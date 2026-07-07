from capymoa.ocl.strategy._ncm import _ncm_classify
import torch
from torch import Tensor, IntTensor, nn
from torch.nn import functional as F
from typing import Optional, Sequence, Callable, cast, override
from capymoa.stream import Schema
from capymoa.base import BatchClassifier
from capymoa.base.events import Handler, Dispatcher
from capymoa.ocl.replay import ReplayBuffer, ReservoirSampler
from capymoa.ocl.evaluation.events import TrainTaskEnd, TestBegin, TestEnd
from copy import deepcopy


def batched_feature_extract(
    feature_extractor: Callable[[Tensor], Tensor],
    x: Tensor,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Extract features from x in batches to avoid OOM."""
    features = []
    for i in range(0, x.shape[0], batch_size):
        batch_x = x[i : i + batch_size].to(device)
        batch_features = feature_extractor(batch_x).cpu()
        features.append(batch_features)
    return torch.cat(features, dim=0)


def icarl_loss(
    logits: Tensor,
    target: Tensor,
    teacher_logits: Tensor | None = None,
    teacher_classes: Sequence[int] = (),
    distillation_weight: float = 1.0,
) -> Tensor:
    """iCaRL hybrid classification and distillation loss.

    Implements iCaRL algorithm 3 loss function (Rebuffi et al., 2017).

    >>> import torch
    >>> logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    >>> target = torch.tensor([0, 1])
    >>> teacher_logits = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    >>> teacher_classes = [0, 1]
    >>> icarl_loss(logits, target, teacher_logits, teacher_classes)
    tensor(0.6920)

    References:
    * https://github.com/ContinualAI/avalanche/blob/eb075be393e1f458b2c352514ff6c17b5a2c0f4e/avalanche/training/losses.py#L38
    * https://github.com/mmasana/FACIL/blob/e09d2c83320a1aa945a6157d4875437515824dc9/src/approach/icarl.py#L179
    * https://github.com/srebuffi/iCaRL/blob/7a3d254da4b3f0f19f67b8c25ba72374674bafaf/iCaRL-Tensorflow/main_resnet_tf.py#L125

    :param logits: Logit tensor ``(B, C)``
    :param target: Integer target tensor ``(B,)``
    :param teacher_logits: Logit tensor ``(B, C)``
    :param teacher_classes: Classes the teacher 'knows' about.
    :param distillation_weight: Interpolation weight in ``[0, 1]`` for replacing
        teacher-known class targets with teacher probabilities.
        ``0`` keeps one-hot targets only; ``1`` matches standard iCaRL behaviour.
    :return: A loss scalar tensor.
    """
    if logits.shape[0] != target.shape[0]:
        raise ValueError("Logits and target must have the same batch size.")
    if teacher_logits is not None and teacher_logits.shape != logits.shape:
        raise ValueError("Teacher and student logits must match shape.")
    if not (0.0 <= distillation_weight <= 1.0):
        raise ValueError("Distillation weight must be in the range [0, 1].")

    num_classes = logits.shape[1]
    target_one_hot = F.one_hot(target, num_classes=num_classes).float()

    # If a teacher is supplied use its knowledge about previous tasks to help.
    if teacher_logits is not None:
        teacher_probs = torch.sigmoid(teacher_logits)
        target_one_hot[:, teacher_classes] = (
            1.0 - distillation_weight
        ) * target_one_hot[:, teacher_classes] + distillation_weight * teacher_probs[
            :, teacher_classes
        ]

    return F.binary_cross_entropy_with_logits(logits, target_one_hot, reduction="mean")


def construct_exemplar_set(features: Tensor, count: int) -> IntTensor:
    """Construct an exemplar set using herding.

    Herding select samples such that the selection's mean is similar to the overall
    mean.

    Implements algorithm 4 (Rebuffi et al., 2017).

    >>> features = torch.tensor([
    ...     [1.0, 0.0],
    ...     [0.0, 1.0],
    ...     [1.0, 1.0]
    ... ])
    >>> construct_exemplar_set(features, 2)
    tensor([2, 0])

    :param features: Features ``(B, F)`` $\rho(x)$
    :param count: Target number of exemplars
    :return: Indices selecting exemplars
    """
    mean = features.mean(0)
    sum_selected = torch.zeros_like(mean)
    selection = torch.zeros(count, dtype=torch.int64, device=features.device)

    for k in range(count):
        scores = torch.norm(mean - 1 / (k + 1) * (features + sum_selected), p=2, dim=1)
        scores[selection[:k]] = float("inf")  # Do not select the same sample twice

        # Select the best sample and update the running score
        best_idx = torch.argmin(scores)
        selection[k] = best_idx
        sum_selected += features[best_idx]

    return cast(IntTensor, selection)


class _iCaRLReplayBuffer(ReplayBuffer):
    """iCaRL replay buffer that maintains an exemplar set using herding."""

    @override
    def update(self) -> None:
        raise NotImplementedError(
            "Use ``icarl_update`` with custom logic to maintain exemplar set."
        )

    def icarl_update(
        self,
        task_x: Tensor,
        task_y: Tensor,
        feature_extractor: Callable[[Tensor], Tensor],
        num_classes: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        # Collect current task samples. This may be a stream subsample depending on
        # the task buffer policy.
        previous = self.array()
        old_x = previous["x"]
        old_y = previous["y"]

        old_classes = torch.unique(old_y)
        task_classes = torch.unique(task_y)

        is_new_class = torch.ones(task_classes.shape[0], dtype=torch.bool)
        if len(old_classes) > 0:
            is_new_class = ~torch.isin(task_classes, old_classes)
        new_classes = task_classes[is_new_class]

        total_classes = len(old_classes) + len(new_classes)
        if total_classes == 0:
            self._count = 0
            return None
        exemplar_count = self.capacity // total_classes

        selected_x: list[Tensor] = []
        selected_y: list[Tensor] = []

        # Reduce old exemplar sets by truncating each class-preserving order.
        for class_id in old_classes.tolist():
            class_mask = old_y == class_id
            class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
            keep = class_indices[:exemplar_count]
            if len(keep) > 0:
                selected_x.append(old_x[keep])
                selected_y.append(old_y[keep])

        # Construct exemplar sets for new classes via herding over task samples.
        for class_id in new_classes.tolist():
            class_mask = task_y == class_id
            class_x = task_x[class_mask]
            if class_x.shape[0] == 0:
                continue
            class_features = batched_feature_extract(
                feature_extractor, class_x, batch_size, device
            )
            count = min(exemplar_count, class_x.shape[0])
            if count == 0:
                continue
            ids = construct_exemplar_set(class_features, count).to(class_x.device)
            selected_x.append(class_x[ids])
            selected_y.append(task_y[class_mask][ids])

        if len(selected_x) == 0:
            self._count = 0
            return None

        x_all = torch.cat(selected_x, dim=0)
        y_all = torch.cat(selected_y, dim=0)

        # Update buffer with selected exemplars.
        self._buffer["x"][: len(y_all)] = x_all  # type: ignore
        self._buffer["y"][: len(y_all)] = y_all  # type: ignore
        self._count = len(y_all)

        features = batched_feature_extract(feature_extractor, x_all, batch_size, device)
        return self.class_means(y_all.to(features.device), features, num_classes)

    def class_means(self, y: Tensor, features: Tensor, num_classes: int) -> Tensor:
        feature_dim = features.shape[1]
        means = torch.zeros(
            (num_classes, feature_dim),
            dtype=features.dtype,
            device=features.device,
        )

        for class_id in torch.unique(y).tolist():
            class_id = int(class_id)
            class_features = features[y == class_id]
            mean = class_features.mean(dim=0)
            means[class_id] = F.normalize(mean, dim=0)

        return means


class ICaRL(BatchClassifier, nn.Module, Handler):
    """Incremental Classifier and Representation Learning

    iCaRL [#f0]_ is a class-incremental learning strategy that maintains a replay buffer
    of exemplars selected using herding and uses a hybrid classification and
    distillation loss to learn from new tasks while retaining performance on old tasks.

    Sources:
    * http://www.github.com/srebuffi/iCaRL
    * https://github.com/mmasana/FACIL/blob/e09d2c83320a1aa945a6157d4875437515824dc9/src/approach/icarl.py
    * https://github.com/ContinualAI/avalanche/blob/eb075be393e1f458b2c352514ff6c17b5a2c0f4e/avalanche/training/losses.py#L14
    * https://github.com/ContinualAI/avalanche/blob/eb075be393e1f458b2c352514ff6c17b5a2c0f4e/avalanche/training/supervised/icarl.py

    Notes:
    * iCaRL will use NCM classification only during holdout evaluation (after each task)
      but not during online evaluation.
    * Offline iCaRL access the entire training data of the current task at the end of
      the task to update the replay buffer with exemplars selected using herding. To
      mimic this behaviour in the online setting, we use a separate task buffer to store
      the current task's data.

    Subscribes to:
    * :py:class:`~capymoa.ocl.evaluation.events.TrainTaskEnd` event to update the replay
      buffer with the current task's data.
    * :py:class:`~capymoa.ocl.evaluation.events.TestBegin` to enable using NCM
      classification during evaluation.
    * :py:class:`~capymoa.ocl.evaluation.events.TestEnd` to disable using NCM
      classification during evaluation.

    ..  [#f0] Rebuffi, S.-A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017, July).
        iCaRL: Incremental Classifier and Representation Learning. The IEEE Conference
        on Computer Vision and Pattern Recognition (CVPR).
    """

    def __init__(
        self,
        schema: Schema,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        feature_extractor: Callable[[Tensor], Tensor],
        capacity: int = 200,
        batch_size: int = 64,
        distillation_weight: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if not (0.0 <= distillation_weight <= 1.0):
            raise ValueError("distillation_weight must be in the range [0, 1].")

        super().__init__(schema, 0)
        nn.Module.__init__(self)
        self.device = device
        self.ncm_enabled = False
        self._model = model
        self._teacher: Optional[nn.Module] = None
        self._teacher_classes: Sequence[int] = []

        # Class means is populated at the end of each task and used for exemplar
        # selection and classification. Of shape ``(C, F)`` where C is the number of
        # classes and F is the number of features.
        self._class_means: Optional[Tensor] = None
        self._batch_size = batch_size
        self._distillation_weight = float(distillation_weight)
        self._optimiser = optimiser
        self._feature_extractor = feature_extractor

        _buffers = {"x": (schema.shape, torch.float32), "y": ((), torch.long)}
        self._buffer = _iCaRLReplayBuffer(capacity, _buffers)

        # Task buffer is used to store data from the current task. This is needed to
        # implement the iCaRL's replay buffer update strategy, which requires access to
        # the current task's data.
        self._task_buffer = ReservoirSampler().build(capacity, _buffers)
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    @override
    def batch_train(self, x: Tensor, y: Tensor) -> None:
        self._task_buffer.update(x=x, y=y)

        # Sample from the replay buffer and current task buffer for training.
        n = x.shape[0]
        if self._buffer.count > 0:
            (replay_x, replay_y) = self._buffer.sample(n).values()
        else:
            replay_x = x[:0]
            replay_y = y[:0]
        train_x = torch.cat([x, replay_x], dim=0).to(self.device)
        train_y = torch.cat([y, replay_y], dim=0).to(self.device)

        self._optimiser.zero_grad()
        logits = self._model(train_x)
        if self._teacher is not None:
            with torch.no_grad():
                teacher_logits = self._teacher(train_x)
            loss = icarl_loss(
                logits,
                train_y,
                teacher_logits,
                self._teacher_classes,
                self._distillation_weight,
            )
        else:
            loss = F.cross_entropy(logits, train_y)
        loss.backward()
        self._optimiser.step()

    @override
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        if self.ncm_enabled and self._class_means is not None:
            features = self._feature_extractor(x)
            features = F.normalize(features, p=2, dim=1)
            return _ncm_classify(features, self._class_means.to(features.device))
        else:
            return torch.sigmoid(self._model(x))

    def attach_with(self, dispatcher: Dispatcher) -> None:
        dispatcher.subscribe(TrainTaskEnd, self._on_train_task_end)
        dispatcher.subscribe(TestBegin, self._enable_ncm)
        dispatcher.subscribe(TestEnd, self._disable_ncm)

    def _on_train_task_end(self, event: TrainTaskEnd) -> None:
        # Update the replay buffer with the current task's data using iCaRL's exemplar
        # selection strategy.
        task_data = self._task_buffer.array()
        was_training = self._model.training
        self._model.eval()
        with torch.no_grad():
            self._class_means = self._buffer.icarl_update(
                task_x=task_data["x"],
                task_y=task_data["y"],
                feature_extractor=self._feature_extractor,
                num_classes=self.schema.get_num_classes(),
                device=self.device,
                batch_size=self._batch_size,
            )
        if was_training:
            self._model.train()
        self._task_buffer.clear()

        # Update teacher model and known classes for distillation in future tasks.
        with torch.no_grad():
            self._teacher = deepcopy(self._model).eval().requires_grad_(False)
            self._teacher.to(self.device)
            self._teacher_classes = torch.unique(self._buffer.array()["y"]).tolist()

    def _enable_ncm(self, event: TestBegin | None = None) -> None:
        self.ncm_enabled = True

    def _disable_ncm(self, event: TestEnd | None = None) -> None:
        self.ncm_enabled = False
