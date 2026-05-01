from typing import Callable

import torch
from torch import Tensor, nn

from capymoa.base import BatchClassifier
from capymoa.ocl.replay import ReplayBuilder, ReservoirSampler
from capymoa.stream._stream import Schema


class DER(BatchClassifier, nn.Module):
    """Dark Experience Replay.

    Dark Experience Replay (DER) [#f1]_ is a replay-based continual learning
    strategy that stores model logits for replay samples and regularises new
    predictions with an MSE loss on those stored logits.

    ..  [#f1] Buzzega, Pietro, Matteo Boschini, Angelo Porrello, Davide Abati,
        and SIMONE CALDERARA. “Dark Experience for General Continual Learning:
        A Strong, Simple Baseline.” Advances in Neural Information Processing
        Systems 33 (2020): 15920–30.
        https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html.

    """

    def __init__(
        self,
        schema: Schema,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        augment: Callable[[Tensor], Tensor],
        alpha: float = 0.5,
        buffer_capacity: int = 200,
        replay_builder: ReplayBuilder | None = None,
        seed: int = 0,
        substeps: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Construct a DER learner.

        :param schema: Stream schema used by the classifier interface.
        :param model: Torch model that outputs class logits.
        :param optimiser: Optimiser used to update ``model`` parameters.
        :param alpha: Weight of the DER replay-logit loss term.
        :param buffer_capacity: Number of replay samples to retain, defaults to 200.
        :param replay_builder: Builder used to construct the replay buffer.
        :param augment: Data augmentation function that takes a batch of
            examples.
        :param substeps: Number of optimization steps to take per batch. Each
            step will use a different random augmentation of the batch and
            replay samples.
        :param seed: Random seed for reproducibility.
        :param device: Compute device.
        """
        super().__init__(schema, 0)
        nn.Module.__init__(self)
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be greater than zero.")

        self.device = device
        self._augment = augment
        self._alpha = alpha
        self._substeps = substeps
        self._model = model
        self._optimiser = optimiser
        self._criterion = torch.nn.CrossEntropyLoss()
        self._logit_loss = torch.nn.MSELoss()
        if replay_builder is None:
            replay_builder = ReservoirSampler()
        self._buffer = replay_builder.build(
            buffer_capacity,
            dict(
                x=(schema.shape, torch.float32),
                z=((schema.get_num_classes(),), torch.float32),
                y=((), torch.long),
            ),
            torch.Generator().manual_seed(seed),
        )
        self.to(device)

    def _train_step(self, x: Tensor, y: Tensor, update_buffer: bool) -> None:
        self._optimiser.zero_grad()

        n = x.shape[0]

        # Update Buffer
        x_t = self._augment(x)  # $x_t$
        z = self._model(x_t)  # $z$
        if update_buffer:
            self._buffer.update(x=x, z=z, y=y)

        # Sample buffer and augment
        xp, zp, _ = self._buffer.sample(n).values()  # $x'$, $z'$, $y'$
        x_tp = self._augment(xp)  # $x_t'$

        reg = self._alpha * self._logit_loss(zp, self._model(x_tp))  # Line 7
        loss = self._criterion(z, y) + reg  # Line 8
        loss.backward()
        self._optimiser.step()

    def batch_train(self, x: Tensor, y: Tensor) -> None:
        """Implementes Algorithm 1 from the DER paper."""
        self._model.train()
        for i in range(self._substeps):
            self._train_step(x, y, update_buffer=i == 0)

    @torch.no_grad()
    def batch_predict_proba(self, x: Tensor) -> Tensor:
        self._model.eval()
        y_hat = self._model(x)
        return torch.softmax(y_hat, dim=1)

    def __str__(self) -> str:
        return f"DER(alpha={self._alpha}, buffer_capacity={self._buffer._capacity})"
