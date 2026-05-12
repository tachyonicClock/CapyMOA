from capymoa.base import BatchClassifier, BatchRegressor
from capymoa.ocl.datasets import TinySplitMNIST
from capymoa.evaluation import prequential_evaluation
import torch
from torch import Tensor
import pytest


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("y_dtype", [torch.long, torch.int64])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_batch_classifier(batch_size, x_dtype, y_dtype, device):
    scenario = TinySplitMNIST()
    shape = (1, 16, 16)

    class MyBatchClassifier(BatchClassifier):
        def __init__(self, schema, x_dtype, y_dtype, device):
            super().__init__(schema)
            self.x_dtype = x_dtype
            self.y_dtype = y_dtype
            self.device = device

        def batch_train(self, x, y):
            assert x.shape[1:] == shape
            assert y.shape == (x.shape[0],)
            assert isinstance(x, Tensor)
            assert isinstance(y, Tensor)
            assert x.dtype == self.x_dtype == x_dtype
            assert y.dtype == self.y_dtype == y_dtype

        def batch_predict_proba(self, x):
            return torch.zeros((x.shape[0], self.schema.get_num_classes()))

    learner = MyBatchClassifier(scenario.schema, x_dtype, y_dtype, device)
    prequential_evaluation(
        scenario.stream, learner, batch_size=batch_size, max_instances=100
    )


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("y_dtype", [torch.float32])
@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_batch_regressor(batch_size, x_dtype, y_dtype, device):
    scenario = TinySplitMNIST()
    scenario.schema._regression = True  # Force into a regression scenario
    shape = (1, 16, 16)

    class MyBatchRegressor(BatchRegressor):
        def __init__(self, schema, x_dtype, y_dtype, device):
            super().__init__(schema)
            self.x_dtype = x_dtype
            self.y_dtype = y_dtype
            self.device = device

        def batch_train(self, x, y):
            assert x.shape[1:] == shape
            assert y.shape == (x.shape[0],)
            assert isinstance(x, Tensor)
            assert isinstance(y, Tensor)
            assert x.dtype == self.x_dtype == x_dtype
            assert y.dtype == self.y_dtype == y_dtype

        def batch_predict(self, x):
            return torch.zeros((x.shape[0],), dtype=self.y_dtype, device=self.device)

    learner = MyBatchRegressor(scenario.schema, x_dtype, y_dtype, device)
    prequential_evaluation(
        scenario.stream, learner, batch_size=batch_size, max_instances=100
    )
