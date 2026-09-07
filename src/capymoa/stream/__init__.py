"""Data stream representations and related utilities."""

from capymoa._optional import lazy_torch_attrs

from . import drift, generator, preprocessing
from ._csv_stream import CSVStream
from ._stream import (
    ARFFStream,
    MOAStream,
    NumpyStream,
    Schema,
    Stream,
)
from ._stream_from_file import stream_from_file

__all__ = [
    "ARFFStream",
    "CSVStream",
    "MOAStream",
    "NumpyStream",
    "Schema",
    "Stream",
    "TorchStream",
    "drift",
    "generator",
    "preprocessing",
    "stream_from_file",
]


#: Names that need PyTorch. Imported on first access so ``import capymoa`` stays
#: torch-free -- see :mod:`capymoa._optional`.
_LAZY = {
    "TorchStream": ".torch",
}

__getattr__, __dir__ = lazy_torch_attrs(__name__, _LAZY, "TorchStream", __all__)
