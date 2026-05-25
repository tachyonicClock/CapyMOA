from typing import (
    Iterable,
    Iterator,
    MutableSequence,
    Optional,
    Sequence,
    MutableMapping,
)
from collections import OrderedDict

from torch import Tensor
from torch.nn import Module


class BufferDict(Module, MutableMapping[str, Tensor]):
    def __init__(self, buffers: dict[str, Tensor]) -> None:
        super().__init__()
        for name, buffer in buffers.items():
            self.register_buffer(name, buffer)

    def __repr__(self) -> str:
        """Return a string representing the BufferDict

        >>> import torch
        >>> bd = BufferDict({"x": torch.zeros((10, 10)), "y": torch.zeros((20, 20))})
        >>> print(bd)
        BufferDict(
            (x): Tensor((10, 10), torch.float32)
            (y): Tensor((20, 20), torch.float32)
        )
        """
        buffer_strs = []
        for name, buffer in self.items():
            buffer_strs.append(
                f"({name}): Tensor({tuple(buffer.shape)}, {buffer.dtype})"
            )
        repr_ = "BufferDict("
        for buffer_str in buffer_strs:
            repr_ += f"\n    {buffer_str}"
        repr_ += "\n)"
        return repr_

    def __getitem__(self, key: str) -> Optional[Tensor]:
        return self._buffers.__getitem__(key)

    def __setitem__(self, key: str, value: Tensor) -> None:
        return self._buffers.__setitem__(key, value)

    def __delitem__(self, key: str):
        return self._buffers.__delitem__(key)

    def __iter__(self) -> Iterator[str]:
        return self._buffers.__iter__()

    def __len__(self) -> int:
        return self._buffers.__len__()

    def __hash__(self) -> int:
        return Module.__hash__(self)


class BufferList(Module, MutableSequence[Tensor]):
    def _attr_name(self, index: int) -> str:
        return f"{index}"

    def __init__(self, buffers: Sequence[Tensor]) -> None:
        super().__init__()
        for i, buffer in enumerate(buffers):
            self.register_buffer(self._attr_name(i), buffer)

    def __getitem__(self, index: int) -> Tensor:
        return self._buffers[self._attr_name(index)]  # type: ignore

    def __len__(self) -> int:
        return len(self._buffers)  # type: ignore

    def __contains__(self, value: object) -> bool:
        return any(buffer is value for buffer in self._buffers.values())  # type: ignore

    def __iter__(self) -> Iterable[Tensor]:
        for i in range(len(self)):
            yield self[i]

    def __delitem__(self, index: int) -> None:
        del self._buffers[self._attr_name(index)]
        self._buffers = OrderedDict(
            (self._attr_name(i), buffer)
            for i, buffer in enumerate(self._buffers.values())  # type: ignore
        )

    def __setitem__(self, index: int, value: Tensor) -> None:
        self._buffers[self._attr_name(index)] = value  # type: ignore

    def insert(self, index: int, value: Tensor) -> None:
        _buffers = list(self._buffers.values())  # type: ignore
        _buffers.insert(index, value)
        self._buffers = OrderedDict(
            (self._attr_name(i), buffer) for i, buffer in enumerate(_buffers)
        )

    def __repr__(self) -> str:
        """Return a string representing the BufferList

        >>> import torch
        >>> bl = BufferList([torch.zeros((10, 10)), torch.zeros((20, 20))])
        >>> print(bl)
        BufferList(
            (0): Tensor((10, 10), torch.float32)
            (1): Tensor((20, 20), torch.float32)
        )
        """
        buffer_strs = []
        for i in range(len(self)):
            buffer = self[i]
            buffer_strs.append(f"({i}): Tensor({tuple(buffer.shape)}, {buffer.dtype})")
        repr_ = "BufferList("
        for buffer_str in buffer_strs:
            repr_ += f"\n    {buffer_str}"
        repr_ += "\n)"
        return repr_
