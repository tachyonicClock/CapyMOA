from capymoa.schema import Schema   # type: ignore
from typing import Sized, Iterable, Generic, TypeVar, Optional, Iterator
from abc import abstractmethod, ABC

_T_co = TypeVar('_T_co', covariant=True)

class Stream(Iterable[_T_co], Generic[_T_co], Sized, ABC):
    
    @abstractmethod
    def get_schema(self) -> Schema:
        pass

    def is_finite(self) -> bool:
        return self.num_instances() is not None

    @abstractmethod
    def num_instances(self) -> Optional[int]:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[_T_co]:
        pass

    def __len__(self) -> int:
        match self.num_instances():
            case None:
                raise ValueError("Stream is infinite")
            case n:
                return n
