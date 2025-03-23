from abc import abstractmethod, ABC

class Identifiable(ABC):

    @property
    @abstractmethod
    def id(self): ...
