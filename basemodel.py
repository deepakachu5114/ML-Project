from abc import ABC, abstractmethod

class BaseModel(ABC):

    hyperparameters : dict
    name : str

    def __init__(self, hyperparameters : dict):
        self.hyperparameters = hyperparameters
        self.name = self.__class__.__name__

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass
