from abc import abstractmethod


class AbstractPipeline:
    """
    Interface for different pipelines. So IDE automatically give you need function
    """
    @abstractmethod
    def run(self) -> None:
        pass
