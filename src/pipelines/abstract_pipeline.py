from abc import abstractmethod


class AbstractPipeline:
    @abstractmethod
    def run(self) -> None:
        pass
