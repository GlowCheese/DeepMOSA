from abc import ABC, abstractmethod

class AbstractLLMSeeding(ABC):
    # TODO: implement this class further!

    @abstractmethod
    def _get_targeted_testcase(*args, **kwargs):
        pass

    @abstractmethod
    def target_uncovered_functions(*args, **kwargs):
        pass