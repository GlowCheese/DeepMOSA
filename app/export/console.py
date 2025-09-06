from .utils import ExperimentStatistics

exs = ExperimentStatistics()
exs.load_everything()
exs.analyse()
exs.report()
