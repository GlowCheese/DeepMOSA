from __future__ import annotations

from custom_logger import getLogger
from typing import final, List, TYPE_CHECKING


logger = getLogger(__name__)

if TYPE_CHECKING:
    from pynguin.setup import TestCluster
    from pynguin.config import (
        Algorithm,
        Configuration,
        CoverageMetric,
        SeedingConfiguration,
        StoppingConfiguration,
        TestCreationConfiguration,
        TypeInferenceConfiguration,
        TestCaseOutputConfiguration,
        SearchAlgorithmConfiguration,
        StatisticsOutputConfiguration
    )
    from pynguin.utils.randomness import Random
    from pynguin.runtimevar import RuntimeVariable
    from pynguin.statistics import StatisticsTracker
    from pynguin.llm.abstractllmseeding import AbstractLLMSeeding
    from pynguin.constants import ConstantProvider, DynamicConstantProvider


@final
class _Globl:
    def __init__(self):
        self.conf: Configuration
        self.test_cluster: TestCluster
        self.constant_provider: ConstantProvider
        self.statistics_tracker: StatisticsTracker
        self.dynamic_constant_provider: DynamicConstantProvider

        # llms
        self.llmseeding: AbstractLLMSeeding

        # configuration
        self.seeding_conf: SeedingConfiguration
        self.stopping_conf: StoppingConfiguration
        self.ga_conf: SearchAlgorithmConfiguration
        self.output_conf: TestCaseOutputConfiguration
        self.test_creation_conf: TestCreationConfiguration
        self.statistics_conf: StatisticsOutputConfiguration
        self.type_inference_conf: TypeInferenceConfiguration
        
        # common stuffs
        self.seed: int
        self.report_dir: str
        self.module_name: str
        self.module_path: str
        self.project_path: str
        self.project_name: str
        self.algorithm: Algorithm

        # others
        self.coverage_metrics: List[CoverageMetric]
        self.output_variables: List[RuntimeVariable]


    def reset(self):
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                try:
                    setattr(self, attr, None)
                except AttributeError:
                    logger.warning(f"Cannot reset attribute '{attr}'")
        logger.info("Globl state has been reset.")


    def set_conf(self, conf: Configuration):
        self.conf = conf
        
        self.seed = conf.seeding.seed
        self.algorithm = conf.algorithm
        self.module_name = conf.module_name
        self.module_path = conf.module_path
        self.project_path = conf.project_path
        self.report_dir = conf.statistics_output.report_dir
        self.project_name = conf.statistics_output.project_name
        
        self.seeding_conf = conf.seeding
        self.stopping_conf = conf.stopping
        self.ga_conf = conf.search_algorithm
        self.output_conf = conf.test_case_output
        self.test_creation_conf = conf.test_creation
        self.statistics_conf = conf.statistics_output
        self.type_inference_conf = conf.type_inference

        self.coverage_metrics = conf.statistics_output.coverage_metrics
        self.output_variables = conf.statistics_output.output_variables


# singleton
Globl = _Globl()
