#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides a configuration interface for the test generator."""

import dataclasses
import enum
import time
from typing import List

from pynguin.runtimevar import RuntimeVariable


class ExportStrategy(str, enum.Enum):
    """Contains all available export strategies.

    These strategies allow to export the generated test cases in different styles,
    such as the style of the `PyTest` framework.  Setting the value to `NONE` will
    prevent exporting of the generated test cases (only reasonable for
    benchmarking, though).
    """

    PY_TEST = "PY_TEST"
    """Export tests in the style of the PyTest framework."""

    NONE = "NONE"
    """Do not export test cases at all."""


class UninterpretedStatementUse(tuple, enum.Enum):
    """Whether to use ASTAssignStatements (aka uninterpreted statements)
    when parsing targeted test cases.
    """

    NONE = (False,)
    """Don't use the statements."""

    ONLY = (True,)
    """Parse test cases with uninterpreted statements, not what the test case
     would have been without uninterpreted statements"""

    BOTH = (True, False)
    """Parse each generated test case with and without uninterpreted statements"""


class Algorithm(str, enum.Enum):
    """Different algorithms supported by Pynguin."""

    DYNAMOSA = "DYNAMOSA"
    """The dynamic many-objective sorting algorithm (cf. Panichella et al. Automated
    test case generation as a many-objective optimisation problem with dynamic selection
    of the targets.  TSE vol. 44 issue 2)."""

    CODAMOSA = "CODAMOSA"
    """MOSA + Codex :)"""

    DEEPMOSA = "DEEPMOSA"
    """CODAMOSA + Enhanced Prompting"""

    MIO = "MIO"
    """The MIO test suite generation algorithm (cf. Andrea Arcuri. Many Independent
    Objective (MIO) Algorithm for Test Suite Generation.  Proc. SBSE 2017)."""

    MOSA = "MOSA"
    """The many-objective sorting algorithm (cf. Panichella et al. Reformulating Branch
    Coverage as a Many-Objective Optimization Problem.  Proc. ICST 2015)."""

    RANDOM = "RANDOM"
    """A feedback-direct random test generation approach similar to the algorithm
    proposed by Randoop (cf. Pacheco et al. Feedback-directed random test generation.
    Proc. ICSE 2007)."""

    RANDOM_TEST_SUITE_SEARCH = "RANDOM_TEST_SUITE_SEARCH"
    """Performs random search on test suites."""

    RANDOM_TEST_CASE_SEARCH = "RANDOM_TEST_CASE_SEARCH"
    """Performs random search on test cases."""

    WHOLE_SUITE = "WHOLE_SUITE"
    """A whole-suite test generation approach similar to the one proposed by EvoSuite
    (cf. Fraser and Arcuri. EvoSuite: Automatic Test Suite Generation for
    Object-Oriented Software. Proc. ESEC/FSE 2011).

    This algorithm can be modified to use an archive (cf. Rojas, José Miguel, et al.
    "A detailed investigation of the effectiveness of whole test suite generation."
    Empirical Software Engineering 22.2 (2017): 852-893.), by using the
    following options: --use-archive True, --seed-from-archive True and
    --filter-covered-targets-from-test-cluster True.
    """


class AssertionGenerator(str, enum.Enum):
    """Different approaches for assertion generation supported by Pynguin."""

    MUTATION_ANALYSIS = "MUTATION_ANALYSIS"
    """Use the mutation analysis approach for assertion generation."""

    CHECKED_MINIMIZING = "CHECKED_MINIMIZING"
    """All assertions that do not increase the checked coverage are removed."""

    SIMPLE = "SIMPLE"
    """Use the simple approach for primitive and none assertion generation."""

    NONE = "NONE"
    """Do not create any assertions."""


class MutationStrategy(str, enum.Enum):
    """Different strategies for creating mutants.

    Only respected when using the MUTATION_ANALYSIS approach for assertion generation.
    """

    FIRST_ORDER_MUTANTS = "FIRST_ORDER_MUTANTS"
    """Generate first order mutants."""

    FIRST_TO_LAST = "FIRST_TO_LAST"
    """Higher order mutation strategy FirstToLast.
    (cf. Mateo et al. Validating Second-Order Mutation at System Level. Article.
    IEEE Transactions on SE 39.4 2013)"""

    BETWEEN_OPERATORS = "BETWEEN_OPERATORS"
    """Higher order mutation strategy BetweenOperators.
    (cf. Mateo et al. Validating Second-Order Mutation at System Level. Article.
    IEEE Transactions on SE 39.4 2013)"""

    RANDOM = "RANDOM"
    """Higher order mutation strategy Random.
    (cf. Mateo et al. Validating Second-Order Mutation at System Level. Article.
    IEEE Transactions on SE 39.4 2013)"""

    EACH_CHOICE = "EACH_CHOICE"
    """Higher order mutation strategy EachChoice.
    (cf. Mateo et al. Validating Second-Order Mutation at System Level. Article.
    IEEE Transactions on SE 39.4 2013)"""


class TypeInferenceStrategy(str, enum.Enum):
    """The different available type-inference strategies."""

    NONE = "NONE"
    """Ignore any type information given in the module under test."""

    TYPE_HINTS = "TYPE_HINTS"
    """Use type information from type hints in the module under test."""


class StatisticsBackend(str, enum.Enum):
    """The different available statistics backends to write statistics."""

    NONE = "NONE"
    """Do not write any statistics."""

    CONSOLE = "CONSOLE"
    """Write statistics to the standard out."""

    CSV = "CSV"
    """Write statistics to a CSV file."""


class CoverageMetric(str, enum.Enum):
    """The different available coverage metrics available for optimisation."""

    BRANCH = "BRANCH"
    """Calculate how many of the possible branches in the code were executed"""

    LINE = "LINE"
    """Calculate how many of the possible lines in the code were executed"""

    CHECKED = "CHECKED"
    """Calculate how many of the possible lines in the
    code are checked by an assertion."""


class Selection(str, enum.Enum):
    """Different selection algorithms to select from."""

    RANK_SELECTION = "RANK_SELECTION"
    """Rank selection."""

    TOURNAMENT_SELECTION = "TOURNAMENT_SELECTION"
    """Tournament selection.  Use `tournament_size` to set size."""


class TestCaseContext(str, enum.Enum):
    """What kind of extra context to pass to the LLM when
    generating test cases for CodaMOSA"""

    NONE = "NONE"
    """Don't add any additional context."""

    SMALLEST = "SMALLEST"
    """Add the smallest 'winning' test case as context"""

    RANDOM = "RANDOM"
    """Add a random test case as context"""


@dataclasses.dataclass
class StatisticsOutputConfiguration:
    """Configuration related to output."""

    report_dir: str = "pynguin-report"
    """Directory in which to put HTML and CSV reports"""

    statistics_backend: StatisticsBackend = StatisticsBackend.CSV
    """Which backend to use to collect data"""

    timeline_interval: int = 1 * 1_000_000_000
    """Time interval in nano-seconds for timeline statistics, i.e., we select a data
    point after each interval.  This can be interpolated, if there is no exact
    value stored at the time-step of the interval, see `timeline_interpolation`.
    The default value is every 1.00s."""

    timeline_interpolation: bool = True
    """Interpolate timeline values"""

    coverage_metrics: list[CoverageMetric] = dataclasses.field(
        default_factory=lambda: [
            CoverageMetric.LINE,
            CoverageMetric.BRANCH,
        ]
    )
    """List of coverage metrics that are optimised during the search"""

    output_variables: list[RuntimeVariable] = dataclasses.field(
        default_factory=lambda: [
            RuntimeVariable.TargetModule,
            RuntimeVariable.Coverage,
        ]
    )
    """List of variables to output to the statistics backend."""

    configuration_id: str = ""
    """Label that identifies the used configuration of Pynguin.  This is only done
    when running experiments."""

    run_id: str = ""
    """Id of the cluster run. Useful for finding the log entries that belong to a
    certain result."""

    project_name: str = ""
    """Label that identifies the project name of Pynguin.  This is useful when
    running experiments."""

    create_coverage_report: bool = True
    """Create a coverage report for the tested module.
    This can be helpful to find hard to cover parts because Pynguin measures coverage
    on bytecode level which might yield different results when compared with other
    tools, e.g., Coverage.py."""

    type_guess_top_n: int = 10
    """When exporting type guesses for parameters, how many guesses per parameter
    should be exported? Expects positive integers."""


@dataclasses.dataclass
class TestCaseOutputConfiguration:
    """Configuration related to test case output."""

    output_path: str = ""
    """Path to an output folder for the generated test cases."""

    export_strategies: List[ExportStrategy] = dataclasses.field(
        default_factory=lambda: [ExportStrategy.PY_TEST]
    )
    """The export strategy determines for which test-runner system the
    generated tests should fit."""

    max_length_test_case: int = 2500
    """The maximum number of statement in as test case (normal + assertion
    statements)"""

    assertion_generation: AssertionGenerator = AssertionGenerator.SIMPLE
    """The generator that shall be used for assertion generation."""

    allow_stale_assertions: bool = False
    """Allow assertion on things that did not change between statement executions."""

    mutation_strategy: MutationStrategy = MutationStrategy.FIRST_ORDER_MUTANTS
    """The strategy that shall be used for creating mutants in the mutation analysis
    assertion generation method."""

    mutation_order: int = 1
    """The order of the generated higher order mutants in the mutation analysis
    assertion generation method."""

    post_process: bool = True
    """Should the results be post processed? For example, truncate test cases after
    statements that raise an exception."""

    float_precision: float = 0.01
    """Precision to use in float comparisons and assertions"""

    format_with_black: bool = False
    """Format the generated test cases using black."""


@dataclasses.dataclass
class SeedingConfiguration:
    """Configuration related to seeding."""

    seed: int = time.time_ns()  # noqa: RUF009
    """A predefined seed value for the random number generator that is used."""

    constant_seeding: bool = True
    """Should the generator use a static constant seeding technique to improve constant
    generation?"""

    large_language_model_seeding: bool = False
    """If set to True, assume we want to use an OpenAI large language
    model to conduct seeding.
    """

    large_language_model_mutation: bool = True
    """If set to True, assume we want to use an OpenAI large language
    model to conduct mutation
    """

    initial_population_seeding: bool = False
    """Should the generator use previously existing testcases to seed the initial
    population?"""

    initial_population_data: str = ""
    """The path to the file with the pre-existing tests. The path has to include the
    file itself."""

    sample_with_replacement: bool = True
    """Should we allow sampling with replacement from previously existing testcases?"""

    allow_expandable_cluster: bool = False
    """Should we create an 'expandable' test cluster, which we can query for new
    functions at seeding/test time?"""

    expand_cluster: bool = False
    """Similar to the above, but create the expanded test cluster from the start.
    """

    max_cluster_recursion: int = 1
    """The maximum level of recursion when calculating the dependencies in the test
    cluster."""

    remove_testcases_without_coverage: bool = False
    """Should we remove seeded test cases that don't have any coverage of the test
    module?"""

    include_partially_parsable: bool = True
    """If true, keep the parsable parts of seed test cases. If False, only retain test
    cases that are fully parsable. """

    seeded_testcases_reuse_probability: float = 0.9
    """Probability of using seeded testcases when initial population seeding is
    enabled."""

    initial_population_mutations: int = 0
    """Number of how often the testcases collected by initial population seeding should
    be mutated to promote diversity"""

    dynamic_constant_seeding: bool = True
    """Enables seeding of constants at runtime."""

    seeded_primitives_reuse_probability: float = 0.2
    """Probability for using seeded primitive values instead of randomly
    generated ones."""

    seeded_dynamic_values_reuse_probability: float = 0.6
    """Probability of using dynamically seeded values when a primitive seeded
     value will be used."""

    seed_from_archive: bool = False
    """When sampling new test cases reuse some from the archive, if one is used."""

    seed_from_archive_probability: float = 0.2
    """Instead of creating a new test case, reuse a covering solution from the archive,
    iff an archive is used."""

    seed_from_archive_mutations: int = 3
    """Number of mutations applied when sampling from the archive."""

    max_dynamic_length: int = 1000
    """Maximum length of strings/bytes that should be stored in the dynamic constant
    pool."""

    max_dynamic_pool_size: int = 50
    """Maximum number of constants of the same type that should be stored in the
    dynamic constant pool."""

    uninterpreted_statements: UninterpretedStatementUse = UninterpretedStatementUse.ONLY
    """Whether to allow uninterpreted assignment statements in the parsed test cases"""


@dataclasses.dataclass
class CodaMosaConfiguration:
    """Configuration for CodaMosa"""

    max_plateau_len: int = 25
    """The number of iterations to let go on before trying to do LLM Seeding"""

    temperature: float = 0.8
    """The temperature to use when querying the model"""

    num_seeds_to_inject: int = 5
    """Number of seeds to query the OpenAI model for"""

    test_case_context: TestCaseContext = TestCaseContext.NONE
    """What extra context to pass to the LLM when querying for a new test case"""

    target_low_coverage_functions: bool = True
    """Whether or not to target low coverage functions. If false, target random
    functions."""

    max_tokens: int = 1024
    """Max number of tokens for the completion response."""


@dataclasses.dataclass
class DeepMosaConfiguration:
    """Configuration for DeepMosa"""

    async_enabled: bool = False
    """Whether DeepMOSA should use asynchoronous strategy or not."""

    max_plateau_len: int = 25
    """The number of iterations to let go on before trying to do LLM Seeding"""

    temperature: float = 0.8
    """The temperature to use when querying the model"""

    num_seeds_to_inject: int = 5
    """Number of seeds to query the OpenAI model for"""

    test_case_context: TestCaseContext = TestCaseContext.NONE
    """What extra context to pass to the LLM when querying for a new test case"""

    target_low_coverage_functions: bool = True
    """Whether or not to target low coverage functions. If false, target random
    functions."""

    max_tokens: int = 1024
    """Max number of tokens for the completion response."""

    reseed_probability: float = 1.0 / 3.0
    """Probability of requesting initial prompt when a predicate is provided."""

    recreate_conversation_probability: float = 0.6
    """Probability of recreating new conversation. Great for attempt
    to try new random set of code dependencies for LLM prompt generation."""

    tau: float = 1.3
    """Temperate parameter used in goal selection softmax formula."""


@dataclasses.dataclass
class MIOPhaseConfiguration:
    """Configuration for a phase of MIO."""

    number_of_tests_per_target: int
    """Number of test cases for each target goal to keep in an archive."""

    random_test_or_from_archive_probability: float
    """Probability [0,1] of sampling a new test at random or choose an existing one in
    an archive."""

    number_of_mutations: int
    """Number of mutations allowed to be done on the same individual before
    sampling a new one."""


@dataclasses.dataclass
class MIOConfiguration:
    """Configuration that is specific to the MIO approach."""

    initial_config = MIOPhaseConfiguration(
        number_of_tests_per_target=10,
        random_test_or_from_archive_probability=0.5,
        number_of_mutations=1,
    )
    """Configuration to use before focused phase."""

    focused_config = MIOPhaseConfiguration(
        number_of_tests_per_target=1,
        random_test_or_from_archive_probability=0.0,
        number_of_mutations=10,
    )
    """Configuration to use in focused phase"""

    exploitation_starts_at_percent: float = 0.5
    """Percentage ]0,1] of search budget after which exploitation is activated, i.e.,
    switching to focused phase."""


@dataclasses.dataclass
class RandomConfiguration:
    """Configuration that is specific to the RANDOM approach."""

    max_sequence_length: int = 10
    """The maximum length of sequences that are generated, 0 means infinite."""

    max_sequences_combined: int = 10
    """The maximum number of combined sequences, 0 means infinite."""


@dataclasses.dataclass
class TypeInferenceConfiguration:
    """Configuration related to type inference."""

    guess_unknown_types: bool = True
    """Should we guess unknown types while constructing parameters?
    This might happen in the following cases:
    The parameter type is unknown, e.g. a parameter is missing a type hint.
    The parameter is not primitive and cannot be created from the test cluster,
    e.g. Callable[...]"""

    type_inference_strategy: TypeInferenceStrategy = TypeInferenceStrategy.TYPE_HINTS
    """The strategy for type-inference that shall be used"""

    type_tracing: bool = True
    """Trace usage of parameters with unknown types to improve type guesses."""


@dataclasses.dataclass
class TestCreationConfiguration:
    """Configuration related to test creation."""

    max_recursion: int = 10
    """Recursion depth when trying to create objects in a test case."""

    max_delta: int = 20
    """Maximum size of delta for numbers during mutation"""

    max_int: int = 2048
    """Maximum size of randomly generated integers (minimum range = -1 * max)"""

    string_length: int = 20
    """Maximum length of randomly generated strings"""

    bytes_length: int = 20
    """Maximum length of randomly generated bytes"""

    collection_size: int = 5
    """Maximum length of randomly generated collections"""

    primitive_reuse_probability: float = 0.5
    """Probability to reuse an existing primitive in a test case, if available.
    Expects values in [0,1]"""

    object_reuse_probability: float = 0.9
    """Probability to reuse an existing object in a test case, if available.
    Expects values in [0,1]"""

    none_weight: float = 1
    """Weight to use None as parameter type during test generation.
    Expects values > 0."""

    any_weight: float = 5
    """Weight to use Any as parameter type during test generation.
    Expects values > 0."""

    original_type_weight: float = 5
    """Weight to use the originally annotated type as parameter type during test
    generation. Expects values > 0."""

    type_tracing_weight: float = 10
    """Weight to use the type guessed from type tracing as parameter type during
    test generation. Expects values > 0."""

    type4py_weight: float = 10
    """Weight to use types inferred from type4py as parameter type during
    test generation. Expects values > 0."""

    type_tracing_kept_guesses: int = 2
    """Amount of kept recently guessed types per parameter, when type tracing
    is used."""

    wrap_var_param_type_probability: float = 0.7
    """Probability to wrap the type required for a *arg or **kwargs parameter
    in a list or dict, respectively. Expects values in [0,1]"""

    negate_type: float = 0.1
    """When inferring a type from proxies, it may also be desirable to negate the chosen
    type, e.g., such that an instance check or a getattr() evaluate to False.
    Expects values in [0,1]"""

    skip_optional_parameter_probability: float = 0.7
    """Probability to skip an optional parameter, i.e., do not fill such a parameter."""

    max_attempts: int = 1000
    """Number of attempts when generating an object before giving up"""

    insertion_uut: float = 0.5
    """Score for selection of insertion of UUT calls"""

    max_size: int = 100
    """Maximum number of test cases in a test suite"""

    use_random_object_for_call: float = 0.1
    """When adding or modifying a call on an object, use a random modifier instead
    of only modifiers for that type. Expects values in [0, 1]."""


@dataclasses.dataclass
class SearchAlgorithmConfiguration:
    """General configuration for search algorithms."""

    min_initial_tests: int = 1
    """Minimum number of tests in initial test suites"""

    max_initial_tests: int = 10
    """Maximum number of tests in initial test suites"""

    population: int = 50
    """Population size of genetic algorithm"""

    chromosome_length: int = 40
    """Maximum length of chromosomes during search"""

    chop_max_length: bool = True
    """Chop statements after exception if length has reached maximum"""

    elite: int = 1
    """Elite size for search algorithm"""

    crossover_rate: float = 0.75
    """Probability of crossover"""

    test_insertion_probability: float = 0.1
    """Initial probability of inserting a new test in a test suite"""

    test_delete_probability: float = 1.0 / 3.0
    """Probability of deleting statements during mutation"""

    test_change_probability: float = 1.0 / 3.0
    """Probability of changing statements during mutation"""

    test_insert_probability: float = 1.0 / 3.0
    """Probability of inserting new statements during mutation"""

    statement_insertion_probability: float = 0.5
    """Initial probability of inserting a new statement in a test case"""

    random_perturbation: float = 0.2
    """Probability to replace a primitive with a random new value rather than adding
    a delta."""

    change_parameter_probability: float = 0.1
    """Probability of replacing parameters when mutating a method or constructor
    statement in a test case.  Expects values in [0,1]"""

    tournament_size: int = 5
    """Number of individuals for tournament selection."""

    rank_bias: float = 1.7
    """Bias for better individuals in rank selection"""

    selection: Selection = Selection.TOURNAMENT_SELECTION
    """The selection operator for genetic algorithms."""

    use_archive: bool = False
    """Some algorithms can be enhanced with an optional archive, e.g. Whole Suite ->
    Whole Suite + Archive. Use this option to enable the usage of an archive.
    Algorithms that always use an archive are not affected by this option."""

    filter_covered_targets_from_test_cluster: bool = False
    """Focus search by filtering out elements from the test cluster when
     they are fully covered."""

    number_of_mutations: int = 1
    """Number of mutations that should be applied in one breeding step."""


@dataclasses.dataclass
class StoppingConfiguration:
    """Configuration related to when Pynguin should stop.

    Note that these are mostly soft-limits rather than hard limits, because
    the search algorithms only check the condition at the start of each algorithm
    iteration.
    """

    maximum_search_time: int = -1
    """Time (in seconds) that can be used for generating tests."""

    maximum_test_executions: int = -1
    """Maximum number of test cases to be executed."""

    maximum_statement_executions: int = -1
    """Maximum number of test cases to be executed."""

    maximum_slicing_time: int = 600
    """Time budget (in seconds) that can be used for slicing."""

    maximum_iterations: int = -1
    """Maximum iterations"""

    maximum_test_execution_timeout: int = 5
    """The maximum time (in seconds) after which a test case times out."""

    maximum_coverage: int = 100
    """The maximum percentage of coverage after which the generation shall stop."""

    maximum_coverage_plateau: int = -1
    """Maximum number of algorithm iterations without coverage change before the
    algorithms stops."""

    minimum_coverage: int = 100
    """Minimum coverage for the plateau-based stopping condition.  Expects values larger
    than 0 but less than 100 to activate the stopping condition; also requires the
    setting of minimum_plateau_iterations."""

    minimum_plateau_iterations: int = -1
    """Minimum iterations without a coverage change to stop early.  Expects values
    larger than 0; also requires the setting of minimum_coverage."""

    test_execution_time_per_statement: int = 1
    """The time (in seconds) per statement that a test is allowed to run
    (up to maximum_test_execution_timeout)."""


@dataclasses.dataclass
class Configuration:
    """General configuration for the test generator."""

    project_path: str
    """Path to the project the generator shall create tests for."""

    module_name: str
    """Name of the module for which the generator shall create tests."""

    module_path: str
    """Literally combination of project_path and module_name"""

    test_case_output: TestCaseOutputConfiguration = dataclasses.field(
        default_factory=TestCaseOutputConfiguration
    )
    """Configuration for how test cases should be output."""

    algorithm: Algorithm = Algorithm.DYNAMOSA
    """The algorithm that shall be used for generation."""

    statistics_output: StatisticsOutputConfiguration = dataclasses.field(
        default_factory=StatisticsOutputConfiguration
    )
    """Statistic Output configuration."""

    stopping: StoppingConfiguration = dataclasses.field(default_factory=StoppingConfiguration)
    """Stopping configuration."""

    seeding: SeedingConfiguration = dataclasses.field(default_factory=SeedingConfiguration)
    """Seeding configuration."""

    type_inference: TypeInferenceConfiguration = dataclasses.field(
        default_factory=TypeInferenceConfiguration
    )
    """Type inference configuration."""

    test_creation: TestCreationConfiguration = dataclasses.field(
        default_factory=TestCreationConfiguration
    )
    """Test creation configuration."""

    search_algorithm: SearchAlgorithmConfiguration = dataclasses.field(
        default_factory=SearchAlgorithmConfiguration
    )
    """Search algorithm configuration."""

    mio: MIOConfiguration = dataclasses.field(default_factory=MIOConfiguration)
    """Configuration used for the MIO algorithm."""

    random: RandomConfiguration = dataclasses.field(default_factory=RandomConfiguration)
    """Configuration used for the RANDOM algorithm."""

    codamosa: CodaMosaConfiguration = dataclasses.field(
        default_factory=CodaMosaConfiguration
    )
    """Condiguration used for CodaMOSA algorithm."""

    deepmosa: DeepMosaConfiguration = dataclasses.field(
        default_factory=DeepMosaConfiguration
    )
    """Condiguration used for DeepMOSA algorithm."""

    ignore_modules: list[str] = dataclasses.field(default_factory=list)
    """Ignore the modules specified here from the module analysis."""

    ignore_methods: list[str] = dataclasses.field(default_factory=list)
    """Ignore the methods specified here from the module analysis."""
