#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides tracking of statistics for various variables and types."""

from __future__ import annotations

import json
import pprint
import queue
import time

from pathlib import Path
from typing import Any, TypeVar
from typing import TYPE_CHECKING
from vendor.custom_logger import getLogger

from pynguin.globl import Globl
from pynguin.config import StatisticsBackend
from . import statisticsbackend as sb
from . import outputvariablefactory as ovf
from pynguin.ga.chromosome import Chromosome

from pynguin.runtimevar import RuntimeVariable


if TYPE_CHECKING:
    from collections.abc import Generator


T = TypeVar("T", int, float)


class StatisticsTracker:
    """A singleton tracker for statistics."""

    def __init__(self) -> None:
        self._variables: queue.Queue = queue.Queue()
        self._search_statistics: _SearchStatistics = _SearchStatistics()

    def reset(self) -> None:
        """Reset the tracker (necessary for testing only)."""
        self._variables = queue.Queue()
        self._search_statistics = _SearchStatistics()

    def track_output_variable(self, runtime_variable: RuntimeVariable, value: Any):
        """Tracks a run-time variable for output.

        Args:
            runtime_variable: The run-time variable
            value: The value to track for the variable
        """
        self._variables.put((runtime_variable, value))

    @property
    def variables(self) -> queue.Queue:
        """Provides the queue of tracked variables.

        Returns:
            The queue of tracked variables
        """
        return self._variables

    @property
    def variables_generator(self) -> Generator[tuple[RuntimeVariable, Any]]:
        """Provides a generator.

        Yields:
            A generator for iteration
        """
        while not self._variables.empty():
            yield self._variables.get()

    @property
    def search_statistics(self) -> _SearchStatistics:
        """Provides the internal search statistics instance.

        Returns:
            The search statistics instance
        """
        return self._search_statistics

    def set_sequence_start_time(self, start_time: int) -> None:
        """This should only be called once, before any sequence data was generated.

        Args:
            start_time: the start time
        """
        self._search_statistics.set_sequence_output_variable_start_time(start_time)

    def current_individual(self, individual: Chromosome) -> None:
        """Called when a new individual is sent.

        The individual represents the best individual of the current generation.

        Args:
            individual: The best individual of the current generation
        """
        self._search_statistics.current_individual(individual)

    def set_output_variable(self, variable: sb.OutputVariable) -> None:
        """Sets an output variable to a value directly.

        Args:
            variable: The variable to be set
        """
        self._search_statistics.set_output_variable(variable)

    def update_output_variable(self, variable: sb.OutputVariable) -> None:
        """Updates an output variable with a value.

        Args:
            variable: The variable to update
        """
        self._search_statistics.update_output_variable(variable)

    def set_output_variable_for_runtime_variable(
        self, variable: RuntimeVariable, value: Any
    ) -> None:
        """Sets an output variable to a value directly.

        Args:
            variable: The variable to be set
            value: the value to be set
        """
        self._search_statistics.set_output_variable_for_runtime_variable(variable, value)

    def update_output_variable_for_runtime_variable(
        self, variable: RuntimeVariable, value: Any
    ) -> None:
        """Updates an output variable with a value directly.

        Args:
            variable: The variable to update
            value: The value to add
        """
        self._search_statistics.update_output_variable_for_runtime_variable(variable, value)

    @property
    def output_variables(self) -> dict[str, sb.OutputVariable]:
        """Provides the output variables.

        Returns:
            The output variables
        """
        return self._search_statistics.output_variables

    def write_statistics(self) -> bool:
        """Write result to disk using selected backend.

        Returns:
            True if the writing was successful
        """
        return self._search_statistics.write_statistics()


class _SearchStatistics:
    """A singleton of SearchStatistics collects all the data values reported.

    Because we cannot guarantee a singleton here without making the code too crazy,
    the only instance of this class that shall exist throughout the whole framework
    is in the `StatisticsTracker`.  The `StatisticsTracker` provides public methods
    for all public methods of this class, which delegate to its instance.
    """

    _logger = getLogger(__name__)

    def __init__(self):
        self._backend: sb.AbstractStatisticsBackend | None = sb.CSVStatisticsBackend()
        self._output_variables: dict[str, sb.OutputVariable] = {}
        self._variable_factories: dict[str, ovf.ChromosomeOutputVariableFactory] = {}
        self._sequence_output_variable_factories: dict[str, ovf.SequenceOutputVariableFactory] = {}
        self._init_factories()
        self.set_output_variable_for_runtime_variable(
            RuntimeVariable.RandomSeed, Globl.seeding_conf.seed
        )
        self._fill_sequence_output_variable_factories()
        self._start_time = time.time_ns()
        self.set_sequence_output_variable_start_time(self._start_time)
        self._best_individual: Chromosome | None = None

    def _init_factories(self) -> None:
        self._variable_factories[RuntimeVariable.Length.name] = (
            self._ChromosomeLengthOutputVariableFactory()
        )
        self._variable_factories[RuntimeVariable.Size.name] = (
            self._ChromosomeSizeOutputVariableFactory()
        )
        self._variable_factories[RuntimeVariable.Coverage.name] = (
            self._ChromosomeCoverageOutputVariableFactory()
        )
        self._variable_factories[RuntimeVariable.Fitness.name] = (
            self._ChromosomeFitnessOutputVariableFactory()
        )

    def _fill_sequence_output_variable_factories(self) -> None:
        self._sequence_output_variable_factories[RuntimeVariable.CoverageTimeline.name] = (
            self._CoverageSequenceOutputVariableFactory()
        )
        self._sequence_output_variable_factories[RuntimeVariable.SizeTimeline.name] = (
            self._SizeSequenceOutputVariableFactory()
        )
        self._sequence_output_variable_factories[RuntimeVariable.LengthTimeline.name] = (
            self._LengthSequenceOutputVariableFactory()
        )
        self._sequence_output_variable_factories[RuntimeVariable.FitnessTimeline.name] = (
            self._FitnessSequenceOutputVariableFactory()
        )
        self._sequence_output_variable_factories[RuntimeVariable.TotalExceptionsTimeline.name] = (
            ovf.DirectSequenceOutputVariableFactory.get_integer(
                RuntimeVariable.TotalExceptionsTimeline
            )
        )

    def set_sequence_output_variable_start_time(self, start_time: int) -> None:
        """Set start time for sequence data.

        Args:
            start_time: the start time
        """
        for factory in self._sequence_output_variable_factories.values():
            factory.set_start_time(start_time)

    def current_individual(self, individual: Chromosome) -> None:
        """Called when a new individual is sent.

        The individual represents the best individual of the current generation.

        Args:
            individual: The best individual of the current generation
        """
        if not self._backend:
            return

        if not isinstance(individual, Chromosome):
            self._logger.warning("SearchStatistics expected a TestSuiteChromosome")
            return

        self._logger.debug("Received individual")
        self._best_individual = individual
        for variable_factory in self._variable_factories.values():
            self.set_output_variable(variable_factory.get_variable(individual))
        for seq_variable_factory in self._sequence_output_variable_factories.values():
            seq_variable_factory.update(individual)

    def set_output_variable(self, variable: sb.OutputVariable) -> None:
        """Sets an output variable to a value directly.

        Args:
            variable: The variable to be set
        """
        if variable.name in self._sequence_output_variable_factories:
            var = self._sequence_output_variable_factories[variable.name]
            assert isinstance(var, ovf.DirectSequenceOutputVariableFactory)
            var.set_value(variable.value)
        else:
            self._output_variables[variable.name] = variable

    def update_output_variable(self, variable: sb.OutputVariable) -> None:
        """Updates an output variable with a new value.

        Args:
            variable: The variable to update
        """
        if variable.name not in self._sequence_output_variable_factories:
            raise AssertionError("Can only be called on sequence variable.")
        var = self._sequence_output_variable_factories[variable.name]
        assert isinstance(var, ovf.DirectSequenceOutputVariableFactory)
        var.update_value(variable.value)

    def set_output_variable_for_runtime_variable(
        self, variable: RuntimeVariable, value: Any
    ) -> None:
        """Sets an output variable to a value directly.

        Args:
            variable: The variable to be set
            value: the value to be set
        """
        self.set_output_variable(sb.OutputVariable(name=variable.name, value=value))

    def update_output_variable_for_runtime_variable(
        self, variable: RuntimeVariable, value: Any
    ) -> None:
        """Updates an output variable with a new value.

        Args:
            variable: The variable to update
            value: The value to add
        """
        self.update_output_variable(sb.OutputVariable(name=variable.name, value=value))

    @property
    def output_variables(self) -> dict[str, sb.OutputVariable]:
        """Provides the output variables.

        Returns:
            The output variables
        """
        return self._output_variables

    def _get_output_variables(
        self, individual, *, skip_missing: bool = True
    ) -> dict[str, sb.OutputVariable]:
        output_variables_map: dict[str, sb.OutputVariable] = {}

        for variable in Globl.output_variables:
            variable_name = variable.name
            if variable_name in self._output_variables:
                # Values directly sent
                output_variables_map[variable_name] = self._output_variables[variable_name]
            elif variable_name in self._variable_factories:
                # Values extracted from the individual
                output_variables_map[variable_name] = self._variable_factories[
                    variable_name
                ].get_variable(individual)
            elif variable_name in self._sequence_output_variable_factories:
                # Time related values, which will be expanded in a list of values
                # through time
                assert Globl.stopping_conf.maximum_search_time >= 0, (
                    "Tracking sequential variables is only possible when using "
                    "maximum search time as a stopping condition"
                )
                for var in self._sequence_output_variable_factories[
                    variable_name
                ].get_output_variables():
                    output_variables_map[var.name] = var

                # HACK: disable area under curve
                # # For every time-series variable, we compute the area under curve, too
                # auc_variable = self._sequence_output_variable_factories[
                #     variable_name
                # ].area_under_curve_output_variable
                # output_variables_map[auc_variable.name] = auc_variable
                # # Additionally, add a normalised version of the area under curve
                # norm_auc_variable = self._sequence_output_variable_factories[
                #     variable_name
                # ].normalised_area_under_curve_output_variable
                # output_variables_map[norm_auc_variable.name] = norm_auc_variable
            elif skip_missing:
                # if variable does not exist, return an empty value instead
                output_variables_map[variable_name] = sb.OutputVariable(
                    name=variable_name, value=""
                )
            else:
                self._logger.error("No obtained value for output variable %s", variable_name)
                return {}

        return output_variables_map

    def write_statistics(self) -> bool:
        """Write result to disk using selected backend.

        Returns:
            True if the writing was successful
        """
        self._logger.info("Writing statistics")
        # reinitialise backend to be sure we got the correct one, prone to failure
        # due to global-object pattern otherwise.
        self._backend = sb.CSVStatisticsBackend()
        if not self._backend:
            return False

        self._output_variables[RuntimeVariable.TotalTime.name] = sb.OutputVariable(
            name=RuntimeVariable.TotalTime.name,
            value=time.time_ns() - self._start_time,
        )

        # self._output_variables

        if not self._best_individual:
            self._logger.error(
                "No statistics has been saved because Pynguin failed to generate any test case"
            )
            return False

        individual = self._best_individual
        output_variables_map = self._get_output_variables(individual)
        self._backend.write_data(output_variables_map)

        if Globl.statistics_conf.statistics_backend == StatisticsBackend.CSV:
            report_dir = Path(Globl.report_dir).resolve()
            if "SignatureInfos" in output_variables_map:
                obj = json.loads(output_variables_map["SignatureInfos"].value)
                output_file = report_dir / "signature-infos.json"
                with output_file.open(mode="w") as f:
                    json.dump(obj, f)
            cfg_file = report_dir / "pynguin-config.txt"
            cfg_file.write_text(pprint.pformat(repr(Globl.conf)))
        return True

    class _ChromosomeLengthOutputVariableFactory(ovf.ChromosomeOutputVariableFactory):
        def __init__(self) -> None:
            super().__init__(RuntimeVariable.Length)

        def get_data(self, individual: Chromosome) -> int:
            return individual.length()

    class _ChromosomeSizeOutputVariableFactory(ovf.ChromosomeOutputVariableFactory):
        def __init__(self) -> None:
            super().__init__(RuntimeVariable.Size)

        def get_data(self, individual: Chromosome) -> int:
            return individual.size()

    class _ChromosomeCoverageOutputVariableFactory(ovf.ChromosomeOutputVariableFactory):
        def __init__(self) -> None:
            super().__init__(RuntimeVariable.Coverage)

        def get_data(self, individual: Chromosome) -> float:
            return individual.get_coverage()

    class _ChromosomeFitnessOutputVariableFactory(ovf.ChromosomeOutputVariableFactory):
        def __init__(self) -> None:
            super().__init__(RuntimeVariable.Fitness)

        def get_data(self, individual: Chromosome) -> float:
            return individual.get_fitness()

    class _CoverageSequenceOutputVariableFactory(ovf.DirectSequenceOutputVariableFactory):
        def __init__(self) -> None:
            super().__init__(RuntimeVariable.CoverageTimeline, 0.0)

        def get_value(self, individual: Chromosome) -> float:
            return individual.get_coverage()
        
        def _get_time_line_value(self, index: int) -> tuple[T, T]:
            if not self._time_stamps:
                # No data, if this is even possible.
                return 0, 0
            interval = Globl.statistics_conf.timeline_interval
            preferred_time = interval * index

            for i in range(len(self._time_stamps)):
                # find the first stamp that is following the time we would like to get
                # the value for
                stamp = self._time_stamps[i]
                if stamp < preferred_time:
                    continue

                if i == 0:
                    # it is the first element, just use it as value
                    return self._values[i], self._time_stamps[i]

                if not Globl.statistics_conf.timeline_interpolation:
                    # if we do not want to interpolate, return last observed value
                    return self._values[i-1], self._time_stamps[i-1]

                # interpolate the value, since we do not have the value for the exact
                # time we want
                time_delta = self._time_stamps[i] - self._time_stamps[i - 1]
                if time_delta > 0:
                    value_delta = float(self._values[i]) - float(self._values[i - 1])
                    ratio = value_delta / time_delta
                    diff = preferred_time - self._time_stamps[i - 1]
                    return float(self._values[i - 1]) + diff * ratio, self._time_stamps[i-1] 

            # no time stamp was higher, just use the last value seen
            return self._values[-1], self._time_stamps[-1]
        
        def get_variable_names_indices(self) -> list[tuple[int, str]]:
            """Provides a list of variable names.

            Returns:
                A list of pairs consisting of variable names and their index.
            """
            return [
                (i + 1, f"CT_{i + 1}")
                for i in range(self._calculate_number_of_intervals())
            ]

        def get_output_variables(self) -> list[sb.OutputVariable[tuple[T, T]]]:
            """Provides the output variables.

            Returns:
                A list of output variables
            """
            return [
                sb.OutputVariable(
                    name=variable_name, 
                    value=self._get_time_line_value(variable_index)
                )
                for variable_index, variable_name in self.get_variable_names_indices()
            ]
        
    # class _TimelineSequenceOutputVariableFactory(ovf.DirectSequenceOutputVariableFactory):
    #     def __init__(self) -> None:
    #         super().__init__(RuntimeVariable.Timeline, 0.0)

    #     def get_value(self, individual: Chromosome) -> float:
    #         return individual.get_coverage()

    class _SizeSequenceOutputVariableFactory(ovf.DirectSequenceOutputVariableFactory):
        def __init__(self) -> None:
            super().__init__(RuntimeVariable.SizeTimeline, 0)

        def get_value(self, individual: Chromosome) -> int:
            return individual.size()

    class _LengthSequenceOutputVariableFactory(ovf.DirectSequenceOutputVariableFactory):
        def __init__(self) -> None:
            super().__init__(RuntimeVariable.LengthTimeline, 0)

        def get_value(self, individual: Chromosome) -> int:
            return individual.length()

    class _FitnessSequenceOutputVariableFactory(ovf.DirectSequenceOutputVariableFactory):
        def __init__(self) -> None:
            super().__init__(RuntimeVariable.FitnessTimeline, 0.0)

        def get_value(self, individual: Chromosome) -> float:
            return individual.get_fitness()
