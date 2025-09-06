#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provide abstract selection function."""

from __future__ import annotations

import random
from math import sqrt
from typing import Generic
from typing import TypeVar
from abc import abstractmethod

from pynguin.globl import Globl

from pynguin.ga.chromosome import Chromosome


T = TypeVar("T", bound=Chromosome)


class SelectionFunction(Generic[T]):
    """Abstract base class for selection functions."""

    def __init__(self) -> None:  # noqa: D107
        self._maximize = True

    @abstractmethod
    def get_index(self, population: list[T]) -> int:
        """Provide an index within the population.

        Args:
            population: A list of chromosomes, the population

        Returns:
            The index within the population  # noqa: DAR202
        """

    def select(self, population: list[T], number: int = 1) -> list[T]:
        """Return N parents.

        Args:
            population: A list of chromosomes, the population
            number: The number of elements to select

        Returns:
            A list of chromosomes that was selected
        """
        return [population[self.get_index(population)] for _ in range(number)]

    @property
    def maximize(self):
        """Do we maximize fitness?

        Returns:
            Whether this is a maximising fitness function
        """
        return self._maximize

    @maximize.setter
    def maximize(self, new_value: bool) -> None:
        """Sets whether this is a maximising fitness function.

        Args:
            new_value: The new value
        """
        self._maximize = new_value


class RankSelection(SelectionFunction[T]):
    """Rank selection."""

    def get_index(self, population: list[T]) -> int:
        """Provides an index in the population that is chosen by rank selection.

        Make sure that the population is sorted. The fittest chromosomes have to come
        first.

        Args:
            population: A list of chromosomes to select from

        Returns:
            The index that should be used for selection
        """
        random_value = random.random()
        bias = Globl.ga_conf.rank_bias
        return int(
            len(population)
            * ((bias - sqrt(bias**2 - (4.0 * (bias - 1.0) * random_value))) / 2.0 / (bias - 1.0))
        )


class TournamentSelection(SelectionFunction[T]):
    """Tournament selection."""

    def get_index(self, population: list[T]) -> int:  # noqa: D102
        new_num = random.randrange(len(population))
        winner = new_num

        tournament_round = 0

        while tournament_round < Globl.ga_conf.tournament_size - 1:
            new_num = random.randrange(len(population))
            selected = population[new_num]

            if (
                selected.get_fitness() > population[winner].get_fitness()
                if self._maximize
                else selected.get_fitness() < population[winner].get_fitness()
            ):
                winner = new_num

            tournament_round += 1

        return winner
