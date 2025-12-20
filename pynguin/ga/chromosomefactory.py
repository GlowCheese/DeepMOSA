#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Factory for chromosome used by the genetic algorithm."""

from abc import abstractmethod
from typing import Generic, TypeVar

from . import chromosome as chrom

T = TypeVar("T", bound=chrom.Chromosome)


class ChromosomeFactory(Generic[T]):
    """A factory that provides new chromosomes."""

    @abstractmethod
    def get_chromosome(self) -> T:
        """Create a new chromosome.

        Returns:
            A new chromosome
        """
