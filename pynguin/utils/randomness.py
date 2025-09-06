#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides a singleton instance of Random that can be seeded."""

from __future__ import annotations

import time
import random
import string

from typing import Any
from typing import TypeVar
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


def set_seed(seed: int | None = None):
    """Set the seed for the random number generator.
    If no seed is given, the current time in nanoseconds is used.

    Args:
        seed: The seed to use for the random number generator
    """
    random.seed(seed or time.time_ns())


def next_char() -> str:
    """Create a random printable ascii char.

    Returns:
        A random printable ascii char
    """
    return random.choice(string.printable)


def next_string(length: int) -> str:
    """Create a random string consisting of printable and with the given length.

    Args:
        length: the desired length

    Returns:
        A string of given length
    """
    return "".join(next_char() for _ in range(length))


def chance(p: float = 0.5) -> bool:
    """Return a bool.

    Args:
        p (float): Probability of returning True, in range [0.0, 1.0].
                   Defaults to 0.5 (equal chance of True/False).

    Returns:
        bool: True with probability `p`, False with probability `1 - p`.
    """
    return random.random() < p


def next_gaussian() -> float:
    """Returns the next pseudorandom.

    Use a Gaussian ("normally") distribution value with mu 0.0 and sigma 1.0.

    Returns:
        The next random number
    """
    return random.gauss(0, 1)


_T = TypeVar("_T")


def choice(sequence: Sequence[_T]) -> _T:
    """Return a random element from a non-empty sequence.

    If the sequence is empty, it raises an `IndexError`.

    Args:
        sequence: The non-empty sequence to choose from

    Returns:
        An randomly selected element of the sequence
    """
    return random.choice(sequence)


def choices(
    population: Sequence[Any],
    weights: Sequence[float] | None = None,
    *,
    cum_weights: Sequence[float] | None = None,
    k: int = 1,
) -> list[Any]:
    """Return a k sized list of population elements chosen with replacement.

    If the relative weights or cumulative weights are not specified, the selections are
    made with equal probability.

    Args:
        population: The non-empty population to choose from
        weights: A sequence of weights
        cum_weights: A sequence of cumulative weights
        k: The size of the sample

    Returns:
        A list of sampled elements from the sequence with respect to the weight
    """
    return random.choices(population, weights, cum_weights=cum_weights, k=k)


def next_byte() -> int:
    """Returns a random byte.

    Returns:
        A random byte.
    """
    return random.getrandbits(8)


def next_bytes(length: int) -> bytes:
    """Create random bytes of given length.

    Args:
        length: the length of the bytes

    Returns:
        Random bytes of given length.
    """
    return bytes(next_byte() for _ in range(length))


def shuffle(sequence: list[Any]) -> None:
    """Shuffle the given sequence in place.

    Args:
        sequence: The sequence to shuffle
    """
    random.shuffle(sequence)


def sample(
    population: Sequence[Any],
    k: int,
) -> list[Any]:
    """Return a k sized list of unique elements chosen from the population.

    If the population is smaller than k, it raises a `ValueError`.

    Args:
        population: The non-empty population to choose from
        k: The size of the sample

    Returns:
        A list of sampled elements from the sequence
    """
    return random.sample(population, k)