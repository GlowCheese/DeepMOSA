#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides a singleton instance of Random that can be seeded."""

from __future__ import annotations

import random
import string
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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
