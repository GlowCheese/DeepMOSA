#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
import string

from hypothesis import given
import hypothesis.strategies as st
from pynguin.utils import randomness


def test_next_char_printable():
    assert randomness.next_char() in string.printable


def test_next_string_length():
    assert len(randomness.next_string(15)) == 15


def test_next_string_printable():
    rand = randomness.next_string(15)
    assert all(char in string.printable for char in rand)


def test_next_string_zero():
    rand = randomness.next_string(0)
    assert rand == ""


def test_next_gaussian():
    rand = randomness.next_gaussian()
    assert isinstance(rand, float)


def test_next_byte():
    rand = randomness.next_byte()
    assert isinstance(rand, int)
    assert 0 <= rand <= 255


def test_next_bytes_zero():
    rand = randomness.next_bytes(0)
    assert rand == b""


def test_next_bytes_fixed():
    rand = randomness.next_bytes(15)
    assert len(rand) == 15


def test_next_bytes_valid_bytes():
    rand = randomness.next_bytes(15)
    assert all(0 <= byte <= 255 for byte in rand)


def test_choice():
    sequence = ["a", "b", "c"]
    result = randomness.choice(sequence)
    assert result in {"a", "b", "c"}


def test_choices():
    sequence = ["a", "b", "c"]
    weights = [0.1, 0.5, 0.3]
    result = randomness.choices(sequence, weights)
    assert len(result) == 1
    assert result[0] in {"a", "b", "c"}
