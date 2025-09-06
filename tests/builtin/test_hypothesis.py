import hypothesis.strategies as st
from hypothesis import given, assume, example


def plus(a, b):
    return a + b


@given(st.integers(), st.integers(), st.lists(st.floats(), min_size=3))
def test_given(a, b, s):
    # a and b are two random number
    assert a + b == b + a
    t = s.copy()
    t.sort()
    assert t == sorted(s)


@given(st.text(min_size=3))
def test_assume(s):
    n = len(s)
    # skip the test if len of s is even 
    assume(n % 2 == 1)
    assert s[(n-1)//2] == s[n//2]


@given(st.integers())
# @example(1302)  # uncomment this to see the magic :)
def test_example(x):
    assert x != 1302
