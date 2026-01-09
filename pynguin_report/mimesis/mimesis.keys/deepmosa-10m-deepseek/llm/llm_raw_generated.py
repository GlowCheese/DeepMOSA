####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_romanize_returns_callable_for_supported_locale():
    from mimesis.enums import Locale
    from mimesis.keys import romanize
    result = romanize(Locale.RU)
    assert callable(result)

def test_romanize_raises_value_error_for_unsupported_locale():
    from mimesis.enums import Locale
    from mimesis.keys import romanize
    try:
        romanize(Locale.EN)
    except ValueError as e:
        assert "Romanization is not available for:" in str(e)

def test_romanize_closure_romanizes_russian_text():
    from mimesis.enums import Locale
    from mimesis.keys import romanize
    romanize_func = romanize(Locale.RU)
    result = romanize_func("Привет")
    assert result == "Privet"

def test_romanize_closure_romanizes_ukrainian_text():
    from mimesis.enums import Locale
    from mimesis.keys import romanize
    romanize_func = romanize(Locale.UK)
    result = romanize_func("Привіт")
    assert result == "Pryvit"

def test_romanize_closure_romanizes_kazakh_text():
    from mimesis.enums import Locale
    from mimesis.keys import romanize
    romanize_func = romanize(Locale.KK)
    result = romanize_func("Сәлем")
    assert result == "Sälem"

def test_romanize_closure_raises_type_error_for_non_string_input():
    from mimesis.enums import Locale
    from mimesis.keys import romanize
    romanize_func = romanize(Locale.RU)
    try:
        romanize_func(123)
    except TypeError as e:
        assert "romanize() requires a string, got" in str(e)

def test_romanize_accepts_locale_string():
    from mimesis.keys import romanize
    romanize_func = romanize("ru")
    result = romanize_func("Привет")
    assert result == "Privet"

def test_romanize_raises_locale_error_for_invalid_locale_string():
    from mimesis.exceptions import LocaleError
    from mimesis.keys import romanize
    try:
        romanize("invalid")
    except LocaleError as e:
        assert "invalid" in str(e)

def test_romanize_raises_locale_error_for_invalid_locale_type():
    from mimesis.exceptions import LocaleError
    from mimesis.keys import romanize
    try:
        romanize(123)
    except LocaleError as e:
        assert "123" in str(e)


# LLM-generated content at query #2
#--------------------------

def test_wrap_default():
    wrapper = wrap()
    result = wrapper("test")
    assert result == "<test>"

def test_wrap_custom():
    wrapper = wrap("[", "]")
    result = wrapper("test")
    assert result == "[test]"

def test_wrap_empty_string():
    wrapper = wrap("(", ")")
    result = wrapper("")
    assert result == "()"

def test_wrap_non_string_raises_typeerror():
    wrapper = wrap()
    try:
        wrapper(123)
        assert False
    except TypeError as e:
        assert str(e) == "wrap() requires a string, got int"

def test_wrap_none_raises_typeerror():
    wrapper = wrap()
    try:
        wrapper(None)
        assert False
    except TypeError as e:
        assert str(e) == "wrap() requires a string, got NoneType"

def test_wrap_list_raises_typeerror():
    wrapper = wrap()
    try:
        wrapper(["a", "b"])
        assert False
    except TypeError as e:
        assert str(e) == "wrap() requires a string, got list"

def test_wrap_with_special_characters():
    wrapper = wrap("***", "***")
    result = wrapper("test")
    assert result == "***test***"

def test_wrap_multiple_calls():
    wrapper = wrap("{", "}")
    result1 = wrapper("first")
    result2 = wrapper("second")
    assert result1 == "{first}"
    assert result2 == "{second}"


# LLM-generated content at query #3
#--------------------------

def test_pipe_with_key_functions():
    def add_prefix(prefix):
        def inner(value, random=None):
            return prefix + value
        return inner
    def to_uppercase(value, random=None):
        return value.upper()
    key_func = pipe(add_prefix("test-"), to_uppercase)
    result = key_func("hello", None)
    assert result == "TEST-HELLO"

def test_pipe_with_single_function():
    def double(value, random=None):
        return value * 2
    key_func = pipe(double)
    result = key_func(5, None)
    assert result == 10

def test_pipe_with_multiple_functions_no_random():
    def increment(value):
        return value + 1
    def square(value):
        return value * value
    key_func = pipe(increment, square)
    result = key_func(3, None)
    assert result == 16

def test_pipe_with_mixed_functions():
    def add_suffix(suffix):
        def inner(value, random=None):
            return value + suffix
        return inner
    def reverse(value):
        return value[::-1]
    key_func = pipe(add_suffix("!!"), reverse)
    result = key_func("hello", None)
    assert result == "!!olleh"

def test_pipe_with_random_parameter():
    def add_random(value, random):
        return value + random.randint(1, 10)
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    key_func = pipe(add_random)
    result = key_func(10, mock_random)
    assert result == 15

def test_pipe_with_nested_pipes():
    def multiply_by(factor):
        def inner(value, random=None):
            return value * factor
        return inner
    inner_pipe = pipe(multiply_by(2), multiply_by(3))
    outer_pipe = pipe(inner_pipe, multiply_by(4))
    result = outer_pipe(5, None)
    assert result == 120

def test_pipe_with_empty_functions():
    key_func = pipe()
    result = key_func("test", None)
    assert result == "test"

def test_pipe_preserves_random_across_functions():
    calls = []
    def track_random(value, random):
        calls.append(random)
        return value
    mock_random = Random()
    key_func = pipe(track_random, track_random)
    result = key_func("data", mock_random)
    assert calls[0] is mock_random
    assert calls[1] is mock_random
    assert result == "data"


# LLM-generated content at query #4
#--------------------------

def test_pipe_applies_functions_in_sequence():
    def add_one(x):
        return x + 1
    def double(x):
        return x * 2
    piped = pipe(add_one, double)
    result = piped(5)
    assert result == 12

def test_pipe_handles_key_func_with_random():
    def add_random(x, random):
        return x + random.randint(1, 10)
    def double(x):
        return x * 2
    piped = pipe(add_random, double)
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    result = piped(5, mock_random)
    assert result == 20

def test_pipe_handles_mix_of_functions():
    def add_one(x):
        return x + 1
    def add_random(x, random):
        return x + random.randint(1, 10)
    def triple(x):
        return x * 3
    piped = pipe(add_one, add_random, triple)
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    result = piped(2, mock_random)
    assert result == 24

def test_pipe_with_single_function():
    def square(x):
        return x * x
    piped = pipe(square)
    result = piped(4)
    assert result == 16

def test_pipe_with_string_operations():
    def lower(s):
        return s.lower()
    def replace_space(s):
        return s.replace(' ', '-')
    def prefix(s):
        return 'user-' + s
    piped = pipe(lower, replace_space, prefix)
    result = piped('John Doe')
    assert result == 'user-john-doe'


# LLM-generated content at query #5
#--------------------------

def test_pipe_key_function_with_random_parameter():
    def add_prefix(prefix):
        def inner(value, random=None):
            return prefix + value
        return inner
    def to_upper(value, random=None):
        return value.upper()
    def repeat(value, random=None):
        if random is None:
            return value + value
        return value * random.randint(1, 3)
    from mimesis.random import Random
    random_instance = Random()
    key_func = pipe(to_upper, add_prefix("TEST-"), repeat)
    result = key_func("hello", random_instance)
    assert isinstance(result, str)
    assert result.startswith("TEST-")
    assert "HELLO" in result

def test_pipe_key_function_without_random_parameter():
    def add_suffix(suffix):
        def inner(value):
            return value + suffix
        return inner
    def capitalize(value):
        return value.capitalize()
    key_func = pipe(capitalize, add_suffix("!!!"))
    result = key_func("test")
    assert result == "Test!!!"

def test_pipe_key_function_mixed_parameters():
    def func_with_random(value, random=None):
        if random is not None:
            return value * random.randint(1, 2)
        return value
    def func_without_random(value):
        return value.upper()
    from mimesis.random import Random
    random_instance = Random()
    key_func = pipe(func_without_random, func_with_random)
    result_with_random = key_func("ab", random_instance)
    assert result_with_random in ["AB", "ABAB"]
    result_without_random = key_func("ab", None)
    assert result_without_random == "AB"

def test_pipe_key_function_single_function():
    def double(value):
        return value * 2
    key_func = pipe(double)
    result = key_func(5)
    assert result == 10

def test_pipe_key_function_multiple_functions():
    def increment(value):
        return value + 1
    def square(value):
        return value * value
    key_func = pipe(increment, square, increment)
    result = key_func(2)
    assert result == 10


# LLM-generated content at query #6
#--------------------------

def test_maybe_returns_value_with_probability():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    key_func = maybe("special", 0.7)
    result = key_func("default", mock_random)
    assert result == "special"

def test_maybe_returns_first_argument_with_probability():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[0]]
    key_func = maybe("special", 0.7)
    result = key_func("default", mock_random)
    assert result == "default"

def test_maybe_with_probability_zero():
    mock_random = Random()
    key_func = maybe("special", 0.0)
    result = key_func("default", mock_random)
    assert result == "default"

def test_maybe_with_probability_one():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    key_func = maybe("special", 1.0)
    result = key_func("default", mock_random)
    assert result == "special"

def test_maybe_with_negative_probability():
    mock_random = Random()
    key_func = maybe("special", -0.5)
    result = key_func("default", mock_random)
    assert result == "default"

def test_maybe_with_probability_greater_than_one():
    mock_random = Random()
    key_func = maybe("special", 1.5)
    result = key_func("default", mock_random)
    assert result == "default"

def test_maybe_with_different_value_types():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    key_func = maybe(123, 0.8)
    result = key_func(456, mock_random)
    assert result == 123

def test_maybe_with_none_value():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    key_func = maybe(None, 0.6)
    result = key_func("not_none", mock_random)
    assert result is None

def test_maybe_with_complex_object():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    complex_obj = {"key": "value"}
    key_func = maybe(complex_obj, 0.9)
    result = key_func("simple", mock_random)
    assert result == complex_obj


# LLM-generated content at query #7
#--------------------------

def test_apply_if_condition_true():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(5)
    assert result == 10

def test_apply_if_condition_false_without_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(-5)
    assert result == -5

def test_apply_if_condition_false_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    func = apply_if(condition, transform, otherwise)
    result = func(-5)
    assert result == -15

def test_apply_if_condition_true_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    func = apply_if(condition, transform, otherwise)
    result = func(5)
    assert result == 10

def test_apply_if_with_string_condition_true():
    condition = lambda s: len(s) > 3
    transform = str.upper
    func = apply_if(condition, transform)
    result = func("hello")
    assert result == "HELLO"

def test_apply_if_with_string_condition_false_without_otherwise():
    condition = lambda s: len(s) > 3
    transform = str.upper
    func = apply_if(condition, transform)
    result = func("hi")
    assert result == "hi"

def test_apply_if_with_string_condition_false_with_otherwise():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    func = apply_if(condition, transform, otherwise)
    result = func("HI")
    assert result == "hi"

def test_apply_if_with_none_otherwise():
    condition = lambda x: x is None
    transform = lambda x: "missing"
    func = apply_if(condition, transform)
    result = func(None)
    assert result == "missing"

def test_apply_if_with_false_condition_and_none_otherwise():
    condition = lambda x: x is None
    transform = lambda x: "missing"
    func = apply_if(condition, transform)
    result = func(42)
    assert result == 42

def test_apply_if_with_otherwise_as_none():
    condition = lambda x: x % 2 == 0
    transform = lambda x: x // 2
    otherwise = None
    func = apply_if(condition, transform, otherwise)
    result = func(3)
    assert result == 3


# LLM-generated content at query #8
#--------------------------

def test_pipe_key_function_with_random_parameter():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    def func1(x, random=None):
        return x + 1
    def func2(x, random=None):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, random_instance)
    assert result == 12

def test_pipe_key_function_without_random_parameter():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    def func1(x):
        return x + 1
    def func2(x):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, random_instance)
    assert result == 12

def test_pipe_key_function_mixed_parameters():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    def func1(x, random=None):
        return x + random.randint(1, 10)
    def func2(x):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, random_instance)
    assert result == (5 + random_instance.randint(1, 10)) * 2

def test_pipe_key_function_single_func():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    def func1(x, random=None):
        return x + 1
    key_func = pipe(func1)
    result = key_func(5, random_instance)
    assert result == 6

def test_pipe_key_function_no_random_passed():
    from mimesis.keys import pipe
    def func1(x):
        return x + 1
    def func2(x):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5)
    assert result == 12


# LLM-generated content at query #9
#--------------------------

def test_apply_if_condition_true_without_otherwise():
    result_func = apply_if(lambda x: x > 0, lambda x: x * 2)
    assert result_func(5) == 10

def test_apply_if_condition_true_with_otherwise():
    result_func = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)
    assert result_func(5) == 10

def test_apply_if_condition_false_without_otherwise():
    result_func = apply_if(lambda x: x > 0, lambda x: x * 2)
    assert result_func(-5) == -5

def test_apply_if_condition_false_with_otherwise():
    result_func = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)
    assert result_func(-5) == -15

def test_apply_if_condition_true_with_string():
    result_func = apply_if(lambda x: len(x) > 3, str.upper, str.lower)
    assert result_func('word') == 'WORD'

def test_apply_if_condition_false_with_string():
    result_func = apply_if(lambda x: len(x) > 3, str.upper, str.lower)
    assert result_func('hi') == 'hi'

def test_apply_if_condition_true_with_none_otherwise():
    result_func = apply_if(lambda x: x is not None, lambda x: x + 1)
    assert result_func(10) == 11

def test_apply_if_condition_false_with_none_otherwise():
    result_func = apply_if(lambda x: x is not None, lambda x: x + 1)
    assert result_func(None) == None


# LLM-generated content at query #10
#--------------------------

def test_apply_if_with_otherwise():
    result = apply_if(lambda x: x > 5, lambda x: x * 2, lambda x: x + 1)(3)
    assert result == 4


# LLM-generated content at query #11
#--------------------------

def test_pipe_with_key_functions():
    def add_one(x, random=None):
        return x + 1
    def double(x, random=None):
        return x * 2
    func = pipe(add_one, double)
    result = func(5, None)
    assert result == 12

def test_pipe_with_regular_functions():
    def add_one(x):
        return x + 1
    def double(x):
        return x * 2
    func = pipe(add_one, double)
    result = func(5, None)
    assert result == 12

def test_pipe_mixed_functions():
    def add_one(x):
        return x + 1
    def double(x, random=None):
        return x * 2
    func = pipe(add_one, double)
    result = func(5, None)
    assert result == 12

def test_pipe_single_function():
    def square(x, random=None):
        return x * x
    func = pipe(square)
    result = func(4, None)
    assert result == 16

def test_pipe_with_random_argument():
    def add_random(x, random):
        return x + random.randint(1, 1)
    def double(x, random=None):
        return x * 2
    from mimesis.random import Random
    rnd = Random()
    func = pipe(add_random, double)
    result = func(5, rnd)
    assert result == 12

def test_pipe_string_operations():
    def lower(s, random=None):
        return s.lower()
    def strip(s, random=None):
        return s.strip()
    func = pipe(lower, strip)
    result = func("  HELLO  ", None)
    assert result == "hello"

def test_pipe_empty_functions():
    func = pipe()
    result = func(42, None)
    assert result == 42


# LLM-generated content at query #12
#--------------------------

def test_pipe_key_handles_random_parameter_correctly():
    def func1(x, random=None):
        return x + 1
    def func2(x):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, random=Random())
    assert result == 12


# LLM-generated content at query #13
#--------------------------

def test_pipe_key_with_random_parameter():
    def func1(x, random=None):
        return x + 1
    def func2(x, random=None):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, Random())
    assert result == 12


# LLM-generated content at query #14
#--------------------------

def test_apply_if_condition_true():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)(5)
    assert result == 10


# LLM-generated content at query #15
#--------------------------

def test_apply_if_condition_true():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(5)
    assert result == 10

def test_apply_if_condition_false_without_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(-5)
    assert result == -5

def test_apply_if_condition_false_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    func = apply_if(condition, transform, otherwise)
    result = func(-5)
    assert result == -15

def test_apply_if_condition_true_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    func = apply_if(condition, transform, otherwise)
    result = func(5)
    assert result == 10

def test_apply_if_with_strings():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    func = apply_if(condition, transform, otherwise)
    result = func("hello")
    assert result == "HELLO"

def test_apply_if_with_strings_condition_false():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    func = apply_if(condition, transform, otherwise)
    result = func("hi")
    assert result == "hi"

def test_apply_if_with_strings_condition_false_with_otherwise():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    func = apply_if(condition, transform, otherwise)
    result = func("hi")
    assert result == "hi"

def test_apply_if_with_none_otherwise():
    condition = lambda x: x is None
    transform = lambda x: "missing"
    func = apply_if(condition, transform)
    result = func(None)
    assert result == "missing"

def test_apply_if_with_false_condition_and_none_otherwise():
    condition = lambda x: x is None
    transform = lambda x: "missing"
    func = apply_if(condition, transform)
    result = func(42)
    assert result == 42

def test_apply_if_with_complex_condition():
    condition = lambda lst: len(lst) > 2 and sum(lst) > 10
    transform = lambda lst: max(lst)
    otherwise = lambda lst: min(lst)
    func = apply_if(condition, transform, otherwise)
    result = func([1, 2, 3, 4])
    assert result == 4

def test_apply_if_with_complex_condition_false():
    condition = lambda lst: len(lst) > 2 and sum(lst) > 10
    transform = lambda lst: max(lst)
    otherwise = lambda lst: min(lst)
    func = apply_if(condition, transform, otherwise)
    result = func([1, 2])
    assert result == 1


# LLM-generated content at query #16
#--------------------------

def test_pipe_with_key_functions():
    from mimesis.keys import pipe
    from mimesis.random import Random
    rnd = Random()
    def add_one(x, random=None):
        return x + 1
    def double(x, random=None):
        return x * 2
    piped = pipe(add_one, double)
    result = piped(5, rnd)
    assert result == 12

def test_pipe_with_mixed_functions():
    from mimesis.keys import pipe
    from mimesis.random import Random
    rnd = Random()
    def add_random(x, random):
        return x + random.randint(1, 10)
    def square(x):
        return x * x
    piped = pipe(add_random, square)
    result = piped(2, rnd)
    assert result >= 9 and result <= 144

def test_pipe_with_string_functions():
    from mimesis.keys import pipe
    from mimesis.random import Random
    rnd = Random()
    def lower(s, random=None):
        return s.lower()
    def reverse(s, random=None):
        return s[::-1]
    piped = pipe(lower, reverse)
    result = piped("Hello", rnd)
    assert result == "olleh"

def test_pipe_with_no_functions():
    from mimesis.keys import pipe
    from mimesis.random import Random
    rnd = Random()
    piped = pipe()
    result = piped("test", rnd)
    assert result == "test"

def test_pipe_with_single_function():
    from mimesis.keys import pipe
    from mimesis.random import Random
    rnd = Random()
    def identity(x, random=None):
        return x
    piped = pipe(identity)
    result = piped(42, rnd)
    assert result == 42

def test_pipe_with_random_parameter_ignored():
    from mimesis.keys import pipe
    from mimesis.random import Random
    rnd = Random()
    def func_no_random(x):
        return x * 3
    piped = pipe(func_no_random)
    result = piped(3, rnd)
    assert result == 9

def test_pipe_sequence_of_transformations():
    from mimesis.keys import pipe
    from mimesis.random import Random
    rnd = Random()
    def append_a(s, random=None):
        return s + "a"
    def append_b(s, random=None):
        return s + "b"
    piped = pipe(append_a, append_b)
    result = piped("start_", rnd)
    assert result == "start_ab"


# LLM-generated content at query #17
#--------------------------

def test_pipe_handles_functions_without_random_parameter():
    def func1(x):
        return x + 1
    def func2(x):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, None)
    assert result == 12


# LLM-generated content at query #18
#--------------------------

def test_pipe_with_single_function():
    def add_one(x):
        return x + 1
    key_func = pipe(add_one)
    result = key_func(5, None)
    assert result == 6

def test_pipe_with_multiple_functions():
    def add_one(x):
        return x + 1
    def multiply_by_two(x):
        return x * 2
    key_func = pipe(add_one, multiply_by_two)
    result = key_func(5, None)
    assert result == 12

def test_pipe_with_functions_using_random():
    def add_random(x, random):
        return x + random.randint(1, 10)
    random_instance = Random()
    key_func = pipe(add_random)
    result = key_func(5, random_instance)
    assert 6 <= result <= 15

def test_pipe_with_mixed_functions():
    def add_one(x):
        return x + 1
    def multiply_random(x, random):
        return x * random.randint(1, 5)
    random_instance = Random()
    key_func = pipe(add_one, multiply_random)
    result = key_func(5, random_instance)
    assert 6 <= result <= 30

def test_pipe_with_string_functions():
    def uppercase(s):
        return s.upper()
    def add_exclamation(s):
        return s + '!'
    key_func = pipe(uppercase, add_exclamation)
    result = key_func('hello', None)
    assert result == 'HELLO!'

def test_pipe_with_no_functions():
    key_func = pipe()
    result = key_func('test', None)
    assert result == 'test'

def test_pipe_with_nested_pipe():
    def add_one(x):
        return x + 1
    inner_pipe = pipe(add_one, add_one)
    outer_pipe = pipe(inner_pipe, add_one)
    result = outer_pipe(5, None)
    assert result == 8


# LLM-generated content at query #19
#--------------------------

def test_pipe_applies_functions_in_sequence():
    def add_one(x):
        return x + 1
    def multiply_two(x):
        return x * 2
    key_func = pipe(add_one, multiply_two)
    result = key_func(5)
    assert result == 12

def test_pipe_with_random_parameter():
    def add_random(x, random):
        return x + random.randint(1, 10)
    def multiply_two(x):
        return x * 2
    key_func = pipe(add_random, multiply_two)
    mock_random = Random()
    mock_random.randint = lambda a, b: 3
    result = key_func(5, mock_random)
    assert result == 16

def test_pipe_with_mixed_functions():
    def to_upper(x):
        return x.upper()
    def add_exclamation(x):
        return x + "!"
    key_func = pipe(to_upper, add_exclamation)
    result = key_func("hello")
    assert result == "HELLO!"

def test_pipe_single_function():
    def square(x):
        return x * x
    key_func = pipe(square)
    result = key_func(4)
    assert result == 16

def test_pipe_no_functions():
    key_func = pipe()
    result = key_func("test")
    assert result == "test"


# LLM-generated content at query #20
#--------------------------

def test_pipe_handles_functions_without_random_parameter():
    def add_one(x):
        return x + 1
    def double(x):
        return x * 2
    piped = pipe(add_one, double)
    result = piped(5, None)
    assert result == 12


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_hash_with_supported_algorithm():
    hash_func = hash_with('sha256')
    result = hash_func('hello')
    expected = '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    assert result == expected

def test_hash_with_unsupported_algorithm():
    try:
        hash_with('unsupported_algo')
        assert False
    except ValueError as e:
        assert str(e) == "Unsupported hash algorithm: unsupported_algo"

def test_hash_with_non_string_input():
    hash_func = hash_with('md5')
    try:
        hash_func(123)
        assert False
    except TypeError as e:
        assert str(e) == "hash_with() requires a string, got int"

def test_hash_with_empty_string():
    hash_func = hash_with('sha1')
    result = hash_func('')
    expected = 'da39a3ee5e6b4b0d3255bfef95601890afd80709'
    assert result == expected

def test_hash_with_different_algorithms():
    hash_func_md5 = hash_with('md5')
    result_md5 = hash_func_md5('test')
    expected_md5 = '098f6bcd4621d373cade4e832627b4f6'
    assert result_md5 == expected_md5
    hash_func_sha512 = hash_with('sha512')
    result_sha512 = hash_func_sha512('test')
    expected_sha512 = 'ee26b0dd4af7e749aa1a8ee3c10ae9923f618980772e473f8819a5d4940e0db27ac185f8a0e1d5f84f88bc887fd67b143732c304cc5fa9ad8e6f57f50028a8ff'
    assert result_sha512 == expected_sha512


# LLM-generated content at query #2
#--------------------------

def test_join_with_default_separator():
    result = join()(['a', 'b', 'c'])
    assert result == 'a, b, c'

def test_join_with_custom_separator():
    result = join(' | ')(['pci', 'promise', 'excel'])
    assert result == 'pci | promise | excel'

def test_join_with_empty_list():
    result = join(';')([])
    assert result == ''

def test_join_with_single_item():
    result = join('-')(['single'])
    assert result == 'single'

def test_join_with_non_string_items():
    result = join(' ')([1, 2, 3])
    assert result == '1 2 3'

def test_join_raises_type_error_for_non_iterable():
    try:
        join()(123)
        assert False
    except TypeError as e:
        assert str(e) == "join() requires iterable, got int"

def test_join_with_tuple():
    result = join('-')(('a', 'b', 'c'))
    assert result == 'a-b-c'

def test_join_with_generator():
    result = join(',')(range(3))
    assert result == '0,1,2'


# LLM-generated content at query #3
#--------------------------

def test_maybe_returns_value_with_probability():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    closure = maybe("special", 0.7)
    result = closure("default", mock_random)
    assert result == "special"

def test_maybe_returns_first_argument_with_probability():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[0]]
    closure = maybe("special", 0.7)
    result = closure("default", mock_random)
    assert result == "default"

def test_maybe_probability_zero_or_negative():
    mock_random = Random()
    closure = maybe("special", 0.0)
    result = closure("default", mock_random)
    assert result == "default"
    closure2 = maybe("special", -0.5)
    result2 = closure2("default", mock_random)
    assert result2 == "default"

def test_maybe_probability_one():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    closure = maybe("special", 1.0)
    result = closure("default", mock_random)
    assert result == "special"

def test_maybe_with_different_value_types():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    closure = maybe(42, 0.5)
    result = closure(100, mock_random)
    assert result == 42

def test_maybe_closure_uses_random_instance():
    random_instance = Random()
    closure = maybe("value", 0.5)
    result = closure("other", random_instance)
    assert result in ["value", "other"]


# LLM-generated content at query #4
#--------------------------

def test_pipe_with_key_functions_using_random():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    def add_one(x, r):
        return x + 1
    def multiply_two(x, r):
        return x * 2
    key_func = pipe(add_one, multiply_two)
    result = key_func(5, random_instance)
    assert result == 12

def test_pipe_with_key_functions_without_random():
    from mimesis.keys import pipe
    def add_one(x):
        return x + 1
    def multiply_two(x):
        return x * 2
    key_func = pipe(add_one, multiply_two)
    result = key_func(5)
    assert result == 12

def test_pipe_with_mixed_key_functions():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    def add_one(x):
        return x + 1
    def multiply_two(x, r):
        return x * 2
    key_func = pipe(add_one, multiply_two)
    result = key_func(5, random_instance)
    assert result == 12

def test_pipe_with_string_transformations():
    from mimesis.keys import pipe
    def lower(s):
        return s.lower()
    def strip(s):
        return s.strip()
    key_func = pipe(lower, strip)
    result = key_func("  HELLO  ")
    assert result == "hello"

def test_pipe_with_empty_functions():
    from mimesis.keys import pipe
    key_func = pipe()
    result = key_func("test")
    assert result == "test"

def test_pipe_with_single_function():
    from mimesis.keys import pipe
    def square(x):
        return x * x
    key_func = pipe(square)
    result = key_func(4)
    assert result == 16


# LLM-generated content at query #5
#--------------------------

def test_romanize_returns_callable_for_supported_locale():
    from mimesis.enums import Locale
    from mimesis.keys import romanize
    romanizer = romanize(Locale.RU)
    result = romanizer("Привет")
    assert isinstance(result, str)

def test_romanize_raises_value_error_for_unsupported_locale():
    from mimesis.enums import Locale
    from mimesis.keys import romanize
    try:
        romanize(Locale.EN)
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {Locale.EN}"

def test_romanize_raises_type_error_when_non_string_passed_to_closure():
    from mimesis.enums import Locale
    from mimesis.keys import romanize
    romanizer = romanize(Locale.UK)
    try:
        romanizer(123)
    except TypeError as e:
        assert str(e) == "romanize() requires a string, got int"

def test_romanize_works_with_string_locale():
    from mimesis.keys import romanize
    romanizer = romanize("ru")
    result = romanizer("Привет")
    assert isinstance(result, str)

def test_romanize_raises_locale_error_for_invalid_string_locale():
    from mimesis.exceptions import LocaleError
    from mimesis.keys import romanize
    try:
        romanize("invalid")
    except LocaleError as e:
        assert str(e) == "invalid"

def test_romanize_raises_locale_error_for_invalid_locale_type():
    from mimesis.exceptions import LocaleError
    from mimesis.keys import romanize
    try:
        romanize(123)
    except LocaleError as e:
        assert str(e) == 123

def test_romanize_translates_common_letters():
    from mimesis.enums import Locale
    from mimesis.keys import romanize
    romanizer = romanize(Locale.KK)
    result = romanizer("ё")
    assert result == "yo"


# LLM-generated content at query #6
#--------------------------

def test_pipe_with_single_function():
    def add_one(x):
        return x + 1
    key_func = pipe(add_one)
    result = key_func(5)
    assert result == 6

def test_pipe_with_multiple_functions():
    def add_one(x):
        return x + 1
    def multiply_by_two(x):
        return x * 2
    key_func = pipe(add_one, multiply_by_two)
    result = key_func(3)
    assert result == 8

def test_pipe_with_string_functions():
    def lower(s):
        return s.lower()
    def strip(s):
        return s.strip()
    key_func = pipe(lower, strip)
    result = key_func("  HELLO  ")
    assert result == "hello"

def test_pipe_with_random_parameter():
    def add_random(x, random):
        return x + random.randint(1, 10)
    def multiply_by_two(x):
        return x * 2
    key_func = pipe(add_random, multiply_by_two)
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    result = key_func(3, mock_random)
    assert result == 16

def test_pipe_with_mixed_function_signatures():
    def add_one(x):
        return x + 1
    def add_random(x, random):
        return x + random.randint(1, 10)
    key_func = pipe(add_one, add_random)
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    result = key_func(3, mock_random)
    assert result == 9

def test_pipe_with_no_functions():
    key_func = pipe()
    result = key_func(42)
    assert result == 42

def test_pipe_with_nested_pipe():
    def add_one(x):
        return x + 1
    inner_pipe = pipe(add_one, add_one)
    outer_pipe = pipe(inner_pipe, add_one)
    result = outer_pipe(1)
    assert result == 4

def test_pipe_with_string_operations():
    def prefix(p):
        return lambda s: p + s
    def suffix(suf):
        return lambda s: s + suf
    key_func = pipe(prefix("pre-"), suffix("-suf"))
    result = key_func("test")
    assert result == "pre-test-suf"

def test_pipe_with_list_operations():
    def append_one(lst):
        lst.append(1)
        return lst
    def reverse_list(lst):
        lst.reverse()
        return lst
    key_func = pipe(append_one, reverse_list)
    result = key_func([2, 3])
    assert result == [1, 3, 2]

def test_pipe_with_dict_operations():
    def add_key(d):
        d["new"] = "value"
        return d
    def remove_key(d):
        d.pop("old", None)
        return d
    key_func = pipe(add_key, remove_key)
    result = key_func({"old": "data"})
    assert result == {"new": "value"}


# LLM-generated content at query #7
#--------------------------

def test_truncate_returns_original_string_when_within_max_length():
    truncator = truncate(10)
    result = truncator("hello")
    assert result == "hello"

def test_truncate_returns_truncated_string_with_default_suffix():
    truncator = truncate(5)
    result = truncator("hello world")
    assert result == "he..."

def test_truncate_returns_truncated_string_with_custom_suffix():
    truncator = truncate(5, suffix="!!")
    result = truncator("hello world")
    assert result == "hel!!"

def test_truncate_raises_value_error_for_non_positive_max_length():
    try:
        truncate(0)
        assert False
    except ValueError as e:
        assert str(e) == "max_length must be positive, got 0"

def test_truncate_raises_type_error_for_non_string_input():
    truncator = truncate(10)
    try:
        truncator(123)
        assert False
    except TypeError as e:
        assert str(e) == "truncate() requires a string, got int"

def test_truncate_exact_max_length_no_truncation():
    truncator = truncate(5)
    result = truncator("hello")
    assert result == "hello"

def test_truncate_exact_max_length_with_suffix_no_truncation():
    truncator = truncate(5, suffix="...")
    result = truncator("hello")
    assert result == "hello"

def test_truncate_max_length_shorter_than_suffix():
    truncator = truncate(2, suffix="...")
    result = truncator("hello")
    assert result == ".."

def test_truncate_empty_string():
    truncator = truncate(5)
    result = truncator("")
    assert result == ""

def test_truncate_negative_max_length_raises_error():
    try:
        truncate(-1)
        assert False
    except ValueError as e:
        assert str(e) == "max_length must be positive, got -1"


# LLM-generated content at query #8
#--------------------------

def test_pipe_key_function_with_random_parameter():
    def add_prefix(prefix):
        def inner(value, random=None):
            return prefix + value
        return inner
    def to_upper(value, random=None):
        return value.upper()
    def repeat(value, random=None):
        if random is None:
            return value + value
        return value * random.randint(1, 3)
    key_func = pipe(to_upper, add_prefix("TEST-"), repeat)
    mock_random = Random()
    mock_random.randint = lambda a, b: 2
    result = key_func("hello", mock_random)
    assert result == "TEST-HELLOTEST-HELLO"


# LLM-generated content at query #9
#--------------------------

def test_pipe_with_key_functions():
    def add_prefix(prefix):
        def inner(value, random=None):
            return prefix + value
        return inner
    def to_upper(value, random=None):
        return value.upper()
    def repeat(value, random=None):
        return value + value
    key_func = pipe(add_prefix("test_"), to_upper, repeat)
    result = key_func("hello", None)
    expected = "TEST_HELLOTEST_HELLO"
    assert result == expected

def test_pipe_with_mixed_functions():
    def add_suffix(suffix):
        def inner(value):
            return value + suffix
        return inner
    def double(value, random=None):
        return value * 2
    key_func = pipe(double, add_suffix("!"))
    result = key_func("hi", None)
    expected = "hihi!"
    assert result == expected

def test_pipe_single_function():
    def increment(value, random=None):
        return value + 1
    key_func = pipe(increment)
    result = key_func(5, None)
    expected = 6
    assert result == expected

def test_pipe_no_functions():
    key_func = pipe()
    result = key_func("anything", None)
    expected = "anything"
    assert result == expected

def test_pipe_with_random_parameter():
    def add_random(value, random):
        return value + str(random.randint(1, 10))
    mock_random = Random()
    mock_random.randint = lambda a, b: 7
    key_func = pipe(add_random)
    result = key_func("num", mock_random)
    expected = "num7"
    assert result == expected

def test_pipe_chain_with_and_without_random():
    def with_random(value, random):
        return value + str(random.randint(1, 10))
    def without_random(value):
        return value.upper()
    mock_random = Random()
    mock_random.randint = lambda a, b: 3
    key_func = pipe(with_random, without_random)
    result = key_func("test", mock_random)
    expected = "TEST3"
    assert result == expected


# LLM-generated content at query #10
#--------------------------

def test_apply_if_condition_true():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    key_func = apply_if(condition, transform)
    result = key_func(5)
    assert result == 10

def test_apply_if_condition_false_without_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    key_func = apply_if(condition, transform)
    result = key_func(-5)
    assert result == -5

def test_apply_if_condition_false_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(-5)
    assert result == -15

def test_apply_if_condition_true_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(5)
    assert result == 10

def test_apply_if_with_string_condition_true():
    condition = lambda s: len(s) > 3
    transform = str.upper
    key_func = apply_if(condition, transform)
    result = key_func("word")
    assert result == "WORD"

def test_apply_if_with_string_condition_false_without_otherwise():
    condition = lambda s: len(s) > 3
    transform = str.upper
    key_func = apply_if(condition, transform)
    result = key_func("hi")
    assert result == "hi"

def test_apply_if_with_string_condition_false_with_otherwise():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    key_func = apply_if(condition, transform, otherwise)
    result = key_func("HI")
    assert result == "hi"

def test_apply_if_with_none_otherwise():
    condition = lambda x: x is None
    transform = lambda x: "transformed"
    key_func = apply_if(condition, transform)
    result = key_func(None)
    assert result == "transformed"

def test_apply_if_with_false_condition_and_none_otherwise():
    condition = lambda x: x is None
    transform = lambda x: "transformed"
    key_func = apply_if(condition, transform)
    result = key_func(42)
    assert result == 42

def test_apply_if_with_complex_condition():
    condition = lambda lst: len(lst) > 2 and sum(lst) > 10
    transform = lambda lst: max(lst)
    otherwise = lambda lst: min(lst)
    key_func = apply_if(condition, transform, otherwise)
    result = key_func([1, 2, 3, 4])
    assert result == 4

def test_apply_if_with_complex_condition_false():
    condition = lambda lst: len(lst) > 2 and sum(lst) > 10
    transform = lambda lst: max(lst)
    otherwise = lambda lst: min(lst)
    key_func = apply_if(condition, transform, otherwise)
    result = key_func([1, 2])
    assert result == 1


# LLM-generated content at query #11
#--------------------------

def test_apply_if_condition_false_without_otherwise():
    condition = lambda x: False
    transform = lambda x: x * 2
    key_func = apply_if(condition, transform)
    result = key_func(5)
    assert result == 5

def test_apply_if_condition_false_with_otherwise():
    condition = lambda x: False
    transform = lambda x: x * 2
    otherwise = lambda x: x + 1
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(5)
    assert result == 6


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_2_evaluates_to_false():
    result = apply_if(lambda x: False, lambda x: x + 1, lambda x: x - 1)(5)
    assert result == 4
    result = apply_if(lambda x: False, lambda x: x.upper())( "test")
    assert result == "test"
    result = apply_if(lambda x: False, lambda x: x * 2, None)(3)
    assert result == 3


# LLM-generated content at query #13
#--------------------------

def test_apply_if_condition_false_without_otherwise():
    condition = lambda x: False
    transform = lambda x: x * 2
    key_func = apply_if(condition, transform)
    result = key_func(5)
    assert result == 5

def test_apply_if_condition_false_with_otherwise():
    condition = lambda x: False
    transform = lambda x: x * 2
    otherwise = lambda x: x + 10
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(5)
    assert result == 15


# LLM-generated content at query #14
#--------------------------

def test_predicate_at_line_2_evaluates_to_false():
    result = apply_if(lambda x: False, lambda x: x + 1, lambda x: x - 1)(5)
    assert result == 4
    result = apply_if(lambda x: False, lambda x: x.upper())( "test")
    assert result == "test"
    result = apply_if(lambda x: False, lambda x: x * 2, None)(3)
    assert result == 3


# LLM-generated content at query #15
#--------------------------

def test_pipe_with_key_functions_using_random():
    from mimesis.keys import pipe
    from mimesis.random import Random
    def add_one(x, random=None):
        return x + 1
    def multiply_two(x, random=None):
        return x * 2
    func = pipe(add_one, multiply_two)
    result = func(5, Random())
    assert result == 12

def test_pipe_with_key_functions_without_random():
    from mimesis.keys import pipe
    def add_one(x):
        return x + 1
    def multiply_two(x):
        return x * 2
    func = pipe(add_one, multiply_two)
    result = func(5)
    assert result == 12

def test_pipe_with_mixed_key_functions():
    from mimesis.keys import pipe
    from mimesis.random import Random
    def add_one(x):
        return x + 1
    def multiply_two(x, random=None):
        return x * 2
    func = pipe(add_one, multiply_two)
    result = func(5, Random())
    assert result == 12

def test_pipe_with_string_transformations():
    from mimesis.keys import pipe
    def upper(s):
        return s.upper()
    def add_exclamation(s):
        return s + '!'
    func = pipe(upper, add_exclamation)
    result = func('hello')
    assert result == 'HELLO!'

def test_pipe_with_empty_functions():
    from mimesis.keys import pipe
    func = pipe()
    result = func('test')
    assert result == 'test'

def test_pipe_with_single_function():
    from mimesis.keys import pipe
    def square(x):
        return x * x
    func = pipe(square)
    result = func(4)
    assert result == 16

def test_pipe_preserves_random_instance():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    captured_random = None
    def capture_random(x, random=None):
        nonlocal captured_random
        captured_random = random
        return x
    func = pipe(capture_random)
    func(10, random_instance)
    assert captured_random is random_instance


# LLM-generated content at query #16
#--------------------------

def test_apply_if_with_condition_true():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    key_func = apply_if(condition, transform)
    result = key_func(5)
    assert result == 10

def test_apply_if_with_condition_false_and_no_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    key_func = apply_if(condition, transform)
    result = key_func(-5)
    assert result == -5

def test_apply_if_with_condition_false_and_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(-5)
    assert result == -15

def test_apply_if_with_string_condition_true():
    condition = lambda s: len(s) > 3
    transform = str.upper
    key_func = apply_if(condition, transform)
    result = key_func("hello")
    assert result == "HELLO"

def test_apply_if_with_string_condition_false_and_otherwise():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    key_func = apply_if(condition, transform, otherwise)
    result = key_func("hi")
    assert result == "hi"

def test_apply_if_with_condition_false_and_otherwise_none():
    condition = lambda x: False
    transform = lambda x: "transformed"
    key_func = apply_if(condition, transform)
    result = key_func("input")
    assert result == "input"

def test_apply_if_with_condition_true_and_otherwise_provided_but_not_used():
    condition = lambda x: True
    transform = lambda x: x + 1
    otherwise = lambda x: x - 1
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(10)
    assert result == 11

def test_apply_if_with_complex_condition():
    condition = lambda lst: len(lst) == 2 and sum(lst) > 10
    transform = lambda lst: sum(lst)
    key_func = apply_if(condition, transform)
    result = key_func([6, 5])
    assert result == 11

def test_apply_if_with_complex_condition_false():
    condition = lambda lst: len(lst) == 2 and sum(lst) > 10
    transform = lambda lst: sum(lst)
    key_func = apply_if(condition, transform)
    result = key_func([1, 2])
    assert result == [1, 2]


# LLM-generated content at query #17
#--------------------------

def test_pipe_with_single_function():
    def add_one(x):
        return x + 1
    key_func = pipe(add_one)
    result = key_func(5)
    assert result == 6

def test_pipe_with_multiple_functions():
    def add_one(x):
        return x + 1
    def multiply_by_two(x):
        return x * 2
    key_func = pipe(add_one, multiply_by_two)
    result = key_func(5)
    assert result == 12

def test_pipe_with_string_functions():
    def upper(s):
        return s.upper()
    def add_exclamation(s):
        return s + "!"
    key_func = pipe(upper, add_exclamation)
    result = key_func("hello")
    assert result == "HELLO!"

def test_pipe_with_random_parameter():
    def add_random(x, random):
        return x + random.randint(1, 10)
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    key_func = pipe(add_random)
    result = key_func(10, mock_random)
    assert result == 15

def test_pipe_mixed_with_and_without_random():
    def add_one(x):
        return x + 1
    def add_random(x, random):
        return x + random.randint(1, 10)
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    key_func = pipe(add_one, add_random)
    result = key_func(10, mock_random)
    assert result == 16

def test_pipe_with_no_functions():
    key_func = pipe()
    result = key_func(42)
    assert result == 42

def test_pipe_with_nested_pipe():
    def add_one(x):
        return x + 1
    def multiply_by_two(x):
        return x * 2
    inner_pipe = pipe(add_one, multiply_by_two)
    outer_pipe = pipe(inner_pipe, add_one)
    result = outer_pipe(5)
    assert result == 13


# LLM-generated content at query #18
#--------------------------

def test_apply_if_with_condition_true():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(5)
    assert result == 10

def test_apply_if_with_condition_false_and_no_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(-5)
    assert result == -5

def test_apply_if_with_condition_false_and_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    func = apply_if(condition, transform, otherwise)
    result = func(-5)
    assert result == -15

def test_apply_if_with_string_condition_true():
    condition = lambda s: len(s) > 3
    transform = str.upper
    func = apply_if(condition, transform)
    result = func("hello")
    assert result == "HELLO"

def test_apply_if_with_string_condition_false_and_otherwise():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    func = apply_if(condition, transform, otherwise)
    result = func("hi")
    assert result == "hi"

def test_apply_if_with_none_otherwise_and_condition_false():
    condition = lambda x: isinstance(x, int)
    transform = lambda x: x + 1
    func = apply_if(condition, transform)
    result = func("not an int")
    assert result == "not an int"

def test_apply_if_with_complex_condition():
    condition = lambda lst: len(lst) == sum(lst)
    transform = lambda lst: [x * 2 for x in lst]
    otherwise = lambda lst: lst[::-1]
    func = apply_if(condition, transform, otherwise)
    result = func([1, 1, 1])
    assert result == [2, 2, 2]

def test_apply_if_with_otherwise_as_none():
    condition = lambda x: x % 2 == 0
    transform = lambda x: x // 2
    func = apply_if(condition, transform, None)
    result = func(4)
    assert result == 2
    result = func(3)
    assert result == 3


# LLM-generated content at query #19
#--------------------------

def test_predicate_at_line_2_evaluates_to_false():
    result = apply_if(lambda x: False, lambda x: x + 1, lambda x: x - 1)(5)
    assert result == 4


# LLM-generated content at query #20
#--------------------------

def test_pipe_with_key_functions():
    def add_one(x, random=None):
        return x + 1
    def double(x, random=None):
        return x * 2
    piped = pipe(add_one, double)
    result = piped(5, None)
    assert result == 12

def test_pipe_with_mixed_functions():
    def add_prefix(x, random=None):
        return "prefix-" + x
    def upper(x):
        return x.upper()
    piped = pipe(add_prefix, upper)
    result = piped("test", None)
    assert result == "PREFIX-TEST"

def test_pipe_single_function():
    def square(x, random=None):
        return x * x
    piped = pipe(square)
    result = piped(4, None)
    assert result == 16

def test_pipe_no_functions():
    piped = pipe()
    result = piped("anything", None)
    assert result == "anything"

def test_pipe_with_random_argument():
    def add_random(x, random):
        return x + random.randint(1, 10)
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    piped = pipe(add_random)
    result = piped(10, mock_random)
    assert result == 15

def test_pipe_chain_with_ignored_random():
    def first(x, random=None):
        return x + 1
    def second(x):
        return x * 2
    def third(x, random=None):
        return x - 3
    piped = pipe(first, second, third)
    result = piped(5, None)
    assert result == 9


# LLM-generated content at query #21
#--------------------------

def test_pipe_key_function_with_random_parameter():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    def add_suffix(value, random=None):
        return f"{value}_{random.randint(1, 10) if random else 0}"
    def uppercase(value):
        return value.upper()
    key_func = pipe(uppercase, add_suffix)
    result = key_func("test", random_instance)
    assert isinstance(result, str)
    assert result.startswith("TEST_")
    assert result[5:].isdigit()
    assert 1 <= int(result[5:]) <= 10


# LLM-generated content at query #22
#--------------------------

def test_apply_if_condition_true():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * -1
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(5)
    assert result == 10


# LLM-generated content at query #23
#--------------------------

def test_apply_if_condition_true():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(5)
    assert result == 10

def test_apply_if_condition_false_without_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(-5)
    assert result == -5

def test_apply_if_condition_false_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    func = apply_if(condition, transform, otherwise)
    result = func(-5)
    assert result == -15

def test_apply_if_with_string_condition_true():
    condition = lambda x: len(x) > 3
    transform = str.upper
    func = apply_if(condition, transform)
    result = func("word")
    assert result == "WORD"

def test_apply_if_with_string_condition_false_without_otherwise():
    condition = lambda x: len(x) > 3
    transform = str.upper
    func = apply_if(condition, transform)
    result = func("hi")
    assert result == "hi"

def test_apply_if_with_string_condition_false_with_otherwise():
    condition = lambda x: len(x) > 3
    transform = str.upper
    otherwise = str.lower
    func = apply_if(condition, transform, otherwise)
    result = func("HI")
    assert result == "hi"

def test_apply_if_otherwise_is_none():
    condition = lambda x: x is None
    transform = lambda x: "None"
    func = apply_if(condition, transform, None)
    result = func(42)
    assert result == 42

def test_apply_if_condition_true_with_complex_transform():
    condition = lambda x: isinstance(x, list)
    transform = sum
    func = apply_if(condition, transform)
    result = func([1, 2, 3])
    assert result == 6

def test_apply_if_condition_false_with_complex_otherwise():
    condition = lambda x: isinstance(x, list)
    transform = sum
    otherwise = lambda x: len(str(x))
    func = apply_if(condition, transform, otherwise)
    result = func(12345)
    assert result == 5


# LLM-generated content at query #24
#--------------------------

def test_apply_if_condition_true():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * -1)(5)
    assert result == 10


# LLM-generated content at query #25
#--------------------------

def test_apply_if_with_condition_true():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(5)
    assert result == 10

def test_apply_if_with_condition_false_and_no_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(-5)
    assert result == -5

def test_apply_if_with_condition_false_and_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    func = apply_if(condition, transform, otherwise)
    result = func(-5)
    assert result == -15

def test_apply_if_with_string_condition_true():
    condition = lambda s: len(s) > 3
    transform = str.upper
    func = apply_if(condition, transform)
    result = func("hello")
    assert result == "HELLO"

def test_apply_if_with_string_condition_false_and_otherwise():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    func = apply_if(condition, transform, otherwise)
    result = func("hi")
    assert result == "hi"

def test_apply_if_with_none_otherwise():
    condition = lambda x: isinstance(x, int)
    transform = lambda x: x + 1
    func = apply_if(condition, transform)
    result = func("string")
    assert result == "string"

def test_apply_if_with_list_input():
    condition = lambda lst: len(lst) > 2
    transform = lambda lst: sum(lst)
    otherwise = lambda lst: max(lst)
    func = apply_if(condition, transform, otherwise)
    result = func([1, 2])
    assert result == 2

def test_apply_if_with_condition_true_and_otherwise_provided():
    condition = lambda x: x % 2 == 0
    transform = lambda x: x // 2
    otherwise = lambda x: x * 2
    func = apply_if(condition, transform, otherwise)
    result = func(4)
    assert result == 2

def test_apply_if_with_condition_false_and_otherwise_provided():
    condition = lambda x: x % 2 == 0
    transform = lambda x: x // 2
    otherwise = lambda x: x * 2
    func = apply_if(condition, transform, otherwise)
    result = func(3)
    assert result == 6


# LLM-generated content at query #26
#--------------------------

def test_apply_if_condition_true():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * -1
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(5)
    assert result == 10


# LLM-generated content at query #27
#--------------------------

def test_pipe_handles_functions_without_random_parameter():
    def add_prefix(text):
        return "user-" + text
    def to_lower(text):
        return text.lower()
    key_func = pipe(to_lower, add_prefix)
    result = key_func("John_Doe", None)
    assert result == "user-john_doe"


# LLM-generated content at query #28
#--------------------------

def test_pipe_with_key_functions():
    def add_one(x, random=None):
        return x + 1
    def double(x, random=None):
        return x * 2
    piped = pipe(add_one, double)
    result = piped(5, None)
    assert result == 12

def test_pipe_with_mixed_functions():
    def add_prefix(x, random=None):
        return "prefix-" + x
    def upper(x):
        return x.upper()
    piped = pipe(add_prefix, upper)
    result = piped("test", None)
    assert result == "PREFIX-TEST"

def test_pipe_with_single_function():
    def square(x, random=None):
        return x * x
    piped = pipe(square)
    result = piped(4, None)
    assert result == 16

def test_pipe_with_no_functions():
    piped = pipe()
    result = piped("anything", None)
    assert result == "anything"

def test_pipe_with_random_argument():
    def add_random(x, random):
        return x + random.randint(1, 10)
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    piped = pipe(add_random)
    result = piped(10, mock_random)
    assert result == 15

def test_pipe_with_three_functions():
    def add_two(x, random=None):
        return x + 2
    def triple(x, random=None):
        return x * 3
    def subtract_one(x, random=None):
        return x - 1
    piped = pipe(add_two, triple, subtract_one)
    result = piped(5, None)
    assert result == 20

def test_pipe_with_string_operations():
    def reverse(s, random=None):
        return s[::-1]
    def capitalize(s):
        return s.capitalize()
    piped = pipe(reverse, capitalize)
    result = piped("hello", None)
    assert result == "Olleh"

def test_pipe_with_list_operations():
    def append_item(lst, random=None):
        lst.append("end")
        return lst
    def extend_list(lst, random=None):
        lst.extend([1, 2])
        return lst
    piped = pipe(append_item, extend_list)
    result = piped(["start"], None)
    assert result == ["start", "end", 1, 2]


# LLM-generated content at query #29
#--------------------------

def test_apply_if_with_otherwise():
    result_func = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * -1)
    assert result_func(5) == 10
    assert result_func(-3) == 3

def test_apply_if_without_otherwise():
    result_func = apply_if(lambda x: x.startswith('a'), lambda x: x.upper())
    assert result_func('apple') == 'APPLE'
    assert result_func('banana') == 'banana'

def test_apply_if_condition_true():
    result_func = apply_if(lambda x: isinstance(x, int), lambda x: x + 1)
    assert result_func(42) == 43

def test_apply_if_condition_false_without_otherwise():
    result_func = apply_if(lambda x: x % 2 == 0, lambda x: x * 2)
    assert result_func(3) == 3

def test_apply_if_with_none_otherwise():
    result_func = apply_if(lambda x: x > 10, lambda x: 'big', None)
    assert result_func(15) == 'big'
    assert result_func(5) == 5


# LLM-generated content at query #30
#--------------------------

def test_pipe_handles_key_functions_without_random_parameter():
    def func1(x):
        return x + 1
    def func2(x):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, None)
    assert result == 12


# LLM-generated content at query #31
#--------------------------

def test_pipe_handles_key_functions_without_random_parameter():
    def func1(x):
        return x + 1
    def func2(x):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, None)
    assert result == 12


# LLM-generated content at query #32
#--------------------------

def test_apply_if_condition_true():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(5)
    assert result == 10

def test_apply_if_condition_false_without_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    func = apply_if(condition, transform)
    result = func(-5)
    assert result == -5

def test_apply_if_condition_false_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    func = apply_if(condition, transform, otherwise)
    result = func(-5)
    assert result == -15

def test_apply_if_condition_true_with_otherwise():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x * 3
    func = apply_if(condition, transform, otherwise)
    result = func(5)
    assert result == 10

def test_apply_if_with_string_condition():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    func = apply_if(condition, transform, otherwise)
    result = func('word')
    assert result == 'WORD'

def test_apply_if_with_string_condition_false():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    func = apply_if(condition, transform, otherwise)
    result = func('hi')
    assert result == 'hi'

def test_apply_if_with_string_condition_false_with_otherwise():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    func = apply_if(condition, transform, otherwise)
    result = func('hi')
    assert result == 'hi'

def test_apply_if_with_none_otherwise():
    condition = lambda x: x is None
    transform = lambda x: 'missing'
    func = apply_if(condition, transform)
    result = func(None)
    assert result == 'missing'

def test_apply_if_with_false_condition_and_none_otherwise():
    condition = lambda x: x is None
    transform = lambda x: 'missing'
    func = apply_if(condition, transform)
    result = func(42)
    assert result == 42

def test_apply_if_with_complex_condition():
    condition = lambda x: isinstance(x, list) and len(x) == 0
    transform = lambda x: ['empty']
    otherwise = lambda x: x + [1]
    func = apply_if(condition, transform, otherwise)
    result = func([])
    assert result == ['empty']

def test_apply_if_with_complex_condition_false():
    condition = lambda x: isinstance(x, list) and len(x) == 0
    transform = lambda x: ['empty']
    otherwise = lambda x: x + [1]
    func = apply_if(condition, transform, otherwise)
    result = func([2, 3])
    assert result == [2, 3, 1]


# LLM-generated content at query #33
#--------------------------

def test_pipe_key_with_random_parameter():
    def add_prefix(value, random=None):
        return f"pre-{value}"
    def add_suffix(value, random=None):
        return f"{value}-suf"
    key_func = pipe(add_prefix, add_suffix)
    result = key_func("test", random=Random())
    assert result == "pre-test-suf"

def test_pipe_key_without_random_parameter():
    def uppercase(value):
        return value.upper()
    def reverse(value):
        return value[::-1]
    key_func = pipe(uppercase, reverse)
    result = key_func("hello")
    assert result == "OLLEH"

def test_pipe_key_mixed_functions():
    def add_random(value, random):
        return f"{value}-{random.randint(1, 10)}"
    def double(value):
        return value * 2
    key_func = pipe(add_random, double)
    random_instance = Random()
    result = key_func("test", random=random_instance)
    assert result.endswith("-suf") == False


# LLM-generated content at query #34
#--------------------------

def test_pipe_with_random_parameter():
    def func1(x, random=None):
        return x + 1
    def func2(x, random=None):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, random=Random())
    assert result == 12


# LLM-generated content at query #35
#--------------------------

def test_pipe_applies_functions_in_sequence():
    def add_one(x):
        return x + 1
    def multiply_by_two(x):
        return x * 2
    key_func = pipe(add_one, multiply_by_two)
    result = key_func(5)
    assert result == 12

def test_pipe_with_single_function():
    def to_uppercase(s):
        return s.upper()
    key_func = pipe(to_uppercase)
    result = key_func("hello")
    assert result == "HELLO"

def test_pipe_with_string_operations():
    def strip_spaces(s):
        return s.strip()
    def replace_dash(s):
        return s.replace('-', '_')
    key_func = pipe(strip_spaces, replace_dash)
    result = key_func("  hello-world  ")
    assert result == "hello_world"

def test_pipe_with_random_parameter():
    def add_random(x, random):
        return x + random.randint(1, 10)
    def double(x):
        return x * 2
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    key_func = pipe(add_random, double)
    result = key_func(10, mock_random)
    assert result == 30

def test_pipe_with_random_parameter_optional():
    def func_without_random(x):
        return x * 3
    def func_with_random(x, random):
        return x + random.randint(1, 10)
    mock_random = Random()
    mock_random.randint = lambda a, b: 3
    key_func = pipe(func_without_random, func_with_random)
    result = key_func(4, mock_random)
    assert result == 15

def test_pipe_with_no_functions():
    key_func = pipe()
    result = key_func("test")
    assert result == "test"

def test_pipe_with_nested_pipes():
    def increment(x):
        return x + 1
    inner_pipe = pipe(increment, increment)
    outer_pipe = pipe(inner_pipe, increment)
    result = outer_pipe(0)
    assert result == 3


# LLM-generated content at query #36
#--------------------------

def test_pipe_key_function_handles_type_error_correctly():
    mock_random = Random()
    def func_raises_type_error(value, random):
        raise TypeError
    def func_returns_modified(value):
        return value + "_modified"
    key_func = pipe(func_raises_type_error, func_returns_modified)
    result = key_func("test", mock_random)
    assert result == "test_modified"


