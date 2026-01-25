####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_romanize_with_russian_locale():
    romanize_ru = romanize(Locale.RU)
    assert romanize_ru("Привет") == "Privet"
    assert romanize_ru("Москва") == "Moskva"

def test_romanize_with_ukrainian_locale():
    romanize_uk = romanize(Locale.UK)
    assert romanize_uk("Привіт") == "Pryvit"
    assert romanize_uk("Київ") == "Kyiv"

def test_romanize_with_kazakh_locale():
    romanize_kk = romanize(Locale.KK)
    assert romanize_kk("Сәлем") == "Sälem"
    assert romanize_kk("Астана") == "Astana"

def test_romanize_with_invalid_locale():
    with pytest.raises(ValueError):
        romanize(Locale.EN)

def test_romanize_with_invalid_string_input():
    romanize_ru = romanize(Locale.RU)
    with pytest.raises(TypeError):
        romanize_ru(123)

def test_romanize_with_string_locale():
    romanize_ru = romanize("ru")
    assert romanize_ru("Привет") == "Privet"

def test_romanize_with_invalid_string_locale():
    with pytest.raises(ValueError):
        romanize("invalid_locale")


# LLM-generated content at query #2
#--------------------------

```python
def test_romanize_raises_value_error_for_unsupported_locale():
    with pytest.raises(ValueError, match="Romanization is not available for: .*"):
        romanize(Locale.EN)


# LLM-generated content at query #3
#--------------------------

```python
def test_truncate_returns_closure():
    truncate_func = truncate(10)
    assert callable(truncate_func)

def test_truncate_with_empty_string():
    truncate_func = truncate(10)
    assert truncate_func("") == ""

def test_truncate_with_string_shorter_than_max_length():
    truncate_func = truncate(10)
    assert truncate_func("short") == "short"

def test_truncate_with_string_equal_to_max_length():
    truncate_func = truncate(10)
    assert truncate_func("1234567890") == "1234567890"

def test_truncate_with_string_longer_than_max_length():
    truncate_func = truncate(10)
    assert truncate_func("1234567890123") == "1234567..."

def test_truncate_with_custom_suffix():
    truncate_func = truncate(10, suffix="...")
    assert truncate_func("1234567890123") == "1234567..."

def test_truncate_with_non_string_input():
    truncate_func = truncate(10)
    try:
        truncate_func(123)
    except TypeError as e:
        assert str(e) == "truncate() requires a string, got int"

def test_truncate_with_non_positive_max_length():
    try:
        truncate(0)
    except ValueError as e:
        assert str(e) == "max_length must be positive, got 0"

def test_truncate_with_negative_max_length():
    try:
        truncate(-5)
    except ValueError as e:
        assert str(e) == "max_length must be positive, got -5"


# LLM-generated content at query #4
#--------------------------

```python
def test_pipe_with_single_function():
    result = pipe(str.upper)("hello", None)
    assert result == "HELLO"

def test_pipe_with_multiple_functions():
    def add_prefix(value: str) -> str:
        return f"prefix-{value}"

    result = pipe(str.upper, add_prefix)("hello", None)
    assert result == "prefix-HELLO"

def test_pipe_with_random_parameter():
    def random_upper(value: str, random: Random) -> str:
        if random.random() > 0.5:
            return value.upper()
        return value.lower()

    random = Random()
    random.random = lambda: 0.6
    result = pipe(random_upper)("hello", random)
    assert result == "HELLO"

def test_pipe_with_no_random_parameter():
    def no_random_func(value: str) -> str:
        return f"no-random-{value}"

    result = pipe(no_random_func)("test", None)
    assert result == "no-random-test"

def test_pipe_with_mixed_functions():
    def add_suffix(value: str) -> str:
        return f"{value}-suffix"

    def random_prefix(value: str, random: Random) -> str:
        return f"random-{random.randint(1, 100)}-{value}"

    random = Random()
    random.randint = lambda a, b: 42
    result = pipe(str.upper, random_prefix, add_suffix)("hello", random)
    assert result == "random-42-HELLO-suffix"


# LLM-generated content at query #5
#--------------------------

```python
def test_apply_if_with_true_condition():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(5)
    assert result == 10

def test_apply_if_with_false_condition_and_no_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(-5)
    assert result == -5

def test_apply_if_with_false_condition_and_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)(-5)
    assert result == -15

def test_apply_if_with_string_condition():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hello")
    assert result == "HELLO"

def test_apply_if_with_string_condition_false():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hi")
    assert result == "hi"

def test_apply_if_with_none_input():
    result = apply_if(lambda x: x is not None, lambda x: "valid", lambda x: "invalid")(None)
    assert result == "invalid"


# LLM-generated content at query #6
#--------------------------

```python
def test_apply_if_applies_transform_when_condition_is_true():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x)(5)
    assert result == 10

def test_apply_if_applies_otherwise_when_condition_is_false():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x)(-5)
    assert result == -5

def test_apply_if_returns_original_value_when_condition_is_false_and_no_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(-5)
    assert result == -5

def test_apply_if_works_with_strings():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hello")
    assert result == "HELLO"

def test_apply_if_works_with_strings_and_otherwise():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hi")
    assert result == "hi"

def test_apply_if_works_with_lists():
    result = apply_if(lambda x: len(x) > 2, lambda x: x + [1], lambda x: x + [0])([1, 2])
    assert result == [1, 2, 1]

def test_apply_if_works_with_lists_and_no_otherwise():
    result = apply_if(lambda x: len(x) > 2, lambda x: x + [1])([1])
    assert result == [1]

def test_apply_if_works_with_custom_objects():
    class TestClass:
        def __init__(self, value):
            self.value = value

    obj = TestClass(10)
    result = apply_if(lambda x: x.value > 5, lambda x: TestClass(x.value * 2), lambda x: TestClass(x.value))(obj)
    assert result.value == 20

def test_apply_if_works_with_custom_objects_and_otherwise():
    class TestClass:
        def __init__(self, value):
            self.value = value

    obj = TestClass(2)
    result = apply_if(lambda x: x.value > 5, lambda x: TestClass(x.value * 2), lambda x: TestClass(x.value))(obj)
    assert result.value == 2


# LLM-generated content at query #7
#--------------------------

```python
def test_pipe_docstring_exists():
    assert pipe.__doc__ is not None


# LLM-generated content at query #8
#--------------------------

```python
def test_pipe_docstring_predicate():
    assert pipe.__doc__.startswith("Pipe multiple key functions together.")


# LLM-generated content at query #9
#--------------------------

```python
def test_maybe_returns_closure():
    key_func = maybe("test_value")
    assert callable(key_func)

def test_maybe_with_valid_probability():
    key_func = maybe("test_value", 0.7)
    random = Random()
    result = key_func("original_value", random)
    assert result in ["original_value", "test_value"]

def test_maybe_with_zero_probability():
    key_func = maybe("test_value", 0.0)
    random = Random()
    result = key_func("original_value", random)
    assert result == "original_value"

def test_maybe_with_one_probability():
    key_func = maybe("test_value", 1.0)
    random = Random()
    result = key_func("original_value", random)
    assert result == "test_value"

def test_maybe_with_invalid_probability():
    key_func = maybe("test_value", -0.5)
    random = Random()
    result = key_func("original_value", random)
    assert result == "original_value"

def test_maybe_with_probability_above_one():
    key_func = maybe("test_value", 1.5)
    random = Random()
    result = key_func("original_value", random)
    assert result == "original_value"


# LLM-generated content at query #10
#--------------------------

```python
def test_pipe_docstring_exists():
    assert pipe.__doc__ is not None


# LLM-generated content at query #11
#--------------------------

```python
def test_pipe_docstring_exists():
    assert pipe.__doc__ is not None


# LLM-generated content at query #12
#--------------------------

```python
def test_pipe_docstring_starts_with_pipe_multiple_key_functions_together():
    assert pipe.__doc__.startswith("Pipe multiple key functions together.")


# LLM-generated content at query #13
#--------------------------

```python
def test_pipe_docstring_starts_with_pipe_multiple_key_functions_together():
    assert pipe.__doc__.startswith("Pipe multiple key functions together.")


# LLM-generated content at query #14
#--------------------------

```python
def test_pipe_applies_single_function():
    result = pipe(str.upper)("hello", None)
    assert result == "HELLO"

def test_pipe_applies_multiple_functions():
    result = pipe(str.lower, str.upper)("Hello", None)
    assert result == "HELLO"

def test_pipe_with_random_parameter():
    random = Random()
    result = pipe(lambda x, r: x + str(r.randint(0, 100)))("value", random)
    assert isinstance(result, str) and result.startswith("value")

def test_pipe_handles_type_error_gracefully():
    def func_without_random(x):
        return x.upper()

    result = pipe(func_without_random)("test", Random())
    assert result == "TEST"

def test_pipe_returns_original_value_when_no_functions():
    result = pipe()("unchanged", None)
    assert result == "unchanged"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_hash_with_sha256():
    hash_func = hash_with("sha256")
    assert hash_func("password") == "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"

def test_hash_with_sha1():
    hash_func = hash_with("sha1")
    assert hash_func("password") == "d3e7130d657733468b10c1fd207c4d62b7180cda"

def test_hash_with_unsupported_algorithm():
    try:
        hash_with("unsupported")
    except ValueError as e:
        assert str(e) == "Unsupported hash algorithm: unsupported"

def test_hash_with_non_string_input():
    hash_func = hash_with("sha256")
    try:
        hash_func(123)
    except TypeError as e:
        assert str(e) == "hash_with() requires a string, got int"


# LLM-generated content at query #2
#--------------------------

```python
def test_suffix_adds_correct_suffix():
    add_io = suffix(".io")
    assert add_io("example") == "example.io"

def test_suffix_with_empty_string():
    add_empty = suffix("")
    assert add_empty("test") == "test"

def test_suffix_raises_type_error_for_non_string():
    add_io = suffix(".io")
    try:
        add_io(123)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "suffix() requires a string, got int"


# LLM-generated content at query #3
#--------------------------

```python
def test_pipe_empty():
    result = pipe()("test")
    assert result == "test"

def test_pipe_single_function():
    def upper(result: str) -> str:
        return result.upper()

    result = pipe(upper)("test")
    assert result == "TEST"

def test_pipe_multiple_functions():
    def upper(result: str) -> str:
        return result.upper()

    def add_prefix(result: str) -> str:
        return f"prefix_{result}"

    result = pipe(upper, add_prefix)("test")
    assert result == "prefix_TEST"

def test_pipe_with_random():
    def add_random_number(result: str, random: Random) -> str:
        return f"{result}_{random.randint(1, 100)}"

    random = Random()
    result = pipe(add_random_number)("test", random)
    assert result.startswith("test_")
    assert result[5:].isdigit()

def test_pipe_with_exception_handling():
    def upper(result: str) -> str:
        return result.upper()

    def add_prefix(result: str, random: Random) -> str:
        return f"prefix_{result}"

    random = Random()
    result = pipe(upper, add_prefix)("test", random)
    assert result == "prefix_TEST"


# LLM-generated content at query #4
#--------------------------

```python
def test_truncate_with_max_length_less_than_string_length():
    truncate_func = truncate(10)
    assert truncate_func("Hello, World!") == "Hello, W..."

def test_truncate_with_max_length_equal_to_string_length():
    truncate_func = truncate(13)
    assert truncate_func("Hello, World!") == "Hello, World!"

def test_truncate_with_max_length_greater_than_string_length():
    truncate_func = truncate(20)
    assert truncate_func("Hello") == "Hello"

def test_truncate_with_custom_suffix():
    truncate_func = truncate(8, "..")
    assert truncate_func("Hello, World!") == "Hello.."

def test_truncate_with_empty_string():
    truncate_func = truncate(5)
    assert truncate_func("") == ""

def test_truncate_with_max_length_zero_raises_value_error():
    try:
        truncate(0)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "max_length must be positive, got 0"

def test_truncate_with_negative_max_length_raises_value_error():
    try:
        truncate(-5)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "max_length must be positive, got -5"

def test_truncate_with_non_string_input_raises_type_error():
    truncate_func = truncate(10)
    try:
        truncate_func(123)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == "truncate() requires a string, got int"


# LLM-generated content at query #5
#--------------------------

```python
def test_pipe_docstring_starts_with_pipe_multiple_key_functions_together():
    assert pipe.__doc__.startswith("Pipe multiple key functions together.")


# LLM-generated content at query #6
#--------------------------

```python
def test_pipe_docstring_starts_with_pipe_multiple_key_functions_together():
    assert pipe.__doc__.startswith("Pipe multiple key functions together.")


# LLM-generated content at query #7
#--------------------------

```python
def test_romanize_with_valid_locale():
    ru_romanize = romanize(Locale.RU)
    assert ru_romanize("Привет") == "Privet"

    uk_romanize = romanize(Locale.UK)
    assert uk_romanize("Привіт") == "Privit"

    kk_romanize = romanize(Locale.KK)
    assert kk_romanize("Сәлем") == "Sälem"

def test_romanize_with_invalid_locale():
    try:
        romanize(Locale.EN)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_romanize_with_string_locale():
    ru_romanize = romanize("ru")
    assert ru_romanize("Привет") == "Privet"

def test_romanize_with_invalid_string_locale():
    try:
        romanize("invalid")
        assert False, "Expected LocaleError"
    except LocaleError:
        pass

def test_romanize_with_non_string_input():
    ru_romanize = romanize(Locale.RU)
    try:
        ru_romanize(123)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_apply_if_returns_transformed_value_when_condition_is_true():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(5)
    assert result == 10

def test_apply_if_returns_otherwise_value_when_condition_is_false():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x + 1)(-3)
    assert result == -2

def test_apply_if_returns_original_value_when_condition_is_false_and_no_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(-3)
    assert result == -3

def test_apply_if_works_with_strings():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hello")
    assert result == "HELLO"

def test_apply_if_works_with_strings_and_false_condition():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hi")
    assert result == "hi"

def test_apply_if_works_with_none_otherwise():
    result = apply_if(lambda x: len(x) > 3, str.upper)("hi")
    assert result == "hi"


# LLM-generated content at query #9
#--------------------------

```python
def test_maybe_returns_closure():
    key_func = maybe("test_value")
    assert callable(key_func)

def test_maybe_with_valid_probability():
    key_func = maybe("test_value", 0.7)
    random = Random()
    result = key_func("original_value", random)
    assert result in ["original_value", "test_value"]

def test_maybe_with_zero_probability():
    key_func = maybe("test_value", 0.0)
    random = Random()
    result = key_func("original_value", random)
    assert result == "original_value"

def test_maybe_with_one_probability():
    key_func = maybe("test_value", 1.0)
    random = Random()
    result = key_func("original_value", random)
    assert result == "test_value"

def test_maybe_with_invalid_probability():
    key_func = maybe("test_value", -0.5)
    random = Random()
    result = key_func("original_value", random)
    assert result == "original_value"

def test_maybe_with_probability_above_one():
    key_func = maybe("test_value", 1.5)
    random = Random()
    result = key_func("original_value", random)
    assert result == "original_value"


# LLM-generated content at query #10
#--------------------------

```python
def test_apply_if_condition_true():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)(5)
    assert result == 10

def test_apply_if_condition_false_with_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)(-5)
    assert result == -15

def test_apply_if_condition_false_without_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(-5)
    assert result == -5

def test_apply_if_string_condition_true():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hello")
    assert result == "HELLO"

def test_apply_if_string_condition_false_with_otherwise():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hi")
    assert result == "hi"

def test_apply_if_string_condition_false_without_otherwise():
    result = apply_if(lambda x: len(x) > 3, str.upper)("hi")
    assert result == "hi"


# LLM-generated content at query #11
#--------------------------

```python
def test_pipe_docstring_starts_with_pipe_multiple_key_functions_together():
    assert pipe.__doc__.startswith("Pipe multiple key functions together.")


# LLM-generated content at query #12
#--------------------------

```python
def test_condition_false():
    condition = lambda x: False
    transform = lambda x: x
    otherwise = lambda x: x
    result = apply_if(condition, transform, otherwise)("test")
    assert result == "test"


# LLM-generated content at query #13
#--------------------------

```python
def test_condition_false():
    condition = lambda x: False
    transform = lambda x: x
    otherwise = lambda x: x
    result = apply_if(condition, transform, otherwise)("test")
    assert result == "test"


# LLM-generated content at query #14
#--------------------------

```python
def test_pipe_docstring_starts_with_pipe_multiple_key_functions_together():
    assert pipe.__doc__.startswith("Pipe multiple key functions together.")


# LLM-generated content at query #15
#--------------------------

```python
def test_condition_false_without_otherwise():
    condition = lambda x: False
    transform = lambda x: x.upper()
    apply_if_fn = apply_if(condition, transform)
    assert apply_if_fn("test") == "test"


# LLM-generated content at query #16
#--------------------------

```python
def test_pipe_empty_functions():
    result = pipe()("test")
    assert result == "test"

def test_pipe_single_function():
    def upper(result: str) -> str:
        return result.upper()

    result = pipe(upper)("test")
    assert result == "TEST"

def test_pipe_multiple_functions():
    def upper(result: str) -> str:
        return result.upper()

    def add_prefix(result: str) -> str:
        return f"prefix_{result}"

    result = pipe(upper, add_prefix)("test")
    assert result == "prefix_TEST"

def test_pipe_with_random():
    def add_random_number(result: str, random: Random) -> str:
        return f"{result}_{random.randint(1, 100)}"

    random = Random()
    result = pipe(add_random_number)("test", random)
    assert result.startswith("test_")
    assert result.split("_")[1].isdigit()

def test_pipe_with_exception_handling():
    def upper(result: str) -> str:
        return result.upper()

    def add_prefix(result: str, random: Random) -> str:
        return f"prefix_{result}"

    random = Random()
    result = pipe(upper, add_prefix)("test", random)
    assert result == "prefix_TEST"


# LLM-generated content at query #17
#--------------------------

```python
def test_condition_false_without_otherwise():
    result = apply_if(lambda x: False, lambda x: x + 1, None)(5)
    assert result == 5


# LLM-generated content at query #18
#--------------------------

```python
def test_pipe_docstring_starts_with_pipe_multiple_key_functions_together():
    assert pipe.__doc__.startswith("Pipe multiple key functions together.")


# LLM-generated content at query #19
#--------------------------

```python
def test_condition_evaluates_to_false():
    result = apply_if(lambda x: False, lambda x: x, lambda x: x)(None)
    assert result is None


# LLM-generated content at query #20
#--------------------------

```python
def test_pipe_docstring_predicate():
    assert pipe.__doc__.startswith("Pipe multiple key functions together.")


# LLM-generated content at query #21
#--------------------------

```python
def test_condition_evaluates_to_false():
    condition = lambda x: False
    transform = lambda x: x
    otherwise = lambda x: x
    result = apply_if(condition, transform, otherwise)("test")
    assert result == "test"


# LLM-generated content at query #22
#--------------------------

```python
def test_apply_if_with_true_condition():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)(5)
    assert result == 10

def test_apply_if_with_false_condition_and_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)(-5)
    assert result == -15

def test_apply_if_with_false_condition_and_no_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(-5)
    assert result == -5

def test_apply_if_with_zero():
    result = apply_if(lambda x: x != 0, lambda x: x + 1, lambda x: x - 1)(0)
    assert result == -1

def test_apply_if_with_string():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hello")
    assert result == "HELLO"

def test_apply_if_with_string_and_false_condition():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hi")
    assert result == "hi"

def test_apply_if_with_none_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, None)(-5)
    assert result == -5


# LLM-generated content at query #23
#--------------------------

```python
def test_apply_if_with_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)(5)
    assert result == 10


# LLM-generated content at query #24
#--------------------------

```python
def test_apply_if_predicate_true():
    condition = lambda x: x > 0
    transform = lambda x: x * 2
    otherwise = lambda x: x / 2
    result = apply_if(condition, transform, otherwise)(5)
    assert result == 10


# LLM-generated content at query #25
#--------------------------

```python
def test_apply_if_with_true_condition():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(5)
    assert result == 10

def test_apply_if_with_false_condition_and_no_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(-5)
    assert result == -5

def test_apply_if_with_false_condition_and_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)(-5)
    assert result == -15

def test_apply_if_with_string_condition_and_transform():
    result = apply_if(lambda x: len(x) > 3, str.upper)("hello")
    assert result == "HELLO"

def test_apply_if_with_string_condition_and_otherwise():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hi")
    assert result == "hi"

def test_apply_if_with_list_condition_and_transform():
    result = apply_if(lambda x: len(x) > 2, lambda x: x + [1])([1, 2, 3])
    assert result == [1, 2, 3, 1]

def test_apply_if_with_list_condition_and_otherwise():
    result = apply_if(lambda x: len(x) > 2, lambda x: x + [1], lambda x: x + [0])([1, 2])
    assert result == [1, 2, 0]


# LLM-generated content at query #26
#--------------------------

```python
def test_apply_if_with_true_condition():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)(1)
    assert result == 2


# LLM-generated content at query #27
#--------------------------

```python
def test_apply_if_returns_transformed_value_when_condition_is_true():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x)(5)
    assert result == 10


# LLM-generated content at query #28
#--------------------------

```python
def test_apply_if_with_true_condition():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(5)
    assert result == 10

def test_apply_if_with_false_condition_and_no_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(-5)
    assert result == -5

def test_apply_if_with_false_condition_and_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x * 3)(-5)
    assert result == -15

def test_apply_if_with_string_condition():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hello")
    assert result == "HELLO"

def test_apply_if_with_string_condition_false():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hi")
    assert result == "hi"


# LLM-generated content at query #29
#--------------------------

```python
def test_apply_if_with_true_condition():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(5)
    assert result == 10

def test_apply_if_with_false_condition_and_no_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2)(-5)
    assert result == -5

def test_apply_if_with_false_condition_and_otherwise():
    result = apply_if(lambda x: x > 0, lambda x: x * 2, lambda x: x + 1)(-5)
    assert result == -4

def test_apply_if_with_string_condition_and_transform():
    result = apply_if(lambda x: len(x) > 3, str.upper)("hello")
    assert result == "HELLO"

def test_apply_if_with_string_condition_and_otherwise():
    result = apply_if(lambda x: len(x) > 3, str.upper, str.lower)("hi")
    assert result == "hi"

def test_apply_if_with_none_value():
    result = apply_if(lambda x: x is not None, lambda x: x + 1)(None)
    assert result is None

def test_apply_if_with_none_value_and_otherwise():
    result = apply_if(lambda x: x is not None, lambda x: x + 1, lambda x: 0)(None)
    assert result == 0


# LLM-generated content at query #30
#--------------------------

```python
def test_pipe_single_function():
    result = pipe(str.upper)("hello", None)
    assert result == "HELLO"

def test_pipe_multiple_functions():
    result = pipe(str.lower, str.upper)("Hello", None)
    assert result == "HELLO"

def test_pipe_with_random():
    def add_random_number(value: str, random: Random) -> str:
        return f"{value}{random.randint(1, 100)}"

    random = Random()
    result = pipe(str.lower, add_random_number)("Hello", random)
    assert result.startswith("hello")
    assert result[5:].isdigit()

def test_pipe_empty_input():
    result = pipe(str.upper)("", None)
    assert result == ""

def test_pipe_with_none():
    result = pipe(str.upper)(None, None)
    assert result is None

def test_pipe_with_exception_handling():
    def custom_func(value: str) -> str:
        return value.upper()

    result = pipe(custom_func)("hello", None)
    assert result == "HELLO"

def test_pipe_with_mixed_functions():
    def add_prefix(value: str) -> str:
        return f"prefix-{value}"

    result = pipe(str.lower, add_prefix)("Hello", None)
    assert result == "prefix-hello"


# LLM-generated content at query #31
#--------------------------

```python
def test_pipe_applies_single_function():
    result = pipe(str.upper)("hello", None)
    assert result == "HELLO"

def test_pipe_applies_multiple_functions():
    result = pipe(str.upper, lambda x: x + "!")("hello", None)
    assert result == "HELLO!"

def test_pipe_handles_none_random():
    result = pipe(lambda x, r: x if r is None else x.upper())("test", None)
    assert result == "test"

def test_pipe_with_random_parameter():
    random = Random()
    result = pipe(lambda x, r: x + str(r.randint(0, 100)))("value", random)
    assert isinstance(result, str) and result.startswith("value")

def test_pipe_empty_functions_list():
    result = pipe()("unchanged", None)
    assert result == "unchanged"


# LLM-generated content at query #32
#--------------------------

```python
def test_pipe_with_single_function():
    def add_one(x, random=None):
        return x + 1

    piped = pipe(add_one)
    assert piped(5) == 6

def test_pipe_with_multiple_functions():
    def add_one(x, random=None):
        return x + 1

    def multiply_two(x, random=None):
        return x * 2

    piped = pipe(add_one, multiply_two)
    assert piped(5) == 12

def test_pipe_with_no_functions():
    piped = pipe()
    assert piped(5) == 5

def test_pipe_with_random_parameter():
    def add_random(x, random):
        return x + random.randint(1, 10)

    random = Random()
    random.seed(42)
    piped = pipe(add_random)
    assert piped(5, random) == 5 + random.randint(1, 10)

def test_pipe_with_mixed_functions():
    def add_one(x, random=None):
        return x + 1

    def multiply_two(x):
        return x * 2

    piped = pipe(add_one, multiply_two)
    assert piped(5) == 12


# LLM-generated content at query #33
#--------------------------

```python
def test_pipe_predicate_false():
    assert not pipe()


# LLM-generated content at query #34
#--------------------------

```python
def test_pipe_predicate_false():
    assert not (functions and all(callable(func) for func in functions))


