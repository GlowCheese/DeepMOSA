####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_maybe_returns_value_with_given_probability():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    key_func = maybe("special", 0.7)
    result = key_func("default", mock_random)
    assert result == "special"

def test_maybe_returns_first_argument_when_probability_not_met():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[0]]
    key_func = maybe("special", 0.3)
    result = key_func("default", mock_random)
    assert result == "default"

def test_maybe_returns_first_argument_when_probability_zero():
    mock_random = Random()
    key_func = maybe("special", 0.0)
    result = key_func("default", mock_random)
    assert result == "default"

def test_maybe_returns_first_argument_when_probability_negative():
    mock_random = Random()
    key_func = maybe("special", -0.5)
    result = key_func("default", mock_random)
    assert result == "default"

def test_maybe_returns_value_when_probability_one():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    key_func = maybe("special", 1.0)
    result = key_func("default", mock_random)
    assert result == "special"

def test_maybe_works_with_different_value_types():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    key_func = maybe(123, 0.8)
    result = key_func(456, mock_random)
    assert result == 123

def test_maybe_works_with_none_value():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    key_func = maybe(None, 0.6)
    result = key_func("not_none", mock_random)
    assert result is None

def test_maybe_returns_first_argument_when_probability_exceeds_one():
    mock_random = Random()
    key_func = maybe("special", 1.5)
    result = key_func("default", mock_random)
    assert result == "default"


# LLM-generated content at query #2
#--------------------------

def test_maybe_returns_original_result_when_probability_is_zero():
    from mimesis.keys import maybe
    from mimesis.random import Random
    random_instance = Random()
    key_func = maybe(value="special", probability=0.0)
    original = "original"
    result = key_func(original, random_instance)
    assert result == original

def test_maybe_returns_original_result_when_probability_is_negative():
    from mimesis.keys import maybe
    from mimesis.random import Random
    random_instance = Random()
    key_func = maybe(value="special", probability=-0.5)
    original = "original"
    result = key_func(original, random_instance)
    assert result == original

def test_maybe_returns_original_result_when_probability_is_greater_than_one():
    from mimesis.keys import maybe
    from mimesis.random import Random
    random_instance = Random()
    key_func = maybe(value="special", probability=1.5)
    original = "original"
    result = key_func(original, random_instance)
    assert result == original


# LLM-generated content at query #3
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

def test_apply_if_otherwise_is_none():
    condition = lambda x: x % 2 == 0
    transform = lambda x: x * 10
    func = apply_if(condition, transform, None)
    result = func(3)
    assert result == 3

def test_apply_if_condition_true_with_complex_object():
    condition = lambda d: "valid" in d
    transform = lambda d: {**d, "processed": True}
    func = apply_if(condition, transform)
    data = {"valid": True, "value": 10}
    result = func(data)
    assert result == {"valid": True, "value": 10, "processed": True}

def test_apply_if_condition_false_with_complex_object_and_otherwise():
    condition = lambda d: "valid" in d
    transform = lambda d: {**d, "processed": True}
    otherwise = lambda d: {**d, "processed": False}
    func = apply_if(condition, transform, otherwise)
    data = {"value": 10}
    result = func(data)
    assert result == {"value": 10, "processed": False}


# LLM-generated content at query #4
#--------------------------

def test_prefix_returns_function():
    func = prefix('user_')
    assert callable(func)

def test_prefix_adds_prefix():
    func = prefix('user_')
    result = func('order')
    assert result == 'user_order'

def test_prefix_works_with_empty_string():
    func = prefix('')
    result = func('test')
    assert result == 'test'

def test_prefix_raises_type_error_for_non_string_input():
    func = prefix('user_')
    try:
        func(123)
        assert False
    except TypeError as e:
        assert str(e) == "prefix() requires a string, got int"

def test_prefix_raises_type_error_for_none_input():
    func = prefix('user_')
    try:
        func(None)
        assert False
    except TypeError as e:
        assert str(e) == "prefix() requires a string, got NoneType"

def test_prefix_with_special_characters():
    func = prefix('pre_')
    result = func('fix@123')
    assert result == 'pre_fix@123'

def test_prefix_with_empty_input_string():
    func = prefix('pre_')
    result = func('')
    assert result == 'pre_'

def test_prefix_multiple_calls():
    func = prefix('test_')
    assert func('one') == 'test_one'
    assert func('two') == 'test_two'


# LLM-generated content at query #5
#--------------------------

def test_romanize_returns_callable_for_supported_locales():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    key_func = romanize(Locale.RU)
    result = isinstance(key_func, type(lambda: None))
    assert result == True
    key_func = romanize(Locale.UK)
    result = isinstance(key_func, type(lambda: None))
    assert result == True
    key_func = romanize(Locale.KK)
    result = isinstance(key_func, type(lambda: None))
    assert result == True

def test_romanize_raises_value_error_for_unsupported_locale():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    try:
        romanize(Locale.EN)
        raised = False
    except ValueError:
        raised = True
    assert raised == True

def test_romanize_accepts_locale_string():
    from mimesis.keys import romanize
    key_func = romanize("ru")
    result = isinstance(key_func, type(lambda: None))
    assert result == True

def test_romanize_raises_locale_error_for_invalid_locale_string():
    from mimesis.locales import LocaleError
    from mimesis.keys import romanize
    try:
        romanize("invalid")
        raised = False
    except LocaleError:
        raised = True
    assert raised == True

def test_romanize_raises_locale_error_for_invalid_locale_type():
    from mimesis.locales import LocaleError
    from mimesis.keys import romanize
    try:
        romanize(123)
        raised = False
    except LocaleError:
        raised = True
    assert raised == True

def test_returned_function_romanizes_russian_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    key_func = romanize(Locale.RU)
    result = key_func("Привет")
    assert result == "Privet"

def test_returned_function_romanizes_ukrainian_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    key_func = romanize(Locale.UK)
    result = key_func("Привіт")
    assert result == "Pryvit"

def test_returned_function_romanizes_kazakh_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    key_func = romanize(Locale.KK)
    result = key_func("Сәлем")
    assert result == "Sälem"

def test_returned_function_handles_common_letters():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    key_func = romanize(Locale.RU)
    result = key_func("ёж")
    assert result == "yozh"

def test_returned_function_raises_type_error_for_non_string_input():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    key_func = romanize(Locale.RU)
    try:
        key_func(123)
        raised = False
    except TypeError:
        raised = True
    assert raised == True

def test_returned_function_returns_empty_string_for_empty_input():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    key_func = romanize(Locale.RU)
    result = key_func("")
    assert result == ""


# LLM-generated content at query #6
#--------------------------

def test_truncate_returns_original_string_when_shorter_than_max_length():
    truncator = truncate(10)
    result = truncator("short")
    assert result == "short"

def test_truncate_returns_original_string_when_equal_to_max_length():
    truncator = truncate(5)
    result = truncator("exact")
    assert result == "exact"

def test_truncate_truncates_and_adds_default_suffix():
    truncator = truncate(10)
    result = truncator("this is a long string")
    assert result == "this is..."

def test_truncate_truncates_and_adds_custom_suffix():
    truncator = truncate(10, suffix="!!")
    result = truncator("this is a long string")
    assert result == "this is !!"

def test_truncate_raises_value_error_for_non_positive_max_length():
    try:
        truncate(0)
        assert False
    except ValueError as e:
        assert str(e) == "max_length must be positive, got 0"

def test_truncate_raises_value_error_for_negative_max_length():
    try:
        truncate(-5)
        assert False
    except ValueError as e:
        assert str(e) == "max_length must be positive, got -5"

def test_truncate_raises_type_error_for_non_string_input():
    truncator = truncate(10)
    try:
        truncator(123)
        assert False
    except TypeError as e:
        assert str(e) == "truncate() requires a string, got int"

def test_truncate_works_with_empty_string():
    truncator = truncate(5)
    result = truncator("")
    assert result == ""

def test_truncate_works_with_suffix_longer_than_max_length():
    truncator = truncate(2, suffix="...")
    result = truncator("hello")
    assert result == ".."

def test_truncate_returns_suffix_only_when_max_length_equals_suffix_length():
    truncator = truncate(3, suffix="...")
    result = truncator("hello")
    assert result == "..."


# LLM-generated content at query #7
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    unsupported_locale = Locale.EN
    try:
        romanize(unsupported_locale)
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {unsupported_locale}"


# LLM-generated content at query #8
#--------------------------

def test_truncate_predicate_false():
    truncate_func = truncate(20)
    result = truncate_func('Ports are created')
    assert len(result) <= 20


# LLM-generated content at query #9
#--------------------------

def test_romanize_raises_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
        assert False
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


# LLM-generated content at query #10
#--------------------------

def test_pipe_with_key_functions_using_random():
    from mimesis.random import Random
    from mimesis.keys import pipe
    def add_one(x, random=None):
        return x + 1
    def multiply_two(x, random=None):
        return x * 2
    key_func = pipe(add_one, multiply_two)
    result = key_func(5, Random())
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
    from mimesis.random import Random
    from mimesis.keys import pipe
    def add_one(x):
        return x + 1
    def multiply_two(x, random=None):
        return x * 2
    key_func = pipe(add_one, multiply_two)
    result = key_func(5, Random())
    assert result == 12

def test_pipe_with_string_operations():
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

def test_pipe_with_nested_pipe():
    from mimesis.keys import pipe
    def add_one(x):
        return x + 1
    def multiply_two(x):
        return x * 2
    inner_pipe = pipe(add_one, multiply_two)
    outer_pipe = pipe(inner_pipe, multiply_two)
    result = outer_pipe(3)
    assert result == 16


# LLM-generated content at query #11
#--------------------------

def test_romanize_returns_callable_for_supported_locales():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer_ru = romanize(Locale.RU)
    romanizer_uk = romanize(Locale.UK)
    romanizer_kk = romanize(Locale.KK)
    assert callable(romanizer_ru)
    assert callable(romanizer_uk)
    assert callable(romanizer_kk)

def test_romanize_raises_value_error_for_unsupported_locale():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    try:
        romanize(Locale.EN)
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {Locale.EN}"

def test_romanize_accepts_locale_string():
    from mimesis.keys import romanize
    romanizer_ru = romanize("ru")
    assert callable(romanizer_ru)

def test_romanize_raises_locale_error_for_invalid_locale_string():
    from mimesis.locales import LocaleError
    from mimesis.keys import romanize
    try:
        romanize("invalid")
    except LocaleError as e:
        assert str(e) == "invalid"

def test_romanize_raises_locale_error_for_invalid_locale_type():
    from mimesis.locales import LocaleError
    from mimesis.keys import romanize
    try:
        romanize(123)
    except LocaleError as e:
        assert str(e) == 123

def test_romanizer_raises_type_error_for_non_string_input():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.RU)
    try:
        romanizer(123)
    except TypeError as e:
        assert str(e) == "romanize() requires a string, got int"

def test_romanizer_translates_russian_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.RU)
    result = romanizer("Привет мир")
    expected = "Privet mir"
    assert result == expected

def test_romanizer_translates_ukrainian_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.UK)
    result = romanizer("Привіт світ")
    expected = "Pryvit svit"
    assert result == expected

def test_romanizer_translates_kazakh_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.KK)
    result = romanizer("Сәлем әлем")
    expected = "Sälem älem"
    assert result == expected

def test_romanizer_handles_empty_string():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.RU)
    result = romanizer("")
    assert result == ""

def test_romanizer_handles_common_letters():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.RU)
    result = romanizer("ёж")
    expected = "yozh"
    assert result == expected


# LLM-generated content at query #12
#--------------------------

def test_validate_locale_returns_locale_when_locale_is_locale_instance():
    locale = Locale.RU
    result = validate_locale(locale)
    assert result == locale


# LLM-generated content at query #13
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
        assert False
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


# LLM-generated content at query #14
#--------------------------

def test_romanize_returns_callable_for_supported_locales():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanize_ru = romanize(Locale.RU)
    romanize_uk = romanize(Locale.UK)
    romanize_kk = romanize(Locale.KK)
    assert callable(romanize_ru)
    assert callable(romanize_uk)
    assert callable(romanize_kk)

def test_romanize_raises_value_error_for_unsupported_locale():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    try:
        romanize(Locale.EN)
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {Locale.EN}"

def test_romanize_raises_value_error_for_unsupported_locale_string():
    from mimesis.keys import romanize
    try:
        romanize("en")
    except ValueError as e:
        assert str(e) == "Romanization is not available for: en"

def test_romanize_raises_type_error_for_non_string_input():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanize_ru = romanize(Locale.RU)
    try:
        romanize_ru(123)
    except TypeError as e:
        assert str(e) == "romanize() requires a string, got int"

def test_romanize_raises_type_error_for_none_input():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanize_ru = romanize(Locale.RU)
    try:
        romanize_ru(None)
    except TypeError as e:
        assert str(e) == "romanize() requires a string, got NoneType"

def test_romanize_works_with_string_locale():
    from mimesis.keys import romanize
    romanize_ru = romanize("ru")
    assert callable(romanize_ru)

def test_romanize_raises_locale_error_for_invalid_locale_string():
    from mimesis.keys import romanize
    try:
        romanize("invalid")
    except Exception as e:
        assert e.__class__.__name__ == "LocaleError"

def test_romanize_raises_locale_error_for_invalid_locale_type():
    from mimesis.keys import romanize
    try:
        romanize(123)
    except Exception as e:
        assert e.__class__.__name__ == "LocaleError"

def test_romanize_translates_russian_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanize_ru = romanize(Locale.RU)
    result = romanize_ru("Привет")
    assert result == "Privet"

def test_romanize_translates_ukrainian_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanize_uk = romanize(Locale.UK)
    result = romanize_uk("Привіт")
    assert result == "Pryvit"

def test_romanize_translates_kazakh_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanize_kk = romanize(Locale.KK)
    result = romanize_kk("Сәлем")
    assert result == "Sälem"

def test_romanize_handles_empty_string():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanize_ru = romanize(Locale.RU)
    result = romanize_ru("")
    assert result == ""

def test_romanize_handles_mixed_case():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanize_ru = romanize(Locale.RU)
    result = romanize_ru("ПрИвЕт")
    assert result == "PrIvEt"

def test_romanize_handles_common_letters():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanize_ru = romanize(Locale.RU)
    result = romanize_ru("ёЁ")
    assert result == "eE"


# LLM-generated content at query #15
#--------------------------

def test_pipe_handles_key_functions_without_random_parameter():
    def add_exclamation(text):
        return text + "!"
    def make_upper(text):
        return text.upper()
    piped = pipe(add_exclamation, make_upper)
    result = piped("hello", None)
    assert result == "HELLO!"


# LLM-generated content at query #16
#--------------------------

def test_pipe_handles_key_functions_without_random_parameter():
    def func1(x):
        return x + 1
    def func2(x):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, None)
    assert result == 12


# LLM-generated content at query #17
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


# LLM-generated content at query #18
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
        assert False
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


# LLM-generated content at query #19
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
    assert result == "TEST_HELLOTEST_HELLO"

def test_pipe_with_mixed_functions():
    def add_suffix(suffix):
        def inner(value):
            return value + suffix
        return inner
    def double(value, random=None):
        return value * 2
    key_func = pipe(double, add_suffix("!"))
    result = key_func("hi", None)
    assert result == "hihi!"

def test_pipe_single_function():
    def increment(value, random=None):
        return value + 1
    key_func = pipe(increment)
    result = key_func(5, None)
    assert result == 6

def test_pipe_no_functions():
    key_func = pipe()
    result = key_func("anything", None)
    assert result == "anything"

def test_pipe_with_random_argument():
    def add_random(value, random):
        return value + str(random.randint(1, 10))
    mock_random = Random()
    mock_random.randint = lambda a, b: 7
    key_func = pipe(add_random)
    result = key_func("num", mock_random)
    assert result == "num7"

def test_pipe_chain_with_and_without_random():
    def with_random(value, random):
        return value + random.choice(["a", "b"])
    def without_random(value):
        return value.upper()
    mock_random = Random()
    mock_random.choice = lambda seq: "b"
    key_func = pipe(with_random, without_random)
    result = key_func("test", mock_random)
    assert result == "TESTB"


# LLM-generated content at query #20
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
        assert False
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


# LLM-generated content at query #21
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    unsupported_locale = Locale.EN
    try:
        romanize(unsupported_locale)
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {unsupported_locale}"


# LLM-generated content at query #22
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


# LLM-generated content at query #23
#--------------------------

def test_pipe_handles_functions_without_random_parameter():
    def func1(x):
        return x + 1
    def func2(x):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, None)
    assert result == 12


# LLM-generated content at query #24
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
        assert False
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


# LLM-generated content at query #25
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    unsupported_locale = Locale.EN
    try:
        romanize(unsupported_locale)
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {unsupported_locale}"


# LLM-generated content at query #26
#--------------------------

def test_pipe_handles_functions_without_random_parameter():
    def lower(s):
        return s.lower()
    def add_prefix(s):
        return 'user-' + s
    key_func = pipe(lower, add_prefix)
    result = key_func('John Doe', None)
    assert result == 'user-john doe'


# LLM-generated content at query #27
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
        assert False
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


# LLM-generated content at query #28
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
        assert False
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


# LLM-generated content at query #29
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
        assert False
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


# LLM-generated content at query #30
#--------------------------

def test_pipe_with_random_parameter():
    def func1(x, random=None):
        return x + 1
    def func2(x):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, Random())
    assert result == 12


# LLM-generated content at query #31
#--------------------------

def test_pipe_handles_functions_without_random_parameter():
    def func1(x):
        return x + 1
    def func2(x):
        return x * 2
    key_func = pipe(func1, func2)
    result = key_func(5, None)
    assert result == 12


# LLM-generated content at query #32
#--------------------------

def test_validate_locale_returns_locale_when_locale_is_locale_instance():
    locale = Locale.RU
    result = validate_locale(locale)
    assert result == locale


# LLM-generated content at query #33
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
        assert False
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_suffix_returns_closure_that_adds_suffix():
    add_io = suffix('.io')
    result = add_io('recipe')
    assert result == 'recipe.io'

def test_suffix_works_with_empty_string():
    add_empty = suffix('')
    result = add_empty('hello')
    assert result == 'hello'

def test_suffix_works_with_multiple_characters():
    add_suffix = suffix('_suffix')
    result = add_suffix('test')
    assert result == 'test_suffix'

def test_suffix_raises_type_error_for_non_string_input():
    add_io = suffix('.io')
    try:
        add_io(123)
        assert False
    except TypeError as e:
        assert str(e) == "suffix() requires a string, got int"

def test_suffix_raises_type_error_for_none_input():
    add_io = suffix('.io')
    try:
        add_io(None)
        assert False
    except TypeError as e:
        assert str(e) == "suffix() requires a string, got NoneType"

def test_suffix_closure_can_be_called_multiple_times():
    add_dot = suffix('.')
    result1 = add_dot('a')
    result2 = add_dot('b')
    assert result1 == 'a.'
    assert result2 == 'b.'

def test_suffix_with_special_characters():
    add_special = suffix('!@#$')
    result = add_special('word')
    assert result == 'word!@#$'

def test_suffix_on_empty_input_string():
    add_ex = suffix('!')
    result = add_ex('')
    assert result == '!'


# LLM-generated content at query #2
#--------------------------

def test_maybe_returns_value_with_given_probability():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    key_func = maybe("special", 0.7)
    result = key_func("default", mock_random)
    assert result == "special"

def test_maybe_returns_first_argument_when_probability_not_met():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[0]]
    key_func = maybe("special", 0.3)
    result = key_func("default", mock_random)
    assert result == "default"

def test_maybe_returns_first_argument_when_probability_zero():
    mock_random = Random()
    key_func = maybe("special", 0.0)
    result = key_func("default", mock_random)
    assert result == "default"

def test_maybe_returns_first_argument_when_probability_negative():
    mock_random = Random()
    key_func = maybe("special", -0.5)
    result = key_func("default", mock_random)
    assert result == "default"

def test_maybe_returns_value_when_probability_one():
    mock_random = Random()
    key_func = maybe("special", 1.0)
    result = key_func("default", mock_random)
    assert result == "special"

def test_maybe_works_with_different_value_types():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    key_func = maybe(42, 0.8)
    result = key_func(0, mock_random)
    assert result == 42

def test_maybe_works_with_none_value():
    mock_random = Random()
    mock_random.choices = lambda population, weights, k: [population[1]]
    key_func = maybe(None, 0.6)
    result = key_func("not_none", mock_random)
    assert result is None

def test_maybe_uses_correct_weights():
    mock_random = Random()
    captured_population = []
    captured_weights = []
    def mock_choices(population, weights, k):
        captured_population.extend(population)
        captured_weights.extend(weights)
        return [population[1]]
    mock_random.choices = mock_choices
    key_func = maybe("value", 0.75)
    result = key_func("result", mock_random)
    assert captured_population == ["result", "value"]
    assert captured_weights == [0.25, 0.75]


# LLM-generated content at query #3
#--------------------------

def test_pipe_with_key_functions():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    def add_one(x, random=None):
        return x + 1
    def double(x, random=None):
        return x * 2
    piped = pipe(add_one, double)
    result = piped(5, random_instance)
    assert result == 12

def test_pipe_with_mixed_functions():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    def add_random(x, random):
        return x + random.randint(1, 10)
    def square(x):
        return x * x
    piped = pipe(add_random, square)
    result = piped(2, random_instance)
    assert isinstance(result, int)

def test_pipe_with_string_operations():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    def upper(s, random=None):
        return s.upper()
    def add_exclamation(s, random=None):
        return s + '!'
    piped = pipe(upper, add_exclamation)
    result = piped('hello', random_instance)
    assert result == 'HELLO!'

def test_pipe_with_no_functions():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    piped = pipe()
    result = piped('test', random_instance)
    assert result == 'test'

def test_pipe_with_single_function():
    from mimesis.keys import pipe
    from mimesis.random import Random
    random_instance = Random()
    def increment(x, random=None):
        return x + 10
    piped = pipe(increment)
    result = piped(5, random_instance)
    assert result == 15


# LLM-generated content at query #4
#--------------------------

def test_prefix_returns_function():
    func = prefix('user_')
    result = func('order')
    assert result == 'user_order'

def test_prefix_raises_type_error():
    func = prefix('user_')
    try:
        func(123)
        assert False
    except TypeError as e:
        assert str(e) == "prefix() requires a string, got int"

def test_prefix_with_empty_string():
    func = prefix('')
    result = func('order')
    assert result == 'order'

def test_prefix_with_empty_input_string():
    func = prefix('user_')
    result = func('')
    assert result == 'user_'

def test_prefix_with_special_characters():
    func = prefix('pre_')
    result = func('test@123')
    assert result == 'pre_test@123'


# LLM-generated content at query #5
#--------------------------

def test_join_with_default_separator():
    key_func = join()
    result = key_func(['a', 'b', 'c'])
    assert result == 'a, b, c'

def test_join_with_custom_separator():
    key_func = join(' | ')
    result = key_func(['pci', 'promise', 'excel'])
    assert result == 'pci | promise | excel'

def test_join_with_empty_list():
    key_func = join('-')
    result = key_func([])
    assert result == ''

def test_join_with_single_item():
    key_func = join(' ')
    result = key_func(['hello'])
    assert result == 'hello'

def test_join_with_non_string_items():
    key_func = join(', ')
    result = key_func([1, 2.5, True])
    assert result == '1, 2.5, True'

def test_join_raises_type_error_for_non_iterable():
    key_func = join()
    try:
        key_func(123)
        assert False
    except TypeError as e:
        assert str(e) == "join() requires iterable, got int"


# LLM-generated content at query #6
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
    result = key_func("hello")
    assert result == "HELLO"

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
    result = key_func("not_none")
    assert result == "not_none"

def test_apply_if_with_otherwise_as_none_argument():
    condition = lambda x: x > 10
    transform = lambda x: x * 10
    key_func = apply_if(condition, transform, None)
    result = key_func(5)
    assert result == 5


# LLM-generated content at query #7
#--------------------------

def test_romanize_with_valid_locale_ru():
    romanizer = romanize(Locale.RU)
    result = romanizer("Привет")
    assert result == "Privet"

def test_romanize_with_valid_locale_uk():
    romanizer = romanize(Locale.UK)
    result = romanizer("Привіт")
    assert result == "Pryvit"

def test_romanize_with_valid_locale_kk():
    romanizer = romanize(Locale.KK)
    result = romanizer("Сәлем")
    assert result == "Sälem"

def test_romanize_with_locale_string():
    romanizer = romanize("ru")
    result = romanizer("Привет")
    assert result == "Privet"

def test_romanize_raises_value_error_for_unsupported_locale():
    try:
        romanize(Locale.EN)
    except ValueError as e:
        assert str(e) == "Romanization is not available for: Locale.EN"

def test_romanize_raises_type_error_for_non_string_input():
    romanizer = romanize(Locale.RU)
    try:
        romanizer(123)
    except TypeError as e:
        assert str(e) == "romanize() requires a string, got int"

def test_romanize_raises_locale_error_for_invalid_locale_string():
    try:
        romanize("invalid")
    except LocaleError as e:
        assert str(e) == "invalid"

def test_romanize_raises_locale_error_for_non_locale_non_string():
    try:
        romanize(123)
    except LocaleError as e:
        assert str(e) == "123"

def test_romanize_common_letters_translation():
    romanizer = romanize(Locale.RU)
    result = romanizer("ёж")
    assert result == "yozh"

def test_romanize_empty_string():
    romanizer = romanize(Locale.RU)
    result = romanizer("")
    assert result == ""


# LLM-generated content at query #8
#--------------------------

def test_validate_locale_with_invalid_locale_type():
    result = validate_locale(123)


# LLM-generated content at query #9
#--------------------------

def test_pipe_with_key_functions():
    def add_prefix(prefix):
        def inner(value, random=None):
            return prefix + value
        return inner
    def add_suffix(suffix):
        def inner(value, random=None):
            return value + suffix
        return inner
    key_func = pipe(add_prefix("pre-"), add_suffix("-suf"))
    result = key_func("middle", None)
    assert result == "pre-middle-suf"

def test_pipe_with_mixed_functions():
    def to_upper(value, random=None):
        return value.upper()
    def repeat(value):
        return value * 2
    key_func = pipe(to_upper, repeat)
    result = key_func("ab", None)
    assert result == "ABAB"

def test_pipe_with_single_function():
    def increment(value, random=None):
        return value + 1
    key_func = pipe(increment)
    result = key_func(5, None)
    assert result == 6

def test_pipe_with_no_functions():
    key_func = pipe()
    result = key_func("test", None)
    assert result == "test"

def test_pipe_with_random_parameter():
    def add_random(value, random):
        return value + str(random.randint(1, 10))
    mock_random = Random()
    mock_random.randint = lambda a, b: 7
    key_func = pipe(add_random)
    result = key_func("value", mock_random)
    assert result == "value7"

def test_pipe_with_nested_pipes():
    def double(value, random=None):
        return value * 2
    def square(value):
        return value ** 2
    inner_pipe = pipe(double, square)
    outer_pipe = pipe(inner_pipe, double)
    result = outer_pipe(3, None)
    assert result == 36


# LLM-generated content at query #10
#--------------------------

def test_predicate_at_line_2_evaluates_to_false():
    test_value = "test"
    condition = lambda x: len(x) > 3
    transform = str.upper
    otherwise = str.lower
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(test_value)
    assert condition(test_value) == False


# LLM-generated content at query #11
#--------------------------

def test_pipe_applies_functions_in_sequence():
    def add_one(x):
        return x + 1
    def double(x):
        return x * 2
    piped = pipe(add_one, double)
    result = piped(5)
    assert result == 12

def test_pipe_with_random_parameter():
    def add_random(x, random):
        return x + random.randint(1, 10)
    def double(x):
        return x * 2
    piped = pipe(add_random, double)
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    result = piped(5, mock_random)
    assert result == 20

def test_pipe_with_mixed_functions():
    def add_one(x):
        return x + 1
    def add_random(x, random):
        return x + random.randint(1, 10)
    piped = pipe(add_one, add_random)
    mock_random = Random()
    mock_random.randint = lambda a, b: 5
    result = piped(5, mock_random)
    assert result == 11

def test_pipe_single_function():
    def square(x):
        return x * x
    piped = pipe(square)
    result = piped(4)
    assert result == 16

def test_pipe_no_functions():
    piped = pipe()
    result = piped(42)
    assert result == 42

def test_pipe_with_string_operations():
    def lower(s):
        return s.lower()
    def strip(s):
        return s.strip()
    piped = pipe(lower, strip)
    result = piped("  HELLO  ")
    assert result == "hello"

def test_pipe_handles_type_error():
    def func_without_random(x):
        return x * 2
    def func_with_random(x, random):
        return x + random.randint(1, 10)
    piped = pipe(func_without_random, func_with_random)
    mock_random = Random()
    mock_random.randint = lambda a, b: 3
    result = piped(5, mock_random)
    assert result == 13

def test_pipe_chain_of_three_functions():
    def add_one(x):
        return x + 1
    def double(x):
        return x * 2
    def square(x):
        return x * x
    piped = pipe(add_one, double, square)
    result = piped(2)
    assert result == 36


# LLM-generated content at query #12
#--------------------------

def test_apply_if_condition_false_without_otherwise():
    condition = lambda x: x > 10
    transform = lambda x: x * 2
    key_func = apply_if(condition, transform)
    result = key_func(5)
    assert result == 5

def test_apply_if_condition_false_with_otherwise():
    condition = lambda x: x.startswith('a')
    transform = lambda x: x.upper()
    otherwise = lambda x: x.lower()
    key_func = apply_if(condition, transform, otherwise)
    result = key_func('Banana')
    assert result == 'banana'

def test_apply_if_condition_false_with_falsy_value():
    condition = lambda x: bool(x)
    transform = lambda x: 'truthy'
    otherwise = lambda x: 'falsy'
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(0)
    assert result == 'falsy'

def test_apply_if_condition_false_with_none_input():
    condition = lambda x: x is not None
    transform = lambda x: 'not none'
    otherwise = lambda x: 'none'
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(None)
    assert result == 'none'

def test_apply_if_condition_false_with_empty_string():
    condition = lambda x: len(x) > 0
    transform = lambda x: 'non-empty'
    otherwise = lambda x: 'empty'
    key_func = apply_if(condition, transform, otherwise)
    result = key_func('')
    assert result == 'empty'


# LLM-generated content at query #13
#--------------------------

def test_truncate_returns_original_string_when_shorter_than_max_length():
    truncator = truncate(10)
    result = truncator("short")
    assert result == "short"

def test_truncate_returns_original_string_when_equal_to_max_length():
    truncator = truncate(5)
    result = truncator("exact")
    assert result == "exact"

def test_truncate_truncates_and_adds_default_suffix():
    truncator = truncate(10)
    result = truncator("This is a long sentence.")
    assert result == "This i..."

def test_truncate_truncates_and_adds_custom_suffix():
    truncator = truncate(10, suffix="!!")
    result = truncator("This is a long sentence.")
    assert result == "This is !!"

def test_truncate_raises_value_error_for_non_positive_max_length():
    try:
        truncate(0)
        assert False
    except ValueError as e:
        assert str(e) == "max_length must be positive, got 0"

def test_truncate_raises_type_error_for_non_string_input():
    truncator = truncate(5)
    try:
        truncator(123)
        assert False
    except TypeError as e:
        assert str(e) == "truncate() requires a string, got int"

def test_truncate_handles_empty_string():
    truncator = truncate(5)
    result = truncator("")
    assert result == ""

def test_truncate_with_suffix_longer_than_max_length():
    truncator = truncate(2, suffix="...")
    result = truncator("hello")
    assert result == "..."


# LLM-generated content at query #14
#--------------------------

def test_romanize_returns_callable_for_supported_locale():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.RU)
    assert callable(romanizer)

def test_romanize_raises_value_error_for_unsupported_locale():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    try:
        romanize(Locale.EN)
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {Locale.EN}"

def test_romanize_accepts_locale_string():
    from mimesis.keys import romanize
    romanizer = romanize("ru")
    assert callable(romanizer)

def test_romanize_raises_locale_error_for_invalid_locale_string():
    from mimesis.keys import romanize
    from mimesis.exceptions import LocaleError
    try:
        romanize("invalid")
    except LocaleError as e:
        assert str(e) == "invalid"

def test_romanize_raises_locale_error_for_invalid_locale_type():
    from mimesis.keys import romanize
    from mimesis.exceptions import LocaleError
    try:
        romanize(123)
    except LocaleError as e:
        assert str(e) == 123

def test_romanizer_raises_type_error_for_non_string_input():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.RU)
    try:
        romanizer(123)
    except TypeError as e:
        assert str(e) == "romanize() requires a string, got int"

def test_romanizer_translates_russian_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.RU)
    result = romanizer("Привет")
    expected = "Privet"
    assert result == expected

def test_romanizer_translates_ukrainian_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.UK)
    result = romanizer("Привіт")
    expected = "Pryvit"
    assert result == expected

def test_romanizer_translates_kazakh_text():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.KK)
    result = romanizer("Сәлем")
    expected = "Sälem"
    assert result == expected

def test_romanizer_handles_empty_string():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.RU)
    result = romanizer("")
    assert result == ""

def test_romanizer_handles_common_letters():
    from mimesis.keys import romanize
    from mimesis.enums import Locale
    romanizer = romanize(Locale.RU)
    result = romanizer("ёЁ")
    expected = "eË"
    assert result == expected


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

def test_apply_if_with_string_condition_true():
    condition = lambda s: len(s) > 3
    transform = str.upper
    func = apply_if(condition, transform)
    result = func("hello")
    assert result == "HELLO"

def test_apply_if_with_string_condition_false():
    condition = lambda s: len(s) > 3
    transform = str.upper
    func = apply_if(condition, transform)
    result = func("hi")
    assert result == "hi"

def test_apply_if_with_string_condition_false_and_otherwise():
    condition = lambda s: len(s) > 3
    transform = str.upper
    otherwise = str.lower
    func = apply_if(condition, transform, otherwise)
    result = func("HI")
    assert result == "hi"

def test_apply_if_with_none_input():
    condition = lambda x: x is not None
    transform = lambda x: "not none"
    otherwise = lambda x: "none"
    func = apply_if(condition, transform, otherwise)
    result = func(None)
    assert result == "none"

def test_apply_if_with_list_input():
    condition = lambda lst: len(lst) > 2
    transform = lambda lst: lst + [4]
    func = apply_if(condition, transform)
    result = func([1, 2, 3])
    assert result == [1, 2, 3, 4]

def test_apply_if_with_list_input_condition_false():
    condition = lambda lst: len(lst) > 2
    transform = lambda lst: lst + [4]
    func = apply_if(condition, transform)
    result = func([1, 2])
    assert result == [1, 2]


# LLM-generated content at query #16
#--------------------------

def test_pipe_with_random_parameter():
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
    key_func = pipe(to_upper, add_prefix("TEST_"), repeat)
    result = key_func("hello", Random())
    assert isinstance(result, str)
    assert result.startswith("TEST_")
    assert result[5:].isupper()

def test_pipe_without_random_parameter():
    def double(value):
        return value * 2
    def increment(value):
        return value + 1
    key_func = pipe(increment, double)
    result = key_func(5, None)
    assert result == 12

def test_pipe_mixed_functions():
    def add_suffix(suffix):
        def inner(value, random=None):
            return value + suffix
        return inner
    def capitalize(value):
        return value.capitalize()
    key_func = pipe(capitalize, add_suffix("!"))
    result = key_func("hello", Random())
    assert result == "Hello!"

def test_pipe_single_function():
    def square(value, random=None):
        return value * value
    key_func = pipe(square)
    result = key_func(4, None)
    assert result == 16

def test_pipe_empty_functions():
    key_func = pipe()
    result = key_func("test", Random())
    assert result == "test"


# LLM-generated content at query #17
#--------------------------

def test_apply_if_predicate_false():
    condition = lambda x: x > 10
    transform = lambda x: x * 2
    otherwise = lambda x: x + 5
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(5)
    assert result == 10


# LLM-generated content at query #18
#--------------------------

def test_romanize_raises_value_error_for_unsupported_locale():
    locale = Locale.EN
    try:
        romanize(locale)
        assert False
    except ValueError as e:
        assert str(e) == f"Romanization is not available for: {locale}"


# LLM-generated content at query #19
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    from typing import Any, Callable
    def apply_if(condition: Callable[[Any], bool], transform: Callable[[Any], Any], otherwise: Callable[[Any], Any] | None = None) -> Callable[[Any], Any]:
        def key(v: Any) -> Any:
            if condition(v):
                return transform(v)
            elif otherwise:
                return otherwise(v)
            return v
        return key
    condition = lambda x: x > 10
    transform = lambda x: x * 2
    otherwise = lambda x: x / 2
    key_func = apply_if(condition, transform, otherwise)
    result = key_func(5)
    assert result == 2.5


# LLM-generated content at query #20
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    result = apply_if(lambda x: False, lambda x: x * 2, lambda x: x + 1)(5)
    assert result == 6


# LLM-generated content at query #21
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
    random_instance = Random()
    result = piped(5, random_instance)
    assert result == (5 + random_instance.randint(1, 10)) * 2

def test_pipe_handles_mixed_functions():
    def add_one(x):
        return x + 1
    def add_random(x, random):
        return x + random.randint(1, 10)
    piped = pipe(add_one, add_random)
    random_instance = Random()
    result = piped(5, random_instance)
    assert result == (5 + 1) + random_instance.randint(1, 10)

def test_pipe_with_single_function():
    def square(x):
        return x * x
    piped = pipe(square)
    result = piped(4)
    assert result == 16

def test_pipe_with_no_functions():
    piped = pipe()
    result = piped(42)
    assert result == 42

def test_pipe_preserves_type_handling():
    def to_str(x):
        return str(x)
    def add_suffix(s):
        return s + "_end"
    piped = pipe(to_str, add_suffix)
    result = piped(100)
    assert result == "100_end"


# LLM-generated content at query #22
#--------------------------

def test_pipe_key_function_with_random_parameter():
    def add_prefix(prefix):
        def func(value, random=None):
            return prefix + value
        return func
    def to_upper(value, random=None):
        return value.upper()
    def repeat(value, random=None):
        if random is None:
            return value + value
        return value * random.randint(1, 3)
    from mimesis.random import Random
    rnd = Random()
    key_func = pipe(to_upper, add_prefix("TEST-"), repeat)
    result = key_func("hello", rnd)
    assert isinstance(result, str)
    assert result.startswith("TEST-")
    assert result[5:].isupper()


# LLM-generated content at query #23
#--------------------------

def test_pipe_with_key_functions():
    def add_one(x, random=None):
        return x + 1
    def double(x, random=None):
        return x * 2
    piped = pipe(add_one, double)
    result = piped(5, None)
    assert result == 12

def test_pipe_with_regular_functions():
    def add_one(x):
        return x + 1
    def double(x):
        return x * 2
    piped = pipe(add_one, double)
    result = piped(5, None)
    assert result == 12

def test_pipe_mixed_functions():
    def add_one(x, random=None):
        return x + 1
    def double(x):
        return x * 2
    piped = pipe(add_one, double)
    result = piped(5, None)
    assert result == 12

def test_pipe_single_function():
    def square(x, random=None):
        return x * x
    piped = pipe(square)
    result = piped(4, None)
    assert result == 16

def test_pipe_with_string_operations():
    def upper(s, random=None):
        return s.upper()
    def add_exclamation(s):
        return s + "!"
    piped = pipe(upper, add_exclamation)
    result = piped("hello", None)
    assert result == "HELLO!"

def test_pipe_with_random_parameter():
    def add_random(x, random):
        return x + random.randint(1, 10)
    def double(x, random=None):
        return x * 2
    mock_random = Random()
    mock_random.randint = lambda a, b: 3
    piped = pipe(add_random, double)
    result = piped(5, mock_random)
    assert result == 16

def test_pipe_with_no_functions():
    piped = pipe()
    result = piped("test", None)
    assert result == "test"

def test_pipe_chain_of_three():
    def add_one(x, random=None):
        return x + 1
    def double(x, random=None):
        return x * 2
    def square(x):
        return x * x
    piped = pipe(add_one, double, square)
    result = piped(3, None)
    assert result == 64


# LLM-generated content at query #24
#--------------------------

def test_pipe_with_key_functions_using_random():
    from mimesis.random import Random
    from mimesis.keys import pipe
    def add_one(x, random=None):
        return x + 1
    def multiply_two(x, random=None):
        return x * 2
    key_func = pipe(add_one, multiply_two)
    result = key_func(5, Random())
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
    from mimesis.random import Random
    from mimesis.keys import pipe
    def add_one(x):
        return x + 1
    def multiply_two(x, random=None):
        return x * 2
    key_func = pipe(add_one, multiply_two)
    result = key_func(5, Random())
    assert result == 12

def test_pipe_with_single_function():
    from mimesis.keys import pipe
    def square(x):
        return x * x
    key_func = pipe(square)
    result = key_func(4)
    assert result == 16

def test_pipe_with_string_operations():
    from mimesis.keys import pipe
    def lower(s):
        return s.lower()
    def strip(s):
        return s.strip()
    key_func = pipe(lower, strip)
    result = key_func("  HELLO  ")
    assert result == "hello"

def test_pipe_with_no_functions():
    from mimesis.keys import pipe
    key_func = pipe()
    result = key_func(42)
    assert result == 42


