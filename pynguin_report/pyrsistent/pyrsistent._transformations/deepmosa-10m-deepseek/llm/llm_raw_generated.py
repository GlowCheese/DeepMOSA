####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
# --------------------------


def test_rex_matches_string():
    matcher = rex("^test.*")
    assert matcher("testing") == True


def test_rex_does_not_match_non_string():
    matcher = rex("^test.*")
    assert matcher(123) == False


def test_rex_returns_false_for_non_matching_string():
    matcher = rex("^test.*")
    assert matcher("hello") == False


def test_rex_matches_empty_string():
    matcher = rex("^$")
    assert matcher("") == True


def test_rex_returns_lambda():
    result = rex(".*")
    assert callable(result)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
# --------------------------


def test_rex_returns_callable():
    matcher = rex(r"^test")
    assert callable(matcher)


def test_rex_matches_correct_string():
    matcher = rex(r"^hello")
    assert matcher("hello world") is not None
    assert matcher("hello") is not None


def test_rex_does_not_match_incorrect_string():
    matcher = rex(r"^hello")
    assert matcher("world hello") is None


def test_rex_returns_none_for_non_string():
    matcher = rex(r"^hello")
    assert matcher(123) is None
    assert matcher(["hello"]) is None


def test_rex_uses_full_pattern():
    matcher = rex(r"^\d{3}-\d{2}$")
    assert matcher("123-45") is not None
    assert matcher("123-456") is None
    assert matcher("abc-12") is None


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
# --------------------------


def test__do_to_path_with_empty_path_and_callable_command():
    structure = [1, 2, 3]
    path = []
    command = len
    result = _do_to_path(structure, path, command)
    assert result == 3


def test__do_to_path_with_empty_path_and_non_callable_command():
    structure = {"a": 1}
    path = []
    command = "new_value"
    result = _do_to_path(structure, path, command)
    assert result == "new_value"


def test__do_to_path_with_single_key_path_and_discard_command():
    structure = {"a": 1, "b": 2}
    path = ["a"]
    command = discard
    result = _do_to_path(structure, path, command)
    assert result == {"b": 2}


def test__do_to_path_with_callable_key_spec_and_discard_command():
    structure = {"a": 1, "b": 2, "c": 3}
    path = [lambda k: k == "b"]
    command = discard
    result = _do_to_path(structure, path, command)
    assert result == {"a": 1, "c": 3}


def test__do_to_path_with_callable_key_spec_two_args_and_discard_command():
    structure = {"a": 1, "b": 2, "c": 3}
    path = [lambda k, v: v == 2]
    command = discard
    result = _do_to_path(structure, path, command)
    assert result == {"a": 1, "c": 3}


def test__do_to_path_with_nested_path_and_update_command():
    structure = {"a": {"b": 1}}
    path = ["a", "b"]
    command = 5
    result = _do_to_path(structure, path, command)
    assert result == {"a": {"b": 5}}


def test__do_to_path_with_non_existent_key_and_discard_command():
    structure = {"a": 1}
    path = ["b"]
    command = discard
    result = _do_to_path(structure, path, command)
    assert result == {"a": 1}


def test__do_to_path_with_list_structure_and_index_path():
    structure = [10, 20, 30]
    path = [1]
    command = 99
    result = _do_to_path(structure, path, command)
    assert result == [10, 99, 30]


def test__do_to_path_with_callable_key_spec_on_list_and_discard_command():
    structure = [10, 20, 30]
    path = [lambda i: i == 0]
    command = discard
    result = _do_to_path(structure, path, command)
    assert result == [20, 30]


# LLM-generated content at query #2
# --------------------------


def test_get_arity_with_no_parameters():
    def f():
        pass

    result = _get_arity(f)
    assert result == 0


def test_get_arity_with_positional_only_parameters():
    def f(a, b, /):
        pass

    result = _get_arity(f)
    assert result == 2


def test_get_arity_with_positional_or_keyword_parameters():
    def f(a, b, c):
        pass

    result = _get_arity(f)
    assert result == 3


def test_get_arity_with_mixed_parameters_and_defaults():
    def f(a, b, c=1, d=2):
        pass

    result = _get_arity(f)
    assert result == 2


def test_get_arity_with_keyword_only_parameters():
    def f(*, a, b):
        pass

    result = _get_arity(f)
    assert result == 0


def test_get_arity_with_var_positional():
    def f(*args):
        pass

    result = _get_arity(f)
    assert result == 0


def test_get_arity_with_var_keyword():
    def f(**kwargs):
        pass

    result = _get_arity(f)
    assert result == 0


def test_get_arity_with_all_parameter_kinds():
    def f(a, b, /, c, d=10, *args, e, f=20, **kwargs):
        pass

    result = _get_arity(f)
    assert result == 3


# LLM-generated content at query #3
# --------------------------


def test__get_arity_with_no_parameters():
    def f():
        pass

    result = _get_arity(f)
    expected = 0
    assert result == expected


def test__get_arity_with_positional_only_parameters():
    def f(a, b, /):
        pass

    result = _get_arity(f)
    expected = 2
    assert result == expected


def test__get_arity_with_positional_or_keyword_parameters():
    def f(a, b):
        pass

    result = _get_arity(f)
    expected = 2
    assert result == expected


def test__get_arity_with_mixed_parameters_and_defaults():
    def f(a, b, c=1, d=2):
        pass

    result = _get_arity(f)
    expected = 2
    assert result == expected


def test__get_arity_with_keyword_only_parameters():
    def f(*, a, b):
        pass

    result = _get_arity(f)
    expected = 0
    assert result == expected


def test__get_arity_with_var_positional():
    def f(*args):
        pass

    result = _get_arity(f)
    expected = 0
    assert result == expected


def test__get_arity_with_var_keyword():
    def f(**kwargs):
        pass

    result = _get_arity(f)
    expected = 0
    assert result == expected


def test__get_arity_with_all_parameter_kinds():
    def f(a, b, /, c, d=10, *args, e, f=20, **kwargs):
        pass

    result = _get_arity(f)
    expected = 3
    assert result == expected


# LLM-generated content at query #4
# --------------------------


def test_rex_matches_correct_string():
    matcher = rex("^test.*")
    result = matcher("testing")
    assert result is not None


def test_rex_does_not_match_incorrect_string():
    matcher = rex("^test.*")
    result = matcher("wrong")
    assert result is None


def test_rex_returns_none_for_non_string_input():
    matcher = rex("^test.*")
    result = matcher(123)
    assert result is None


def test_rex_matches_empty_string_with_empty_pattern():
    matcher = rex("")
    result = matcher("")
    assert result is not None


def test_rex_handles_special_regex_characters():
    matcher = rex("^\\d+$")
    result = matcher("123")
    assert result is not None


def test_rex_does_not_match_partial_string_without_wildcard():
    matcher = rex("^test$")
    result = matcher("testing")
    assert result is None


# LLM-generated content at query #5
# --------------------------


def test__get_keys_and_values_with_callable_unary():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1), ("c", 3)]
    assert result == expected


def test__get_keys_and_values_with_callable_binary():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2), ("c", 3)]
    assert result == expected


def test__get_keys_and_values_with_callable_arity_error():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


def test__get_keys_and_values_with_non_callable_key_mapping():
    structure = {"a": 1, "b": 2}
    key_spec = "a"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected


def test__get_keys_and_values_with_non_callable_key_sequence():
    structure = [10, 20, 30]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20)]
    assert result == expected


def test__get_keys_and_values_with_non_callable_key_missing_mapping():
    structure = {"a": 1}
    key_spec = "b"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", _EMPTY_SENTINEL)]
    assert result == expected


def test__get_keys_and_values_with_non_callable_key_missing_sequence():
    structure = [10]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    expected = [(5, _EMPTY_SENTINEL)]
    assert result == expected


def test__get_keys_and_values_with_callable_on_sequence_unary():
    structure = [5, 10, 15]
    key_spec = lambda i: i % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 5), (2, 15)]
    assert result == expected


def test__get_keys_and_values_with_callable_on_sequence_binary():
    structure = [5, 10, 15]
    key_spec = lambda i, v: v > 5
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 10), (2, 15)]
    assert result == expected


def test__get_keys_and_values_with_object_attr():
    class Obj:
        def __init__(self):
            self.x = 100

    structure = Obj()
    key_spec = "x"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("x", 100)]
    assert result == expected


def test__get_keys_and_values_with_object_attr_missing():
    class Obj:
        def __init__(self):
            self.x = 100

    structure = Obj()
    key_spec = "y"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("y", _EMPTY_SENTINEL)]
    assert result == expected
