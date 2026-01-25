####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test__do_to_path_empty_path_with_callable():
    result = _do_to_path([1, 2, 3], [], lambda x: sum(x))
    assert result == 6


def test__do_to_path_empty_path_with_non_callable():
    result = _do_to_path([1, 2, 3], [], [4, 5])
    assert result == [4, 5]


def test__do_to_path_single_key_spec():
    structure = {'a': 1, 'b': 2}
    path = ['a']
    command = lambda x: x * 2
    result = _do_to_path(structure, path, command)
    assert result == {'a': 2, 'b': 2}


def test__do_to_path_with_unary_predicate():
    structure = {0: 'a', 1: 'b', 2: 'c'}
    path = [lambda k: k % 2 == 0]
    command = lambda x: x.upper()
    result = _do_to_path(structure, path, command)
    assert result == {0: 'A', 1: 'b', 2: 'C'}


def test__do_to_path_with_binary_predicate():
    structure = {'x': 5, 'y': 10, 'z': 15}
    path = [lambda k, v: v > 7]
    command = lambda x: x * 2
    result = _do_to_path(structure, path, command)
    assert result == {'x': 5, 'y': 20, 'z': 30}


def test__do_to_path_with_discard_command():
    structure = {'a': 1, 'b': 2, 'c': 3}
    path = ['b']
    command = discard
    result = _do_to_path(structure, path, command)
    assert result == {'a': 1, 'c': 3}


def test__do_to_path_with_nested_path():
    structure = {'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}}
    path = ['a', 'x']
    command = lambda x: x + 10
    result = _do_to_path(structure, path, command)
    assert result == {'a': {'x': 11, 'y': 2}, 'b': {'x': 3, 'y': 4}}


def test__do_to_path_with_list_structure():
    structure = [10, 20, 30]
    path = [1]
    command = lambda x: x * 3
    result = _do_to_path(structure, path, command)
    assert result == [10, 60, 30]


def test__do_to_path_with_predicate_on_list():
    structure = [5, 10, 15]
    path = [lambda k: k == 2]
    command = lambda x: x - 5
    result = _do_to_path(structure, path, command)
    assert result == [5, 10, 10]


# LLM-generated content at query #2
#--------------------------

def test_rex_returns_callable():
    import re
    result = rex("^test.*")
    assert callable(result)

def test_rex_matches_correct_string():
    import re
    matcher = rex("^hello.*")
    assert matcher("hello world") is not None
    assert matcher("hello") is not None

def test_rex_does_not_match_incorrect_string():
    import re
    matcher = rex("^hello.*")
    assert matcher("world hello") is None

def test_rex_returns_none_for_non_string_input():
    import re
    matcher = rex("^test.*")
    assert matcher(123) is None
    assert matcher(None) is None
    assert matcher([]) is None

def test_rex_uses_fullmatch_behavior_with_end_anchor():
    import re
    matcher = rex("^exact$")
    assert matcher("exact") is not None
    assert matcher("exact extra") is None

def test_rex_pattern_with_special_characters():
    import re
    matcher = rex("^\d+\.\d+$")
    assert matcher("123.456") is not None
    assert matcher("abc.def") is None

def test_rex_case_sensitive_by_default():
    import re
    matcher = rex("^CaseSensitive$")
    assert matcher("CaseSensitive") is not None
    assert matcher("casesensitive") is None


# LLM-generated content at query #3
#--------------------------

def test__get_keys_and_values_with_dict_and_callable_unary():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("a", 1), ("c", 3)]

def test__get_keys_and_values_with_dict_and_callable_binary():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", 2), ("c", 3)]

def test__get_keys_and_values_with_list_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(0, 10), (2, 30)]

def test__get_keys_and_values_with_list_and_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20), (2, 30)]

def test__get_keys_and_values_with_dict_and_non_callable_key():
    structure = {"x": 100, "y": 200}
    key_spec = "x"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("x", 100)]

def test__get_keys_and_values_with_list_and_non_callable_key():
    structure = [5, 6, 7]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 6)]

def test__get_keys_and_values_with_dict_and_callable_arity_zero():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_dict_and_callable_arity_three():
    structure = {"a": 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_empty_dict_and_callable():
    structure = {}
    key_spec = lambda k: True
    result = _get_keys_and_values(structure, key_spec)
    assert result == []

def test__get_keys_and_values_with_empty_list_and_callable():
    structure = []
    key_spec = lambda k, v: True
    result = _get_keys_and_values(structure, key_spec)
    assert result == []

def test__get_keys_and_values_with_non_existent_non_callable_key():
    structure = {"a": 1}
    key_spec = "b"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_list_and_out_of_range_non_callable_key():
    structure = [1, 2, 3]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(5, _EMPTY_SENTINEL)]


# LLM-generated content at query #4
#--------------------------

def test_get_arity_with_no_parameters():
    def f():
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_one_positional_parameter():
    def f(a):
        pass
    result = _get_arity(f)
    assert result == 1

def test_get_arity_with_multiple_positional_parameters():
    def f(a, b, c):
        pass
    result = _get_arity(f)
    assert result == 3

def test_get_arity_with_keyword_only_parameter():
    def f(*, a):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_varargs():
    def f(*args):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_varkwargs():
    def f(**kwargs):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_positional_and_keyword_only():
    def f(a, *, b):
        pass
    result = _get_arity(f)
    assert result == 1

def test_get_arity_with_default_parameter():
    def f(a, b=1):
        pass
    result = _get_arity(f)
    assert result == 1

def test_get_arity_with_all_default_parameters():
    def f(a=1, b=2):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_positional_or_keyword_parameter():
    def f(a):
        pass
    result = _get_arity(f)
    assert result == 1

def test_get_arity_with_positional_only_parameter():
    def f(a, /):
        pass
    result = _get_arity(f)
    assert result == 1

def test_get_arity_with_mixed_parameter_types():
    def f(a, b=2, /, c, d=4, *, e, f=6):
        pass
    result = _get_arity(f)
    assert result == 2


# LLM-generated content at query #5
#--------------------------

def test_get_keys_and_values_with_unary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1), ("c", 3)]
    assert result == expected

def test_get_keys_and_values_with_binary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2), ("c", 3)]
    assert result == expected

def test_get_keys_and_values_with_non_callable_key():
    structure = {"a": 1, "b": 2}
    key_spec = "a"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected


# LLM-generated content at query #6
#--------------------------

def test_items_with_dict():
    result = _items({'a': 1, 'b': 2})
    assert list(result) == [('a', 1), ('b', 2)]

def test_items_with_list():
    result = _items([10, 20, 30])
    assert list(result) == [(0, 10), (1, 20), (2, 30)]

def test_items_with_tuple():
    result = _items(('x', 'y', 'z'))
    assert list(result) == [(0, 'x'), (1, 'y'), (2, 'z')]

def test_items_with_empty_dict():
    result = _items({})
    assert list(result) == []

def test_items_with_empty_list():
    result = _items([])
    assert list(result) == []

def test_items_with_string():
    result = _items('ab')
    assert list(result) == [(0, 'a'), (1, 'b')]


# LLM-generated content at query #7
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature

    def func_with_default(param1, param2=10):
        pass

    def func_without_default(param1, param2):
        pass

    def func_with_keyword_only(*, param1):
        pass

    def func_with_var_positional(*args):
        pass

    def func_with_mixed(param1, param2=20, *, kwparam, param3=30):
        pass

    params_with_default = list(signature(func_with_default).parameters.values())
    param_with_default = params_with_default[1]
    result_default = param_with_default.default is Parameter.empty
    assert result_default == False

    params_without_default = list(signature(func_without_default).parameters.values())
    param_without_default = params_without_default[0]
    result_no_default = param_without_default.default is Parameter.empty
    assert result_no_default == True

    params_keyword_only = list(signature(func_with_keyword_only).parameters.values())
    param_keyword_only = params_keyword_only[0]
    result_kind_check = param_keyword_only.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result_kind_check == False

    params_var_pos = list(signature(func_with_var_positional).parameters.values())
    param_var_pos = params_var_pos[0]
    result_var_pos_kind = param_var_pos.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result_var_pos_kind == False

    params_mixed = list(signature(func_with_mixed).parameters.values())
    param_with_default_mixed = params_mixed[1]
    result_default_mixed = param_with_default_mixed.default is Parameter.empty
    assert result_default_mixed == False

    param_kw_mixed = params_mixed[2]
    result_kw_kind_mixed = param_kw_mixed.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result_kw_kind_mixed == False


# LLM-generated content at query #8
#--------------------------

def test_unary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1), ("c", 3)]
    assert result == expected

def test_binary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2), ("c", 3)]
    assert result == expected


# LLM-generated content at query #9
#--------------------------

def test_rex_predicate_evaluates_to_false():
    import re
    result = rex("^a.*")("b")
    assert result is False


# LLM-generated content at query #10
#--------------------------

def test_items_returns_list_of_enumerated_items_when_structure_has_no_items_method():
    structure = [10, 20, 30]
    result = _items(structure)
    expected = list(enumerate(structure))
    assert result == expected


# LLM-generated content at query #11
#--------------------------

def test_get_arity_with_no_parameters():
    def f():
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_positional_only_parameters():
    def f(a, b, c):
        pass
    result = _get_arity(f)
    assert result == 3

def test_get_arity_with_positional_or_keyword_parameters():
    def f(a, b, c):
        pass
    result = _get_arity(f)
    assert result == 3

def test_get_arity_with_default_parameters():
    def f(a, b=2, c=3):
        pass
    result = _get_arity(f)
    assert result == 1

def test_get_arity_with_keyword_only_parameters():
    def f(*, a, b, c):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_var_positional_parameter():
    def f(*args):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_var_keyword_parameter():
    def f(**kwargs):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_mixed_parameter_types():
    def f(a, b=2, *args, c, d=4, **kwargs):
        pass
    result = _get_arity(f)
    assert result == 1


# LLM-generated content at query #12
#--------------------------

def test_update_structure_with_empty_path_and_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'b': 2})
    assert result == expected

def test_update_structure_with_non_empty_path_and_callable_command():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'x': 10})})
    kvs = [('a', pmap({'x': 10}))]
    path = ['x']
    command = lambda v: v * 2
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'x': 20})})
    assert result == expected

def test_update_structure_with_empty_sentinel_and_non_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = []
    command = lambda v: 100
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 1, 'b': 100})
    assert result == expected

def test_update_structure_with_empty_sentinel_and_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 1})
    assert result == expected

def test_update_structure_with_multiple_kvs_and_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2, 'c': 3})
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'c': 3})
    assert result == expected

def test_update_structure_with_nested_path_and_callable_command():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': pmap({'c': 5})})})
    kvs = [('a', pmap({'b': pmap({'c': 5})}))]
    path = ['b', 'c']
    command = lambda v: v + 10
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'b': pmap({'c': 15})})})
    assert result == expected

def test_update_structure_with_no_change_in_value():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = lambda v: v
    result = _update_structure(structure, kvs, path, command)
    assert result == structure

def test_update_structure_with_empty_kvs():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = []
    path = []
    command = lambda v: 99
    result = _update_structure(structure, kvs, path, command)
    assert result == structure


# LLM-generated content at query #13
#--------------------------

def test_predicate_at_line_5_evaluates_to_false():
    from inspect import signature, Parameter

    def func_with_default_param(a, b=1):
        pass

    param_values = list(signature(func_with_default_param).parameters.values())
    param_b = param_values[1]
    result = param_b.default is Parameter.empty and param_b.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result is False


# LLM-generated content at query #14
#--------------------------

def test_predicate_at_line_1_evaluates_to_false():
    result = _get_keys_and_values({}, lambda: None)
    assert result == []


# LLM-generated content at query #15
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature
    def dummy_func(a, b=1, *, c, d=2, **kwargs):
        pass
    params = signature(dummy_func).parameters.values()
    param_a = next(p for p in params if p.name == 'a')
    param_b = next(p for p in params if p.name == 'b')
    param_c = next(p for p in params if p.name == 'c')
    param_d = next(p for p in params if p.name == 'd')
    result_a = param_a.default is Parameter.empty and param_a.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    result_b = param_b.default is Parameter.empty and param_b.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    result_c = param_c.default is Parameter.empty and param_c.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    result_d = param_d.default is Parameter.empty and param_d.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result_a == True
    assert result_b == False
    assert result_c == False
    assert result_d == False


# LLM-generated content at query #16
#--------------------------

def test_predicate_at_line_1_evaluates_to_false():
    result = _get_keys_and_values({}, lambda x: True)
    assert result == []


# LLM-generated content at query #17
#--------------------------

def test_items_returns_list_of_enumerated_items_when_structure_has_no_items_method():
    structure = [10, 20, 30]
    result = _items(structure)
    expected = list(enumerate(structure))
    assert result == expected


# LLM-generated content at query #18
#--------------------------

def test__get_keys_and_values_with_mapping_and_callable_unary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, key_spec)
    expected = [('a', 1), ('c', 3)]
    assert sorted(result) == sorted(expected)

def test__get_keys_and_values_with_mapping_and_callable_binary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', 2), ('c', 3)]
    assert sorted(result) == sorted(expected)

def test__get_keys_and_values_with_sequence_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 10), (2, 30)]
    assert sorted(result) == sorted(expected)

def test__get_keys_and_values_with_sequence_and_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20), (2, 30)]
    assert sorted(result) == sorted(expected)

def test__get_keys_and_values_with_mapping_and_non_callable_key():
    structure = {'x': 100, 'y': 200}
    key_spec = 'x'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('x', 100)]
    assert result == expected

def test__get_keys_and_values_with_sequence_and_non_callable_key():
    structure = [5, 6, 7]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 6)]
    assert result == expected

def test__get_keys_and_values_with_callable_arity_zero():
    structure = {'a': 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_callable_arity_three():
    structure = {'a': 1}
    key_spec = lambda k, v, x: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_non_existent_non_callable_key_in_mapping():
    structure = {'a': 1}
    key_spec = 'b'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', _EMPTY_SENTINEL)]
    assert result == expected

def test__get_keys_and_values_with_out_of_range_non_callable_key_in_sequence():
    structure = [1, 2]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    expected = [(5, _EMPTY_SENTINEL)]
    assert result == expected


# LLM-generated content at query #19
#--------------------------

def test_predicate_at_line_1_evaluates_to_false():
    result = _get_keys_and_values({}, lambda: None)
    assert result == []


# LLM-generated content at query #20
#--------------------------

def test_predicate_at_line_1_evaluates_to_false():
    result = _get_keys_and_values({}, lambda: None)
    assert result == []


# LLM-generated content at query #21
#--------------------------

def test_update_structure_with_empty_path_and_discard_command():
    from pyrsistent import pmap, m
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._helpers import discard
    structure = m(a=1, b=2)
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = m()
    assert result == expected


# LLM-generated content at query #22
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature
    def func_with_default(param1, param2=10):
        pass
    parameters = signature(func_with_default).parameters.values()
    param_with_default = next(p for p in parameters if p.name == 'param2')
    result = param_with_default.default is Parameter.empty
    assert result == False


# LLM-generated content at query #23
#--------------------------

def test_update_structure_with_empty_path_and_discard_command():
    from pyrsistent import pmap
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._helpers import discard
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({})
    assert result == expected

def test_update_structure_with_non_empty_path_and_discard_command():
    from pyrsistent import pmap
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._helpers import discard
    structure = pmap({'a': pmap({'x': 10})})
    kvs = [('a', pmap({'x': 10}))]
    path = ['x']
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({})})
    assert result == expected

def test_update_structure_with_empty_path_and_non_discard_command():
    from pyrsistent import pmap
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._helpers import set
    structure = pmap({'a': 1})
    kvs = [('b', 2)]
    path = []
    command = set
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 1, 'b': 2})
    assert result == expected

def test_update_structure_with_empty_sentinel_and_discard_command():
    from pyrsistent import pmap
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._helpers import discard
    structure = pmap({'a': pmap({'x': 10})})
    kvs = [('a', _EMPTY_SENTINEL)]
    path = ['x']
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'x': 10})})
    assert result == expected

def test_update_structure_with_empty_sentinel_and_non_discard_command():
    from pyrsistent import pmap
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._helpers import set
    structure = pmap({'a': pmap({'x': 10})})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = ['y']
    command = set
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'x': 10}), 'b': pmap({'y': None})})
    assert result == expected


# LLM-generated content at query #24
#--------------------------

def test_items_with_dict():
    result = _items({'a': 1, 'b': 2})
    assert sorted(list(result)) == [('a', 1), ('b', 2)]

def test_items_with_list():
    result = _items([10, 20, 30])
    assert list(result) == [(0, 10), (1, 20), (2, 30)]

def test_items_with_tuple():
    result = _items((5, 15, 25))
    assert list(result) == [(0, 5), (1, 15), (2, 25)]

def test_items_with_empty_dict():
    result = _items({})
    assert list(result) == []

def test_items_with_empty_list():
    result = _items([])
    assert list(result) == []

def test_items_with_single_element():
    result = _items([99])
    assert list(result) == [(0, 99)]


# LLM-generated content at query #25
#--------------------------

def test_rex_returns_lambda_for_string_matching():
    import re
    result = rex("^test.*")
    assert callable(result)
    assert result("test_string") is not None
    assert result("other_string") is None
    assert result(123) is False


# LLM-generated content at query #26
#--------------------------

def test_predicate_arity_other_than_one_or_two_raises_error():
    def predicate_with_zero_args():
        return True
    structure = {}
    key_spec = predicate_with_zero_args
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_three_args_raises_error():
    def predicate_with_three_args(a, b, c):
        return True
    structure = {}
    key_spec = predicate_with_three_args
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #27
#--------------------------

def test_callable_key_spec_with_arity_1():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1), ("c", 3)]
    assert result == expected

def test_callable_key_spec_with_arity_2():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2), ("c", 3)]
    assert result == expected

def test_callable_key_spec_with_arity_0_raises_error():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_callable_key_spec_with_arity_3_raises_error():
    structure = {"a": 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_non_callable_key_spec():
    structure = {"a": 1, "b": 2}
    key_spec = "a"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected


# LLM-generated content at query #28
#--------------------------

def test_unary_predicate():
    key_spec = lambda k: k % 2 == 0
    structure = {0: 'a', 1: 'b', 2: 'c'}
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 'a'), (2, 'c')]
    assert result == expected

def test_binary_predicate():
    key_spec = lambda k, v: k > 0 and v.startswith('b')
    structure = {0: 'a', 1: 'b', 2: 'bc'}
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 'b'), (2, 'bc')]
    assert result == expected

def test_non_callable_key():
    key_spec = 'x'
    structure = {'x': 1, 'y': 2}
    result = _get_keys_and_values(structure, key_spec)
    expected = [('x', 1)]
    assert result == expected

def test_predicate_with_zero_arity():
    key_spec = lambda: True
    structure = {'a': 1}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_arity_three():
    key_spec = lambda a, b, c: True
    structure = {'a': 1}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #29
#--------------------------

def test_get_keys_and_values_with_unary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1), ("c", 3)]
    assert result == expected

def test_get_keys_and_values_with_binary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2), ("c", 3)]
    assert result == expected

def test_get_keys_and_values_with_non_callable_key():
    structure = {"a": 1, "b": 2}
    key_spec = "a"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected

def test_get_keys_and_values_with_predicate_arity_error():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_get_keys_and_values_with_unary_predicate_on_list():
    structure = [10, 20, 30]
    key_spec = lambda i: i % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 10), (2, 30)]
    assert result == expected

def test_get_keys_and_values_with_binary_predicate_on_list():
    structure = [10, 20, 30]
    key_spec = lambda i, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20), (2, 30)]
    assert result == expected


# LLM-generated content at query #30
#--------------------------

def test_update_structure_with_path_and_command_not_discard():
    from pyrsistent import pmap
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._pset import discard
    from pyrsistent._helpers import _do_to_path
    structure = pmap({"a": 1})
    kvs = [("b", 2)]
    path = ["x"]
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert isinstance(result, type(structure))


# LLM-generated content at query #31
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    from pyrsistent import pmap
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._helpers import discard
    structure = pmap({"a": 1})
    kvs = [("a", 1)]
    path = ["key"]
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({"a": 1})


# LLM-generated content at query #32
#--------------------------

def test_update_structure_with_empty_path_and_discard_command():
    from pyrsistent import pmap
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._pmap import discard
    structure = pmap({1: 2, 3: 4})
    kvs = [(1, _EMPTY_SENTINEL), (5, 6)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({3: 4})
    assert result == expected


# LLM-generated content at query #33
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    from pyrsistent import pmap
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._pset import discard
    from pyrsistent._helpers import _update_structure
    structure = pmap({"a": 1})
    kvs = [("a", 2)]
    path = ["key"]
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({"a": 1})


# LLM-generated content at query #34
#--------------------------

def test_update_structure_with_empty_path_and_discard_command():
    from pyrsistent import pmap
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._helpers import discard
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({})
    assert result == expected


# LLM-generated content at query #35
#--------------------------

def test_predicate_at_line_1_evaluates_to_false():
    result = _get_keys_and_values({}, lambda: None)
    assert result == []


# LLM-generated content at query #36
#--------------------------

def test_items_without_attribute_error_returns_items():
    class MockStructure:
        def items(self):
            return [("key", "value")]
    structure = MockStructure()
    result = _items(structure)
    assert result == [("key", "value")]

def test_items_with_attribute_error_returns_enumerated_list():
    structure = ["a", "b", "c"]
    result = _items(structure)
    assert result == [(0, "a"), (1, "b"), (2, "c")]

def test_items_with_attribute_error_on_empty_structure_returns_empty_list():
    structure = []
    result = _items(structure)
    assert result == []

def test_items_with_attribute_error_on_non_list_iterable():
    structure = ("x", "y")
    result = _items(structure)
    assert result == [(0, "x"), (1, "y")]


# LLM-generated content at query #37
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature

    def func_with_default_param(a, b=1):
        pass

    params = signature(func_with_default_param).parameters.values()
    param_b = next(p for p in params if p.name == 'b')
    result = param_b.default is Parameter.empty
    assert result == False


# LLM-generated content at query #38
#--------------------------

def test_callable_key_spec_with_arity_1():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1), ("c", 3)]
    assert result == expected

def test_callable_key_spec_with_arity_2():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2), ("c", 3)]
    assert result == expected

def test_callable_key_spec_with_arity_0_raises_error():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_callable_key_spec_with_arity_3_raises_error():
    structure = {"a": 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_non_callable_key_spec():
    structure = {"a": 1, "b": 2}
    key_spec = "a"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected

def test_callable_key_spec_with_arity_1_on_list():
    structure = [10, 20, 30]
    key_spec = lambda i: i % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 10), (2, 30)]
    assert result == expected

def test_callable_key_spec_with_arity_2_on_list():
    structure = [10, 20, 30]
    key_spec = lambda i, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20), (2, 30)]
    assert result == expected


# LLM-generated content at query #39
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    from pyrsistent._pmap import pmap
    from pyrsistent import m
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._transformations import discard
    from pyrsistent._transformations import _update_structure
    structure = m()
    kvs = [("key", "value")]
    path = ["some", "path"]
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result is not None


# LLM-generated content at query #40
#--------------------------

def test_update_structure_with_empty_path_and_discard_command():
    from pyrsistent._pmap import pmap
    from pyrsistent._pvector import pvector
    from pyrsistent._pset import pset
    from pyrsistent._pclass import PClass
    from pyrsistent._precord import PRecord
    from pyrsistent._pbag import pbag
    from pyrsistent._pdeque import pdeque
    from pyrsistent._pdict import pdict
    from pyrsistent._plist import plist
    from pyrsistent._checked_types import CheckedPMap, CheckedPVector, CheckedPSet
    from pyrsistent._checked_types import CheckedValue
    from pyrsistent._field_common import field
    from pyrsistent._helpers import freeze
    from pyrsistent._transformations import discard
    from pyrsistent._transformations import _EMPTY_SENTINEL
    from pyrsistent._transformations import _update_structure
    from pyrsistent._transformations import _do_to_path
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap()


# LLM-generated content at query #41
#--------------------------

def test__get_keys_and_values_with_callable_unary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('a', 1), ('c', 3)]

def test__get_keys_and_values_with_callable_binary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('b', 2), ('c', 3)]

def test__get_keys_and_values_with_callable_arity_error():
    structure = {'a': 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_non_callable_key():
    structure = {'a': 1, 'b': 2}
    key_spec = 'a'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('a', 1)]

def test__get_keys_and_values_with_non_callable_key_missing():
    structure = {'a': 1, 'b': 2}
    key_spec = 'c'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('c', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_list_structure():
    structure = [10, 20, 30]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20)]

def test__get_keys_and_values_with_list_structure_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k == 0
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(0, 10)]

def test__get_keys_and_values_with_list_structure_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v == 20
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20)]


# LLM-generated content at query #42
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature
    def mock_func_with_default_param(default_value="default"):
        pass
    param = list(signature(mock_func_with_default_param).parameters.values())[0]
    result = param.default is Parameter.empty and param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result is False


# LLM-generated content at query #43
#--------------------------

def test_get_keys_and_values_with_unary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("a", 1), ("c", 3)]

def test_get_keys_and_values_with_binary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", 2), ("c", 3)]

def test_get_keys_and_values_with_non_callable_key():
    structure = {"a": 1, "b": 2}
    key_spec = "a"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("a", 1)]

def test_get_keys_and_values_with_predicate_arity_error():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_get_keys_and_values_with_empty_structure_and_unary_predicate():
    structure = {}
    key_spec = lambda k: True
    result = _get_keys_and_values(structure, key_spec)
    assert result == []

def test_get_keys_and_values_with_sequence_and_unary_predicate():
    structure = [10, 20, 30]
    key_spec = lambda i: i % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(0, 10), (2, 30)]

def test_get_keys_and_values_with_sequence_and_binary_predicate():
    structure = [10, 20, 30]
    key_spec = lambda i, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20), (2, 30)]


# LLM-generated content at query #44
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    from pyrsistent._pmap import pmap
    from pyrsistent._pvector import pvector
    from pyrsistent import m, v
    from pyrsistent._pclass import _EMPTY_SENTINEL
    from pyrsistent._transformations import _update_structure, discard
    structure = m()
    kvs = [("key", "value")]
    path = v("nested", "path")
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert isinstance(result, pmap)
    structure = pvector([1, 2, 3])
    kvs = [(0, 10)]
    path = v()
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert isinstance(result, type(structure))
    structure = m("a", m("b", 1))
    kvs = [("a", m("b", 2))]
    path = v()
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result == m("a", m("b", 2))
    structure = m()
    kvs = []
    path = v()
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert result == m()
    structure = pvector([1, 2])
    kvs = [(0, _EMPTY_SENTINEL)]
    path = v()
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result == pvector([pmap(), 2])
    structure = m("x", 1)
    kvs = [("x", _EMPTY_SENTINEL)]
    path = v()
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert result == m("x", 1)


# LLM-generated content at query #45
#--------------------------

def test_update_structure_with_discard_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'b': 2})

def test_update_structure_with_discard_command_and_nested_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'x': 1, 'y': 2})})
    kvs = [('a', pmap({'x': 1, 'y': 2}))]
    path = ['x']
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': pmap({'y': 2})})

def test_update_structure_with_discard_command_and_multiple_kvs():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2, 'c': 3})
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'c': 3})

def test_update_structure_with_callable_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': 2})

def test_update_structure_with_callable_command_and_nested_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'x': 1})})
    kvs = [('a', pmap({'x': 1}))]
    path = ['x']
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': pmap({'x': 2})})

def test_update_structure_with_non_callable_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = 100
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': 100})

def test_update_structure_with_empty_sentinel_value_and_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': 1})

def test_update_structure_with_empty_sentinel_value_and_callable_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = []
    command = lambda x: 5
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': 1, 'b': 5})

def test_update_structure_with_empty_sentinel_value_and_nested_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = ['x']
    command = lambda x: 10
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': 1, 'b': pmap({'x': 10})})

def test_update_structure_with_unchanged_value_and_non_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'x': 1})})
    kvs = [('a', pmap({'x': 1}))]
    path = ['x']
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': pmap({'x': 1})})

def test_update_structure_with_unchanged_value_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': 1})


# LLM-generated content at query #46
#--------------------------

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

def test_get_arity_with_keyword_only_parameters():
    def f(*, a, b):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_var_positional_parameters():
    def f(*args):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_var_keyword_parameters():
    def f(**kwargs):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_default_parameters():
    def f(a, b=1, c=2):
        pass
    result = _get_arity(f)
    assert result == 1

def test_get_arity_with_mixed_parameters():
    def f(a, b, /, c, d=4, *, e, f=6):
        pass
    result = _get_arity(f)
    assert result == 3


# LLM-generated content at query #47
#--------------------------

def test_unary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1), ("c", 3)]
    assert result == expected

def test_binary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2), ("c", 3)]
    assert result == expected


# LLM-generated content at query #48
#--------------------------

def test_predicate_arity_other_than_one_or_two_raises_value_error():
    def predicate_with_zero_args():
        return True
    structure = {}
    key_spec = predicate_with_zero_args
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #49
#--------------------------

def test_update_structure_with_discard_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == pmap({'b': 2})

def test_update_structure_with_discard_command_and_non_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'x': 10}), 'b': 2})
    kvs = [('a', pmap({'x': 10}))]
    result = _update_structure(structure, kvs, ['x'], discard)
    assert result == pmap({'a': pmap({}), 'b': 2})

def test_update_structure_with_callable_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    command = lambda x: x * 2
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 2})

def test_update_structure_with_non_callable_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    command = 100
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 100})

def test_update_structure_with_empty_sentinel_value_and_non_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    command = lambda x: x * 2
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 1, 'b': pmap({})})

def test_update_structure_with_empty_sentinel_value_and_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == pmap({'a': 1})

def test_update_structure_with_multiple_kvs_and_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2, 'c': 3})
    kvs = [('a', 1), ('c', 3)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == pmap({'b': 2})

def test_update_structure_with_nested_structure_and_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'x': pmap({'i': 10})})})
    kvs = [('a', pmap({'x': pmap({'i': 10})}))]
    command = lambda x: x * 2
    result = _update_structure(structure, kvs, ['x', 'i'], command)
    assert result == pmap({'a': pmap({'x': pmap({'i': 20})})})

def test_update_structure_with_result_equal_to_original_value():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    command = lambda x: x
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 1})

def test_update_structure_with_empty_kvs_list():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = []
    command = lambda x: x * 2
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 1})


# LLM-generated content at query #50
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    from pyrsistent._pmap import pmap
    from pyrsistent._pvector import pvector
    from pyrsistent import m, v
    from pyrsistent._transform import _EMPTY_SENTINEL, discard
    from pyrsistent._transform import _update_structure
    structure = m()
    kvs = [("key", "value")]
    path = ["nested"]
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert isinstance(result, type(pmap()))


# LLM-generated content at query #51
#--------------------------

def test_predicate_with_arity_0_evaluates_to_false():
    def predicate():
        return True
    structure = {"a": 1}
    try:
        _get_keys_and_values(structure, predicate)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #52
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature

    def func_with_default_param(a, b=1):
        pass

    param = list(signature(func_with_default_param).parameters.values())[1]
    result = param.default is Parameter.empty
    assert result == False


# LLM-generated content at query #53
#--------------------------

def test_update_structure_with_discard_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == pmap({'b': 2})

def test_update_structure_with_discard_command_and_nested_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'x': 10, 'y': 20})})
    kvs = [('a', pmap({'x': 10, 'y': 20}))]
    result = _update_structure(structure, kvs, ['x'], discard)
    assert result == pmap({'a': pmap({'y': 20})})

def test_update_structure_with_callable_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    command = lambda x: x * 2
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 2})

def test_update_structure_with_non_callable_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    command = 100
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 100})

def test_update_structure_with_discard_command_and_sentinel_value():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == pmap({'a': 1})

def test_update_structure_with_command_and_sentinel_value_creating_new_node():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    command = 5
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 1, 'b': 5})

def test_update_structure_with_multiple_kvs_and_discard_reversed():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2, 'c': 3})
    kvs = [('a', 1), ('b', 2)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == pmap({'c': 3})

def test_update_structure_with_nested_structure_and_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': pmap({'c': 10})})})
    kvs = [('a', pmap({'b': pmap({'c': 10})}))]
    command = lambda x: x + 5
    result = _update_structure(structure, kvs, ['b', 'c'], command)
    assert result == pmap({'a': pmap({'b': pmap({'c': 15})})})

def test_update_structure_with_no_change_returns_same_structure():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    command = lambda x: x
    result = _update_structure(structure, kvs, [], command)
    assert result is structure

def test_update_structure_with_empty_kvs_returns_unchanged_structure():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = []
    command = discard
    result = _update_structure(structure, kvs, [], command)
    assert result == structure


# LLM-generated content at query #54
#--------------------------

def test_get_keys_and_values_with_unary_predicate():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k == 'b'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', 2)]
    assert result == expected

def test_get_keys_and_values_with_binary_predicate():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', 2), ('c', 3)]
    assert result == expected

def test_get_keys_and_values_with_non_callable_key():
    structure = {'a': 1, 'b': 2}
    key_spec = 'a'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('a', 1)]
    assert result == expected

def test_get_keys_and_values_with_predicate_arity_one():
    structure = [10, 20, 30]
    key_spec = lambda i: i == 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20)]
    assert result == expected

def test_get_keys_and_values_with_predicate_arity_two():
    structure = [10, 20, 30]
    key_spec = lambda i, v: v == 30
    result = _get_keys_and_values(structure, key_spec)
    expected = [(2, 30)]
    assert result == expected


# LLM-generated content at query #55
#--------------------------

def test_update_structure_discard_leaf():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=1, y=2))
    kvs = [('a', m(x=1, y=2))]
    result = _update_structure(structure, kvs, [], discard)
    expected = m()
    assert result == expected

def test_update_structure_discard_nested():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=1, y=2))
    kvs = [('a', m(x=1, y=2))]
    result = _update_structure(structure, kvs, ['x'], discard)
    expected = m(a=m(y=2))
    assert result == expected

def test_update_structure_discard_multiple_keys():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=1, y=2), b=m(z=3))
    kvs = [('a', m(x=1, y=2)), ('b', m(z=3))]
    result = _update_structure(structure, kvs, [], discard)
    expected = m()
    assert result == expected

def test_update_structure_discard_nonexistent_key():
    from pyrsistent import pmap, m, v
    structure = m(a=1)
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], discard)
    expected = m(a=1)
    assert result == expected

def test_update_structure_set_leaf():
    from pyrsistent import pmap, m, v
    structure = m(a=1)
    kvs = [('a', 1)]
    result = _update_structure(structure, kvs, [], 42)
    expected = m(a=42)
    assert result == expected

def test_update_structure_set_nested():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=1))
    kvs = [('a', m(x=1))]
    result = _update_structure(structure, kvs, ['x'], 99)
    expected = m(a=m(x=99))
    assert result == expected

def test_update_structure_set_new_empty_node():
    from pyrsistent import pmap, m, v
    structure = m()
    kvs = [('a', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, ['x'], 100)
    expected = m(a=m(x=100))
    assert result == expected

def test_update_structure_set_with_callable():
    from pyrsistent import pmap, m, v
    structure = m(a=5)
    kvs = [('a', 5)]
    def increment(x):
        return x + 1
    result = _update_structure(structure, kvs, [], increment)
    expected = m(a=6)
    assert result == expected

def test_update_structure_discard_from_vector():
    from pyrsistent import pmap, m, v
    structure = v(10, 20, 30)
    kvs = [(0, 10), (1, 20), (2, 30)]
    result = _update_structure(structure, kvs, [], discard)
    expected = v()
    assert result == expected

def test_update_structure_discard_specific_index_from_vector():
    from pyrsistent import pmap, m, v
    structure = v(10, 20, 30)
    kvs = [(1, 20)]
    result = _update_structure(structure, kvs, [], discard)
    expected = v(10, 30)
    assert result == expected

def test_update_structure_no_change_when_value_unchanged():
    from pyrsistent import pmap, m, v
    structure = m(a=1, b=2)
    kvs = [('a', 1)]
    result = _update_structure(structure, kvs, [], 1)
    assert result is structure

def test_update_structure_empty_path_and_command():
    from pyrsistent import pmap, m, v
    structure = m(a=1)
    kvs = [('a', 1)]
    result = _update_structure(structure, kvs, [], discard)
    expected = m()
    assert result == expected

def test_update_structure_with_sentinel_and_discard():
    from pyrsistent import pmap, m, v
    structure = m(a=1)
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], discard)
    expected = m(a=1)
    assert result == expected


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_get_arity_positional_only():
    def func(a, b, c): pass
    result = _get_arity(func)
    expected = 3
    assert result == expected

def test_get_arity_with_defaults():
    def func(a, b=2, c=3): pass
    result = _get_arity(func)
    expected = 1
    assert result == expected

def test_get_arity_keyword_only():
    def func(*, a, b, c): pass
    result = _get_arity(func)
    expected = 0
    assert result == expected

def test_get_arity_var_positional():
    def func(a, *args): pass
    result = _get_arity(func)
    expected = 1
    assert result == expected

def test_get_arity_mixed():
    def func(a, b=5, *args, c, d=10, **kwargs): pass
    result = _get_arity(func)
    expected = 1
    assert result == expected

def test_get_arity_no_parameters():
    def func(): pass
    result = _get_arity(func)
    expected = 0
    assert result == expected

def test_get_arity_positional_or_keyword():
    def func(a, b, c=30): pass
    result = _get_arity(func)
    expected = 2
    assert result == expected

def test_get_arity_builtin():
    result = _get_arity(len)
    expected = 1
    assert result == expected


# LLM-generated content at query #2
#--------------------------

def test_items_with_dict():
    structure = {'a': 1, 'b': 2}
    result = _items(structure)
    expected = [('a', 1), ('b', 2)]
    assert sorted(result) == sorted(expected)

def test_items_with_list():
    structure = ['x', 'y', 'z']
    result = _items(structure)
    expected = [(0, 'x'), (1, 'y'), (2, 'z')]
    assert result == expected

def test_items_with_tuple():
    structure = (10, 20, 30)
    result = _items(structure)
    expected = [(0, 10), (1, 20), (2, 30)]
    assert result == expected

def test_items_with_empty_dict():
    structure = {}
    result = _items(structure)
    expected = []
    assert result == expected

def test_items_with_empty_list():
    structure = []
    result = _items(structure)
    expected = []
    assert result == expected

def test_items_with_string():
    structure = 'ab'
    result = _items(structure)
    expected = [(0, 'a'), (1, 'b')]
    assert result == expected


# LLM-generated content at query #3
#--------------------------

def test__do_to_path_with_empty_path_and_callable_command():
    structure = [1, 2, 3]
    result = _do_to_path(structure, [], lambda x: sum(x))
    assert result == 6


def test__do_to_path_with_empty_path_and_non_callable_command():
    structure = {'a': 1}
    result = _do_to_path(structure, [], 'new_value')
    assert result == 'new_value'


def test__do_to_path_with_single_key_path_and_callable_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    result = _do_to_path(structure, ['a'], lambda x: x * 2)
    assert result == pmap({'a': 2, 'b': 2})


def test__do_to_path_with_single_key_path_and_non_callable_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    result = _do_to_path(structure, ['a'], 100)
    assert result == pmap({'a': 100})


def test__do_to_path_with_nested_path_and_callable_command():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': 5})})
    result = _do_to_path(structure, ['a', 'b'], lambda x: x + 1)
    assert result == pmap({'a': pmap({'b': 6})})


def test__do_to_path_with_callable_key_spec_unary():
    from pyrsistent import pmap
    structure = pmap({'x': 1, 'y': 2})
    result = _do_to_path(structure, [lambda k: k == 'x'], lambda v: v * 10)
    assert result == pmap({'x': 10, 'y': 2})


def test__do_to_path_with_callable_key_spec_binary():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    result = _do_to_path(structure, [lambda k, v: v > 1], lambda v: v * 2)
    assert result == pmap({'a': 1, 'b': 4})


def test__do_to_path_with_discard_command_on_single_key():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    result = _do_to_path(structure, ['a'], discard)
    assert result == pmap({'b': 2})


def test__do_to_path_with_discard_command_on_multiple_keys_via_callable():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2, 'c': 3})
    result = _do_to_path(structure, [lambda k, v: v % 2 == 0], discard)
    assert result == pmap({'a': 1, 'c': 3})


def test__do_to_path_with_nonexistent_key_and_non_callable_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    result = _do_to_path(structure, ['b'], 99)
    assert result == pmap({'a': 1, 'b': 99})


def test__do_to_path_with_nonexistent_key_and_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    result = _do_to_path(structure, ['b'], discard)
    assert result == pmap({'a': 1})


def test__do_to_path_with_list_structure_and_index_path():
    from pyrsistent import pvector
    structure = pvector([10, 20, 30])
    result = _do_to_path(structure, [1], lambda x: x * 2)
    assert result == pvector([10, 40, 30])


def test__do_to_path_with_list_structure_and_callable_index():
    from pyrsistent import pvector
    structure = pvector([5, 15, 25])
    result = _do_to_path(structure, [lambda i: i == 2], lambda x: x + 5)
    assert result == pvector([5, 15, 30])


def test__do_to_path_with_discard_on_list_structure():
    from pyrsistent import pvector
    structure = pvector([1, 2, 3, 4])
    result = _do_to_path(structure, [1], discard)
    assert result == pvector([1, 3, 4])


# LLM-generated content at query #4
#--------------------------

def test_items_without_attribute_error():
    structure = {"a": 1, "b": 2}
    result = _items(structure)
    assert result == [("a", 1), ("b", 2)]

def test_items_with_attribute_error():
    structure = [10, 20, 30]
    result = _items(structure)
    assert result == [(0, 10), (1, 20), (2, 30)]


# LLM-generated content at query #5
#--------------------------

def test_rex_returns_lambda():
    import re
    matcher = rex("^test.*")
    assert callable(matcher)
    assert matcher("test_string") is not None
    assert matcher("other_string") is None
    assert matcher(123) is None

def test_rex_matches_correct_pattern():
    import re
    matcher = rex("^a[0-9]+$")
    assert matcher("a123") is not None
    assert matcher("a") is None
    assert matcher("b123") is None

def test_rex_with_special_characters():
    import re
    matcher = rex("^\\d+\\.\\d+$")
    assert matcher("3.14") is not None
    assert matcher("abc") is None
    assert matcher("123") is None

def test_rex_empty_string():
    import re
    matcher = rex("^$")
    assert matcher("") is not None
    assert matcher("a") is None

def test_rex_case_sensitive():
    import re
    matcher = rex("^Hello$")
    assert matcher("Hello") is not None
    assert matcher("hello") is None
    assert matcher("HELLO") is None


# LLM-generated content at query #6
#--------------------------

def test_update_structure_discard_leaf():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=1, y=2))
    kvs = [('a', m(x=1, y=2))]
    result = _update_structure(structure, kvs, [], discard)
    expected = m()
    assert result == expected

def test_update_structure_discard_nested():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=1, y=2))
    kvs = [('a', m(x=1, y=2))]
    result = _update_structure(structure, kvs, ['x'], discard)
    expected = m(a=m(y=2))
    assert result == expected

def test_update_structure_discard_multiple_keys():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=1, y=2), b=m(z=3))
    kvs = [('a', m(x=1, y=2)), ('b', m(z=3))]
    result = _update_structure(structure, kvs, [], discard)
    expected = m()
    assert result == expected

def test_update_structure_discard_non_existing_key():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=1))
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], discard)
    expected = m(a=m(x=1))
    assert result == expected

def test_update_structure_set_leaf():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=1))
    kvs = [('a', m(x=1))]
    result = _update_structure(structure, kvs, [], lambda s: m(y=2))
    expected = m(a=m(y=2))
    assert result == expected

def test_update_structure_set_nested():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=m(y=1)))
    kvs = [('a', m(x=m(y=1)))]
    result = _update_structure(structure, kvs, ['x'], lambda s: m(z=2))
    expected = m(a=m(x=m(z=2)))
    assert result == expected

def test_update_structure_create_new_key():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=1))
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], lambda s: m(y=2))
    expected = m(a=m(x=1), b=m(y=2))
    assert result == expected

def test_update_structure_create_nested_new_key():
    from pyrsistent import pmap, m, v
    structure = m(a=m(x=1))
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, ['y'], lambda s: 2)
    expected = m(a=m(x=1), b=m(y=2))
    assert result == expected

def test_update_structure_with_vector():
    from pyrsistent import pmap, m, v
    structure = v(m(x=1), m(x=2))
    kvs = [(0, m(x=1)), (1, m(x=2))]
    result = _update_structure(structure, kvs, ['x'], lambda s: s * 2)
    expected = v(m(x=2), m(x=4))
    assert result == expected

def test_update_structure_discard_vector_reverse():
    from pyrsistent import pmap, m, v
    structure = v(1, 2, 3)
    kvs = [(0, 1), (1, 2), (2, 3)]
    result = _update_structure(structure, kvs, [], discard)
    expected = v()
    assert result == expected


# LLM-generated content at query #7
#--------------------------

def test_items_without_attribute_error():
    structure = [1, 2, 3]
    result = _items(structure)
    assert result == [(0, 1), (1, 2), (2, 3)]

def test_items_with_attribute_error():
    structure = {"a": 1, "b": 2}
    result = _items(structure)
    assert result == [("a", 1), ("b", 2)]


# LLM-generated content at query #8
#--------------------------

def test__get_keys_and_values_with_mapping_and_callable_unary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('a', 1), ('c', 3)]

def test__get_keys_and_values_with_mapping_and_callable_binary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('b', 2), ('c', 3)]

def test__get_keys_and_values_with_mapping_and_non_callable_key():
    structure = {'a': 1, 'b': 2}
    key_spec = 'a'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('a', 1)]

def test__get_keys_and_values_with_sequence_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k in [0, 2]
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(0, 10), (2, 30)]

def test__get_keys_and_values_with_sequence_and_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20), (2, 30)]

def test__get_keys_and_values_with_sequence_and_non_callable_key():
    structure = [10, 20, 30]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20)]

def test__get_keys_and_values_with_callable_arity_zero_raises_error():
    structure = {'a': 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_callable_arity_three_raises_error():
    structure = {'a': 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_non_existent_non_callable_key_returns_sentinel():
    structure = {'a': 1}
    key_spec = 'b'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('b', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_object_and_non_callable_key():
    class TestObject:
        x = 5
    structure = TestObject()
    key_spec = 'x'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('x', 5)]

def test__get_keys_and_values_with_object_and_non_existent_non_callable_key_returns_sentinel():
    class TestObject:
        x = 5
    structure = TestObject()
    key_spec = 'y'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('y', _EMPTY_SENTINEL)]


# LLM-generated content at query #9
#--------------------------

def test_get_arity_with_no_parameters():
    def f():
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_positional_only():
    def f(a, b, /):
        pass
    result = _get_arity(f)
    assert result == 2

def test_get_arity_with_positional_or_keyword():
    def f(a, b, c):
        pass
    result = _get_arity(f)
    assert result == 3

def test_get_arity_with_keyword_only():
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

def test_get_arity_with_default_values():
    def f(a, b=1, c=2):
        pass
    result = _get_arity(f)
    assert result == 1

def test_get_arity_mixed_parameters():
    def f(a, b, /, c, d=4, *, e, f=6):
        pass
    result = _get_arity(f)
    assert result == 3

def test_get_arity_on_builtin():
    result = _get_arity(print)
    assert result >= 0

def test_get_arity_on_lambda():
    f = lambda x, y: x + y
    result = _get_arity(f)
    assert result == 2


# LLM-generated content at query #10
#--------------------------

def test_predicate_at_line_1_evaluates_to_false():
    result = _get_keys_and_values({}, lambda: None)
    assert result == []


# LLM-generated content at query #11
#--------------------------

def test_predicate_at_line_1_evaluates_to_false():
    result = _get_keys_and_values({}, lambda: None)
    assert result == []


# LLM-generated content at query #12
#--------------------------

def test_predicate_arity_other_than_1_or_2_raises_value_error():
    def predicate_three_args(a, b, c):
        return True
    structure = {}
    try:
        _get_keys_and_values(structure, predicate_three_args)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #13
#--------------------------

def test_predicate_arity_other_than_1_or_2_raises_value_error():
    def predicate_three_args(a, b, c):
        return True
    structure = {}
    try:
        _get_keys_and_values(structure, predicate_three_args)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #14
#--------------------------

def test_get_keys_and_values_with_mapping_and_callable_unary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, key_spec)
    expected = [('a', 1), ('c', 3)]
    assert sorted(result) == sorted(expected)

def test_get_keys_and_values_with_mapping_and_callable_binary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', 2), ('c', 3)]
    assert sorted(result) == sorted(expected)

def test_get_keys_and_values_with_sequence_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 10), (2, 30)]
    assert sorted(result) == sorted(expected)

def test_get_keys_and_values_with_sequence_and_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20), (2, 30)]
    assert sorted(result) == sorted(expected)

def test_get_keys_and_values_with_mapping_and_non_callable_key():
    structure = {'x': 100, 'y': 200}
    key_spec = 'x'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('x', 100)]
    assert result == expected

def test_get_keys_and_values_with_sequence_and_non_callable_key():
    structure = [5, 6, 7]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 6)]
    assert result == expected

def test_get_keys_and_values_with_callable_arity_zero():
    structure = {'a': 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_get_keys_and_values_with_callable_arity_three():
    structure = {'a': 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_get_keys_and_values_with_non_existent_non_callable_key_in_mapping():
    structure = {'a': 1}
    key_spec = 'b'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', _EMPTY_SENTINEL)]
    assert result == expected

def test_get_keys_and_values_with_out_of_range_non_callable_key_in_sequence():
    structure = [1, 2]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    expected = [(5, _EMPTY_SENTINEL)]
    assert result == expected


# LLM-generated content at query #15
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature

    def func_with_default(param1, param2=10):
        pass

    def func_without_default(param1, param2):
        pass

    def func_with_keyword_only(*, param1):
        pass

    def func_with_var_positional(*args):
        pass

    params = list(signature(func_with_default).parameters.values())
    param_with_default = params[1]
    result = param_with_default.default is Parameter.empty
    assert result == False

    params = list(signature(func_without_default).parameters.values())
    param_without_default = params[0]
    result = param_without_default.default is Parameter.empty
    assert result == True

    params = list(signature(func_with_keyword_only).parameters.values())
    keyword_only_param = params[0]
    result = keyword_only_param.default is Parameter.empty
    assert result == True

    params = list(signature(func_with_var_positional).parameters.values())
    var_positional_param = params[0]
    result = var_positional_param.default is Parameter.empty
    assert result == True


# LLM-generated content at query #16
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    from pyrsistent import pmap
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._precord import _update_structure
    from pyrsistent._precord import discard
    structure = pmap({"a": 1})
    kvs = [("a", 1)]
    path = ["key"]
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({"a": 1})


# LLM-generated content at query #17
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature
    def dummy_func(a, b=1, *, c, d=2):
        pass
    params = signature(dummy_func).parameters.values()
    param_c = next(p for p in params if p.name == 'c')
    result = param_c.default is Parameter.empty and param_c.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result == False


# LLM-generated content at query #18
#--------------------------

def test_items_with_dict():
    structure = {'a': 1, 'b': 2}
    result = _items(structure)
    expected = [('a', 1), ('b', 2)]
    assert sorted(result) == sorted(expected)

def test_items_with_list():
    structure = ['x', 'y', 'z']
    result = _items(structure)
    expected = [(0, 'x'), (1, 'y'), (2, 'z')]
    assert list(result) == expected

def test_items_with_tuple():
    structure = (10, 20, 30)
    result = _items(structure)
    expected = [(0, 10), (1, 20), (2, 30)]
    assert list(result) == expected

def test_items_with_empty_dict():
    structure = {}
    result = _items(structure)
    expected = []
    assert list(result) == expected

def test_items_with_empty_list():
    structure = []
    result = _items(structure)
    expected = []
    assert list(result) == expected

def test_items_with_string():
    structure = 'ab'
    result = _items(structure)
    expected = [(0, 'a'), (1, 'b')]
    assert list(result) == expected


# LLM-generated content at query #19
#--------------------------

def test_predicate_arity_other_than_one_or_two_raises_error():
    def predicate_with_zero_args():
        return True
    structure = {"a": 1}
    try:
        _get_keys_and_values(structure, predicate_with_zero_args)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_three_args_raises_error():
    def predicate_with_three_args(a, b, c):
        return True
    structure = {"a": 1}
    try:
        _get_keys_and_values(structure, predicate_with_three_args)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #20
#--------------------------

def test_unary_predicate():
    key_spec = lambda k: k > 0
    structure = {1: 'a', -2: 'b', 3: 'c'}
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 'a'), (3, 'c')]
    assert result == expected

def test_binary_predicate():
    key_spec = lambda k, v: k > 0 and v.startswith('a')
    structure = {1: 'apple', -2: 'banana', 3: 'apricot'}
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 'apple'), (3, 'apricot')]
    assert result == expected

def test_non_callable_key():
    key_spec = 'key1'
    structure = {'key1': 'value1', 'key2': 'value2'}
    result = _get_keys_and_values(structure, key_spec)
    expected = [('key1', 'value1')]
    assert result == expected

def test_predicate_with_zero_arity():
    key_spec = lambda: True
    structure = {1: 'a', 2: 'b'}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_three_arity():
    key_spec = lambda a, b, c: True
    structure = {1: 'a', 2: 'b'}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #21
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature
    def dummy_func(a, b=1, *, c, d=2, **kwargs):
        pass
    params = signature(dummy_func).parameters.values()
    param_a = next(p for p in params if p.name == 'a')
    param_b = next(p for p in params if p.name == 'b')
    param_c = next(p for p in params if p.name == 'c')
    param_d = next(p for p in params if p.name == 'd')
    assert param_a.default is Parameter.empty
    assert param_b.default is not Parameter.empty
    assert param_c.default is Parameter.empty
    assert param_d.default is not Parameter.empty
    assert param_a.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert param_b.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert param_c.kind not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert param_d.kind not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    predicate_result = param_b.default is Parameter.empty and param_b.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert predicate_result == False


# LLM-generated content at query #22
#--------------------------

def test_get_arity_with_no_parameters():
    def f():
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_positional_only_parameters():
    def f(a, b, c):
        pass
    result = _get_arity(f)
    assert result == 3

def test_get_arity_with_positional_or_keyword_parameters():
    def f(a, b, c):
        pass
    result = _get_arity(f)
    assert result == 3

def test_get_arity_with_keyword_only_parameters():
    def f(*, a, b, c):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_var_positional_parameter():
    def f(*args):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_var_keyword_parameter():
    def f(**kwargs):
        pass
    result = _get_arity(f)
    assert result == 0

def test_get_arity_with_default_parameters():
    def f(a, b=2, c=3):
        pass
    result = _get_arity(f)
    assert result == 1

def test_get_arity_with_mixed_parameters():
    def f(a, b, c=3, d=4, *args, e, f=6, **kwargs):
        pass
    result = _get_arity(f)
    assert result == 2

def test_get_arity_with_class_method():
    class C:
        def method(self, a, b):
            pass
    c = C()
    result = _get_arity(c.method)
    assert result == 2

def test_get_arity_with_static_method():
    class C:
        @staticmethod
        def method(a, b):
            pass
    result = _get_arity(C.method)
    assert result == 2

def test_get_arity_with_lambda():
    f = lambda x, y: x + y
    result = _get_arity(f)
    assert result == 2

def test_get_arity_with_builtin_function():
    result = _get_arity(len)
    assert result == 1


# LLM-generated content at query #23
#--------------------------

def test_unary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("a", 1), ("c", 3)]

def test_binary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", 2), ("c", 3)]

def test_non_callable_key_spec():
    structure = {"a": 1, "b": 2}
    key_spec = "a"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("a", 1)]

def test_predicate_with_zero_arity():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_three_arity():
    structure = {"a": 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_unary_predicate_with_sequence():
    structure = [10, 20, 30]
    key_spec = lambda i: i % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(0, 10), (2, 30)]

def test_binary_predicate_with_sequence():
    structure = [10, 20, 30]
    key_spec = lambda i, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20), (2, 30)]

def test_predicate_filters_all():
    structure = {"a": 1, "b": 2}
    key_spec = lambda k: False
    result = _get_keys_and_values(structure, key_spec)
    assert result == []

def test_predicate_keeps_all():
    structure = {"a": 1, "b": 2}
    key_spec = lambda k: True
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("a", 1), ("b", 2)]


# LLM-generated content at query #24
#--------------------------

def test_callable_key_spec_with_arity_1():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k == "b"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2)]
    assert result == expected

def test_callable_key_spec_with_arity_2():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v == 2
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2)]
    assert result == expected

def test_callable_key_spec_with_arity_0_raises_error():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_callable_key_spec_with_arity_3_raises_error():
    structure = {"a": 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #25
#--------------------------

def test_callable_with_arity_1():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k == "b"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", 2)]

def test_callable_with_arity_2():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", 2), ("c", 3)]

def test_callable_with_arity_0_raises_error():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_callable_with_arity_3_raises_error():
    structure = {"a": 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_non_callable_key_spec():
    structure = {"a": 1, "b": 2}
    key_spec = "a"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("a", 1)]


# LLM-generated content at query #26
#--------------------------

def test_unary_predicate():
    key_spec = lambda k: k == "a"
    structure = {"a": 1, "b": 2}
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected

def test_binary_predicate():
    key_spec = lambda k, v: v > 1
    structure = {"a": 1, "b": 2, "c": 3}
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2), ("c", 3)]
    assert result == expected

def test_non_callable_key_spec():
    key_spec = "a"
    structure = {"a": 1, "b": 2}
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected

def test_predicate_with_arity_zero():
    key_spec = lambda: True
    structure = {"a": 1}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_arity_three():
    key_spec = lambda a, b, c: True
    structure = {"a": 1}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_unary_predicate_with_sequence():
    key_spec = lambda i: i % 2 == 0
    structure = [10, 20, 30, 40]
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 10), (2, 30)]
    assert result == expected

def test_binary_predicate_with_sequence():
    key_spec = lambda i, v: v > 25
    structure = [10, 20, 30, 40]
    result = _get_keys_and_values(structure, key_spec)
    expected = [(2, 30), (3, 40)]
    assert result == expected


# LLM-generated content at query #27
#--------------------------

def test_rex_returns_function():
    import re
    result = rex("test.*")
    assert callable(result)

def test_rex_function_matches_correct_string():
    import re
    matcher = rex("^abc")
    assert matcher("abc123") is not None
    assert matcher("xyz") is None

def test_rex_function_returns_none_for_non_string():
    import re
    matcher = rex(".*")
    assert matcher(123) is None
    assert matcher(["list"]) is None

def test_rex_function_uses_fullmatch_behavior():
    import re
    matcher = rex("hello")
    assert matcher("hello") is not None
    assert matcher("hello world") is None

def test_rex_function_with_special_regex_chars():
    import re
    matcher = rex(r"\d+")
    assert matcher("123") is not None
    assert matcher("abc") is None

def test_rex_function_case_sensitive():
    import re
    matcher = rex("Test")
    assert matcher("Test") is not None
    assert matcher("test") is None

def test_rex_function_with_empty_string():
    import re
    matcher = rex("")
    assert matcher("") is not None
    assert matcher("a") is None


# LLM-generated content at query #28
#--------------------------

def test_rex_returns_lambda_for_string_matching():
    import re
    predicate = rex("^test.*")
    result = predicate("test_string")
    assert result == True


# LLM-generated content at query #29
#--------------------------

def test_get_keys_and_values_with_callable_arity_0():
    def zero_arity():
        return True
    structure = {"a": 1}
    try:
        _get_keys_and_values(structure, zero_arity)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_get_keys_and_values_with_callable_arity_3():
    def three_arity(x, y, z):
        return True
    structure = {"a": 1}
    try:
        _get_keys_and_values(structure, three_arity)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #30
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    from pyrsistent._pmap import pmap
    from pyrsistent import m
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._field_common import discard
    structure = m()
    kvs = [("key", "value")]
    path = ["some_path"]
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result is not None


# LLM-generated content at query #31
#--------------------------

def test_rex_returns_true_for_matching_string():
    predicate = rex("^test.*")
    result = predicate("test_string")
    assert result is True

def test_rex_returns_false_for_non_matching_string():
    predicate = rex("^test.*")
    result = predicate("no_match")
    assert result is False

def test_rex_returns_false_for_non_string_key():
    predicate = rex("^test.*")
    result = predicate(123)
    assert result is False

def test_rex_matches_exact_string():
    predicate = rex("^exact$")
    result = predicate("exact")
    assert result is True

def test_rex_does_not_match_partial_exact_string():
    predicate = rex("^exact$")
    result = predicate("exact_extra")
    assert result is False

def test_rex_matches_empty_string():
    predicate = rex("^$")
    result = predicate("")
    assert result is True

def test_rex_matches_with_dot_wildcard():
    predicate = rex("^t.st$")
    result = predicate("test")
    assert result is True

def test_rex_does_not_match_with_dot_wildcard_wrong_length():
    predicate = rex("^t.st$")
    result = predicate("tesst")
    assert result is False

def test_rex_matches_character_class():
    predicate = rex("^[0-9]+$")
    result = predicate("123")
    assert result is True

def test_rex_does_not_match_character_class():
    predicate = rex("^[0-9]+$")
    result = predicate("abc")
    assert result is False


# LLM-generated content at query #32
#--------------------------

def test__get_keys_and_values_with_mapping_and_callable_unary():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("a", 1), ("c", 3)]

def test__get_keys_and_values_with_mapping_and_callable_binary():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", 2), ("c", 3)]

def test__get_keys_and_values_with_sequence_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k == 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20)]

def test__get_keys_and_values_with_sequence_and_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v == 20
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20)]

def test__get_keys_and_values_with_mapping_and_non_callable_key():
    structure = {"x": 100, "y": 200}
    key_spec = "x"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("x", 100)]

def test__get_keys_and_values_with_sequence_and_non_callable_index():
    structure = [5, 6, 7]
    key_spec = 2
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(2, 7)]

def test__get_keys_and_values_with_callable_zero_arity():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_callable_three_arity():
    structure = {"a": 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_non_existent_key():
    structure = {"a": 1}
    key_spec = "b"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_non_existent_index():
    structure = [10, 20]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(5, _EMPTY_SENTINEL)]


# LLM-generated content at query #33
#--------------------------

def test_unary_predicate():
    key_spec = lambda k: k % 2 == 0
    structure = {0: 'a', 1: 'b', 2: 'c'}
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 'a'), (2, 'c')]
    assert result == expected

def test_binary_predicate():
    key_spec = lambda k, v: k > 0 and v.startswith('b')
    structure = {0: 'a', 1: 'b', 2: 'bcd'}
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 'b'), (2, 'bcd')]
    assert result == expected

def test_non_callable_key():
    key_spec = 'x'
    structure = {'x': 42, 'y': 100}
    result = _get_keys_and_values(structure, key_spec)
    expected = [('x', 42)]
    assert result == expected

def test_predicate_with_arity_error():
    key_spec = lambda: True
    structure = {0: 'a'}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_arity_three():
    key_spec = lambda a, b, c: True
    structure = {0: 'a'}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_empty_structure_with_unary_predicate():
    key_spec = lambda k: True
    structure = {}
    result = _get_keys_and_values(structure, key_spec)
    expected = []
    assert result == expected

def test_empty_structure_with_binary_predicate():
    key_spec = lambda k, v: True
    structure = {}
    result = _get_keys_and_values(structure, key_spec)
    expected = []
    assert result == expected


# LLM-generated content at query #34
#--------------------------

def test_unary_predicate_returns_filtered_keys_and_values():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1), ("c", 3)]
    assert result == expected

def test_binary_predicate_returns_filtered_keys_and_values():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2), ("c", 3)]
    assert result == expected

def test_non_callable_key_spec_returns_single_key_value_pair():
    structure = {"a": 1, "b": 2}
    key_spec = "a"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected

def test_unary_predicate_with_sequence_structure():
    structure = [10, 20, 30]
    key_spec = lambda i: i % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 10), (2, 30)]
    assert result == expected

def test_binary_predicate_with_sequence_structure():
    structure = [10, 20, 30]
    key_spec = lambda i, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20), (2, 30)]
    assert result == expected

def test_unary_predicate_with_empty_structure():
    structure = {}
    key_spec = lambda k: True
    result = _get_keys_and_values(structure, key_spec)
    expected = []
    assert result == expected

def test_binary_predicate_with_empty_structure():
    structure = {}
    key_spec = lambda k, v: True
    result = _get_keys_and_values(structure, key_spec)
    expected = []
    assert result == expected

def test_non_callable_key_spec_with_missing_key_returns_sentinel():
    structure = {"a": 1}
    key_spec = "b"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", _EMPTY_SENTINEL)]
    assert result == expected


# LLM-generated content at query #35
#--------------------------

def test__get_keys_and_values_with_dict_and_callable_unary():
    d = {'a': 1, 'b': 2, 'c': 3}
    result = _get_keys_and_values(d, lambda k: k in ['a', 'c'])
    assert sorted(result) == [('a', 1), ('c', 3)]

def test__get_keys_and_values_with_dict_and_callable_binary():
    d = {'a': 1, 'b': 2, 'c': 3}
    result = _get_keys_and_values(d, lambda k, v: v > 1)
    assert sorted(result) == [('b', 2), ('c', 3)]

def test__get_keys_and_values_with_list_and_callable_unary():
    lst = [10, 20, 30]
    result = _get_keys_and_values(lst, lambda k: k == 1)
    assert result == [(1, 20)]

def test__get_keys_and_values_with_list_and_callable_binary():
    lst = [10, 20, 30]
    result = _get_keys_and_values(lst, lambda k, v: v == 30)
    assert result == [(2, 30)]

def test__get_keys_and_values_with_dict_and_non_callable_key():
    d = {'x': 100, 'y': 200}
    result = _get_keys_and_values(d, 'x')
    assert result == [('x', 100)]

def test__get_keys_and_values_with_list_and_non_callable_key():
    lst = [5, 6, 7]
    result = _get_keys_and_values(lst, 2)
    assert result == [(2, 7)]

def test__get_keys_and_values_with_dict_and_non_existent_key():
    d = {'a': 1}
    result = _get_keys_and_values(d, 'b')
    assert result == [('b', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_list_and_out_of_range_index():
    lst = [10, 20]
    result = _get_keys_and_values(lst, 5)
    assert result == [(5, _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_callable_arity_zero():
    d = {'a': 1}
    try:
        _get_keys_and_values(d, lambda: True)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_callable_arity_three():
    d = {'a': 1}
    try:
        _get_keys_and_values(d, lambda x, y, z: True)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_empty_dict_and_callable():
    d = {}
    result = _get_keys_and_values(d, lambda k: True)
    assert result == []

def test__get_keys_and_values_with_empty_list_and_callable():
    lst = []
    result = _get_keys_and_values(lst, lambda k, v: True)
    assert result == []


# LLM-generated content at query #36
#--------------------------

def test_update_structure_discard_with_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1), ('b', 2)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == pmap({})
def test_update_structure_discard_specific_key():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == pmap({'b': 2})
def test_update_structure_discard_non_existent_key():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == pmap({'a': 1})
def test_update_structure_update_leaf_with_callable():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 2})
def test_update_structure_update_leaf_with_value():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    command = 100
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 100})
def test_update_structure_nested_update():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': 1})})
    kvs = [('a', pmap({'b': 1}))]
    path = ['b']
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': pmap({'b': 2})})
def test_update_structure_create_new_nested_node():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({})})
    kvs = [('a', pmap({}))]
    path = ['b']
    command = 5
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': pmap({'b': 5})})
def test_update_structure_discard_nested_key():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': 1, 'c': 2})})
    kvs = [('a', pmap({'b': 1, 'c': 2}))]
    path = ['b']
    result = _update_structure(structure, kvs, path, discard)
    assert result == pmap({'a': pmap({'c': 2})})
def test_update_structure_with_empty_sentinel_and_command():
    from pyrsistent import pmap
    structure = pmap({})
    kvs = [('a', _EMPTY_SENTINEL)]
    command = 10
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 10})
def test_update_structure_with_empty_sentinel_and_discard():
    from pyrsistent import pmap
    structure = pmap({})
    kvs = [('a', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == pmap({})
def test_update_structure_multiple_kvs_update():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1), ('b', 2)]
    command = lambda x: x * 2
    result = _update_structure(structure, kvs, [], command)
    assert result == pmap({'a': 2, 'b': 4})
def test_update_structure_multiple_kvs_discard_reversed():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1), ('b', 2)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == pmap({})
def test_update_structure_no_change_when_result_equals_value():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    command = lambda x: x
    result = _update_structure(structure, kvs, [], command)
    assert result is structure


# LLM-generated content at query #37
#--------------------------

def test_predicate_at_line_4_evaluates_to_true():
    from pyrsistent._pmap import pmap
    from pyrsistent import pmap
    from pyrsistent._pvector import pvector
    from pyrsistent._pset import pset
    from pyrsistent._pbag import pbag
    from pyrsistent._pdeque import pdeque
    from pyrsistent._pclass import PClass
    from pyrsistent._checked_types import CheckedPMap, CheckedPVector, CheckedPSet
    from pyrsistent._field_common import field
    from pyrsistent._precord import PRecord
    from pyrsistent._plist import plist
    from pyrsistent._pdict import pdict
    from pyrsistent._helpers import freeze, thaw, m, v, s, b, d, q, l, pset, pvector, pmap, pbag, pdeque, plist, pdict
    from pyrsistent._transformations import discard, inc, dec, add, subtract, multiply, divide, transform, _EMPTY_SENTINEL, _do_to_path, _update_structure
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({})


# LLM-generated content at query #38
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    from pyrsistent._pmap import pmap
    from pyrsistent import m
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._field_common import discard
    structure = m()
    kvs = []
    path = []
    command = None
    result = _update_structure(structure, kvs, path, command)
    assert result is not None


# LLM-generated content at query #39
#--------------------------

def test_update_structure_discard_leaf():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': 1})})
    kvs = [('a', pmap({'b': 1}))]
    path = ['b']
    result = _update_structure(structure, kvs, path, discard)
    expected = pmap({'a': pmap({})})
    assert result == expected

def test_update_structure_discard_nonexistent():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({})})
    kvs = [('a', pmap({}))]
    path = ['b']
    result = _update_structure(structure, kvs, path, discard)
    expected = pmap({'a': pmap({})})
    assert result == expected

def test_update_structure_discard_multiple_keys():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': 1, 'c': 2})})
    kvs = [('a', pmap({'b': 1, 'c': 2}))]
    path = ['b']
    result = _update_structure(structure, kvs, path, discard)
    expected = pmap({'a': pmap({'c': 2})})
    assert result == expected

def test_update_structure_update_leaf():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': 1})})
    kvs = [('a', pmap({'b': 1}))]
    path = ['b']
    command = lambda x: 2
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'b': 2})})
    assert result == expected

def test_update_structure_update_empty_sentinel():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({})})
    kvs = [('a', pmap({}))]
    path = ['b']
    command = lambda x: 1
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'b': 1})})
    assert result == expected

def test_update_structure_discard_empty_sentinel():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({})})
    kvs = [('a', pmap({}))]
    path = ['b']
    result = _update_structure(structure, kvs, path, discard)
    expected = pmap({'a': pmap({})})
    assert result == expected

def test_update_structure_multiple_kvs():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': 1}), 'c': pmap({'b': 2})})
    kvs = [('a', pmap({'b': 1})), ('c', pmap({'b': 2}))]
    path = ['b']
    command = lambda x: x * 2
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'b': 2}), 'c': pmap({'b': 4})})
    assert result == expected

def test_update_structure_no_path_no_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = 2
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 2})
    assert result == expected

def test_update_structure_no_path_callable_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 2})
    assert result == expected

def test_update_structure_discard_reverse_order():
    from pyrsistent import pvector
    structure = pvector([pvector([1, 2]), pvector([3, 4])])
    kvs = [(0, pvector([1, 2])), (1, pvector([3, 4]))]
    path = [0]
    result = _update_structure(structure, kvs, path, discard)
    expected = pvector([pvector([2]), pvector([4])])
    assert result == expected


# LLM-generated content at query #40
#--------------------------

def test_update_structure_with_discard_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'b': 2})
    assert result == expected

def test_update_structure_with_discard_command_and_non_existent_key():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('c', _EMPTY_SENTINEL)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert result == structure

def test_update_structure_with_callable_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 2})
    assert result == expected

def test_update_structure_with_non_callable_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = 100
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 100})
    assert result == expected

def test_update_structure_with_nested_path_and_command():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': 1})})
    kvs = [('a', pmap({'b': 1}))]
    path = ['b']
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'b': 2})})
    assert result == expected

def test_update_structure_with_sentinel_value_and_discard():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert result == structure

def test_update_structure_with_sentinel_value_and_non_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = []
    command = 5
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 1, 'b': 5})
    assert result == expected

def test_update_structure_with_sentinel_value_and_nested_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = ['c']
    command = 10
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 1, 'b': pmap({'c': 10})})
    assert result == expected

def test_update_structure_with_multiple_kvs_and_discard_reversed():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2, 'c': 3})
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'c': 3})
    assert result == expected

def test_update_structure_with_evolver_persistent():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result == structure


# LLM-generated content at query #41
#--------------------------

def test__get_keys_and_values_with_dict_and_callable_unary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, key_spec)
    expected = [('a', 1), ('c', 3)]
    assert sorted(result) == sorted(expected)

def test__get_keys_and_values_with_dict_and_callable_binary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', 2), ('c', 3)]
    assert sorted(result) == sorted(expected)

def test__get_keys_and_values_with_list_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 10), (2, 30)]
    assert sorted(result) == sorted(expected)

def test__get_keys_and_values_with_list_and_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20), (2, 30)]
    assert sorted(result) == sorted(expected)

def test__get_keys_and_values_with_dict_and_non_callable_key():
    structure = {'x': 100, 'y': 200}
    key_spec = 'x'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('x', 100)]
    assert result == expected

def test__get_keys_and_values_with_list_and_non_callable_key():
    structure = [5, 6, 7]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 6)]
    assert result == expected

def test__get_keys_and_values_with_callable_arity_zero_raises_error():
    structure = {'a': 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_callable_arity_three_raises_error():
    structure = {'a': 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_non_existent_non_callable_key_returns_empty_sentinel():
    structure = {'a': 1}
    key_spec = 'b'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', _EMPTY_SENTINEL)]
    assert result == expected

def test__get_keys_and_values_with_empty_structure_and_callable():
    structure = {}
    key_spec = lambda k: True
    result = _get_keys_and_values(structure, key_spec)
    assert result == []

def test__get_keys_and_values_with_empty_structure_and_non_callable():
    structure = {}
    key_spec = 'missing'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('missing', _EMPTY_SENTINEL)]
    assert result == expected


# LLM-generated content at query #42
#--------------------------

def test_unary_predicate():
    key_spec = lambda k: k == "a"
    structure = {"a": 1, "b": 2}
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected

def test_binary_predicate():
    key_spec = lambda k, v: v > 1
    structure = {"a": 1, "b": 2}
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2)]
    assert result == expected

def test_non_callable_key_spec():
    key_spec = "a"
    structure = {"a": 1, "b": 2}
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected


# LLM-generated content at query #43
#--------------------------

def test_items_with_dict():
    structure = {'a': 1, 'b': 2}
    result = _items(structure)
    expected = [('a', 1), ('b', 2)]
    assert sorted(result) == sorted(expected)

def test_items_with_list():
    structure = ['x', 'y', 'z']
    result = _items(structure)
    expected = [(0, 'x'), (1, 'y'), (2, 'z')]
    assert list(result) == expected

def test_items_with_tuple():
    structure = (10, 20, 30)
    result = _items(structure)
    expected = [(0, 10), (1, 20), (2, 30)]
    assert list(result) == expected

def test_items_with_empty_dict():
    structure = {}
    result = _items(structure)
    assert list(result) == []

def test_items_with_empty_list():
    structure = []
    result = _items(structure)
    assert list(result) == []

def test_items_with_single_element():
    structure = ['only']
    result = _items(structure)
    assert list(result) == [(0, 'only')]


# LLM-generated content at query #44
#--------------------------

def test_get_keys_and_values_with_callable_unary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, key_spec)
    expected = [('a', 1), ('c', 3)]
    assert result == expected

def test_get_keys_and_values_with_callable_binary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', 2), ('c', 3)]
    assert result == expected

def test_get_keys_and_values_with_callable_arity_error():
    structure = {'a': 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_get_keys_and_values_with_non_callable_key():
    structure = {'a': 1, 'b': 2}
    key_spec = 'a'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('a', 1)]
    assert result == expected

def test_get_keys_and_values_with_non_callable_key_missing():
    structure = {'a': 1, 'b': 2}
    key_spec = 'c'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('c', _EMPTY_SENTINEL)]
    assert result == expected

def test_get_keys_and_values_with_sequence_structure():
    structure = [10, 20, 30]
    key_spec = lambda k: k == 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20)]
    assert result == expected

def test_get_keys_and_values_with_sequence_structure_non_callable():
    structure = [10, 20, 30]
    key_spec = 2
    result = _get_keys_and_values(structure, key_spec)
    expected = [(2, 30)]
    assert result == expected

def test_get_keys_and_values_with_sequence_structure_non_callable_missing():
    structure = [10, 20, 30]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    expected = [(5, _EMPTY_SENTINEL)]
    assert result == expected

def test_get_keys_and_values_with_object_structure():
    class TestObject:
        def __init__(self):
            self.x = 100
            self.y = 200
    structure = TestObject()
    key_spec = 'x'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('x', 100)]
    assert result == expected

def test_get_keys_and_values_with_object_structure_missing():
    class TestObject:
        def __init__(self):
            self.x = 100
    structure = TestObject()
    key_spec = 'z'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('z', _EMPTY_SENTINEL)]
    assert result == expected


