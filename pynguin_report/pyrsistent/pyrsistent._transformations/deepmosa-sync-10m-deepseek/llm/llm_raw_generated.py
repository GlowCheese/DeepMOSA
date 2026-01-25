####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
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
    def f(a, b=2, /, c=3, *, d):
        pass
    result = _get_arity(f)
    assert result == 1


# LLM-generated content at query #3
#--------------------------

def test__get_keys_and_values_with_dict_and_callable_unary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('a', 1), ('c', 3)]

def test__get_keys_and_values_with_dict_and_callable_binary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('b', 2), ('c', 3)]

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
    structure = {'x': 100, 'y': 200}
    key_spec = 'x'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('x', 100)]

def test__get_keys_and_values_with_list_and_non_callable_key():
    structure = [5, 6, 7]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 6)]

def test__get_keys_and_values_with_dict_and_callable_arity_zero():
    structure = {'a': 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_dict_and_callable_arity_three():
    structure = {'a': 1}
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

def test__get_keys_and_values_with_non_existent_key():
    structure = {'a': 1}
    key_spec = 'b'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('b', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_non_existent_index():
    structure = [1, 2]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(5, _EMPTY_SENTINEL)]


# LLM-generated content at query #4
#--------------------------

def test__get_keys_and_values_with_dict_and_callable_unary():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k == "b"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", 2)]

def test__get_keys_and_values_with_dict_and_callable_binary():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", 2), ("c", 3)]

def test__get_keys_and_values_with_list_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k == 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20)]

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
    key_spec = 2
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(2, 7)]

def test__get_keys_and_values_with_callable_arity_zero_raises_error():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_callable_arity_three_raises_error():
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
    key_spec = lambda k: True
    result = _get_keys_and_values(structure, key_spec)
    assert result == []

def test__get_keys_and_values_with_non_existent_non_callable_key():
    structure = {"a": 1}
    key_spec = "b"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", _EMPTY_SENTINEL)]


# LLM-generated content at query #5
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


# LLM-generated content at query #6
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature
    def dummy_func(a, b=1, *, c, d=2):
        pass
    params = signature(dummy_func).parameters.values()
    param_c = next(p for p in params if p.name == 'c')
    result = param_c.default is Parameter.empty and param_c.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result is False


# LLM-generated content at query #7
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
    assert result_a is True
    assert result_b is False
    assert result_c is False
    assert result_d is False


# LLM-generated content at query #8
#--------------------------

def test__get_keys_and_values_with_dict_and_callable_unary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('a', 1), ('c', 3)]

def test__get_keys_and_values_with_dict_and_callable_binary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('b', 2), ('c', 3)]

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
    structure = {'x': 100, 'y': 200}
    key_spec = 'x'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('x', 100)]

def test__get_keys_and_values_with_list_and_non_callable_key():
    structure = [5, 6, 7]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 6)]

def test__get_keys_and_values_with_dict_and_callable_zero_arity():
    structure = {'a': 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_dict_and_callable_three_arity():
    structure = {'a': 1}
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

def test__get_keys_and_values_with_non_existent_key():
    structure = {'a': 1}
    key_spec = 'b'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('b', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_non_existent_index():
    structure = [1, 2]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(5, _EMPTY_SENTINEL)]


# LLM-generated content at query #9
#--------------------------

def test_items_with_dict():
    result = _items({'a': 1, 'b': 2})
    assert sorted(list(result)) == [('a', 1), ('b', 2)]

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

def test_items_with_single_element():
    result = _items([99])
    assert list(result) == [(0, 99)]

def test_items_with_string():
    result = _items('ab')
    assert list(result) == [(0, 'a'), (1, 'b')]


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

def test_predicate_at_line_5_evaluates_to_false():
    from inspect import signature, Parameter
    def dummy_func(a, b=1, *, c, d=2):
        pass
    params = signature(dummy_func).parameters.values()
    param_c = next(p for p in params if p.name == 'c')
    result = param_c.default is Parameter.empty and param_c.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result is False


# LLM-generated content at query #12
#--------------------------

def test_items_without_attribute_error_returns_items():
    structure = {"a": 1, "b": 2}
    result = _items(structure)
    expected = [("a", 1), ("b", 2)]
    assert list(result) == expected

def test_items_with_attribute_error_returns_enumerated_list():
    structure = [10, 20, 30]
    result = _items(structure)
    expected = [(0, 10), (1, 20), (2, 30)]
    assert list(result) == expected

def test_items_with_empty_list_returns_empty_enumerated_list():
    structure = []
    result = _items(structure)
    expected = []
    assert list(result) == expected

def test_items_with_tuple_returns_enumerated_tuple():
    structure = (5, 15, 25)
    result = _items(structure)
    expected = [(0, 5), (1, 15), (2, 25)]
    assert list(result) == expected

def test_items_with_string_returns_enumerated_characters():
    structure = "abc"
    result = _items(structure)
    expected = [(0, "a"), (1, "b"), (2, "c")]
    assert list(result) == expected


# LLM-generated content at query #13
#--------------------------

def test_unary_predicate_returns_matching_keys_and_values():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1), ("c", 3)]
    assert result == expected

def test_binary_predicate_returns_matching_keys_and_values():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2)]
    assert result == expected

def test_non_callable_key_spec_returns_single_key_value_pair():
    structure = {"a": 1, "b": 2}
    key_spec = "a"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected


# LLM-generated content at query #14
#--------------------------

def test_update_structure_discard_leaf():
    from pyrsistent import pmap, m
    structure = m(a=m(x=1, y=2))
    kvs = [('a', m(x=1, y=2))]
    result = _update_structure(structure, kvs, [], discard)
    assert result == m()

def test_update_structure_discard_nested():
    from pyrsistent import pmap, m
    structure = m(a=m(x=1, y=2))
    kvs = [('a', m(x=1, y=2))]
    result = _update_structure(structure, kvs, ['x'], discard)
    assert result == m(a=m(y=2))

def test_update_structure_discard_multiple_keys():
    from pyrsistent import pmap, m
    structure = m(a=m(x=1, y=2), b=m(z=3))
    kvs = [('a', m(x=1, y=2)), ('b', m(z=3))]
    result = _update_structure(structure, kvs, [], discard)
    assert result == m()

def test_update_structure_discard_non_existing_key():
    from pyrsistent import pmap, m
    structure = m(a=1)
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == m(a=1)

def test_update_structure_set_new_leaf():
    from pyrsistent import pmap, m
    structure = m()
    kvs = [('a', _EMPTY_SENTINEL)]
    command = lambda x: 10
    result = _update_structure(structure, kvs, [], command)
    assert result == m(a=10)

def test_update_structure_set_nested_new():
    from pyrsistent import pmap, m
    structure = m()
    kvs = [('a', _EMPTY_SENTINEL)]
    command = lambda x: m(b=5)
    result = _update_structure(structure, kvs, [], command)
    assert result == m(a=m(b=5))

def test_update_structure_update_existing_leaf():
    from pyrsistent import pmap, m
    structure = m(a=1)
    kvs = [('a', 1)]
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, [], command)
    assert result == m(a=2)

def test_update_structure_update_nested_existing():
    from pyrsistent import pmap, m
    structure = m(a=m(b=2))
    kvs = [('a', m(b=2))]
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, ['b'], command)
    assert result == m(a=m(b=3))

def test_update_structure_with_empty_pmap_leaf():
    from pyrsistent import pmap, m
    structure = m()
    kvs = [('a', _EMPTY_SENTINEL)]
    command = lambda x: pmap()
    result = _update_structure(structure, kvs, [], command)
    assert result == m(a=pmap())

def test_update_structure_no_change():
    from pyrsistent import pmap, m
    structure = m(a=1, b=2)
    kvs = [('a', 1)]
    command = lambda x: x
    result = _update_structure(structure, kvs, [], command)
    assert result == m(a=1, b=2)

def test_update_structure_discard_reverse_order():
    from pyrsistent import pvector, v
    structure = v(10, 20, 30)
    kvs = [(0, 10), (1, 20), (2, 30)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == v()

def test_update_structure_with_callable_command():
    from pyrsistent import pmap, m
    structure = m(a=5)
    kvs = [('a', 5)]
    command = str
    result = _update_structure(structure, kvs, [], command)
    assert result == m(a='5')


# LLM-generated content at query #15
#--------------------------

def test_predicate_at_line_5_evaluates_to_false():
    from inspect import signature, Parameter

    def func_with_default_param(a, b=1):
        pass

    param_values = list(signature(func_with_default_param).parameters.values())
    param_b = param_values[1]
    result = param_b.default is Parameter.empty and param_b.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result is False


# LLM-generated content at query #16
#--------------------------

def test_predicate_arity_other_than_one_or_two_raises_value_error():
    def predicate_with_zero_args():
        return True
    structure = {}
    try:
        _get_keys_and_values(structure, predicate_with_zero_args)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_three_args_raises_value_error():
    def predicate_with_three_args(a, b, c):
        return True
    structure = {}
    try:
        _get_keys_and_values(structure, predicate_with_three_args)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #17
#--------------------------

def test_get_keys_and_values_with_mapping_and_callable_unary():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k in ["a", "c"]
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1), ("c", 3)]
    assert sorted(result) == sorted(expected)

def test_get_keys_and_values_with_mapping_and_callable_binary():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2), ("c", 3)]
    assert sorted(result) == sorted(expected)

def test_get_keys_and_values_with_sequence_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k == 0 or k == 2
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 10), (2, 30)]
    assert sorted(result) == sorted(expected)

def test_get_keys_and_values_with_sequence_and_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v == 20
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20)]
    assert result == expected

def test_get_keys_and_values_with_callable_arity_zero():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_get_keys_and_values_with_callable_arity_three():
    structure = {"a": 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_get_keys_and_values_with_mapping_and_non_callable_key():
    structure = {"a": 1, "b": 2}
    key_spec = "a"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("a", 1)]
    assert result == expected

def test_get_keys_and_values_with_sequence_and_non_callable_key():
    structure = [10, 20, 30]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20)]
    assert result == expected

def test_get_keys_and_values_with_non_callable_key_missing():
    structure = {"a": 1}
    key_spec = "b"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", _EMPTY_SENTINEL)]
    assert result == expected

def test_get_keys_and_values_with_sequence_and_non_callable_key_out_of_range():
    structure = [10, 20]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    expected = [(5, _EMPTY_SENTINEL)]
    assert result == expected

def test_get_keys_and_values_with_object_and_non_callable_key():
    class TestObj:
        x = 42
    structure = TestObj()
    key_spec = "x"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("x", 42)]
    assert result == expected

def test_get_keys_and_values_with_object_and_non_callable_key_missing():
    class TestObj:
        pass
    structure = TestObj()
    key_spec = "y"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("y", _EMPTY_SENTINEL)]
    assert result == expected


# LLM-generated content at query #18
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature

    def func_with_default(param1, param2=10):
        pass

    def func_without_default(param1, param2):
        pass

    def func_with_keyword_only(*, kwarg):
        pass

    def func_with_var_positional(*args):
        pass

    def func_with_var_keyword(**kwargs):
        pass

    params_with_default = list(signature(func_with_default).parameters.values())
    param_with_default = params_with_default[1]
    result_default = param_with_default.default is Parameter.empty
    assert result_default == False

    params_without_default = list(signature(func_without_default).parameters.values())
    param_without_default = params_without_default[0]
    result_no_default = param_without_default.default is Parameter.empty
    assert result_no_default == True

    params_kw_only = list(signature(func_with_keyword_only).parameters.values())
    param_kw_only = params_kw_only[0]
    result_kw_only = param_kw_only.default is Parameter.empty
    assert result_kw_only == True

    params_var_pos = list(signature(func_with_var_positional).parameters.values())
    param_var_pos = params_var_pos[0]
    result_var_pos = param_var_pos.default is Parameter.empty
    assert result_var_pos == True

    params_var_kw = list(signature(func_with_var_keyword).parameters.values())
    param_var_kw = params_var_kw[0]
    result_var_kw = param_var_kw.default is Parameter.empty
    assert result_var_kw == True


# LLM-generated content at query #19
#--------------------------

def test__do_to_path_empty_path_with_callable():
    result = _do_to_path([1, 2, 3], [], lambda x: sum(x))
    assert result == 6

def test__do_to_path_empty_path_with_non_callable():
    result = _do_to_path([1, 2, 3], [], [4, 5, 6])
    assert result == [4, 5, 6]

def test__do_to_path_single_key_spec():
    structure = {'a': 1, 'b': 2}
    path = ['a']
    command = lambda x: x * 2
    result = _do_to_path(structure, path, command)
    assert result == {'a': 2, 'b': 2}

def test__do_to_path_with_callable_key_spec_unary():
    structure = {0: 'a', 1: 'b', 2: 'c'}
    path = [lambda k: k % 2 == 0]
    command = lambda x: x.upper()
    result = _do_to_path(structure, path, command)
    assert result == {0: 'A', 1: 'b', 2: 'C'}

def test__do_to_path_with_callable_key_spec_binary():
    structure = {'x': 10, 'y': 20, 'z': 30}
    path = [lambda k, v: v > 15]
    command = lambda x: x * 2
    result = _do_to_path(structure, path, command)
    assert result == {'x': 10, 'y': 40, 'z': 60}

def test__do_to_path_with_discard_command():
    structure = {'a': 1, 'b': 2, 'c': 3}
    path = ['b']
    command = discard
    result = _do_to_path(structure, path, command)
    assert result == {'a': 1, 'c': 3}

def test__do_to_path_with_nested_path():
    structure = {'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}}
    path = ['a', 'y']
    command = lambda x: x * 10
    result = _do_to_path(structure, path, command)
    assert result == {'a': {'x': 1, 'y': 20}, 'b': {'x': 3, 'y': 4}}

def test__do_to_path_with_missing_key_and_discard():
    structure = {'a': 1, 'b': 2}
    path = ['c']
    command = discard
    result = _do_to_path(structure, path, command)
    assert result == {'a': 1, 'b': 2}

def test__do_to_path_with_missing_key_and_non_discard():
    from pyrsistent._pmap import pmap
    structure = {'a': 1, 'b': 2}
    path = ['c']
    command = lambda x: 99
    result = _do_to_path(structure, path, command)
    assert result == {'a': 1, 'b': 2, 'c': 99}

def test__do_to_path_with_sequence_structure():
    structure = [10, 20, 30]
    path = [1]
    command = lambda x: x + 5
    result = _do_to_path(structure, path, command)
    assert result == [10, 25, 30]

def test__do_to_path_with_callable_key_spec_on_sequence():
    structure = [5, 10, 15, 20]
    path = [lambda k: k > 1]
    command = lambda x: x * 2
    result = _do_to_path(structure, path, command)
    assert result == [5, 10, 30, 40]


# LLM-generated content at query #20
#--------------------------

def test_rex_matches_string():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123") is not None
    assert matcher("test_") is None
    assert matcher("123") is None
    assert matcher("test_abc") is None

def test_rex_non_string_input():
    matcher = rex(r"^test$")
    assert matcher(123) is False
    assert matcher(None) is False
    assert matcher([]) is False
    assert matcher({"key": "value"}) is False

def test_rex_partial_match():
    matcher = rex(r"^test_\d+$")
    assert matcher("test_123_extra") is None
    assert matcher("prefix_test_123") is None

def test_rex_empty_pattern():
    matcher = rex(r"")
    assert matcher("") is not None
    assert matcher("any_string") is not None

def test_rex_special_characters():
    matcher = rex(r"^a\.b$")
    assert matcher("a.b") is not None
    assert matcher("aXb") is None

def test_rex_case_sensitive():
    matcher = rex(r"^Test$")
    assert matcher("Test") is not None
    assert matcher("test") is None


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

def test_predicate_at_line_6_evaluates_to_false():
    from inspect import Parameter, signature

    def func_with_default(param1, param2=1):
        pass

    def func_without_default(param1, param2):
        pass

    def func_with_keyword_only(*, param1):
        pass

    def func_with_var_positional(*args):
        pass

    parameters = signature(func_with_default).parameters.values()
    param_with_default = next(p for p in parameters if p.name == 'param2')
    result_with_default = param_with_default.default is Parameter.empty
    assert result_with_default == False

    parameters = signature(func_without_default).parameters.values()
    param_without_default = next(p for p in parameters if p.name == 'param1')
    result_without_default = param_without_default.default is Parameter.empty
    assert result_without_default == True

    parameters = signature(func_with_keyword_only).parameters.values()
    param_keyword_only = next(p for p in parameters if p.name == 'param1')
    result_keyword_only = param_keyword_only.default is Parameter.empty
    assert result_keyword_only == True

    parameters = signature(func_with_var_positional).parameters.values()
    param_var_positional = next(p for p in parameters if p.name == 'args')
    result_var_positional = param_var_positional.default is Parameter.empty
    assert result_var_positional == True


# LLM-generated content at query #23
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

def test_get_arity_with_lambda():
    f = lambda x, y: x + y
    result = _get_arity(f)
    assert result == 2

def test_get_arity_with_builtin_function():
    result = _get_arity(len)
    assert result == 1


# LLM-generated content at query #24
#--------------------------

def test_predicate_with_arity_other_than_1_or_2():
    def predicate_with_zero_args():
        return True
    structure = {}
    try:
        _get_keys_and_values(structure, predicate_with_zero_args)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_arity_three():
    def predicate_with_three_args(a, b, c):
        return True
    structure = {}
    try:
        _get_keys_and_values(structure, predicate_with_three_args)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #25
#--------------------------

def test_unary_predicate():
    key_spec = lambda k: k % 2 == 0
    structure = {0: 'a', 1: 'b', 2: 'c'}
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 'a'), (2, 'c')]
    assert result == expected

def test_binary_predicate():
    key_spec = lambda k, v: v == 'b'
    structure = {0: 'a', 1: 'b', 2: 'c'}
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 'b')]
    assert result == expected

def test_non_callable_key():
    key_spec = 1
    structure = {0: 'a', 1: 'b', 2: 'c'}
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 'b')]
    assert result == expected

def test_predicate_with_arity_zero():
    key_spec = lambda: True
    structure = {0: 'a', 1: 'b'}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_arity_three():
    key_spec = lambda a, b, c: True
    structure = {0: 'a', 1: 'b'}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_unary_predicate_with_sequence():
    key_spec = lambda i: i > 0
    structure = ['a', 'b', 'c']
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 'b'), (2, 'c')]
    assert result == expected

def test_binary_predicate_with_sequence():
    key_spec = lambda i, v: v == 'c'
    structure = ['a', 'b', 'c']
    result = _get_keys_and_values(structure, key_spec)
    expected = [(2, 'c')]
    assert result == expected


# LLM-generated content at query #26
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


# LLM-generated content at query #27
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


# LLM-generated content at query #28
#--------------------------

def test__get_keys_and_values_with_mapping_and_string_key():
    structure = {'a': 1, 'b': 2}
    result = _get_keys_and_values(structure, 'a')
    assert result == [('a', 1)]

def test__get_keys_and_values_with_mapping_and_non_existent_key():
    structure = {'a': 1, 'b': 2}
    result = _get_keys_and_values(structure, 'c')
    assert result == [('c', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_sequence_and_integer_key():
    structure = [10, 20, 30]
    result = _get_keys_and_values(structure, 1)
    assert result == [(1, 20)]

def test__get_keys_and_values_with_sequence_and_out_of_range_key():
    structure = [10, 20, 30]
    result = _get_keys_and_values(structure, 5)
    assert result == [(5, _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_object_and_attribute_key():
    class TestObject:
        x = 100
    structure = TestObject()
    result = _get_keys_and_values(structure, 'x')
    assert result == [('x', 100)]

def test__get_keys_and_values_with_object_and_non_existent_attribute_key():
    class TestObject:
        x = 100
    structure = TestObject()
    result = _get_keys_and_values(structure, 'y')
    assert result == [('y', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_callable_unary_predicate_on_mapping():
    structure = {'a': 1, 'b': 2, 'c': 3}
    predicate = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, predicate)
    assert sorted(result) == [('a', 1), ('c', 3)]

def test__get_keys_and_values_with_callable_unary_predicate_on_sequence():
    structure = [10, 20, 30]
    predicate = lambda k: k % 2 == 0
    result = _get_keys_and_values(structure, predicate)
    assert result == [(0, 10), (2, 30)]

def test__get_keys_and_values_with_callable_binary_predicate_on_mapping():
    structure = {'a': 1, 'b': 2, 'c': 3}
    predicate = lambda k, v: v > 1
    result = _get_keys_and_values(structure, predicate)
    assert sorted(result) == [('b', 2), ('c', 3)]

def test__get_keys_and_values_with_callable_binary_predicate_on_sequence():
    structure = [10, 20, 30]
    predicate = lambda k, v: v > 15
    result = _get_keys_and_values(structure, predicate)
    assert result == [(1, 20), (2, 30)]

def test__get_keys_and_values_with_callable_zero_arity_raises_error():
    structure = {'a': 1}
    predicate = lambda: True
    try:
        _get_keys_and_values(structure, predicate)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_callable_three_arity_raises_error():
    structure = {'a': 1}
    predicate = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, predicate)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #29
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
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_missing_non_callable_key_in_dict():
    structure = {'a': 1}
    key_spec = 'b'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', _EMPTY_SENTINEL)]
    assert result == expected

def test__get_keys_and_values_with_missing_non_callable_key_in_list():
    structure = [1, 2]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    expected = [(5, _EMPTY_SENTINEL)]
    assert result == expected

def test__get_keys_and_values_with_object_and_non_callable_key():
    class TestObj:
        attr = 42
    structure = TestObj()
    key_spec = 'attr'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('attr', 42)]
    assert result == expected

def test__get_keys_and_values_with_object_and_missing_non_callable_key():
    class TestObj:
        pass
    structure = TestObj()
    key_spec = 'missing'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('missing', _EMPTY_SENTINEL)]
    assert result == expected


# LLM-generated content at query #30
#--------------------------

def test__get_keys_and_values_with_mapping_and_callable_unary():
    d = {'a': 1, 'b': 2, 'c': 3}
    result = _get_keys_and_values(d, lambda k: k in ['a', 'c'])
    assert result == [('a', 1), ('c', 3)]

def test__get_keys_and_values_with_mapping_and_callable_binary():
    d = {'a': 1, 'b': 2, 'c': 3}
    result = _get_keys_and_values(d, lambda k, v: v > 1)
    assert result == [('b', 2), ('c', 3)]

def test__get_keys_and_values_with_mapping_and_non_callable_key():
    d = {'a': 1, 'b': 2}
    result = _get_keys_and_values(d, 'a')
    assert result == [('a', 1)]

def test__get_keys_and_values_with_sequence_and_callable_unary():
    lst = [10, 20, 30]
    result = _get_keys_and_values(lst, lambda idx: idx % 2 == 0)
    assert result == [(0, 10), (2, 30)]

def test__get_keys_and_values_with_sequence_and_callable_binary():
    lst = [10, 20, 30]
    result = _get_keys_and_values(lst, lambda idx, val: val > 15)
    assert result == [(1, 20), (2, 30)]

def test__get_keys_and_values_with_sequence_and_non_callable_key():
    lst = [10, 20, 30]
    result = _get_keys_and_values(lst, 1)
    assert result == [(1, 20)]

def test__get_keys_and_values_with_object_and_callable_unary():
    class Obj:
        x = 5
        y = 10
    o = Obj()
    result = _get_keys_and_values(o, lambda attr: attr == 'x')
    assert result == [('x', 5)]

def test__get_keys_and_values_with_object_and_callable_binary():
    class Obj:
        x = 5
        y = 10
    o = Obj()
    result = _get_keys_and_values(o, lambda attr, val: val == 10)
    assert result == [('y', 10)]

def test__get_keys_and_values_with_object_and_non_callable_key():
    class Obj:
        x = 5
    o = Obj()
    result = _get_keys_and_values(o, 'x')
    assert result == [('x', 5)]

def test__get_keys_and_values_with_callable_arity_zero_raises_error():
    d = {'a': 1}
    try:
        _get_keys_and_values(d, lambda: True)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_callable_arity_three_raises_error():
    d = {'a': 1}
    try:
        _get_keys_and_values(d, lambda x, y, z: True)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_non_existent_key_returns_sentinel():
    d = {'a': 1}
    result = _get_keys_and_values(d, 'b')
    assert result == [('b', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_sequence_index_out_of_range_returns_sentinel():
    lst = [10, 20]
    result = _get_keys_and_values(lst, 5)
    assert result == [(5, _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_object_missing_attribute_returns_sentinel():
    class Obj:
        x = 5
    o = Obj()
    result = _get_keys_and_values(o, 'y')
    assert result == [('y', _EMPTY_SENTINEL)]


# LLM-generated content at query #31
#--------------------------

def test__get_keys_and_values_with_dict_and_callable_unary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, key_spec)
    assert sorted(result) == [('a', 1), ('c', 3)]

def test__get_keys_and_values_with_dict_and_callable_binary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert sorted(result) == [('b', 2), ('c', 3)]

def test__get_keys_and_values_with_dict_and_non_callable_key():
    structure = {'a': 1, 'b': 2}
    key_spec = 'a'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('a', 1)]

def test__get_keys_and_values_with_list_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda i: i % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(0, 10), (2, 30)]

def test__get_keys_and_values_with_list_and_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda i, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20), (2, 30)]

def test__get_keys_and_values_with_list_and_non_callable_key():
    structure = [10, 20, 30]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20)]

def test__get_keys_and_values_with_callable_arity_zero_raises():
    structure = {'a': 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_callable_arity_three_raises():
    structure = {'a': 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_missing_non_callable_key_returns_sentinel():
    structure = {'a': 1}
    key_spec = 'b'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('b', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_object_and_callable_unary():
    class TestObject:
        def __init__(self):
            self.x = 5
            self.y = 10
    structure = TestObject()
    key_spec = lambda k: k == 'x'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('x', 5)]

def test__get_keys_and_values_with_object_and_callable_binary():
    class TestObject:
        def __init__(self):
            self.x = 5
            self.y = 10
    structure = TestObject()
    key_spec = lambda k, v: v == 10
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('y', 10)]

def test__get_keys_and_values_with_object_and_non_callable_key():
    class TestObject:
        def __init__(self):
            self.x = 5
    structure = TestObject()
    key_spec = 'x'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('x', 5)]

def test__get_keys_and_values_with_object_missing_non_callable_key_returns_sentinel():
    class TestObject:
        pass
    structure = TestObject()
    key_spec = 'z'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('z', _EMPTY_SENTINEL)]


# LLM-generated content at query #32
#--------------------------

def test_callable_key_spec_with_arity_1():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k == "b"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2)]
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


# LLM-generated content at query #33
#--------------------------

def test_update_structure_discard_with_empty_path():
    from pyrsistent import pmap, m
    structure = m(a=1, b=2)
    kvs = [('a', 1)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == m(b=2)

def test_update_structure_discard_with_nested_path():
    from pyrsistent import pmap, m
    structure = m(a=m(b=1, c=2))
    kvs = [('a', m(b=1, c=2))]
    result = _update_structure(structure, kvs, ['b'], discard)
    assert result == m(a=m(c=2))

def test_update_structure_discard_multiple_keys_reversed():
    from pyrsistent import pmap, m, v
    structure = v(10, 20, 30)
    kvs = [(0, 10), (2, 30)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == v(20)

def test_update_structure_discard_key_not_present():
    from pyrsistent import pmap, m
    structure = m(a=1)
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == m(a=1)

def test_update_structure_update_with_new_value():
    from pyrsistent import pmap, m
    structure = m(a=1)
    kvs = [('a', 1)]
    result = _update_structure(structure, kvs, [], lambda x: 100)
    assert result == m(a=100)

def test_update_structure_update_with_nested_path():
    from pyrsistent import pmap, m
    structure = m(a=m(b=1))
    kvs = [('a', m(b=1))]
    result = _update_structure(structure, kvs, ['b'], lambda x: 99)
    assert result == m(a=m(b=99))

def test_update_structure_expand_with_empty_sentinel():
    from pyrsistent import pmap, m
    structure = m()
    kvs = [('new', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], lambda x: 42)
    assert result == m(new=42)

def test_update_structure_expand_nested_with_empty_sentinel():
    from pyrsistent import pmap, m
    structure = m()
    kvs = [('a', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, ['b'], lambda x: 5)
    assert result == m(a=m(b=5))

def test_update_structure_no_change_when_result_equals_value():
    from pyrsistent import pmap, m
    structure = m(a=1)
    kvs = [('a', 1)]
    result = _update_structure(structure, kvs, [], lambda x: x)
    assert result is structure

def test_update_structure_with_multiple_kvs():
    from pyrsistent import pmap, m
    structure = m(a=1, b=2)
    kvs = [('a', 1), ('b', 2)]
    result = _update_structure(structure, kvs, [], lambda x: x * 10)
    assert result == m(a=10, b=20)


# LLM-generated content at query #34
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

def test_get_keys_and_values_with_invalid_arity_predicate():
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
    expected = []
    assert result == expected

def test_get_keys_and_values_with_empty_structure_and_binary_predicate():
    structure = {}
    key_spec = lambda k, v: True
    result = _get_keys_and_values(structure, key_spec)
    expected = []
    assert result == expected


# LLM-generated content at query #35
#--------------------------

def test_unary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k: k == "b"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", 2)]

def test_binary_predicate():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = lambda k, v: v == 2
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", 2)]

def test_predicate_with_arity_one():
    structure = [10, 20, 30]
    key_spec = lambda i: i == 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20)]

def test_predicate_with_arity_two():
    structure = [10, 20, 30]
    key_spec = lambda i, v: v == 30
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(2, 30)]

def test_predicate_filters_multiple_items():
    structure = {"x": 5, "y": 10, "z": 5}
    key_spec = lambda k, v: v == 5
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("x", 5), ("z", 5)]

def test_predicate_filters_none():
    structure = {"a": 1, "b": 2}
    key_spec = lambda k: k == "c"
    result = _get_keys_and_values(structure, key_spec)
    assert result == []


# LLM-generated content at query #36
#--------------------------

def test_update_structure_discard_leaf():
    from pyrsistent import pmap, m
    structure = m(a=m(x=1, y=2))
    kvs = [('a', m(x=1, y=2))]
    result = _update_structure(structure, kvs, [], discard)
    expected = m()
    assert result == expected

def test_update_structure_discard_nested():
    from pyrsistent import pmap, m
    structure = m(a=m(b=m(c=1)))
    kvs = [('a', m(b=m(c=1)))]
    result = _update_structure(structure, kvs, ['b'], discard)
    expected = m(a=m())
    assert result == expected

def test_update_structure_discard_multiple_keys():
    from pyrsistent import pmap, m
    structure = m(a=1, b=2, c=3)
    kvs = [('a', 1), ('b', 2)]
    result = _update_structure(structure, kvs, [], discard)
    expected = m(c=3)
    assert result == expected

def test_update_structure_discard_non_existent_key():
    from pyrsistent import pmap, m
    structure = m(a=1)
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], discard)
    expected = m(a=1)
    assert result == expected

def test_update_structure_update_leaf():
    from pyrsistent import pmap, m
    structure = m(a=1)
    kvs = [('a', 1)]
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, [], command)
    expected = m(a=2)
    assert result == expected

def test_update_structure_update_nested():
    from pyrsistent import pmap, m
    structure = m(a=m(b=1))
    kvs = [('a', m(b=1))]
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, ['b'], command)
    expected = m(a=m(b=2))
    assert result == expected

def test_update_structure_insert_new_empty_pmap():
    from pyrsistent import pmap, m
    structure = m()
    kvs = [('a', _EMPTY_SENTINEL)]
    command = lambda x: m(b=1)
    result = _update_structure(structure, kvs, [], command)
    expected = m(a=m(b=1))
    assert result == expected

def test_update_structure_no_change():
    from pyrsistent import pmap, m
    structure = m(a=1)
    kvs = [('a', 1)]
    command = lambda x: x
    result = _update_structure(structure, kvs, [], command)
    assert result == structure

def test_update_structure_with_callable_command():
    from pyrsistent import pmap, m
    structure = m(a=m(b=1, c=2))
    kvs = [('a', m(b=1, c=2))]
    command = lambda x: m(b=x['b'] + 10, c=x['c'] + 20)
    result = _update_structure(structure, kvs, [], command)
    expected = m(a=m(b=11, c=22))
    assert result == expected

def test_update_structure_discard_reverse_order():
    from pyrsistent import pvector, v
    structure = v(10, 20, 30)
    kvs = [(0, 10), (2, 30)]
    result = _update_structure(structure, kvs, [], discard)
    expected = v(20)
    assert result == expected


# LLM-generated content at query #37
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

def test_update_structure_with_discard_command_and_nested_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'x': 10, 'y': 20})})
    kvs = [('a', pmap({'x': 10, 'y': 20}))]
    path = ['x']
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'y': 20})})
    assert result == expected

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

def test_update_structure_with_empty_sentinel_value_and_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 1})
    assert result == expected

def test_update_structure_with_empty_sentinel_value_and_non_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = []
    command = lambda x: 5
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 1, 'b': 5})
    assert result == expected

def test_update_structure_with_multiple_kvs_and_discard_command_reversed():
    from pyrsistent import pmap
    structure = pmap({0: 'a', 1: 'b', 2: 'c'})
    kvs = [(0, 'a'), (1, 'b'), (2, 'c')]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({})
    assert result == expected

def test_update_structure_with_nested_structure_and_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': pmap({'c': 1})})})
    kvs = [('a', pmap({'b': pmap({'c': 1})}))]
    path = ['b', 'c']
    command = lambda x: x + 1
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'b': pmap({'c': 2})})})
    assert result == expected

def test_update_structure_with_result_equal_to_original_value():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result == structure

def test_update_structure_with_empty_kvs_list():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = []
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert result == structure


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
    structure = {'a': 1, 'b': 2}
    result = _do_to_path(structure, ['a'], lambda x: x * 2)
    assert result == {'a': 2, 'b': 2}

def test__do_to_path_with_single_key_path_and_non_callable_command():
    structure = {'a': 1, 'b': 2}
    result = _do_to_path(structure, ['a'], 100)
    assert result == {'a': 100, 'b': 2}

def test__do_to_path_with_callable_key_spec_and_arity_1():
    structure = {0: 'a', 1: 'b', 2: 'c'}
    result = _do_to_path(structure, [lambda k: k % 2 == 0], lambda x: x.upper())
    assert result == {0: 'A', 1: 'b', 2: 'C'}

def test__do_to_path_with_callable_key_spec_and_arity_2():
    structure = {'a': 1, 'b': 2, 'c': 3}
    result = _do_to_path(structure, [lambda k, v: v > 1], lambda x: x * 10)
    assert result == {'a': 1, 'b': 20, 'c': 30}

def test__do_to_path_with_multiple_path_segments():
    structure = {'a': {'x': 1, 'y': 2}, 'b': {'x': 3, 'y': 4}}
    result = _do_to_path(structure, ['a', 'x'], lambda v: v + 100)
    assert result == {'a': {'x': 101, 'y': 2}, 'b': {'x': 3, 'y': 4}}

def test__do_to_path_with_discard_command_on_missing_key():
    structure = {'a': 1, 'b': 2}
    result = _do_to_path(structure, ['c'], discard)
    assert result == {'a': 1, 'b': 2}

def test__do_to_path_with_discard_command_on_existing_key():
    structure = {'a': 1, 'b': 2}
    result = _do_to_path(structure, ['a'], discard)
    assert result == {'b': 2}

def test__do_to_path_with_discard_command_on_list_structure():
    structure = [10, 20, 30]
    result = _do_to_path(structure, [1], discard)
    assert result == [10, 30]

def test__do_to_path_with_callable_key_spec_and_discard():
    structure = {0: 'a', 1: 'b', 2: 'c'}
    result = _do_to_path(structure, [lambda k: k == 1], discard)
    assert result == {0: 'a', 2: 'c'}

def test__do_to_path_with_empty_sentinel_and_pmap_creation():
    structure = {}
    result = _do_to_path(structure, ['new_key', 'nested_key'], 'value')
    from pyrsistent._pmap import pmap
    expected = {'new_key': {'nested_key': 'value'}}
    assert result == expected

def test__do_to_path_with_non_dict_structure_and_index_key():
    structure = [100, 200, 300]
    result = _do_to_path(structure, [0], lambda x: x / 10)
    assert result == [10, 200, 300]


# LLM-generated content at query #2
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
    key_spec = lambda k: k in [0, 2]
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

def test_get_keys_and_values_with_missing_key_in_mapping():
    structure = {'a': 1}
    key_spec = 'b'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', _EMPTY_SENTINEL)]
    assert result == expected

def test_get_keys_and_values_with_missing_index_in_sequence():
    structure = [10]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    expected = [(5, _EMPTY_SENTINEL)]
    assert result == expected

def test_get_keys_and_values_with_object_and_attribute():
    class Obj:
        attr = 42
    structure = Obj()
    key_spec = 'attr'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('attr', 42)]
    assert result == expected

def test_get_keys_and_values_with_object_and_missing_attribute():
    class Obj:
        pass
    structure = Obj()
    key_spec = 'missing'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('missing', _EMPTY_SENTINEL)]
    assert result == expected

def test_get_keys_and_values_callable_with_zero_arity_raises_error():
    structure = {}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_get_keys_and_values_callable_with_three_arity_raises_error():
    structure = {}
    key_spec = lambda a, b, c: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #3
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

def test_get_arity_with_mixed_parameters_and_defaults():
    def f(a, b=2, c=3):
        pass
    result = _get_arity(f)
    assert result == 1

def test_get_arity_with_positional_only_parameters():
    def f(a, b, /, c):
        pass
    result = _get_arity(f)
    assert result == 2

def test_get_arity_with_all_parameter_types():
    def f(a, b=2, /, c, d=4, *args, e, f=6, **kwargs):
        pass
    result = _get_arity(f)
    assert result == 2


# LLM-generated content at query #4
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


# LLM-generated content at query #5
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

def test_get_arity_with_positional_and_keyword_parameters():
    def f(a, b, c=3):
        pass
    result = _get_arity(f)
    assert result == 2

def test_get_arity_with_keyword_only_parameters():
    def f(*, a, b):
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

def test_get_arity_with_positional_only_parameters():
    def f(a, b, /, c):
        pass
    result = _get_arity(f)
    assert result == 3

def test_get_arity_with_mixed_parameter_kinds():
    def f(a, b=2, *args, c, d=4, **kwargs):
        pass
    result = _get_arity(f)
    assert result == 1


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

def test_items_with_single_element():
    result = _items([99])
    assert list(result) == [(0, 99)]


# LLM-generated content at query #7
#--------------------------

def test__get_keys_and_values_with_dict_and_callable_unary():
    d = {'a': 1, 'b': 2, 'c': 3}
    result = _get_keys_and_values(d, lambda k: k in ['a', 'c'])
    assert result == [('a', 1), ('c', 3)]

def test__get_keys_and_values_with_dict_and_callable_binary():
    d = {'a': 1, 'b': 2, 'c': 3}
    result = _get_keys_and_values(d, lambda k, v: v > 1)
    assert result == [('b', 2), ('c', 3)]

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

def test__get_keys_and_values_with_callable_arity_zero():
    d = {'a': 1}
    try:
        _get_keys_and_values(d, lambda: True)
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_callable_arity_three():
    d = {'a': 1}
    try:
        _get_keys_and_values(d, lambda x, y, z: True)
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_missing_key_in_dict():
    d = {'a': 1}
    result = _get_keys_and_values(d, 'b')
    assert result == [('b', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_out_of_range_index_in_list():
    lst = [10, 20]
    result = _get_keys_and_values(lst, 5)
    assert result == [(5, _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_empty_structure_and_callable():
    d = {}
    result = _get_keys_and_values(d, lambda k: True)
    assert result == []

def test__get_keys_and_values_with_object_having_items():
    class CustomDict:
        def items(self):
            return [('key1', 'val1'), ('key2', 'val2')]
    obj = CustomDict()
    result = _get_keys_and_values(obj, lambda k, v: '2' in k)
    assert result == [('key2', 'val2')]

def test__get_keys_and_values_with_object_having_getitem():
    class CustomList:
        def __init__(self, data):
            self.data = data
        def __getitem__(self, idx):
            return self.data[idx]
        def __len__(self):
            return len(self.data)
    obj = CustomList([100, 200, 300])
    result = _get_keys_and_values(obj, 1)
    assert result == [(1, 200)]


# LLM-generated content at query #8
#--------------------------

def test_callable_with_arity_0():
    def zero_arity():
        return True
    try:
        _get_keys_and_values({}, zero_arity)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_callable_with_arity_3():
    def three_arity(a, b, c):
        return True
    try:
        _get_keys_and_values({}, three_arity)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #9
#--------------------------

def test__get_keys_and_values_with_mapping_and_callable_unary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, key_spec)
    assert sorted(result) == [('a', 1), ('c', 3)]

def test__get_keys_and_values_with_mapping_and_callable_binary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert sorted(result) == [('b', 2), ('c', 3)]

def test__get_keys_and_values_with_sequence_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(0, 10), (2, 30)]

def test__get_keys_and_values_with_sequence_and_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20), (2, 30)]

def test__get_keys_and_values_with_mapping_and_non_callable_key():
    structure = {'x': 100, 'y': 200}
    key_spec = 'x'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('x', 100)]

def test__get_keys_and_values_with_sequence_and_non_callable_key():
    structure = ['first', 'second', 'third']
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 'second')]

def test__get_keys_and_values_with_mapping_and_missing_non_callable_key():
    structure = {'a': 1}
    key_spec = 'missing'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('missing', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_sequence_and_out_of_range_non_callable_key():
    structure = [5, 6]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(5, _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_callable_zero_arity_raises_error():
    structure = {'a': 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_callable_three_arity_raises_error():
    structure = {'a': 1}
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #10
#--------------------------

def test_callable_with_arity_0_raises_value_error():
    def zero_arity():
        return True
    structure = {"a": 1}
    try:
        _get_keys_and_values(structure, zero_arity)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #11
#--------------------------

def test_predicate_at_line_1_evaluates_to_false():
    from module import _get_keys_and_values
    structure = {"a": 1, "b": 2}
    key_spec = lambda k: False
    result = _get_keys_and_values(structure, key_spec)
    expected = []
    assert result == expected


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_5_evaluates_to_false():
    from inspect import signature, Parameter

    def func_with_default_param(a, b=1):
        pass

    def func_with_keyword_only_param(*, a):
        pass

    def func_with_var_positional_param(*args):
        pass

    def func_with_var_keyword_param(**kwargs):
        pass

    def func_with_positional_only_param(a, /, b):
        pass

    params = list(signature(func_with_default_param).parameters.values())
    param_b = params[1]
    result = param_b.default is Parameter.empty and param_b.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result == False

    params = list(signature(func_with_keyword_only_param).parameters.values())
    param_a = params[0]
    result = param_a.default is Parameter.empty and param_a.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result == False

    params = list(signature(func_with_var_positional_param).parameters.values())
    param_args = params[0]
    result = param_args.default is Parameter.empty and param_args.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result == False

    params = list(signature(func_with_var_keyword_param).parameters.values())
    param_kwargs = params[0]
    result = param_kwargs.default is Parameter.empty and param_kwargs.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result == False

    params = list(signature(func_with_positional_only_param).parameters.values())
    param_a = params[0]
    result = param_a.default is Parameter.empty and param_a.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result == True

    param_b = params[1]
    result = param_b.default is Parameter.empty and param_b.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result == True


# LLM-generated content at query #13
#--------------------------

def test_items_without_attribute_error():
    structure = {"a": 1, "b": 2}
    result = _items(structure)
    assert result == [("a", 1), ("b", 2)]

def test_items_with_attribute_error():
    structure = [10, 20, 30]
    result = _items(structure)
    assert result == [(0, 10), (1, 20), (2, 30)]

def test_items_empty_list():
    structure = []
    result = _items(structure)
    assert result == []

def test_items_tuple():
    structure = (100, 200, 300)
    result = _items(structure)
    assert result == [(0, 100), (1, 200), (2, 300)]

def test_items_string():
    structure = "abc"
    result = _items(structure)
    assert result == [(0, "a"), (1, "b"), (2, "c")]


# LLM-generated content at query #14
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

def test_non_callable_key():
    structure = {"a": 1, "b": 2, "c": 3}
    key_spec = "b"
    result = _get_keys_and_values(structure, key_spec)
    expected = [("b", 2)]
    assert result == expected

def test_predicate_with_arity_zero():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_arity_three():
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
    expected = [(0, 10), (2, 30)]
    assert result == expected

def test_binary_predicate_with_sequence():
    structure = [10, 20, 30]
    key_spec = lambda i, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20), (2, 30)]
    assert result == expected

def test_unary_predicate_no_match():
    structure = {"x": 5, "y": 6}
    key_spec = lambda k: k == "z"
    result = _get_keys_and_values(structure, key_spec)
    expected = []
    assert result == expected

def test_binary_predicate_no_match():
    structure = {"x": 5, "y": 6}
    key_spec = lambda k, v: v > 10
    result = _get_keys_and_values(structure, key_spec)
    expected = []
    assert result == expected


# LLM-generated content at query #15
#--------------------------

def test__get_keys_and_values_with_dict_and_callable_unary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k: k in ['a', 'c']
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('a', 1), ('c', 3)]

def test__get_keys_and_values_with_dict_and_callable_binary():
    structure = {'a': 1, 'b': 2, 'c': 3}
    key_spec = lambda k, v: v > 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('b', 2), ('c', 3)]

def test__get_keys_and_values_with_list_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k == 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20)]

def test__get_keys_and_values_with_list_and_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v == 20
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20)]

def test__get_keys_and_values_with_dict_and_non_callable_key():
    structure = {'x': 100, 'y': 200}
    key_spec = 'x'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('x', 100)]

def test__get_keys_and_values_with_list_and_non_callable_index():
    structure = [5, 6, 7]
    key_spec = 2
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(2, 7)]

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
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_non_existent_key():
    structure = {'a': 1, 'b': 2}
    key_spec = 'c'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('c', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_non_existent_index():
    structure = [10, 20]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(5, _EMPTY_SENTINEL)]


# LLM-generated content at query #16
#--------------------------

def test_predicate_arity_other_than_1_or_2_raises_value_error():
    def predicate_with_zero_args():
        return True
    structure = {}
    key_spec = predicate_with_zero_args
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_three_args_raises_value_error():
    def predicate_with_three_args(a, b, c):
        return True
    structure = {}
    key_spec = predicate_with_three_args
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #17
#--------------------------

def test_predicate_at_line_5_evaluates_to_false():
    from inspect import signature, Parameter

    def func_with_default_param(a, b=1):
        pass

    def func_with_keyword_only_param(a, *, b):
        pass

    def func_with_var_positional_param(*args):
        pass

    def func_with_var_keyword_param(**kwargs):
        pass

    def func_with_positional_only_param(a, /, b):
        pass

    params1 = list(signature(func_with_default_param).parameters.values())
    param_b = params1[1]
    result1 = param_b.default is Parameter.empty and param_b.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result1 == False

    params2 = list(signature(func_with_keyword_only_param).parameters.values())
    param_b_kw = params2[1]
    result2 = param_b_kw.default is Parameter.empty and param_b_kw.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result2 == False

    params3 = list(signature(func_with_var_positional_param).parameters.values())
    param_args = params3[0]
    result3 = param_args.default is Parameter.empty and param_args.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result3 == False

    params4 = list(signature(func_with_var_keyword_param).parameters.values())
    param_kwargs = params4[0]
    result4 = param_kwargs.default is Parameter.empty and param_kwargs.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result4 == False

    params5 = list(signature(func_with_positional_only_param).parameters.values())
    param_a = params5[0]
    result5 = param_a.default is Parameter.empty and param_a.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result5 == True
    param_b_pos = params5[1]
    result6 = param_b_pos.default is Parameter.empty and param_b_pos.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result6 == True


# LLM-generated content at query #18
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

def test__get_keys_and_values_with_list_and_non_callable_index():
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
    key_spec = lambda a, b, c: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test__get_keys_and_values_with_non_existent_key_returns_empty_sentinel():
    structure = {'a': 1}
    key_spec = 'b'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('b', _EMPTY_SENTINEL)]
    assert result == expected

def test__get_keys_and_values_with_object_and_attribute():
    class TestObject:
        attr = 42
    structure = TestObject()
    key_spec = 'attr'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('attr', 42)]
    assert result == expected

def test__get_keys_and_values_with_object_and_missing_attribute_returns_empty_sentinel():
    class TestObject:
        pass
    structure = TestObject()
    key_spec = 'missing_attr'
    result = _get_keys_and_values(structure, key_spec)
    expected = [('missing_attr', _EMPTY_SENTINEL)]
    assert result == expected


# LLM-generated content at query #19
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
    assert (param_b.default is Parameter.empty) == False
    assert (param_d.default is Parameter.empty) == False


# LLM-generated content at query #20
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
    assert result_a is True
    assert result_b is False
    assert result_c is False
    assert result_d is False


# LLM-generated content at query #21
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
    key_spec = 'x'
    structure = {'x': 10, 'y': 20}
    result = _get_keys_and_values(structure, key_spec)
    expected = [('x', 10)]
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

def test_unary_predicate_with_sequence():
    key_spec = lambda i: i % 2 == 0
    structure = ['a', 'b', 'c', 'd']
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 'a'), (2, 'c')]
    assert result == expected

def test_binary_predicate_with_sequence():
    key_spec = lambda i, v: i % 2 == 0 and v.isupper()
    structure = ['A', 'b', 'C', 'd']
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 'A'), (2, 'C')]
    assert result == expected

def test_non_callable_key_with_sequence():
    key_spec = 2
    structure = ['a', 'b', 'c', 'd']
    result = _get_keys_and_values(structure, key_spec)
    expected = [(2, 'c')]
    assert result == expected


# LLM-generated content at query #22
#--------------------------

def test_predicate_at_line_5_evaluates_to_false():
    from inspect import signature, Parameter
    def func_with_default(a, b=1):
        pass
    param = list(signature(func_with_default).parameters.values())[1]
    result = param.default is Parameter.empty and param.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    assert result == False


# LLM-generated content at query #23
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


# LLM-generated content at query #24
#--------------------------

def test_predicate_arity_other_than_one_or_two_raises_value_error():
    def predicate_with_zero_args():
        return True
    structure = {}
    try:
        _get_keys_and_values(structure, predicate_with_zero_args)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_three_arguments_raises_value_error():
    def predicate_with_three_args(a, b, c):
        return True
    structure = {}
    try:
        _get_keys_and_values(structure, predicate_with_three_args)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #25
#--------------------------

def test_get_keys_and_values_with_callable_arity_other_than_1_or_2():
    def invalid_arity_predicate(a, b, c):
        return True
    structure = {}
    try:
        _get_keys_and_values(structure, invalid_arity_predicate)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #26
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

def test__get_keys_and_values_with_sequence_and_callable_unary():
    structure = [10, 20, 30]
    key_spec = lambda k: k % 2 == 0
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(0, 10), (2, 30)]

def test__get_keys_and_values_with_sequence_and_callable_binary():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v > 15
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 20), (2, 30)]

def test__get_keys_and_values_with_mapping_and_non_callable_key():
    structure = {'x': 100, 'y': 200}
    key_spec = 'x'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('x', 100)]

def test__get_keys_and_values_with_sequence_and_non_callable_key():
    structure = [5, 6, 7]
    key_spec = 1
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(1, 6)]

def test__get_keys_and_values_with_missing_key_in_mapping():
    structure = {'a': 1}
    key_spec = 'missing'
    result = _get_keys_and_values(structure, key_spec)
    assert result == [('missing', _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_missing_index_in_sequence():
    structure = [1, 2]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(5, _EMPTY_SENTINEL)]

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
    key_spec = lambda x, y, z: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #27
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
    structure = {1: 'a'}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_three_arity():
    key_spec = lambda a, b, c: True
    structure = {1: 'a'}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"


# LLM-generated content at query #28
#--------------------------

def test_items_returns_list_of_enumerated_items_for_non_dict():
    result = _items([10, 20, 30])
    expected = list(enumerate([10, 20, 30]))
    assert result == expected


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

def test__get_keys_and_values_with_dict_and_non_existent_non_callable_key():
    structure = {"a": 1}
    key_spec = "b"
    result = _get_keys_and_values(structure, key_spec)
    assert result == [("b", _EMPTY_SENTINEL)]

def test__get_keys_and_values_with_list_and_out_of_range_non_callable_key():
    structure = [1, 2]
    key_spec = 5
    result = _get_keys_and_values(structure, key_spec)
    assert result == [(5, _EMPTY_SENTINEL)]


# LLM-generated content at query #31
#--------------------------

def test_rex_returns_callable():
    import re
    matcher = rex("^test.*")
    assert callable(matcher)

def test_rex_matches_correct_string():
    import re
    matcher = rex("^a.*z$")
    assert matcher("abcz") == True

def test_rex_does_not_match_incorrect_string():
    import re
    matcher = rex("^a.*z$")
    assert matcher("abcy") == False

def test_rex_returns_false_for_non_string():
    import re
    matcher = rex(".*")
    assert matcher(123) == False
    assert matcher(None) == False
    assert matcher([]) == False

def test_rex_uses_fullmatch_equivalent():
    import re
    matcher = rex("a.*z")
    assert matcher("a test z") == True
    assert matcher("start a test z end") == False

def test_rex_with_empty_pattern():
    import re
    matcher = rex("")
    assert matcher("") == True
    assert matcher("any") == True

def test_rex_with_special_regex_characters():
    import re
    matcher = rex("^\\d+\\.\\d+$")
    assert matcher("123.456") == True
    assert matcher("123") == False
    assert matcher("abc.def") == False


# LLM-generated content at query #32
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

def test_get_keys_and_values_with_invalid_arity_predicate():
    structure = {"a": 1}
    key_spec = lambda: True
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_get_keys_and_values_with_empty_structure():
    structure = {}
    key_spec = lambda k: True
    result = _get_keys_and_values(structure, key_spec)
    assert result == []

def test_get_keys_and_values_with_sequence_structure():
    structure = [10, 20, 30]
    key_spec = lambda k, v: v % 20 == 0
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20)]
    assert result == expected


# LLM-generated content at query #33
#--------------------------

def test__items_with_dict():
    structure = {'a': 1, 'b': 2}
    result = _items(structure)
    expected = [('a', 1), ('b', 2)]
    assert sorted(result) == sorted(expected)

def test__items_with_list():
    structure = ['x', 'y', 'z']
    result = _items(structure)
    expected = [(0, 'x'), (1, 'y'), (2, 'z')]
    assert result == expected

def test__items_with_tuple():
    structure = (10, 20, 30)
    result = _items(structure)
    expected = [(0, 10), (1, 20), (2, 30)]
    assert result == expected

def test__items_with_empty_dict():
    structure = {}
    result = _items(structure)
    expected = []
    assert result == expected

def test__items_with_empty_list():
    structure = []
    result = _items(structure)
    expected = []
    assert result == expected

def test__items_with_string():
    structure = 'ab'
    result = _items(structure)
    expected = [(0, 'a'), (1, 'b')]
    assert result == expected


# LLM-generated content at query #34
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
    structure = {'x': 42, 'y': 0}
    result = _get_keys_and_values(structure, key_spec)
    expected = [('x', 42)]
    assert result == expected

def test_predicate_with_arity_error():
    key_spec = lambda: True
    structure = {}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_on_sequence():
    key_spec = lambda i: i == 1
    structure = ['a', 'b', 'c']
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 'b')]
    assert result == expected


# LLM-generated content at query #35
#--------------------------

def test_update_structure_with_discard_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({})
    assert result == expected

def test_update_structure_with_callable_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = lambda x: x * 2
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 2, 'b': 4})
    assert result == expected

def test_update_structure_with_non_callable_command_and_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = 100
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 100, 'b': 100})
    assert result == expected

def test_update_structure_with_discard_command_and_non_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'x': 10}), 'b': pmap({'y': 20})})
    kvs = [('a', pmap({'x': 10})), ('b', pmap({'y': 20}))]
    path = ['x']
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({}), 'b': pmap({'y': 20})})
    assert result == expected

def test_update_structure_with_callable_command_and_non_empty_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'x': 10}), 'b': pmap({'y': 20})})
    kvs = [('a', pmap({'x': 10})), ('b', pmap({'y': 20}))]
    path = ['x']
    command = lambda x: x + 5
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'x': 15}), 'b': pmap({'y': 20})})
    assert result == expected

def test_update_structure_with_empty_sentinel_value_and_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 1})
    assert result == expected

def test_update_structure_with_empty_sentinel_value_and_non_discard_command():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = []
    command = 99
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 1, 'b': 99})
    assert result == expected

def test_update_structure_with_empty_sentinel_value_and_nested_path():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'x': 10})})
    kvs = [('b', _EMPTY_SENTINEL)]
    path = ['y']
    command = 50
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'x': 10}), 'b': pmap({'y': 50})})
    assert result == expected

def test_update_structure_with_reversed_discard_for_sequence():
    from pyrsistent import pvector
    structure = pvector([10, 20, 30])
    kvs = [(0, 10), (1, 20), (2, 30)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    expected = pvector([])
    assert result == expected

def test_update_structure_with_no_change_when_result_equals_original():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result is structure


# LLM-generated content at query #36
#--------------------------

def test_update_structure_with_empty_path_and_discard_command():
    from pyrsistent import m
    from pyrsistent._pvector import _EMPTY_SENTINEL
    from pyrsistent._helpers import discard
    structure = m(a=1, b=2)
    kvs = [('a', 1), ('b', 2)]
    path = []
    command = discard
    result = _update_structure(structure, kvs, path, command)
    assert not path and command is discard


# LLM-generated content at query #37
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

def test_predicate_with_arity_0():
    key_spec = lambda: True
    structure = {"a": 1}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_predicate_with_arity_3():
    key_spec = lambda a, b, c: True
    structure = {"a": 1}
    try:
        _get_keys_and_values(structure, key_spec)
        assert False
    except ValueError as e:
        assert str(e) == "callable in transform path must take 1 or 2 arguments"

def test_unary_predicate_with_sequence():
    key_spec = lambda i: i % 2 == 0
    structure = [10, 20, 30]
    result = _get_keys_and_values(structure, key_spec)
    expected = [(0, 10), (2, 30)]
    assert result == expected

def test_binary_predicate_with_sequence():
    key_spec = lambda i, v: v > 15
    structure = [10, 20, 30]
    result = _get_keys_and_values(structure, key_spec)
    expected = [(1, 20), (2, 30)]
    assert result == expected


# LLM-generated content at query #38
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
    structure = pmap({'a': pmap({'x': 1, 'y': 2})})
    kvs = [('a', pmap({'x': 1, 'y': 2}))]
    path = []
    result = _update_structure(structure, kvs, path, discard)
    expected = pmap({})
    assert result == expected

def test_update_structure_discard_reverse_order():
    from pyrsistent import pvector
    structure = pvector([pvector([1, 2]), pvector([3, 4])])
    kvs = [(0, pvector([1, 2])), (1, pvector([3, 4]))]
    path = []
    result = _update_structure(structure, kvs, path, discard)
    expected = pvector([])
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

def test_update_structure_update_with_empty_sentinel():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({})})
    kvs = [('a', pmap({}))]
    path = ['b']
    command = lambda x: 1
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'b': 1})})
    assert result == expected

def test_update_structure_update_multiple_keys():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': 1}), 'c': pmap({'d': 2})})
    kvs = [('a', pmap({'b': 1})), ('c', pmap({'d': 2}))]
    path = ['b']
    command = lambda x: 3
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'b': 3}), 'c': pmap({'d': 2})})
    assert result == expected

def test_update_structure_no_change():
    from pyrsistent import pmap
    structure = pmap({'a': pmap({'b': 1})})
    kvs = [('a', pmap({'b': 1}))]
    path = ['b']
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': pmap({'b': 1})})
    assert result == expected

def test_update_structure_empty_path_update():
    from pyrsistent import pmap
    structure = pmap({'a': 1})
    kvs = [('a', 1)]
    path = []
    command = lambda x: 2
    result = _update_structure(structure, kvs, path, command)
    expected = pmap({'a': 2})
    assert result == expected

def test_update_structure_empty_path_discard():
    from pyrsistent import pmap
    structure = pmap({'a': 1, 'b': 2})
    kvs = [('a', 1), ('b', 2)]
    path = []
    result = _update_structure(structure, kvs, path, discard)
    expected = pmap({})
    assert result == expected


# LLM-generated content at query #39
#--------------------------

def test_update_structure_discard_with_empty_path():
    from pyrsistent import m, v
    structure = m(a=1, b=2)
    kvs = [('a', 1), ('b', 2)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == m(b=2)

def test_update_structure_discard_with_nested_path():
    from pyrsistent import m, v
    structure = m(a=m(x=1, y=2), b=m(z=3))
    kvs = [('a', m(x=1, y=2))]
    result = _update_structure(structure, kvs, ['x'], discard)
    assert result == m(a=m(y=2), b=m(z=3))

def test_update_structure_discard_reverse_order_for_vectors():
    from pyrsistent import m, v
    structure = v(10, 20, 30)
    kvs = [(0, 10), (1, 20), (2, 30)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == v()

def test_update_structure_discard_skip_empty_sentinel():
    from pyrsistent import m
    structure = m(a=1)
    kvs = [('b', _EMPTY_SENTINEL)]
    result = _update_structure(structure, kvs, [], discard)
    assert result == m(a=1)

def test_update_structure_update_with_empty_sentinel_and_pmap():
    from pyrsistent import m
    structure = m()
    kvs = [('new_key', _EMPTY_SENTINEL)]
    command = lambda x: 42
    result = _update_structure(structure, kvs, [], command)
    assert result == m(new_key=42)

def test_update_structure_update_with_nested_path_and_modification():
    from pyrsistent import m
    structure = m(a=m(b=1))
    kvs = [('a', m(b=1))]
    command = lambda x: x * 2
    result = _update_structure(structure, kvs, ['b'], command)
    assert result == m(a=m(b=2))

def test_update_structure_no_change_when_result_equals_original():
    from pyrsistent import m
    structure = m(a=5)
    kvs = [('a', 5)]
    command = lambda x: x
    result = _update_structure(structure, kvs, [], command)
    assert result is structure

def test_update_structure_update_multiple_keys():
    from pyrsistent import m
    structure = m(a=1, b=2)
    kvs = [('a', 1), ('b', 2)]
    command = lambda x: x + 10
    result = _update_structure(structure, kvs, [], command)
    assert result == m(a=11, b=12)


# LLM-generated content at query #40
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    from pyrsistent._pmap import pmap
    from pyrsistent._pvector import pvector
    from pyrsistent._pset import pset
    from pyrsistent._precord import PRecord
    from pyrsistent._pclass import PClass
    from pyrsistent._field_common import discard
    from pyrsistent._transformations import _EMPTY_SENTINEL, _do_to_path, _update_structure
    structure = pmap({'a': 1})
    kvs = [('a', 2)]
    path = ['b']
    command = lambda x: x
    result = _update_structure(structure, kvs, path, command)
    assert result == pmap({'a': pmap({'b': 2})})


