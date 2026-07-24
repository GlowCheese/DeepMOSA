####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_valid_key_value_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_invalid_key_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_invalid_value_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'invalid'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_with_invariant_violation_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v > 0, 'Value non-positive')]
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_multiple_invariants_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v > 0, 'Value non-positive')]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_initial_as_same_type_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result is not original

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_checked_key_type_and_create():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    try:
        CustomMap({'raw_key': 'raw_value'})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_checked_value_type_and_create():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    try:
        CustomMap({KeyType(): 'raw_value'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_with_initial_as_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial_dict = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_dict)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial_dict

def test_constructor_with_no_arguments():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_positional_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5

def test_constructor_with_keyword_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap(initial={1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5


# LLM-generated content at query #2
#--------------------------

def test_maybe_parse_user_type_with_preserved_type():
    result = maybe_parse_user_type(list)
    assert result == [list]

def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]

def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    result = maybe_parse_user_type((int, str))
    assert result == (int, str)

def test_maybe_parse_user_type_with_nested_iterable():
    result = maybe_parse_user_type([(int, float), str])
    assert result == (int, float, str)

def test_maybe_parse_user_type_with_empty_iterable():
    result = maybe_parse_user_type([])
    assert result == ()

def test_maybe_parse_user_type_raises_type_error_for_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False
    except TypeError:
        assert True

def test_maybe_parse_user_type_with_tuple_of_strings():
    result = maybe_parse_user_type(("str", "int"))
    assert result == ("str", "int")

def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(["bool", "float"])
    assert result == ("bool", "float")

def test_maybe_parse_user_type_with_mixed_iterable():
    result = maybe_parse_user_type([int, "str", (float, bool)])
    assert result == (int, "str", float, bool)


# LLM-generated content at query #3
#--------------------------

def test_store_invariants_adds_wrapped_invariants():
    def inv1(self):
        return True, ()
    def inv2(self):
        return False, ('error',)
    class Base:
        pass
    Base.__dict__['_invariant_'] = inv1
    dct = {}
    store_invariants(dct, (Base,), '_invariants_', '_invariant_')
    result = dct['_invariants_'][0](None)
    assert result == (True, ())

def test_store_invariants_merges_multiple_inherited_invariants():
    def invA(self):
        return True, ()
    def invB(self):
        return False, ('err',)
    class A:
        pass
    A.__dict__['_invariant_'] = invA
    class B(A):
        pass
    B.__dict__['_invariant_'] = invB
    dct = {}
    store_invariants(dct, (B,), '_invariants_', '_invariant_')
    invariants = dct['_invariants_']
    assert len(invariants) == 2
    result1 = invariants[0](None)
    result2 = invariants[1](None)
    assert result1 == (True, ())
    assert result2 == (False, ('err',))

def test_store_invariants_handles_diamond_inheritance_without_duplicates():
    def invX(self):
        return True, ()
    class X:
        pass
    X.__dict__['_invariant_'] = invX
    class A(X):
        pass
    class B(X):
        pass
    class C(A, B):
        pass
    dct = {}
    store_invariants(dct, (C,), '_invariants_', '_invariant_')
    invariants = dct['_invariants_']
    assert len(invariants) == 1
    result = invariants[0](None)
    assert result == (True, ())

def test_store_invariants_raises_type_error_for_non_callable():
    class Base:
        pass
    Base.__dict__['_invariant_'] = 'not callable'
    dct = {}
    try:
        store_invariants(dct, (Base,), '_invariants_', '_invariant_')
        assert False
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test_store_invariants_includes_invariant_from_current_dict():
    def inv_local(self):
        return (False, ('local',))
    def inv_base(self):
        return (True, ())
    class Base:
        pass
    Base.__dict__['_invariant_'] = inv_base
    dct = {'_invariant_': inv_local}
    store_invariants(dct, (Base,), '_invariants_', '_invariant_')
    invariants = dct['_invariants_']
    assert len(invariants) == 2
    result_local = invariants[0](None)
    result_base = invariants[1](None)
    assert result_local == (False, ('local',))
    assert result_base == (True, ())

def test_store_invariants_wraps_invariant_returning_tuple_of_tuples():
    def inv_composite(self):
        return ((True, ()), (False, ('err1',)), (True, ()))
    class Base:
        pass
    Base.__dict__['_invariant_'] = inv_composite
    dct = {}
    store_invariants(dct, (Base,), '_invariants_', '_invariant_')
    wrapped_inv = dct['_invariants_'][0]
    result = wrapped_inv(None)
    assert result == (False, ('err1',))

def test_store_invariants_with_no_invariants_found():
    class Base:
        pass
    dct = {}
    store_invariants(dct, (Base,), '_invariants_', '_invariant_')
    invariants = dct['_invariants_']
    assert invariants == ()

def test_store_invariants_wrap_invariant_preserves_simple_bool_result():
    def inv_simple(self):
        return False, ('simple',)
    class Base:
        pass
    Base.__dict__['_invariant_'] = inv_simple
    dct = {}
    store_invariants(dct, (Base,), '_invariants_', '_invariant_')
    wrapped_inv = dct['_invariants_'][0]
    result = wrapped_inv(None)
    assert result == (False, ('simple',))


# LLM-generated content at query #4
#--------------------------

def test_constructor_with_valid_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_invalid_key_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
    except CheckedKeyTypeError:
        pass
    else:
        assert False, "Expected CheckedKeyTypeError"

def test_constructor_with_invalid_value_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'invalid'})
    except CheckedTypeError:
        pass
    else:
        assert False, "Expected CheckedTypeError"

def test_constructor_with_invariant_violation_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
    except InvariantException:
        pass
    else:
        assert False, "Expected InvariantException"

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert result == {1: 1.0}

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    result = IntToFloatMap({1: 2.0})
    assert result == {1: 2.0}

def test_constructor_with_multiple_invariants_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    try:
        IntToFloatMap({-1: 2.0})
    except InvariantException as e:
        assert 'Key negative' in str(e.error_codes)
    else:
        assert False, "Expected InvariantException"

def test_constructor_with_checked_type_key():
    class KeyType(CheckedType):
        __type__ = int
    class ValueType(CheckedType):
        __type__ = float
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap({1: 2.5})
    assert result == {1: 2.5}

def test_constructor_with_checked_type_value():
    class KeyType(CheckedType):
        __type__ = int
    class ValueType(CheckedType):
        __type__ = float
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap({1: 2.5})
    assert result == {1: 2.5}

def test_constructor_with_checked_type_and_invariant():
    class KeyType(CheckedType):
        __type__ = int
    class ValueType(CheckedType):
        __type__ = float
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
        __invariant__ = lambda k, v: (v > k, 'Value not greater than key')
    result = CustomMap({1: 2.5})
    assert result == {1: 2.5}

def test_constructor_with_checked_type_and_invariant_violation():
    class KeyType(CheckedType):
        __type__ = int
    class ValueType(CheckedType):
        __type__ = float
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
        __invariant__ = lambda k, v: (v > k, 'Value not greater than key')
    try:
        CustomMap({2: 1.5})
    except InvariantException:
        pass
    else:
        assert False, "Expected InvariantException"

def test_constructor_with_same_class_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result is original

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert result == {1: 1.5}

def test_constructor_with_undefined_size_constant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=_UNDEFINED_CHECKED_PMAP_SIZE)
    assert result == {1: 1.5}

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert str(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_with_inheritance():
    class BaseMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    class DerivedMap(BaseMap):
        pass
    result = DerivedMap({1: 1.5})
    assert result == {1: 1.5}
    assert isinstance(result, DerivedMap)

def test_constructor_with_no_type_specification():
    class AnyMap(CheckedPMap):
        pass
    result = AnyMap({1: 'a', 'b': 2})
    assert result == {1: 'a', 'b': 2}

def test_constructor_with_union_key_type():
    class IntOrStrMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = float
    result = IntOrStrMap({1: 1.5, 'a': 2.5})
    assert result == {1: 1.5, 'a': 2.5}

def test_constructor_with_union_value_type():
    class FloatOrStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = (float, str)
    result = FloatOrStrMap({1: 1.5, 2: 'text'})
    assert result == {1: 1.5, 2: 'text'}

def test_constructor_with_union_type_violation():
    class IntOrStrMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = float
    try:
        IntOrStrMap({1.5: 1.5})
    except CheckedKeyTypeError:
        pass
    else:
        assert False, "Expected CheckedKeyTypeError"


# LLM-generated content at query #5
#--------------------------

def test_store_invariants_all_callable():
    dct = {}
    bases = ()
    destination_name = 'invariants'
    source_name = '_invariant'
    invariant_func = lambda x: True
    dct[source_name] = invariant_func
    store_invariants(dct, bases, destination_name, source_name)
    assert destination_name in dct
    assert isinstance(dct[destination_name], tuple)
    assert len(dct[destination_name]) == 1
    assert callable(dct[destination_name][0])


# LLM-generated content at query #6
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_checkedpmap_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(initial)
    assert result == {1: 1.5}
    assert result is initial

def test_constructor_type_check_key_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_check_value_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'string'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_invariant_success():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert result == {1: 1.0}

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert result == {1: 1.5}

def test_constructor_with_checked_key_type_create():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return int(source_data)
    class MyMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = int
    result = MyMap({'1': 5})
    assert result == {1: 5}

def test_constructor_with_checked_value_type_create():
    class ValueType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return str(source_data)
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = ValueType
    result = MyMap({1: 5})
    assert result == {1: '5'}

def test_constructor_with_both_checked_types_create():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return int(source_data)
    class ValueType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return str(source_data)
    class MyMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = MyMap({'1': 5})
    assert result == {1: '5'}

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert str(result) == "IntToFloatMap({1: 1.5})"


# LLM-generated content at query #7
#--------------------------

def test_store_invariants_adds_tuple_of_wrapped_invariants():
    from unittest.mock import Mock
    mock_dct = {}
    mock_bases = []
    mock_source_name = 'invariant'
    mock_destination_name = 'invariants'
    mock_invariant1 = Mock()
    mock_invariant2 = Mock()
    mock_wrap_invariant = Mock(side_effect=lambda x: x)
    import sys
    original_all_dicts = sys.modules[__name__]._all_dicts if '_all_dicts' in dir(sys.modules[__name__]) else None
    mock_all_dicts = Mock(return_value=[])
    sys.modules[__name__]._all_dicts = mock_all_dicts
    mock_dct[mock_source_name] = mock_invariant1
    store_invariants(mock_dct, mock_bases, mock_destination_name, mock_source_name)
    assert mock_dct[mock_destination_name] == (mock_invariant1,)
    if original_all_dicts:
        sys.modules[__name__]._all_dicts = original_all_dicts


# LLM-generated content at query #8
#--------------------------

def test_store_invariants_all_callable():
    from unittest.mock import Mock
    mock_dct = {}
    mock_bases = []
    mock_source_name = 'invariant'
    mock_callable1 = Mock()
    mock_callable2 = Mock()
    mock_ns1 = {mock_source_name: mock_callable1}
    mock_ns2 = {mock_source_name: mock_callable2}
    def mock_all_dicts(bases):
        return [mock_ns2]
    import sys
    original_all_dicts = sys.modules[__name__]._all_dicts if '_all_dicts' in dir(sys.modules[__name__]) else None
    sys.modules[__name__]._all_dicts = mock_all_dicts
    mock_dct[mock_source_name] = mock_callable1
    store_invariants(mock_dct, mock_bases, 'dest', mock_source_name)
    assert isinstance(mock_dct['dest'], tuple)
    assert len(mock_dct['dest']) == 2
    if original_all_dicts is not None:
        sys.modules[__name__]._all_dicts = original_all_dicts
    else:
        del sys.modules[__name__]._all_dicts


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_18_evaluates_to_true():
    from typing import Iterable
    class _preserved_iterable_types:
        pass
    class CustomType:
        pass
    t = CustomType
    is_type = isinstance(t, type)
    is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
    is_string = isinstance(t, str)
    is_iterable = isinstance(t, Iterable)
    predicate = is_type and not is_iterable
    assert predicate == True


# LLM-generated content at query #10
#--------------------------

def test__store_types_single_dict():
    dct = {'source': int}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == (int,)

def test__store_types_multiple_dicts():
    dct1 = {'source': str}
    dct2 = {'source': float}
    bases = [type('Base1', (), dct1), type('Base2', (), dct2)]
    dct = {}
    _store_types(dct, bases, 'dest', 'source')
    assert dct['dest'] == (str, float)

def test__store_types_with_iterable_source():
    dct = {'source': [int, str]}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == (int, str)

def test__store_types_with_string_type():
    dct = {'source': 'MyClass'}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == ('MyClass',)

def test__store_types_with_preserved_iterable_type():
    dct = {'source': list}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == (list,)

def test__store_types_missing_source():
    dct = {}
    _store_types(dct, [], 'dest', 'source')
    assert 'dest' not in dct

def test__store_types_combine_dict_and_bases():
    base_dict = {'source': bool}
    base = type('Base', (), base_dict)
    dct = {'source': bytes}
    _store_types(dct, [base], 'dest', 'source')
    assert dct['dest'] == (bytes, bool)

def test__store_types_nested_iterables():
    dct = {'source': [[int, str], float]}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == (int, str, float)

def test__store_types_invalid_type_raises():
    dct = {'source': 123}
    try:
        _store_types(dct, [], 'dest', 'source')
        assert False
    except TypeError:
        pass

def test__store_types_empty_iterable():
    dct = {'source': []}
    _store_types(dct, [], 'dest', 'source')
    assert dct['dest'] == ()


# LLM-generated content at query #11
#--------------------------

def test_CheckedType_constructor_initialization():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_18_evaluates_to_true():
    from collections.abc import Iterable
    class _preserved_iterable_types:
        pass
    class CustomType:
        pass
    t = CustomType
    is_type = isinstance(t, type)
    is_iterable = isinstance(t, Iterable)
    result = is_type and not is_iterable
    assert result == True


# LLM-generated content at query #13
#--------------------------

def test_invariant_errors_no_errors():
    elem = 5
    invariants = [lambda x: (True, None), lambda x: (True, "ok")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_with_errors():
    elem = "test"
    invariants = [lambda x: (False, "error1"), lambda x: (True, "ok"), lambda x: (False, "error2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["error1", "error2"]

def test_invariant_errors_empty_invariants():
    elem = object()
    invariants = []
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_all_true():
    elem = [1, 2, 3]
    invariants = [lambda x: (True, "data1"), lambda x: (True, "data2")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_all_false():
    elem = None
    invariants = [lambda x: (False, "err1"), lambda x: (False, "err2"), lambda x: (False, "err3")]
    result = _invariant_errors(elem, invariants)
    assert result == ["err1", "err2", "err3"]


# LLM-generated content at query #14
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_correct_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.0, 3: 3.5})
    assert result[1] == 1.0
    assert result[3] == 3.5

def test_constructor_with_checked_type_key():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key = KeyType()
    value = ValueType()
    result = CustomMap({key: value})
    assert result[key] == value

def test_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException as e:
        assert 'Invalid mapping' in str(e)

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result == {1: 1.0, 2: 2.0}

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    try:
        IntToFloatMap({-1: 1.0})
        assert False
    except InvariantException as e:
        assert 'Key negative' in str(e)
    try:
        IntToFloatMap({1: -1.0})
        assert False
    except InvariantException as e:
        assert 'Value negative' in str(e)

def test_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.0})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'a'})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_checkedpmap_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result == original
    assert result is not original

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert result == {1: 1.5}

def test_constructor_returns_same_type():
    class MyMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    instance = MyMap({'a': 1})
    assert type(instance) is MyMap

def test_constructor_with_nested_checked_types():
    class Inner(CheckedType):
        pass
    class Outer(CheckedPMap):
        __key_type__ = str
        __value_type__ = Inner
    inner = Inner()
    result = Outer({'x': inner})
    assert result['x'] is inner

def test_constructor_with_inheritance():
    class BaseMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    class DerivedMap(BaseMap):
        pass
    result = DerivedMap({1: 2.5})
    assert isinstance(result, DerivedMap)
    assert result == {1: 2.5}

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    instance = IntToFloatMap({1: 1.5})
    assert repr(instance) == "IntToFloatMap({1: 1.5})"


# LLM-generated content at query #15
#--------------------------

def test_invariant_errors_no_errors():
    elem = 5
    invariants = [lambda x: (True, None), lambda x: (True, "ok")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_with_errors():
    elem = "test"
    invariants = [lambda x: (False, "error1"), lambda x: (True, "ok"), lambda x: (False, "error2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["error1", "error2"]

def test_invariant_errors_empty_invariants():
    elem = object()
    invariants = []
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_all_true():
    elem = [1, 2, 3]
    invariants = [lambda x: (True, "a"), lambda x: (True, "b"), lambda x: (True, "c")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_all_false():
    elem = None
    invariants = [lambda x: (False, "err1"), lambda x: (False, "err2"), lambda x: (False, "err3")]
    result = _invariant_errors(elem, invariants)
    assert result == ["err1", "err2", "err3"]


# LLM-generated content at query #16
#--------------------------

def test__checked_type_create_with_same_class():
    class MockClass:
        _checked_types = []
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance

def test__checked_type_create_without_checked_types():
    class MockClass:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result.data == source_data

def test__checked_type_create_with_checked_type_and_matching_data():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls()
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    source_data = [CheckedType()]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result.data == source_data

def test__checked_type_create_with_checked_type_and_non_matching_data():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls()
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    source_data = [1]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert all(isinstance(item, CheckedType) for item in result.data)

def test__checked_type_create_with_checked_type_and_ignore_extra():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls()
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    source_data = [1]
    result = _checked_type_create(MockClass, source_data, ignore_extra=True)
    assert isinstance(result, MockClass)
    assert all(isinstance(item, CheckedType) for item in result.data)

def test__checked_type_create_with_multiple_checked_types_and_matching_data():
    class CheckedTypeA:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls()
    class CheckedTypeB:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls()
    class MockClass:
        _checked_types = ['__main__.CheckedTypeA', '__main__.CheckedTypeB']
        def __init__(self, data):
            self.data = data
    source_data = [CheckedTypeA()]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert result.data == source_data

def test__checked_type_create_with_multiple_checked_types_and_non_matching_data():
    class CheckedTypeA:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls()
    class CheckedTypeB:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls()
    class MockClass:
        _checked_types = ['__main__.CheckedTypeA', '__main__.CheckedTypeB']
        def __init__(self, data):
            self.data = data
    source_data = [1]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)
    assert all(isinstance(item, (CheckedTypeA, CheckedTypeB)) for item in result.data)


# LLM-generated content at query #17
#--------------------------

def test_constructor_with_valid_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_type_check_failure_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_check_failure_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'a'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_invariant_success():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert result == {1: 1.0}

def test_constructor_with_checked_type_key():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key_instance = KeyType()
    value_instance = ValueType()
    result = CustomMap({key_instance: value_instance})
    assert result == {key_instance: value_instance}

def test_constructor_with_checked_type_create_key():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, data):
            return cls()
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap.create({'raw_key': 'raw_value'})
    assert isinstance(list(result.keys())[0], KeyType)
    assert isinstance(list(result.values())[0], ValueType)

def test_constructor_with_checked_type_create_value():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        @classmethod
        def create(cls, data):
            return cls()
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap.create({'raw_key': 'raw_value'})
    assert isinstance(list(result.keys())[0], KeyType)
    assert isinstance(list(result.values())[0], ValueType)

def test_constructor_persistent_returns_same_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    evolver = IntToFloatMap().evolver()
    evolver.set(1, 1.5)
    result = evolver.persistent()
    assert isinstance(result, IntToFloatMap)
    assert result == {1: 1.5}

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert str(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert result == {1: 1.5}
    assert isinstance(result, IntToFloatMap)

def test_constructor_serialize():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __serializer__ = lambda format, k, v: (k, v)
    result = IntToFloatMap({1: 1.5})
    serialized = result.serialize()
    assert serialized == {1: 1.5}

def test_constructor_create_with_same_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap.create(original)
    assert result is original

def test_constructor_reduce_for_pickling():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    instance = IntToFloatMap({1: 1.5})
    reduced = instance.__reduce__()
    assert reduced[0] is _restore_pickle
    assert reduced[1] == (IntToFloatMap, {1: 1.5})


# LLM-generated content at query #18
#--------------------------

def test_maybe_parse_user_type_with_preserved_type():
    result = maybe_parse_user_type(list)
    assert result == [list]

def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]

def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    result = maybe_parse_user_type((int, str))
    assert result == (int, str)

def test_maybe_parse_user_type_with_nested_iterable():
    result = maybe_parse_user_type([(int, str), float])
    assert result == (int, str, float)

def test_maybe_parse_user_type_with_empty_iterable():
    result = maybe_parse_user_type([])
    assert result == ()

def test_maybe_parse_user_type_raises_type_error_for_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False
    except TypeError:
        assert True

def test_maybe_parse_user_type_with_tuple_of_strings():
    result = maybe_parse_user_type(("str", "int"))
    assert result == ("str", "int")

def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(["str", "int"])
    assert result == ("str", "int")

def test_maybe_parse_user_type_with_mixed_iterable():
    result = maybe_parse_user_type([int, "str"])
    assert result == (int, "str")


# LLM-generated content at query #19
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert result == initial
    assert isinstance(result, IntToFloatMap)

def test_constructor_type_checking_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert result == {1: 1.5}

def test_constructor_type_checking_invalid_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_checking_invalid_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'string'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert result == {1: 1.0}

def test_constructor_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.0})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_checked_type_key():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key_instance = KeyType()
    value_instance = ValueType()
    result = CustomMap({key_instance: value_instance})
    assert result == {key_instance: value_instance}

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [
            lambda k, v: (k >= 0, 'Key must be non-negative'),
            lambda k, v: (v > 0, 'Value must be positive')
        ]
    result = IntToFloatMap({1: 1.5})
    assert result == {1: 1.5}

def test_constructor_with_multiple_invariants_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [
            lambda k, v: (k >= 0, 'Key must be non-negative'),
            lambda k, v: (v > 0, 'Value must be positive')
        ]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == 'IntToFloatMap({1: 1.5})'

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert result == {1: 1.5}

def test_constructor_with_initial_as_checkedpmap():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(initial)
    assert result is initial


# LLM-generated content at query #20
#--------------------------

def test_CheckedType_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #21
#--------------------------

def test_constructor_with_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector()
    assert len(result) == 0
    assert isinstance(result, TestVector)


def test_constructor_with_list_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector([1, 2, 3])
    assert list(result) == [1, 2, 3]
    assert isinstance(result, TestVector)


def test_constructor_with_python_pvector_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    pv = python_pvector([4, 5, 6])
    result = TestVector(pv)
    assert list(result) == [4, 5, 6]
    assert isinstance(result, TestVector)


def test_constructor_type_check_success():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector([7, 8, 9])
    assert all(isinstance(x, int) for x in result)


def test_constructor_type_check_failure():
    class TestVector(CheckedPVector):
        __type__ = int
    try:
        TestVector([10, 'invalid', 12])
        assert False
    except TypeError:
        assert True


def test_constructor_invariant_check_success():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1.5, 2, 3.0])
    assert list(result) == [1.5, 2, 3.0]


def test_constructor_invariant_check_failure():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -2, 3])
        assert False
    except InvariantException:
        assert True


def test_constructor_with_tuple_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector((13, 14, 15))
    assert list(result) == [13, 14, 15]


def test_constructor_with_generator_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector(x for x in range(3))
    assert list(result) == [0, 1, 2]


def test_constructor_preserves_subclass_identity():
    class SpecialVector(CheckedPVector):
        __type__ = str
    result = SpecialVector(['a', 'b'])
    assert type(result) is SpecialVector


# LLM-generated content at query #22
#--------------------------

def test_invariant_errors_returns_empty_list_when_all_invariants_valid():
    elem = "test_element"
    invariants = [lambda x: (True, "ok1"), lambda x: (True, "ok2")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_returns_data_from_invalid_invariants():
    elem = 5
    invariants = [lambda x: (True, "ok"), lambda x: (False, "error1"), lambda x: (False, "error2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["error1", "error2"]

def test_invariant_errors_returns_empty_list_for_empty_invariants():
    elem = None
    invariants = []
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_passes_elem_to_each_invariant():
    captured = []
    def capturing_invariant(e):
        captured.append(e)
        return (True, "ok")
    elem = object()
    invariants = [capturing_invariant, capturing_invariant]
    _invariant_errors(elem, invariants)
    assert captured == [elem, elem]

def test_invariant_errors_handles_mixed_valid_and_invalid():
    elem = []
    invariants = [lambda x: (False, "bad"), lambda x: (True, "good"), lambda x: (False, "worse")]
    result = _invariant_errors(elem, invariants)
    assert result == ["bad", "worse"]


# LLM-generated content at query #23
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 1.5})
        assert False
    except InvariantException as e:
        assert 'Invalid mapping' in str(e.error_codes)

def test_constructor_type_error_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_error_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'a'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_with_checked_type_key():
    class KeyType(CheckedType):
        __type__ = int
    class ValueType(CheckedType):
        __type__ = float
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap({1: 1.5})
    assert isinstance(result, CustomMap)
    assert result[1] == 1.5

def test_constructor_with_checked_type_value():
    class KeyType(CheckedType):
        __type__ = int
    class ValueType(CheckedType):
        __type__ = float
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap({1: 1.5})
    assert isinstance(result, CustomMap)
    assert result[1] == 1.5

def test_constructor_with_checked_type_create():
    class KeyType(CheckedType):
        __type__ = int
        @classmethod
        def create(cls, data):
            return int(data)
    class ValueType(CheckedType):
        __type__ = float
        @classmethod
        def create(cls, data):
            return float(data)
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap({'1': '1.5'})
    assert isinstance(result, CustomMap)
    assert result[1] == 1.5

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k > 0, 'Key must be positive'), lambda k, v: (v > 0, 'Value must be positive')]
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_multiple_invariants_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k > 0, 'Key must be positive'), lambda k, v: (v > 0, 'Value must be positive')]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException as e:
        errors = e.error_codes
        assert any('Key must be positive' in str(error) for error in errors)

def test_constructor_with_self_as_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(initial)
    assert result is initial

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert str(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5

def test_constructor_with_undefined_size():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=_UNDEFINED_CHECKED_PMAP_SIZE)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5


# LLM-generated content at query #24
#--------------------------

def test_constructor_with_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector()
    assert len(result) == 0
    assert isinstance(result, TestVector)


def test_constructor_with_list_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector([1, 2, 3])
    assert list(result) == [1, 2, 3]
    assert isinstance(result, TestVector)


def test_constructor_with_python_pvector_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    pv = python_pvector([4, 5, 6])
    result = TestVector(pv)
    assert list(result) == [4, 5, 6]
    assert isinstance(result, TestVector)


def test_constructor_type_check_failure():
    class TestVector(CheckedPVector):
        __type__ = int
    try:
        TestVector([1, 'a', 3])
        assert False
    except TypeError:
        assert True


def test_constructor_invariant_check_failure():
    class Positives(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -2, 3])
        assert False
    except InvariantException:
        assert True


def test_constructor_invariant_check_success():
    class Positives(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 2, 3])
    assert list(result) == [1, 2, 3]
    assert isinstance(result, Positives)


def test_constructor_with_multiple_types():
    class TestVector(CheckedPVector):
        __type__ = (int, float)
    result = TestVector([1, 2.5, 3])
    assert list(result) == [1, 2.5, 3]
    assert isinstance(result, TestVector)


def test_constructor_with_no_type_specified():
    class TestVector(CheckedPVector):
        pass
    result = TestVector([1, 'a', 3.5])
    assert list(result) == [1, 'a', 3.5]
    assert isinstance(result, TestVector)


# LLM-generated content at query #25
#--------------------------

def test__restore_pickle():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields):
            return (cls, data, _factory_fields)
    data = "test_data"
    result = MockClass._restore_pickle(data)
    assert result == (MockClass, data, set())


# LLM-generated content at query #26
#--------------------------

def test_merge_invariant_results_all_true():
    result = [(True, []), (True, []), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_one_false():
    result = [(True, []), (False, ["error1"]), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)

def test_merge_invariant_results_multiple_false():
    result = [(False, ["error1"]), (False, ["error2"]), (False, ["error3"])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2", "error3")

def test_merge_invariant_results_empty_input():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_mixed_data():
    result = [(True, []), (False, ["err1", "err2"]), (False, ["err3"])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("err1", "err2", "err3")


# LLM-generated content at query #27
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({})
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_valid_key_value_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_invalid_key_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_invalid_value_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'invalid'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_with_invariant_violation_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result == {1: 1.0, 2: 2.0}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_multiple_invariants_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_checked_type_key():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key = KeyType()
    value = ValueType()
    result = CustomMap({key: value})
    assert result == {key: value}
    assert isinstance(result, CustomMap)

def test_constructor_with_initial_as_same_class_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(initial)
    assert result == {1: 1.5}
    assert result is initial

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert result == {1: 1.5}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_size_parameter_and_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({}, size=0)
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_initial_as_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial_dict = {1: 1.5, 2: 2.5}
    result = IntToFloatMap(initial_dict)
    assert result == initial_dict
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_initial_as_iterable_of_pairs():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial_pairs = [(1, 1.5), (2, 2.5)]
    result = IntToFloatMap(dict(initial_pairs))
    assert result == dict(initial_pairs)
    assert isinstance(result, IntToFloatMap)

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert str(result) == "IntToFloatMap({1: 1.5})"


# LLM-generated content at query #28
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert dict(result) == initial

def test_constructor_type_check_failure_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {'a': 1.5}
    try:
        IntToFloatMap(initial)
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_check_failure_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 'string'}
    try:
        IntToFloatMap(initial)
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    initial = {1: 2.5}
    try:
        IntToFloatMap(initial)
        assert False
    except InvariantException:
        assert True

def test_constructor_invariant_success():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    initial = {1: 1.0, 2: 2.0}
    result = IntToFloatMap(initial)
    assert dict(result) == initial

def test_constructor_with_checked_key_type_create():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return int(source_data)
    class MyMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = int
    initial = {'1': 10, '2': 20}
    result = MyMap(initial)
    assert dict(result) == {1: 10, 2: 20}

def test_constructor_with_checked_value_type_create():
    class ValueType(CheckedType):
        @classmethod
        def create(cls, source_data):
            return str(source_data)
    class MyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = ValueType
    initial = {1: 10, 2: 20}
    result = MyMap(initial)
    assert dict(result) == {1: '10', 2: '20'}

def test_constructor_with_existing_checkedpmap():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result is original

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == "IntToFloatMap({1: 1.5})"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_store_invariants_stores_wrapped_invariants():
    def inv1():
        return True, ()
    def inv2():
        return False, ("error",)
    class Base1:
        invariant = inv1
    class Base2:
        pass
    dct = {}
    store_invariants(dct, (Base1, Base2), 'invariants', 'invariant')
    result = dct['invariants']
    assert isinstance(result, tuple)
    assert len(result) == 1
    wrapped = result[0]
    assert callable(wrapped)
    assert wrapped() == (True, ())

def test_store_invariants_inherits_from_multiple_bases():
    def inv1():
        return True, ()
    def inv2():
        return False, ("error2",)
    class BaseA:
        invariant = inv1
    class BaseB:
        invariant = inv2
    dct = {}
    store_invariants(dct, (BaseA, BaseB), 'invariants', 'invariant')
    result = dct['invariants']
    assert len(result) == 2
    assert result[0]() == (True, ())
    assert result[1]() == (False, ("error2",))

def test_store_invariants_merges_local_and_inherited():
    def base_inv():
        return True, ()
    def local_inv():
        return False, ("local",)
    class Base:
        invariant = base_inv
    dct = {'invariant': local_inv}
    store_invariants(dct, (Base,), 'invariants', 'invariant')
    result = dct['invariants']
    assert len(result) == 2
    assert result[0]() == (False, ("local",))
    assert result[1]() == (True, ())

def test_store_invariants_raises_typeerror_for_non_callable():
    class Base:
        invariant = "not callable"
    dct = {}
    try:
        store_invariants(dct, (Base,), 'invariants', 'invariant')
        assert False
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'

def test_store_invariants_handles_empty_invariants():
    class Base:
        pass
    dct = {}
    store_invariants(dct, (Base,), 'invariants', 'invariant')
    result = dct['invariants']
    assert result == ()

def test_store_invariants_wraps_invariant_returning_tuple_of_results():
    def inv():
        return [(True, ()), (False, ("err1",)), (True, ())]
    class Base:
        invariant = inv
    dct = {}
    store_invariants(dct, (Base,), 'invariants', 'invariant')
    wrapped = dct['invariants'][0]
    result = wrapped()
    assert result == (False, ("err1",))

def test_store_invariants_avoids_duplicate_inheritance():
    class GrandBase:
        def invariant():
            return True, ()
    class Parent1(GrandBase):
        pass
    class Parent2(GrandBase):
        pass
    class Child(Parent1, Parent2):
        pass
    dct = {}
    store_invariants(dct, (Child,), 'invariants', 'invariant')
    result = dct['invariants']
    assert len(result) == 1
    assert result[0]() == (True, ())

def test_store_invariants_preserves_wrapping_for_bool_result():
    def inv():
        return False, ("msg",)
    class Base:
        invariant = inv
    dct = {}
    store_invariants(dct, (Base,), 'invariants', 'invariant')
    wrapped = dct['invariants'][0]
    result = wrapped()
    assert result == (False, ("msg",))


# LLM-generated content at query #2
#--------------------------

def test_invariant_errors_no_errors():
    elem = 5
    invariants = [lambda x: (True, None), lambda x: (True, "ok")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_some_errors():
    elem = "test"
    invariants = [lambda x: (True, None), lambda x: (False, "error1"), lambda x: (False, "error2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["error1", "error2"]

def test_invariant_errors_all_errors():
    elem = []
    invariants = [lambda x: (False, "err1"), lambda x: (False, "err2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["err1", "err2"]

def test_invariant_errors_empty_invariants():
    elem = object()
    invariants = []
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_with_different_data_types():
    elem = 10
    invariants = [lambda x: (False, 123), lambda x: (False, {"key": "value"}), lambda x: (True, "skip")]
    result = _invariant_errors(elem, invariants)
    assert result == [123, {"key": "value"}]


# LLM-generated content at query #3
#--------------------------

def test_maybe_parse_user_type_with_preserved_type():
    result = maybe_parse_user_type(list)
    assert result == [list]

def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]

def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]

def test_maybe_parse_user_type_with_iterable_of_types():
    result = maybe_parse_user_type((int, str))
    assert result == (int, str)

def test_maybe_parse_user_type_with_nested_iterable():
    result = maybe_parse_user_type(([int, str], float))
    assert result == (int, str, float)

def test_maybe_parse_user_type_with_empty_iterable():
    result = maybe_parse_user_type([])
    assert result == ()

def test_maybe_parse_user_type_raises_type_error_for_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False
    except TypeError:
        assert True

def test_maybe_parse_user_type_with_tuple_of_strings():
    result = maybe_parse_user_type(("int", "str"))
    assert result == ("int", "str")

def test_maybe_parse_user_type_with_list_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)

def test_maybe_parse_user_type_with_complex_nested_iterable():
    result = maybe_parse_user_type([(int, str), [float, bool]])
    assert result == (int, str, float, bool)


# LLM-generated content at query #4
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_correct_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_constructor_with_incorrect_key_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.0})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_incorrect_value_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'a'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_with_invariant_violation_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.0})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_constructor_with_checked_key_type():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key = KeyType()
    value = ValueType()
    result = CustomMap({key: value})
    assert isinstance(result, CustomMap)
    assert result[key] == value

def test_constructor_with_checked_value_type():
    class KeyType:
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key = KeyType()
    value = ValueType()
    result = CustomMap({key: value})
    assert isinstance(result, CustomMap)
    assert result[key] == value

def test_constructor_with_checked_types_and_create():
    class CheckedKey(CheckedType):
        @classmethod
        def create(cls, data):
            return cls()
    class CheckedValue(CheckedType):
        @classmethod
        def create(cls, data):
            return cls()
    class CustomMap(CheckedPMap):
        __key_type__ = CheckedKey
        __value_type__ = CheckedValue
    result = CustomMap.create({'raw_key': 'raw_value'})
    assert isinstance(result, CustomMap)
    assert all(isinstance(k, CheckedKey) for k in result.keys())
    assert all(isinstance(v, CheckedValue) for v in result.values())

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_self_as_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result is original

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert str(result) == "IntToFloatMap({1: 1.5})"


# LLM-generated content at query #5
#--------------------------

def test_predicate_at_line_18_evaluates_to_true():
    from collections.abc import Iterable
    class _preserved_iterable_types:
        pass
    class CustomType:
        pass
    class CustomIterableType(Iterable):
        def __iter__(self):
            return iter([])
    t = CustomType
    is_type = isinstance(t, type)
    is_iterable = isinstance(t, Iterable)
    result = is_type and not is_iterable
    assert result == True


# LLM-generated content at query #6
#--------------------------

def test__checked_type_create_with_instance_of_cls():
    class MockCls:
        _checked_types = []
    instance = MockCls()
    result = _checked_type_create(MockCls, instance)
    assert result is instance

def test__checked_type_create_without_checked_types():
    class MockCls:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCls, source_data)
    assert isinstance(result, MockCls)
    assert result.data == source_data

def test__checked_type_create_with_checked_type_and_mismatched_data():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
        def __init__(self, data):
            self.data = data
    class MockCls:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    source_data = [5, 6]
    result = _checked_type_create(MockCls, source_data)
    assert isinstance(result, MockCls)
    assert all(isinstance(item, CheckedType) for item in result.data)
    assert [item.data for item in result.data] == source_data

def test__checked_type_create_with_checked_type_and_matching_data():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
        def __init__(self, data):
            self.data = data
    class MockCls:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    checked_instance = CheckedType(10)
    source_data = [checked_instance]
    result = _checked_type_create(MockCls, source_data)
    assert isinstance(result, MockCls)
    assert result.data == source_data
    assert result.data[0] is checked_instance

def test__checked_type_create_with_ignore_extra_true():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data, ignore_extra)
        def __init__(self, data, ignore_extra):
            self.data = data
            self.ignore_extra = ignore_extra
    class MockCls:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    source_data = [7]
    result = _checked_type_create(MockCls, source_data, ignore_extra=True)
    assert isinstance(result, MockCls)
    assert isinstance(result.data[0], CheckedType)
    assert result.data[0].data == 7
    assert result.data[0].ignore_extra == True


# LLM-generated content at query #7
#--------------------------

def test_wrap_invariant_with_single_bool_result():
    def invariant_single():
        return True, "data"
    wrapped = wrap_invariant(invariant_single)
    result = wrapped()
    assert result == (True, "data")

def test_wrap_invariant_with_single_false_bool_result():
    def invariant_single():
        return False, "error"
    wrapped = wrap_invariant(invariant_single)
    result = wrapped()
    assert result == (False, "error")

def test_wrap_invariant_with_multiple_results():
    def invariant_multiple():
        return [(True, "ok1"), (False, "err1"), (True, "ok2"), (False, "err2")]
    wrapped = wrap_invariant(invariant_multiple)
    result = wrapped()
    assert result[0] == False
    assert result[1] == ("err1", "err2")

def test_wrap_invariant_with_all_true_multiple_results():
    def invariant_all_true():
        return [(True, "ok1"), (True, "ok2")]
    wrapped = wrap_invariant(invariant_all_true)
    result = wrapped()
    assert result == (True, ("ok1", "ok2"))

def test_wrap_invariant_with_args_and_kwargs():
    def invariant_with_args(a, b, c=3):
        if a + b + c > 10:
            return False, "sum too large"
        return True, "acceptable"
    wrapped = wrap_invariant(invariant_with_args)
    result = wrapped(5, 4, c=2)
    assert result == (False, "sum too large")
    result2 = wrapped(1, 2, c=3)
    assert result2 == (True, "acceptable")

def test_wrap_invariant_with_empty_multiple_results():
    def invariant_empty():
        return []
    wrapped = wrap_invariant(invariant_empty)
    result = wrapped()
    assert result == (True, ())


# LLM-generated content at query #8
#--------------------------

def test_check_types_with_valid_types():
    expected_types = [int, str]
    source_class = list
    it = [1, "hello", 2, "world"]
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_type():
    expected_types = [int]
    source_class = list
    it = [1, "invalid", 3]
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == (int,)
        assert e.actual_type == str

def test_check_types_with_no_expected_types():
    expected_types = None
    source_class = list
    it = [1, "hello", 3.14]
    _check_types(it, expected_types, source_class)

def test_check_types_with_empty_iterable():
    expected_types = [int, str]
    source_class = list
    it = []
    _check_types(it, expected_types, source_class)

def test_check_types_with_custom_exception_type():
    expected_types = [int]
    source_class = list
    it = ["invalid"]
    exception_type = ValueError
    try:
        _check_types(it, expected_types, source_class, exception_type)
        assert False
    except ValueError as e:
        assert isinstance(e, ValueError)

def test_check_types_with_type_string():
    expected_types = ["builtins.int", "builtins.str"]
    source_class = list
    it = [1, "hello"]
    _check_types(it, expected_types, source_class)

def test_check_types_with_mixed_type_and_string():
    expected_types = [int, "builtins.str"]
    source_class = list
    it = [1, "hello"]
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_type_string():
    expected_types = ["builtins.int"]
    source_class = list
    it = ["invalid"]
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == (int,)
        assert e.actual_type == str


# LLM-generated content at query #9
#--------------------------

def test_constructor_with_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector()
    assert len(result) == 0
    assert isinstance(result, TestVector)


def test_constructor_with_list_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    result = TestVector([1, 2, 3])
    assert list(result) == [1, 2, 3]
    assert isinstance(result, TestVector)


def test_constructor_with_python_pvector_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    pvec = python_pvector([4, 5, 6])
    result = TestVector(pvec)
    assert list(result) == [4, 5, 6]
    assert isinstance(result, TestVector)


def test_constructor_type_check_failure():
    class TestVector(CheckedPVector):
        __type__ = int
    try:
        TestVector([1, 'a', 3])
        assert False
    except TypeError:
        assert True


def test_constructor_invariant_check_failure():
    class Positives(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    try:
        Positives([1, -2, 3])
        assert False
    except InvariantException:
        assert True


def test_constructor_invariant_check_success():
    class Positives(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    result = Positives([1, 2, 3])
    assert list(result) == [1, 2, 3]
    assert isinstance(result, Positives)


def test_constructor_with_multiple_types():
    class TestVector(CheckedPVector):
        __type__ = (int, float)
    result = TestVector([1, 2.5, 3])
    assert list(result) == [1, 2.5, 3]
    assert isinstance(result, TestVector)


def test_constructor_with_no_type_specified():
    class TestVector(CheckedPVector):
        pass
    result = TestVector([1, 'a', 3.5])
    assert list(result) == [1, 'a', 3.5]
    assert isinstance(result, TestVector)


# LLM-generated content at query #10
#--------------------------

def test_check_types_with_expected_types_and_matching_element():
    class MockType:
        pass
    class MockSourceClass:
        __name__ = "MockSourceClass"
    class MockExceptionType(Exception):
        pass
    it = [MockType()]
    expected_types = [MockType]
    source_class = MockSourceClass
    exception_type = MockExceptionType
    _check_types(it, expected_types, source_class, exception_type)


# LLM-generated content at query #11
#--------------------------

def test_check_types_with_valid_types():
    expected_types = [int, str]
    source_class = list
    it = [1, "hello", 2, "world"]
    _check_types(it, expected_types, source_class)

def test_check_types_with_invalid_type():
    expected_types = [int, str]
    source_class = list
    it = [1, "hello", 2.5]
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == (int, str)
        assert e.actual_type == float
        assert e.value == 2.5

def test_check_types_with_type_string():
    expected_types = ["builtins.int", "builtins.str"]
    source_class = list
    it = [1, "hello"]
    _check_types(it, expected_types, source_class)

def test_check_types_with_type_string_invalid():
    expected_types = ["builtins.int", "builtins.str"]
    source_class = list
    it = [1, "hello", []]
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == (int, str)
        assert e.actual_type == list
        assert e.value == []

def test_check_types_with_empty_expected_types():
    expected_types = []
    source_class = list
    it = [1, "hello", []]
    _check_types(it, expected_types, source_class)

def test_check_types_with_custom_exception_type():
    class CustomError(Exception):
        pass
    expected_types = [int]
    source_class = list
    it = ["not an int"]
    try:
        _check_types(it, expected_types, source_class, CustomError)
        assert False
    except CustomError:
        assert True

def test_check_types_with_single_expected_type():
    expected_types = [int]
    source_class = list
    it = [1, 2, 3]
    _check_types(it, expected_types, source_class)

def test_check_types_with_single_expected_type_invalid():
    expected_types = [int]
    source_class = list
    it = [1, 2, "three"]
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == (int,)
        assert e.actual_type == str
        assert e.value == "three"

def test_check_types_with_none_values():
    expected_types = [int, type(None)]
    source_class = list
    it = [1, None, 2]
    _check_types(it, expected_types, source_class)

def test_check_types_with_none_values_invalid():
    expected_types = [int, type(None)]
    source_class = list
    it = [1, None, "invalid"]
    try:
        _check_types(it, expected_types, source_class)
        assert False
    except CheckedValueTypeError as e:
        assert e.source_class == list
        assert e.expected_types == (int, type(None))
        assert e.actual_type == str
        assert e.value == "invalid"


# LLM-generated content at query #12
#--------------------------

def test_store_invariants_with_no_inheritance():
    class Base:
        pass
    dct = {}
    store_invariants(dct, (Base,), 'invariants', 'invariant')
    assert dct['invariants'] == ()

def test_store_invariants_with_single_invariant():
    class Base:
        def invariant(self):
            return True, ()
    dct = {}
    store_invariants(dct, (Base,), 'invariants', 'invariant')
    assert len(dct['invariants']) == 1
    assert dct['invariants'][0](None) == (True, ())

def test_store_invariants_with_multiple_inheritance():
    class A:
        def invariant(self):
            return True, ()
    class B:
        def invariant(self):
            return False, ('error',)
    class C(A, B):
        pass
    dct = {}
    store_invariants(dct, (C,), 'invariants', 'invariant')
    assert len(dct['invariants']) == 2
    results = [inv(None) for inv in dct['invariants']]
    assert (True, ()) in results
    assert (False, ('error',)) in results

def test_store_invariants_with_diamond_inheritance():
    class Base:
        def invariant(self):
            return True, ()
    class A(Base):
        pass
    class B(Base):
        pass
    class C(A, B):
        pass
    dct = {}
    store_invariants(dct, (C,), 'invariants', 'invariant')
    assert len(dct['invariants']) == 1
    assert dct['invariants'][0](None) == (True, ())

def test_store_invariants_with_local_override():
    class Base:
        def invariant(self):
            return True, ()
    dct = {'invariant': lambda self: (False, ('local',))}
    store_invariants(dct, (Base,), 'invariants', 'invariant')
    assert len(dct['invariants']) == 1
    assert dct['invariants'][0](None) == (False, ('local',))

def test_store_invariants_with_multiple_invariants_in_chain():
    class Base:
        def invariant(self):
            return True, ()
    class Derived(Base):
        def invariant(self):
            return False, ('derived',)
    dct = {}
    store_invariants(dct, (Derived,), 'invariants', 'invariant')
    assert len(dct['invariants']) == 2
    results = [inv(None) for inv in dct['invariants']]
    assert (True, ()) in results
    assert (False, ('derived',)) in results

def test_store_invariants_with_non_callable_raises_typeerror():
    class Base:
        invariant = 'not a function'
    dct = {}
    try:
        store_invariants(dct, (Base,), 'invariants', 'invariant')
        assert False
    except TypeError:
        assert True

def test_store_invariants_with_wrapped_invariant_returning_list():
    def invariant_returning_list(self):
        return [(True, ()), (False, ('a',)), (True, ())]
    dct = {'invariant': invariant_returning_list}
    store_invariants(dct, (), 'invariants', 'invariant')
    result = dct['invariants'][0](None)
    assert result == (False, ('a',))

def test_store_invariants_with_wrapped_invariant_returning_single_bool():
    def invariant_returning_bool(self):
        return False, ('single',)
    dct = {'invariant': invariant_returning_bool}
    store_invariants(dct, (), 'invariants', 'invariant')
    result = dct['invariants'][0](None)
    assert result == (False, ('single',))

def test_store_invariants_destination_name_different():
    class Base:
        def check(self):
            return True, ()
    dct = {}
    store_invariants(dct, (Base,), 'checks', 'check')
    assert 'checks' in dct
    assert len(dct['checks']) == 1
    assert dct['checks'][0](None) == (True, ())


# LLM-generated content at query #13
#--------------------------

def test_checked_type_create_with_checked_type_and_mismatched_data():
    class MockCheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}"

    class MockTypes:
        _checked_types = [MockCheckedType]

    def get_types(types):
        return types

    types = get_types(MockTypes._checked_types)
    checked_type = next((t for t in types if issubclass(t, MockCheckedType)), None)
    source_data = ["data1", "data2"]
    result = MockTypes([checked_type.create(data, ignore_extra=False) if not any(isinstance(data, t) for t in types) else data for data in source_data])
    assert result == ["created_data1", "created_data2"]


# LLM-generated content at query #14
#--------------------------

def test_predicate_at_line_11_evaluates_to_false():
    class MockDict:
        def __init__(self, data):
            self.data = data
        def __getitem__(self, key):
            return self.data[key]
        def __setitem__(self, key, value):
            pass
    def _all_dicts(bases):
        return bases
    def wrap_invariant(inv):
        return inv
    dct = MockDict({})
    bases = []
    destination_name = "dest"
    source_name = "source"
    invariants = []
    result = not all(callable(invariant) for invariant in invariants)
    assert result == False


# LLM-generated content at query #15
#--------------------------

def test_store_invariants_adds_tuple_of_wrapped_invariants():
    from unittest.mock import Mock, patch
    mock_invariant1 = Mock()
    mock_invariant2 = Mock()
    mock_wrap = Mock(side_effect=lambda x: f'wrapped_{x}')
    dct = {}
    bases = []
    with patch('_all_dicts', return_value=[]):
        with patch('wrap_invariant', mock_wrap):
            store_invariants(dct, bases, 'dest', 'src')
    assert dct['dest'] == ('wrapped_mock_invariant1',)
    assert mock_wrap.call_count == 1

def test_store_invariants_inherits_from_bases():
    from unittest.mock import Mock, patch
    mock_invariant1 = Mock()
    mock_invariant2 = Mock()
    mock_wrap = Mock(side_effect=lambda x: f'wrapped_{x}')
    dct = {'src': mock_invariant1}
    base_dict = {'src': mock_invariant2}
    bases = [Mock()]
    with patch('_all_dicts', return_value=[base_dict]):
        with patch('wrap_invariant', mock_wrap):
            store_invariants(dct, bases, 'dest', 'src')
    assert len(dct['dest']) == 2
    assert dct['dest'][0] == 'wrapped_mock_invariant1'
    assert dct['dest'][1] == 'wrapped_mock_invariant2'
    assert mock_wrap.call_count == 2

def test_store_invariants_raises_on_non_callable():
    from unittest.mock import Mock, patch
    dct = {'src': 'not_callable'}
    bases = []
    with patch('_all_dicts', return_value=[]):
        try:
            store_invariants(dct, bases, 'dest', 'src')
            assert False
        except TypeError as e:
            assert str(e) == 'Invariants must be callable'

def test_store_invariants_skips_missing_keys():
    from unittest.mock import Mock, patch
    mock_invariant = Mock()
    mock_wrap = Mock(side_effect=lambda x: f'wrapped_{x}')
    dct = {}
    base_dict = {'src': mock_invariant}
    bases = [Mock()]
    with patch('_all_dicts', return_value=[base_dict]):
        with patch('wrap_invariant', mock_wrap):
            store_invariants(dct, bases, 'dest', 'src')
    assert dct['dest'] == ('wrapped_mock_invariant',)
    assert mock_wrap.call_count == 1

def test_store_invariants_empty_result_when_no_sources():
    from unittest.mock import patch
    dct = {}
    bases = []
    with patch('_all_dicts', return_value=[]):
        store_invariants(dct, bases, 'dest', 'src')
    assert dct['dest'] == ()


# LLM-generated content at query #16
#--------------------------

def test_store_invariants_adds_destination():
    class Base:
        def base_inv(self):
            return True, ()
    class Test:
        def test_inv(self):
            return True, ()
    dct = {}
    store_invariants(dct, (Base, Test), 'invariants', 'inv')
    assert 'invariants' in dct
    assert isinstance(dct['invariants'], tuple)
    assert len(dct['invariants']) == 0

def test_store_invariants_collects_from_dct():
    def inv1(self):
        return True, ()
    dct = {'inv': inv1}
    store_invariants(dct, (), 'invariants', 'inv')
    assert len(dct['invariants']) == 1
    assert dct['invariants'][0] is inv1

def test_store_invariants_collects_from_base_classes():
    class Base:
        def inv(self):
            return True, ()
    dct = {}
    store_invariants(dct, (Base,), 'invariants', 'inv')
    assert len(dct['invariants']) == 1

def test_store_invariants_collects_from_multiple_bases():
    class Base1:
        def inv(self):
            return True, ()
    class Base2:
        def inv(self):
            return True, ()
    dct = {}
    store_invariants(dct, (Base1, Base2), 'invariants', 'inv')
    assert len(dct['invariants']) == 2

def test_store_invariants_inherits_from_base_hierarchy():
    class GrandBase:
        def inv(self):
            return True, ()
    class Base(GrandBase):
        pass
    dct = {}
    store_invariants(dct, (Base,), 'invariants', 'inv')
    assert len(dct['invariants']) == 1

def test_store_invariants_avoids_duplicates():
    class Base:
        def inv(self):
            return True, ()
    class Derived(Base):
        pass
    dct = {}
    store_invariants(dct, (Base, Derived), 'invariants', 'inv')
    assert len(dct['invariants']) == 1

def test_store_invariants_combines_dct_and_bases():
    class Base:
        def inv(self):
            return True, ()
    def local_inv(self):
        return True, ()
    dct = {'inv': local_inv}
    store_invariants(dct, (Base,), 'invariants', 'inv')
    assert len(dct['invariants']) == 2

def test_store_invariants_wraps_invariants():
    def invariant(self):
        return [(True, ()), (False, ('error',))]
    dct = {'inv': invariant}
    store_invariants(dct, (), 'invariants', 'inv')
    wrapped = dct['invariants'][0]
    result = wrapped(None)
    assert result == (False, ('error',))

def test_store_invariants_raises_on_non_callable():
    dct = {'inv': 'not a function'}
    try:
        store_invariants(dct, (), 'invariants', 'inv')
        assert False
    except TypeError:
        pass

def test_store_invariants_raises_on_non_callable_in_base():
    class Base:
        inv = 'not a function'
    dct = {}
    try:
        store_invariants(dct, (Base,), 'invariants', 'inv')
        assert False
    except TypeError:
        pass


# LLM-generated content at query #17
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({})
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_valid_key_value_pairs():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_invalid_key_type_raises_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_with_invalid_value_type_raises_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'invalid'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_with_invariant_violation_raises_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert result == {1: 1.0}
    assert isinstance(result, IntToFloatMap)

def test_constructor_with_checked_key_type_create():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap.create({KeyType(): ValueType()})
    assert isinstance(result, CustomMap)

def test_constructor_with_initial_as_same_type_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result == original
    assert result is not original

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert result == {1: 1.5}
    assert isinstance(result, IntToFloatMap)

def test_constructor_repr_output():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    instance = IntToFloatMap({1: 1.5})
    assert repr(instance) == "IntToFloatMap({1: 1.5})"


# LLM-generated content at query #18
#--------------------------

def test_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_valid_key_value_types():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_invalid_key_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'invalid': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_invalid_value_type_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'invalid'})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_with_invariant_violation_raises():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_multiple_invariants_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_with_checked_key_type():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key = KeyType()
    value = ValueType()
    result = CustomMap({key: value})
    assert isinstance(result, CustomMap)
    assert result[key] == value

def test_constructor_with_initial_as_same_class_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result is not original

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_invalid_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=5)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_undefined_size_constant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25}, _UNDEFINED_CHECKED_PMAP_SIZE)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_repr_output():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_str_output():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert str(result) == "IntToFloatMap({1: 1.5})"


# LLM-generated content at query #19
#--------------------------

def test_checked_pmap_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_checked_pmap_constructor_type_check_failure_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {'a': 1.5}
    try:
        IntToFloatMap(initial)
        assert False
    except CheckedKeyTypeError:
        assert True

def test_checked_pmap_constructor_type_check_failure_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 'string'}
    try:
        IntToFloatMap(initial)
        assert False
    except CheckedTypeError:
        assert True

def test_checked_pmap_constructor_invariant_failure():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    initial = {1: 2.5}
    try:
        IntToFloatMap(initial)
        assert False
    except InvariantException:
        assert True

def test_checked_pmap_constructor_invariant_success():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    initial = {1: 1.0, 2: 2.0}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_checked_pmap_constructor_with_checked_type_source():
    class InnerChecked(CheckedType):
        pass
    class OuterMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = InnerChecked
    source = {'a': InnerChecked()}
    result = OuterMap(source)
    assert isinstance(result, OuterMap)
    assert result['a'] == source['a']

def test_checked_pmap_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    expected_repr = "IntToFloatMap({1: 1.5})"
    assert repr(result) == expected_repr
    assert str(result) == expected_repr

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5

def test_checked_pmap_constructor_passing_checked_pmap_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result is original


# LLM-generated content at query #20
#--------------------------

def test_invariant_errors_returns_empty_list_when_all_invariants_valid():
    elem = "test_element"
    invariants = [lambda x: (True, "ok1"), lambda x: (True, "ok2")]
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_returns_data_from_invalid_invariants():
    elem = 5
    invariants = [lambda x: (True, "ok"), lambda x: (False, "error1"), lambda x: (False, "error2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["error1", "error2"]

def test_invariant_errors_returns_empty_list_for_empty_invariants():
    elem = None
    invariants = []
    result = _invariant_errors(elem, invariants)
    assert result == []

def test_invariant_errors_passes_elem_to_each_invariant():
    captured = []
    def capturing_invariant(e):
        captured.append(e)
        return (True, None)
    elem = object()
    invariants = [capturing_invariant, capturing_invariant]
    _invariant_errors(elem, invariants)
    assert captured == [elem, elem]

def test_invariant_errors_ignores_data_from_valid_invariants():
    elem = []
    invariants = [lambda x: (True, "ignore1"), lambda x: (False, "include1"), lambda x: (True, "ignore2")]
    result = _invariant_errors(elem, invariants)
    assert result == ["include1"]


# LLM-generated content at query #21
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_with_checkedpmap_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 1
    assert result[1] == 1.5

def test_constructor_type_check_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({"a": 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_constructor_type_check_value():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: "string"})
        assert False
    except CheckedTypeError:
        assert True

def test_constructor_invariant_check():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_constructor_invariant_check_success():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0

def test_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    try:
        IntToFloatMap({-1: 1.0})
        assert False
    except InvariantException as e:
        assert len(e.error_codes) == 1
        assert e.error_codes[0] == 'Key negative'

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({}, size=0)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_size_and_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 1
    assert result[1] == 1.5

def test_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert repr(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_str():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5})
    assert str(result) == "IntToFloatMap({1: 1.5})"

def test_constructor_with_checked_key_type():
    class KeyMap(CheckedPMap):
        __key_type__ = CheckedType
        __value_type__ = int
    class MyCheckedType(CheckedType):
        pass
    my_instance = MyCheckedType()
    result = KeyMap({my_instance: 5})
    assert isinstance(result, KeyMap)
    assert result[my_instance] == 5

def test_constructor_with_checked_value_type():
    class ValueMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = CheckedType
    class MyCheckedType(CheckedType):
        pass
    my_instance = MyCheckedType()
    result = ValueMap({1: my_instance})
    assert isinstance(result, ValueMap)
    assert result[1] is my_instance


# LLM-generated content at query #22
#--------------------------

def test_constructor_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_constructor_respects_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({"a": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

def test_constructor_respects_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: "string"})
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_constructor_enforces_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass

def test_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=1)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5

def test_constructor_with_checked_type_key():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key_instance = KeyType()
    value_instance = ValueType()
    result = CustomMap({key_instance: value_instance})
    assert isinstance(result, CustomMap)
    assert result[key_instance] == value_instance

def test_constructor_with_checked_type_creation():
    class KeyType(CheckedType):
        @classmethod
        def create(cls, data):
            return cls()
    class ValueType(CheckedType):
        @classmethod
        def create(cls, data):
            return cls()
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    result = CustomMap({"raw_key": "raw_value"})
    assert isinstance(result, CustomMap)
    assert all(isinstance(k, KeyType) for k in result.keys())
    assert all(isinstance(v, ValueType) for v in result.values())

def test_constructor_returns_same_instance_if_already_checked_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result is original


# LLM-generated content at query #23
#--------------------------

def test_store_types_single_dict():
    dct = {"source": int}
    _store_types(dct, [], "destination", "source")
    assert dct["destination"] == (int,)

def test_store_types_multiple_dicts():
    dct = {"source": str}
    base1 = type("Base1", (), {"source": float})
    base2 = type("Base2", (), {})
    _store_types(dct, [base1, base2], "destination", "source")
    assert dct["destination"] == (str, float)

def test_store_types_with_iterable_source():
    dct = {"source": [int, str]}
    _store_types(dct, [], "destination", "source")
    assert dct["destination"] == (int, str)

def test_store_types_missing_source():
    dct = {}
    base = type("Base", (), {"source": bool})
    _store_types(dct, [base], "destination", "source")
    assert dct["destination"] == (bool,)

def test_store_types_no_source_anywhere():
    dct = {}
    _store_types(dct, [], "destination", "source")
    assert dct["destination"] == ()

def test_store_types_preserved_iterable_type():
    dct = {"source": list}
    _store_types(dct, [], "destination", "source")
    assert dct["destination"] == (list,)

def test_store_types_string_type():
    dct = {"source": "CustomType"}
    _store_types(dct, [], "destination", "source")
    assert dct["destination"] == ("CustomType",)

def test_store_types_nested_iterables():
    dct = {"source": [[int, str], float]}
    _store_types(dct, [], "destination", "source")
    assert dct["destination"] == (int, str, float)

def test_store_types_duplicate_sources():
    dct = {"source": int}
    base = type("Base", (), {"source": int})
    _store_types(dct, [base], "destination", "source")
    assert dct["destination"] == (int, int)

def test_store_types_mixed_types():
    dct = {"source": (int, "str")}
    _store_types(dct, [], "destination", "source")
    assert dct["destination"] == (int, "str")


# LLM-generated content at query #24
#--------------------------

def test__restore_pickle():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields):
            return (cls, data, _factory_fields)
    data = {"key": "value"}
    result = MockClass._restore_pickle(data)
    expected = (MockClass, data, set())
    assert result == expected


# LLM-generated content at query #25
#--------------------------

def test_checked_type_create_predicate_false():
    class MockCheckedType:
        pass
    class MockClass:
        _checked_types = []
    cls = MockClass
    source_data = []
    result = cls._checked_type_create(cls, source_data)
    assert result is not None


# LLM-generated content at query #26
#--------------------------

def test___new___creates_checked_pmap_from_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test___new___creates_empty_checked_pmap():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap()
    assert result == {}
    assert isinstance(result, IntToFloatMap)

def test___new___creates_checked_pmap_with_size():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)

def test___new___enforces_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        pass

def test___new___enforces_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'a'})
        assert False
    except CheckedTypeError:
        pass

def test___new___enforces_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        pass

def test___new___accepts_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0})
    assert result == {1: 1.0}
    assert isinstance(result, IntToFloatMap)

def test___new___creates_from_checked_pmap_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result == {1: 1.5}
    assert result is original

def test___new___creates_from_iterable_of_pairs():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap([(1, 1.5), (2, 2.25)])
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, IntToFloatMap)


# LLM-generated content at query #27
#--------------------------

def test_checked_pmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0

def test_checked_pmap_constructor_with_valid_key_value_pairs():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({'a': 1.5})
        assert False
    except CheckedKeyTypeError:
        assert True

def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    try:
        IntToFloatMap({1: 'invalid'})
        assert False
    except CheckedTypeError:
        assert True

def test_checked_pmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    try:
        IntToFloatMap({1: 2.5})
        assert False
    except InvariantException:
        assert True

def test_checked_pmap_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0

def test_checked_pmap_constructor_with_multiple_invariants():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_checked_pmap_constructor_with_multiple_invariants_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = [lambda k, v: (k >= 0, 'Key negative'), lambda k, v: (v >= 0, 'Value negative')]
    try:
        IntToFloatMap({-1: 1.5})
        assert False
    except InvariantException as e:
        assert 'Key negative' in str(e.error_codes)

def test_checked_pmap_constructor_with_checked_type_key():
    class KeyType(CheckedType):
        pass
    class ValueType(CheckedType):
        pass
    class CustomMap(CheckedPMap):
        __key_type__ = KeyType
        __value_type__ = ValueType
    key_instance = KeyType()
    value_instance = ValueType()
    result = CustomMap({key_instance: value_instance})
    assert isinstance(result, CustomMap)
    assert result[key_instance] == value_instance

def test_checked_pmap_constructor_with_initial_as_same_type_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert result is original

def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25

def test_checked_pmap_constructor_with_undefined_size_constant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    result = IntToFloatMap({1: 1.5}, size=_UNDEFINED_CHECKED_PMAP_SIZE)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5


# LLM-generated content at query #28
#--------------------------

def test_merge_invariant_results_all_true():
    result = [(True, []), (True, []), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_one_false():
    result = [(True, []), (False, ["error1"]), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)

def test_merge_invariant_results_multiple_false():
    result = [(False, ["error1"]), (True, []), (False, ["error2", "error3"])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2", "error3")

def test_merge_invariant_results_all_false():
    result = [(False, ["a"]), (False, ["b"]), (False, ["c"])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("a", "b", "c")

def test_merge_invariant_results_empty_input():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_single_true():
    result = [(True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_single_false():
    result = [(False, ["only error"])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("only error",)

def test_merge_invariant_results_false_with_empty_data():
    result = [(False, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ()

def test_merge_invariant_results_mixed_data_types():
    result = [(True, []), (False, [1, 2]), (False, ["text"])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == (1, 2, "text")


# LLM-generated content at query #29
#--------------------------

def test_merge_invariant_results_all_true():
    result = [(True, []), (True, []), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_one_false():
    result = [(True, []), (False, "error1"), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)

def test_merge_invariant_results_multiple_false():
    result = [(False, "error1"), (False, "error2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2", "error3")

def test_merge_invariant_results_mixed_false():
    result = [(True, []), (False, "error1"), (False, "error2"), (True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2")

def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_single_true():
    result = [(True, [])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_single_false():
    result = [(False, "error")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error",)

def test_merge_invariant_results_data_types():
    result = [(True, []), (False, 123), (False, {"key": "value"})]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == (123, {"key": "value"})


# LLM-generated content at query #30
#--------------------------

def test__restore_pickle():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields):
            return data, _factory_fields

    data = "test_data"
    result = MockClass._restore_pickle(data)
    expected = ("test_data", set())
    assert result == expected


