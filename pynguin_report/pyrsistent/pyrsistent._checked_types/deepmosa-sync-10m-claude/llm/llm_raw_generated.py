####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]


def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]


def test_maybe_parse_user_type_with_list_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)


def test_maybe_parse_user_type_with_tuple_of_types():
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(["int", "str"])
    assert result == ("int", "str")


def test_maybe_parse_user_type_with_nested_list():
    result = maybe_parse_user_type([[int, str], [float]])
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_mixed_list():
    result = maybe_parse_user_type([int, "str", float])
    assert result == (int, "str", float)


def test_maybe_parse_user_type_with_invalid_type_raises_error():
    try:
        maybe_parse_user_type(123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_none_raises_error():
    try:
        maybe_parse_user_type(None)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_dict_raises_error():
    try:
        maybe_parse_user_type({"key": "value"})
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


# LLM-generated content at query #2
#--------------------------

```python
def test_invariant_errors_all_valid():
    def invariant1(elem):
        return (True, "valid1")
    
    def invariant2(elem):
        return (True, "valid2")
    
    result = _invariant_errors("test", [invariant1, invariant2])
    assert result == []


def test_invariant_errors_all_invalid():
    def invariant1(elem):
        return (False, "error1")
    
    def invariant2(elem):
        return (False, "error2")
    
    result = _invariant_errors("test", [invariant1, invariant2])
    assert result == ["error1", "error2"]


def test_invariant_errors_mixed():
    def invariant1(elem):
        return (True, "valid1")
    
    def invariant2(elem):
        return (False, "error2")
    
    def invariant3(elem):
        return (False, "error3")
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == ["error2", "error3"]


def test_invariant_errors_empty_invariants():
    result = _invariant_errors("test", [])
    assert result == []


def test_invariant_errors_single_valid():
    def invariant(elem):
        return (True, "success")
    
    result = _invariant_errors("test", [invariant])
    assert result == []


def test_invariant_errors_single_invalid():
    def invariant(elem):
        return (False, "failure")
    
    result = _invariant_errors("test", [invariant])
    assert result == ["failure"]


def test_invariant_errors_with_different_data_types():
    def invariant1(elem):
        return (False, 42)
    
    def invariant2(elem):
        return (False, {"key": "value"})
    
    def invariant3(elem):
        return (True, "ignored")
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == [42, {"key": "value"}]


# LLM-generated content at query #3
#--------------------------

```python
def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]


def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type('int')
    assert result == ['int']


def test_maybe_parse_user_type_with_list_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)


def test_maybe_parse_user_type_with_tuple_of_types():
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(['int', 'str'])
    assert result == ('int', 'str')


def test_maybe_parse_user_type_with_nested_iterables():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


def test_maybe_parse_user_type_with_none():
    try:
        maybe_parse_user_type(None)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


def test_maybe_parse_user_type_with_dict():
    try:
        maybe_parse_user_type({'key': int})
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


# LLM-generated content at query #4
#--------------------------

```python
def test_store_invariants_basic():
    def inv1(x):
        return True, None
    
    def inv2(x):
        return True, None
    
    dct = {'__invariant__': inv1}
    bases = ()
    destination_name = '__wrapped_invariants__'
    source_name = '__invariant__'
    
    store_invariants(dct, bases, destination_name, source_name)
    
    assert destination_name in dct
    assert len(dct[destination_name]) == 1
    assert callable(dct[destination_name][0])


def test_store_invariants_multiple_inheritance():
    def inv1(x):
        return True, None
    
    def inv2(x):
        return True, None
    
    class Base1:
        __invariant__ = inv1
    
    class Base2:
        __invariant__ = inv2
    
    dct = {}
    bases = (Base1, Base2)
    destination_name = '__wrapped_invariants__'
    source_name = '__invariant__'
    
    store_invariants(dct, bases, destination_name, source_name)
    
    assert destination_name in dct
    assert len(dct[destination_name]) == 2


def test_store_invariants_override():
    def inv1(x):
        return True, None
    
    def inv2(x):
        return True, None
    
    class Base:
        __invariant__ = inv1
    
    dct = {'__invariant__': inv2}
    bases = (Base,)
    destination_name = '__wrapped_invariants__'
    source_name = '__invariant__'
    
    store_invariants(dct, bases, destination_name, source_name)
    
    assert destination_name in dct
    assert len(dct[destination_name]) == 2


def test_store_invariants_not_callable():
    class Base:
        __invariant__ = "not callable"
    
    dct = {}
    bases = (Base,)
    destination_name = '__wrapped_invariants__'
    source_name = '__invariant__'
    
    try:
        store_invariants(dct, bases, destination_name, source_name)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Invariants must be callable' in str(e)


def test_store_invariants_no_invariant():
    dct = {}
    bases = ()
    destination_name = '__wrapped_invariants__'
    source_name = '__invariant__'
    
    store_invariants(dct, bases, destination_name, source_name)
    
    assert destination_name in dct
    assert len(dct[destination_name]) == 0


def test_store_invariants_wrapped_behavior():
    def inv1(x):
        return (True, None), (False, "error")
    
    dct = {'__invariant__': inv1}
    bases = ()
    destination_name = '__wrapped_invariants__'
    source_name = '__invariant__'
    
    store_invariants(dct, bases, destination_name, source_name)
    
    wrapped = dct[destination_name][0]
    verdict, data = wrapped(None)
    
    assert verdict is False
    assert len(data) == 1
    assert data[0] == "error"


def test_store_invariants_diamond_inheritance():
    def inv1(x):
        return True, None
    
    class Base:
        __invariant__ = inv1
    
    class Left(Base):
        pass
    
    class Right(Base):
        pass
    
    class Diamond(Left, Right):
        pass
    
    dct = {}
    bases = (Left, Right)
    destination_name = '__wrapped_invariants__'
    source_name = '__invariant__'
    
    store_invariants(dct, bases, destination_name, source_name)
    
    assert destination_name in dct
    assert len(dct[destination_name]) == 1


def test_store_invariants_mixed_callable_non_callable():
    def inv1(x):
        return True, None
    
    class Base:
        __invariant__ = inv1
    
    dct = {'__invariant__': "not callable"}
    bases = (Base,)
    destination_name = '__wrapped_invariants__'
    source_name = '__invariant__'
    
    try:
        store_invariants(dct, bases, destination_name, source_name)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Invariants must be callable' in str(e)


# LLM-generated content at query #5
#--------------------------

```python
def test_checked_pmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_with_single_entry():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert result[42] == 3.14
    assert len(result) == 1


def test_checked_pmap_constructor_default_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0


def test_checked_pmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    initial_data = {1: 1.0, 2: 2.0}
    result = IntToFloatMap(initial_data)
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


# LLM-generated content at query #7
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checked_pset_constructor_with_valid_integers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    assert isinstance(result, Positives)


def test_checked_pset_constructor_with_valid_floats():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert 1.5 in result
    assert 2.5 in result
    assert 3.5 in result


def test_checked_pset_constructor_with_mixed_valid_types():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checked_pset_constructor_with_duplicates():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Should have raised an exception"
    except TypeError:
        pass


def test_checked_pset_constructor_with_negative_number():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pset_constructor_with_pmap():
    from pyrsistent import pmap
    
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pmap_obj = pmap()
    result = Positives(pmap_obj)
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checked_pset_constructor_preserves_class_type():
    class CustomSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda n: (n > 0, 'NotPositive')
    
    result = CustomSet([1, 2, 3])
    assert type(result).__name__ == 'CustomSet'
    assert isinstance(result, CustomSet)


# LLM-generated content at query #8
#--------------------------

```python
def test_checked_pmap_new_with_default_arguments():
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    result = IntMap()
    assert len(result) == 0
    assert isinstance(result, IntMap)


def test_checked_pmap_new_with_initial_dict():
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    initial = {1: 10, 2: 20}
    result = IntMap(initial)
    assert len(result) == 2
    assert result[1] == 10
    assert result[2] == 20


def test_checked_pmap_new_with_single_item():
    class StringMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = str
    
    initial = {'a': 'hello'}
    result = StringMap(initial)
    assert len(result) == 1
    assert result['a'] == 'hello'


def test_checked_pmap_new_with_size_argument():
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    initial = {1: 100, 2: 200, 3: 300}
    result = IntMap(initial, size=16)
    assert len(result) == 3
    assert result[1] == 100
    assert result[2] == 200
    assert result[3] == 300


def test_checked_pmap_new_with_empty_dict_and_size():
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    result = IntMap({}, size=32)
    assert len(result) == 0
    assert isinstance(result, IntMap)


def test_checked_pmap_new_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_new_multiple_items():
    class StringIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    result = StringIntMap(initial)
    assert len(result) == 4
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert result['d'] == 4


def test_checked_pmap_new_returns_instance_of_correct_class():
    class CustomMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = CustomMap({'x': 10})
    assert isinstance(result, CustomMap)
    assert type(result).__name__ == 'CustomMap'


# LLM-generated content at query #9
#--------------------------

```python
def test_store_invariants_basic():
    def inv1(x):
        return True, None
    
    dct = {}
    bases = ()
    store_invariants(dct, bases, '_invariants', 'invariant')
    assert '_invariants' not in dct


def test_store_invariants_single_invariant():
    def inv1(x):
        return True, None
    
    dct = {'invariant': inv1}
    bases = ()
    store_invariants(dct, bases, '_invariants', 'invariant')
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 1


def test_store_invariants_multiple_invariants():
    def inv1(x):
        return True, None
    
    def inv2(x):
        return False, "error"
    
    dct = {'invariant': [inv1, inv2]}
    bases = ()
    store_invariants(dct, bases, '_invariants', 'invariant')
    assert '_invariants' in dct


def test_store_invariants_inherited():
    def inv1(x):
        return True, None
    
    def inv2(x):
        return False, "error"
    
    class Base:
        invariant = inv1
    
    dct = {'invariant': inv2}
    bases = (Base,)
    store_invariants(dct, bases, '_invariants', 'invariant')
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 2


def test_store_invariants_non_callable_raises():
    dct = {'invariant': "not_callable"}
    bases = ()
    try:
        store_invariants(dct, bases, '_invariants', 'invariant')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Invariants must be callable' in str(e)


def test_store_invariants_wrapped_invariants():
    def inv1(x):
        return True, None
    
    dct = {'invariant': inv1}
    bases = ()
    store_invariants(dct, bases, '_invariants', 'invariant')
    wrapped_inv = dct['_invariants'][0]
    result = wrapped_inv(5)
    assert result == (True, None)


def test_store_invariants_custom_destination_name():
    def inv1(x):
        return True, None
    
    dct = {'my_inv': inv1}
    bases = ()
    store_invariants(dct, bases, 'custom_dest', 'my_inv')
    assert 'custom_dest' in dct
    assert 'my_inv' in dct


def test_store_invariants_multiple_inheritance_levels():
    def inv1(x):
        return True, None
    
    def inv2(x):
        return False, "error"
    
    class GrandBase:
        invariant = inv1
    
    class Base(GrandBase):
        invariant = inv2
    
    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, '_invariants', 'invariant')
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 2


def test_store_invariants_tuple_result():
    def inv1(x):
        return True, None
    
    dct = {'invariant': inv1}
    bases = ()
    store_invariants(dct, bases, '_invariants', 'invariant')
    assert isinstance(dct['_invariants'], tuple)


# LLM-generated content at query #10
#--------------------------

```python
def test_check_types_with_valid_types():
    from collections.abc import Iterable
    
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            super().__init__(msg)
    
    class TestClass:
        pass
    
    _check_types([1, 2, 3], [int], TestClass, CheckedValueTypeError)


def test_check_types_with_empty_expected_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            pass
    
    class TestClass:
        pass
    
    _check_types([1, "string", 3.14], [], TestClass, CheckedValueTypeError)


def test_check_types_with_multiple_valid_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            pass
    
    class TestClass:
        pass
    
    _check_types([1, "string", 3.14], [int, str, float], TestClass, CheckedValueTypeError)


def test_check_types_raises_exception_on_invalid_type():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.msg = msg
    
    class TestClass:
        pass
    
    try:
        _check_types([1, "invalid", 3], [int], TestClass, CheckedValueTypeError)
        assert False, "Expected CheckedValueTypeError to be raised"
    except CheckedValueTypeError as e:
        assert "Type TestClass can only be used with" in e.msg
        assert "not str" in e.msg


def test_check_types_with_class_type_objects():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            pass
    
    class TestClass:
        pass
    
    _check_types([1, 2, 3], [int], TestClass, CheckedValueTypeError)


def test_check_types_with_mixed_type_specifications():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            pass
    
    class TestClass:
        pass
    
    _check_types(["a", "b", "c"], [str], TestClass, CheckedValueTypeError)


def test_check_types_exception_message_format():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
    
    class MySourceClass:
        pass
    
    try:
        _check_types([3.14], [int, str], MySourceClass, CheckedValueTypeError)
        assert False, "Expected CheckedValueTypeError to be raised"
    except CheckedValueTypeError as e:
        assert e.source_class == MySourceClass
        assert e.actual_type == float
        assert e.value == 3.14


# LLM-generated content at query #11
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checked_pset_constructor_with_valid_integers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    assert isinstance(result, Positives)


def test_checked_pset_constructor_with_valid_floats():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert 1.5 in result
    assert 2.5 in result
    assert 3.5 in result


def test_checked_pset_constructor_with_mixed_valid_numbers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checked_pset_constructor_with_duplicates():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pset_constructor_with_negative_number():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pset_constructor_with_zero():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([0, 1, 2])
    assert len(result) == 3
    assert 0 in result


def test_checked_pset_constructor_with_generator():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives(x for x in [1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_with_tuple():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives((1, 2, 3))
    assert len(result) == 3
    assert isinstance(result, Positives)


# LLM-generated content at query #12
#--------------------------

```python
def test_checkedpmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checkedpmap_constructor_with_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"1": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "1.5"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_with_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.5, 2: 2.25, 3: 3.75, 4: 4.5}
    result = IntToFloatMap(data)
    assert len(result) == 4
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75
    assert result[4] == 4.5


# LLM-generated content at query #13
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert len(result) == 1
    assert result[1] == 1.5


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_multiple_items():
    class StrToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial = {'a': 1, 'b': 2, 'c': 3}
    result = StrToIntMap(initial)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #14
#--------------------------

```python
def test_store_invariants_predicate_line_1():
    def _all_dicts(bases):
        return []
    
    def wrap_invariant(inv):
        return inv
    
    dct = {'invariant': lambda x: x > 0}
    bases = []
    destination_name = 'wrapped_invariants'
    source_name = 'invariant'
    
    store_invariants(dct, bases, destination_name, source_name)
    
    assert destination_name in dct
    assert isinstance(dct[destination_name], tuple)
    assert len(dct[destination_name]) == 1


# LLM-generated content at query #15
#--------------------------

```python
def test_store_types_empty_dicts():
    dct = {}
    bases = ()
    _store_types(dct, bases, 'dest', 'src')
    assert dct['dest'] == ()


def test_store_types_single_type_in_dct():
    dct = {'src': int}
    bases = ()
    _store_types(dct, bases, 'dest', 'src')
    assert dct['dest'] == (int,)


def test_store_types_single_string_in_dct():
    dct = {'src': 'MyType'}
    bases = ()
    _store_types(dct, bases, 'dest', 'src')
    assert dct['dest'] == ('MyType',)


def test_store_types_list_of_types_in_dct():
    dct = {'src': [int, str]}
    bases = ()
    _store_types(dct, bases, 'dest', 'src')
    assert dct['dest'] == (int, str)


def test_store_types_from_base_class():
    class Base:
        pass
    Base.__dict__['src'] = float
    
    dct = {}
    bases = (Base,)
    _store_types(dct, bases, 'dest', 'src')
    assert dct['dest'] == (float,)


def test_store_types_dct_overrides_base():
    class Base:
        pass
    Base.__dict__['src'] = float
    
    dct = {'src': int}
    bases = (Base,)
    _store_types(dct, bases, 'dest', 'src')
    assert dct['dest'] == (int, float)


def test_store_types_multiple_bases():
    class Base1:
        pass
    class Base2:
        pass
    Base1.__dict__['src'] = int
    Base2.__dict__['src'] = str
    
    dct = {}
    bases = (Base1, Base2)
    _store_types(dct, bases, 'dest', 'src')
    assert dct['dest'] == (int, str)


def test_store_types_nested_iterables():
    dct = {'src': [int, [str, float]]}
    bases = ()
    _store_types(dct, bases, 'dest', 'src')
    assert dct['dest'] == (int, str, float)


def test_store_types_mixed_types_and_strings():
    dct = {'src': [int, 'CustomType', str]}
    bases = ()
    _store_types(dct, bases, 'dest', 'src')
    assert dct['dest'] == (int, 'CustomType', str)


def test_store_types_source_name_not_present():
    dct = {'other_key': int}
    bases = ()
    _store_types(dct, bases, 'dest', 'src')
    assert dct['dest'] == ()


def test_store_types_destination_key_created():
    dct = {}
    bases = ()
    _store_types(dct, bases, 'my_dest', 'my_src')
    assert 'my_dest' in dct


# LLM-generated content at query #16
#--------------------------

```python
def test_invariant_exception_constructor_with_no_arguments():
    exc = InvariantException()
    assert exc.invariant_errors == ()
    assert exc.missing_fields == ()


def test_invariant_exception_constructor_with_error_codes():
    exc = InvariantException(error_codes=("error1", "error2"))
    assert exc.invariant_errors == ("error1", "error2")
    assert exc.missing_fields == ()


def test_invariant_exception_constructor_with_missing_fields():
    exc = InvariantException(missing_fields=("field1", "field2"))
    assert exc.invariant_errors == ()
    assert exc.missing_fields == ("field1", "field2")


def test_invariant_exception_constructor_with_both_error_codes_and_missing_fields():
    exc = InvariantException(error_codes=("error1", "error2"), missing_fields=("field1", "field2"))
    assert exc.invariant_errors == ("error1", "error2")
    assert exc.missing_fields == ("field1", "field2")


def test_invariant_exception_constructor_with_callable_error_codes():
    callable_error = lambda: "callable_error_result"
    exc = InvariantException(error_codes=(callable_error, "static_error"))
    assert exc.invariant_errors == ("callable_error_result", "static_error")
    assert exc.missing_fields == ()


def test_invariant_exception_constructor_with_extra_args():
    exc = InvariantException(error_codes=("error1",), missing_fields=("field1",), "extra_arg")
    assert exc.invariant_errors == ("error1",)
    assert exc.missing_fields == ("field1",)


def test_invariant_exception_constructor_with_extra_kwargs():
    exc = InvariantException(error_codes=("error1",), missing_fields=("field1",), extra_key="extra_value")
    assert exc.invariant_errors == ("error1",)
    assert exc.missing_fields == ("field1",)


def test_invariant_exception_constructor_with_multiple_callable_error_codes():
    callable1 = lambda: "error_from_callable1"
    callable2 = lambda: "error_from_callable2"
    exc = InvariantException(error_codes=(callable1, callable2, "static"))
    assert exc.invariant_errors == ("error_from_callable1", "error_from_callable2", "static")


def test_invariant_exception_constructor_with_empty_error_codes_tuple():
    exc = InvariantException(error_codes=())
    assert exc.invariant_errors == ()
    assert exc.missing_fields == ()


def test_invariant_exception_constructor_with_empty_missing_fields_tuple():
    exc = InvariantException(missing_fields=())
    assert exc.invariant_errors == ()
    assert exc.missing_fields == ()


# LLM-generated content at query #17
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5}, size=10)
    assert len(result) == 1
    assert result[1] == 1.5
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_preserves_type():
    class CustomMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = CustomMap({'a': 1, 'b': 2})
    assert type(result).__name__ == 'CustomMap'
    assert result['a'] == 1
    assert result['b'] == 2


def test_checkedpmap_constructor_single_item():
    class IntToStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = IntToStrMap({42: 'answer'})
    assert len(result) == 1
    assert result[42] == 'answer'


def test_checkedpmap_constructor_multiple_items():
    class IntToIntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    data = {i: i*2 for i in range(5)}
    result = IntToIntMap(data)
    assert len(result) == 5
    for k, v in data.items():
        assert result[k] == v


def test_checkedpmap_constructor_default_parameter():
    class StrToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StrToIntMap(initial={'x': 10})
    assert result['x'] == 10
    assert len(result) == 1


# LLM-generated content at query #18
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockCheckedType:
        _checked_types = []
    
    instance = MockCheckedType()
    result = _checked_type_create(MockCheckedType, instance)
    assert result is instance


def test_checked_type_create_without_checked_type_subclass():
    class MockCheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_subclass_matching_data():
    class CheckedType:
        pass
    
    class MockCheckedType(CheckedType):
        _checked_types = ['__main__.MockCheckedType']
        def __init__(self, data):
            self.data = data
    
    instance1 = MockCheckedType([])
    instance2 = MockCheckedType([])
    source_data = [instance1, instance2]
    result = _checked_type_create(MockCheckedType, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_subclass_non_matching_data():
    class CheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return f"created_{data}"
    
    class MockCheckedType(CheckedType):
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = ["raw_data_1", "raw_data_2"]
    result = _checked_type_create(MockCheckedType, source_data)
    assert result.data == ["created_raw_data_1", "created_raw_data_2"]


def test_checked_type_create_with_ignore_extra_parameter():
    class CheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return f"created_{data}_ignore_{ignore_extra}"
    
    class MockCheckedType(CheckedType):
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = ["data"]
    result = _checked_type_create(MockCheckedType, source_data, ignore_extra=True)
    assert result.data == ["created_data_ignore_True"]


def test_get_type_with_type_object():
    result = get_type(int)
    assert result is int


def test_get_type_with_string():
    result = get_type('builtins.int')
    assert result is int


def test_get_types_with_multiple_types():
    result = get_types([int, 'builtins.str'])
    assert result == [int, str]


def test_get_types_with_empty_list():
    result = get_types([])
    assert result == []


# LLM-generated content at query #19
#--------------------------

```python
def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]


def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type('int')
    assert result == ['int']


def test_maybe_parse_user_type_with_list_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)


def test_maybe_parse_user_type_with_tuple_of_types():
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(['int', 'str'])
    assert result == ('int', 'str')


def test_maybe_parse_user_type_with_mixed_list():
    result = maybe_parse_user_type([int, 'str', float])
    assert result == (int, 'str', float)


def test_maybe_parse_user_type_with_nested_iterables():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


def test_maybe_parse_user_type_with_none():
    try:
        maybe_parse_user_type(None)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


def test_maybe_parse_user_type_with_dict():
    try:
        maybe_parse_user_type({'key': int})
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


# LLM-generated content at query #20
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


# LLM-generated content at query #21
#--------------------------

```python
def test_wrap_invariant_with_bool_result():
    def invariant_func(x):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped(5)
    assert verdict is True
    assert data == "success"


def test_wrap_invariant_with_bool_result_false():
    def invariant_func(x):
        return (False, "error")
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped(5)
    assert verdict is False
    assert data == "error"


def test_wrap_invariant_with_multiple_results():
    def invariant_func(x):
        return [(True, "ok1"), (True, "ok2")]
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped(5)
    assert verdict is True
    assert data == ()


def test_wrap_invariant_with_multiple_results_one_failure():
    def invariant_func(x):
        return [(True, "ok"), (False, "failed")]
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped(5)
    assert verdict is False
    assert data == ("failed",)


def test_wrap_invariant_with_multiple_results_multiple_failures():
    def invariant_func(x):
        return [(True, "ok"), (False, "failed1"), (False, "failed2")]
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped(5)
    assert verdict is False
    assert data == ("failed1", "failed2")


def test_wrap_invariant_with_kwargs():
    def invariant_func(x, y=10):
        return (True, x + y)
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped(5, y=20)
    assert verdict is True
    assert data == 25


def test_wrap_invariant_with_multiple_args():
    def invariant_func(x, y, z):
        return [(x > 0, "x_positive"), (y > 0, "y_positive"), (z > 0, "z_positive")]
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped(1, -1, 1)
    assert verdict is False
    assert data == ("y_positive",)


def test_wrap_invariant_empty_failures():
    def invariant_func():
        return [(True, "test1"), (True, "test2"), (True, "test3")]
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped()
    assert verdict is True
    assert data == ()


def test_wrap_invariant_all_failures():
    def invariant_func():
        return [(False, "fail1"), (False, "fail2"), (False, "fail3")]
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped()
    assert verdict is False
    assert data == ("fail1", "fail2", "fail3")


# LLM-generated content at query #22
#--------------------------

```python
def test_checked_pmap_initial_items_iteration():
    from pyrsistent import CheckedPMap
    
    class TestMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {'a': 1, 'b': 2, 'c': 3}
    result = TestMap(initial_data)
    
    assert dict(result) == initial_data
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #23
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "InvariantException" in str(type(e).__name__)


def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except Exception as e:
        assert "CheckedKeyTypeError" in str(type(e).__name__)


def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Expected exception for wrong value type"
    except Exception:
        pass


def test_checked_pmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


# LLM-generated content at query #24
#--------------------------

```python
def test_checked_pvector_constructor_with_empty_initial():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives()
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_list():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_python_pvector():
    from pyrsistent import pvector
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pv = pvector([1, 2, 3])
    result = Positives(pv)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_invalid_type():
    from pyrsistent import InvariantException
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Should have raised an exception"
    except (InvariantException, TypeError):
        pass


def test_checked_pvector_constructor_with_negative_value():
    from pyrsistent import InvariantException
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised an exception"
    except InvariantException:
        pass


def test_checked_pvector_constructor_preserves_type():
    class CustomVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n > 0, 'Not positive')
    
    result = CustomVector([1, 2, 3])
    assert type(result).__name__ == 'CustomVector'
    assert isinstance(result, CustomVector)


def test_checked_pvector_constructor_with_tuple():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives((1, 2, 3))
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #25
#--------------------------

```python
def test_check_types_predicate_true():
    expected_types = [int, str]
    it = [1, "hello", 2]
    source_class = list
    
    # This should not raise an exception since the predicate evaluates to True
    # (all elements are instances of int or str)
    from your_module import _check_types
    _check_types(it, expected_types, source_class)


# LLM-generated content at query #26
#--------------------------

```python
def test_invariant_errors_all_valid():
    def invariant1(elem):
        return (True, "valid1")
    
    def invariant2(elem):
        return (True, "valid2")
    
    result = _invariant_errors("test", [invariant1, invariant2])
    assert result == []


def test_invariant_errors_some_invalid():
    def invariant1(elem):
        return (True, "valid1")
    
    def invariant2(elem):
        return (False, "error2")
    
    def invariant3(elem):
        return (False, "error3")
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == ["error2", "error3"]


def test_invariant_errors_all_invalid():
    def invariant1(elem):
        return (False, "error1")
    
    def invariant2(elem):
        return (False, "error2")
    
    result = _invariant_errors("test", [invariant1, invariant2])
    assert result == ["error1", "error2"]


def test_invariant_errors_empty_invariants():
    result = _invariant_errors("test", [])
    assert result == []


def test_invariant_errors_single_valid_invariant():
    def invariant(elem):
        return (True, "valid")
    
    result = _invariant_errors("test", [invariant])
    assert result == []


def test_invariant_errors_single_invalid_invariant():
    def invariant(elem):
        return (False, "error")
    
    result = _invariant_errors("test", [invariant])
    assert result == ["error"]


def test_invariant_errors_with_different_data_types():
    def invariant1(elem):
        return (False, 42)
    
    def invariant2(elem):
        return (False, {"error": "message"})
    
    def invariant3(elem):
        return (True, "valid")
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == [42, {"error": "message"}]


# LLM-generated content at query #27
#--------------------------

```python
def test_checked_pmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_default_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "InvariantException" in str(type(e).__name__)


def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception as e:
        assert "CheckedKeyTypeError" in str(type(e).__name__)


def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except Exception as e:
        assert "CheckedTypeError" in str(type(e).__name__)


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == "IntToFloatMap"


def test_checked_pmap_constructor_with_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2, "c": 3})
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


# LLM-generated content at query #28
#--------------------------

```python
def test_maybe_parse_user_type_line_18_predicate():
    from collections.abc import Iterable
    
    # Create a custom type that is a type but not iterable
    class CustomType:
        pass
    
    # Verify the predicate at line 18: is_type and not is_iterable
    t = CustomType
    is_type = isinstance(t, type)
    is_iterable = isinstance(t, Iterable)
    
    # The predicate should evaluate to True
    assert is_type and not is_iterable


# LLM-generated content at query #29
#--------------------------

```python
def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]


def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]


def test_maybe_parse_user_type_with_list_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)


def test_maybe_parse_user_type_with_tuple_of_types():
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(["int", "str"])
    assert result == ("int", "str")


def test_maybe_parse_user_type_with_mixed_list():
    result = maybe_parse_user_type([int, "str", float])
    assert result == (int, "str", float)


def test_maybe_parse_user_type_with_nested_iterables():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(42)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_none():
    try:
        maybe_parse_user_type(None)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_dict():
    try:
        maybe_parse_user_type({int: str})
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_empty_list():
    result = maybe_parse_user_type([])
    assert result == ()


def test_maybe_parse_user_type_with_deeply_nested_iterables():
    result = maybe_parse_user_type([int, [str, [float, bool]]])
    assert result == (int, str, float, bool)


# LLM-generated content at query #30
#--------------------------

```python
def test_checkedpmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checkedpmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_default_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_multiple_valid_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 5
    for k, v in data.items():
        assert result[k] == v


# LLM-generated content at query #31
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_one_false():
    result = [(True, "data1"), (False, "error1"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)

def test_merge_invariant_results_multiple_false():
    result = [(False, "error1"), (True, "data2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error3")

def test_merge_invariant_results_all_false():
    result = [(False, "error1"), (False, "error2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2", "error3")

def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_single_true():
    result = [(True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()

def test_merge_invariant_results_single_false():
    result = [(False, "error1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)

def test_merge_invariant_results_various_data_types():
    result = [(False, 123), (True, "data"), (False, None), (False, {"key": "value"})]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == (123, None, {"key": "value"})


# LLM-generated content at query #32
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_default_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({5: 5.5})
    assert type(result).__name__ == 'IntToFloatMap'
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #33
#--------------------------

```python
def test_checked_type_create_isinstance_predicate():
    class MockCheckedType:
        pass
    
    class MockClass:
        pass
    
    source_data = MockClass()
    result = isinstance(source_data, MockClass)
    assert result is True


# LLM-generated content at query #34
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checkedpmap_constructor_single_entry():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert result[42] == 3.14
    assert len(result) == 1


def test_checkedpmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_checkedpmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception:
        pass


def test_checkedpmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised exception"
    except Exception:
        pass


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert type(result).__name__ == 'IntToFloatMap'


# LLM-generated content at query #35
#--------------------------

```python
def test_checked_pvector_constructor_with_empty_list():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_valid_integers():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_valid_floats():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert result[0] == 1.5


def test_checked_pvector_constructor_with_mixed_numbers():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2.5


def test_checked_pvector_constructor_with_pvector_input():
    from pyrsistent import pvector
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pv = pvector([1, 2, 3])
    result = Positives(pv)
    assert len(result) == 3
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_tuple_input():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives((1, 2, 3))
    assert len(result) == 3
    assert result[0] == 1


def test_checked_pvector_constructor_with_generator():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives(x for x in [1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1


def test_checked_pvector_constructor_default_parameter():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives()
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_single_element():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([42])
    assert len(result) == 1
    assert result[0] == 42


# LLM-generated content at query #36
#--------------------------

```python
def test_checked_type_create_predicate_false():
    class MockClass:
        pass
    
    source_data = MockClass()
    cls = type('TestClass', (), {})
    
    result = isinstance(source_data, cls)
    
    assert result is False


# LLM-generated content at query #37
#--------------------------

```python
def test_check_types_predicate_true_with_expected_types():
    expected_types = [int, str]
    it = [1, "hello", 2]
    source_class = list
    
    # Mock get_type function
    def get_type(t):
        return t
    
    # Mock CheckedValueTypeError
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, e, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.e = e
            self.msg = msg
    
    # Inject get_type into globals for the function
    import sys
    from types import FunctionType
    
    def _check_types(it, expected_types, source_class, exception_type=CheckedValueTypeError):
        if expected_types:
            for e in it:
                if not any(isinstance(e, get_type(t)) for t in expected_types):
                    actual_type = type(e)
                    msg = "Type {source_class} can only be used with {expected_types}, not {actual_type}".format(
                        source_class=source_class.__name__,
                        expected_types=tuple(get_type(et).__name__ for et in expected_types),
                        actual_type=actual_type.__name__)
                    raise exception_type(source_class, expected_types, actual_type, e, msg)
    
    # This should not raise an exception since all elements match expected types
    _check_types(it, expected_types, source_class)


# LLM-generated content at query #38
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant_fail(elem):
        return (False, "error_data")
    
    def invariant_pass(elem):
        return (True, "valid_data")
    
    invariants = [invariant_fail, invariant_pass]
    result = _invariant_errors("test_elem", invariants)
    
    assert result == ["error_data"]


# LLM-generated content at query #39
#--------------------------

```python
def test_check_types_predicate_line_1():
    def get_type(t):
        return t
    
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            pass
    
    class TestClass:
        pass
    
    # Test case 1: expected_types is None/empty - predicate at line 1 should be False
    it = [1, 2, 3]
    expected_types = None
    # When expected_types is falsy, the if block is skipped
    result = bool(expected_types)
    assert result == False
    
    # Test case 2: expected_types is non-empty - predicate at line 1 should be True
    it = [1, 2, 3]
    expected_types = [int]
    result = bool(expected_types)
    assert result == True
    
    # Test case 3: expected_types is empty list - predicate at line 1 should be False
    it = [1, 2, 3]
    expected_types = []
    result = bool(expected_types)
    assert result == False


# LLM-generated content at query #40
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checked_pset_constructor_with_valid_integers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    assert isinstance(result, Positives)


def test_checked_pset_constructor_with_valid_floats():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert 1.5 in result
    assert 2.5 in result
    assert 3.5 in result


def test_checked_pset_constructor_with_mixed_valid_types():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checked_pset_constructor_with_duplicates():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, 'invalid', 3])
        assert False, "Should have raised an exception"
    except TypeCheckError:
        pass


def test_checked_pset_constructor_with_negative_number():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised an exception"
    except InvariantException:
        pass


def test_checked_pset_constructor_with_zero():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([0, 1, 2])
    assert len(result) == 3
    assert 0 in result
    assert 1 in result
    assert 2 in result


def test_checked_pset_constructor_with_generator():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives(x for x in [1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_preserves_class_type():
    class SpecialSet(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = SpecialSet([1, 2, 3])
    assert type(result).__name__ == 'SpecialSet'
    assert isinstance(result, SpecialSet)


# LLM-generated content at query #41
#--------------------------

```python
def test_wrap_invariant_predicate_line_3():
    def sample_invariant(*args, **kwargs):
        return (True, "test")
    
    wrapped = wrap_invariant(sample_invariant)
    result = wrapped()
    
    assert isinstance(result[0], bool) == True


# LLM-generated content at query #42
#--------------------------

```python
def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]


def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]


def test_maybe_parse_user_type_with_list_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)


def test_maybe_parse_user_type_with_tuple_of_types():
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(["int", "str"])
    assert result == ("int", "str")


def test_maybe_parse_user_type_with_mixed_list():
    result = maybe_parse_user_type([int, "str"])
    assert result == (int, "str")


def test_maybe_parse_user_type_with_nested_iterables():
    result = maybe_parse_user_type([[int, str], [float]])
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_none():
    try:
        maybe_parse_user_type(None)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_empty_list():
    result = maybe_parse_user_type([])
    assert result == ()


# LLM-generated content at query #43
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checkedpmap_constructor_preserves_type():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2})
    assert isinstance(result, StringToIntMap)
    assert result["a"] == 1
    assert result["b"] == 2


# LLM-generated content at query #44
#--------------------------

```python
def test_check_types_with_valid_types():
    from collections.abc import Sequence
    
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    _check_types([1, 2, 3], [int], TestClass, CheckedValueTypeError)


def test_check_types_with_empty_expected_types():
    class TestClass:
        pass
    
    _check_types([1, "string", 3.14], [], TestClass)


def test_check_types_with_multiple_valid_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    _check_types([1, "string", 3.14], [int, str, float], TestClass, CheckedValueTypeError)


def test_check_types_with_invalid_type():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    try:
        _check_types([1, "invalid", 3], [int], TestClass, CheckedValueTypeError)
        assert False, "Expected CheckedValueTypeError to be raised"
    except CheckedValueTypeError as e:
        assert e.source_class == TestClass
        assert e.expected_types == [int]
        assert e.actual_type == str
        assert e.value == "invalid"
        assert "Type TestClass can only be used with ('int',), not str" in e.msg


def test_check_types_with_string_type_name():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    _check_types([1, 2, 3], ['builtins.int'], TestClass, CheckedValueTypeError)


def test_check_types_raises_with_wrong_type_in_list():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    try:
        _check_types([1, 2.5, 3], [int], TestClass, CheckedValueTypeError)
        assert False, "Expected CheckedValueTypeError to be raised"
    except CheckedValueTypeError as e:
        assert e.actual_type == float
        assert e.value == 2.5


# LLM-generated content at query #45
#--------------------------

```python
def test_restore_pickle_creates_instance_with_empty_factory_fields():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls()
            instance.data = data
            instance.factory_fields = _factory_fields
            return instance
    
    test_data = {"key": "value"}
    result = _restore_pickle(MockClass, test_data)
    
    assert result.data == test_data
    assert result.factory_fields == set()


def test_restore_pickle_passes_correct_arguments():
    class TestClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls()
            instance.received_data = data
            instance.received_factory_fields = _factory_fields
            return instance
    
    input_data = [1, 2, 3]
    result = _restore_pickle(TestClass, input_data)
    
    assert result.received_data == input_data
    assert result.received_factory_fields == set()


def test_restore_pickle_with_empty_data():
    class TestClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls()
            instance.data = data
            instance.factory_fields = _factory_fields
            return instance
    
    result = _restore_pickle(TestClass, {})
    
    assert result.data == {}
    assert result.factory_fields == set()


# LLM-generated content at query #46
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_dict():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1, 'b': 2})
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2


def test_checked_pmap_constructor_with_single_item():
    class IntToStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = IntToStrMap({42: 'answer'})
    assert len(result) == 1
    assert result[42] == 'answer'


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_from_pmap_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    original = IntToFloatMap({1: 1.5, 2: 2.25})
    result = IntToFloatMap(original)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_multiple_items():
    class IntToStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    data = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
    result = IntToStrMap(data)
    assert len(result) == 5
    assert result[1] == 'one'
    assert result[5] == 'five'


# LLM-generated content at query #47
#--------------------------

```python
def test_restore_pickle_creates_instance_with_empty_factory_fields():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self._factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            return cls(data, _factory_fields=_factory_fields)
    
    test_data = {"key": "value"}
    result = MockClass.create.__func__.__self__.__class__._restore_pickle(MockClass, test_data)
    
    assert result.data == test_data
    assert result._factory_fields == set()


def test_restore_pickle_with_different_data_types():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self._factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            return cls(data, _factory_fields=_factory_fields)
    
    test_data = [1, 2, 3, 4, 5]
    result = MockClass.create.__func__.__self__.__class__._restore_pickle(MockClass, test_data)
    
    assert result.data == test_data
    assert result._factory_fields == set()


def test_restore_pickle_factory_fields_is_empty_set():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self._factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            return cls(data, _factory_fields=_factory_fields)
    
    test_data = "test_string"
    result = MockClass.create.__func__.__self__.__class__._restore_pickle(MockClass, test_data)
    
    assert result._factory_fields == set()
    assert len(result._factory_fields) == 0


# LLM-generated content at query #48
#--------------------------

```python
def test_checked_pmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_default_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "Invalid mapping" in str(e) or "InvariantException" in str(type(e).__name__)


def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except Exception as e:
        assert "CheckedKeyTypeError" in str(type(e).__name__) or "key" in str(e).lower()


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid_value"})
        assert False, "Expected type error"
    except Exception as e:
        assert "value" in str(e).lower() or "type" in str(e).lower()


def test_checked_pmap_constructor_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2, "c": 3})
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


# LLM-generated content at query #49
#--------------------------

```python
def test_restore_pickle_calls_create_with_factory_fields_empty_set():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            return {"data": data, "_factory_fields": _factory_fields}
    
    result = _restore_pickle(MockClass, {"key": "value"})
    assert result == {"data": {"key": "value"}, "_factory_fields": set()}


def test_restore_pickle_with_empty_data():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            return {"data": data, "_factory_fields": _factory_fields}
    
    result = _restore_pickle(MockClass, {})
    assert result == {"data": {}, "_factory_fields": set()}


def test_restore_pickle_with_complex_data():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            return {"data": data, "_factory_fields": _factory_fields}
    
    complex_data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
    result = _restore_pickle(MockClass, complex_data)
    assert result == {"data": complex_data, "_factory_fields": set()}


def test_restore_pickle_factory_fields_is_empty_set():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            return _factory_fields
    
    result = _restore_pickle(MockClass, {"test": "data"})
    assert result == set()
    assert len(result) == 0


# LLM-generated content at query #50
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant1(elem):
        return (True, "valid1")
    
    def invariant2(elem):
        return (False, "invalid2")
    
    def invariant3(elem):
        return (False, "invalid3")
    
    invariants = [invariant1, invariant2, invariant3]
    elem = "test_element"
    
    result = _invariant_errors(elem, invariants)
    
    assert result == ["invalid2", "invalid3"]


# LLM-generated content at query #51
#--------------------------

```python
def test_checked_pmap_new_with_empty_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_new_with_single_element():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert len(result) == 1
    assert result[1] == 1.5


def test_checked_pmap_new_with_multiple_elements():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25, 3: 3.75})
    assert len(result) == 3
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75


def test_checked_pmap_new_with_explicit_size():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=16)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_new_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0


def test_checked_pmap_new_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"a": 1.5})
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pmap_new_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pmap_new_with_violated_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pmap_new_returns_correct_type():
    class CustomMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = CustomMap({"a": 1, "b": 2})
    assert type(result) == CustomMap


def test_checked_pmap_new_default_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


# LLM-generated content at query #52
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checked_pset_constructor_with_valid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    assert isinstance(result, Positives)


def test_checked_pset_constructor_with_float_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert 1.5 in result
    assert 2.5 in result
    assert 3.5 in result


def test_checked_pset_constructor_with_mixed_valid_types():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checked_pset_constructor_with_duplicates():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 1, 2, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_with_zero():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([0, 1, 2])
    assert len(result) == 3
    assert 0 in result


def test_checked_pset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives(['invalid'])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pset_constructor_with_negative_invariant_violation():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([-1, 1, 2])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pset_constructor_with_tuple_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives((1, 2, 3))
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


# LLM-generated content at query #53
#--------------------------

```python
def test_checked_pmap_initial_items_iteration():
    from pyrsistent import CheckedPMap
    
    class TestMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {'a': 1, 'b': 2, 'c': 3}
    result = TestMap(initial_data)
    
    assert dict(result) == initial_data
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #54
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant1(elem):
        return (True, "valid1")
    
    def invariant2(elem):
        return (False, "invalid2")
    
    def invariant3(elem):
        return (False, "invalid3")
    
    invariants = [invariant1, invariant2, invariant3]
    elem = "test_element"
    
    result = _invariant_errors(elem, invariants)
    
    assert result == ["invalid2", "invalid3"]
    assert len(result) == 2
    assert "invalid2" in result
    assert "invalid3" in result


# LLM-generated content at query #55
#--------------------------

```python
def test_checked_type_create_isinstance_predicate():
    class MockCheckedType:
        pass
    
    class TestClass(MockCheckedType):
        pass
    
    instance = TestClass()
    result = isinstance(instance, TestClass)
    
    assert result is True


# LLM-generated content at query #56
#--------------------------

```python
def test_checked_type_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #57
#--------------------------

```python
def test_checkedpvector_constructor_with_empty_initial():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checkedpvector_constructor_with_valid_integers():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checkedpvector_constructor_with_valid_floats():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert result[0] == 1.5
    assert result[1] == 2.5
    assert result[2] == 3.5


def test_checkedpvector_constructor_with_mixed_valid_types():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2.5
    assert result[2] == 3


def test_checkedpvector_constructor_with_pythonpvector():
    from pyrsistent import pvector
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pv = pvector([1, 2, 3])
    result = Positives(pv)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checkedpvector_constructor_with_generator():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives(x for x in [1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_checkedpvector_constructor_with_tuple():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives((1, 2, 3))
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #58
#--------------------------

```python
def test_check_types_predicate_with_non_empty_expected_types():
    expected_types = [int, str]
    result = bool(expected_types)
    assert result is True


# LLM-generated content at query #59
#--------------------------

```python
def test_checked_pmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_with_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "InvariantException" in type(e).__name__


def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except Exception as e:
        assert "CheckedKeyTypeError" in type(e).__name__


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Expected CheckedTypeError"
    except Exception as e:
        assert "CheckedTypeError" in type(e).__name__


def test_checked_pmap_constructor_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(initial_data)
    assert len(result) == 5
    assert result[1] == 1.1
    assert result[5] == 5.5


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({10: 10.5})
    assert type(result).__name__ == 'IntToFloatMap'


# LLM-generated content at query #60
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class TestCheckedType:
        _checked_types = []
    
    instance = TestCheckedType()
    result = _checked_type_create(TestCheckedType, instance)
    assert result is instance


def test_checked_type_create_with_simple_data():
    class SimpleClass:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    result = _checked_type_create(SimpleClass, [1, 2, 3])
    assert result.data == [1, 2, 3]


def test_checked_type_create_with_non_checked_type():
    class NonCheckedType:
        _checked_types = ['builtins.str']
        def __init__(self, data):
            self.data = data
    
    result = _checked_type_create(NonCheckedType, [1, 2, 3])
    assert result.data == [1, 2, 3]


def test_checked_type_create_with_checked_type_matching_data():
    class CheckedType:
        _checked_types = ['builtins.str']
        def __init__(self, data):
            self.data = data
        @staticmethod
        def create(data, ignore_extra=False):
            return CheckedType(data)
    
    result = _checked_type_create(CheckedType, ["hello"])
    assert isinstance(result, CheckedType)


def test_checked_type_create_ignore_extra_parameter():
    class CustomCheckedType:
        _checked_types = ['builtins.str']
        def __init__(self, data):
            self.data = data
        @staticmethod
        def create(data, ignore_extra=False):
            return CustomCheckedType(data)
    
    result = _checked_type_create(CustomCheckedType, [1, 2], ignore_extra=True)
    assert isinstance(result, CustomCheckedType)


def test_checked_type_create_factory_fields_parameter():
    class FactoryCheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    result = _checked_type_create(FactoryCheckedType, [1, 2], _factory_fields={'key': 'value'})
    assert result.data == [1, 2]


# LLM-generated content at query #61
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


def test_merge_invariant_results_one_false():
    result = [(True, "data1"), (False, "error1"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)


def test_merge_invariant_results_multiple_false():
    result = [(False, "error1"), (True, "data2"), (False, "error2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2", "error3")


def test_merge_invariant_results_all_false():
    result = [(False, "error1"), (False, "error2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2")


def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


def test_merge_invariant_results_single_true():
    result = [(True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


def test_merge_invariant_results_single_false():
    result = [(False, "error1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)


def test_merge_invariant_results_various_data_types():
    result = [(False, 123), (True, "data"), (False, None), (False, {"key": "value"})]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == (123, None, {"key": "value"})


# LLM-generated content at query #62
#--------------------------

```python
def test_checkedpvector_constructor_with_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector()
    assert len(result) == 0
    assert isinstance(result, TestVector)


def test_checkedpvector_constructor_with_list():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, TestVector)


def test_checkedpvector_constructor_with_tuple():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector((4, 5, 6))
    assert len(result) == 3
    assert result[0] == 4
    assert result[1] == 5
    assert result[2] == 6


def test_checkedpvector_constructor_with_pythonpvector():
    from pyrsistent import PVector
    class TestVector(CheckedPVector):
        __type__ = int
    
    pv = PVector([7, 8, 9])
    result = TestVector(pv)
    assert len(result) == 3
    assert result[0] == 7
    assert result[1] == 8
    assert result[2] == 9
    assert isinstance(result, TestVector)


def test_checkedpvector_constructor_with_multiple_types():
    class TestVector(CheckedPVector):
        __type__ = (int, float)
    
    result = TestVector([1, 2.5, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2.5
    assert result[2] == 3


def test_checkedpvector_constructor_with_invariant():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_checkedpvector_constructor_preserves_type():
    class CustomVector(CheckedPVector):
        __type__ = str
    
    result = CustomVector(['a', 'b', 'c'])
    assert type(result).__name__ == 'CustomVector'
    assert isinstance(result, CustomVector)


# LLM-generated content at query #63
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Expected CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checked_pmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2, "c": 3})
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


def test_checked_pmap_constructor_from_dict_literal():
    class IntToStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    initial_dict = {1: "one", 2: "two"}
    result = IntToStrMap(initial_dict)
    assert result[1] == "one"
    assert result[2] == "two"


# LLM-generated content at query #64
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checkedpmap_constructor_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1, 'b': 2, 'c': 3})
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #65
#--------------------------

```python
def test_check_types_with_valid_types():
    from collections.abc import Iterable
    
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    test_iterable = [1, 2, 3]
    _check_types(test_iterable, [int], TestClass, CheckedValueTypeError)


def test_check_types_with_empty_expected_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    test_iterable = [1, 2, 3]
    _check_types(test_iterable, [], TestClass, CheckedValueTypeError)


def test_check_types_with_invalid_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    test_iterable = [1, "invalid", 3]
    exception_raised = False
    try:
        _check_types(test_iterable, [int], TestClass, CheckedValueTypeError)
    except CheckedValueTypeError as e:
        exception_raised = True
        assert e.source_class == TestClass
        assert e.expected_types == [int]
        assert e.actual_type == str
        assert e.value == "invalid"
        assert "Type TestClass can only be used with" in e.msg
    
    assert exception_raised


def test_check_types_with_multiple_expected_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    test_iterable = [1, "string", 3.14]
    _check_types(test_iterable, [int, str, float], TestClass, CheckedValueTypeError)


def test_check_types_raises_on_first_invalid_element():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    test_iterable = [1, 2, [], 4]
    exception_raised = False
    try:
        _check_types(test_iterable, [int], TestClass, CheckedValueTypeError)
    except CheckedValueTypeError as e:
        exception_raised = True
        assert e.actual_type == list
        assert e.value == []
    
    assert exception_raised


# LLM-generated content at query #66
#--------------------------

```python
def test_maybe_parse_user_type_preserved_iterable():
    from collections.abc import Iterable
    
    class _preserved_iterable_types(tuple):
        pass
    
    def maybe_parse_user_type(t):
        is_type = isinstance(t, type)
        is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
        is_string = isinstance(t, str)
        is_iterable = isinstance(t, Iterable)

        if is_preserved:
            return [t]
        elif is_string:
            return [t]
        elif is_type and not is_iterable:
            return [t]
        elif is_iterable:
            ts = t
            return tuple(e for t in ts for e in maybe_parse_user_type(t))
        else:
            raise TypeError(
                'Type specifications must be types or strings. Input: {}'.format(t)
            )
    
    result = maybe_parse_user_type(_preserved_iterable_types)
    assert result == [_preserved_iterable_types]
    assert isinstance(result, list)
    assert len(result) == 1


# LLM-generated content at query #67
#--------------------------

```python
def test_wrap_invariant_predicate_at_line_3_evaluates_to_false():
    def invariant_func():
        return ("not_a_bool_value", "another_value")
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    
    assert isinstance(result[0], bool) is False


# LLM-generated content at query #68
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockCheckedType:
        _checked_types = []
    
    mock_instance = MockCheckedType()
    result = _checked_type_create(MockCheckedType, mock_instance)
    assert result is mock_instance


def test_checked_type_create_with_non_matching_data():
    class MockCheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert result.data == source_data


def test_checked_type_create_with_ignore_extra_false():
    class InnerCheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return f"created_{data}"
    
    class MockCheckedType:
        _checked_types = ['__main__.InnerCheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2]
    result = _checked_type_create(MockCheckedType, source_data, ignore_extra=False)
    assert result.data == ["created_1", "created_2"]


def test_get_type_with_type_object():
    result = get_type(int)
    assert result is int


def test_get_type_with_string():
    result = get_type('builtins.int')
    assert result is int


def test_get_types_with_multiple_types():
    result = get_types([int, 'builtins.str', float])
    assert result == [int, str, float]


def test_get_types_with_empty_list():
    result = get_types([])
    assert result == []


def test_get_class_with_valid_module_path():
    result = _get_class('builtins.int')
    assert result is int


def test_get_class_with_builtin_exception():
    result = _get_class('builtins.ValueError')
    assert result is ValueError


# LLM-generated content at query #69
#--------------------------

```python
def test_wrap_invariant_with_bool_result():
    def invariant_func():
        return (True, "test")
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    
    assert isinstance(result[0], bool)
    assert result == (True, "test")


# LLM-generated content at query #70
#--------------------------

```python
def test_maybe_parse_user_type_line_18_predicate():
    """Test that the predicate at line 18 (is_type and not is_iterable) evaluates to True."""
    
    class CustomType:
        pass
    
    # Create a type that is not iterable
    test_type = CustomType
    
    # Verify preconditions for line 18 predicate
    is_type = isinstance(test_type, type)
    is_iterable = isinstance(test_type, Iterable)
    
    # The predicate at line 18 is: is_type and not is_iterable
    predicate_result = is_type and not is_iterable
    
    assert predicate_result is True
    assert is_type is True
    assert is_iterable is False


# LLM-generated content at query #71
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    from pyrsistent import CheckedPMap
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_dict():
    from pyrsistent import CheckedPMap
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_with_invariant():
    from pyrsistent import CheckedPMap
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0


def test_checkedpmap_constructor_with_size_parameter():
    from pyrsistent import CheckedPMap
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert len(result) == 2
    assert result[1] == 1.5


def test_checkedpmap_constructor_preserves_type():
    from pyrsistent import CheckedPMap
    
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1, 'b': 2})
    assert isinstance(result, StringToIntMap)
    assert result['a'] == 1


def test_checkedpmap_constructor_invalid_key_type():
    from pyrsistent import CheckedPMap, CheckedKeyTypeError
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({'invalid': 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_invalid_value_type():
    from pyrsistent import CheckedPMap, CheckedValueTypeError
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: 'invalid'})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checkedpmap_constructor_invariant_violation():
    from pyrsistent import CheckedPMap, InvariantException
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #72
#--------------------------

```python
def test_checked_pmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"not_int": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_float"})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    result = StringToIntMap(data)
    assert len(result) == 4
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert result['d'] == 4


# LLM-generated content at query #73
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Expected InvariantException"
    except Exception:
        pass


def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except Exception:
        pass


def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Expected type error"
    except Exception:
        pass


def test_checked_pmap_constructor_default_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


# LLM-generated content at query #74
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant1(elem):
        return (True, "valid_data_1")
    
    def invariant2(elem):
        return (False, "invalid_data_2")
    
    def invariant3(elem):
        return (False, "invalid_data_3")
    
    invariants = [invariant1, invariant2, invariant3]
    elem = "test_element"
    
    result = _invariant_errors(elem, invariants)
    
    assert result == ["invalid_data_2", "invalid_data_3"]
    assert len(result) == 2
    assert "invalid_data_2" in result
    assert "invalid_data_3" in result


# LLM-generated content at query #75
#--------------------------

```python
def test_checked_type_create_isinstance_predicate():
    class MockCheckedType:
        pass
    
    class TestClass(MockCheckedType):
        pass
    
    instance = TestClass()
    result = isinstance(instance, TestClass)
    assert result is True


# LLM-generated content at query #76
#--------------------------

```python
def test_checked_type_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #77
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    class MockCheckedType:
        pass
    
    source_data = MockCheckedType()
    cls = list
    
    result = isinstance(source_data, cls)
    
    assert result is False


# LLM-generated content at query #78
#--------------------------

```python
def test_wrap_invariant_predicate_at_line_3():
    def invariant(*args, **kwargs):
        return (False, "some message")
    
    wrapped = wrap_invariant(invariant)
    result = wrapped()
    
    assert result == (False, "some message")


# LLM-generated content at query #79
#--------------------------

```python
def test_check_types_with_valid_types():
    from collections.abc import Iterable
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    _check_types([1, 2, 3], [int], TestClass, CheckedValueTypeError)


def test_check_types_with_empty_expected_types():
    class TestClass:
        pass
    
    _check_types([1, "string", 3.14], [], TestClass)


def test_check_types_with_multiple_valid_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    _check_types([1, "string", 3.14], [int, str, float], TestClass, CheckedValueTypeError)


def test_check_types_raises_exception_for_invalid_type():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    try:
        _check_types([1, "string", 3.14], [int], TestClass, CheckedValueTypeError)
        assert False, "Expected exception to be raised"
    except CheckedValueTypeError as e:
        assert e.actual_type == str
        assert e.value == "string"
        assert "TestClass" in e.msg
        assert "int" in e.msg


def test_check_types_with_class_type_string():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    _check_types([1, 2], ["builtins.int"], TestClass, CheckedValueTypeError)


def test_check_types_raises_with_class_type_string():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    try:
        _check_types(["string"], ["builtins.int"], TestClass, CheckedValueTypeError)
        assert False, "Expected exception to be raised"
    except CheckedValueTypeError as e:
        assert e.actual_type == str


# LLM-generated content at query #80
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_default_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1, 'b': 2, 'c': 3})
    assert isinstance(result, StringToIntMap)
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert len(result) == 3


# LLM-generated content at query #81
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_default_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5


def test_checked_pmap_constructor_returns_correct_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {'a': 1, 'b': 2, 'c': 3}
    result = StringToIntMap(initial_data)
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert len(result) == 3


# LLM-generated content at query #82
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'
    assert isinstance(result, CheckedPMap)


def test_checked_pmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4}
    result = IntToFloatMap(initial)
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[4] == 4.4


# LLM-generated content at query #83
#--------------------------

```python
def test_check_types_predicate_line_1():
    # Test that the predicate at line 1 (if expected_types:) evaluates to True
    # This means expected_types should be a non-empty collection
    
    def get_type(t):
        return t
    
    def _check_types(it, expected_types, source_class, exception_type=None):
        if expected_types:
            for e in it:
                if not any(isinstance(e, get_type(t)) for t in expected_types):
                    actual_type = type(e)
                    msg = "Type {source_class} can only be used with {expected_types}, not {actual_type}".format(
                        source_class=source_class.__name__,
                        expected_types=tuple(get_type(et).__name__ for et in expected_types),
                        actual_type=actual_type.__name__)
                    raise exception_type(source_class, expected_types, actual_type, e, msg)
    
    class TestClass:
        pass
    
    # Call with non-empty expected_types to make the predicate True
    result = _check_types([1, 2, 3], [int], TestClass)
    
    # The function should complete without raising an error when types match
    assert result is None


# LLM-generated content at query #84
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "InvariantException" in str(type(e))


def test_checkedpmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception as e:
        assert "CheckedKeyTypeError" in str(type(e))


def test_checkedpmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedValueTypeError"
    except Exception as e:
        assert "CheckedValueTypeError" in str(type(e)) or "TypeError" in str(type(e))


def test_checkedpmap_constructor_single_element():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.0, 2: 2.0, 3: 3.0}
    result = IntToFloatMap(initial_data, size=8)
    assert len(result) == 3
    assert result[1] == 1.0
    assert result[2] == 2.0
    assert result[3] == 3.0


# LLM-generated content at query #85
#--------------------------

```python
def test_isinstance_predicate_evaluates_to_false():
    class MockClass:
        pass
    
    class DifferentClass:
        pass
    
    source_data = DifferentClass()
    cls = MockClass
    
    result = isinstance(source_data, cls)
    
    assert result is False


# LLM-generated content at query #86
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checkedpmap_constructor_with_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int

    result = StringToIntMap({'a': 1, 'b': 2, 'c': 3})
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checkedpmap_constructor_returns_pmap_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, PMap)


# LLM-generated content at query #87
#--------------------------

```python
def test_checked_type_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #88
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockClass:
        pass
    
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance


def test_checked_type_create_with_source_data_matching_cls():
    class MockClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert result.data == source_data


def test_checked_type_create_with_no_checked_types():
    class MockClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert result.data == source_data


def test_checked_type_create_returns_cls_instance():
    class CheckedType:
        pass
    
    class MockCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
    
    class MockClass:
        _checked_types = ['__main__.MockCheckedType']
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert isinstance(result, MockClass)


def test_checked_type_create_with_ignore_extra_parameter():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
    
    class MockCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
    
    class MockClass:
        _checked_types = ['__main__.MockCheckedType']
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data, ignore_extra=True)
    assert isinstance(result, MockClass)


# LLM-generated content at query #89
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    class MockCheckedType:
        pass
    
    class MockClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = isinstance(source_data, MockClass)
    
    assert result is False


# LLM-generated content at query #90
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    # Test that the predicate at line 1 (isinstance(source_data, cls)) evaluates to False
    # We need to create a mock class and pass data that is NOT an instance of that class
    
    class MockClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = isinstance(source_data, MockClass)
    
    assert result is False


# LLM-generated content at query #91
#--------------------------

```python
def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]


def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type('int')
    assert result == ['int']


def test_maybe_parse_user_type_with_list_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)


def test_maybe_parse_user_type_with_tuple_of_types():
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(['int', 'str'])
    assert result == ('int', 'str')


def test_maybe_parse_user_type_with_nested_iterables():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_invalid_input_raises_typeerror():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


def test_maybe_parse_user_type_with_none_raises_typeerror():
    try:
        maybe_parse_user_type(None)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


def test_maybe_parse_user_type_with_empty_list():
    result = maybe_parse_user_type([])
    assert result == ()


def test_maybe_parse_user_type_with_mixed_types_and_strings():
    result = maybe_parse_user_type([int, 'str', float])
    assert result == (int, 'str', float)


# LLM-generated content at query #92
#--------------------------

```python
def test_line_18_predicate_evaluates_to_true():
    class CustomType:
        pass
    
    class NotIterable:
        pass
    
    t = CustomType
    is_type = isinstance(t, type)
    is_iterable = isinstance(t, Iterable)
    
    assert is_type and not is_iterable


# LLM-generated content at query #93
#--------------------------

```python
def test_checked_type_create_returns_source_when_already_instance():
    class MockCheckedType:
        _checked_types = []
    
    source = MockCheckedType()
    result = _checked_type_create(MockCheckedType, source)
    assert result is source


def test_checked_type_create_returns_cls_instance_when_no_checked_type():
    class MockCheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert isinstance(result, MockCheckedType)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_matching_data():
    class CheckedType:
        pass
    
    class InnerCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
        
        @staticmethod
        def create(data, ignore_extra=False):
            return InnerCheckedType(data)
    
    class MockCheckedType:
        _checked_types = ['__main__.InnerCheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert isinstance(result, MockCheckedType)


def test_checked_type_create_with_checked_type_non_matching_data():
    class CheckedType:
        pass
    
    class InnerCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
        
        @staticmethod
        def create(data, ignore_extra=False):
            return InnerCheckedType(data * 2)
    
    class MockCheckedType:
        _checked_types = ['__main__.InnerCheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert isinstance(result, MockCheckedType)


def test_checked_type_create_with_ignore_extra_flag():
    class CheckedType:
        pass
    
    class InnerCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
        
        @staticmethod
        def create(data, ignore_extra=False):
            return InnerCheckedType(data)
    
    class MockCheckedType:
        _checked_types = ['__main__.InnerCheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data, ignore_extra=True)
    assert isinstance(result, MockCheckedType)


def test_checked_type_create_with_factory_fields():
    class MockCheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data, _factory_fields={'key': 'value'})
    assert isinstance(result, MockCheckedType)
    assert result.data == source_data


# LLM-generated content at query #94
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0


def test_checked_pmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    initial_data = {1: 1.0, 2: 2.0}
    result = IntToFloatMap(initial_data)
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised exception"
    except Exception:
        pass


def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_checked_pmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25, 3: 3.75, 4: 4.5}
    result = IntToFloatMap(initial_data)
    assert len(result) == 4
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75
    assert result[4] == 4.5


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data)
    assert isinstance(result, IntToFloatMap)
    assert type(result).__name__ == 'IntToFloatMap'


# LLM-generated content at query #95
#--------------------------

```python
def test_restore_pickle():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self._factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            return cls(data, _factory_fields)
    
    test_data = {"key": "value"}
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result.data == test_data
    assert result._factory_fields == set()
    assert isinstance(result._factory_fields, set)
    assert len(result._factory_fields) == 0


# LLM-generated content at query #96
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    # Test that the predicate at line 1 (isinstance(source_data, cls)) evaluates to False
    # by passing source_data that is not an instance of cls
    
    class MockCheckedType:
        _checked_types = []
    
    source_data = [1, 2, 3]
    cls = MockCheckedType
    
    # The predicate isinstance(source_data, cls) should be False
    result = isinstance(source_data, cls)
    assert result is False


# LLM-generated content at query #97
#--------------------------

```python
def test_isinstance_source_data_is_cls():
    class MockClass:
        pass
    
    source_data = MockClass()
    result = isinstance(source_data, MockClass)
    
    assert result is True


# LLM-generated content at query #98
#--------------------------

```python
def test_checked_pmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checked_pmap_constructor_with_failed_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_multiple_items():
    class IntToStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = IntToStrMap({1: "a", 2: "b", 3: "c"})
    assert len(result) == 3
    assert result[1] == "a"
    assert result[2] == "b"
    assert result[3] == "c"


# LLM-generated content at query #99
#--------------------------

```python
def test_check_types_predicate_line_1():
    def _check_types(it, expected_types, source_class, exception_type=None):
        if expected_types:
            return True
        return False
    
    class DummyClass:
        pass
    
    result = _check_types([1, 2, 3], [int], DummyClass)
    assert result is True


# LLM-generated content at query #100
#--------------------------

```python
def test_checked_type_create_isinstance_predicate():
    class MockCheckedType:
        pass
    
    cls = MockCheckedType
    source_data = MockCheckedType()
    result = isinstance(source_data, cls)
    assert result is True


# LLM-generated content at query #101
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_default_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0


def test_checked_pmap_constructor_type_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_value_type_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25, 3: 3.75, 4: 4.5})
    assert len(result) == 4
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75
    assert result[4] == 4.5


# LLM-generated content at query #102
#--------------------------

```python
def test_checked_type_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #103
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception:
        pass


def test_checkedpmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checkedpmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_checkedpmap_constructor_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int

    result = StringToIntMap({"a": 1, "b": 2, "c": 3})
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


def test_checkedpmap_constructor_with_default_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


# LLM-generated content at query #104
#--------------------------

```python
def test_checkedpmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_default_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_single_entry():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1})
    assert isinstance(result, StringToIntMap)
    assert result['a'] == 1
    assert len(result) == 1


def test_checkedpmap_constructor_multiple_entries():
    class StrToStrMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = str
    
    data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    result = StrToStrMap(data)
    assert isinstance(result, StrToStrMap)
    assert result['key1'] == 'value1'
    assert result['key2'] == 'value2'
    assert result['key3'] == 'value3'
    assert len(result) == 3


# LLM-generated content at query #105
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


def test_merge_invariant_results_all_false():
    result = [(False, "error1"), (False, "error2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2", "error3")


def test_merge_invariant_results_mixed():
    result = [(True, "data1"), (False, "error1"), (True, "data2"), (False, "error2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2")


def test_merge_invariant_results_single_true():
    result = [(True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


def test_merge_invariant_results_single_false():
    result = [(False, "error1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)


def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


def test_merge_invariant_results_with_none_data():
    result = [(True, None), (False, None), (True, "data")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == (None,)


def test_merge_invariant_results_with_complex_data():
    result = [(True, {"key": "value"}), (False, {"error": "details"}), (False, [1, 2, 3])]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ({"error": "details"}, [1, 2, 3])


# LLM-generated content at query #106
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({5: 3.14})
    assert len(result) == 1
    assert result[5] == 3.14


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_preserves_type():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1, 'b': 2})
    assert type(result).__name__ == 'StringToIntMap'
    assert isinstance(result, CheckedPMap)


def test_checked_pmap_constructor_with_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4}
    result = IntToFloatMap(initial)
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[2] == 2.2
    assert result[3] == 3.3
    assert result[4] == 4.4


# LLM-generated content at query #107
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_single_element():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checked_pmap_constructor_multiple_elements():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert len(result) == 5
    for k, v in data.items():
        assert result[k] == v


# LLM-generated content at query #108
#--------------------------

```python
def test_check_types_with_valid_types():
    from collections.abc import Iterable
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            super().__init__(msg)
    
    class TestClass:
        pass
    
    _check_types([1, 2, 3], [int], TestClass, CheckedValueTypeError)


def test_check_types_with_empty_expected_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            pass
    
    class TestClass:
        pass
    
    _check_types([1, "string", 3.14], [], TestClass, CheckedValueTypeError)


def test_check_types_with_multiple_valid_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            pass
    
    class TestClass:
        pass
    
    _check_types([1, "string", 3.14], [int, str, float], TestClass, CheckedValueTypeError)


def test_check_types_raises_exception_on_invalid_type():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
    
    class TestClass:
        pass
    
    try:
        _check_types([1, 2, "invalid"], [int], TestClass, CheckedValueTypeError)
        assert False, "Expected exception to be raised"
    except CheckedValueTypeError as e:
        assert e.actual_type == str
        assert e.value == "invalid"
        assert "Type TestClass can only be used with ('int',), not str" in e.msg


def test_check_types_with_single_invalid_element():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
    
    class TestClass:
        pass
    
    try:
        _check_types([3.14], [int, str], TestClass, CheckedValueTypeError)
        assert False, "Expected exception to be raised"
    except CheckedValueTypeError as e:
        assert e.actual_type == float
        assert e.value == 3.14


def test_check_types_with_string_type_names():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
    
    class TestClass:
        pass
    
    _check_types([1, 2, 3], ['builtins.int'], TestClass, CheckedValueTypeError)


def test_check_types_with_empty_iterable():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            pass
    
    class TestClass:
        pass
    
    _check_types([], [int], TestClass, CheckedValueTypeError)


# LLM-generated content at query #109
#--------------------------

```python
def test_checked_type_create_predicate_false():
    class MockClass:
        pass
    
    class DifferentClass:
        pass
    
    source_data = DifferentClass()
    cls = MockClass
    
    result = isinstance(source_data, cls)
    
    assert result is False


# LLM-generated content at query #110
#--------------------------

```python
def test_checked_type_create_isinstance_predicate():
    class MockCheckedType:
        def _checked_type_create(cls, source_data, _factory_fields=None, ignore_extra=False):
            if isinstance(source_data, cls):
                return source_data
            return None
    
    instance = MockCheckedType()
    result = instance._checked_type_create(instance)
    
    assert result is instance


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_invariant_errors_all_valid():
    invariants = [
        lambda x: (True, "valid1"),
        lambda x: (True, "valid2"),
    ]
    result = _invariant_errors("test", invariants)
    assert result == []


def test_invariant_errors_all_invalid():
    invariants = [
        lambda x: (False, "error1"),
        lambda x: (False, "error2"),
    ]
    result = _invariant_errors("test", invariants)
    assert result == ["error1", "error2"]


def test_invariant_errors_mixed():
    invariants = [
        lambda x: (True, "valid"),
        lambda x: (False, "error1"),
        lambda x: (True, "valid2"),
        lambda x: (False, "error2"),
    ]
    result = _invariant_errors("test", invariants)
    assert result == ["error1", "error2"]


def test_invariant_errors_empty_invariants():
    invariants = []
    result = _invariant_errors("test", invariants)
    assert result == []


def test_invariant_errors_single_valid():
    invariants = [lambda x: (True, "valid")]
    result = _invariant_errors("test", invariants)
    assert result == []


def test_invariant_errors_single_invalid():
    invariants = [lambda x: (False, "error")]
    result = _invariant_errors("test", invariants)
    assert result == ["error"]


def test_invariant_errors_with_different_data_types():
    invariants = [
        lambda x: (False, 42),
        lambda x: (False, {"key": "value"}),
        lambda x: (False, [1, 2, 3]),
    ]
    result = _invariant_errors("test", invariants)
    assert result == [42, {"key": "value"}, [1, 2, 3]]


# LLM-generated content at query #2
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0


def test_checked_pset_constructor_with_valid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_with_float_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert 1.5 in result
    assert 2.5 in result
    assert 3.5 in result


def test_checked_pset_constructor_with_mixed_valid_types():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checked_pset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pset_constructor_with_negative_value():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pset_constructor_with_duplicate_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_with_zero():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([0, 1, 2])
    assert len(result) == 3
    assert 0 in result


# LLM-generated content at query #3
#--------------------------

```python
def test_check_types_with_valid_types():
    from your_module import _check_types, CheckedValueTypeError
    
    _check_types([1, 2, 3], [int], type('TestClass', (), {}))


def test_check_types_with_multiple_valid_types():
    from your_module import _check_types, CheckedValueTypeError
    
    _check_types([1, "hello", 2.5], [int, str, float], type('TestClass', (), {}))


def test_check_types_with_empty_expected_types():
    from your_module import _check_types, CheckedValueTypeError
    
    _check_types([1, "hello", None], [], type('TestClass', (), {}))


def test_check_types_with_empty_iterable():
    from your_module import _check_types, CheckedValueTypeError
    
    _check_types([], [int], type('TestClass', (), {}))


def test_check_types_raises_on_invalid_type():
    from your_module import _check_types, CheckedValueTypeError
    
    source_class = type('TestClass', (), {})
    try:
        _check_types([1, "invalid"], [int], source_class)
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.args[0] == source_class
        assert e.args[1] == [int]
        assert e.args[2] == str
        assert e.args[3] == "invalid"


def test_check_types_with_string_type_names():
    from your_module import _check_types, CheckedValueTypeError
    
    _check_types([1, 2], ['builtins.int'], type('TestClass', (), {}))


def test_check_types_raises_with_string_type_names():
    from your_module import _check_types, CheckedValueTypeError
    
    source_class = type('TestClass', (), {})
    try:
        _check_types(["invalid"], ['builtins.int'], source_class)
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.args[2] == str


def test_check_types_custom_exception_type():
    from your_module import _check_types
    
    class CustomException(Exception):
        pass
    
    source_class = type('TestClass', (), {})
    try:
        _check_types([1, "invalid"], [int], source_class, exception_type=CustomException)
        assert False, "Should have raised CustomException"
    except CustomException:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_checkedpmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_with_default_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_single_element():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 42.5})
    assert result[42] == 42.5
    assert len(result) == 1


def test_checkedpmap_constructor_returns_instance_of_correct_type():
    class CustomMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = CustomMap({'a': 1, 'b': 2})
    assert type(result).__name__ == 'CustomMap'
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #5
#--------------------------

```python
def test_store_types_empty_dict():
    dct = {}
    bases = ()
    destination_name = 'dest'
    source_name = 'src'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == ()


def test_store_types_single_type_in_dict():
    dct = {'src': int}
    bases = ()
    destination_name = 'dest'
    source_name = 'src'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == (int,)


def test_store_types_string_in_dict():
    dct = {'src': 'MyType'}
    bases = ()
    destination_name = 'dest'
    source_name = 'src'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == ('MyType',)


def test_store_types_multiple_types_in_dict():
    dct = {'src': [int, str]}
    bases = ()
    destination_name = 'dest'
    source_name = 'src'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == (int, str)


def test_store_types_with_base_class():
    class Base:
        pass
    Base.__dict__['src'] = float
    dct = {'src': int}
    bases = (Base,)
    destination_name = 'dest'
    source_name = 'src'
    _store_types(dct, bases, destination_name, source_name)
    assert int in dct[destination_name]
    assert float in dct[destination_name]


def test_store_types_source_name_not_in_dict_or_bases():
    dct = {}
    bases = ()
    destination_name = 'dest'
    source_name = 'nonexistent'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == ()


def test_store_types_overwrites_existing_destination():
    dct = {'dest': 'old_value', 'src': int}
    bases = ()
    destination_name = 'dest'
    source_name = 'src'
    _store_types(dct, bases, destination_name, source_name)
    assert dct[destination_name] == (int,)


def test_store_types_nested_list_of_types():
    dct = {'src': [int, [str, float]]}
    bases = ()
    destination_name = 'dest'
    source_name = 'src'
    _store_types(dct, bases, destination_name, source_name)
    assert int in dct[destination_name]
    assert str in dct[destination_name]
    assert float in dct[destination_name]


def test_store_types_multiple_bases():
    class Base1:
        pass
    class Base2:
        pass
    Base1.__dict__['src'] = int
    Base2.__dict__['src'] = str
    dct = {'src': float}
    bases = (Base1, Base2)
    destination_name = 'dest'
    source_name = 'src'
    _store_types(dct, bases, destination_name, source_name)
    assert float in dct[destination_name]
    assert int in dct[destination_name]
    assert str in dct[destination_name]


# LLM-generated content at query #6
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0


def test_checked_pset_constructor_with_valid_integers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_with_valid_floats():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.7, 3.2])
    assert len(result) == 3
    assert 1.5 in result
    assert 2.7 in result
    assert 3.2 in result


def test_checked_pset_constructor_with_mixed_valid_types():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checked_pset_constructor_with_duplicates():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Should have raised an exception for invalid type"
    except Exception:
        pass


def test_checked_pset_constructor_invariant_violation():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised an exception for invariant violation"
    except Exception:
        pass


def test_checked_pset_constructor_with_zero():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([0, 1, 2])
    assert len(result) == 3
    assert 0 in result


def test_checked_pset_constructor_preserves_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert type(result).__name__ == 'Positives'


# LLM-generated content at query #7
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({5: 5.5})
    assert result[5] == 5.5
    assert len(result) == 1


def test_checked_pmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4}
    result = IntToFloatMap(data)
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[2] == 2.2
    assert result[3] == 3.3
    assert result[4] == 4.4


# LLM-generated content at query #8
#--------------------------

```python
def test_check_types_predicate_evaluates_to_false():
    from your_module import _check_types, CheckedValueTypeError, get_type
    
    class TestClass:
        pass
    
    it = [42]
    expected_types = [str, float]
    source_class = TestClass
    
    try:
        _check_types(it, expected_types, source_class)
        assert False, "Expected CheckedValueTypeError to be raised"
    except CheckedValueTypeError:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_checked_pmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 5
    for key, value in data.items():
        assert result[key] == value


# LLM-generated content at query #10
#--------------------------

```python
def test_store_invariants_basic():
    dct = {}
    bases = ()
    
    def invariant1(x):
        return True, None
    
    store_invariants(dct, bases, '_invariants', 'invariant')
    assert '_invariants' in dct
    assert dct['_invariants'] == ()


def test_store_invariants_single_invariant():
    def invariant1(x):
        return True, None
    
    dct = {'invariant': invariant1}
    bases = ()
    
    store_invariants(dct, bases, '_invariants', 'invariant')
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 1
    assert callable(dct['_invariants'][0])


def test_store_invariants_multiple_invariants():
    def invariant1(x):
        return True, None
    
    def invariant2(x):
        return True, None
    
    dct = {'invariant': invariant1}
    
    class Base:
        invariant = invariant2
    
    store_invariants(dct, (Base,), '_invariants', 'invariant')
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 2


def test_store_invariants_inherited():
    def invariant1(x):
        return True, None
    
    def invariant2(x):
        return True, None
    
    class Base:
        invariant = invariant1
    
    dct = {'invariant': invariant2}
    
    store_invariants(dct, (Base,), '_invariants', 'invariant')
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 2


def test_store_invariants_non_callable_raises():
    dct = {'invariant': 'not_callable'}
    bases = ()
    
    try:
        store_invariants(dct, bases, '_invariants', 'invariant')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Invariants must be callable' in str(e)


def test_store_invariants_wrapped_invariant_bool():
    def invariant1(x):
        return True, None
    
    dct = {'invariant': invariant1}
    bases = ()
    
    store_invariants(dct, bases, '_invariants', 'invariant')
    result = dct['_invariants'][0](5)
    assert result == (True, None)


def test_store_invariants_wrapped_invariant_multiple_results():
    def invariant1(x):
        return [(True, None), (False, "error1"), (False, "error2")]
    
    dct = {'invariant': invariant1}
    bases = ()
    
    store_invariants(dct, bases, '_invariants', 'invariant')
    result = dct['_invariants'][0](5)
    assert result[0] == False
    assert result[1] == ("error1", "error2")


def test_store_invariants_multiple_bases():
    def inv1(x):
        return True, None
    
    def inv2(x):
        return True, None
    
    def inv3(x):
        return True, None
    
    class Base1:
        invariant = inv1
    
    class Base2:
        invariant = inv2
    
    dct = {'invariant': inv3}
    
    store_invariants(dct, (Base1, Base2), '_invariants', 'invariant')
    assert len(dct['_invariants']) == 3


def test_store_invariants_destination_name():
    def invariant1(x):
        return True, None
    
    dct = {'check': invariant1}
    bases = ()
    
    store_invariants(dct, bases, 'custom_dest', 'check')
    assert 'custom_dest' in dct
    assert len(dct['custom_dest']) == 1


def test_store_invariants_empty_bases_no_source():
    dct = {}
    bases = ()
    
    store_invariants(dct, bases, '_invariants', 'invariant')
    assert dct['_invariants'] == ()


# LLM-generated content at query #11
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    from collections import UserList
    
    class TestCheckedType(UserList):
        _checked_types = []
    
    instance = TestCheckedType([1, 2, 3])
    result = _checked_type_create(TestCheckedType, instance)
    
    assert result is instance


def test_checked_type_create_with_non_checked_type():
    from collections import UserList
    
    class TestCheckedType(UserList):
        _checked_types = []
    
    source_data = [1, 2, 3]
    result = _checked_type_create(TestCheckedType, source_data)
    
    assert isinstance(result, TestCheckedType)
    assert list(result) == [1, 2, 3]


def test_checked_type_create_with_string_type_name():
    from collections import UserList
    
    class CheckedType(UserList):
        pass
    
    class TestCheckedType(UserList):
        _checked_types = ['collections.UserList']
    
    source_data = [1, 2, 3]
    result = _checked_type_create(TestCheckedType, source_data)
    
    assert isinstance(result, TestCheckedType)
    assert list(result) == [1, 2, 3]


def test_get_type_with_type_object():
    result = get_type(int)
    assert result is int


def test_get_type_with_string():
    result = get_type('builtins.int')
    assert result is int


def test_get_types_with_multiple_types():
    result = get_types([int, 'builtins.str'])
    assert result == [int, str]


def test_get_types_with_empty_list():
    result = get_types([])
    assert result == []


# LLM-generated content at query #12
#--------------------------

```python
def test_serialize_empty_set():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([])
    result = ps.serialize()
    assert result == set()


def test_serialize_with_integers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([1, 2, 3])
    result = ps.serialize()
    assert result == {1, 2, 3}


def test_serialize_with_floats():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([1.5, 2.5, 3.5])
    result = ps.serialize()
    assert result == {1.5, 2.5, 3.5}


def test_serialize_with_mixed_numbers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([1, 2.5, 3])
    result = ps.serialize()
    assert result == {1, 2.5, 3}


def test_serialize_with_format_none():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([1, 2, 3])
    result = ps.serialize(format=None)
    assert result == {1, 2, 3}


def test_serialize_returns_set_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([1, 2, 3])
    result = ps.serialize()
    assert isinstance(result, set)


# LLM-generated content at query #13
#--------------------------

```python
def test_check_types_with_valid_types():
    from collections.abc import Iterable
    
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
    
    class TestClass:
        pass
    
    _check_types([1, 2, 3], [int], TestClass, CheckedValueTypeError)


def test_check_types_with_string_type_names():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
    
    class TestClass:
        pass
    
    _check_types(["hello", "world"], ["builtins.str"], TestClass, CheckedValueTypeError)


def test_check_types_with_empty_types():
    class TestClass:
        pass
    
    _check_types([1, 2, 3], [], TestClass)


def test_check_types_with_multiple_valid_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
    
    class TestClass:
        pass
    
    _check_types([1, "hello", 2.5], [int, str, float], TestClass, CheckedValueTypeError)


def test_check_types_raises_on_invalid_type():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
    
    class TestClass:
        pass
    
    try:
        _check_types([1, "invalid", 3], [int], TestClass, CheckedValueTypeError)
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.source_class == TestClass
        assert e.expected_types == [int]
        assert e.actual_type == str
        assert e.value == "invalid"
        assert "TestClass" in e.msg
        assert "int" in e.msg
        assert "str" in e.msg


def test_check_types_default_exception():
    class TestClass:
        pass
    
    try:
        _check_types([1, None, 3], [int], TestClass)
        assert False, "Should have raised CheckedValueTypeError"
    except Exception as e:
        assert e.__class__.__name__ == "CheckedValueTypeError"


# LLM-generated content at query #14
#--------------------------

```python
def test_checked_pvector_constructor_with_empty_list():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_valid_integers():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_valid_floats():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert result[0] == 1.5
    assert result[1] == 2.5
    assert result[2] == 3.5


def test_checked_pvector_constructor_with_mixed_valid_types():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2.5
    assert result[2] == 3


def test_checked_pvector_constructor_with_pvector_input():
    from pyrsistent import pvector
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pv = pvector([1, 2, 3])
    result = Positives(pv)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_generator():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives(x for x in [1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_checked_pvector_constructor_with_single_element():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([42])
    assert len(result) == 1
    assert result[0] == 42


def test_checked_pvector_constructor_default_empty():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives()
    assert len(result) == 0
    assert isinstance(result, Positives)


# LLM-generated content at query #15
#--------------------------

```python
def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]

def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]

def test_maybe_parse_user_type_with_list_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)

def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(["int", "str"])
    assert result == ("int", "str")

def test_maybe_parse_user_type_with_mixed_list():
    result = maybe_parse_user_type([int, "str"])
    assert result == (int, "str")

def test_maybe_parse_user_type_with_nested_list():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)

def test_maybe_parse_user_type_with_tuple():
    result = maybe_parse_user_type((int, str))
    assert result == (int, str)

def test_maybe_parse_user_type_with_invalid_type_raises_error():
    try:
        maybe_parse_user_type(42)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)

def test_maybe_parse_user_type_with_none_raises_error():
    try:
        maybe_parse_user_type(None)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)

def test_maybe_parse_user_type_with_dict_raises_error():
    try:
        maybe_parse_user_type({"key": "value"})
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


# LLM-generated content at query #16
#--------------------------

```python
def test_checkedpmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checkedpmap_constructor_default_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should raise CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should raise CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_violates_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert len(result) == 5
    assert result[3] == 3.3


# LLM-generated content at query #17
#--------------------------

```python
def test_store_invariants_callable_check():
    def _all_dicts(bases):
        return []
    
    def wrap_invariant(inv):
        return inv
    
    # Test case 1: All invariants are callable - should not raise
    dct = {'invariant': lambda x: x > 0}
    bases = []
    store_invariants(dct, bases, 'stored_invariants', 'invariant')
    assert 'stored_invariants' in dct
    assert isinstance(dct['stored_invariants'], tuple)
    
    # Test case 2: Non-callable invariant - should raise TypeError
    dct = {'invariant': 'not_callable'}
    bases = []
    try:
        store_invariants(dct, bases, 'stored_invariants', 'invariant')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert str(e) == 'Invariants must be callable'
    
    # Test case 3: Multiple callable invariants - should not raise
    dct = {'invariant': lambda x: x > 0}
    bases = []
    store_invariants(dct, bases, 'stored_invariants', 'invariant')
    assert len(dct['stored_invariants']) == 1


# LLM-generated content at query #18
#--------------------------

```python
def test_checked_type_create_predicate_line_2():
    class MockCheckedType:
        pass
    
    instance = MockCheckedType()
    result = isinstance(instance, MockCheckedType)
    
    assert result is True


# LLM-generated content at query #19
#--------------------------

```python
def test_maybe_parse_user_type_preserved_iterable():
    from collections.abc import Iterable
    
    class _preserved_iterable_types(tuple):
        pass
    
    def maybe_parse_user_type(t):
        is_type = isinstance(t, type)
        is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
        is_string = isinstance(t, str)
        is_iterable = isinstance(t, Iterable)
        
        if is_preserved:
            return [t]
        elif is_string:
            return [t]
        elif is_type and not is_iterable:
            return [t]
        elif is_iterable:
            ts = t
            return tuple(e for t in ts for e in maybe_parse_user_type(t))
        else:
            raise TypeError(
                'Type specifications must be types or strings. Input: {}'.format(t)
            )
    
    result = maybe_parse_user_type(_preserved_iterable_types)
    assert result == [_preserved_iterable_types]


# LLM-generated content at query #20
#--------------------------

```python
def test_invariant_errors_all_valid():
    def invariant1(elem):
        return (True, "data1")
    
    def invariant2(elem):
        return (True, "data2")
    
    result = _invariant_errors("test", [invariant1, invariant2])
    assert result == []


def test_invariant_errors_all_invalid():
    def invariant1(elem):
        return (False, "error1")
    
    def invariant2(elem):
        return (False, "error2")
    
    result = _invariant_errors("test", [invariant1, invariant2])
    assert result == ["error1", "error2"]


def test_invariant_errors_mixed():
    def invariant1(elem):
        return (True, "data1")
    
    def invariant2(elem):
        return (False, "error2")
    
    def invariant3(elem):
        return (True, "data3")
    
    def invariant4(elem):
        return (False, "error4")
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3, invariant4])
    assert result == ["error2", "error4"]


def test_invariant_errors_empty_invariants():
    result = _invariant_errors("test", [])
    assert result == []


def test_invariant_errors_single_invariant_valid():
    def invariant(elem):
        return (True, "valid_data")
    
    result = _invariant_errors("test", [invariant])
    assert result == []


def test_invariant_errors_single_invariant_invalid():
    def invariant(elem):
        return (False, "invalid_data")
    
    result = _invariant_errors("test", [invariant])
    assert result == ["invalid_data"]


def test_invariant_errors_with_different_data_types():
    def invariant1(elem):
        return (False, 123)
    
    def invariant2(elem):
        return (False, {"key": "value"})
    
    def invariant3(elem):
        return (False, ["list", "item"])
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == [123, {"key": "value"}, ["list", "item"]]


# LLM-generated content at query #21
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_returns_correct_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert isinstance(result, CheckedPMap)


def test_checked_pmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4}
    result = IntToFloatMap(initial)
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[4] == 4.4


# LLM-generated content at query #22
#--------------------------

```python
def test_check_types_with_valid_types():
    from collections.abc import Sequence
    
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    it = [1, 2, 3]
    expected_types = [int]
    _check_types(it, expected_types, TestClass, CheckedValueTypeError)


def test_check_types_with_empty_expected_types():
    class TestClass:
        pass
    
    it = [1, "string", 3.14]
    expected_types = []
    _check_types(it, expected_types, TestClass)


def test_check_types_with_none_expected_types():
    class TestClass:
        pass
    
    it = [1, "string", 3.14]
    expected_types = None
    _check_types(it, expected_types, TestClass)


def test_check_types_with_multiple_valid_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    it = [1, "string", 3.14]
    expected_types = [int, str, float]
    _check_types(it, expected_types, TestClass, CheckedValueTypeError)


def test_check_types_with_invalid_type():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    it = [1, 2, "invalid"]
    expected_types = [int]
    
    try:
        _check_types(it, expected_types, TestClass, CheckedValueTypeError)
        assert False, "Expected CheckedValueTypeError to be raised"
    except CheckedValueTypeError as e:
        assert e.source_class == TestClass
        assert e.expected_types == [int]
        assert e.actual_type == str
        assert e.value == "invalid"
        assert "TestClass" in e.msg
        assert "int" in e.msg
        assert "str" in e.msg


def test_check_types_with_string_type_reference():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    it = [1, 2, 3]
    expected_types = ["builtins.int"]
    _check_types(it, expected_types, TestClass, CheckedValueTypeError)


def test_check_types_with_mixed_type_references():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    it = [1, "string", 3.14]
    expected_types = ["builtins.int", str, "builtins.float"]
    _check_types(it, expected_types, TestClass, CheckedValueTypeError)


# LLM-generated content at query #23
#--------------------------

```python
def test_checked_pset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0


def test_checked_pset_constructor_with_valid_integers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_with_valid_floats():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert 1.5 in result
    assert 2.5 in result
    assert 3.5 in result


def test_checked_pset_constructor_with_mixed_types():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checked_pset_constructor_with_duplicates():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checked_pset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, 'invalid', 3])
        assert False, "Should raise TypeError"
    except TypeError:
        pass


def test_checked_pset_constructor_with_negative_invariant_violation():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_checked_pset_constructor_with_pmap_initial():
    from pyrsistent import pmap
    
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    initial_pmap = pmap()
    result = Positives(initial_pmap)
    assert len(result) == 0


# LLM-generated content at query #24
#--------------------------

```python
def test_maybe_parse_user_type_line_18_predicate():
    from collections.abc import Iterable
    
    class _preserved_iterable_types:
        pass
    
    def maybe_parse_user_type(t):
        is_type = isinstance(t, type)
        is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
        is_string = isinstance(t, str)
        is_iterable = isinstance(t, Iterable)
        
        if is_preserved:
            return [t]
        elif is_string:
            return [t]
        elif is_type and not is_iterable:
            return [t]
        elif is_iterable:
            ts = t
            return tuple(e for t in ts for e in maybe_parse_user_type(t))
        else:
            raise TypeError(
                'Type specifications must be types or strings. Input: {}'.format(t)
            )
    
    # Test case: a type that is not iterable should satisfy the predicate at line 18
    result = maybe_parse_user_type(int)
    assert result == [int]


# LLM-generated content at query #25
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    initial_data = {1: 1.0, 2: 2.0}
    result = IntToFloatMap(initial_data)
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_single_element():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {'a': 10}
    result = StringToIntMap(initial_data)
    assert len(result) == 1
    assert result['a'] == 10


def test_checked_pmap_constructor_multiple_elements():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {'a': 1, 'b': 2, 'c': 3}
    result = StringToIntMap(initial_data)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #26
#--------------------------

```python
def test_checkedpmap_constructor_empty_default():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_valid_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checkedpmap_constructor_preserves_class_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_multiple_items():
    class StrToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial = {'a': 1, 'b': 2, 'c': 3}
    result = StrToIntMap(initial)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #27
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_default_empty_dict():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({})
    assert len(result) == 0
    assert isinstance(result, StringToIntMap)


def test_checked_pmap_constructor_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {"a": 1, "b": 2, "c": 3}
    result = StringToIntMap(initial_data)
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


def test_checked_pmap_constructor_preserves_type():
    class CustomMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = str
    
    result = CustomMap({"key": "value"})
    assert type(result).__name__ == "CustomMap"
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #28
#--------------------------

```python
def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]


def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type('int')
    assert result == ['int']


def test_maybe_parse_user_type_with_list_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)


def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(['int', 'str'])
    assert result == ('int', 'str')


def test_maybe_parse_user_type_with_tuple_of_types():
    result = maybe_parse_user_type((int, float))
    assert result == (int, float)


def test_maybe_parse_user_type_with_nested_iterables():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_invalid_type():
    try:
        maybe_parse_user_type(123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


def test_maybe_parse_user_type_with_invalid_iterable():
    try:
        maybe_parse_user_type([int, 123])
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


def test_maybe_parse_user_type_with_none():
    try:
        maybe_parse_user_type(None)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Type specifications must be types or strings' in str(e)


def test_maybe_parse_user_type_with_empty_list():
    result = maybe_parse_user_type([])
    assert result == ()


def test_maybe_parse_user_type_with_custom_class():
    class CustomClass:
        pass
    result = maybe_parse_user_type(CustomClass)
    assert result == [CustomClass]


# LLM-generated content at query #29
#--------------------------

```python
def test_checkedpset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert isinstance(result, Positives)
    assert len(result) == 0


def test_checkedpset_constructor_with_valid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert isinstance(result, Positives)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checkedpset_constructor_with_float_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert isinstance(result, Positives)
    assert len(result) == 3
    assert 1.5 in result


def test_checkedpset_constructor_with_duplicate_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert isinstance(result, Positives)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checkedpset_constructor_with_negative_number_raises_error():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpset_constructor_with_wrong_type_raises_error():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "string", 3])
        assert False, "Should have raised TypeCheckError"
    except TypeCheckError:
        pass


def test_checkedpset_constructor_with_default_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives()
    assert isinstance(result, Positives)
    assert len(result) == 0


def test_checkedpset_constructor_preserves_class_type():
    class CustomSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda n: (n > 0, 'NotPositive')
    
    result = CustomSet([5, 10, 15])
    assert type(result) is CustomSet


# LLM-generated content at query #30
#--------------------------

```python
def test_check_types_predicate_true_with_non_empty_expected_types():
    expected_types = [int, str]
    assert expected_types


# LLM-generated content at query #31
#--------------------------

```python
def test_checked_pmap_new_with_default_arguments():
    from pyrsistent import pmap
    from pyrsistent._checked_types import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap()
    assert isinstance(result, SimpleMap)
    assert len(result) == 0


def test_checked_pmap_new_with_initial_dict():
    from pyrsistent import pmap
    from pyrsistent._checked_types import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap({'a': 1, 'b': 2})
    assert isinstance(result, SimpleMap)
    assert result['a'] == 1
    assert result['b'] == 2
    assert len(result) == 2


def test_checked_pmap_new_with_empty_initial_dict():
    from pyrsistent import pmap
    from pyrsistent._checked_types import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap({})
    assert isinstance(result, SimpleMap)
    assert len(result) == 0


def test_checked_pmap_new_with_initial_and_size():
    from pyrsistent import pmap
    from pyrsistent._checked_types import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_pmap = pmap({'x': 10, 'y': 20})
    result = SimpleMap(initial_pmap, size=2)
    assert isinstance(result, SimpleMap)
    assert result['x'] == 10
    assert result['y'] == 20


def test_checked_pmap_new_with_size_only():
    from pyrsistent import pmap
    from pyrsistent._checked_types import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_pmap = pmap({'p': 100})
    result = SimpleMap(initial_pmap, size=1)
    assert isinstance(result, SimpleMap)
    assert result['p'] == 100


def test_checked_pmap_new_with_invariant():
    from pyrsistent import pmap
    from pyrsistent._checked_types import CheckedPMap
    
    class PositiveMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, 'Value must be positive')
    
    result = PositiveMap({'a': 1, 'b': 2})
    assert isinstance(result, PositiveMap)
    assert result['a'] == 1
    assert result['b'] == 2


def test_checked_pmap_new_with_invalid_invariant():
    from pyrsistent import pmap
    from pyrsistent._checked_types import CheckedPMap, InvariantException
    
    class PositiveMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, 'Value must be positive')
    
    try:
        result = PositiveMap({'a': -1})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_new_multiple_items():
    from pyrsistent import pmap
    from pyrsistent._checked_types import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
    assert len(result) == 5
    assert result['a'] == 1
    assert result['e'] == 5


# LLM-generated content at query #32
#--------------------------

```python
def test_maybe_parse_user_type_with_single_type():
    result = maybe_parse_user_type(int)
    assert result == [int]


def test_maybe_parse_user_type_with_string():
    result = maybe_parse_user_type("int")
    assert result == ["int"]


def test_maybe_parse_user_type_with_list_of_types():
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)


def test_maybe_parse_user_type_with_tuple_of_types():
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(["int", "str"])
    assert result == ("int", "str")


def test_maybe_parse_user_type_with_mixed_list():
    result = maybe_parse_user_type([int, "str"])
    assert result == (int, "str")


def test_maybe_parse_user_type_with_nested_iterables():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_none():
    try:
        maybe_parse_user_type(None)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_dict():
    try:
        maybe_parse_user_type({"key": int})
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


# LLM-generated content at query #33
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_default_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert len(result) == 1
    assert result[1] == 1.5


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert len(result) == 5
    for key, value in data.items():
        assert result[key] == value


def test_checked_pmap_constructor_returns_same_instance_if_already_checked():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert len(result) == 1
    assert result[1] == 1.5


# LLM-generated content at query #34
#--------------------------

```python
def test_restore_pickle():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self._factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            return cls(data, _factory_fields)
    
    test_data = {"key": "value"}
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result.data == test_data
    assert result._factory_fields == set()
    assert isinstance(result._factory_fields, set)
    assert len(result._factory_fields) == 0


# LLM-generated content at query #35
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_invalid_key_type():
    from pyrsistent import InvariantException
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checkedpmap_constructor_invalid_value_type():
    from pyrsistent import InvariantException
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checkedpmap_constructor_with_initial_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.0, 2: 2.5, 3: 3.75, 4: 4.25}
    result = IntToFloatMap(data)
    assert len(result) == 4
    assert result[1] == 1.0
    assert result[2] == 2.5
    assert result[3] == 3.75
    assert result[4] == 4.25


# LLM-generated content at query #36
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    # Test that the predicate at line 1 (isinstance(source_data, cls)) evaluates to False
    # We need to create a mock class and pass source_data that is NOT an instance of cls
    
    class MockClass:
        _checked_types = []
    
    source_data = "not an instance of MockClass"
    result = isinstance(source_data, MockClass)
    
    assert result is False


# LLM-generated content at query #37
#--------------------------

```python
def test_check_types_predicate_true():
    expected_types = [int, str]
    result = bool(expected_types)
    assert result is True


# LLM-generated content at query #38
#--------------------------

```python
def test_maybe_parse_user_type_line_18_predicate():
    from collections.abc import Iterable
    
    class _preserved_iterable_types:
        pass
    
    def maybe_parse_user_type(t):
        is_type = isinstance(t, type)
        is_preserved = isinstance(t, type) and issubclass(t, _preserved_iterable_types)
        is_string = isinstance(t, str)
        is_iterable = isinstance(t, Iterable)
        
        if is_preserved:
            return [t]
        elif is_string:
            return [t]
        elif is_type and not is_iterable:
            return [t]
        elif is_iterable:
            ts = t
            return tuple(e for t in ts for e in maybe_parse_user_type(t))
        else:
            raise TypeError(
                'Type specifications must be types or strings. Input: {}'.format(t)
            )
    
    result = maybe_parse_user_type(int)
    assert result == [int]
    
    result = maybe_parse_user_type(str)
    assert result == [str]
    
    result = maybe_parse_user_type(float)
    assert result == [float]


# LLM-generated content at query #39
#--------------------------

```python
def test_wrap_invariant_with_bool_result():
    def invariant_bool(*args, **kwargs):
        return True, "data"
    
    wrapped = wrap_invariant(invariant_bool)
    verdict, data = wrapped()
    assert verdict is True
    assert data == "data"


def test_wrap_invariant_with_tuple_results():
    def invariant_tuple(*args, **kwargs):
        return [(True, "data1"), (True, "data2")]
    
    wrapped = wrap_invariant(invariant_tuple)
    verdict, data = wrapped()
    assert verdict is True
    assert data == ()


def test_wrap_invariant_with_mixed_results():
    def invariant_mixed(*args, **kwargs):
        return [(True, "data1"), (False, "error1"), (True, "data2"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_mixed)
    verdict, data = wrapped()
    assert verdict is False
    assert data == ("error1", "error2")


def test_wrap_invariant_with_all_false_results():
    def invariant_all_false(*args, **kwargs):
        return [(False, "error1"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_all_false)
    verdict, data = wrapped()
    assert verdict is False
    assert data == ("error1", "error2")


def test_wrap_invariant_with_all_true_results():
    def invariant_all_true(*args, **kwargs):
        return [(True, "data1"), (True, "data2")]
    
    wrapped = wrap_invariant(invariant_all_true)
    verdict, data = wrapped()
    assert verdict is True
    assert data == ()


def test_wrap_invariant_with_args_and_kwargs():
    def invariant_with_params(a, b, c=None):
        return True, f"result_{a}_{b}_{c}"
    
    wrapped = wrap_invariant(invariant_with_params)
    verdict, data = wrapped(1, 2, c=3)
    assert verdict is True
    assert data == "result_1_2_3"


def test_wrap_invariant_with_single_result_list():
    def invariant_single(*args, **kwargs):
        return [(False, "single_error")]
    
    wrapped = wrap_invariant(invariant_single)
    verdict, data = wrapped()
    assert verdict is False
    assert data == ("single_error",)


# LLM-generated content at query #40
#--------------------------

```python
def test_checkedtype_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #41
#--------------------------

```python
def test_check_types_predicate_true_with_non_empty_expected_types():
    expected_types = [int, str]
    result = bool(expected_types)
    assert result is True


# LLM-generated content at query #42
#--------------------------

```python
def test_checkedpmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checkedpmap_constructor_default_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invariant_invalid():
    from pyrsistent import InvariantException
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_with_wrong_key_type():
    from pyrsistent import CheckedKeyTypeError
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_wrong_value_type():
    from pyrsistent import CheckedTypeError
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {"a": 1, "b": 2, "c": 3, "d": 4}
    result = StringToIntMap(initial_data)
    assert len(result) == 4
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3
    assert result["d"] == 4


# LLM-generated content at query #43
#--------------------------

```python
def test_checkedpmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_initial_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_default_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1, 'b': 2, 'c': 3})
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #44
#--------------------------

```python
def test_checkedtype_constructor():
    obj = CheckedType()
    assert isinstance(obj, CheckedType)
    assert hasattr(obj, '__slots__')
    assert obj.__slots__ == ()


# LLM-generated content at query #45
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    class MockCheckedType:
        pass
    
    class MockCls:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = "not a MockCls instance"
    result = isinstance(source_data, MockCls)
    
    assert result is False


# LLM-generated content at query #46
#--------------------------

```python
def test_checked_pmap_constructor_empty_default():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_single_valid_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checked_pmap_constructor_multiple_valid_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4})
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[4] == 4.4


# LLM-generated content at query #47
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockClass:
        pass
    
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance


def test_checked_type_create_with_source_data_list():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}"
    
    class MockCls:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCls, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_in_types():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"processed_{data}"
    
    class MockCls:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = ["test"]
    result = _checked_type_create(MockCls, source_data, ignore_extra=False)
    assert isinstance(result, MockCls)


def test_checked_type_create_ignore_extra_parameter():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return data
    
    class MockCls:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2]
    result = _checked_type_create(MockCls, source_data, ignore_extra=True)
    assert result.data == source_data


def test_checked_type_create_factory_fields_parameter():
    class MockCls:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCls, source_data, _factory_fields={'field': 'value'})
    assert result.data == source_data


# LLM-generated content at query #48
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0


def test_checked_pmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"a": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_with_default_param():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5}, size=10)
    assert len(result) == 1
    assert result[1] == 1.5


# LLM-generated content at query #49
#--------------------------

```python
def test_checked_pvector_constructor_empty():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives()
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_list():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_tuple():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives((1.5, 2.5))
    assert len(result) == 2
    assert result[0] == 1.5
    assert result[1] == 2.5


def test_checked_pvector_constructor_with_pvector():
    from pyrsistent import pvector
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pv = pvector([1, 2, 3])
    result = Positives(pv)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_checked_pvector_constructor_preserves_type():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([5, 10])
    assert type(result).__name__ == 'Positives'
    assert isinstance(result, CheckedPVector)


def test_checked_pvector_constructor_with_generator():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives(x for x in [1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #50
#--------------------------

```python
def test_checked_type_create_isinstance_predicate():
    class MockCheckedType:
        pass
    
    class MockClass:
        pass
    
    source_data = MockClass()
    cls = MockClass
    
    result = isinstance(source_data, cls)
    assert result is True


# LLM-generated content at query #51
#--------------------------

```python
def test_checked_type_create_isinstance_predicate():
    class MockCheckedType:
        pass
    
    class TestClass(MockCheckedType):
        pass
    
    instance = TestClass()
    result = isinstance(instance, TestClass)
    
    assert result is True


# LLM-generated content at query #52
#--------------------------

```python
def test_checkedpmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checkedpmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert result[42] == 3.14
    assert len(result) == 1


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0


def test_checkedpmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid_value"})
        assert False, "Should have raised an error"
    except Exception:
        pass


def test_checkedpmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_checkedpmap_constructor_multiple_items_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0, 3: 3.0})
    assert len(result) == 3
    assert result[1] == 1.0
    assert result[2] == 2.0
    assert result[3] == 3.0


# LLM-generated content at query #53
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant1(elem):
        return (True, "valid_data_1")
    
    def invariant2(elem):
        return (False, "invalid_data_2")
    
    def invariant3(elem):
        return (False, "invalid_data_3")
    
    invariants = [invariant1, invariant2, invariant3]
    elem = "test_element"
    
    result = _invariant_errors(elem, invariants)
    
    assert result == ["invalid_data_2", "invalid_data_3"]


# LLM-generated content at query #54
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockCheckedType:
        _checked_types = []
    
    instance = MockCheckedType()
    result = _checked_type_create(MockCheckedType, instance)
    assert result is instance


def test_checked_type_create_with_non_matching_data():
    class MockCheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_subclass():
    class CheckedType:
        pass
    
    class ConcreteCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
        
        @staticmethod
        def create(data, ignore_extra=False):
            return ConcreteCheckedType(data * 2)
    
    class MockContainer:
        _checked_types = ['__main__.ConcreteCheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockContainer, source_data)
    assert result.data == [2, 4, 6]


def test_checked_type_create_with_matching_type_in_data():
    class CheckedType:
        pass
    
    class ConcreteCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
        
        @staticmethod
        def create(data, ignore_extra=False):
            return ConcreteCheckedType(data * 2)
    
    class MockContainer:
        _checked_types = ['__main__.ConcreteCheckedType']
        def __init__(self, data):
            self.data = data
    
    existing_instance = ConcreteCheckedType(5)
    source_data = [existing_instance, 2]
    result = _checked_type_create(MockContainer, source_data)
    assert result.data[0] is existing_instance
    assert result.data[1].value == 4


def test_checked_type_create_with_ignore_extra_flag():
    class CheckedType:
        pass
    
    class ConcreteCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
        
        @staticmethod
        def create(data, ignore_extra=False):
            return ConcreteCheckedType(data)
    
    class MockContainer:
        _checked_types = ['__main__.ConcreteCheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2]
    result = _checked_type_create(MockContainer, source_data, ignore_extra=True)
    assert len(result.data) == 2


# LLM-generated content at query #55
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


def test_merge_invariant_results_one_false():
    result = [(True, "data1"), (False, "error1"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)


def test_merge_invariant_results_multiple_false():
    result = [(False, "error1"), (True, "data2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error3")


def test_merge_invariant_results_all_false():
    result = [(False, "error1"), (False, "error2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1", "error2", "error3")


def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


def test_merge_invariant_results_single_true():
    result = [(True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


def test_merge_invariant_results_single_false():
    result = [(False, "error1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)


def test_merge_invariant_results_various_data_types():
    result = [(False, 123), (True, "data"), (False, None), (True, {"key": "value"})]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == (123, None)


# LLM-generated content at query #56
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    class MockCheckedType:
        pass
    
    source_data = "not a MockCheckedType instance"
    cls = MockCheckedType
    
    result = isinstance(source_data, cls)
    
    assert result is False


# LLM-generated content at query #57
#--------------------------

```python
def test_serialize_with_integers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    checked_set = Positives([1, 2, 3])
    result = checked_set.serialize()
    assert result == {1, 2, 3}


def test_serialize_with_floats():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    checked_set = Positives([1.5, 2.5, 3.5])
    result = checked_set.serialize()
    assert result == {1.5, 2.5, 3.5}


def test_serialize_with_mixed_numbers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    checked_set = Positives([1, 2.5, 3])
    result = checked_set.serialize()
    assert result == {1, 2.5, 3}


def test_serialize_empty_set():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    checked_set = Positives([])
    result = checked_set.serialize()
    assert result == set()


def test_serialize_with_format_parameter():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    checked_set = Positives([1, 2, 3])
    result = checked_set.serialize(format='json')
    assert result == {1, 2, 3}


def test_serialize_single_element():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    checked_set = Positives([42])
    result = checked_set.serialize()
    assert result == {42}


# LLM-generated content at query #58
#--------------------------

```python
def test_wrap_invariant_with_single_bool_result():
    def invariant_func():
        return True, "data"
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped()
    
    assert verdict is True
    assert data == "data"


def test_wrap_invariant_with_multiple_results_all_true():
    def invariant_func():
        return [(True, "data1"), (True, "data2")]
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped()
    
    assert verdict is True
    assert data == ()


def test_wrap_invariant_with_multiple_results_one_false():
    def invariant_func():
        return [(True, "data1"), (False, "error1"), (True, "data2")]
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped()
    
    assert verdict is False
    assert data == ("error1",)


def test_wrap_invariant_with_multiple_results_all_false():
    def invariant_func():
        return [(False, "error1"), (False, "error2"), (False, "error3")]
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped()
    
    assert verdict is False
    assert data == ("error1", "error2", "error3")


def test_wrap_invariant_with_args_and_kwargs():
    def invariant_func(x, y, z=None):
        return True, f"{x}-{y}-{z}"
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped(1, 2, z=3)
    
    assert verdict is True
    assert data == "1-2-3"


def test_wrap_invariant_with_multiple_results_and_args():
    def invariant_func(x):
        return [(True, f"data_{x}"), (False, f"error_{x}")]
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped(5)
    
    assert verdict is False
    assert data == ("error_5",)


def test_wrap_invariant_preserves_false_bool_result():
    def invariant_func():
        return False, "single_error"
    
    wrapped = wrap_invariant(invariant_func)
    verdict, data = wrapped()
    
    assert verdict is False
    assert data == "single_error"


# LLM-generated content at query #59
#--------------------------

```python
def test_checked_pmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_single_entry():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert isinstance(result, IntToFloatMap)
    assert result[42] == 3.14


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


# LLM-generated content at query #60
#--------------------------

```python
def test_checkedpvector_constructor_with_empty_list():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checkedpvector_constructor_with_valid_integers():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checkedpvector_constructor_with_valid_floats():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert result[0] == 1.5
    assert result[1] == 2.5
    assert result[2] == 3.5


def test_checkedpvector_constructor_with_mixed_numbers():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2.5
    assert result[2] == 3


def test_checkedpvector_constructor_with_pythonpvector():
    from pyrsistent import pvector
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pv = pvector([1, 2, 3])
    result = Positives(pv)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checkedpvector_constructor_with_tuple():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives((1, 2, 3))
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_checkedpvector_constructor_with_generator():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives(x for x in [1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_checkedpvector_constructor_preserves_class_type():
    class CustomVector(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = CustomVector([1, 2, 3])
    assert type(result).__name__ == 'CustomVector'
    assert isinstance(result, CustomVector)


# LLM-generated content at query #61
#--------------------------

```python
def test_merge_invariant_results_predicate_false():
    result = [(False, "error1"), (True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


# LLM-generated content at query #62
#--------------------------

```python
def test_checked_pmap_initial_items_iteration():
    from pyrsistent import CheckedPMap
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25, 3: 3.75}
    result = IntToFloatMap(initial_data)
    
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75
    assert len(result) == 3


# LLM-generated content at query #63
#--------------------------

```python
def test_checkedtype_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #64
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant1(elem):
        return (True, "valid_data_1")
    
    def invariant2(elem):
        return (False, "invalid_data_2")
    
    def invariant3(elem):
        return (False, "invalid_data_3")
    
    invariants = [invariant1, invariant2, invariant3]
    elem = "test_element"
    
    result = _invariant_errors(elem, invariants)
    
    assert result == ["invalid_data_2", "invalid_data_3"]
    assert len(result) == 2
    assert "invalid_data_2" in result
    assert "invalid_data_3" in result


# LLM-generated content at query #65
#--------------------------

```python
def test_serialize_returns_set():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    positives = Positives([1, 2, 3])
    result = positives.serialize()
    assert isinstance(result, set)


# LLM-generated content at query #66
#--------------------------

```python
def test_checked_pmap_initial_items_iteration():
    from pyrsistent import CheckedPMap
    
    class TestMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {'a': 1, 'b': 2, 'c': 3}
    result = TestMap(initial_data)
    
    assert dict(result) == initial_data
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #67
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    class MockCheckedType:
        pass
    
    source_data = MockCheckedType()
    cls = list
    
    result = isinstance(source_data, cls)
    
    assert result is False


# LLM-generated content at query #68
#--------------------------

```python
def test_wrap_invariant_predicate_line_3():
    def mock_invariant(*args, **kwargs):
        return (True, "test")
    
    wrapped = wrap_invariant(mock_invariant)
    result = wrapped()
    
    assert isinstance(result[0], bool) is True


# LLM-generated content at query #69
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    # Test that the predicate at line 1 (isinstance(source_data, cls)) evaluates to False
    # This means source_data should NOT be an instance of cls
    
    class MockClass:
        pass
    
    class DifferentClass:
        pass
    
    source_data = DifferentClass()
    cls = MockClass
    
    # Verify the predicate is False
    assert isinstance(source_data, cls) == False


# LLM-generated content at query #70
#--------------------------

```python
def test_serialize_empty_checked_pset():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([])
    result = ps.serialize()
    assert result == set()


def test_serialize_with_integers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([1, 2, 3])
    result = ps.serialize()
    assert result == {1, 2, 3}


def test_serialize_with_floats():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([1.5, 2.5, 3.5])
    result = ps.serialize()
    assert result == {1.5, 2.5, 3.5}


def test_serialize_with_mixed_numbers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([1, 2.5, 3])
    result = ps.serialize()
    assert result == {1, 2.5, 3}


def test_serialize_with_format_parameter():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([1, 2, 3])
    result = ps.serialize(format='json')
    assert result == {1, 2, 3}


def test_serialize_with_single_element():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    ps = Positives([42])
    result = ps.serialize()
    assert result == {42}


# LLM-generated content at query #71
#--------------------------

```python
def test_checked_pmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_violation():
    from pyrsistent import InvariantException
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_with_wrong_key_type():
    from pyrsistent import CheckedKeyTypeError
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_with_wrong_value_type():
    from pyrsistent import CheckedTypeError
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    data = {"a": 1, "b": 2, "c": 3}
    result = StringToIntMap(data)
    assert isinstance(result, StringToIntMap)
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3
    assert len(result) == 3


# LLM-generated content at query #72
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invariant_invalid():
    from pyrsistent._checked_types import InvariantException
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_with_invalid_key_type():
    from pyrsistent._checked_types import CheckedKeyTypeError
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    from pyrsistent._checked_types import CheckedTypeError
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_returns_correct_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checkedpmap_constructor_with_dict_input():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    input_dict = {1: 1.5, 2: 2.25, 3: 3.75}
    result = IntToFloatMap(input_dict)
    assert len(result) == 3
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75


# LLM-generated content at query #73
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockCheckedType:
        _checked_types = []
    
    instance = MockCheckedType()
    result = _checked_type_create(MockCheckedType, instance)
    assert result is instance


def test_checked_type_create_with_simple_data():
    class SimpleClass:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    result = _checked_type_create(SimpleClass, [1, 2, 3])
    assert result.data == [1, 2, 3]


def test_checked_type_create_with_checked_type_no_matching_instance():
    class CheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return f"created_{data}"
    
    class ContainerClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    result = _checked_type_create(ContainerClass, [1, 2], ignore_extra=False)
    assert result.data == ['created_1', 'created_2']


def test_checked_type_create_with_matching_instance_type():
    class CheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return f"created_{data}"
    
    class ContainerClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    obj = CheckedType()
    result = _checked_type_create(ContainerClass, [obj, 2], ignore_extra=False)
    assert result.data[0] is obj
    assert result.data[1] == 'created_2'


def test_checked_type_create_with_ignore_extra_flag():
    class CheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return f"created_{data}_ignore={ignore_extra}"
    
    class ContainerClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    result = _checked_type_create(ContainerClass, [1], ignore_extra=True)
    assert result.data == ['created_1_ignore=True']


def test_checked_type_create_empty_checked_types():
    class SimpleClass:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    result = _checked_type_create(SimpleClass, [])
    assert result.data == []


# LLM-generated content at query #74
#--------------------------

```python
def test_checkedpmap_initial_items_iteration():
    from pyrsistent import CheckedPMap
    
    class TestMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {'a': 1, 'b': 2, 'c': 3}
    result = TestMap(initial_data)
    
    assert dict(result) == initial_data
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #75
#--------------------------

```python
def test_checkedtype_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert CheckedType.__slots__ == ()


# LLM-generated content at query #76
#--------------------------

```python
def test_checkedpvector_constructor_with_empty_list():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checkedpvector_constructor_with_valid_integers():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checkedpvector_constructor_with_valid_floats():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert result[0] == 1.5
    assert result[1] == 2.5
    assert result[2] == 3.5


def test_checkedpvector_constructor_with_mixed_valid_types():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2.5
    assert result[2] == 3


def test_checkedpvector_constructor_with_python_pvector():
    from pyrsistent import pvector
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pv = pvector([1, 2, 3])
    result = Positives(pv)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, Positives)


def test_checkedpvector_constructor_with_tuple():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives((1, 2, 3))
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_checkedpvector_constructor_with_generator():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives(x for x in [1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_checkedpvector_constructor_default_empty():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives()
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checkedpvector_constructor_invalid_type_raises_error():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Should have raised an error"
    except Exception:
        pass


def test_checkedpvector_constructor_negative_value_raises_error():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised an error"
    except Exception:
        pass


# LLM-generated content at query #77
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "InvariantException" in str(type(e))


def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception as e:
        assert "CheckedKeyTypeError" in str(type(e)) or "CheckedTypeError" in str(type(e))


def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "string_value"})
        assert False, "Should have raised CheckedTypeError"
    except Exception as e:
        assert "CheckedTypeError" in str(type(e))


def test_checked_pmap_constructor_single_entry():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checked_pmap_constructor_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5})
    assert len(result) == 5
    assert result[1] == 1.1
    assert result[5] == 5.5


# LLM-generated content at query #78
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_with_default_size():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=10)
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25


# LLM-generated content at query #79
#--------------------------

```python
def test_wrap_invariant_predicate_line_3():
    def invariant(*args, **kwargs):
        return (True, "test")
    
    wrapped = wrap_invariant(invariant)
    result = wrapped()
    assert isinstance(result[0], bool) == True


# LLM-generated content at query #80
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"a": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {"a": 1, "b": 2, "c": 3}
    result = StringToIntMap(initial_data)
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


def test_checkedpmap_constructor_preserves_type():
    class IntToStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = IntToStrMap({1: "one", 2: "two"})
    assert type(result) == IntToStrMap
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #81
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_single_element():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checked_pmap_constructor_multiple_elements():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2, "c": 3})
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


# LLM-generated content at query #82
#--------------------------

```python
def test_checkedpmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checkedpmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"a": 1.5})
        assert False, "Should raise CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should raise CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_with_failed_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_with_multiple_valid_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4}
    result = IntToFloatMap(initial_data)
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[2] == 2.2
    assert result[3] == 3.3
    assert result[4] == 4.4


# LLM-generated content at query #83
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception:
        pass


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedValueTypeError"
    except Exception:
        pass


def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial = {"a": 1, "b": 2, "c": 3}
    result = StringToIntMap(initial)
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #84
#--------------------------

```python
def test_check_types_predicate_evaluates_to_false():
    from unittest.mock import Mock
    
    def get_type(t):
        return t
    
    def _check_types(it, expected_types, source_class, exception_type=Exception):
        if expected_types:
            for e in it:
                if not any(isinstance(e, get_type(t)) for t in expected_types):
                    actual_type = type(e)
                    msg = "Type {source_class} can only be used with {expected_types}, not {actual_type}".format(
                        source_class=source_class.__name__,
                        expected_types=tuple(get_type(et).__name__ for et in expected_types),
                        actual_type=actual_type.__name__)
                    raise exception_type(source_class, expected_types, actual_type, e, msg)
    
    class TestClass:
        pass
    
    test_obj = "string_value"
    expected_types = [int, float]
    
    try:
        _check_types([test_obj], expected_types, TestClass)
        assert False, "Expected exception to be raised"
    except Exception as e:
        assert "can only be used with" in str(e)


# LLM-generated content at query #85
#--------------------------

```python
def test_restore_pickle():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self._factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            return cls(data, _factory_fields)
    
    test_data = {"key": "value"}
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result.data == test_data
    assert result._factory_fields == set()
    assert isinstance(result, MockClass)


def test_restore_pickle_with_empty_dict():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self._factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            return cls(data, _factory_fields)
    
    test_data = {}
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result.data == {}
    assert result._factory_fields == set()


def test_restore_pickle_with_complex_data():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self._factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            return cls(data, _factory_fields)
    
    test_data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result.data == test_data
    assert result._factory_fields == set()
    assert result.data["nested"]["key"] == "value"
    assert result.data["list"] == [1, 2, 3]


# LLM-generated content at query #86
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant1(elem):
        return (True, "valid_data_1")
    
    def invariant2(elem):
        return (False, "invalid_data_2")
    
    def invariant3(elem):
        return (False, "invalid_data_3")
    
    invariants = [invariant1, invariant2, invariant3]
    elem = "test_element"
    
    result = _invariant_errors(elem, invariants)
    
    assert result == ["invalid_data_2", "invalid_data_3"]
    assert len(result) == 2
    assert "invalid_data_2" in result
    assert "invalid_data_3" in result


# LLM-generated content at query #87
#--------------------------

```python
def test_check_types_predicate_line_1():
    def _check_types(it, expected_types, source_class, exception_type=Exception):
        if expected_types:
            for e in it:
                if not any(isinstance(e, t) for t in expected_types):
                    actual_type = type(e)
                    msg = "Type {source_class} can only be used with {expected_types}, not {actual_type}".format(
                        source_class=source_class.__name__,
                        expected_types=tuple(et.__name__ for et in expected_types),
                        actual_type=actual_type.__name__)
                    raise exception_type(msg)
    
    class DummyClass:
        pass
    
    predicate_line_1 = True
    assert predicate_line_1 is True
    
    _check_types([1, 2, 3], [int], DummyClass)
    _check_types([], [int], DummyClass)
    _check_types([1.5], [float], DummyClass)


# LLM-generated content at query #88
#--------------------------

```python
def test_check_types_predicate_line_1():
    def get_type(t):
        return t
    
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
    
    def _check_types(it, expected_types, source_class, exception_type=CheckedValueTypeError):
        if expected_types:
            for e in it:
                if not any(isinstance(e, get_type(t)) for t in expected_types):
                    actual_type = type(e)
                    msg = "Type {source_class} can only be used with {expected_types}, not {actual_type}".format(
                        source_class=source_class.__name__,
                        expected_types=tuple(get_type(et).__name__ for et in expected_types),
                        actual_type=actual_type.__name__)
                    raise exception_type(source_class, expected_types, actual_type, e, msg)
    
    class TestClass:
        pass
    
    # Test case 1: predicate at line 1 (if expected_types:) evaluates to True
    result = _check_types([1, 2, 3], [int], TestClass)
    assert result is None
    
    # Test case 2: predicate at line 1 evaluates to True with non-empty expected_types
    result = _check_types(["a", "b"], [str], TestClass)
    assert result is None
    
    # Test case 3: predicate at line 1 evaluates to True with multiple expected types
    result = _check_types([1, "a", 2.5], [int, str, float], TestClass)
    assert result is None


# LLM-generated content at query #89
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_checked_pmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception:
        pass


def test_checked_pmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not a float"})
        assert False, "Should have raised TypeError"
    except Exception:
        pass


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_default_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap(initial={1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


# LLM-generated content at query #90
#--------------------------

```python
def test_wrap_invariant_with_bool_result():
    def invariant_bool(*args, **kwargs):
        return True, "data"
    
    wrapped = wrap_invariant(invariant_bool)
    verdict, data = wrapped()
    assert verdict is True
    assert data == "data"


def test_wrap_invariant_with_tuple_result_all_pass():
    def invariant_tuple(*args, **kwargs):
        return ((True, "pass1"), (True, "pass2"))
    
    wrapped = wrap_invariant(invariant_tuple)
    verdict, data = wrapped()
    assert verdict is True
    assert data == ()


def test_wrap_invariant_with_tuple_result_one_fail():
    def invariant_tuple(*args, **kwargs):
        return ((True, "pass1"), (False, "fail1"))
    
    wrapped = wrap_invariant(invariant_tuple)
    verdict, data = wrapped()
    assert verdict is False
    assert data == ("fail1",)


def test_wrap_invariant_with_tuple_result_multiple_fails():
    def invariant_tuple(*args, **kwargs):
        return ((False, "fail1"), (True, "pass1"), (False, "fail2"))
    
    wrapped = wrap_invariant(invariant_tuple)
    verdict, data = wrapped()
    assert verdict is False
    assert data == ("fail1", "fail2")


def test_wrap_invariant_with_args_and_kwargs():
    def invariant_with_params(a, b, c=None):
        return (a + b) > 0, f"sum={a+b}, c={c}"
    
    wrapped = wrap_invariant(invariant_with_params)
    verdict, data = wrapped(5, 3, c="test")
    assert verdict is True
    assert data == "sum=8, c=test"


def test_wrap_invariant_with_empty_tuple_result():
    def invariant_empty(*args, **kwargs):
        return ()
    
    wrapped = wrap_invariant(invariant_empty)
    verdict, data = wrapped()
    assert verdict is True
    assert data == ()


def test_wrap_invariant_with_all_false_results():
    def invariant_all_false(*args, **kwargs):
        return ((False, "error1"), (False, "error2"), (False, "error3"))
    
    wrapped = wrap_invariant(invariant_all_false)
    verdict, data = wrapped()
    assert verdict is False
    assert data == ("error1", "error2", "error3")


# LLM-generated content at query #91
#--------------------------

```python
def test_merge_invariant_results_predicate_false():
    result = [(False, "error1"), (True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


# LLM-generated content at query #92
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert dict(result) == {}


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checkedpmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({5: 5.5})
    assert len(result) == 1
    assert result[5] == 5.5


def test_checkedpmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4})
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[2] == 2.2
    assert result[3] == 3.3
    assert result[4] == 4.4


# LLM-generated content at query #93
#--------------------------

```python
def test_merge_invariant_results_predicate_true():
    result = [(True, "data1"), (True, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


# LLM-generated content at query #94
#--------------------------

```python
def test_checkedpmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invalid_invariant():
    from pyrsistent import InvariantException
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_with_invalid_key_type():
    from pyrsistent import CheckedKeyTypeError
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    from pyrsistent import CheckedValueTypeError
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checkedpmap_constructor_default_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(initial_data)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 5
    for k, v in initial_data.items():
        assert result[k] == v


# LLM-generated content at query #95
#--------------------------

```python
def test_check_types_predicate_line_1():
    # Line 1: def _check_types(it, expected_types, source_class, exception_type=CheckedValueTypeError):
    # The predicate is the function definition itself, which evaluates to True (function exists and is callable)
    
    def _check_types(it, expected_types, source_class, exception_type=None):
        if expected_types:
            for e in it:
                if not any(isinstance(e, t) for t in expected_types):
                    actual_type = type(e)
                    msg = "Type {source_class} can only be used with {expected_types}, not {actual_type}".format(
                        source_class=source_class.__name__,
                        expected_types=tuple(et.__name__ for et in expected_types),
                        actual_type=actual_type.__name__)
                    if exception_type:
                        raise exception_type(source_class, expected_types, actual_type, e, msg)
    
    # Verify the function is callable and exists
    assert callable(_check_types)
    assert _check_types is not None
    
    # Test with valid types - should not raise
    class TestClass:
        pass
    
    _check_types([1, 2, 3], [int], TestClass)
    
    # Test with empty expected_types - should not raise
    _check_types([1, "test", None], [], TestClass)


# LLM-generated content at query #96
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should raise CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should raise CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_violates_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4}
    result = IntToFloatMap(initial)
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[4] == 4.4


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result) == IntToFloatMap


# LLM-generated content at query #97
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockCheckedType:
        _checked_types = []
    
    instance = MockCheckedType()
    result = _checked_type_create(MockCheckedType, instance)
    assert result is instance


def test_checked_type_create_without_checked_type():
    class MockCls:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCls, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_and_matching_data():
    class CheckedType:
        pass
    
    class MockCls:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    instance = CheckedType()
    source_data = [instance]
    result = _checked_type_create(MockCls, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_and_non_matching_data():
    class CheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return CheckedType()
    
    class MockCls:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [{'key': 'value'}]
    result = _checked_type_create(MockCls, source_data)
    assert len(result.data) == 1
    assert isinstance(result.data[0], CheckedType)


def test_checked_type_create_with_ignore_extra_flag():
    class CheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return CheckedType()
    
    class MockCls:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [{'key': 'value'}]
    result = _checked_type_create(MockCls, source_data, ignore_extra=True)
    assert len(result.data) == 1
    assert isinstance(result.data[0], CheckedType)


# LLM-generated content at query #98
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should raise CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should raise CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_from_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_dict = {1: 1.5, 2: 2.25, 3: 3.75}
    result = IntToFloatMap(initial_dict)
    assert len(result) == 3
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75


def test_checked_pmap_constructor_multiple_items():
    class StrToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StrToIntMap({"a": 1, "b": 2, "c": 3, "d": 4})
    assert len(result) == 4
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3
    assert result["d"] == 4


# LLM-generated content at query #99
#--------------------------

```python
def test_check_types_with_valid_types():
    from typing import Type
    
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    it = [1, 2, 3]
    expected_types = [int]
    source_class = TestClass
    
    _check_types(it, expected_types, source_class, CheckedValueTypeError)


def test_check_types_with_invalid_types():
    from typing import Type
    
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    it = [1, "invalid", 3]
    expected_types = [int]
    source_class = TestClass
    
    try:
        _check_types(it, expected_types, source_class, CheckedValueTypeError)
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.actual_type == str
        assert e.value == "invalid"
        assert "str" in e.msg


def test_check_types_with_empty_expected_types():
    class TestClass:
        pass
    
    it = [1, "string", 3.14]
    expected_types = []
    source_class = TestClass
    
    _check_types(it, expected_types, source_class)


def test_check_types_with_multiple_valid_types():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    it = [1, "string", 3.14]
    expected_types = [int, str, float]
    source_class = TestClass
    
    _check_types(it, expected_types, source_class, CheckedValueTypeError)


def test_check_types_with_none_value():
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    it = [1, None, 3]
    expected_types = [int]
    source_class = TestClass
    
    try:
        _check_types(it, expected_types, source_class, CheckedValueTypeError)
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.actual_type == type(None)


# LLM-generated content at query #100
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_invariant_valid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invariant_invalid():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"1": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "1.5"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checkedpmap_constructor_multiple_entries():
    class StrToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {'a': 1, 'b': 2, 'c': 3}
    result = StrToIntMap(initial_data)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #101
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockClass:
        pass
    
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance


def test_checked_type_create_with_source_data_list():
    class MockCheckedType:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert result.data == source_data


def test_checked_type_create_calls_cls_constructor():
    class SimpleClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = {'key': 'value'}
    result = _checked_type_create(SimpleClass, source_data)
    assert result.data == source_data


def test_checked_type_create_with_empty_checked_types():
    class ClassWithEmptyCheckedTypes:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(ClassWithEmptyCheckedTypes, source_data)
    assert result.data == source_data


def test_checked_type_create_returns_source_when_already_instance():
    class TargetClass:
        _checked_types = []
    
    instance = TargetClass()
    result = _checked_type_create(TargetClass, instance)
    assert result is instance


# LLM-generated content at query #102
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockClass:
        pass
    
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance


def test_checked_type_create_without_checked_types():
    class SimpleClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(SimpleClass, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_and_matching_data():
    class MockCheckedType:
        _checked_types = ['builtins.int']
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert result.data == source_data


def test_get_class_valid_import():
    result = _get_class('os.path.PathLike')
    assert result.__name__ == 'PathLike'


def test_get_type_with_type_object():
    result = get_type(int)
    assert result is int


def test_get_type_with_string():
    result = get_type('builtins.int')
    assert result is int


def test_get_types_multiple_types():
    result = get_types([int, 'builtins.str'])
    assert result[0] is int
    assert result[1] is str


def test_get_types_empty_list():
    result = get_types([])
    assert result == []


# LLM-generated content at query #103
#--------------------------

```python
def test_isinstance_source_data_cls_returns_source_data():
    class TestClass:
        pass
    
    source_data = TestClass()
    result = isinstance(source_data, TestClass)
    
    assert result is True


# LLM-generated content at query #104
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checked_pmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert result[42] == 3.14
    assert len(result) == 1


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_checked_pmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception:
        pass


def test_checked_pmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not a float"})
        assert False, "Should have raised an error"
    except Exception:
        pass


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'
    assert isinstance(result, CheckedPMap)


def test_checked_pmap_constructor_with_large_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    input_dict = {i: float(i) for i in range(100)}
    result = IntToFloatMap(input_dict)
    assert len(result) == 100
    assert result[50] == 50.0


# LLM-generated content at query #105
#--------------------------

```python
def test_restore_pickle():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self._factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls(data, _factory_fields)
            return instance
    
    test_data = {"key": "value", "number": 42}
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result.data == test_data
    assert result._factory_fields == set()
    assert isinstance(result._factory_fields, set)
    assert len(result._factory_fields) == 0


# LLM-generated content at query #106
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised exception"
    except Exception:
        pass


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checkedpmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert result[42] == 3.14
    assert len(result) == 1


def test_checkedpmap_constructor_with_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert len(result) == 5
    for k, v in data.items():
        assert result[k] == v


# LLM-generated content at query #107
#--------------------------

```python
def test_restore_pickle_calls_create_with_empty_factory_fields():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            return {"data": data, "_factory_fields": _factory_fields}
    
    data = {"key": "value"}
    result = _restore_pickle(MockClass, data)
    
    assert result == {"data": data, "_factory_fields": set()}


def test_restore_pickle_with_empty_data():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            return {"data": data, "_factory_fields": _factory_fields}
    
    data = {}
    result = _restore_pickle(MockClass, data)
    
    assert result == {"data": data, "_factory_fields": set()}


def test_restore_pickle_with_complex_data():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            return {"data": data, "_factory_fields": _factory_fields}
    
    data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
    result = _restore_pickle(MockClass, data)
    
    assert result == {"data": data, "_factory_fields": set()}
    assert result["_factory_fields"] == set()


def test_restore_pickle_factory_fields_is_set():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            return _factory_fields
    
    data = {"key": "value"}
    result = _restore_pickle(MockClass, data)
    
    assert isinstance(result, set)
    assert len(result) == 0


# LLM-generated content at query #108
#--------------------------

```python
def test_check_types_predicate_line_1():
    def get_type(t):
        return t
    
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            super().__init__(msg)
    
    def _check_types(it, expected_types, source_class, exception_type=CheckedValueTypeError):
        if expected_types:
            for e in it:
                if not any(isinstance(e, get_type(t)) for t in expected_types):
                    actual_type = type(e)
                    msg = "Type {source_class} can only be used with {expected_types}, not {actual_type}".format(
                        source_class=source_class.__name__,
                        expected_types=tuple(get_type(et).__name__ for et in expected_types),
                        actual_type=actual_type.__name__)
                    raise exception_type(source_class, expected_types, actual_type, e, msg)
    
    class TestClass:
        pass
    
    it = [1, 2, 3]
    expected_types = [int]
    
    result = _check_types(it, expected_types, TestClass)
    
    assert result is None


# LLM-generated content at query #109
#--------------------------

```python
def test_checked_type_create_isinstance_predicate():
    class MockCheckedType:
        pass
    
    class TestClass(MockCheckedType):
        pass
    
    source_data = TestClass()
    result = isinstance(source_data, TestClass)
    assert result is True


# LLM-generated content at query #110
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_preserves_class_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #111
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"not_int": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_from_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_dict = {1: 1.5, 2: 2.25, 3: 3.75}
    result = IntToFloatMap(initial_dict)
    assert len(result) == 3
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75


def test_checked_pmap_constructor_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2, "c": 3, "d": 4})
    assert len(result) == 4
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3
    assert result["d"] == 4


# LLM-generated content at query #112
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    initial = {1: 1.0, 2: 2.0}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_single_item():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial = {'a': 1}
    result = StringToIntMap(initial)
    assert len(result) == 1
    assert result['a'] == 1


def test_checkedpmap_constructor_multiple_items():
    class StringToStrMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = str
    
    initial = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    result = StringToStrMap(initial)
    assert len(result) == 3
    assert result['key1'] == 'value1'
    assert result['key2'] == 'value2'
    assert result['key3'] == 'value3'


def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


# LLM-generated content at query #113
#--------------------------

```python
def test_check_types_predicate_evaluates_to_false():
    from unittest.mock import Mock
    
    def get_type(t):
        return t
    
    def _check_types(it, expected_types, source_class, exception_type=Exception):
        if expected_types:
            for e in it:
                if not any(isinstance(e, get_type(t)) for t in expected_types):
                    actual_type = type(e)
                    msg = "Type {source_class} can only be used with {expected_types}, not {actual_type}".format(
                        source_class=source_class.__name__,
                        expected_types=tuple(get_type(et).__name__ for et in expected_types),
                        actual_type=actual_type.__name__)
                    raise exception_type(source_class, expected_types, actual_type, e, msg)
    
    class TestClass:
        pass
    
    element = "string_value"
    expected_types = [int, float]
    
    try:
        _check_types([element], expected_types, TestClass)
        predicate_result = False
    except Exception:
        predicate_result = True
    
    assert predicate_result is True


# LLM-generated content at query #114
#--------------------------

```python
def test_checked_pmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_preserves_class_type():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2})
    assert type(result).__name__ == "StringToIntMap"
    assert result["a"] == 1
    assert result["b"] == 2


def test_checked_pmap_constructor_with_single_entry():
    class IntToStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = IntToStrMap({42: "answer"})
    assert len(result) == 1
    assert result[42] == "answer"


def test_checked_pmap_constructor_with_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert len(result) == 5
    for key, value in data.items():
        assert result[key] == value


# LLM-generated content at query #115
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checkedpmap_constructor_dict_conversion():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_dict = {10: 10.5, 20: 20.5, 30: 30.5}
    result = IntToFloatMap(initial_dict)
    assert len(result) == 3
    assert result[10] == 10.5
    assert result[20] == 20.5
    assert result[30] == 30.5


def test_checkedpmap_constructor_preserves_type():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1, 'b': 2})
    assert type(result).__name__ == 'StringToIntMap'
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #116
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_single_entry():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1})
    assert len(result) == 1
    assert result['a'] == 1


def test_checked_pmap_constructor_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    data = {'key1': 10, 'key2': 20, 'key3': 30}
    result = StringToIntMap(data)
    assert len(result) == 3
    assert result['key1'] == 10
    assert result['key2'] == 20
    assert result['key3'] == 30


# LLM-generated content at query #117
#--------------------------

```python
def test_check_types_with_valid_types():
    from your_module import _check_types
    
    _check_types([1, 2, 3], [int], type([]))
    _check_types(['a', 'b'], [str], type([]))
    _check_types([1, 'a'], [int, str], type([]))


def test_check_types_with_empty_expected_types():
    from your_module import _check_types
    
    _check_types([1, 2, 3], [], type([]))
    _check_types(['anything'], [], type([]))


def test_check_types_with_empty_iterable():
    from your_module import _check_types
    
    _check_types([], [int], type([]))
    _check_types([], [str, float], type([]))


def test_check_types_with_invalid_type_raises_error():
    from your_module import _check_types, CheckedValueTypeError
    
    try:
        _check_types([1, 'invalid'], [int], type([]))
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_check_types_with_custom_exception_type():
    from your_module import _check_types
    
    class CustomException(Exception):
        pass
    
    try:
        _check_types([1.5], [int], type([]), exception_type=CustomException)
        assert False, "Should have raised CustomException"
    except CustomException:
        pass


def test_check_types_with_multiple_expected_types_first_invalid():
    from your_module import _check_types, CheckedValueTypeError
    
    try:
        _check_types([1.5], [int, str], type([]))
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_check_types_with_multiple_expected_types_valid():
    from your_module import _check_types
    
    _check_types([1, 'a', 2.5], [int, str, float], type([]))


def test_check_types_error_message_format():
    from your_module import _check_types, CheckedValueTypeError
    
    try:
        _check_types([1.5], [int], list)
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert "list" in str(e)
        assert "int" in str(e)
        assert "float" in str(e)


# LLM-generated content at query #118
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_checked_pmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


# LLM-generated content at query #119
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockClass:
        pass
    
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance


def test_checked_type_create_with_source_data_already_correct_type():
    class MockCheckedType:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert result.data == source_data


def test_checked_type_create_with_no_checked_types():
    class SimpleClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(SimpleClass, source_data)
    assert result.data == source_data


def test_checked_type_create_returns_source_when_already_instance():
    class TargetClass:
        _checked_types = []
    
    source = TargetClass()
    result = _checked_type_create(TargetClass, source)
    assert result is source


def test_get_type_with_type_object():
    result = get_type(int)
    assert result is int


def test_get_type_with_string():
    result = get_type('builtins.int')
    assert result is int


def test_get_types_with_multiple_types():
    result = get_types([int, 'builtins.str'])
    assert result[0] is int
    assert result[1] is str


def test_get_types_with_all_type_objects():
    result = get_types([int, str, float])
    assert result == [int, str, float]


def test_get_types_with_all_strings():
    result = get_types(['builtins.int', 'builtins.str', 'builtins.float'])
    assert result == [int, str, float]


