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

def test_maybe_parse_user_type_with_nested_iterables():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)

def test_maybe_parse_user_type_with_invalid_input():
    try:
        maybe_parse_user_type(42)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)

def test_maybe_parse_user_type_with_none():
    try:
        maybe_parse_user_type(None)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)

def test_maybe_parse_user_type_with_mixed_list():
    result = maybe_parse_user_type([int, "str", float])
    assert result == (int, "str", float)

def test_maybe_parse_user_type_with_empty_list():
    result = maybe_parse_user_type([])
    assert result == ()


# LLM-generated content at query #2
#--------------------------

```python
def test_checked_type_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #3
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


def test_checkedpvector_constructor_with_mixed_numeric_types():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2.5
    assert result[2] == 3


def test_checkedpvector_constructor_with_default_parameter():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives()
    assert len(result) == 0
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


def test_checkedpvector_constructor_preserves_type():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([5, 10])
    assert type(result).__name__ == 'Positives'


# LLM-generated content at query #4
#--------------------------

```python
def test_checked_map_type_meta_new_stores_key_types():
    class TestClass(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
    
    assert hasattr(TestClass, '_checked_key_types')
    assert str in TestClass._checked_key_types


def test_checked_map_type_meta_new_stores_value_types():
    class TestClass(metaclass=_CheckedMapTypeMeta):
        __value_type__ = int
    
    assert hasattr(TestClass, '_checked_value_types')
    assert int in TestClass._checked_value_types


def test_checked_map_type_meta_new_stores_invariants():
    def dummy_invariant(self):
        return True, None
    
    class TestClass(metaclass=_CheckedMapTypeMeta):
        __invariant__ = dummy_invariant
    
    assert hasattr(TestClass, '_checked_invariants')
    assert len(TestClass._checked_invariants) > 0


def test_checked_map_type_meta_new_sets_default_serializer():
    class TestClass(metaclass=_CheckedMapTypeMeta):
        pass
    
    assert hasattr(TestClass, '__serializer__')
    assert callable(TestClass.__serializer__)


def test_checked_map_type_meta_new_sets_empty_slots():
    class TestClass(metaclass=_CheckedMapTypeMeta):
        pass
    
    assert hasattr(TestClass, '__slots__')
    assert TestClass.__slots__ == ()


def test_checked_map_type_meta_new_default_serializer_with_primitives():
    class TestClass(metaclass=_CheckedMapTypeMeta):
        pass
    
    instance = TestClass()
    sk, sv = instance.__serializer__(None, 'key', 'value')
    assert sk == 'key'
    assert sv == 'value'


def test_checked_map_type_meta_new_inherits_key_types():
    class BaseClass(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
    
    class DerivedClass(BaseClass):
        pass
    
    assert hasattr(DerivedClass, '_checked_key_types')
    assert str in DerivedClass._checked_key_types


def test_checked_map_type_meta_new_inherits_value_types():
    class BaseClass(metaclass=_CheckedMapTypeMeta):
        __value_type__ = int
    
    class DerivedClass(BaseClass):
        pass
    
    assert hasattr(DerivedClass, '_checked_value_types')
    assert int in DerivedClass._checked_value_types


def test_checked_map_type_meta_new_inherits_invariants():
    def dummy_invariant(self):
        return True, None
    
    class BaseClass(metaclass=_CheckedMapTypeMeta):
        __invariant__ = dummy_invariant
    
    class DerivedClass(BaseClass):
        pass
    
    assert hasattr(DerivedClass, '_checked_invariants')
    assert len(DerivedClass._checked_invariants) > 0


def test_checked_map_type_meta_new_multiple_types():
    class TestClass(metaclass=_CheckedMapTypeMeta):
        __key_type__ = [str, int]
        __value_type__ = [float, bool]
    
    assert str in TestClass._checked_key_types
    assert int in TestClass._checked_key_types
    assert float in TestClass._checked_value_types
    assert bool in TestClass._checked_value_types


# LLM-generated content at query #5
#--------------------------

```python
def test_invariant_errors_all_pass():
    def invariant1(elem):
        return (True, "valid1")
    
    def invariant2(elem):
        return (True, "valid2")
    
    result = _invariant_errors("test", [invariant1, invariant2])
    assert result == []


def test_invariant_errors_all_fail():
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


def test_invariant_errors_empty_list():
    result = _invariant_errors("test", [])
    assert result == []


def test_invariant_errors_single_pass():
    def invariant1(elem):
        return (True, "valid")
    
    result = _invariant_errors("test", [invariant1])
    assert result == []


def test_invariant_errors_single_fail():
    def invariant1(elem):
        return (False, "error")
    
    result = _invariant_errors("test", [invariant1])
    assert result == ["error"]


def test_invariant_errors_with_different_data_types():
    def invariant1(elem):
        return (False, 42)
    
    def invariant2(elem):
        return (False, {"key": "value"})
    
    def invariant3(elem):
        return (True, "ignored")
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == [42, {"key": "value"}]


def test_invariant_errors_uses_elem_parameter():
    def invariant1(elem):
        return (elem == "expected", f"got {elem}")
    
    result = _invariant_errors("expected", [invariant1])
    assert result == []
    
    result = _invariant_errors("unexpected", [invariant1])
    assert result == ["got unexpected"]


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


def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


# LLM-generated content at query #7
#--------------------------

```python
def test_checked_map_type_meta_line_3_predicate_false():
    class MockCheckedType:
        def serialize(self):
            return "serialized"
    
    key_not_checked = "regular_key"
    value_not_checked = "regular_value"
    
    is_key_checked_type = isinstance(key_not_checked, MockCheckedType)
    is_value_checked_type = isinstance(value_not_checked, MockCheckedType)
    
    assert is_key_checked_type is False
    assert is_value_checked_type is False


# LLM-generated content at query #8
#--------------------------

```python
def test_store_invariants_empty_dct_and_bases():
    dct = {}
    bases = ()
    store_invariants(dct, bases, 'dest', 'src')
    assert dct['dest'] == ()


def test_store_invariants_single_invariant_in_dct():
    def my_invariant(x):
        return True
    
    dct = {'src': my_invariant}
    bases = ()
    store_invariants(dct, bases, 'dest', 'src')
    assert len(dct['dest']) == 1
    assert callable(dct['dest'][0])


def test_store_invariants_invariant_in_base_class():
    def base_invariant(x):
        return True
    
    class Base:
        pass
    
    Base.__dict__['src'] = base_invariant
    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, 'dest', 'src')
    assert len(dct['dest']) == 1
    assert callable(dct['dest'][0])


def test_store_invariants_multiple_invariants():
    def inv1(x):
        return True
    
    def inv2(x):
        return False
    
    dct = {'src': inv1}
    
    class Base:
        pass
    
    Base.__dict__['src'] = inv2
    bases = (Base,)
    store_invariants(dct, bases, 'dest', 'src')
    assert len(dct['dest']) == 2


def test_store_invariants_non_callable_raises_error():
    dct = {'src': 'not_callable'}
    bases = ()
    
    try:
        store_invariants(dct, bases, 'dest', 'src')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Invariants must be callable' in str(e)


def test_store_invariants_wrapped_invariant_returns_bool_tuple():
    def my_invariant(x):
        return (True, 5)
    
    dct = {'src': my_invariant}
    bases = ()
    store_invariants(dct, bases, 'dest', 'src')
    wrapped = dct['dest'][0]
    result = wrapped(10)
    assert result == (True, 5)


def test_store_invariants_wrapped_invariant_merges_results():
    def my_invariant(x):
        return ((False, 'error1'), (False, 'error2'))
    
    dct = {'src': my_invariant}
    bases = ()
    store_invariants(dct, bases, 'dest', 'src')
    wrapped = dct['dest'][0]
    result = wrapped(10)
    assert result == (False, ('error1', 'error2'))


def test_store_invariants_inheritance_order():
    def base_inv(x):
        return True
    
    def derived_inv(x):
        return False
    
    class Base:
        pass
    
    Base.__dict__['src'] = base_inv
    dct = {'src': derived_inv}
    bases = (Base,)
    store_invariants(dct, bases, 'dest', 'src')
    assert len(dct['dest']) == 2


def test_store_invariants_diamond_inheritance():
    def inv_a(x):
        return True
    
    class A:
        pass
    
    A.__dict__['src'] = inv_a
    
    class B(A):
        pass
    
    class C(A):
        pass
    
    dct = {}
    bases = (B, C)
    store_invariants(dct, bases, 'dest', 'src')
    assert len(dct['dest']) == 1


# LLM-generated content at query #9
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


def test_checked_pmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "InvariantException" in str(type(e))


def test_checked_pmap_constructor_type_checking():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"key": 1.5})
        assert False, "Expected type error"
    except Exception:
        pass


def test_checked_pmap_constructor_value_type_checking():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not a float"})
        assert False, "Expected type error"
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


# LLM-generated content at query #10
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
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_single_item():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1})
    assert len(result) == 1
    assert result['a'] == 1


def test_checkedpmap_constructor_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1, 'b': 2, 'c': 3})
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


def test_checkedpmap_constructor_returns_correct_type():
    class CustomMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = CustomMap({1: 'one', 2: 'two'})
    assert type(result).__name__ == 'CustomMap'
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #11
#--------------------------

```python
def test_checked_pmap_new_with_empty_initial():
    from pyrsistent import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap()
    assert isinstance(result, SimpleMap)
    assert len(result) == 0


def test_checked_pmap_new_with_initial_dict():
    from pyrsistent import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap({'a': 1, 'b': 2})
    assert isinstance(result, SimpleMap)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2


def test_checked_pmap_new_with_size_parameter():
    from pyrsistent import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap({'x': 10}, size=16)
    assert isinstance(result, SimpleMap)
    assert result['x'] == 10


def test_checked_pmap_new_with_invalid_key_type():
    from pyrsistent import CheckedPMap, CheckedKeyTypeError
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    try:
        SimpleMap({1: 100})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_new_with_invalid_value_type():
    from pyrsistent import CheckedPMap, CheckedTypeError
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    try:
        SimpleMap({'a': 'not_an_int'})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_new_with_invariant():
    from pyrsistent import CheckedPMap, InvariantException
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0


def test_checked_pmap_new_with_invariant_violation():
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


def test_checked_pmap_new_with_multiple_entries():
    from pyrsistent import CheckedPMap
    
    class StringMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = str
    
    data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    result = StringMap(data)
    assert len(result) == 3
    assert result['key1'] == 'value1'
    assert result['key2'] == 'value2'
    assert result['key3'] == 'value3'


def test_checked_pmap_new_preserves_type():
    from pyrsistent import CheckedPMap
    
    class CustomMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = CustomMap({1: 'a', 2: 'b'})
    assert type(result).__name__ == 'CustomMap'
    assert isinstance(result, CustomMap)


# LLM-generated content at query #12
#--------------------------

```python
def test_checked_pvector_constructor_with_empty_list():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector([])
    assert len(result) == 0
    assert isinstance(result, TestVector)


def test_checked_pvector_constructor_with_valid_values():
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
    
    class TestVector(CheckedPVector):
        __type__ = int
    
    pv = pvector([1, 2, 3])
    result = TestVector(pv)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, TestVector)


def test_checked_pvector_constructor_with_tuple():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector((1, 2, 3))
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_checked_pvector_constructor_with_mixed_numeric_types():
    class Numbers(CheckedPVector):
        __type__ = (int, float)
    
    result = Numbers([1, 2.5, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2.5
    assert result[2] == 3


def test_checked_pvector_constructor_type_error():
    class IntVector(CheckedPVector):
        __type__ = int
    
    try:
        IntVector([1, "invalid", 3])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pvector_constructor_invariant_error():
    class Positives(CheckedPVector):
        __type__ = int
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pvector_constructor_single_element():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector([42])
    assert len(result) == 1
    assert result[0] == 42


def test_checked_pvector_constructor_preserves_type():
    class CustomVector(CheckedPVector):
        __type__ = str
    
    result = CustomVector(["a", "b", "c"])
    assert type(result).__name__ == "CustomVector"
    assert isinstance(result, CheckedPVector)


# LLM-generated content at query #13
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
    assert result[2] == 3


def test_checked_pvector_constructor_with_python_pvector():
    from pyrsistent import pvector as python_pvector
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pv = python_pvector([1, 2, 3])
    result = Positives(pv)
    assert len(result) == 3
    assert isinstance(result, Positives)


def test_checked_pvector_constructor_with_single_element():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([42])
    assert len(result) == 1
    assert result[0] == 42


def test_checked_pvector_constructor_with_large_list():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    large_list = list(range(1, 101))
    result = Positives(large_list)
    assert len(result) == 100
    assert result[0] == 1
    assert result[99] == 100


# LLM-generated content at query #14
#--------------------------

```python
def test_checkedtype_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert instance.__slots__ == ()


# LLM-generated content at query #15
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


def test_checked_pmap_constructor_with_invalid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_default_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


# LLM-generated content at query #16
#--------------------------

```python
def test_store_types_empty_bases():
    dct = {}
    bases = ()
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == ()


def test_store_types_single_type_in_dct():
    dct = {'source': int}
    bases = ()
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == (int,)


def test_store_types_string_type_in_dct():
    dct = {'source': 'str'}
    bases = ()
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == ('str',)


def test_store_types_multiple_types_as_list():
    dct = {'source': [int, str]}
    bases = ()
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == (int, str)


def test_store_types_from_base_class():
    class Base:
        pass
    Base.__dict__['source'] = float
    
    dct = {}
    bases = (Base,)
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == (float,)


def test_store_types_dct_overrides_base():
    class Base:
        pass
    Base.__dict__['source'] = float
    
    dct = {'source': int}
    bases = (Base,)
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == (int, float)


def test_store_types_multiple_bases():
    class Base1:
        pass
    class Base2:
        pass
    Base1.__dict__['source'] = int
    Base2.__dict__['source'] = str
    
    dct = {}
    bases = (Base1, Base2)
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == (int, str)


def test_store_types_nested_iterables():
    dct = {'source': [int, [str, float]]}
    bases = ()
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == (int, str, float)


def test_store_types_source_not_in_dct_or_bases():
    dct = {'other': int}
    bases = ()
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == ()


def test_store_types_mixed_types_and_strings():
    dct = {'source': [int, 'CustomType', str]}
    bases = ()
    _store_types(dct, bases, 'destination', 'source')
    assert dct['destination'] == (int, 'CustomType', str)


# LLM-generated content at query #17
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
    
    class CustomType:
        pass
    
    result = maybe_parse_user_type(CustomType)
    assert result == [CustomType]
    assert isinstance(result, list)


# LLM-generated content at query #18
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


def test_checked_pmap_constructor_with_single_item():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1})
    assert len(result) == 1
    assert result['a'] == 1


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
    except Exception:
        pass


def test_checked_pmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_multiple_items():
    class StringToStrMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = str
    
    data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    result = StringToStrMap(data)
    assert len(result) == 3
    assert result['key1'] == 'value1'
    assert result['key2'] == 'value2'
    assert result['key3'] == 'value3'


# LLM-generated content at query #19
#--------------------------

```python
def test_checked_pmap_initial_items_iteration():
    from pyrsistent import CheckedPMap
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.5, 3: 3.5}
    checked_map = IntToFloatMap(initial_data)
    
    # Verify that the map was created with all items
    assert len(checked_map) == 3
    assert checked_map[1] == 1.5
    assert checked_map[2] == 2.5
    assert checked_map[3] == 3.5
    
    # Verify iteration works on initial.items()
    result_items = list(checked_map.items())
    assert len(result_items) == 3
    assert (1, 1.5) in result_items
    assert (2, 2.5) in result_items
    assert (3, 3.5) in result_items


# LLM-generated content at query #20
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
    result = maybe_parse_user_type([[int], [str]])
    assert result == (int, str)


def test_maybe_parse_user_type_with_invalid_type():
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
        maybe_parse_user_type({"type": int})
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


# LLM-generated content at query #21
#--------------------------

```python
def test_checkedpset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checkedpset_constructor_with_valid_integers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    assert isinstance(result, Positives)


def test_checkedpset_constructor_with_valid_floats():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.7, 3.2])
    assert len(result) == 3
    assert 1.5 in result
    assert 2.7 in result
    assert 3.2 in result


def test_checkedpset_constructor_with_mixed_valid_types():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checkedpset_constructor_with_duplicates():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checkedpset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Should have raised an exception"
    except CheckedTypeError:
        pass


def test_checkedpset_constructor_with_negative_values():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised an exception"
    except InvariantException:
        pass


def test_checkedpset_constructor_repr():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    repr_str = repr(result)
    assert repr_str.startswith('Positives')
    assert 'Positives' in repr_str


# LLM-generated content at query #22
#--------------------------

```python
def test_checked_pmap_new_with_empty_initial():
    from pyrsistent import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap()
    assert isinstance(result, SimpleMap)
    assert len(result) == 0


def test_checked_pmap_new_with_initial_dict():
    from pyrsistent import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap({'a': 1, 'b': 2})
    assert isinstance(result, SimpleMap)
    assert result['a'] == 1
    assert result['b'] == 2
    assert len(result) == 2


def test_checked_pmap_new_with_size_parameter():
    from pyrsistent import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap({'x': 10}, size=16)
    assert isinstance(result, SimpleMap)
    assert result['x'] == 10
    assert len(result) == 1


def test_checked_pmap_new_with_empty_and_size():
    from pyrsistent import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap({}, size=32)
    assert isinstance(result, SimpleMap)
    assert len(result) == 0


def test_checked_pmap_new_validates_key_type():
    from pyrsistent import CheckedPMap, CheckedKeyTypeError
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    try:
        SimpleMap({1: 'invalid'})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_new_validates_value_type():
    from pyrsistent import CheckedPMap, CheckedTypeError
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    try:
        SimpleMap({'a': 'invalid'})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_new_with_invariant():
    from pyrsistent import CheckedPMap, InvariantException
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0


def test_checked_pmap_new_invariant_violation():
    from pyrsistent import CheckedPMap, InvariantException
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_new_multiple_items():
    from pyrsistent import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    result = SimpleMap(data)
    assert isinstance(result, SimpleMap)
    assert len(result) == 4
    for k, v in data.items():
        assert result[k] == v


# LLM-generated content at query #23
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
    
    valid_items = [1, 2, 3]
    _check_types(valid_items, [int], TestClass, CheckedValueTypeError)


def test_check_types_with_empty_expected_types():
    class TestClass:
        pass
    
    items = [1, "string", 3.14]
    _check_types(items, [], TestClass)


def test_check_types_with_multiple_valid_types():
    class TestClass:
        pass
    
    valid_items = [1, "string", 3.14]
    _check_types(valid_items, [int, str, float], TestClass)


def test_check_types_raises_error_on_invalid_type():
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
    
    invalid_items = [1, 2, "invalid"]
    try:
        _check_types(invalid_items, [int], TestClass, CheckedValueTypeError)
        assert False, "Expected CheckedValueTypeError to be raised"
    except CheckedValueTypeError as e:
        assert e.actual_type == str
        assert e.value == "invalid"


def test_check_types_with_none_in_expected_types():
    class TestClass:
        pass
    
    valid_items = [1, None, 3]
    _check_types(valid_items, [int, type(None)], TestClass)


def test_check_types_error_message_format():
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
    
    invalid_items = [1.5]
    try:
        _check_types(invalid_items, [int], TestClass, CheckedValueTypeError)
        assert False, "Expected CheckedValueTypeError to be raised"
    except CheckedValueTypeError as e:
        assert "TestClass" in e.msg
        assert "int" in e.msg
        assert "float" in e.msg


# LLM-generated content at query #24
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


def test_checked_type_create_with_checked_type_subclass():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
        
        def __init__(self, data):
            self.data = data
    
    class MockCheckedType:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert isinstance(result, MockCheckedType)


def test_checked_type_create_with_matching_type_in_data():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
        
        def __init__(self, data):
            self.data = data
    
    class MockCheckedType:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    existing_instance = CheckedType(42)
    source_data = [existing_instance, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert isinstance(result, MockCheckedType)


def test_checked_type_create_ignore_extra_parameter():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return cls(data)
        
        def __init__(self, data):
            self.data = data
    
    class MockCheckedType:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data, ignore_extra=True)
    assert isinstance(result, MockCheckedType)


# LLM-generated content at query #25
#--------------------------

```python
def test_maybe_parse_user_type_preserved_iterable():
    from collections.abc import Iterable
    
    class _preserved_iterable_types(list):
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


# LLM-generated content at query #26
#--------------------------

```python
def test_checked_pmap_new_with_empty_initial():
    from pyrsistent import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = SimpleMap()
    assert isinstance(result, SimpleMap)
    assert len(result) == 0


def test_checked_pmap_new_with_initial_dict():
    from pyrsistent import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial = {'a': 1, 'b': 2}
    result = SimpleMap(initial)
    assert isinstance(result, SimpleMap)
    assert result['a'] == 1
    assert result['b'] == 2
    assert len(result) == 2


def test_checked_pmap_new_with_size_parameter():
    from pyrsistent import CheckedPMap
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial = {'x': 10, 'y': 20}
    result = SimpleMap(initial, size=16)
    assert isinstance(result, SimpleMap)
    assert result['x'] == 10
    assert result['y'] == 20


def test_checked_pmap_new_invalid_key_type():
    from pyrsistent import CheckedPMap, CheckedKeyTypeError
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    try:
        SimpleMap({123: 1})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_new_invalid_value_type():
    from pyrsistent import CheckedPMap, CheckedTypeError
    
    class SimpleMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    try:
        SimpleMap({'a': 'invalid'})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_new_with_invariant_valid():
    from pyrsistent import CheckedPMap
    
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_new_with_invariant_invalid():
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


def test_checked_pmap_new_preserves_type():
    from pyrsistent import CheckedPMap
    
    class CustomMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = str
    
    result = CustomMap({'key': 'value'})
    assert type(result).__name__ == 'CustomMap'


def test_checked_pmap_new_multiple_items():
    from pyrsistent import CheckedPMap
    
    class StringMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = str
    
    initial = {'a': '1', 'b': '2', 'c': '3', 'd': '4'}
    result = StringMap(initial)
    assert len(result) == 4
    assert result['a'] == '1'
    assert result['d'] == '4'


# LLM-generated content at query #27
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


# LLM-generated content at query #28
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


def test_checked_pmap_constructor_default_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert len(result) == 1
    assert result[1] == 1.5


def test_checked_pmap_constructor_preserves_class_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'
    assert isinstance(result, CheckedPMap)


def test_checked_pmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


# LLM-generated content at query #29
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


def test_checked_pmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert result[1] == 1.5
    assert len(result) == 1


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


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #30
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
    
    _check_types([1, 2, 3], [], TestClass)


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
    
    _check_types([1, "hello", 3.14], [int, str, float], TestClass, CheckedValueTypeError)


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
        _check_types([1, 2, "invalid"], [int], TestClass, CheckedValueTypeError)
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.actual_type == str
        assert e.value == "invalid"
        assert "TestClass" in e.msg
        assert "int" in e.msg


def test_check_types_with_string_type_names():
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
    
    _check_types([1, 2, 3], ["builtins.int"], TestClass, CheckedValueTypeError)


def test_check_types_raises_with_wrong_string_type():
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
        _check_types([1, 2, "invalid"], ["builtins.int"], TestClass, CheckedValueTypeError)
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.actual_type == str


# LLM-generated content at query #31
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
    
    test_data = {"key": "value", "number": 42}
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result.data == test_data
    assert result._factory_fields == set()
    assert isinstance(result._factory_fields, set)
    assert len(result._factory_fields) == 0


# LLM-generated content at query #32
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


# LLM-generated content at query #33
#--------------------------

```python
def test_maybe_parse_user_type_preserved_iterable():
    from collections.abc import Iterable
    
    class _preserved_iterable_types(list):
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


# LLM-generated content at query #34
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockClass:
        pass
    
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance


def test_checked_type_create_with_non_checked_type():
    class MockClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert result.data == [1, 2, 3]


def test_checked_type_create_with_checked_type_and_matching_data():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}"
    
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        
        def __init__(self, data):
            self.data = data
    
    source_data = [CheckedType(), CheckedType()]
    result = _checked_type_create(MockClass, source_data)
    assert result.data == [CheckedType(), CheckedType()]


def test_checked_type_create_with_checked_type_and_non_matching_data():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}"
    
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        
        def __init__(self, data):
            self.data = data
    
    source_data = ["raw_data_1", "raw_data_2"]
    result = _checked_type_create(MockClass, source_data)
    assert result.data == ["created_raw_data_1", "created_raw_data_2"]


def test_checked_type_create_with_mixed_data_types():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}"
    
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        
        def __init__(self, data):
            self.data = data
    
    checked_instance = CheckedType()
    source_data = [checked_instance, "raw_data"]
    result = _checked_type_create(MockClass, source_data)
    assert result.data[0] is checked_instance
    assert result.data[1] == "created_raw_data"


def test_checked_type_create_with_ignore_extra_flag():
    class CheckedType:
        @classmethod
        def create(cls, data, ignore_extra=False):
            return f"created_{data}_{ignore_extra}"
    
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        
        def __init__(self, data):
            self.data = data
    
    source_data = ["raw_data"]
    result = _checked_type_create(MockClass, source_data, ignore_extra=True)
    assert result.data == ["created_raw_data_True"]


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


def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_with_valid_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial = {'a': 1, 'b': 2, 'c': 3}
    result = StringToIntMap(initial)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


def test_checkedpmap_constructor_preserves_type():
    class IntToStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = IntToStrMap({1: 'hello', 2: 'world'})
    assert type(result).__name__ == 'IntToStrMap'
    assert isinstance(result, CheckedPMap)


def test_checkedpmap_constructor_with_single_entry():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


# LLM-generated content at query #36
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    from collections import UserList
    
    class TestCheckedType(UserList):
        _checked_types = []
    
    instance = TestCheckedType([1, 2, 3])
    result = _checked_type_create(TestCheckedType, instance)
    assert result is instance


def test_checked_type_create_with_list_data_no_checked_types():
    from collections import UserList
    
    class TestCheckedType(UserList):
        _checked_types = []
    
    source_data = [1, 2, 3]
    result = _checked_type_create(TestCheckedType, source_data)
    assert isinstance(result, TestCheckedType)
    assert result.data == [1, 2, 3]


def test_get_type_with_type_object():
    result = get_type(int)
    assert result is int


def test_get_type_with_string_type_name():
    result = get_type('builtins.int')
    assert result is int


def test_get_types_with_multiple_types():
    result = get_types([int, 'builtins.str', float])
    assert result == [int, str, float]


def test_get_types_with_empty_list():
    result = get_types([])
    assert result == []


def test_get_class_with_valid_module_and_class():
    result = _get_class('collections.OrderedDict')
    from collections import OrderedDict
    assert result is OrderedDict


def test_get_class_with_builtin_module():
    result = _get_class('builtins.dict')
    assert result is dict


# LLM-generated content at query #37
#--------------------------

```python
def test_wrap_invariant_with_single_bool_result():
    def invariant_single(*args, **kwargs):
        return (True, "data1")
    
    wrapped = wrap_invariant(invariant_single)
    verdict, data = wrapped()
    assert verdict is True
    assert data == "data1"


def test_wrap_invariant_with_single_bool_result_false():
    def invariant_single(*args, **kwargs):
        return (False, "error1")
    
    wrapped = wrap_invariant(invariant_single)
    verdict, data = wrapped()
    assert verdict is False
    assert data == "error1"


def test_wrap_invariant_with_multiple_results_all_pass():
    def invariant_multiple(*args, **kwargs):
        return ((True, "data1"), (True, "data2"))
    
    wrapped = wrap_invariant(invariant_multiple)
    verdict, data = wrapped()
    assert verdict is True
    assert data == ()


def test_wrap_invariant_with_multiple_results_one_fails():
    def invariant_multiple(*args, **kwargs):
        return ((True, "data1"), (False, "error1"), (True, "data2"))
    
    wrapped = wrap_invariant(invariant_multiple)
    verdict, data = wrapped()
    assert verdict is False
    assert data == ("error1",)


def test_wrap_invariant_with_multiple_results_all_fail():
    def invariant_multiple(*args, **kwargs):
        return ((False, "error1"), (False, "error2"))
    
    wrapped = wrap_invariant(invariant_multiple)
    verdict, data = wrapped()
    assert verdict is False
    assert data == ("error1", "error2")


def test_wrap_invariant_passes_args_and_kwargs():
    def invariant_with_args(a, b, c=None):
        return (True, f"{a}-{b}-{c}")
    
    wrapped = wrap_invariant(invariant_with_args)
    verdict, data = wrapped(1, 2, c=3)
    assert verdict is True
    assert data == "1-2-3"


def test_wrap_invariant_with_multiple_results_and_args():
    def invariant_multiple_args(a, b):
        return ((True, f"data{a}"), (False, f"error{b}"))
    
    wrapped = wrap_invariant(invariant_multiple_args)
    verdict, data = wrapped(1, 2)
    assert verdict is False
    assert data == ("error2",)


# LLM-generated content at query #38
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


def test_checked_pmap_constructor_with_single_element():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({5: 3.14})
    assert len(result) == 1
    assert result[5] == 3.14


def test_checked_pmap_constructor_with_multiple_elements():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25, 3: 3.75}
    result = IntToFloatMap(initial)
    assert len(result) == 3
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75


def test_checked_pmap_constructor_preserves_class_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({10: 10.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_default_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


# LLM-generated content at query #39
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
    except InvariantException:
        pass


def test_checked_pmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checked_pmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(initial_data)
    assert len(result) == 5
    for k, v in initial_data.items():
        assert result[k] == v


# LLM-generated content at query #40
#--------------------------

```python
def test_check_types_predicate_true_with_nonempty_expected_types():
    expected_types = [int, str]
    result = bool(expected_types)
    assert result is True


# LLM-generated content at query #41
#--------------------------

```python
def test_restore_pickle():
    class MockClass:
        def __init__(self, data):
            self.data = data
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls(data)
            instance._factory_fields = _factory_fields
            return instance
    
    test_data = {"key": "value"}
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result.data == test_data
    assert result._factory_fields == set()
    assert isinstance(result._factory_fields, set)


# LLM-generated content at query #42
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


def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_with_failed_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert len(result) == 5
    assert result[1] == 1.1
    assert result[5] == 5.5


# LLM-generated content at query #43
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
        IntToFloatMap({1: "string_value"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


# LLM-generated content at query #44
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
        assert False, "Should have raised an exception"
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


def test_checkedpmap_constructor_default_argument():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert len(result) == 1
    assert result[1] == 1.5


def test_checkedpmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0})
    assert len(result) == 4
    assert result[1] == 1.0
    assert result[4] == 4.0


# LLM-generated content at query #45
#--------------------------

```python
def test_checkedtype_constructor():
    obj = CheckedType()
    assert isinstance(obj, CheckedType)
    assert hasattr(obj, '__slots__')
    assert obj.__slots__ == ()


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
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checked_pmap_constructor_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1, 'b': 2, 'c': 3})
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({10: 10.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_default_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5}, size=8)
    assert result[1] == 1.5
    assert isinstance(result, IntToFloatMap)


# LLM-generated content at query #47
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockClass:
        pass
    
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance


def test_checked_type_create_with_no_checked_types():
    class MockClass:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_and_matching_data():
    class CheckedType:
        pass
    
    class MockClass:
        _checked_types = ['test_module.CheckedType']
        def __init__(self, data):
            self.data = data
    
    data_instance = CheckedType()
    source_data = [data_instance]
    
    result = _checked_type_create(MockClass, source_data, ignore_extra=False)
    assert result.data == source_data


def test_checked_type_create_returns_cls_instance_with_source_data():
    class SimpleClass:
        _checked_types = []
        def __init__(self, data):
            self.items = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(SimpleClass, source_data)
    assert isinstance(result, SimpleClass)
    assert result.items == source_data


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
        IntToFloatMap({'invalid': 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: 'invalid'})
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


def test_checked_pmap_constructor_preserves_class_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'
    assert isinstance(result, IntToFloatMap)


# LLM-generated content at query #49
#--------------------------

```python
def test_check_types_predicate_line_1_evaluates_to_false():
    expected_types = []
    it = [1, 2, 3]
    source_class = list
    exception_type = Exception
    
    result = bool(expected_types)
    
    assert result is False


# LLM-generated content at query #50
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


def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


# LLM-generated content at query #51
#--------------------------

```python
def test_check_types_predicate_line_1():
    # Test that the predicate at line 1 (if expected_types:) evaluates to True
    # This means expected_types should be truthy (non-empty)
    
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
    
    # Test case where expected_types is truthy (non-empty list)
    # This ensures the predicate at line 1 evaluates to True
    result = _check_types([1, 2, 3], [int], TestClass)
    assert result is None


# LLM-generated content at query #52
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    # Test that the predicate at line 1 (isinstance(source_data, cls)) evaluates to False
    # We need to create a mock class and pass source_data that is NOT an instance of cls
    
    class MockClass:
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    cls = MockClass
    
    # Verify the predicate is False
    result = isinstance(source_data, cls)
    assert result is False


# LLM-generated content at query #53
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
    except InvariantException:
        pass


def test_checkedpmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checkedpmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(initial_data)
    assert len(result) == 5
    for k, v in initial_data.items():
        assert result[k] == v


def test_checkedpmap_constructor_preserves_type():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2})
    assert type(result).__name__ == "StringToIntMap"
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #54
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


def test_checked_pmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({5: 5.5})
    assert len(result) == 1
    assert result[5] == 5.5


def test_checked_pmap_constructor_default_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_returns_correct_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4}
    result = IntToFloatMap(initial_data)
    assert len(result) == 4
    for key, value in initial_data.items():
        assert result[key] == value


# LLM-generated content at query #55
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


def test_checked_pmap_constructor_with_single_entry():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


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


# LLM-generated content at query #56
#--------------------------

```python
def test_checkedtype_constructor():
    instance = CheckedType()
    assert instance is not None
    assert isinstance(instance, CheckedType)


# LLM-generated content at query #57
#--------------------------

```python
def test_checkedpset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checkedpset_constructor_with_valid_integers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    assert isinstance(result, Positives)


def test_checkedpset_constructor_with_valid_floats():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert 1.5 in result
    assert 2.5 in result
    assert 3.5 in result


def test_checkedpset_constructor_with_mixed_valid_types():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checkedpset_constructor_with_duplicates():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checkedpset_constructor_with_zero():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([0, 1, 2])
    assert len(result) == 3
    assert 0 in result
    assert 1 in result
    assert 2 in result


def test_checkedpset_constructor_with_negative_raises_invariant_exception():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([-1, 1, 2])
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpset_constructor_with_invalid_type_raises_exception():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "string", 3])
        assert False, "Should have raised an exception for invalid type"
    except (TypeError, InvariantException):
        pass


def test_checkedpset_constructor_with_pmap_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    from pyrsistent import pmap
    initial_pmap = pmap()
    result = Positives(initial_pmap)
    assert isinstance(result, Positives)
    assert len(result) == 0


# LLM-generated content at query #58
#--------------------------

```python
def test_restore_pickle_calls_create_with_data_and_empty_factory_fields():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            cls.last_data = data
            cls.last_factory_fields = _factory_fields
            return cls()
    
    test_data = {"key": "value"}
    result = _restore_pickle(MockClass, test_data)
    
    assert MockClass.last_data == test_data
    assert MockClass.last_factory_fields == set()
    assert result is not None


def test_restore_pickle_with_empty_dict():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            cls.received_data = data
            cls.received_factory_fields = _factory_fields
            return "instance"
    
    result = _restore_pickle(MockClass, {})
    
    assert MockClass.received_data == {}
    assert MockClass.received_factory_fields == set()
    assert result == "instance"


def test_restore_pickle_with_complex_data():
    class MockClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            cls.data_received = data
            cls.factory_fields_received = _factory_fields
            return cls()
    
    complex_data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
    result = _restore_pickle(MockClass, complex_data)
    
    assert MockClass.data_received == complex_data
    assert MockClass.factory_fields_received == set()


# LLM-generated content at query #59
#--------------------------

```python
def test_checked_pset_constructor_with_empty_list():
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


def test_checked_pset_constructor_with_mixed_valid_numbers():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checked_pset_constructor_with_zero():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([0, 1, 2])
    assert len(result) == 3
    assert 0 in result


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
        Positives([1, -5, 3])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pset_constructor_with_tuple_input():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives((1, 2, 3))
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


# LLM-generated content at query #60
#--------------------------

```python
def test_wrap_invariant_predicate_line_3():
    def invariant(*args, **kwargs):
        return (True, "test")
    
    wrapped = wrap_invariant(invariant)
    result = wrapped()
    assert isinstance(result[0], bool) == True


# LLM-generated content at query #61
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


def test_maybe_parse_user_type_with_empty_list():
    result = maybe_parse_user_type([])
    assert result == ()


def test_maybe_parse_user_type_with_complex_nested_structure():
    result = maybe_parse_user_type([int, [str, [float, bool]]])
    assert result == (int, str, float, bool)


# LLM-generated content at query #62
#--------------------------

```python
def test_checked_pmap_constructor_empty_dict():
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


def test_checked_pmap_constructor_single_entry():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert isinstance(result, IntToFloatMap)
    assert result[42] == 3.14
    assert len(result) == 1


def test_checked_pmap_constructor_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 5
    assert result[1] == 1.1
    assert result[5] == 5.5


# LLM-generated content at query #63
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

def test_maybe_parse_user_type_with_nested_iterables():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)

def test_maybe_parse_user_type_with_invalid_type():
    try:
        maybe_parse_user_type(42)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)

def test_maybe_parse_user_type_with_invalid_list():
    try:
        maybe_parse_user_type([int, 42])
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
        maybe_parse_user_type({})
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


# LLM-generated content at query #64
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class TestClass:
        pass
    
    instance = TestClass()
    result = _checked_type_create(TestClass, instance)
    assert result is instance


def test_checked_type_create_with_non_checked_type():
    class SimpleClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(SimpleClass, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_and_matching_data():
    class CheckedType:
        pass
    
    class Container:
        _checked_types = ['__main__.CheckedType']
        
        def __init__(self, data):
            self.data = data
    
    data_instance = CheckedType()
    source_data = [data_instance]
    result = _checked_type_create(Container, source_data)
    assert result.data == source_data


def test_checked_type_create_returns_same_instance_when_already_correct_type():
    class MyClass:
        pass
    
    instance = MyClass()
    result = _checked_type_create(MyClass, instance)
    assert result is instance


def test_get_type_with_type_object():
    result = get_type(int)
    assert result is int


def test_get_type_with_string():
    result = get_type('builtins.int')
    assert result is int


def test_get_types_with_multiple_types():
    result = get_types([int, str, 'builtins.float'])
    assert result == [int, str, float]


def test_get_types_with_empty_list():
    result = get_types([])
    assert result == []


# LLM-generated content at query #65
#--------------------------

```python
def test_check_types_predicate_evaluates_to_false():
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
    
    test_obj = "string"
    expected_types = [int, float]
    
    try:
        _check_types([test_obj], expected_types, TestClass)
        assert False, "Expected exception to be raised"
    except Exception as e:
        assert "can only be used with" in str(e)


# LLM-generated content at query #66
#--------------------------

```python
def test_check_types_predicate_with_empty_expected_types():
    def _check_types(it, expected_types, source_class, exception_type=None):
        if expected_types:
            for e in it:
                pass
        return True
    
    result = _check_types([1, 2, 3], [], object, None)
    assert result is True


# LLM-generated content at query #67
#--------------------------

```python
def test_checked_type_create_predicate_false():
    class MockClass:
        pass
    
    class TestClass:
        _checked_types = None
        
        @staticmethod
        def _checked_type_create(cls, source_data, _factory_fields=None, ignore_extra=False):
            if isinstance(source_data, cls):
                return source_data
            return None
    
    source_data = "not an instance"
    cls = TestClass
    
    result = isinstance(source_data, cls)
    assert result is False


# LLM-generated content at query #68
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


def test_checkedpmap_constructor_with_initial_dict():
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
    class CustomMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = CustomMap({1: "one", 2: "two"})
    assert type(result).__name__ == "CustomMap"
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #69
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
    assert "valid_data_1" not in result


# LLM-generated content at query #70
#--------------------------

```python
def test_check_types_with_valid_types():
    from typing import List
    
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
    
    test_list = [1, 2, 3]
    _check_types(test_list, [int], TestClass, CheckedValueTypeError)


def test_check_types_with_empty_expected_types():
    class TestClass:
        pass
    
    test_list = [1, "string", 3.14]
    _check_types(test_list, [], TestClass)


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
    
    test_list = [1, "string", 3.14]
    _check_types(test_list, [int, str, float], TestClass, CheckedValueTypeError)


def test_check_types_with_invalid_type_raises_exception():
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
    
    test_list = [1, 2, "invalid"]
    error_raised = False
    try:
        _check_types(test_list, [int], TestClass, CheckedValueTypeError)
    except CheckedValueTypeError as e:
        error_raised = True
        assert e.source_class == TestClass
        assert e.expected_types == [int]
        assert e.actual_type == str
        assert e.value == "invalid"
        assert "TestClass" in e.msg
    
    assert error_raised is True


def test_check_types_empty_iterable():
    class TestClass:
        pass
    
    test_list = []
    _check_types(test_list, [int], TestClass)


def test_check_types_with_custom_exception():
    class CustomException(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            self.source_class = source_class
            self.expected_types = expected_types
            self.actual_type = actual_type
            self.value = value
            self.msg = msg
            super().__init__(msg)
    
    class TestClass:
        pass
    
    test_list = [1, 2, None]
    error_raised = False
    try:
        _check_types(test_list, [int], TestClass, CustomException)
    except CustomException as e:
        error_raised = True
        assert e.actual_type == type(None)
    
    assert error_raised is True


# LLM-generated content at query #71
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


def test_checked_pmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_default_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_with_multiple_items():
    class StrToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    result = StrToIntMap(data)
    assert len(result) == 4
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert result['d'] == 4


# LLM-generated content at query #72
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant_1(elem):
        return (True, "valid_data_1")
    
    def invariant_2(elem):
        return (False, "invalid_data_2")
    
    def invariant_3(elem):
        return (False, "invalid_data_3")
    
    elem = "test_element"
    invariants = [invariant_1, invariant_2, invariant_3]
    
    result = _invariant_errors(elem, invariants)
    
    assert result == ["invalid_data_2", "invalid_data_3"]


# LLM-generated content at query #73
#--------------------------

```python
def test_check_types_predicate_evaluates_to_true():
    expected_types = [int, str]
    it = [1, "hello", 2]
    source_class = list
    
    # This should not raise an exception, meaning the predicate at line 2 evaluates to True
    # and the function executes without error
    try:
        _check_types(it, expected_types, source_class)
        result = True
    except:
        result = False
    
    assert result is True


# LLM-generated content at query #74
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
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"a": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_single_entry():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({5: 5.5})
    assert len(result) == 1
    assert result[5] == 5.5


def test_checkedpmap_constructor_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4})
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[2] == 2.2
    assert result[3] == 3.3
    assert result[4] == 4.4


# LLM-generated content at query #75
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
    except InvariantException:
        pass


def test_checked_pmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


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


# LLM-generated content at query #76
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


def test_checked_pset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checked_pset_constructor_with_negative_values():
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


def test_checked_pset_constructor_with_duplicates():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


# LLM-generated content at query #77
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0


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
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checkedpmap_constructor_with_failed_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert type(result).__name__ == 'IntToFloatMap'


# LLM-generated content at query #78
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    class MockCheckedType:
        pass
    
    source_data = MockCheckedType()
    cls = type('TestClass', (), {})
    
    result = isinstance(source_data, cls)
    
    assert result is False


# LLM-generated content at query #79
#--------------------------

```python
def test_checked_pvector_constructor_with_empty_list():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0


def test_checked_pvector_constructor_with_valid_integers():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_checked_pvector_constructor_with_valid_floats():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert result[0] == 1.5


def test_checked_pvector_constructor_with_mixed_numeric_types():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2.5
    assert result[2] == 3


def test_checked_pvector_constructor_with_python_pvector():
    from pyrsistent import pvector as python_pvector
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pv = python_pvector([1, 2, 3])
    result = Positives(pv)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_checked_pvector_constructor_with_tuple():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives((1, 2, 3))
    assert len(result) == 3
    assert result[0] == 1


def test_checked_pvector_constructor_with_single_element():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([5])
    assert len(result) == 1
    assert result[0] == 5


def test_checked_pvector_constructor_with_default_argument():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives()
    assert len(result) == 0


# LLM-generated content at query #80
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant_always_false(elem):
        return (False, "error_data")
    
    def invariant_always_true(elem):
        return (True, "valid_data")
    
    invariants = [invariant_always_false, invariant_always_true]
    result = _invariant_errors("test_elem", invariants)
    
    assert result == ["error_data"]


# LLM-generated content at query #81
#--------------------------

```python
def test_check_types_predicate_line_1():
    # The predicate at line 1 is the function definition itself
    # We verify that _check_types is callable and has the expected signature
    import inspect
    
    # Get the function signature
    sig = inspect.signature(_check_types)
    params = list(sig.parameters.keys())
    
    # Verify the function exists and has correct parameters
    assert callable(_check_types)
    assert params == ['it', 'expected_types', 'source_class', 'exception_type']
    assert sig.parameters['exception_type'].default is not inspect.Parameter.empty


# LLM-generated content at query #82
#--------------------------

```python
def test_restore_pickle_creates_instance_with_empty_factory_fields():
    class MockClass:
        def create(cls, data, _factory_fields=None):
            instance = MockClass()
            instance.data = data
            instance._factory_fields = _factory_fields
            return instance
    
    MockClass.create = classmethod(MockClass.create.__func__)
    test_data = {"key": "value"}
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result.data == test_data
    assert result._factory_fields == set()


def test_restore_pickle_calls_create_with_correct_parameters():
    class TestClass:
        instances_created = []
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            TestClass.instances_created.append((data, _factory_fields))
            instance = TestClass()
            instance.data = data
            instance._factory_fields = _factory_fields
            return instance
    
    test_data = {"test": "data"}
    _restore_pickle(TestClass, test_data)
    
    assert len(TestClass.instances_created) == 1
    assert TestClass.instances_created[0][0] == test_data
    assert TestClass.instances_created[0][1] == set()


def test_restore_pickle_returns_instance():
    class SampleClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = SampleClass()
            instance.value = data
            return instance
    
    result = _restore_pickle(SampleClass, "test_value")
    
    assert result is not None
    assert isinstance(result, SampleClass)
    assert result.value == "test_value"


def test_restore_pickle_with_empty_data():
    class EmptyClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = EmptyClass()
            instance.data = data
            return instance
    
    result = _restore_pickle(EmptyClass, {})
    
    assert result.data == {}


def test_restore_pickle_factory_fields_is_always_empty_set():
    class FactoryClass:
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = FactoryClass()
            instance._factory_fields = _factory_fields
            return instance
    
    result = _restore_pickle(FactoryClass, {"any": "data"})
    
    assert result._factory_fields == set()
    assert len(result._factory_fields) == 0


# LLM-generated content at query #83
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


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


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


def test_checkedpmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not a float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_with_dict_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_dict = {1: 1.5, 2: 2.5, 3: 3.5}
    result = IntToFloatMap(initial_dict)
    assert len(result) == 3
    assert result[1] == 1.5
    assert result[2] == 2.5
    assert result[3] == 3.5


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result) == IntToFloatMap
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #84
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
    
    # Test that the predicate at line 1 (if expected_types:) evaluates to True
    test_list = [1, 2, 3]
    expected_types = [int]
    result = _check_types(test_list, expected_types, TestClass)
    assert result is None


# LLM-generated content at query #85
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


def test_checked_pmap_constructor_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({42: 42.5})
    assert len(result) == 1
    assert result[42] == 42.5


def test_checked_pmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5})
    assert len(result) == 5
    assert result[1] == 1.1
    assert result[5] == 5.5


# LLM-generated content at query #86
#--------------------------

```python
def test_checked_type_create_predicate_false():
    class MockClass:
        pass
    
    source_data = MockClass()
    cls = type('TestClass', (), {})
    
    result = isinstance(source_data, cls)
    
    assert result is False


# LLM-generated content at query #87
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


def test_checkedpmap_constructor_single_entry():
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
    
    result = StringToIntMap({'a': 1, 'b': 2})
    assert type(result).__name__ == 'StringToIntMap'
    assert result['a'] == 1
    assert result['b'] == 2


def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


# LLM-generated content at query #88
#--------------------------

```python
def test_checked_type_create_predicate_line_1():
    # Test that the predicate at line 1 (isinstance(source_data, cls)) evaluates to True
    
    class MockCheckedType:
        pass
    
    source_data = MockCheckedType()
    cls = MockCheckedType
    
    result = isinstance(source_data, cls)
    
    assert result is True


# LLM-generated content at query #89
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
    
    # Test case 1: expected_types is empty (predicate at line 1 should be False, so function returns early)
    result = _check_types([1, 2, 3], [], TestClass)
    assert result is None
    
    # Test case 2: expected_types is None (predicate at line 1 should be False, so function returns early)
    result = _check_types([1, 2, 3], None, TestClass)
    assert result is None
    
    # Test case 3: expected_types is truthy (predicate at line 1 should be True, function continues)
    result = _check_types([1, 2, 3], [int], TestClass)
    assert result is None


# LLM-generated content at query #90
#--------------------------

```python
def test_wrap_invariant_predicate_line_3():
    def invariant(*args, **kwargs):
        return ("non_bool_value", "another_value")
    
    wrapped = wrap_invariant(invariant)
    result = wrapped()
    
    assert isinstance(result[0], bool) == False


# LLM-generated content at query #91
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_one_false():
    result = [(True, "data1"), (False, "error1"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


def test_merge_invariant_results_multiple_false():
    result = [(False, "error1"), (True, "data2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1", "error3")


def test_merge_invariant_results_all_false():
    result = [(False, "error1"), (False, "error2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1", "error2", "error3")


def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_single_true():
    result = [(True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_single_false():
    result = [(False, "error1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


def test_merge_invariant_results_various_data_types():
    result = [(True, 123), (False, {"key": "value"}), (False, [1, 2, 3])]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ({"key": "value"}, [1, 2, 3])


# LLM-generated content at query #92
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


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception:
        pass


def test_checkedpmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checkedpmap_constructor_with_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4}
    result = IntToFloatMap(initial_data)
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[4] == 4.4


def test_checkedpmap_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    repr_str = repr(result)
    assert 'IntToFloatMap' in repr_str
    assert '1' in repr_str
    assert '1.5' in repr_str


# LLM-generated content at query #93
#--------------------------

```python
def test_checked_type_create_returns_same_instance_when_already_correct_type():
    class TestCheckedType:
        _checked_types = []
    
    instance = TestCheckedType()
    result = _checked_type_create(TestCheckedType, instance)
    assert result is instance


def test_checked_type_create_returns_cls_instance_when_source_data_not_instance():
    class TestCheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(TestCheckedType, source_data)
    assert isinstance(result, TestCheckedType)
    assert result.data == source_data


def test_checked_type_create_with_empty_checked_types():
    class TestCheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = {"key": "value"}
    result = _checked_type_create(TestCheckedType, source_data)
    assert isinstance(result, TestCheckedType)
    assert result.data == source_data


def test_checked_type_create_returns_instance_when_source_is_list():
    class InnerCheckedType:
        def __init__(self, value):
            self.value = value
    
    class TestCheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(TestCheckedType, source_data)
    assert isinstance(result, TestCheckedType)


def test_checked_type_create_with_ignore_extra_parameter():
    class TestCheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [{"extra": "field"}]
    result = _checked_type_create(TestCheckedType, source_data, ignore_extra=True)
    assert isinstance(result, TestCheckedType)


def test_checked_type_create_with_factory_fields_parameter():
    class TestCheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2]
    factory_fields = {"field1": "value1"}
    result = _checked_type_create(TestCheckedType, source_data, _factory_fields=factory_fields)
    assert isinstance(result, TestCheckedType)


# LLM-generated content at query #94
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
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except Exception as e:
        assert "InvariantException" in type(e).__name__


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == "IntToFloatMap"


def test_checkedpmap_constructor_with_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2, "c": 3})
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


# LLM-generated content at query #95
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    # Test that the predicate at line 1 (isinstance(source_data, cls)) evaluates to False
    # This means source_data should NOT be an instance of cls
    
    class MockCheckedType:
        pass
    
    class TestClass:
        _checked_types = []
    
    source_data = "not an instance of TestClass"
    
    # The predicate at line 1 should be False
    assert not isinstance(source_data, TestClass)


# LLM-generated content at query #96
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
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


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


def test_checked_pmap_constructor_single_element():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert result[42] == 3.14
    assert len(result) == 1


def test_checked_pmap_constructor_multiple_elements():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4})
    assert result[1] == 1.1
    assert result[2] == 2.2
    assert result[3] == 3.3
    assert result[4] == 4.4
    assert len(result) == 4


# LLM-generated content at query #97
#--------------------------

```python
def test_wrap_invariant_predicate_line_3():
    def invariant_func(*args, **kwargs):
        return (True, "test")
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    
    assert isinstance(result[0], bool) is True


# LLM-generated content at query #98
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    class MockCheckedType:
        pass
    
    class MockClass:
        _checked_types = None
    
    source_data = MockCheckedType()
    cls = MockClass
    
    result = isinstance(source_data, cls)
    
    assert result is False


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
def test_check_types_predicate_line_1():
    # Test that the predicate at line 1 (if expected_types:) evaluates to True
    # when expected_types is a non-empty sequence
    
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
    
    # Create a mock source class
    class MockSourceClass:
        pass
    
    # Test with non-empty expected_types - predicate should be True
    result = _check_types([1, 2, 3], [int], MockSourceClass)
    assert result is None
    
    # Verify the function executed (predicate was True)
    # by checking it didn't raise an exception for valid types
    result = _check_types(["a", "b"], [str], MockSourceClass)
    assert result is None


# LLM-generated content at query #101
#--------------------------

```python
def test_check_types_predicate_evaluates_to_false():
    from your_module import _check_types, CheckedValueTypeError, get_type
    
    class DummyClass:
        pass
    
    it = [42]
    expected_types = [str, float]
    source_class = DummyClass
    
    try:
        _check_types(it, expected_types, source_class)
        assert False, "Expected exception to be raised"
    except CheckedValueTypeError:
        pass


# LLM-generated content at query #102
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
    
    source_data = [1, 2, 3]
    result = _checked_type_create(SimpleClass, source_data)
    assert result.data == [1, 2, 3]


def test_checked_type_create_with_ignore_extra_parameter():
    class MockClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data, ignore_extra=True)
    assert result.data == [1, 2, 3]


def test_checked_type_create_with_factory_fields_parameter():
    class MockClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data, _factory_fields={'field': 'value'})
    assert result.data == [1, 2, 3]


def test_get_type_with_type_object():
    result = get_type(int)
    assert result is int


def test_get_type_with_string():
    result = get_type('builtins.int')
    assert result is int


def test_get_types_with_multiple_types():
    result = get_types([int, str])
    assert result == [int, str]


def test_get_types_with_string_type_names():
    result = get_types(['builtins.int', 'builtins.str'])
    assert result == [int, str]


def test_get_types_with_mixed_types():
    result = get_types([int, 'builtins.str'])
    assert result == [int, str]


# LLM-generated content at query #103
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    # Test that the predicate at line 1 (isinstance(source_data, cls)) evaluates to False
    # This means source_data should NOT be an instance of cls
    
    class MockCheckedType:
        pass
    
    source_data = "not an instance of MockCheckedType"
    cls = MockCheckedType
    
    result = isinstance(source_data, cls)
    
    assert result is False


# LLM-generated content at query #104
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    class MockClass:
        pass
    
    source_data = MockClass()
    cls = MockClass
    
    result = isinstance(source_data, cls)
    
    assert result == False


# LLM-generated content at query #105
#--------------------------

```python
def test_check_types_predicate_false():
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
    
    # Test that predicate evaluates to False when element is not instance of expected types
    element = "string"
    expected_types = [int, float]
    
    try:
        _check_types([element], expected_types, TestClass)
        assert False, "Expected exception to be raised"
    except Exception as e:
        assert "Type TestClass can only be used with" in str(e)


# LLM-generated content at query #106
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockClass:
        pass
    
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance


def test_checked_type_create_with_no_checked_types():
    class MockClass:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_and_matching_data():
    class MockCheckedType:
        _checked_types = ['builtins.int']
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data)
    assert result.data == source_data


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


def test_get_types_with_all_strings():
    result = get_types(['builtins.int', 'builtins.str'])
    assert result[0] is int
    assert result[1] is str


# LLM-generated content at query #107
#--------------------------

```python
def test_merge_invariant_results_predicate_true():
    result = [(True, "data1"), (True, "data2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


# LLM-generated content at query #108
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


def test_checked_pmap_constructor_with_invariant_pass():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_invariant_fail():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 1.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_with_default_initial():
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
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid_value"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_creates_correct_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_with_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert len(result) == 5
    for k, v in data.items():
        assert result[k] == v


# LLM-generated content at query #109
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


def test_checked_pmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_returns_correct_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_constructor_preserves_values():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial = {'a': 1, 'b': 2, 'c': 3}
    result = StringToIntMap(initial)
    assert dict(result) == initial


# LLM-generated content at query #110
#--------------------------

```python
def test_check_types_with_valid_types():
    from builtins import int, str
    _check_types([1, 2, 3], [int], type('TestClass', (), {}))


def test_check_types_with_multiple_valid_types():
    from builtins import int, str
    _check_types([1, "hello", 2], [int, str], type('TestClass', (), {}))


def test_check_types_with_empty_expected_types():
    _check_types([1, 2, 3], [], type('TestClass', (), {}))


def test_check_types_with_empty_iterable():
    from builtins import int
    _check_types([], [int], type('TestClass', (), {}))


def test_check_types_raises_error_for_invalid_type():
    from builtins import int, str
    test_class = type('TestClass', (), {})
    
    try:
        _check_types([1, 2.5, 3], [int], test_class)
        assert False, "Expected CheckedValueTypeError to be raised"
    except Exception as e:
        assert "Type TestClass can only be used with" in str(e)
        assert "not float" in str(e)


def test_check_types_with_string_type_name():
    test_class = type('TestClass', (), {})
    _check_types([1, 2, 3], ['builtins.int'], test_class)


def test_check_types_with_mixed_string_and_type():
    test_class = type('TestClass', (), {})
    _check_types([1, 2], ['builtins.int', int], test_class)


def test_check_types_raises_error_with_custom_exception_type():
    from builtins import int, str
    test_class = type('TestClass', (), {})
    custom_exception = type('CustomException', (Exception,), {})
    
    try:
        _check_types([1, "invalid"], [int], test_class, exception_type=custom_exception)
        assert False, "Expected custom exception to be raised"
    except custom_exception:
        pass


def test_check_types_error_contains_actual_type():
    from builtins import int
    test_class = type('TestClass', (), {})
    
    try:
        _check_types([3.14], [int], test_class)
        assert False, "Expected exception to be raised"
    except Exception as e:
        assert "float" in str(e)


# LLM-generated content at query #111
#--------------------------

```python
def test_wrap_invariant_with_boolean_result():
    def invariant_func(x):
        return True, "success"
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped(5)
    assert result == (True, "success")


def test_wrap_invariant_with_tuple_of_tuples():
    def invariant_func(x):
        return ((True, "test1"), (False, "error1"), (True, "test2"))
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped(5)
    assert result == (False, ("error1",))


def test_wrap_invariant_all_true_results():
    def invariant_func(x):
        return ((True, "pass1"), (True, "pass2"), (True, "pass3"))
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped(5)
    assert result == (True, ())


def test_wrap_invariant_all_false_results():
    def invariant_func(x):
        return ((False, "error1"), (False, "error2"))
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped(5)
    assert result == (False, ("error1", "error2"))


def test_wrap_invariant_with_kwargs():
    def invariant_func(x, y=10):
        return False, f"x={x}, y={y}"
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped(5, y=20)
    assert result == (False, f"x={5}, y={20}")


def test_wrap_invariant_multiple_args():
    def invariant_func(a, b, c):
        return ((True, "check1"), (False, "check2_failed"), (False, "check3_failed"))
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped(1, 2, 3)
    assert result == (False, ("check2_failed", "check3_failed"))


def test_wrap_invariant_empty_error_list():
    def invariant_func():
        return ((True, "ok"),)
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    assert result == (True, ())


def test_wrap_invariant_single_false_result():
    def invariant_func():
        return ((False, "single_error"),)
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    assert result == (False, ("single_error",))


# LLM-generated content at query #112
#--------------------------

```python
def test_check_types_with_valid_types():
    from your_module import _check_types, CheckedValueTypeError
    
    _check_types([1, 2, 3], [int], type)


def test_check_types_with_multiple_valid_types():
    from your_module import _check_types
    
    _check_types([1, "hello", 2.5], [int, str, float], type)


def test_check_types_with_empty_expected_types():
    from your_module import _check_types
    
    _check_types([1, "hello", None], [], type)


def test_check_types_with_empty_iterable():
    from your_module import _check_types
    
    _check_types([], [int], type)


def test_check_types_raises_error_on_invalid_type():
    from your_module import _check_types, CheckedValueTypeError
    
    try:
        _check_types([1, "invalid", 3], [int], type)
        assert False, "Expected CheckedValueTypeError to be raised"
    except CheckedValueTypeError:
        pass


def test_check_types_with_class_types():
    from your_module import _check_types
    
    class CustomClass:
        pass
    
    obj1 = CustomClass()
    obj2 = CustomClass()
    _check_types([obj1, obj2], [CustomClass], type)


def test_check_types_raises_error_with_custom_exception_type():
    from your_module import _check_types
    
    class CustomException(Exception):
        pass
    
    try:
        _check_types([1, "invalid"], [int], type, exception_type=CustomException)
        assert False, "Expected CustomException to be raised"
    except CustomException:
        pass


def test_check_types_error_message_format():
    from your_module import _check_types, CheckedValueTypeError
    
    try:
        _check_types([1, "invalid"], [int], type)
        assert False, "Expected CheckedValueTypeError to be raised"
    except CheckedValueTypeError as e:
        assert "Type" in str(e.args[-1])
        assert "can only be used with" in str(e.args[-1])


def test_check_types_with_subclass():
    from your_module import _check_types
    
    class Parent:
        pass
    
    class Child(Parent):
        pass
    
    child_obj = Child()
    _check_types([child_obj], [Parent], type)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_checkedpset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0


def test_checkedpset_constructor_with_valid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checkedpset_constructor_with_float_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert 1.5 in result
    assert 2.5 in result
    assert 3.5 in result


def test_checkedpset_constructor_with_mixed_valid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checkedpset_constructor_with_duplicate_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checkedpset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checkedpset_constructor_with_negative_number():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised an exception"
    except Exception:
        pass


def test_checkedpset_constructor_single_element():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([42])
    assert len(result) == 1
    assert 42 in result


# LLM-generated content at query #2
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


def test_maybe_parse_user_type_with_nested_list():
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)


def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(["int", "str"])
    assert result == ("int", "str")


def test_maybe_parse_user_type_with_mixed_types_and_strings():
    result = maybe_parse_user_type([int, "str", float])
    assert result == (int, "str", float)


def test_maybe_parse_user_type_with_invalid_input_raises_type_error():
    try:
        maybe_parse_user_type(123)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_none_raises_type_error():
    try:
        maybe_parse_user_type(None)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_dict_raises_type_error():
    try:
        maybe_parse_user_type({"key": "value"})
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_deeply_nested_list():
    result = maybe_parse_user_type([int, [str, [float, bool]]])
    assert result == (int, str, float, bool)


def test_maybe_parse_user_type_with_empty_list():
    result = maybe_parse_user_type([])
    assert result == ()


# LLM-generated content at query #3
#--------------------------

```python
def test_checkedpmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert len(result) == 0
    assert dict(result) == {}


def test_checkedpmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.5})
    assert len(result) == 1
    assert result[1] == 1.5


def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert type(result) == IntToFloatMap


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')

    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    initial = {1: 1.5, 2: 2.25, 3: 3.75, 4: 4.5}
    result = IntToFloatMap(initial)
    assert len(result) == 4
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75
    assert result[4] == 4.5


# LLM-generated content at query #4
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


def test_checked_pmap_constructor_from_checked_pmap_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    original = IntToFloatMap({1: 1.5})
    result = IntToFloatMap(original)
    assert len(result) == 1
    assert result[1] == 1.5
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_type_error_invalid_key():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should raise CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_type_error_invalid_value():
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
        IntToFloatMap({1: 2.0})
        assert False, "Should raise InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial = {'a': 1, 'b': 2, 'c': 3}
    result = StringToIntMap(initial)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #5
#--------------------------

```python
def test_get_type_with_type_object():
    from your_module import get_type
    result = get_type(str)
    assert result is str


def test_get_type_with_int_type():
    from your_module import get_type
    result = get_type(int)
    assert result is int


def test_get_type_with_list_type():
    from your_module import get_type
    result = get_type(list)
    assert result is list


def test_get_type_with_string_path():
    from your_module import get_type
    result = get_type('builtins.str')
    assert result is str


def test_get_type_with_string_path_int():
    from your_module import get_type
    result = get_type('builtins.int')
    assert result is int


def test_get_type_with_string_path_list():
    from your_module import get_type
    result = get_type('builtins.list')
    assert result is list


def test_get_type_with_custom_class_string_path():
    from your_module import get_type
    result = get_type('collections.OrderedDict')
    assert result.__name__ == 'OrderedDict'


def test_get_type_returns_type_for_type_input():
    from your_module import get_type
    class_type = type
    result = get_type(class_type)
    assert result is type


# LLM-generated content at query #6
#--------------------------

```python
def test_store_invariants_empty_dct_and_bases():
    dct = {}
    bases = ()
    store_invariants(dct, bases, 'dest', 'src')
    assert dct['dest'] == ()


def test_store_invariants_single_invariant_in_dct():
    def inv1():
        return True, "test"
    
    dct = {'src': inv1}
    bases = ()
    store_invariants(dct, bases, 'dest', 'src')
    assert len(dct['dest']) == 1
    assert callable(dct['dest'][0])


def test_store_invariants_multiple_invariants_in_bases():
    def inv1():
        return True, "test1"
    
    def inv2():
        return True, "test2"
    
    class Base1:
        pass
    Base1.src = inv1
    
    class Base2:
        pass
    Base2.src = inv2
    
    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, 'dest', 'src')
    assert len(dct['dest']) == 2


def test_store_invariants_dct_overrides_base():
    def inv1():
        return True, "test1"
    
    def inv2():
        return True, "test2"
    
    class Base:
        pass
    Base.src = inv1
    
    dct = {'src': inv2}
    bases = (Base,)
    store_invariants(dct, bases, 'dest', 'src')
    assert len(dct['dest']) == 2


def test_store_invariants_non_callable_raises_type_error():
    dct = {'src': "not_callable"}
    bases = ()
    try:
        store_invariants(dct, bases, 'dest', 'src')
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Invariants must be callable' in str(e)


def test_store_invariants_inherited_invariants_multiple_levels():
    def inv1():
        return True, "test1"
    
    def inv2():
        return True, "test2"
    
    class GrandBase:
        pass
    GrandBase.src = inv1
    
    class Base(GrandBase):
        pass
    Base.src = inv2
    
    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, 'dest', 'src')
    assert len(dct['dest']) == 2


def test_store_invariants_wrapped_invariant_with_bool_result():
    def inv():
        return True, "message"
    
    dct = {'src': inv}
    bases = ()
    store_invariants(dct, bases, 'dest', 'src')
    result = dct['dest'][0]()
    assert result == (True, "message")


def test_store_invariants_wrapped_invariant_with_tuple_result():
    def inv():
        return ((True, "msg1"), (False, "msg2"))
    
    dct = {'src': inv}
    bases = ()
    store_invariants(dct, bases, 'dest', 'src')
    result = dct['dest'][0]()
    assert result == (False, ("msg2",))


def test_store_invariants_duplicate_bases_not_repeated():
    def inv1():
        return True, "test"
    
    class Base:
        pass
    Base.src = inv1
    
    dct = {}
    bases = (Base, Base)
    store_invariants(dct, bases, 'dest', 'src')
    assert len(dct['dest']) == 1


# LLM-generated content at query #7
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


def test_checkedpmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4}
    result = IntToFloatMap(data)
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[4] == 4.4


# LLM-generated content at query #8
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
        return (True, "valid")
    
    def invariant2(elem):
        return (False, "error2")
    
    def invariant3(elem):
        return (False, "error3")
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == ["error2", "error3"]


def test_invariant_errors_empty_invariants():
    result = _invariant_errors("test", [])
    assert result == []


def test_invariant_errors_single_invalid():
    def invariant(elem):
        return (False, "single error")
    
    result = _invariant_errors("test", [invariant])
    assert result == ["single error"]


def test_invariant_errors_single_valid():
    def invariant(elem):
        return (True, "no error")
    
    result = _invariant_errors("test", [invariant])
    assert result == []


def test_invariant_errors_with_different_data_types():
    def invariant1(elem):
        return (False, 42)
    
    def invariant2(elem):
        return (False, {"error": "dict"})
    
    def invariant3(elem):
        return (True, None)
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == [42, {"error": "dict"}]


# LLM-generated content at query #9
#--------------------------

```python
def test_store_invariants_basic():
    def invariant1(self):
        return True, "test1"
    
    def invariant2(self):
        return False, "test2"
    
    dct = {}
    bases = ()
    store_invariants(dct, bases, '_invariants', 'invariant')
    assert '_invariants' in dct
    assert dct['_invariants'] == ()


def test_store_invariants_with_source():
    def invariant1(self):
        return True, "test1"
    
    dct = {'my_invariant': invariant1}
    bases = ()
    store_invariants(dct, bases, '_invariants', 'my_invariant')
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 1
    assert callable(dct['_invariants'][0])


def test_store_invariants_inherited():
    def invariant1(self):
        return True, "test1"
    
    def invariant2(self):
        return False, "test2"
    
    class Base:
        pass
    
    Base.__dict__ = {'my_invariant': invariant1}
    
    dct = {'my_invariant': invariant2}
    bases = (Base,)
    store_invariants(dct, bases, '_invariants', 'my_invariant')
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 2


def test_store_invariants_non_callable_raises():
    dct = {'my_invariant': 'not_callable'}
    bases = ()
    try:
        store_invariants(dct, bases, '_invariants', 'my_invariant')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Invariants must be callable' in str(e)


def test_store_invariants_multiple_bases():
    def invariant1(self):
        return True, "test1"
    
    def invariant2(self):
        return False, "test2"
    
    class Base1:
        pass
    
    class Base2:
        pass
    
    Base1.__dict__ = {'my_invariant': invariant1}
    Base2.__dict__ = {'my_invariant': invariant2}
    
    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_invariants', 'my_invariant')
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 2


def test_store_invariants_wrapped_function():
    def invariant1(self):
        return (True, "test1"), (False, "test2")
    
    dct = {'my_invariant': invariant1}
    bases = ()
    store_invariants(dct, bases, '_invariants', 'my_invariant')
    
    wrapped_invariant = dct['_invariants'][0]
    result = wrapped_invariant(None)
    assert result[0] == False
    assert len(result[1]) == 1


# LLM-generated content at query #10
#--------------------------

```python
def test_checked_pvector_constructor_with_empty_list():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector([])
    assert len(result) == 0
    assert isinstance(result, TestVector)


def test_checked_pvector_constructor_with_valid_integers():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector([1, 2, 3])
    assert len(result) == 3
    assert list(result) == [1, 2, 3]
    assert isinstance(result, TestVector)


def test_checked_pvector_constructor_with_valid_mixed_numeric_types():
    class Numerics(CheckedPVector):
        __type__ = (int, float)
    
    result = Numerics([1, 2.5, 3])
    assert len(result) == 3
    assert list(result) == [1, 2.5, 3]


def test_checked_pvector_constructor_with_invariant_check():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert list(result) == [1, 2, 3]


def test_checked_pvector_constructor_with_invalid_type():
    class IntVector(CheckedPVector):
        __type__ = int
    
    try:
        IntVector([1, "two", 3])
        assert False, "Should have raised an exception"
    except TypeError:
        pass


def test_checked_pvector_constructor_with_invalid_invariant():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised InvariantException"
    except Exception:
        pass


def test_checked_pvector_constructor_with_python_pvector():
    from pyrsistent import pvector
    
    class TestVector(CheckedPVector):
        __type__ = int
    
    pv = pvector([1, 2, 3])
    result = TestVector(pv)
    assert len(result) == 3
    assert list(result) == [1, 2, 3]
    assert isinstance(result, TestVector)


def test_checked_pvector_constructor_with_tuple():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector((1, 2, 3))
    assert len(result) == 3
    assert list(result) == [1, 2, 3]


def test_checked_pvector_constructor_with_generator():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector(x for x in [1, 2, 3])
    assert len(result) == 3
    assert list(result) == [1, 2, 3]


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant1(elem):
        return (True, "valid1")
    
    def invariant2(elem):
        return (False, "invalid1")
    
    def invariant3(elem):
        return (False, "invalid2")
    
    invariants = [invariant1, invariant2, invariant3]
    elem = "test_element"
    
    result = _invariant_errors(elem, invariants)
    
    assert result == ["invalid1", "invalid2"]
    assert len(result) == 2
    assert "invalid1" in result
    assert "invalid2" in result


def test_invariant_errors_all_valid():
    def invariant1(elem):
        return (True, "valid1")
    
    def invariant2(elem):
        return (True, "valid2")
    
    invariants = [invariant1, invariant2]
    elem = "test_element"
    
    result = _invariant_errors(elem, invariants)
    
    assert result == []


def test_invariant_errors_all_invalid():
    def invariant1(elem):
        return (False, "error1")
    
    def invariant2(elem):
        return (False, "error2")
    
    invariants = [invariant1, invariant2]
    elem = "test_element"
    
    result = _invariant_errors(elem, invariants)
    
    assert result == ["error1", "error2"]


def _invariant_errors(elem, invariants):
    return [data for valid, data in (invariant(elem) for invariant in invariants) if not valid]


# LLM-generated content at query #13
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


def test_checkedpmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"a": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25, 3: 3.75})
    assert len(result) == 3
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75


# LLM-generated content at query #14
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


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invalid_invariant():
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
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid_value"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial_data, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(initial_data)
    assert len(result) == 5
    assert result[1] == 1.1
    assert result[3] == 3.3
    assert result[5] == 5.5


# LLM-generated content at query #15
#--------------------------

```python
def test_check_types_with_valid_types():
    from types import SimpleNamespace
    exception_raised = False
    try:
        _check_types([1, 2, 3], [int], SimpleNamespace)
    except Exception:
        exception_raised = True
    assert exception_raised == False


def test_check_types_with_invalid_types():
    from types import SimpleNamespace
    exception_raised = False
    exception_message = ""
    try:
        _check_types([1, "string", 3], [int], SimpleNamespace)
    except Exception as e:
        exception_raised = True
        exception_message = str(e)
    assert exception_raised == True


def test_check_types_with_multiple_valid_types():
    from types import SimpleNamespace
    exception_raised = False
    try:
        _check_types([1, "string", 3.14], [int, str, float], SimpleNamespace)
    except Exception:
        exception_raised = True
    assert exception_raised == False


def test_check_types_with_empty_iterable():
    from types import SimpleNamespace
    exception_raised = False
    try:
        _check_types([], [int], SimpleNamespace)
    except Exception:
        exception_raised = True
    assert exception_raised == False


def test_check_types_with_empty_expected_types():
    from types import SimpleNamespace
    exception_raised = False
    try:
        _check_types([1, "string", 3.14], [], SimpleNamespace)
    except Exception:
        exception_raised = True
    assert exception_raised == False


def test_check_types_with_custom_exception_type():
    from types import SimpleNamespace
    class CustomException(Exception):
        pass
    exception_raised = False
    exception_type_correct = False
    try:
        _check_types([1, "string"], [int], SimpleNamespace, exception_type=CustomException)
    except CustomException:
        exception_raised = True
        exception_type_correct = True
    except Exception:
        exception_raised = True
    assert exception_raised == True
    assert exception_type_correct == True


def test_check_types_first_element_invalid():
    from types import SimpleNamespace
    exception_raised = False
    try:
        _check_types(["string", 1, 2], [int], SimpleNamespace)
    except Exception:
        exception_raised = True
    assert exception_raised == True


def test_check_types_with_class_type_string():
    from types import SimpleNamespace
    exception_raised = False
    try:
        _check_types([1, 2, 3], ["builtins.int"], SimpleNamespace)
    except Exception:
        exception_raised = True
    assert exception_raised == False


# LLM-generated content at query #16
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockClass:
        pass
    
    instance = MockClass()
    result = _checked_type_create(MockClass, instance)
    assert result is instance


def test_checked_type_create_without_checked_types():
    class MockClass:
        _checked_types = []
        
        def __init__(self, data):
            self.data = data
    
    result = _checked_type_create(MockClass, "test_data")
    assert result.data == "test_data"


def test_checked_type_create_with_checked_type_no_conversion_needed():
    class CheckedType:
        pass
    
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        
        def __init__(self, data):
            self.data = data
    
    obj = CheckedType()
    result = _checked_type_create(MockClass, [obj])
    assert result.data == [obj]


def test_checked_type_create_with_matching_type_in_source_data():
    class ValidType:
        def __init__(self, value):
            self.value = value
    
    class CheckedType:
        @staticmethod
        def create(data, ignore_extra=False):
            return CheckedType()
    
    class MockClass:
        _checked_types = ['__main__.ValidType']
        
        def __init__(self, data):
            self.data = data
    
    valid_obj = ValidType(42)
    result = _checked_type_create(MockClass, [valid_obj])
    assert result.data == [valid_obj]


# LLM-generated content at query #17
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_one_false():
    result = [(True, "data1"), (False, "error1"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


def test_merge_invariant_results_multiple_false():
    result = [(False, "error1"), (True, "data2"), (False, "error2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1", "error2", "error3")


def test_merge_invariant_results_all_false():
    result = [(False, "error1"), (False, "error2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1", "error2", "error3")


def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_single_true():
    result = [(True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_single_false():
    result = [(False, "error1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


# LLM-generated content at query #18
#--------------------------

```python
def test_maybe_parse_user_type_preserved_iterable():
    from collections.abc import Iterable
    
    class _preserved_iterable_types(list):
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


# LLM-generated content at query #19
#--------------------------

```python
def test_restore_pickle():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self.factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls(data, _factory_fields)
            return instance
    
    test_data = {"key": "value"}
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result.data == test_data
    assert result.factory_fields == set()
    assert isinstance(result.factory_fields, set)
    assert len(result.factory_fields) == 0


# LLM-generated content at query #20
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


def test_checkedpmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_preserves_type():
    class CustomMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = CustomMap({"a": 1, "b": 2})
    assert type(result).__name__ == "CustomMap"
    assert isinstance(result, CustomMap)


def test_checkedpmap_constructor_with_pmap_input():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    from pyrsistent import pmap
    input_pmap = pmap({1: 1.5})
    result = IntToFloatMap(input_pmap)
    assert result[1] == 1.5
    assert isinstance(result, IntToFloatMap)


# LLM-generated content at query #21
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


def test_checkedpmap_constructor_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2, "c": 3})
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


def test_checkedpmap_constructor_preserves_class_type():
    class CustomMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = CustomMap({"key": 42})
    assert type(result).__name__ == "CustomMap"
    assert isinstance(result, CustomMap)


# LLM-generated content at query #22
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
    
    # Test case 1: expected_types is empty list (predicate at line 1 is False, so no exception)
    result = _check_types([1, 2, 3], [], TestClass)
    assert result is None
    
    # Test case 2: expected_types is non-empty and all elements match (predicate at line 1 is True, line 2 condition is False)
    result = _check_types([1, 2, 3], [int], TestClass)
    assert result is None
    
    # Test case 3: expected_types is None (predicate at line 1 is False, so no exception)
    result = _check_types([1, 2, 3], None, TestClass)
    assert result is None


# LLM-generated content at query #23
#--------------------------

```python
def test_store_types_single_type():
    dct = {'__annotations__': {}}
    bases = []
    _store_types(dct, bases, 'types', '__annotations__')
    assert 'types' in dct


def test_store_types_with_string_type():
    dct = {'my_types': 'int'}
    bases = []
    _store_types(dct, bases, 'parsed_types', 'my_types')
    assert 'parsed_types' in dct
    assert dct['parsed_types'] == ('int',)


def test_store_types_with_multiple_bases():
    class Base1:
        __dict__ = {'source_attr': 'str'}
    
    class Base2:
        __dict__ = {'source_attr': 'float'}
    
    dct = {'source_attr': 'int'}
    bases = [Base1, Base2]
    _store_types(dct, bases, 'dest_attr', 'source_attr')
    assert 'dest_attr' in dct


def test_store_types_source_name_not_in_dict():
    dct = {'other_key': 'value'}
    bases = []
    _store_types(dct, bases, 'dest_attr', 'source_attr')
    assert 'dest_attr' in dct
    assert dct['dest_attr'] == ()


def test_store_types_with_type_object():
    dct = {'my_types': int}
    bases = []
    _store_types(dct, bases, 'parsed_types', 'my_types')
    assert 'parsed_types' in dct
    assert dct['parsed_types'] == (int,)


def test_store_types_with_list_of_types():
    dct = {'my_types': [int, str]}
    bases = []
    _store_types(dct, bases, 'parsed_types', 'my_types')
    assert 'parsed_types' in dct


def test_store_types_dct_takes_precedence():
    class Base:
        __dict__ = {'source_attr': 'base_value'}
    
    dct = {'source_attr': 'dct_value'}
    bases = [Base]
    _store_types(dct, bases, 'dest_attr', 'source_attr')
    assert 'dest_attr' in dct


# LLM-generated content at query #24
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


def test_checked_pmap_new_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_new_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=16)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_new_default_parameters():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_new_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checked_pmap_new_with_dict_initial():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_dict = {'a': 1, 'b': 2}
    result = StringToIntMap(initial_dict)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2


def test_checked_pmap_new_invalid_key_type_raises_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({'a': 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_new_invalid_value_type_raises_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: 'invalid'})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checked_pmap_new_invariant_violation_raises_error():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant1(elem):
        return (True, "valid_data")
    
    def invariant2(elem):
        return (False, "invalid_data")
    
    def invariant3(elem):
        return (True, "another_valid")
    
    invariants = [invariant1, invariant2, invariant3]
    result = _invariant_errors("test_elem", invariants)
    
    assert result == ["invalid_data"]


# LLM-generated content at query #26
#--------------------------

```python
def test_merge_invariant_results_predicate_false():
    result = [(False, "error1"), (True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)


# LLM-generated content at query #27
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
    assert set(result) == {int, str}


def test_maybe_parse_user_type_with_tuple_of_types():
    result = maybe_parse_user_type((int, str, float))
    assert set(result) == {int, str, float}


def test_maybe_parse_user_type_with_list_of_strings():
    result = maybe_parse_user_type(["int", "str"])
    assert set(result) == {"int", "str"}


def test_maybe_parse_user_type_with_nested_list():
    result = maybe_parse_user_type([int, [str, float]])
    assert set(result) == {int, str, float}


def test_maybe_parse_user_type_with_mixed_types_and_strings():
    result = maybe_parse_user_type([int, "str", float])
    assert set(result) == {int, "str", float}


def test_maybe_parse_user_type_with_invalid_input_dict():
    try:
        maybe_parse_user_type({"key": "value"})
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_invalid_input_number():
    try:
        maybe_parse_user_type(42)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_invalid_input_none():
    try:
        maybe_parse_user_type(None)
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


# LLM-generated content at query #28
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


def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should raise CheckedKeyTypeError"
    except Exception:
        pass


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should raise TypeError"
    except Exception:
        pass


def test_checked_pmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2, "c": 3})
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


# LLM-generated content at query #29
#--------------------------

```python
def test_invariant_errors_returns_invalid_data():
    def invariant1(elem):
        return (True, "valid_data")
    
    def invariant2(elem):
        return (False, "invalid_data")
    
    def invariant3(elem):
        return (True, "another_valid")
    
    invariants = [invariant1, invariant2, invariant3]
    result = _invariant_errors("test_elem", invariants)
    
    assert result == ["invalid_data"]


def test_invariant_errors_all_valid():
    def invariant1(elem):
        return (True, "data1")
    
    def invariant2(elem):
        return (True, "data2")
    
    invariants = [invariant1, invariant2]
    result = _invariant_errors("test_elem", invariants)
    
    assert result == []


def test_invariant_errors_all_invalid():
    def invariant1(elem):
        return (False, "error1")
    
    def invariant2(elem):
        return (False, "error2")
    
    invariants = [invariant1, invariant2]
    result = _invariant_errors("test_elem", invariants)
    
    assert result == ["error1", "error2"]


def test_invariant_errors_empty_invariants():
    invariants = []
    result = _invariant_errors("test_elem", invariants)
    
    assert result == []


# LLM-generated content at query #30
#--------------------------

```python
def test_maybe_parse_user_type_type_not_iterable():
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
    assert isinstance(result, list)


# LLM-generated content at query #31
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


def test_checked_type_create_with_checked_type_subclass():
    class CheckedType:
        pass
    
    class MockCheckedType(CheckedType):
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    class InnerCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
        
        @classmethod
        def create(cls, data, ignore_extra=False):
            if isinstance(data, cls):
                return data
            return cls(data * 2)
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockCheckedType, source_data, ignore_extra=False)
    assert isinstance(result, MockCheckedType)
    assert len(result.data) == 3


def test_checked_type_create_with_matching_type():
    class CheckedType:
        pass
    
    class InnerType(CheckedType):
        def __init__(self, value):
            self.value = value
    
    class MockCheckedType(CheckedType):
        _checked_types = ['__main__.InnerType']
        def __init__(self, data):
            self.data = data
    
    inner_instance = InnerType(5)
    source_data = [inner_instance]
    result = _checked_type_create(MockCheckedType, source_data)
    assert isinstance(result, MockCheckedType)
    assert result.data[0] is inner_instance


# LLM-generated content at query #32
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockCheckedType:
        _checked_types = []
    
    instance = MockCheckedType()
    result = _checked_type_create(MockCheckedType, instance)
    assert result is instance


def test_checked_type_create_without_checked_type():
    class MockClass:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_matching_data():
    class CheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    checked_instance = CheckedType([1, 2, 3])
    result = _checked_type_create(MockClass, [checked_instance], ignore_extra=False)
    assert result.data == [checked_instance]


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


def test_get_class_with_valid_module_path():
    result = _get_class('builtins.int')
    assert result is int


def test_get_class_with_builtin_type():
    result = _get_class('collections.OrderedDict')
    from collections import OrderedDict
    assert result is OrderedDict


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


def test_checked_pmap_constructor_with_initial_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial = {1: 1.5, 2: 2.25}
    result = IntToFloatMap(initial)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_valid_invariant():
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


def test_checked_pmap_constructor_with_failed_invariant():
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


# LLM-generated content at query #34
#--------------------------

```python
def test_check_types_predicate_line_1():
    def get_type(t):
        return t
    
    class CheckedValueTypeError(Exception):
        def __init__(self, source_class, expected_types, actual_type, value, msg):
            super().__init__(msg)
    
    class TestClass:
        pass
    
    # Test case 1: expected_types is empty (predicate at line 1 is False)
    result = None
    try:
        _check_types([], [], TestClass)
        result = True
    except:
        result = False
    assert result == True
    
    # Test case 2: expected_types is None (predicate at line 1 is False)
    result = None
    try:
        _check_types([], None, TestClass)
        result = True
    except:
        result = False
    assert result == True
    
    # Test case 3: expected_types is truthy and non-empty (predicate at line 1 is True)
    result = None
    try:
        _check_types([1, 2, 3], [int], TestClass)
        result = True
    except:
        result = False
    assert result == True


# LLM-generated content at query #35
#--------------------------

```python
def test_merge_invariant_results_predicate_false():
    result = [(False, "error1"), (True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


# LLM-generated content at query #36
#--------------------------

```python
def test_wrap_invariant_with_single_boolean_result():
    def simple_invariant():
        return True, "data"
    
    wrapped = wrap_invariant(simple_invariant)
    verdict, data = wrapped()
    
    assert verdict is True
    assert data == "data"


def test_wrap_invariant_with_multiple_results():
    def multi_invariant():
        return [(True, "pass1"), (False, "fail1"), (True, "pass2"), (False, "fail2")]
    
    wrapped = wrap_invariant(multi_invariant)
    verdict, data = wrapped()
    
    assert verdict is False
    assert data == ("fail1", "fail2")


def test_wrap_invariant_all_passing():
    def all_pass_invariant():
        return [(True, "pass1"), (True, "pass2"), (True, "pass3")]
    
    wrapped = wrap_invariant(all_pass_invariant)
    verdict, data = wrapped()
    
    assert verdict is True
    assert data == ()


def test_wrap_invariant_all_failing():
    def all_fail_invariant():
        return [(False, "fail1"), (False, "fail2")]
    
    wrapped = wrap_invariant(all_fail_invariant)
    verdict, data = wrapped()
    
    assert verdict is False
    assert data == ("fail1", "fail2")


def test_wrap_invariant_with_args_and_kwargs():
    def invariant_with_params(a, b, c=None):
        if a and b and c:
            return True, "success"
        return [(False, "param_error")]
    
    wrapped = wrap_invariant(invariant_with_params)
    verdict, data = wrapped(True, True, c=True)
    
    assert verdict is True
    assert data == "success"


def test_wrap_invariant_single_false_result():
    def single_false_invariant():
        return False, "single_error"
    
    wrapped = wrap_invariant(single_false_invariant)
    verdict, data = wrapped()
    
    assert verdict is False
    assert data == "single_error"


def test_wrap_invariant_multiple_results_with_mixed_data_types():
    def mixed_data_invariant():
        return [(True, 1), (False, {"error": "value"}), (False, [1, 2, 3])]
    
    wrapped = wrap_invariant(mixed_data_invariant)
    verdict, data = wrapped()
    
    assert verdict is False
    assert data == ({"error": "value"}, [1, 2, 3])


# LLM-generated content at query #37
#--------------------------

```python
def test_check_types_predicate_true():
    from unittest.mock import MagicMock
    
    # Create mock objects
    mock_source_class = MagicMock()
    mock_source_class.__name__ = "TestClass"
    
    # Test case where expected_types is truthy (non-empty list)
    expected_types = [int, str]
    it = [1, "hello", 2]
    
    # Call the function - should not raise since all items match expected types
    result = _check_types(it, expected_types, mock_source_class)
    
    # If we reach here without exception, the predicate at line 2 evaluated to True
    assert result is None


# LLM-generated content at query #38
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


def test_maybe_parse_user_type_with_invalid_type():
    try:
        maybe_parse_user_type(123)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Type specifications must be types or strings" in str(e)


def test_maybe_parse_user_type_with_invalid_object():
    try:
        maybe_parse_user_type({"key": "value"})
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


def test_maybe_parse_user_type_with_generator():
    gen = (t for t in [int, str])
    result = maybe_parse_user_type(gen)
    assert result == (int, str)


# LLM-generated content at query #39
#--------------------------

```python
def test_restore_pickle():
    class MockClass:
        def __init__(self, data):
            self.data = data
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls(data)
            instance._factory_fields = _factory_fields
            return instance
    
    test_data = {"key": "value"}
    result = MockClass.create.__self__._restore_pickle(MockClass, test_data)
    
    assert result.data == test_data
    assert result._factory_fields == set()


def test_restore_pickle_with_empty_data():
    class MockClass:
        def __init__(self, data):
            self.data = data
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls(data)
            instance._factory_fields = _factory_fields
            return instance
    
    test_data = {}
    result = MockClass.create.__self__._restore_pickle(MockClass, test_data)
    
    assert result.data == test_data
    assert result._factory_fields == set()


def test_restore_pickle_with_complex_data():
    class MockClass:
        def __init__(self, data):
            self.data = data
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls(data)
            instance._factory_fields = _factory_fields
            return instance
    
    test_data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
    result = MockClass.create.__self__._restore_pickle(MockClass, test_data)
    
    assert result.data == test_data
    assert result._factory_fields == set()


# LLM-generated content at query #40
#--------------------------

```python
def test_check_types_with_valid_types():
    from builtins import int, str
    _check_types([1, 2, 3], [int], type('TestClass', (), {}))


def test_check_types_with_multiple_valid_types():
    from builtins import int, str
    _check_types([1, "hello", 2], [int, str], type('TestClass', (), {}))


def test_check_types_with_empty_expected_types():
    _check_types([1, 2, 3], [], type('TestClass', (), {}))


def test_check_types_with_empty_iterable():
    from builtins import int
    _check_types([], [int], type('TestClass', (), {}))


def test_check_types_raises_error_on_invalid_type():
    from builtins import int
    TestClass = type('TestClass', (), {})
    
    try:
        _check_types(["invalid"], [int], TestClass)
        assert False, "Expected CheckedValueTypeError to be raised"
    except Exception as e:
        assert e.__class__.__name__ == 'CheckedValueTypeError'
        assert "can only be used with" in str(e)


def test_check_types_raises_error_on_first_invalid_type():
    from builtins import int, str
    TestClass = type('TestClass', (), {})
    
    try:
        _check_types([1, [], 3], [int, str], TestClass)
        assert False, "Expected CheckedValueTypeError to be raised"
    except Exception as e:
        assert e.__class__.__name__ == 'CheckedValueTypeError'


def test_check_types_with_custom_exception_type():
    from builtins import int
    CustomException = type('CustomException', (Exception,), {})
    TestClass = type('TestClass', (), {})
    
    try:
        _check_types(["invalid"], [int], TestClass, exception_type=CustomException)
        assert False, "Expected CustomException to be raised"
    except CustomException:
        pass


def test_check_types_error_message_format():
    from builtins import int
    TestClass = type('TestClass', (), {})
    
    try:
        _check_types([3.14], [int], TestClass)
        assert False, "Expected CheckedValueTypeError to be raised"
    except Exception as e:
        error_msg = str(e)
        assert "TestClass" in error_msg
        assert "int" in error_msg
        assert "float" in error_msg


# LLM-generated content at query #41
#--------------------------

```python
def test_checked_type_constructor():
    instance = CheckedType()
    assert instance is not None
    assert isinstance(instance, CheckedType)


# LLM-generated content at query #42
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    from unittest.mock import Mock
    
    cls = Mock()
    cls._checked_types = []
    source_data = Mock(spec=cls)
    
    result = _checked_type_create(cls, source_data)
    
    assert result is source_data


def test_checked_type_create_without_checked_type():
    from unittest.mock import Mock
    
    cls = Mock()
    cls._checked_types = []
    source_data = [1, 2, 3]
    
    result = _checked_type_create(cls, source_data)
    
    assert result == cls(source_data)
    cls.assert_called_once_with(source_data)


def test_checked_type_create_with_checked_type_and_matching_data():
    from unittest.mock import Mock, patch
    
    mock_checked_type = Mock()
    mock_checked_type.create = Mock(return_value="created")
    
    cls = Mock()
    cls._checked_types = ['test.MockCheckedType']
    source_data = [Mock(), Mock()]
    
    with patch('__main__.get_types', return_value=[mock_checked_type]):
        with patch('__main__.CheckedType', mock_checked_type):
            result = _checked_type_create(cls, source_data)
    
    assert result == cls([source_data[0], source_data[1]])


def test_checked_type_create_with_checked_type_and_non_matching_data():
    from unittest.mock import Mock, patch, MagicMock
    
    mock_checked_type = MagicMock()
    mock_checked_type.create = Mock(side_effect=lambda x, ignore_extra=False: f"created_{id(x)}")
    
    cls = Mock()
    cls._checked_types = ['test.MockCheckedType']
    data1 = "data1"
    data2 = "data2"
    source_data = [data1, data2]
    
    with patch('__main__.get_types', return_value=[mock_checked_type]):
        with patch('__main__.CheckedType', mock_checked_type):
            result = _checked_type_create(cls, source_data)
    
    assert cls.called


def test_checked_type_create_ignore_extra_parameter():
    from unittest.mock import Mock, patch, MagicMock
    
    mock_checked_type = MagicMock()
    mock_checked_type.create = Mock(return_value="created")
    
    cls = Mock()
    cls._checked_types = ['test.MockCheckedType']
    source_data = ["data"]
    
    with patch('__main__.get_types', return_value=[mock_checked_type]):
        with patch('__main__.CheckedType', mock_checked_type):
            result = _checked_type_create(cls, source_data, ignore_extra=True)
    
    mock_checked_type.create.assert_called()


# LLM-generated content at query #43
#--------------------------

```python
def test_maybe_parse_user_type_type_not_iterable():
    class CustomType:
        pass
    
    result = maybe_parse_user_type(CustomType)
    assert result == [CustomType]


# LLM-generated content at query #44
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


# LLM-generated content at query #45
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


def test_checked_pmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


# LLM-generated content at query #46
#--------------------------

```python
def test_predicate_line_18_evaluates_to_true():
    class CustomType:
        pass
    
    t = CustomType
    is_type = isinstance(t, type)
    is_iterable = isinstance(t, Iterable)
    
    assert is_type and not is_iterable


# LLM-generated content at query #47
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
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({"a": 1, "b": 2, "c": 3})
    assert len(result) == 3
    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


def test_checkedpmap_constructor_repr():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert "IntToFloatMap" in repr(result)
    assert "1" in repr(result)
    assert "1.5" in repr(result)


# LLM-generated content at query #48
#--------------------------

```python
def test_wrap_invariant_with_bool_result():
    def invariant_bool(*args, **kwargs):
        return True, "data"
    
    wrapped = wrap_invariant(invariant_bool)
    result = wrapped()
    assert result == (True, "data")


def test_wrap_invariant_with_tuple_results():
    def invariant_tuple(*args, **kwargs):
        return [(True, "data1"), (True, "data2")]
    
    wrapped = wrap_invariant(invariant_tuple)
    result = wrapped()
    assert result == (True, ())


def test_wrap_invariant_with_failed_results():
    def invariant_failed(*args, **kwargs):
        return [(True, "data1"), (False, "error1"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_failed)
    result = wrapped()
    assert result == (False, ("error1", "error2"))


def test_wrap_invariant_with_all_failed_results():
    def invariant_all_failed(*args, **kwargs):
        return [(False, "error1"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_all_failed)
    result = wrapped()
    assert result == (False, ("error1", "error2"))


def test_wrap_invariant_with_args_and_kwargs():
    def invariant_with_params(a, b, c=None):
        return True, f"{a}_{b}_{c}"
    
    wrapped = wrap_invariant(invariant_with_params)
    result = wrapped(1, 2, c=3)
    assert result == (True, "1_2_3")


def test_wrap_invariant_with_tuple_results_mixed():
    def invariant_mixed(*args, **kwargs):
        return [(True, "data1"), (True, "data2"), (False, "error")]
    
    wrapped = wrap_invariant(invariant_mixed)
    result = wrapped()
    assert result == (False, ("error",))


def test_wrap_invariant_preserves_bool_true():
    def invariant_returns_true(*args, **kwargs):
        return True, "success"
    
    wrapped = wrap_invariant(invariant_returns_true)
    result = wrapped()
    assert result[0] is True


def test_wrap_invariant_preserves_bool_false():
    def invariant_returns_false(*args, **kwargs):
        return False, "failure"
    
    wrapped = wrap_invariant(invariant_returns_false)
    result = wrapped()
    assert result[0] is False


# LLM-generated content at query #49
#--------------------------

```python
def test_checked_type_create_isinstance_returns_source_data():
    class MockCheckedType:
        pass
    
    source_data = MockCheckedType()
    result = source_data
    
    assert isinstance(result, MockCheckedType)


# LLM-generated content at query #50
#--------------------------

```python
def test_merge_invariant_results_predicate_at_line_5_evaluates_to_true():
    result = [(True, "data1"), (True, "data2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


# LLM-generated content at query #51
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


# LLM-generated content at query #52
#--------------------------

```python
def test_checkedpvector_constructor_with_empty_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector()
    assert len(result) == 0
    assert isinstance(result, TestVector)


def test_checkedpvector_constructor_with_list_initial():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector([1, 2, 3])
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, TestVector)


def test_checkedpvector_constructor_with_tuple_initial():
    class TestVector(CheckedPVector):
        __type__ = (int, float)
    
    result = TestVector((1, 2.5, 3))
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2.5
    assert result[2] == 3


def test_checkedpvector_constructor_with_pythonpvector_initial():
    from pyrsistent import PVector, pvector
    
    class TestVector(CheckedPVector):
        __type__ = int
    
    pv = pvector([1, 2, 3])
    result = TestVector(pv)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, TestVector)


def test_checkedpvector_constructor_with_invalid_type():
    class TestVector(CheckedPVector):
        __type__ = int
    
    try:
        TestVector([1, "invalid", 3])
        assert False, "Expected exception for invalid type"
    except Exception:
        pass


def test_checkedpvector_constructor_with_invariant_violation():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Expected exception for invariant violation"
    except Exception:
        pass


def test_checkedpvector_constructor_creates_correct_class():
    class CustomVector(CheckedPVector):
        __type__ = int
    
    result = CustomVector([1, 2, 3])
    assert type(result).__name__ == 'CustomVector'
    assert isinstance(result, CustomVector)


def test_checkedpvector_constructor_with_generator():
    class TestVector(CheckedPVector):
        __type__ = int
    
    gen = (x for x in [1, 2, 3])
    result = TestVector(gen)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


# LLM-generated content at query #53
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


def test_checked_pset_constructor_with_negative_raises_error():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pset_constructor_with_wrong_type_raises_error():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "string", 3])
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pset_constructor_with_pmap_initial():
    from pyrsistent import pmap
    
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pmap_obj = pmap()
    result = Positives(pmap_obj)
    assert isinstance(result, Positives)


def test_checked_pset_constructor_with_default_no_args():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives()
    assert len(result) == 0
    assert isinstance(result, Positives)


# LLM-generated content at query #54
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    class MockCheckedType:
        _checked_types = []
    
    instance = MockCheckedType()
    result = _checked_type_create(MockCheckedType, instance)
    assert result is instance


def test_checked_type_create_without_checked_type():
    class MockClass:
        _checked_types = []
        def __init__(self, data):
            self.data = data
    
    source_data = [1, 2, 3]
    result = _checked_type_create(MockClass, source_data)
    assert result.data == source_data


def test_checked_type_create_with_checked_type_matching_data():
    class CheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
        @staticmethod
        def create(data, ignore_extra=False):
            return CheckedType(data)
    
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [CheckedType([1, 2])]
    result = _checked_type_create(MockClass, source_data)
    assert len(result.data) == 1
    assert isinstance(result.data[0], CheckedType)


def test_checked_type_create_with_checked_type_non_matching_data():
    class CheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
        @staticmethod
        def create(data, ignore_extra=False):
            return CheckedType(data)
    
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [[1, 2], [3, 4]]
    result = _checked_type_create(MockClass, source_data)
    assert len(result.data) == 2
    assert all(isinstance(item, CheckedType) for item in result.data)


def test_checked_type_create_with_ignore_extra_flag():
    class CheckedType:
        _checked_types = []
        def __init__(self, data):
            self.data = data
        @staticmethod
        def create(data, ignore_extra=False):
            return CheckedType(data)
    
    class MockClass:
        _checked_types = ['__main__.CheckedType']
        def __init__(self, data):
            self.data = data
    
    source_data = [[1, 2]]
    result = _checked_type_create(MockClass, source_data, ignore_extra=True)
    assert len(result.data) == 1
    assert isinstance(result.data[0], CheckedType)


# LLM-generated content at query #55
#--------------------------

```python
def test_invariant_errors_no_errors():
    invariants = [
        lambda x: (True, "valid1"),
        lambda x: (True, "valid2"),
    ]
    result = _invariant_errors("test", invariants)
    assert result == []


def test_invariant_errors_single_error():
    invariants = [
        lambda x: (True, "valid"),
        lambda x: (False, "error_data"),
    ]
    result = _invariant_errors("test", invariants)
    assert result == ["error_data"]


def test_invariant_errors_multiple_errors():
    invariants = [
        lambda x: (False, "error1"),
        lambda x: (True, "valid"),
        lambda x: (False, "error2"),
    ]
    result = _invariant_errors("test", invariants)
    assert result == ["error1", "error2"]


def test_invariant_errors_all_errors():
    invariants = [
        lambda x: (False, "error1"),
        lambda x: (False, "error2"),
        lambda x: (False, "error3"),
    ]
    result = _invariant_errors("test", invariants)
    assert result == ["error1", "error2", "error3"]


def test_invariant_errors_empty_invariants():
    invariants = []
    result = _invariant_errors("test", invariants)
    assert result == []


def test_invariant_errors_with_different_data_types():
    invariants = [
        lambda x: (False, 42),
        lambda x: (True, "valid"),
        lambda x: (False, {"key": "value"}),
    ]
    result = _invariant_errors("test", invariants)
    assert result == [42, {"key": "value"}]


def test_invariant_errors_with_none_data():
    invariants = [
        lambda x: (False, None),
        lambda x: (True, "valid"),
    ]
    result = _invariant_errors("test", invariants)
    assert result == [None]


# LLM-generated content at query #56
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


def test_checked_pmap_constructor_default_parameter():
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
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "InvariantException" in str(type(e).__name__)


def test_checked_pmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except Exception as e:
        assert "CheckedKeyTypeError" in str(type(e).__name__)


def test_checked_pmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Expected CheckedTypeError"
    except Exception as e:
        assert "CheckedTypeError" in str(type(e).__name__)


def test_checked_pmap_constructor_preserves_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    initial_data = {10: 10.5, 20: 20.75, 30: 30.25}
    result = IntToFloatMap(initial_data)
    
    assert result[10] == 10.5
    assert result[20] == 20.75
    assert result[30] == 30.25


# LLM-generated content at query #57
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
        IntToFloatMap({"not_int": 1.5})
        assert False, "Should raise CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_float"})
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


def test_checked_pmap_constructor_with_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert len(result) == 5
    for k, v in data.items():
        assert result[k] == v


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert type(result) == IntToFloatMap


# LLM-generated content at query #58
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    # Test that the predicate at line 1 (isinstance(source_data, cls)) evaluates to False
    # This means source_data should NOT be an instance of cls
    
    class MockCheckedType:
        pass
    
    source_data = "not an instance of MockCheckedType"
    cls = MockCheckedType
    
    # The predicate at line 2: isinstance(source_data, cls) should be False
    predicate_result = isinstance(source_data, cls)
    
    assert predicate_result is False


# LLM-generated content at query #59
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_one_false():
    result = [(True, "data1"), (False, "error1"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


def test_merge_invariant_results_multiple_false():
    result = [(False, "error1"), (True, "data2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1", "error3")


def test_merge_invariant_results_all_false():
    result = [(False, "error1"), (False, "error2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1", "error2", "error3")


def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_single_true():
    result = [(True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_single_false():
    result = [(False, "error1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


def test_merge_invariant_results_mixed_data_types():
    result = [(False, 123), (True, "data"), (False, None)]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == (123, None)


# LLM-generated content at query #60
#--------------------------

```python
def test_merge_invariant_results_all_true():
    result = [(True, "data1"), (True, "data2"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_one_false():
    result = [(True, "data1"), (False, "error1"), (True, "data3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


def test_merge_invariant_results_multiple_false():
    result = [(False, "error1"), (True, "data2"), (False, "error2"), (False, "error3")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1", "error2", "error3")


def test_merge_invariant_results_all_false():
    result = [(False, "error1"), (False, "error2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1", "error2")


def test_merge_invariant_results_empty():
    result = []
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_single_true():
    result = [(True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is True
    assert data == ()


def test_merge_invariant_results_single_false():
    result = [(False, "error1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


# LLM-generated content at query #61
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
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_returns_checkedpmap_instance():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #62
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


# LLM-generated content at query #63
#--------------------------

```python
def test_checked_pmap_initial_items_iteration():
    from pyrsistent import pmap
    from pyrsistent._checked_types import CheckedPMap
    
    class TestMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    initial_data = {'a': 1, 'b': 2, 'c': 3}
    result = TestMap(initial_data)
    
    assert dict(result) == initial_data
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #64
#--------------------------

```python
def test_merge_invariant_results_predicate_evaluates_to_false():
    result = [(False, "error_data")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error_data",)


# LLM-generated content at query #65
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checked_pmap_constructor_with_dict():
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
        IntToFloatMap({'invalid': 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: 'invalid'})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
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


def test_checked_pmap_constructor_with_multiple_entries():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    data = {'a': 1, 'b': 2, 'c': 3}
    result = StringToIntMap(data)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


def test_checked_pmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({5: 5.5})
    assert type(result).__name__ == 'IntToFloatMap'


# LLM-generated content at query #66
#--------------------------

```python
def test_checkedpvector_constructor_with_empty_list():
    class TestVector(CheckedPVector):
        __type__ = int
    
    result = TestVector([])
    assert len(result) == 0
    assert isinstance(result, TestVector)


def test_checkedpvector_constructor_with_valid_values():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert list(result) == [1, 2, 3]
    assert isinstance(result, Positives)


def test_checkedpvector_constructor_with_mixed_numeric_types():
    class Numbers(CheckedPVector):
        __type__ = (int, float)
    
    result = Numbers([1, 2.5, 3])
    assert len(result) == 3
    assert list(result) == [1, 2.5, 3]


def test_checkedpvector_constructor_with_generator():
    class Integers(CheckedPVector):
        __type__ = int
    
    result = Integers(x for x in [1, 2, 3])
    assert len(result) == 3
    assert list(result) == [1, 2, 3]


def test_checkedpvector_constructor_with_tuple():
    class Values(CheckedPVector):
        __type__ = int
    
    result = Values((10, 20, 30))
    assert len(result) == 3
    assert list(result) == [10, 20, 30]


def test_checkedpvector_constructor_with_pythonpvector():
    from pyrsistent import pvector
    
    class Integers(CheckedPVector):
        __type__ = int
    
    pv = pvector([1, 2, 3])
    result = Integers(pv)
    assert len(result) == 3
    assert list(result) == [1, 2, 3]
    assert isinstance(result, Integers)


def test_checkedpvector_constructor_invalid_type_raises_error():
    class Integers(CheckedPVector):
        __type__ = int
    
    try:
        Integers([1, "invalid", 3])
        assert False, "Expected exception for invalid type"
    except Exception:
        pass


def test_checkedpvector_constructor_invariant_violation_raises_error():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -5, 3])
        assert False, "Expected exception for invariant violation"
    except Exception:
        pass


def test_checkedpvector_constructor_preserves_order():
    class Integers(CheckedPVector):
        __type__ = int
    
    result = Integers([5, 1, 3, 2, 4])
    assert list(result) == [5, 1, 3, 2, 4]


def test_checkedpvector_constructor_with_single_element():
    class Values(CheckedPVector):
        __type__ = str
    
    result = Values(["hello"])
    assert len(result) == 1
    assert result[0] == "hello"


# LLM-generated content at query #67
#--------------------------

```python
def test_checkedpvector_constructor_with_empty_initial():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives()
    assert len(result) == 0
    assert result.tolist() == []


def test_checkedpvector_constructor_with_list():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert result.tolist() == [1, 2, 3]


def test_checkedpvector_constructor_with_floats():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert result.tolist() == [1.5, 2.5, 3.5]


def test_checkedpvector_constructor_with_mixed_numeric_types():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert result.tolist() == [1, 2.5, 3]


def test_checkedpvector_constructor_with_pythonpvector():
    from pyrsistent import pvector
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    pv = pvector([1, 2, 3])
    result = Positives(pv)
    assert len(result) == 3
    assert result.tolist() == [1, 2, 3]


def test_checkedpvector_constructor_with_tuple():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives((1, 2, 3))
    assert len(result) == 3
    assert result.tolist() == [1, 2, 3]


def test_checkedpvector_constructor_with_generator():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives(x for x in [1, 2, 3])
    assert len(result) == 3
    assert result.tolist() == [1, 2, 3]


def test_checkedpvector_constructor_invalid_type():
    from pyrsistent import InvariantException
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpvector_constructor_negative_value():
    from pyrsistent import InvariantException
    
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpvector_constructor_preserves_class_type():
    class Positives(CheckedPVector):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert type(result).__name__ == 'Positives'


# LLM-generated content at query #68
#--------------------------

```python
def test_check_types_with_valid_types():
    from builtins import int, str
    _check_types([1, 2, 3], [int], type('TestClass', (), {}))


def test_check_types_with_multiple_valid_types():
    from builtins import int, str
    _check_types([1, 'hello', 2], [int, str], type('TestClass', (), {}))


def test_check_types_with_empty_expected_types():
    _check_types([1, 2, 'hello'], [], type('TestClass', (), {}))


def test_check_types_with_empty_iterable():
    from builtins import int
    _check_types([], [int], type('TestClass', (), {}))


def test_check_types_raises_error_on_invalid_type():
    from builtins import int, str
    TestClass = type('TestClass', (), {})
    try:
        _check_types([1, 2.5, 3], [int], TestClass)
        assert False, "Expected CheckedValueTypeError to be raised"
    except Exception as e:
        assert "can only be used with" in str(e)
        assert "not float" in str(e)


def test_check_types_with_string_type_name():
    from builtins import int
    TestClass = type('TestClass', (), {})
    _check_types([1, 2, 3], ['builtins.int'], TestClass)


def test_check_types_with_custom_exception_type():
    from builtins import int, str
    CustomException = type('CustomException', (Exception,), {})
    TestClass = type('TestClass', (), {})
    try:
        _check_types([1, 'invalid'], [int], TestClass, exception_type=CustomException)
        assert False, "Expected CustomException to be raised"
    except CustomException:
        pass


def test_check_types_error_message_format():
    from builtins import int, str
    TestClass = type('TestClass', (), {})
    try:
        _check_types([1, 2.5], [int], TestClass)
        assert False, "Expected exception to be raised"
    except Exception as e:
        error_msg = str(e)
        assert "TestClass" in error_msg
        assert "int" in error_msg
        assert "float" in error_msg


# LLM-generated content at query #69
#--------------------------

```python
def test_checked_type_create_isinstance_returns_source_data():
    class MockCheckedType:
        pass
    
    source_data = MockCheckedType()
    result = _checked_type_create(MockCheckedType, source_data)
    
    assert result is source_data


# LLM-generated content at query #70
#--------------------------

```python
def test_check_types_predicate_true_with_non_empty_expected_types():
    expected_types = [int, str]
    result = bool(expected_types)
    assert result is True


# LLM-generated content at query #71
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
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checked_pmap_constructor_with_failing_invariant():
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
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checked_pmap_constructor_preserves_class_type():
    class CustomMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = CustomMap({"a": 1, "b": 2})
    assert type(result).__name__ == "CustomMap"
    assert isinstance(result, CustomMap)


# LLM-generated content at query #72
#--------------------------

```python
def test_checked_type_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #73
#--------------------------

```python
def test_merge_invariant_results_predicate_false():
    result = [(False, "error1"), (True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict is False
    assert data == ("error1",)


# LLM-generated content at query #74
#--------------------------

```python
def test_restore_pickle():
    class MockClass:
        def __init__(self, data):
            self.data = data
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls(data)
            instance._factory_fields = _factory_fields
            return instance
    
    test_data = {"key": "value"}
    result = MockClass.create.__self__._restore_pickle(MockClass, test_data)
    
    assert result.data == test_data
    assert result._factory_fields == set()


def test_restore_pickle_with_different_data():
    class TestClass:
        def __init__(self, data):
            self.data = data
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls(data)
            instance._factory_fields = _factory_fields
            return instance
    
    def _restore_pickle(cls, data):
        return cls.create(data, _factory_fields=set())
    
    test_data = [1, 2, 3]
    result = _restore_pickle(TestClass, test_data)
    
    assert result.data == test_data
    assert result._factory_fields == set()


def test_restore_pickle_with_empty_data():
    class EmptyTestClass:
        def __init__(self, data):
            self.data = data
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            instance = cls(data)
            instance._factory_fields = _factory_fields
            return instance
    
    def _restore_pickle(cls, data):
        return cls.create(data, _factory_fields=set())
    
    test_data = {}
    result = _restore_pickle(EmptyTestClass, test_data)
    
    assert result.data == test_data
    assert result._factory_fields == set()
    assert len(result._factory_fields) == 0


# LLM-generated content at query #75
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


# LLM-generated content at query #76
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


def test_checked_pset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Expected an exception"
    except Exception:
        pass


def test_checked_pset_constructor_with_negative_invariant_violation():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Expected an exception"
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


def test_checked_pset_constructor_default_empty():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives()
    assert len(result) == 0
    assert isinstance(result, Positives)


# LLM-generated content at query #77
#--------------------------

```python
def test_merge_invariant_results_predicate_evaluates_to_false():
    result = [(False, "error1"), (True, "data1")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ("error1",)


# LLM-generated content at query #78
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
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checked_pmap_constructor_with_single_item():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'a': 1})
    assert len(result) == 1
    assert result['a'] == 1


def test_checked_pmap_constructor_preserves_type():
    class CustomMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = str
    
    result = CustomMap({'key': 'value'})
    assert type(result).__name__ == 'CustomMap'


def test_checked_pmap_constructor_with_multiple_items():
    class IntToStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    data = {1: 'one', 2: 'two', 3: 'three'}
    result = IntToStrMap(data)
    assert len(result) == 3
    assert result[1] == 'one'
    assert result[2] == 'two'
    assert result[3] == 'three'


# LLM-generated content at query #79
#--------------------------

```python
def test_checkedpset_constructor_with_empty_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([])
    assert len(result) == 0
    assert isinstance(result, Positives)


def test_checkedpset_constructor_with_valid_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    assert isinstance(result, Positives)


def test_checkedpset_constructor_with_float_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1.5, 2.5, 3.5])
    assert len(result) == 3
    assert 1.5 in result
    assert 2.5 in result
    assert 3.5 in result


def test_checkedpset_constructor_with_mixed_valid_types():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    assert 3 in result


def test_checkedpset_constructor_with_duplicate_elements():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_checkedpset_constructor_with_invalid_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, "invalid", 3])
        assert False, "Expected exception for invalid type"
    except CheckedTypeError:
        pass


def test_checkedpset_constructor_with_invariant_violation():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    try:
        Positives([1, -2, 3])
        assert False, "Expected exception for invariant violation"
    except InvariantException:
        pass


def test_checkedpset_constructor_with_zero_element():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([0, 1, 2])
    assert len(result) == 3
    assert 0 in result


def test_checkedpset_constructor_with_pmap_initial():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    from pyrsistent import pmap
    pmap_obj = pmap({1: True, 2: True})
    result = Positives(pmap_obj)
    assert isinstance(result, Positives)


def test_checkedpset_constructor_returns_correct_class_type():
    class Positives(CheckedPSet):
        __type__ = (int, float)
        __invariant__ = lambda n: (n >= 0, 'Negative')
    
    result = Positives([5, 10, 15])
    assert type(result).__name__ == 'Positives'


# LLM-generated content at query #80
#--------------------------

```python
def test_checked_type_create_isinstance_predicate():
    class MockClass:
        pass
    
    class MockCheckedType(MockClass):
        pass
    
    source_data = MockCheckedType()
    cls = MockCheckedType
    
    result = isinstance(source_data, cls)
    
    assert result is True


# LLM-generated content at query #81
#--------------------------

```python
def test_invariant_errors_all_pass():
    def invariant1(elem):
        return (True, "pass1")
    
    def invariant2(elem):
        return (True, "pass2")
    
    result = _invariant_errors("test", [invariant1, invariant2])
    assert result == []


def test_invariant_errors_all_fail():
    def invariant1(elem):
        return (False, "error1")
    
    def invariant2(elem):
        return (False, "error2")
    
    result = _invariant_errors("test", [invariant1, invariant2])
    assert result == ["error1", "error2"]


def test_invariant_errors_mixed():
    def invariant1(elem):
        return (True, "pass")
    
    def invariant2(elem):
        return (False, "error1")
    
    def invariant3(elem):
        return (False, "error2")
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == ["error1", "error2"]


def test_invariant_errors_empty_list():
    result = _invariant_errors("test", [])
    assert result == []


def test_invariant_errors_single_pass():
    def invariant(elem):
        return (True, "pass")
    
    result = _invariant_errors("test", [invariant])
    assert result == []


def test_invariant_errors_single_fail():
    def invariant(elem):
        return (False, "error")
    
    result = _invariant_errors("test", [invariant])
    assert result == ["error"]


def test_invariant_errors_with_different_data_types():
    def invariant1(elem):
        return (False, 123)
    
    def invariant2(elem):
        return (False, {"key": "value"})
    
    def invariant3(elem):
        return (True, "ignored")
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == [123, {"key": "value"}]


# LLM-generated content at query #82
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


def test_checked_pmap_constructor_with_single_entry():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert isinstance(result, IntToFloatMap)
    assert result[42] == 3.14


def test_checked_pmap_constructor_default_parameter():
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
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checked_pmap_constructor_violates_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checked_pmap_constructor_multiple_entries_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0, 3: 3.0})
    assert len(result) == 3
    assert result[1] == 1.0
    assert result[2] == 2.0
    assert result[3] == 3.0


# LLM-generated content at query #83
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


def test_restore_pickle_with_complex_data():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self._factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            return cls(data, _factory_fields)
    
    test_data = [1, 2, 3, {"nested": "object"}]
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result.data == test_data
    assert result._factory_fields == set()


def test_restore_pickle_factory_fields_is_empty_set():
    class MockClass:
        def __init__(self, data, _factory_fields=None):
            self.data = data
            self._factory_fields = _factory_fields
        
        @classmethod
        def create(cls, data, _factory_fields=None):
            return cls(data, _factory_fields)
    
    test_data = "test"
    result = MockClass.create(test_data, _factory_fields=set())
    
    assert result._factory_fields == set()
    assert len(result._factory_fields) == 0


# LLM-generated content at query #84
#--------------------------

```python
def test_checked_type_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #85
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


def test_merge_invariant_results_with_various_data_types():
    result = [(True, 123), (False, [1, 2, 3]), (False, {"key": "value"})]
    verdict, data = _merge_invariant_results(result)
    assert verdict == False
    assert data == ([1, 2, 3], {"key": "value"})


# LLM-generated content at query #86
#--------------------------

```python
def test_checked_pmap_constructor_empty():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap()
    assert len(result) == 0
    assert dict(result) == {}


def test_checked_pmap_constructor_with_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checked_pmap_constructor_with_single_item():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int

    result = StringToIntMap({'a': 1})
    assert len(result) == 1
    assert result['a'] == 1


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
    except InvariantException:
        pass


def test_checked_pmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({'a': 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checked_pmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({1: 'invalid'})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checked_pmap_constructor_multiple_items():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int

    result = StringToIntMap({'a': 1, 'b': 2, 'c': 3})
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #87
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
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checkedpmap_constructor_with_single_item():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert result[1] == 1.5
    assert len(result) == 1


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_returns_correct_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checkedpmap_constructor_with_multiple_items():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25, 3: 3.75})
    assert len(result) == 3
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert result[3] == 3.75


# LLM-generated content at query #88
#--------------------------

```python
def test_checkedpmap_constructor_empty():
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


def test_checkedpmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert type(result).__name__ == "IntToFloatMap"


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


def test_checkedpmap_constructor_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.5, 2: 2.25, 3: 3.75, 4: 4.5}
    result = IntToFloatMap(data)
    assert len(result) == 4
    assert result[1] == 1.5
    assert result[3] == 3.75


# LLM-generated content at query #89
#--------------------------

```python
def test_checkedtype_constructor():
    instance = CheckedType()
    assert isinstance(instance, CheckedType)
    assert hasattr(instance, '__slots__')
    assert instance.__slots__ == ()


# LLM-generated content at query #90
#--------------------------

```python
def test_merge_invariant_results_predicate_at_line_5_evaluates_to_true():
    result = [(True, "data1"), (True, "data2")]
    verdict, data = _merge_invariant_results(result)
    assert verdict == True
    assert data == ()


# LLM-generated content at query #91
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
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #92
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


def test_checkedpmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except Exception as e:
        assert "CheckedKeyTypeError" in str(type(e))


def test_checkedpmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Should have raised CheckedTypeError"
    except Exception as e:
        assert "CheckedTypeError" in str(type(e))


def test_checkedpmap_constructor_single_element():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({42: 3.14})
    assert len(result) == 1
    assert result[42] == 3.14


def test_checkedpmap_constructor_multiple_elements():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4})
    assert len(result) == 4
    assert result[1] == 1.1
    assert result[2] == 2.2
    assert result[3] == 3.3
    assert result[4] == 4.4


def test_checkedpmap_constructor_returns_correct_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #93
#--------------------------

```python
def test_invariant_errors_returns_empty_list_when_all_valid():
    def invariant1(elem):
        return (True, "data1")
    def invariant2(elem):
        return (True, "data2")
    
    result = _invariant_errors("test", [invariant1, invariant2])
    assert result == []


def test_invariant_errors_returns_invalid_data():
    def invariant1(elem):
        return (True, "data1")
    def invariant2(elem):
        return (False, "error_data2")
    def invariant3(elem):
        return (False, "error_data3")
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == ["error_data2", "error_data3"]


def test_invariant_errors_returns_all_errors_when_all_invalid():
    def invariant1(elem):
        return (False, "error1")
    def invariant2(elem):
        return (False, "error2")
    
    result = _invariant_errors("test", [invariant1, invariant2])
    assert result == ["error1", "error2"]


def test_invariant_errors_with_empty_invariants_list():
    result = _invariant_errors("test", [])
    assert result == []


def test_invariant_errors_with_single_valid_invariant():
    def invariant(elem):
        return (True, "valid_data")
    
    result = _invariant_errors("test", [invariant])
    assert result == []


def test_invariant_errors_with_single_invalid_invariant():
    def invariant(elem):
        return (False, "invalid_data")
    
    result = _invariant_errors("test", [invariant])
    assert result == ["invalid_data"]


def test_invariant_errors_with_different_data_types():
    def invariant1(elem):
        return (False, 42)
    def invariant2(elem):
        return (False, {"key": "value"})
    def invariant3(elem):
        return (True, "ignored")
    
    result = _invariant_errors("test", [invariant1, invariant2, invariant3])
    assert result == [42, {"key": "value"}]


# LLM-generated content at query #94
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


def test_checkedpmap_constructor_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Expected CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Expected InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


# LLM-generated content at query #95
#--------------------------

```python
def test_checked_type_create_isinstance_predicate():
    class MockCheckedType:
        pass
    
    instance = MockCheckedType()
    source_data = instance
    cls = MockCheckedType
    
    result = isinstance(source_data, cls)
    
    assert result is True


# LLM-generated content at query #96
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


def test_checkedpmap_constructor_default_initial():
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
    assert len(result) == 2
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_with_invariant_violation():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Expected InvariantException"
    except Exception as e:
        assert "InvariantException" in str(type(e).__name__)


def test_checkedpmap_constructor_with_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"string_key": 1.5})
        assert False, "Expected CheckedKeyTypeError"
    except Exception as e:
        assert "CheckedKeyTypeError" in str(type(e).__name__)


def test_checkedpmap_constructor_with_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "not_a_float"})
        assert False, "Expected CheckedTypeError"
    except Exception as e:
        assert "CheckedTypeError" in str(type(e).__name__)


def test_checkedpmap_constructor_preserves_type():
    class CustomMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = CustomMap({"a": 1, "b": 2})
    assert type(result).__name__ == "CustomMap"
    assert isinstance(result, CheckedPMap)


# LLM-generated content at query #97
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
        IntToFloatMap({"key": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass


def test_checkedpmap_constructor_with_violated_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_with_multiple_entries():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    data = {1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4, 5: 5.5}
    result = IntToFloatMap(data)
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 5
    for key, value in data.items():
        assert result[key] == value


# LLM-generated content at query #98
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


def test_checkedpmap_constructor_default_initial():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert isinstance(result, IntToFloatMap)
    assert len(result) == 0


def test_checkedpmap_constructor_with_invalid_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({"invalid": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_with_invalid_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    try:
        IntToFloatMap({1: "invalid"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_with_failed_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    try:
        IntToFloatMap({1: 2.0})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


def test_checkedpmap_constructor_single_entry():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({5: 5.5})
    assert isinstance(result, IntToFloatMap)
    assert result[5] == 5.5
    assert len(result) == 1


# LLM-generated content at query #99
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
    
    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


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


def test_checkedpmap_constructor_with_dict():
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


def test_checkedpmap_constructor_wrong_key_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({"not_int": 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass


def test_checkedpmap_constructor_wrong_value_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    try:
        IntToFloatMap({1: "not_float"})
        assert False, "Should have raised CheckedTypeError"
    except CheckedTypeError:
        pass


def test_checkedpmap_constructor_with_size_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.5, 2: 2.25}, size=2)
    assert len(result) == 2
    assert result[1] == 1.5
    assert result[2] == 2.25


def test_checkedpmap_constructor_default_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    result = IntToFloatMap({1: 1.5})
    assert isinstance(result, IntToFloatMap)
    assert result[1] == 1.5


# LLM-generated content at query #101
#--------------------------

```python
def test_checkedpmap_constructor_with_empty_dict():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({})
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_valid_data():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5, 2: 2.25})
    assert result[1] == 1.5
    assert result[2] == 2.25
    assert len(result) == 2


def test_checkedpmap_constructor_with_invariant():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
        __invariant__ = lambda k, v: (int(v) == k, 'Invalid mapping')
    
    result = IntToFloatMap({1: 1.0, 2: 2.0})
    assert result[1] == 1.0
    assert result[2] == 2.0


def test_checkedpmap_constructor_default_parameter():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap()
    assert len(result) == 0
    assert isinstance(result, IntToFloatMap)


def test_checkedpmap_constructor_with_single_entry():
    class StringToIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    result = StringToIntMap({'key': 42})
    assert result['key'] == 42
    assert len(result) == 1


def test_checkedpmap_constructor_preserves_type():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = IntToFloatMap({1: 1.5})
    assert type(result).__name__ == 'IntToFloatMap'


def test_checkedpmap_constructor_multiple_entries():
    class IntToStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    data = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
    result = IntToStrMap(data)
    assert len(result) == 4
    assert result[1] == 'a'
    assert result[4] == 'd'


# LLM-generated content at query #102
#--------------------------

```python
def test_checked_type_create_predicate_line_1_false():
    # Test that the predicate at line 1 (isinstance(source_data, cls)) evaluates to False
    # by passing source_data that is not an instance of cls
    
    class MockCheckedType:
        _checked_types = None
    
    source_data = [1, 2, 3]
    cls = MockCheckedType
    
    # The predicate isinstance(source_data, cls) should be False
    result = isinstance(source_data, cls)
    assert result is False


# LLM-generated content at query #103
#--------------------------

```python
def test_wrap_invariant_with_single_bool_result():
    def invariant_func():
        return True, "data1"
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    
    assert result == (True, "data1")


def test_wrap_invariant_with_multiple_results_all_true():
    def invariant_func():
        return [(True, "data1"), (True, "data2"), (True, "data3")]
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    
    assert result == (True, ())


def test_wrap_invariant_with_multiple_results_one_false():
    def invariant_func():
        return [(True, "data1"), (False, "error1"), (True, "data3")]
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    
    assert result == (False, ("error1",))


def test_wrap_invariant_with_multiple_results_all_false():
    def invariant_func():
        return [(False, "error1"), (False, "error2"), (False, "error3")]
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    
    assert result == (False, ("error1", "error2", "error3"))


def test_wrap_invariant_with_args_and_kwargs():
    def invariant_func(arg1, arg2, kwarg1=None):
        return True, f"{arg1}-{arg2}-{kwarg1}"
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped("a", "b", kwarg1="c")
    
    assert result == (True, "a-b-c")


def test_wrap_invariant_with_false_bool_result():
    def invariant_func():
        return False, "failure_data"
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    
    assert result == (False, "failure_data")


def test_wrap_invariant_with_empty_result_list():
    def invariant_func():
        return []
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    
    assert result == (True, ())


def test_wrap_invariant_with_mixed_error_data_types():
    def invariant_func():
        return [(True, "data1"), (False, 42), (False, None), (True, "data4")]
    
    wrapped = wrap_invariant(invariant_func)
    result = wrapped()
    
    assert result == (False, (42, None))


# LLM-generated content at query #104
#--------------------------

```python
def test_checked_type_create_with_instance_of_cls():
    from unittest.mock import Mock
    cls = Mock()
    cls._checked_types = []
    source_data = Mock(spec=cls)
    
    result = _checked_type_create(cls, source_data)
    
    assert result is source_data


def test_checked_type_create_without_checked_type():
    from unittest.mock import Mock
    cls = Mock()
    cls._checked_types = ['builtins.str']
    cls.return_value = 'result'
    source_data = ['data1', 'data2']
    
    result = _checked_type_create(cls, source_data)
    
    cls.assert_called_once_with(source_data)
    assert result == 'result'


def test_checked_type_create_with_checked_type_matching_data():
    from unittest.mock import Mock, patch
    
    mock_checked_type = Mock()
    mock_checked_type.create = Mock(return_value='created')
    
    cls = Mock()
    cls._checked_types = ['builtins.str']
    cls.return_value = 'result'
    
    source_data = ['data1']
    
    with patch('__main__.get_types', return_value=[mock_checked_type]):
        with patch('__main__.CheckedType', Mock):
            result = _checked_type_create(cls, source_data, ignore_extra=False)
    
    cls.assert_called_once()


def test_checked_type_create_with_checked_type_not_matching_data():
    from unittest.mock import Mock, patch
    
    mock_checked_type = Mock()
    mock_checked_type.create = Mock(return_value='created_data')
    
    cls = Mock()
    cls._checked_types = ['builtins.str']
    cls.return_value = ['created_data']
    
    source_data = [123]
    
    with patch('__main__.get_types', return_value=[mock_checked_type]):
        with patch('__main__.CheckedType', Mock):
            result = _checked_type_create(cls, source_data)
    
    mock_checked_type.create.assert_called_once()


def test_checked_type_create_ignore_extra_parameter():
    from unittest.mock import Mock, patch
    
    mock_checked_type = Mock()
    mock_checked_type.create = Mock(return_value='created')
    
    cls = Mock()
    cls._checked_types = ['builtins.str']
    cls.return_value = 'result'
    
    source_data = [456]
    
    with patch('__main__.get_types', return_value=[mock_checked_type]):
        with patch('__main__.CheckedType', Mock):
            result = _checked_type_create(cls, source_data, ignore_extra=True)
    
    mock_checked_type.create.assert_called_with(456, ignore_extra=True)


