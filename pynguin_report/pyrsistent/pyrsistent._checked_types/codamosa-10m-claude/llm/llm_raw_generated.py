####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CheckedPSet_serialize():
    """Test serialize method of CheckedPSet"""
    
    # Test basic serialization of CheckedPSet with integers
    class IntSet(CheckedPSet):
        __type__ = (int,)
    
    int_set = IntSet([1, 2, 3])
    serialized = int_set.serialize()
    assert isinstance(serialized, set)
    assert serialized == {1, 2, 3}
    
    # Test serialization with custom serializer
    class CustomSet(CheckedPSet):
        __type__ = (int,)
        
        def __serializer__(self, format, value):
            return value * 2
    
    custom_set = CustomSet([1, 2, 3])
    serialized = custom_set.serialize()
    assert serialized == {2, 4, 6}
    
    # Test serialization with CheckedType elements
    class PositiveInts(CheckedPSet):
        __type__ = (int,)
        __invariant__ = lambda n: (n > 0, 'Must be positive')
    
    pos_set = PositiveInts([1, 2, 3])
    serialized = pos_set.serialize()
    assert serialized == {1, 2, 3}
    
    # Test serialization with nested CheckedType
    class NestedSet(CheckedPSet):
        __type__ = (CheckedPSet,)
    
    inner_set = IntSet([1, 2])
    nested_set = NestedSet([inner_set])
    serialized = nested_set.serialize()
    assert len(serialized) == 1
    
    # Test serialization with format parameter
    class FormattedSet(CheckedPSet):
        __type__ = (str,)
    
    formatted_set = FormattedSet(['a', 'b', 'c'])
    serialized = formatted_set.serialize(format='json')
    assert isinstance(serialized, set)
    assert serialized == {'a', 'b', 'c'}
    
    # Test serialization with empty set
    empty_set = IntSet([])
    serialized = empty_set.serialize()
    assert serialized == set()
    
    # Test serialization with floats
    class FloatSet(CheckedPSet):
        __type__ = (float,)
    
    float_set = FloatSet([1.1, 2.2, 3.3])
    serialized = float_set.serialize()
    assert isinstance(serialized, set)
    assert len(serialized) == 3
    assert 1.1 in serialized
    assert 2.2 in serialized
    assert 3.3 in serialized


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pyrsistent._pmap import pmap


def test_CheckedPSet___new__():
    """Test CheckedPSet.__new__ with various initialization scenarios."""
    
    # Test 1: Create CheckedPSet with empty initial value (default)
    class SimpleSet(CheckedPSet):
        __type__ = int
    
    result = SimpleSet()
    assert isinstance(result, CheckedPSet)
    assert len(result) == 0
    
    # Test 2: Create CheckedPSet with list of values
    result = SimpleSet([1, 2, 3])
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    
    # Test 3: Create CheckedPSet with tuple of values
    result = SimpleSet((4, 5, 6))
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 4 in result
    
    # Test 4: Create CheckedPSet with set of values
    result = SimpleSet({7, 8, 9})
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 7 in result
    
    # Test 5: Create CheckedPSet with PMap initial value
    pmap_val = pmap()
    result = SimpleSet.__new__(SimpleSet, pmap_val)
    assert isinstance(result, CheckedPSet)
    
    # Test 6: Type checking during initialization
    class StrictSet(CheckedPSet):
        __type__ = int
    
    with pytest.raises(CheckedValueTypeError):
        StrictSet([1, 2, "invalid"])
    
    # Test 7: Multiple valid types
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, float)
    
    result = MultiTypeSet([1, 2.5, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2.5 in result
    
    # Test 8: Invariant checking during initialization
    class PositiveSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, 'Must be positive')
    
    with pytest.raises(InvariantException):
        PositiveSet([1, 2, -3])
    
    # Test 9: Valid invariant check
    result = PositiveSet([1, 2, 3])
    assert len(result) == 3
    
    # Test 10: Duplicate values in initial data
    result = SimpleSet([1, 1, 2, 2, 3])
    assert len(result) == 3  # Sets eliminate duplicates
    
    # Test 11: Create from iterator
    result = SimpleSet(x for x in [10, 11, 12])
    assert len(result) == 3
    assert 10 in result
    
    # Test 12: Subclass returns correct type
    class MySet(CheckedPSet):
        __type__ = int
    
    result = MySet([1, 2, 3])
    assert type(result).__name__ == 'MySet'


# LLM-generated content at query #3
#--------------------------

def test_wrap_invariant():
    # Test with a single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "success")
    
    # Test with a single boolean result that fails
    def invariant_single_fail(x):
        return (False, "failure")
    
    wrapped_single_fail = wrap_invariant(invariant_single_fail)
    result = wrapped_single_fail(5)
    assert result == (False, "failure")
    
    # Test with multiple results that all pass
    def invariant_multiple_pass(x):
        return [(True, "check1"), (True, "check2"), (True, "check3")]
    
    wrapped_multiple_pass = wrap_invariant(invariant_multiple_pass)
    result = wrapped_multiple_pass(5)
    assert result == (True, ())
    
    # Test with multiple results where some fail
    def invariant_multiple_fail(x):
        return [(True, "check1"), (False, "error1"), (True, "check3"), (False, "error2")]
    
    wrapped_multiple_fail = wrap_invariant(invariant_multiple_fail)
    result = wrapped_multiple_fail(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test with multiple results where all fail
    def invariant_multiple_all_fail(x):
        return [(False, "error1"), (False, "error2")]
    
    wrapped_multiple_all_fail = wrap_invariant(invariant_multiple_all_fail)
    result = wrapped_multiple_all_fail(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test that wrapped function preserves kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x must be greater than y")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_kwargs(15, y=10)
    assert result == (True, "x must be greater than y")
    
    result = wrapped_kwargs(5, y=10)
    assert result == (False, "x must be greater than y")


# LLM-generated content at query #4
#--------------------------

def test_CheckedPSet_serialize():
    class TestSet(CheckedPSet):
        __type__ = (int,)
    
    # Test basic serialization
    s = TestSet([1, 2, 3])
    result = s.serialize()
    assert isinstance(result, set)
    assert result == {1, 2, 3}
    
    # Test serialization with CheckedType elements
    class Inner(CheckedPVector):
        __type__ = (int,)
    
    class OuterSet(CheckedPSet):
        __type__ = (CheckedPVector,)
    
    inner1 = Inner([1, 2])
    inner2 = Inner([3, 4])
    outer = OuterSet([inner1, inner2])
    result = outer.serialize()
    assert isinstance(result, set)
    assert len(result) == 2
    assert all(isinstance(item, list) for item in result)
    
    # Test serialization with format parameter
    s = TestSet([1, 2, 3])
    result = s.serialize(format='json')
    assert isinstance(result, set)
    assert result == {1, 2, 3}
    
    # Test empty set serialization
    empty_set = TestSet([])
    result = empty_set.serialize()
    assert isinstance(result, set)
    assert result == set()
    
    # Test serialization with mixed CheckedType and primitive
    class MixedSet(CheckedPSet):
        __type__ = (int, str)
    
    mixed = MixedSet([1, 'hello', 2, 'world'])
    result = mixed.serialize()
    assert isinstance(result, set)
    assert result == {1, 'hello', 2, 'world'}


# LLM-generated content at query #5
#--------------------------

```python
def test_CheckedPSet___new__():
    """Test CheckedPSet.__new__ with various input types"""
    
    # Test 1: Create CheckedPSet with empty initial value
    class SimpleSet(CheckedPSet):
        __type__ = int
    
    result = SimpleSet()
    assert isinstance(result, CheckedPSet)
    assert len(result) == 0
    
    # Test 2: Create CheckedPSet with list of valid types
    result = SimpleSet([1, 2, 3])
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    
    # Test 3: Create CheckedPSet with tuple of valid types
    result = SimpleSet((4, 5, 6))
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    assert 4 in result
    
    # Test 4: Create CheckedPSet with set of valid types
    result = SimpleSet({7, 8, 9})
    assert isinstance(result, CheckedPSet)
    assert len(result) == 3
    
    # Test 5: Create CheckedPSet with duplicate values (sets deduplicate)
    result = SimpleSet([1, 1, 2, 2, 3])
    assert len(result) == 3
    
    # Test 6: Create CheckedPSet with multiple allowed types
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    result = MultiTypeSet([1, "hello", 2, "world"])
    assert len(result) == 4
    assert 1 in result
    assert "hello" in result
    
    # Test 7: Invalid type should raise CheckedValueTypeError
    try:
        SimpleSet([1, 2, "invalid"])
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass
    
    # Test 8: Create CheckedPSet with PMap initial value
    pmap_initial = pmap({1: True, 2: True})
    result = SimpleSet(pmap_initial)
    assert isinstance(result, CheckedPSet)
    
    # Test 9: CheckedPSet with invariants
    class PositiveSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda n: (n > 0, 'Must be positive')
    
    result = PositiveSet([1, 2, 3])
    assert len(result) == 3
    
    # Test 10: Invariant violation should raise InvariantException
    try:
        PositiveSet([1, -1, 2])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0
    
    # Test 11: Create with generator
    result = SimpleSet(x for x in [10, 11, 12])
    assert len(result) == 3
    assert 10 in result
    
    # Test 12: Create CheckedPSet preserves type
    result = SimpleSet([100])
    assert type(result) == SimpleSet


# LLM-generated content at query #6
#--------------------------

```python
def test_CheckedPSet_serialize():
    class IntSet(CheckedPSet):
        __type__ = int
    
    # Test basic serialization
    s = IntSet([1, 2, 3])
    result = s.serialize()
    assert isinstance(result, set)
    assert result == {1, 2, 3}
    
    # Test with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        
        @staticmethod
        def __serializer__(format, value):
            return value * 2
    
    s2 = CustomSerializerSet([1, 2, 3])
    result2 = s2.serialize()
    assert result2 == {2, 4, 6}
    
    # Test with format parameter
    class FormatAwareSet(CheckedPSet):
        __type__ = int
        
        @staticmethod
        def __serializer__(format, value):
            if format == 'double':
                return value * 2
            return value
    
    s3 = FormatAwareSet([1, 2, 3])
    result3 = s3.serialize(format='double')
    assert result3 == {2, 4, 6}
    
    result4 = s3.serialize(format='normal')
    assert result4 == {1, 2, 3}
    
    # Test with CheckedType values
    class InnerChecked(CheckedPVector):
        __type__ = int
    
    class OuterSet(CheckedPSet):
        __type__ = CheckedType
        
        @staticmethod
        def __serializer__(format, value):
            if isinstance(value, CheckedType):
                return value.serialize()
            return value
    
    inner = InnerChecked([1, 2])
    outer = OuterSet([inner])
    result5 = outer.serialize()
    assert result5 == {[1, 2]}
    
    # Test empty set serialization
    s4 = IntSet([])
    result6 = s4.serialize()
    assert result6 == set()


# LLM-generated content at query #7
#--------------------------

def test_wrap_invariant():
    # Test case 1: Invariant returns a single boolean result
    def invariant_single(x):
        return (True, "valid")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "valid")
    
    # Test case 2: Invariant returns False
    def invariant_false(x):
        return (False, "invalid")
    
    wrapped_false = wrap_invariant(invariant_false)
    result = wrapped_false(5)
    assert result == (False, "invalid")
    
    # Test case 3: Invariant returns multiple results that need merging
    def invariant_multiple(x):
        return [
            (True, "check1_pass"),
            (True, "check2_pass"),
            (False, "check3_fail")
        ]
    
    wrapped_multiple = wrap_invariant(invariant_multiple)
    result = wrapped_multiple(5)
    assert result == (False, ("check3_fail",))
    
    # Test case 4: Invariant returns multiple results, all passing
    def invariant_multiple_pass(x):
        return [
            (True, "check1_pass"),
            (True, "check2_pass"),
            (True, "check3_pass")
        ]
    
    wrapped_multiple_pass = wrap_invariant(invariant_multiple_pass)
    result = wrapped_multiple_pass(5)
    assert result == (True, ())
    
    # Test case 5: Invariant returns multiple results with multiple failures
    def invariant_multiple_fails(x):
        return [
            (False, "check1_fail"),
            (True, "check2_pass"),
            (False, "check3_fail")
        ]
    
    wrapped_multiple_fails = wrap_invariant(invariant_multiple_fails)
    result = wrapped_multiple_fails(5)
    assert result == (False, ("check1_fail", "check3_fail"))
    
    # Test case 6: Invariant with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x < y, "x should be less than y")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_kwargs(5, y=20)
    assert result == (True, "x should be less than y")
    
    result = wrapped_kwargs(25, y=20)
    assert result == (False, "x should be less than y")


# LLM-generated content at query #8
#--------------------------

def test_CheckedPSet_serialize():
    class IntSet(CheckedPSet):
        __type__ = int
    
    # Test basic serialization
    s = IntSet([1, 2, 3])
    result = s.serialize()
    assert isinstance(result, set)
    assert result == {1, 2, 3}
    
    # Test with custom serializer
    class CustomSet(CheckedPSet):
        __type__ = int
        
        @staticmethod
        def __serializer__(fmt, value):
            return value * 2
    
    cs = CustomSet([1, 2, 3])
    result = cs.serialize()
    assert result == {2, 4, 6}
    
    # Test with CheckedType elements
    class NestedInt(CheckedPVector):
        __type__ = int
    
    class NestedSet(CheckedPSet):
        __type__ = NestedInt
    
    ns = NestedSet([NestedInt([1, 2]), NestedInt([3, 4])])
    result = ns.serialize()
    assert isinstance(result, set)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, list)
    
    # Test empty set
    empty_set = IntSet([])
    result = empty_set.serialize()
    assert result == set()
    
    # Test with format parameter
    class FormattedSet(CheckedPSet):
        __type__ = int
        
        @staticmethod
        def __serializer__(fmt, value):
            if fmt == 'double':
                return value * 2
            return value
    
    fs = FormattedSet([1, 2, 3])
    result = fs.serialize(format='double')
    assert result == {2, 4, 6}
    
    result = fs.serialize(format=None)
    assert result == {1, 2, 3}


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from enum import Enum


def test_CheckedPSet_serialize():
    """Test serialize method of CheckedPSet"""
    
    # Test basic serialization with integers
    class IntSet(CheckedPSet):
        __type__ = int
    
    int_set = IntSet([1, 2, 3])
    serialized = int_set.serialize()
    assert isinstance(serialized, set)
    assert serialized == {1, 2, 3}
    
    # Test serialization with strings
    class StringSet(CheckedPSet):
        __type__ = str
    
    string_set = StringSet(['a', 'b', 'c'])
    serialized = string_set.serialize()
    assert isinstance(serialized, set)
    assert serialized == {'a', 'b', 'c'}
    
    # Test serialization with multiple types
    class MixedSet(CheckedPSet):
        __type__ = (int, str)
    
    mixed_set = MixedSet([1, 'a', 2, 'b'])
    serialized = mixed_set.serialize()
    assert isinstance(serialized, set)
    assert serialized == {1, 'a', 2, 'b'}
    
    # Test serialization with CheckedType objects
    class CustomCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
        
        def serialize(self, format=None):
            return f"serialized_{self.value}"
        
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls(source_data)
    
    class CustomSet(CheckedPSet):
        __type__ = CustomCheckedType
    
    obj1 = CustomCheckedType(10)
    obj2 = CustomCheckedType(20)
    custom_set = CustomSet([obj1, obj2])
    serialized = custom_set.serialize()
    assert isinstance(serialized, set)
    assert 'serialized_10' in serialized
    assert 'serialized_20' in serialized
    
    # Test serialization with format parameter
    class FormattedSet(CheckedPSet):
        __type__ = int
    
    formatted_set = FormattedSet([1, 2, 3])
    serialized = formatted_set.serialize(format='json')
    assert isinstance(serialized, set)
    assert serialized == {1, 2, 3}
    
    # Test empty set serialization
    class EmptySet(CheckedPSet):
        __type__ = int
    
    empty_set = EmptySet([])
    serialized = empty_set.serialize()
    assert isinstance(serialized, set)
    assert len(serialized) == 0


# LLM-generated content at query #10
#--------------------------

```python
def test_store_invariants():
    # Test basic invariant storage
    dct = {'__invariants__': lambda self: (True, None)}
    bases = []
    store_invariants(dct, bases, '__invariants__', '__invariants__')
    assert '__invariants__' in dct
    assert len(dct['__invariants__']) == 1
    assert callable(dct['__invariants__'][0])

    # Test invariant inheritance from base classes
    base_invariant = lambda self: (True, None)
    base_dict = {'__invariants__': base_invariant}
    
    class BaseClass:
        __dict__ = base_dict
        __bases__ = ()
    
    dct = {'__invariants__': lambda self: (True, None)}
    bases = [BaseClass]
    store_invariants(dct, bases, '__invariants__', '__invariants__')
    assert len(dct['__invariants__']) == 2

    # Test with no invariants defined
    dct = {}
    bases = []
    store_invariants(dct, bases, '__invariants__', '__invariants__')
    assert dct['__invariants__'] == ()

    # Test that non-callable invariants raise TypeError
    dct = {'__invariants__': 'not_callable'}
    bases = []
    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants(dct, bases, '__invariants__', '__invariants__')

    # Test multiple invariants in hierarchy
    inv1 = lambda self: (True, None)
    inv2 = lambda self: (True, None)
    inv3 = lambda self: (True, None)
    
    class Base1:
        __dict__ = {'__invariants__': inv1}
        __bases__ = ()
    
    class Base2:
        __dict__ = {'__invariants__': inv2}
        __bases__ = ()
    
    dct = {'__invariants__': inv3}
    bases = [Base1, Base2]
    store_invariants(dct, bases, '__invariants__', '__invariants__')
    assert len(dct['__invariants__']) == 3

    # Test with different source and destination names
    dct = {'custom_invariant': lambda self: (True, None)}
    bases = []
    store_invariants(dct, bases, '__invariants__', 'custom_invariant')
    assert '__invariants__' in dct
    assert len(dct['__invariants__']) == 1

    # Test that invariants are wrapped
    test_invariant = lambda self: (True, None)
    dct = {'__invariants__': test_invariant}
    bases = []
    store_invariants(dct, bases, '__invariants__', '__invariants__')
    wrapped_invariant = dct['__invariants__'][0]
    assert callable(wrapped_invariant)
    assert wrapped_invariant != test_invariant  # Should be wrapped

    # Test mixed callable and non-callable raises error
    dct = {'__invariants__': [lambda self: (True, None), 'not_callable']}
    bases = []
    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants(dct, bases, '__invariants__', '__invariants__')

    # Test circular base class reference handling
    class CircularBase:
        __dict__ = {'__invariants__': lambda self: (True, None)}
        __bases__ = ()
    
    dct = {}
    bases = [CircularBase, CircularBase]  # Same base twice
    store_invariants(dct, bases, '__invariants__', '__invariants__')
    # Should handle duplicates gracefully through _all_dicts
    assert len(dct['__invariants__']) >= 1


# LLM-generated content at query #11
#--------------------------

```python
def test_CheckedPSet_serialize():
    # Test basic serialization of CheckedPSet
    class SimpleCheckedSet(CheckedPSet):
        __type__ = (int,)
    
    s = SimpleCheckedSet([1, 2, 3])
    result = s.serialize()
    assert isinstance(result, set)
    assert result == {1, 2, 3}


def test_CheckedPSet_serialize_with_checked_type_elements():
    # Test serialization with CheckedType elements
    class InnerChecked(CheckedPVector):
        __type__ = (int,)
    
    class OuterCheckedSet(CheckedPSet):
        __type__ = (InnerChecked,)
    
    inner1 = InnerChecked([1, 2])
    inner2 = InnerChecked([3, 4])
    s = OuterCheckedSet([inner1, inner2])
    result = s.serialize()
    
    assert isinstance(result, set)
    assert len(result) == 2
    # Serialized CheckedPVector becomes a list
    assert [1, 2] in result or [3, 4] in result


def test_CheckedPSet_serialize_empty():
    # Test serialization of empty CheckedPSet
    class EmptyCheckedSet(CheckedPSet):
        __type__ = (int,)
    
    s = EmptyCheckedSet([])
    result = s.serialize()
    assert isinstance(result, set)
    assert result == set()


def test_CheckedPSet_serialize_mixed_types():
    # Test serialization with multiple allowed types
    class MixedCheckedSet(CheckedPSet):
        __type__ = (int, str, float)
    
    s = MixedCheckedSet([1, "hello", 3.14])
    result = s.serialize()
    assert isinstance(result, set)
    assert 1 in result
    assert "hello" in result
    assert 3.14 in result


def test_CheckedPSet_serialize_with_format():
    # Test serialization with format parameter
    class FormattedCheckedSet(CheckedPSet):
        __type__ = (int,)
    
    s = FormattedCheckedSet([1, 2, 3])
    result = s.serialize(format="json")
    assert isinstance(result, set)
    assert result == {1, 2, 3}


def test_CheckedPSet_serialize_custom_serializer():
    # Test serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = (int,)
        
        @staticmethod
        def __serializer__(fmt, value):
            return value * 2
    
    s = CustomSerializerSet([1, 2, 3])
    result = s.serialize()
    assert isinstance(result, set)
    assert result == {2, 4, 6}


def test_CheckedPSet_serialize_with_none_type():
    # Test serialization allowing None values
    class NullableCheckedSet(CheckedPSet):
        __type__ = optional(int)
    
    s = NullableCheckedSet([1, 2, None])
    result = s.serialize()
    assert isinstance(result, set)
    assert 1 in result
    assert 2 in result
    assert None in result


# LLM-generated content at query #12
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Invariant returns a single boolean result
    def single_result_invariant(x):
        return (True, "valid")
    
    wrapped = wrap_invariant(single_result_invariant)
    result = wrapped(5)
    assert result == (True, "valid")
    
    # Test case 2: Invariant returns False with error data
    def failing_invariant(x):
        return (False, "error_message")
    
    wrapped = wrap_invariant(failing_invariant)
    result = wrapped(5)
    assert result == (False, "error_message")
    
    # Test case 3: Invariant returns multiple test results (all passing)
    def multi_result_passing_invariant(x):
        return [
            (True, "test1_passed"),
            (True, "test2_passed"),
            (True, "test3_passed")
        ]
    
    wrapped = wrap_invariant(multi_result_passing_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test case 4: Invariant returns multiple test results (some failing)
    def multi_result_failing_invariant(x):
        return [
            (True, "test1_passed"),
            (False, "test2_failed"),
            (True, "test3_passed"),
            (False, "test4_failed")
        ]
    
    wrapped = wrap_invariant(multi_result_failing_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("test2_failed", "test4_failed")
    
    # Test case 5: Invariant returns all failing results
    def multi_result_all_failing_invariant(x):
        return [
            (False, "test1_failed"),
            (False, "test2_failed")
        ]
    
    wrapped = wrap_invariant(multi_result_all_failing_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("test1_failed", "test2_failed")
    
    # Test case 6: Wrapped invariant with arguments and kwargs
    def invariant_with_args(x, y, z=None):
        return (True, f"x={x}, y={y}, z={z}")
    
    wrapped = wrap_invariant(invariant_with_args)
    result = wrapped(1, 2, z=3)
    assert result == (True, "x=1, y=2, z=3")
    
    # Test case 7: Single result with False
    def single_false_invariant():
        return (False, "single_error")
    
    wrapped = wrap_invariant(single_false_invariant)
    result = wrapped()
    assert result == (False, "single_error")


# LLM-generated content at query #13
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant returning a single (bool, data) tuple
    def simple_invariant(x):
        return (True, "valid")
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "valid")
    
    # Test with a failing simple invariant
    def failing_invariant(x):
        return (False, "invalid")
    
    wrapped = wrap_invariant(failing_invariant)
    result = wrapped(5)
    assert result == (False, "invalid")
    
    # Test with multiple invariant results that all pass
    def multi_pass_invariant(x):
        return [
            (True, "check1"),
            (True, "check2"),
            (True, "check3")
        ]
    
    wrapped = wrap_invariant(multi_pass_invariant)
    result = wrapped(5)
    assert result[0] is True
    assert result[1] == ("check1", "check2", "check3")
    
    # Test with multiple invariant results where some fail
    def multi_fail_invariant(x):
        return [
            (True, "check1"),
            (False, "error1"),
            (True, "check2"),
            (False, "error2")
        ]
    
    wrapped = wrap_invariant(multi_fail_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test with all failing invariant results
    def all_fail_invariant(x):
        return [
            (False, "error1"),
            (False, "error2")
        ]
    
    wrapped = wrap_invariant(all_fail_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test that wrapped function preserves arguments
    def check_args_invariant(x, y, z=None):
        return (True, f"x={x}, y={y}, z={z}")
    
    wrapped = wrap_invariant(check_args_invariant)
    result = wrapped(1, 2, z=3)
    assert result == (True, "x=1, y=2, z=3")


# LLM-generated content at query #14
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with a single type
    result = maybe_parse_user_type(int)
    assert result == [int]
    
    # Test with a string
    result = maybe_parse_user_type('int')
    assert result == ['int']
    
    # Test with Enum (preserved iterable type)
    class MyEnum(Enum):
        A = 1
        B = 2
    
    result = maybe_parse_user_type(MyEnum)
    assert result == [MyEnum]
    
    # Test with a list of types
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)
    
    # Test with a tuple of types
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)
    
    # Test with nested iterables
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)
    
    # Test with strings in iterable
    result = maybe_parse_user_type(['int', 'str'])
    assert result == ('int', 'str')
    
    # Test with mixed types and strings
    result = maybe_parse_user_type([int, 'str', float])
    assert result == (int, 'str', float)
    
    # Test with single string in iterable
    result = maybe_parse_user_type(['int'])
    assert result == ('int',)
    
    # Test invalid input - non-type, non-string, non-iterable
    with pytest.raises(TypeError, match='Type specifications must be types or strings'):
        maybe_parse_user_type(42)
    
    # Test invalid input - dict (iterable but empty or with non-type items)
    with pytest.raises(TypeError, match='Type specifications must be types or strings'):
        maybe_parse_user_type({1, 2})
    
    # Test with complex nested structure
    result = maybe_parse_user_type([int, [str, [float, bool]]])
    assert result == (int, str, float, bool)
    
    # Test empty iterable
    result = maybe_parse_user_type([])
    assert result == ()
    
    # Test with PMap (iterable type)
    result = maybe_parse_user_type(pmap({'a': 1}))
    assert result == ()
    
    # Test with list containing Enum
    result = maybe_parse_user_type([int, MyEnum])
    assert result == (int, MyEnum)


# LLM-generated content at query #15
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant returning a single (bool, data) tuple
    def simple_invariant(x):
        return (True, "success")
    
    wrapped_simple = wrap_invariant(simple_invariant)
    result = wrapped_simple(5)
    assert result == (True, "success")
    
    # Test with a failing simple invariant
    def failing_invariant(x):
        return (False, "error_message")
    
    wrapped_failing = wrap_invariant(failing_invariant)
    result = wrapped_failing(5)
    assert result == (False, "error_message")
    
    # Test with multiple invariant results (list of tuples)
    def multi_invariant(x):
        return [
            (True, "check1_passed"),
            (True, "check2_passed"),
            (False, "check3_failed")
        ]
    
    wrapped_multi = wrap_invariant(multi_invariant)
    result = wrapped_multi(5)
    assert result[0] is False
    assert "check3_failed" in result[1]
    assert "check1_passed" not in result[1]
    
    # Test with all passing multiple invariant results
    def all_pass_invariant(x):
        return [
            (True, "check1_passed"),
            (True, "check2_passed")
        ]
    
    wrapped_all_pass = wrap_invariant(all_pass_invariant)
    result = wrapped_all_pass(5)
    assert result[0] is True
    assert result[1] == ("check1_passed", "check2_passed")
    
    # Test with all failing multiple invariant results
    def all_fail_invariant(x):
        return [
            (False, "error1"),
            (False, "error2"),
            (False, "error3")
        ]
    
    wrapped_all_fail = wrap_invariant(all_fail_invariant)
    result = wrapped_all_fail(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2", "error3")
    
    # Test that wrapped function preserves kwargs and args
    def multi_arg_invariant(a, b, c=None):
        return (True, f"args: {a}, {b}, {c}")
    
    wrapped_multi_arg = wrap_invariant(multi_arg_invariant)
    result = wrapped_multi_arg(1, 2, c=3)
    assert result[0] is True
    assert "1, 2, 3" in result[1]


# LLM-generated content at query #16
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with a single type
    result = maybe_parse_user_type(int)
    assert result == [int]
    
    # Test with a string
    result = maybe_parse_user_type('int')
    assert result == ['int']
    
    # Test with a list of types
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)
    
    # Test with a tuple of types
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)
    
    # Test with nested iterables
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)
    
    # Test with Enum type (preserved iterable)
    class Color(Enum):
        RED = 1
        GREEN = 2
    
    result = maybe_parse_user_type(Color)
    assert result == [Color]
    
    # Test with list containing Enum
    result = maybe_parse_user_type([Color, int])
    assert result == (Color, int)
    
    # Test with mixed strings and types in iterable
    result = maybe_parse_user_type([int, 'str'])
    assert result == (int, 'str')
    
    # Test with deeply nested iterables
    result = maybe_parse_user_type([int, [str, [float]]])
    assert result == (int, str, float)
    
    # Test error case: invalid type
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)
    
    # Test error case: object that is not type, string, or iterable
    with pytest.raises(TypeError):
        maybe_parse_user_type(3.14)
    
    # Test with empty iterable
    result = maybe_parse_user_type([])
    assert result == ()
    
    # Test with single element in iterable
    result = maybe_parse_user_type([int])
    assert result == (int,)
    
    # Test with string in iterable
    result = maybe_parse_user_type(['CustomType'])
    assert result == ('CustomType',)


# LLM-generated content at query #17
#--------------------------

def test_wrap_invariant():
    # Test with single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "success")
    
    # Test with single boolean False result
    def invariant_single_fail(x):
        return (False, "error")
    
    wrapped_single_fail = wrap_invariant(invariant_single_fail)
    result = wrapped_single_fail(5)
    assert result == (False, "error")
    
    # Test with multiple results to merge (all True)
    def invariant_multiple_pass(x):
        return [
            (True, "check1"),
            (True, "check2"),
            (True, "check3")
        ]
    
    wrapped_multiple_pass = wrap_invariant(invariant_multiple_pass)
    result = wrapped_multiple_pass(5)
    assert result == (True, ("check1", "check2", "check3"))
    
    # Test with multiple results to merge (some False)
    def invariant_multiple_fail(x):
        return [
            (True, "check1"),
            (False, "error1"),
            (True, "check2"),
            (False, "error2")
        ]
    
    wrapped_multiple_fail = wrap_invariant(invariant_multiple_fail)
    result = wrapped_multiple_fail(5)
    assert result == (False, ("error1", "error2"))
    
    # Test with multiple results all False
    def invariant_multiple_all_fail(x):
        return [
            (False, "error1"),
            (False, "error2")
        ]
    
    wrapped_multiple_all_fail = wrap_invariant(invariant_multiple_all_fail)
    result = wrapped_multiple_all_fail(5)
    assert result == (False, ("error1", "error2"))
    
    # Test with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x must be greater than y")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_kwargs(15, y=10)
    assert result == (True, "x must be greater than y")
    
    result = wrapped_kwargs(5, y=10)
    assert result == (False, "x must be greater than y")


# LLM-generated content at query #18
#--------------------------

def test_wrap_invariant():
    # Test with single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test with single boolean result (False)
    def invariant_single_fail(x):
        return (False, "error")
    
    wrapped_fail = wrap_invariant(invariant_single_fail)
    result_fail = wrapped_fail(5)
    assert result_fail == (False, "error")
    
    # Test with multiple results to merge
    def invariant_multiple(x):
        return [
            (True, "check1_pass"),
            (True, "check2_pass"),
            (False, "check3_fail")
        ]
    
    wrapped_multiple = wrap_invariant(invariant_multiple)
    result_multiple = wrapped_multiple(5)
    assert result_multiple[0] is False
    assert result_multiple[1] == ("check3_fail",)
    
    # Test with all passing multiple results
    def invariant_multiple_pass(x):
        return [
            (True, "check1_pass"),
            (True, "check2_pass")
        ]
    
    wrapped_multiple_pass = wrap_invariant(invariant_multiple_pass)
    result_multiple_pass = wrapped_multiple_pass(5)
    assert result_multiple_pass[0] is True
    assert result_multiple_pass[1] == ()
    
    # Test with kwargs
    def invariant_with_kwargs(x, y=10):
        return (y > 5, "y_check")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result_kwargs = wrapped_kwargs(5, y=20)
    assert result_kwargs == (True, "y_check")
    
    result_kwargs_fail = wrapped_kwargs(5, y=3)
    assert result_kwargs_fail == (False, "y_check")


# LLM-generated content at query #19
#--------------------------

```python
import pytest


def test_wrap_invariant():
    # Test case 1: Single boolean result
    def invariant_single_bool(x):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single_bool)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test case 2: Single boolean result with failure
    def invariant_single_bool_fail(x):
        return (False, "error")
    
    wrapped = wrap_invariant(invariant_single_bool_fail)
    result = wrapped(5)
    assert result == (False, "error")
    
    # Test case 3: Multiple test results that all pass
    def invariant_multiple_pass(x):
        return [
            (True, "test1_pass"),
            (True, "test2_pass"),
            (True, "test3_pass")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_pass)
    result = wrapped(5)
    assert result[0] is True
    assert result[1] == ("test1_pass", "test2_pass", "test3_pass")
    
    # Test case 4: Multiple test results with some failures
    def invariant_multiple_mixed(x):
        return [
            (True, "test1_pass"),
            (False, "test2_fail"),
            (True, "test3_pass"),
            (False, "test4_fail")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_mixed)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("test2_fail", "test4_fail")
    
    # Test case 5: Multiple test results all fail
    def invariant_multiple_all_fail(x):
        return [
            (False, "error1"),
            (False, "error2")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_all_fail)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test case 6: Invariant with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "comparison_result")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(5, y=3)
    assert result == (True, "comparison_result")
    
    result = wrapped(5, y=10)
    assert result == (False, "comparison_result")
    
    # Test case 7: Multiple results with single item (edge case)
    def invariant_single_item_list(x):
        return [(True, "only_test")]
    
    wrapped = wrap_invariant(invariant_single_item_list)
    result = wrapped(5)
    assert result[0] is True
    assert result[1] == ("only_test",)


# LLM-generated content at query #20
#--------------------------

def test_CheckedPSet_serialize():
    class IntSet(CheckedPSet):
        __type__ = int

    # Test basic serialization
    s = IntSet([1, 2, 3])
    result = s.serialize()
    assert isinstance(result, set)
    assert result == {1, 2, 3}

    # Test empty set serialization
    empty_set = IntSet([])
    assert empty_set.serialize() == set()

    # Test serialization with custom serializer
    class CustomSet(CheckedPSet):
        __type__ = int
        
        @staticmethod
        def __serializer__(format, value):
            return value * 2

    cs = CustomSet([1, 2, 3])
    result = cs.serialize()
    assert result == {2, 4, 6}

    # Test serialization with CheckedType elements
    class NestedSet(CheckedPSet):
        __type__ = int

    nested = NestedSet([5, 10, 15])
    result = nested.serialize(format='json')
    assert result == {5, 10, 15}

    # Test serialization preserves set type
    s2 = IntSet([7, 8, 9])
    serialized = s2.serialize()
    assert isinstance(serialized, set)
    assert len(serialized) == 3


# LLM-generated content at query #21
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Invariant returning a single (bool, data) tuple
    def single_invariant(x):
        return (True, "success")
    
    wrapped_single = wrap_invariant(single_invariant)
    result = wrapped_single(5)
    assert result == (True, "success")
    
    # Test case 2: Invariant returning False with error data
    def failing_invariant(x):
        return (False, "error_message")
    
    wrapped_failing = wrap_invariant(failing_invariant)
    result = wrapped_failing(5)
    assert result == (False, "error_message")
    
    # Test case 3: Invariant returning multiple test results that all pass
    def multi_pass_invariant(x):
        return [
            (True, "test1_passed"),
            (True, "test2_passed"),
            (True, "test3_passed")
        ]
    
    wrapped_multi_pass = wrap_invariant(multi_pass_invariant)
    result = wrapped_multi_pass(5)
    assert result[0] is True
    assert result[1] == ()
    
    # Test case 4: Invariant returning multiple test results with some failures
    def multi_fail_invariant(x):
        return [
            (True, "test1_passed"),
            (False, "test2_failed"),
            (True, "test3_passed"),
            (False, "test4_failed")
        ]
    
    wrapped_multi_fail = wrap_invariant(multi_fail_invariant)
    result = wrapped_multi_fail(5)
    assert result[0] is False
    assert result[1] == ("test2_failed", "test4_failed")
    
    # Test case 5: Invariant returning multiple test results all failing
    def multi_all_fail_invariant(x):
        return [
            (False, "error1"),
            (False, "error2")
        ]
    
    wrapped_multi_all_fail = wrap_invariant(multi_all_fail_invariant)
    result = wrapped_multi_all_fail(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test case 6: Invariant with kwargs
    def kwargs_invariant(x, y=10):
        return (x > y, "x_greater_than_y")
    
    wrapped_kwargs = wrap_invariant(kwargs_invariant)
    result = wrapped_kwargs(15, y=10)
    assert result == (True, "x_greater_than_y")
    
    result = wrapped_kwargs(5, y=10)
    assert result == (False, "x_greater_than_y")


# LLM-generated content at query #22
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single bool and data tuple
    def simple_invariant(x):
        return (True, "valid")
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "valid")
    
    # Test with a simple invariant that returns False
    def failing_invariant(x):
        return (False, "invalid")
    
    wrapped = wrap_invariant(failing_invariant)
    result = wrapped(5)
    assert result == (False, "invalid")
    
    # Test with multiple invariant results that all pass
    def multi_pass_invariant(x):
        return [
            (True, "check1"),
            (True, "check2"),
            (True, "check3")
        ]
    
    wrapped = wrap_invariant(multi_pass_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test with multiple invariant results with some failures
    def multi_fail_invariant(x):
        return [
            (True, "check1"),
            (False, "check2_failed"),
            (True, "check3"),
            (False, "check4_failed")
        ]
    
    wrapped = wrap_invariant(multi_fail_invariant)
    result = wrapped(5)
    assert result == (False, ("check2_failed", "check4_failed"))
    
    # Test with all failing results
    def all_fail_invariant(x):
        return [
            (False, "error1"),
            (False, "error2")
        ]
    
    wrapped = wrap_invariant(all_fail_invariant)
    result = wrapped(5)
    assert result == (False, ("error1", "error2"))
    
    # Test that wrapped function preserves arguments
    def invariant_with_args(x, y, z=None):
        return (x > 0 and y > 0, "args_check")
    
    wrapped = wrap_invariant(invariant_with_args)
    result = wrapped(1, 2, z=3)
    assert result == (True, "args_check")
    
    result = wrapped(-1, 2, z=3)
    assert result == (False, "args_check")


# LLM-generated content at query #23
#--------------------------

def test_wrap_invariant():
    # Test with single boolean result
    def invariant_single(x):
        return (True, "valid")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "valid")
    
    # Test with single boolean False result
    def invariant_single_false(x):
        return (False, "invalid")
    
    wrapped_single_false = wrap_invariant(invariant_single_false)
    result = wrapped_single_false(5)
    assert result == (False, "invalid")
    
    # Test with multiple boolean results to merge
    def invariant_multiple(x):
        return [(True, "check1"), (True, "check2"), (True, "check3")]
    
    wrapped_multiple = wrap_invariant(invariant_multiple)
    result = wrapped_multiple(5)
    assert result == (True, ())
    
    # Test with multiple results where some fail
    def invariant_multiple_fail(x):
        return [(True, "check1"), (False, "error1"), (True, "check3"), (False, "error2")]
    
    wrapped_multiple_fail = wrap_invariant(invariant_multiple_fail)
    result = wrapped_multiple_fail(5)
    assert result[0] is False
    assert "error1" in result[1]
    assert "error2" in result[1]
    
    # Test with all failing results
    def invariant_all_fail(x):
        return [(False, "error1"), (False, "error2")]
    
    wrapped_all_fail = wrap_invariant(invariant_all_fail)
    result = wrapped_all_fail(5)
    assert result[0] is False
    assert len(result[1]) == 2
    
    # Test with kwargs
    def invariant_with_kwargs(x, y=10):
        return (y > x, "y must be greater than x")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_kwargs(5, y=15)
    assert result == (True, "y must be greater than x")
    
    result = wrapped_kwargs(5, y=3)
    assert result == (False, "y must be greater than x")


# LLM-generated content at query #24
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: invariant returns a single (bool, data) tuple
    def invariant_single(x):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test case 2: invariant returns False with error data
    def invariant_single_false(x):
        return (False, "error")
    
    wrapped = wrap_invariant(invariant_single_false)
    result = wrapped(5)
    assert result == (False, "error")
    
    # Test case 3: invariant returns multiple results that need merging
    def invariant_multiple(x):
        return [(True, "check1"), (True, "check2"), (False, "check3")]
    
    wrapped = wrap_invariant(invariant_multiple)
    result = wrapped(5)
    assert result == (False, ("check3",))
    
    # Test case 4: invariant returns multiple results all passing
    def invariant_multiple_all_pass(x):
        return [(True, "check1"), (True, "check2")]
    
    wrapped = wrap_invariant(invariant_multiple_all_pass)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test case 5: invariant returns multiple results with multiple failures
    def invariant_multiple_failures(x):
        return [(True, "check1"), (False, "error1"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_multiple_failures)
    result = wrapped(5)
    assert result == (False, ("error1", "error2"))
    
    # Test case 6: invariant with kwargs
    def invariant_with_kwargs(x, y=None):
        return (x > 0, "x must be positive")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(5, y=10)
    assert result == (True, "x must be positive")
    
    # Test case 7: invariant returns multiple empty results
    def invariant_multiple_empty(x):
        return []
    
    wrapped = wrap_invariant(invariant_multiple_empty)
    result = wrapped(5)
    assert result == (True, ())


# LLM-generated content at query #25
#--------------------------

```python
import pytest


def test_wrap_invariant():
    # Test case 1: invariant returns a single (bool, data) tuple
    def simple_invariant(x):
        return (True, "success")
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test case 2: invariant returns False with error data
    def failing_invariant(x):
        return (False, "error_message")
    
    wrapped = wrap_invariant(failing_invariant)
    result = wrapped(5)
    assert result == (False, "error_message")
    
    # Test case 3: invariant returns multiple test results that need merging
    def multi_test_invariant(x):
        return [
            (True, "test1_pass"),
            (True, "test2_pass"),
            (False, "test3_fail")
        ]
    
    wrapped = wrap_invariant(multi_test_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("test3_fail",)
    
    # Test case 4: invariant returns multiple results, all passing
    def multi_test_all_pass_invariant(x):
        return [
            (True, "test1_pass"),
            (True, "test2_pass"),
            (True, "test3_pass")
        ]
    
    wrapped = wrap_invariant(multi_test_all_pass_invariant)
    result = wrapped(5)
    assert result[0] is True
    assert result[1] == ()
    
    # Test case 5: invariant returns multiple results with multiple failures
    def multi_test_multi_fail_invariant(x):
        return [
            (True, "test1_pass"),
            (False, "test2_fail"),
            (False, "test3_fail")
        ]
    
    wrapped = wrap_invariant(multi_test_multi_fail_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("test2_fail", "test3_fail")
    
    # Test case 6: invariant with keyword arguments
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x_greater_than_y")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(15, y=10)
    assert result == (True, "x_greater_than_y")
    
    result = wrapped(5, y=10)
    assert result == (False, "x_greater_than_y")
    
    # Test case 7: invariant returns empty list (all pass scenario)
    def empty_results_invariant():
        return []
    
    wrapped = wrap_invariant(empty_results_invariant)
    result = wrapped()
    assert result[0] is True
    assert result[1] == ()


# LLM-generated content at query #26
#--------------------------

```python
import pytest


def test_wrap_invariant():
    # Test with a simple invariant that returns (bool, data)
    def simple_invariant(x):
        return (x > 0, "x must be positive")
    
    wrapped = wrap_invariant(simple_invariant)
    assert wrapped(5) == (True, "x must be positive")
    assert wrapped(-1) == (True, "x must be positive")
    
    
    # Test with an invariant that returns multiple results
    def multi_invariant(x, y):
        results = [
            (x > 0, "x must be positive"),
            (y > 0, "y must be positive"),
            (x + y < 100, "sum must be less than 100")
        ]
        return results
    
    wrapped_multi = wrap_invariant(multi_invariant)
    result = wrapped_multi(5, 10)
    assert result[0] is True
    assert len(result[1]) == 0
    
    result = wrapped_multi(-5, 10)
    assert result[0] is False
    assert len(result[1]) == 1
    assert result[1][0] == "x must be positive"
    
    result = wrapped_multi(5, -10)
    assert result[0] is False
    assert len(result[1]) == 1
    assert result[1][0] == "y must be positive"
    
    result = wrapped_multi(60, 50)
    assert result[0] is False
    assert len(result[1]) == 1
    assert result[1][0] == "sum must be less than 100"
    
    result = wrapped_multi(-5, -10)
    assert result[0] is False
    assert len(result[1]) == 2
    
    
    # Test with invariant returning callable error
    def invariant_with_callable():
        return (False, lambda: "delayed error message")
    
    wrapped_callable = wrap_invariant(invariant_with_callable)
    verdict, errors = wrapped_callable()
    assert verdict is True
    assert errors == ()
    
    
    # Test with mixed results - some pass, some fail
    def mixed_invariant(val):
        return [
            (val > 0, "positive"),
            (val < 100, "less than 100"),
            (val % 2 == 0, "even")
        ]
    
    wrapped_mixed = wrap_invariant(mixed_invariant)
    verdict, errors = wrapped_mixed(50)
    assert verdict is True
    assert errors == ()
    
    verdict, errors = wrapped_mixed(101)
    assert verdict is False
    assert "less than 100" in errors
    
    verdict, errors = wrapped_mixed(51)
    assert verdict is False
    assert "even" in errors
    
    
    # Test with empty results list
    def empty_invariant():
        return []
    
    wrapped_empty = wrap_invariant(empty_invariant)
    verdict, errors = wrapped_empty()
    assert verdict is True
    assert errors == ()


# LLM-generated content at query #27
#--------------------------

def test_wrap_invariant():
    # Test with a single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "success")
    
    # Test with a single boolean result that fails
    def invariant_single_fail(x):
        return (False, "error")
    
    wrapped_single_fail = wrap_invariant(invariant_single_fail)
    result = wrapped_single_fail(5)
    assert result == (False, "error")
    
    # Test with multiple results that all pass
    def invariant_multi_pass(x):
        return [(True, "check1"), (True, "check2"), (True, "check3")]
    
    wrapped_multi_pass = wrap_invariant(invariant_multi_pass)
    result = wrapped_multi_pass(5)
    assert result == (True, ())
    
    # Test with multiple results where some fail
    def invariant_multi_fail(x):
        return [(True, "check1"), (False, "error1"), (True, "check3"), (False, "error2")]
    
    wrapped_multi_fail = wrap_invariant(invariant_multi_fail)
    result = wrapped_multi_fail(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test with multiple results where all fail
    def invariant_multi_all_fail(x):
        return [(False, "error1"), (False, "error2")]
    
    wrapped_multi_all_fail = wrap_invariant(invariant_multi_all_fail)
    result = wrapped_multi_all_fail(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x must be greater than y")
    
    wrapped_with_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_with_kwargs(15, y=10)
    assert result == (True, "x must be greater than y")
    
    result = wrapped_with_kwargs(5, y=10)
    assert result == (False, "x must be greater than y")


# LLM-generated content at query #28
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Single boolean result
    def invariant_single_bool(x):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single_bool)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test case 2: Single boolean result with failure
    def invariant_single_bool_fail(x):
        return (False, "error")
    
    wrapped = wrap_invariant(invariant_single_bool_fail)
    result = wrapped(5)
    assert result == (False, "error")
    
    # Test case 3: Multiple results to merge - all pass
    def invariant_multiple_pass(x):
        return [
            (True, "check1"),
            (True, "check2"),
            (True, "check3")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_pass)
    result = wrapped(5)
    assert result[0] is True
    assert result[1] == ()
    
    # Test case 4: Multiple results to merge - some fail
    def invariant_multiple_fail(x):
        return [
            (True, "check1"),
            (False, "error1"),
            (True, "check2"),
            (False, "error2")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_fail)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test case 5: Multiple results with all failures
    def invariant_all_fail(x):
        return [
            (False, "error1"),
            (False, "error2")
        ]
    
    wrapped = wrap_invariant(invariant_all_fail)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test case 6: Single result in list format
    def invariant_single_in_list(x):
        return [(True, "single")]
    
    wrapped = wrap_invariant(invariant_single_in_list)
    result = wrapped(5)
    assert result[0] is True
    assert result[1] == ()
    
    # Test case 7: Invariant with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x must be greater than y")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(15, y=5)
    assert result == (True, "x must be greater than y")
    
    result = wrapped(3, y=5)
    assert result == (False, "x must be greater than y")


# LLM-generated content at query #29
#--------------------------

def test_wrap_invariant():
    # Test with simple boolean return (verdict, data)
    def simple_invariant(x):
        return (True, "success")
    
    wrapped_simple = wrap_invariant(simple_invariant)
    result = wrapped_simple(5)
    assert result == (True, "success")
    
    # Test with simple boolean return (False, data)
    def failing_invariant(x):
        return (False, "failure")
    
    wrapped_failing = wrap_invariant(failing_invariant)
    result = wrapped_failing(5)
    assert result == (False, "failure")
    
    # Test with multiple results to merge (all passing)
    def multi_invariant_pass(x):
        return [
            (True, "check1"),
            (True, "check2"),
            (True, "check3")
        ]
    
    wrapped_multi_pass = wrap_invariant(multi_invariant_pass)
    result = wrapped_multi_pass(5)
    assert result == (True, ())
    
    # Test with multiple results to merge (some failing)
    def multi_invariant_fail(x):
        return [
            (True, "check1"),
            (False, "error1"),
            (True, "check3"),
            (False, "error2")
        ]
    
    wrapped_multi_fail = wrap_invariant(multi_invariant_fail)
    result = wrapped_multi_fail(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x must be greater than y")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_kwargs(5, y=3)
    assert result == (True, "x must be greater than y")
    
    result = wrapped_kwargs(2, y=3)
    assert result == (False, "x must be greater than y")
    
    # Test with all failing multiple results
    def multi_invariant_all_fail(x):
        return [
            (False, "error1"),
            (False, "error2")
        ]
    
    wrapped_all_fail = wrap_invariant(multi_invariant_all_fail)
    result = wrapped_all_fail(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")


# LLM-generated content at query #30
#--------------------------

def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean result
    def simple_invariant(x):
        return (x > 0, "x must be positive")
    
    wrapped_simple = wrap_invariant(simple_invariant)
    verdict, data = wrapped_simple(5)
    assert verdict is True
    assert data == "x must be positive"
    
    verdict, data = wrapped_simple(-5)
    assert verdict is False
    assert data == "x must be positive"
    
    # Test with an invariant that returns multiple test results
    def multi_invariant(x, y):
        return [
            (x > 0, "x must be positive"),
            (y > 0, "y must be positive"),
            (x + y < 100, "sum must be less than 100")
        ]
    
    wrapped_multi = wrap_invariant(multi_invariant)
    
    # All tests pass
    verdict, data = wrapped_multi(10, 20)
    assert verdict is True
    assert data == ()
    
    # One test fails
    verdict, data = wrapped_multi(-5, 20)
    assert verdict is False
    assert len(data) == 1
    assert data[0] == "x must be positive"
    
    # Multiple tests fail
    verdict, data = wrapped_multi(-5, -10)
    assert verdict is False
    assert len(data) == 2
    assert "x must be positive" in data
    assert "y must be positive" in data
    
    # Test with all tests failing
    verdict, data = wrapped_multi(-5, -10)
    assert verdict is False
    assert len(data) == 2
    
    # Test with sum exceeding limit
    verdict, data = wrapped_multi(60, 50)
    assert verdict is False
    assert len(data) == 1
    assert data[0] == "sum must be less than 100"
    
    # Test with empty results list
    def empty_invariant(x):
        return []
    
    wrapped_empty = wrap_invariant(empty_invariant)
    verdict, data = wrapped_empty(10)
    assert verdict is True
    assert data == ()


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__CheckedMapTypeMeta___new__():
    """Test _CheckedMapTypeMeta.__new__ method"""
    
    # Test basic metaclass creation
    class TestMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
    
    assert hasattr(TestMap, '_checked_key_types')
    assert hasattr(TestMap, '_checked_value_types')
    assert hasattr(TestMap, '__serializer__')
    assert TestMap.__slots__ == ()
    assert TestMap._checked_key_types == (str,)
    assert TestMap._checked_value_types == (int,)
    
    # Test with multiple types
    class MultiTypeMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = (str, int)
        __value_type__ = (int, float)
    
    assert MultiTypeMap._checked_key_types == (str, int)
    assert MultiTypeMap._checked_value_types == (int, float)
    
    # Test with invariant
    class MapWithInvariant(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
        __invariant__ = lambda self: (True, 'test')
    
    assert hasattr(MapWithInvariant, '_checked_invariants')
    assert len(MapWithInvariant._checked_invariants) > 0
    
    # Test inheritance
    class BaseMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
    
    class DerivedMap(BaseMap):
        pass
    
    assert DerivedMap._checked_key_types == (str,)
    assert DerivedMap._checked_value_types == (int,)
    
    # Test default serializer
    test_map = TestMap()
    serializer = test_map.__serializer__
    assert callable(serializer)
    
    # Test serializer with non-CheckedType values
    result = serializer(None, 'key', 42)
    assert result == ('key', 42)
    
    # Test serializer with CheckedType values
    class CustomCheckedType(CheckedType):
        def serialize(self, format=None):
            return 'serialized'
        
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
    
    class MapWithCheckedTypes(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
    
    map_instance = MapWithCheckedTypes()
    serializer = map_instance.__serializer__
    
    checked_key = CustomCheckedType()
    checked_value = CustomCheckedType()
    
    result = serializer(None, checked_key, checked_value)
    assert result == ('serialized', 'serialized')
    
    # Test that slots are always set to empty tuple
    class SlotTestMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
    
    assert SlotTestMap.__slots__ == ()
    
    # Test with no type specifications
    class NoTypeMap(metaclass=_CheckedMapTypeMeta):
        pass
    
    assert NoTypeMap._checked_key_types == ()
    assert NoTypeMap._checked_value_types == ()
    assert hasattr(NoTypeMap, '__serializer__')


# LLM-generated content at query #2
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Single boolean result
    def invariant_single_bool(x):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single_bool)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test case 2: Single boolean result that fails
    def invariant_single_bool_fail(x):
        return (False, "error message")
    
    wrapped = wrap_invariant(invariant_single_bool_fail)
    result = wrapped(5)
    assert result == (False, "error message")
    
    # Test case 3: Multiple test results (list of tuples)
    def invariant_multiple_results(x):
        return [
            (True, "test1 passed"),
            (True, "test2 passed"),
        ]
    
    wrapped = wrap_invariant(invariant_multiple_results)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test case 4: Multiple test results with one failure
    def invariant_multiple_results_with_failure(x):
        return [
            (True, "test1 passed"),
            (False, "test2 failed"),
            (True, "test3 passed"),
        ]
    
    wrapped = wrap_invariant(invariant_multiple_results_with_failure)
    result = wrapped(5)
    assert result[0] is False
    assert "test2 failed" in result[1]
    
    # Test case 5: Multiple test results all fail
    def invariant_all_fail(x):
        return [
            (False, "error1"),
            (False, "error2"),
            (False, "error3"),
        ]
    
    wrapped = wrap_invariant(invariant_all_fail)
    result = wrapped(5)
    assert result[0] is False
    assert len(result[1]) == 3
    assert "error1" in result[1]
    assert "error2" in result[1]
    assert "error3" in result[1]
    
    # Test case 6: Invariant with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, f"x={x}, y={y}")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(5, y=3)
    assert result == (True, f"x=5, y=3")
    
    # Test case 7: Empty results list
    def invariant_empty_results():
        return []
    
    wrapped = wrap_invariant(invariant_empty_results)
    result = wrapped()
    assert result == (True, ())


# LLM-generated content at query #3
#--------------------------

def test_maybe_parse_user_type():
    # Test with a single type
    result = maybe_parse_user_type(int)
    assert result == [int]
    
    # Test with a string
    result = maybe_parse_user_type("int")
    assert result == ["int"]
    
    # Test with a preserved type (Enum)
    result = maybe_parse_user_type(Enum)
    assert result == [Enum]
    
    # Test with a list of types
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)
    
    # Test with a tuple of types
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)
    
    # Test with a list containing strings
    result = maybe_parse_user_type(["int", "str"])
    assert result == ("int", "str")
    
    # Test with mixed list of types and strings
    result = maybe_parse_user_type([int, "str", float])
    assert result == (int, "str", float)
    
    # Test with nested iterables
    result = maybe_parse_user_type([[int, str], float])
    assert result == (int, str, float)
    
    # Test with invalid input (non-type, non-string, non-iterable)
    with pytest.raises(TypeError, match="Type specifications must be types or strings"):
        maybe_parse_user_type(42)
    
    with pytest.raises(TypeError, match="Type specifications must be types or strings"):
        maybe_parse_user_type(3.14)
    
    with pytest.raises(TypeError, match="Type specifications must be types or strings"):
        maybe_parse_user_type(None)
    
    # Test with dict (iterable but should work)
    result = maybe_parse_user_type({int: str})
    assert set(result) == {int, str}
    
    # Test with empty iterable
    result = maybe_parse_user_type([])
    assert result == ()
    
    # Test with custom class
    class CustomClass:
        pass
    
    result = maybe_parse_user_type(CustomClass)
    assert result == [CustomClass]
    
    # Test with Enum subclass
    class Color(Enum):
        RED = 1
        GREEN = 2
    
    result = maybe_parse_user_type(Color)
    assert result == [Color]


# LLM-generated content at query #4
#--------------------------

```python
def test_get_type():
    # Test with a type object - should return the type as-is
    assert get_type(int) == int
    assert get_type(str) == str
    assert get_type(list) == list
    assert get_type(CheckedType) == CheckedType
    
    # Test with a string representing a built-in type
    assert get_type('builtins.int') == int
    assert get_type('builtins.str') == str
    assert get_type('builtins.list') == list
    
    # Test with a string representing a class from this module
    assert get_type('pyrsistent._checked_types.CheckedType') == CheckedType
    assert get_type('pyrsistent._checked_types.InvariantException') == InvariantException
    assert get_type('pyrsistent._checked_types.CheckedTypeError') == CheckedTypeError
    
    # Test that invalid string raises appropriate error
    with pytest.raises(ValueError):
        get_type('invalid_module_name')
    
    with pytest.raises((ImportError, ModuleNotFoundError)):
        get_type('nonexistent.module.Class')
    
    with pytest.raises(AttributeError):
        get_type('builtins.NonexistentClass')


# LLM-generated content at query #5
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with a single type
    result = maybe_parse_user_type(int)
    assert result == [int]
    
    # Test with a string
    result = maybe_parse_user_type("MyType")
    assert result == ["MyType"]
    
    # Test with Enum (preserved iterable type)
    class MyEnum(Enum):
        A = 1
        B = 2
    
    result = maybe_parse_user_type(MyEnum)
    assert result == [MyEnum]
    
    # Test with a list of types
    result = maybe_parse_user_type([int, str])
    assert set(result) == {int, str}
    
    # Test with a tuple of types
    result = maybe_parse_user_type((int, str, float))
    assert set(result) == {int, str, float}
    
    # Test with nested iterables
    result = maybe_parse_user_type([int, [str, float]])
    assert set(result) == {int, str, float}
    
    # Test with a list containing strings
    result = maybe_parse_user_type(["Type1", "Type2"])
    assert set(result) == {"Type1", "Type2"}
    
    # Test with mixed types and strings in iterable
    result = maybe_parse_user_type([int, "CustomType"])
    assert set(result) == {int, "CustomType"}
    
    # Test with invalid input (non-type, non-string, non-iterable)
    import pytest
    with pytest.raises(TypeError, match="Type specifications must be types or strings"):
        maybe_parse_user_type(42)
    
    with pytest.raises(TypeError, match="Type specifications must be types or strings"):
        maybe_parse_user_type(3.14)
    
    with pytest.raises(TypeError, match="Type specifications must be types or strings"):
        maybe_parse_user_type(None)
    
    # Test with complex nested structure
    result = maybe_parse_user_type([int, (str, [float, "Custom"])])
    assert set(result) == {int, str, float, "Custom"}
    
    # Test with empty iterable
    result = maybe_parse_user_type([])
    assert result == []


# LLM-generated content at query #6
#--------------------------

```python
def test__CheckedTypeMeta___new__():
    """Test _CheckedTypeMeta.__new__ creates class with correct attributes."""
    
    # Test basic metaclass instantiation
    class TestChecked(metaclass=_CheckedTypeMeta):
        __type__ = str
        
        def __invariant__(self):
            return True, "valid"
    
    assert hasattr(TestChecked, '_checked_types')
    assert hasattr(TestChecked, '_checked_invariants')
    assert hasattr(TestChecked, '__serializer__')
    assert TestChecked.__slots__ == ()
    assert TestChecked._checked_types == (str,)
    assert len(TestChecked._checked_invariants) == 1


def test__CheckedTypeMeta___new__with_multiple_types():
    """Test metaclass with multiple types."""
    
    class TestChecked(metaclass=_CheckedTypeMeta):
        __type__ = (str, int)
    
    assert TestChecked._checked_types == (str, int)


def test__CheckedTypeMeta___new__with_inherited_invariants():
    """Test metaclass inherits invariants from base classes."""
    
    class BaseChecked(metaclass=_CheckedTypeMeta):
        def __invariant__(self):
            return True, "base_valid"
    
    class DerivedChecked(BaseChecked):
        def __invariant__(self):
            return True, "derived_valid"
    
    assert len(DerivedChecked._checked_invariants) == 2


def test__CheckedTypeMeta___new__default_serializer():
    """Test default serializer is set."""
    
    class TestChecked(metaclass=_CheckedTypeMeta):
        pass
    
    instance = TestChecked()
    serializer = TestChecked.__serializer__
    assert callable(serializer)
    
    # Test serializer with non-CheckedType value
    assert serializer(instance, None, "test") == "test"
    assert serializer(instance, None, 42) == 42


def test__CheckedTypeMeta___new__serializer_with_checked_type():
    """Test default serializer with CheckedType values."""
    
    class MockCheckedType(CheckedType):
        def serialize(self, format=None):
            return {"serialized": True}
        
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()
    
    class TestChecked(metaclass=_CheckedTypeMeta):
        pass
    
    instance = TestChecked()
    serializer = TestChecked.__serializer__
    mock_obj = MockCheckedType()
    
    result = serializer(instance, None, mock_obj)
    assert result == {"serialized": True}


def test__CheckedTypeMeta___new__with_string_type():
    """Test metaclass with string type specification."""
    
    class TestChecked(metaclass=_CheckedTypeMeta):
        __type__ = 'CustomType'
    
    assert TestChecked._checked_types == ('CustomType',)


def test__CheckedTypeMeta___new__invalid_invariant_raises_error():
    """Test that non-callable invariant raises TypeError."""
    
    with raises(TypeError, match='Invariants must be callable'):
        class TestChecked(metaclass=_CheckedTypeMeta):
            __invariant__ = "not_callable"


def test__CheckedTypeMeta___new__slots_always_empty():
    """Test that __slots__ is always set to empty tuple."""
    
    class TestChecked(metaclass=_CheckedTypeMeta):
        __type__ = int
    
    assert TestChecked.__slots__ == ()
    assert not hasattr(TestChecked, '__dict__')


def test__CheckedTypeMeta___new__inherits_types_from_base():
    """Test that types are inherited from base classes."""
    
    class BaseChecked(metaclass=_CheckedTypeMeta):
        __type__ = str
    
    class DerivedChecked(BaseChecked):
        __type__ = int
    
    # Derived should have both types
    assert int in DerivedChecked._checked_types
    assert str in DerivedChecked._checked_types


# LLM-generated content at query #7
#--------------------------

```python
def test__CheckedMapTypeMeta___new__():
    """Test that _CheckedMapTypeMeta.__new__ correctly processes type and invariant metadata."""
    
    # Test basic class creation with key and value types
    class TestCheckedMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
        __invariant__ = lambda self: (True, 'valid')
    
    assert hasattr(TestCheckedMap, '_checked_key_types')
    assert hasattr(TestCheckedMap, '_checked_value_types')
    assert hasattr(TestCheckedMap, '_checked_invariants')
    assert TestCheckedMap._checked_key_types == (str,)
    assert TestCheckedMap._checked_value_types == (int,)
    assert len(TestCheckedMap._checked_invariants) == 1
    
    # Test that __serializer__ is set to default
    assert hasattr(TestCheckedMap, '__serializer__')
    assert callable(TestCheckedMap.__serializer__)
    
    # Test that __slots__ is set to empty tuple
    assert TestCheckedMap.__slots__ == ()


def test__CheckedMapTypeMeta___new__with_multiple_types():
    """Test _CheckedMapTypeMeta.__new__ with multiple types."""
    
    class TestCheckedMapMulti(metaclass=_CheckedMapTypeMeta):
        __key_type__ = (str, int)
        __value_type__ = (int, float)
    
    assert TestCheckedMapMulti._checked_key_types == (str, int)
    assert TestCheckedMapMulti._checked_value_types == (int, float)


def test__CheckedMapTypeMeta___new__with_inheritance():
    """Test _CheckedMapTypeMeta.__new__ correctly inherits types from base classes."""
    
    class BaseMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
    
    class DerivedMap(BaseMap):
        __key_type__ = (str, int)
    
    assert DerivedMap._checked_key_types == (str, int)
    assert DerivedMap._checked_value_types == (int,)


def test__CheckedMapTypeMeta___new__default_serializer():
    """Test that the default serializer in _CheckedMapTypeMeta works correctly."""
    
    class TestMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
    
    serializer = TestMap.__serializer__
    
    # Test with regular values
    sk, sv = serializer(None, None, 'key', 42)
    assert sk == 'key'
    assert sv == 42
    
    # Test with CheckedType values
    class TestCheckedValue(CheckedType):
        def serialize(self, format=None):
            return "serialized"
        
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            pass
    
    checked_val = TestCheckedValue()
    sk, sv = serializer(None, None, 'key', checked_val)
    assert sk == 'key'
    assert sv == 'serialized'


def test__CheckedMapTypeMeta___new__with_invariants():
    """Test that invariants are properly stored and wrapped."""
    
    def test_invariant(self):
        return (True, 'test')
    
    class TestMapWithInvariant(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
        __invariant__ = test_invariant
    
    assert len(TestMapWithInvariant._checked_invariants) == 1
    assert callable(TestMapWithInvariant._checked_invariants[0])


def test__CheckedMapTypeMeta___new__no_types_specified():
    """Test _CheckedMapTypeMeta.__new__ when no types are specified."""
    
    class MinimalMap(metaclass=_CheckedMapTypeMeta):
        pass
    
    assert MinimalMap._checked_key_types == ()
    assert MinimalMap._checked_value_types == ()
    assert MinimalMap._checked_invariants == ()
    assert hasattr(MinimalMap, '__serializer__')
    assert MinimalMap.__slots__ == ()


# LLM-generated content at query #8
#--------------------------

def test_maybe_parse_user_type():
    # Test with a single type
    result = maybe_parse_user_type(int)
    assert result == [int]
    
    # Test with a string
    result = maybe_parse_user_type("int")
    assert result == ["int"]
    
    # Test with a list of types
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)
    
    # Test with a tuple of types
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)
    
    # Test with nested iterables
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)
    
    # Test with Enum (preserved iterable type)
    result = maybe_parse_user_type(Enum)
    assert result == [Enum]
    
    # Test with mixed types and strings in iterable
    result = maybe_parse_user_type([int, "str", float])
    assert result == (int, "str", float)
    
    # Test with deeply nested iterables
    result = maybe_parse_user_type([int, [str, [float]]])
    assert result == (int, str, float)
    
    # Test with invalid input - not a type, string, or iterable
    with pytest.raises(TypeError, match="Type specifications must be types or strings"):
        maybe_parse_user_type(123)
    
    # Test with invalid input - None
    with pytest.raises(TypeError, match="Type specifications must be types or strings"):
        maybe_parse_user_type(None)
    
    # Test with empty list
    result = maybe_parse_user_type([])
    assert result == ()
    
    # Test with single type in list
    result = maybe_parse_user_type([int])
    assert result == (int,)


# LLM-generated content at query #9
#--------------------------

def test_wrap_invariant():
    # Test with single boolean result
    def invariant_single(x):
        return (True, "valid")
    
    wrapped_single = wrap_invariant(invariant_single)
    verdict, data = wrapped_single(5)
    assert verdict is True
    assert data == "valid"
    
    # Test with single boolean result - failure case
    def invariant_single_fail(x):
        return (False, "invalid")
    
    wrapped_single_fail = wrap_invariant(invariant_single_fail)
    verdict, data = wrapped_single_fail(5)
    assert verdict is False
    assert data == "invalid"
    
    # Test with multiple boolean results - all pass
    def invariant_multi_pass(x):
        return [
            (True, "check1"),
            (True, "check2"),
            (True, "check3")
        ]
    
    wrapped_multi_pass = wrap_invariant(invariant_multi_pass)
    verdict, data = wrapped_multi_pass(5)
    assert verdict is True
    assert data == ("check1", "check2", "check3")
    
    # Test with multiple boolean results - some fail
    def invariant_multi_fail(x):
        return [
            (True, "check1"),
            (False, "check2"),
            (True, "check3"),
            (False, "check4")
        ]
    
    wrapped_multi_fail = wrap_invariant(invariant_multi_fail)
    verdict, data = wrapped_multi_fail(5)
    assert verdict is False
    assert data == ("check2", "check4")
    
    # Test with multiple boolean results - all fail
    def invariant_multi_all_fail(x):
        return [
            (False, "error1"),
            (False, "error2")
        ]
    
    wrapped_multi_all_fail = wrap_invariant(invariant_multi_all_fail)
    verdict, data = wrapped_multi_all_fail(5)
    assert verdict is False
    assert data == ("error1", "error2")
    
    # Test with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x must be greater than y")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    verdict, data = wrapped_kwargs(15, y=10)
    assert verdict is True
    assert data == "x must be greater than y"
    
    verdict, data = wrapped_kwargs(5, y=10)
    assert verdict is False
    assert data == "x must be greater than y"


# LLM-generated content at query #10
#--------------------------

```python
def test_CheckedPMap_serialize():
    class IntToFloatMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float

    # Test basic serialization
    m = IntToFloatMap({1: 1.5, 2: 2.25})
    result = m.serialize()
    assert result == {1: 1.5, 2: 2.25}
    assert isinstance(result, dict)

    # Test empty map serialization
    empty_map = IntToFloatMap({})
    assert empty_map.serialize() == {}

    # Test with custom serializer
    class CustomSerializerMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

        def __serializer__(self, format, key, value):
            return str(key), value.upper()

    m2 = CustomSerializerMap({1: "hello", 2: "world"})
    result2 = m2.serialize()
    assert result2 == {"1": "HELLO", "2": "WORLD"}

    # Test serialization with CheckedType values
    class PositivesVector(CheckedPVector):
        __type__ = (int, float)

    class MapWithCheckedValues(CheckedPMap):
        __key_type__ = str
        __value_type__ = PositivesVector

    vec = PositivesVector([1, 2, 3])
    m3 = MapWithCheckedValues({"nums": vec})
    result3 = m3.serialize()
    assert result3 == {"nums": [1, 2, 3]}

    # Test serialization with format parameter
    m4 = IntToFloatMap({1: 1.5})
    result4 = m4.serialize(format="json")
    assert result4 == {1: 1.5}

    # Test with multiple entries
    m5 = IntToFloatMap({1: 1.1, 2: 2.2, 3: 3.3, 4: 4.4})
    result5 = m5.serialize()
    assert len(result5) == 4
    assert result5[1] == 1.1
    assert result5[4] == 4.4


# LLM-generated content at query #11
#--------------------------

```python
def test_get_type():
    # Test with actual type object
    assert get_type(int) is int
    assert get_type(str) is str
    assert get_type(list) is list
    assert get_type(dict) is dict
    
    # Test with string type reference
    assert get_type('builtins.int') is int
    assert get_type('builtins.str') is str
    assert get_type('builtins.list') is list
    assert get_type('builtins.dict') is dict
    
    # Test with custom classes
    assert get_type(CheckedType) is CheckedType
    assert get_type(InvariantException) is InvariantException
    assert get_type('enum.Enum') is Enum
    
    # Test that string references resolve to the correct type
    resolved_type = get_type('collections.abc.Iterable')
    assert resolved_type is Iterable


# LLM-generated content at query #12
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Single boolean result
    def invariant_single_bool(x):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single_bool)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test case 2: Single boolean result with failure
    def invariant_single_bool_fail(x):
        return (False, "error")
    
    wrapped = wrap_invariant(invariant_single_bool_fail)
    result = wrapped(5)
    assert result == (False, "error")
    
    # Test case 3: Multiple test results to merge - all pass
    def invariant_multiple_pass(x):
        return [
            (True, "test1_pass"),
            (True, "test2_pass"),
            (True, "test3_pass")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_pass)
    result = wrapped(5)
    assert result[0] is True
    assert result[1] == ()
    
    # Test case 4: Multiple test results to merge - some fail
    def invariant_multiple_mixed(x):
        return [
            (True, "test1_pass"),
            (False, "test2_fail"),
            (True, "test3_pass"),
            (False, "test4_fail")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_mixed)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("test2_fail", "test4_fail")
    
    # Test case 5: Multiple test results to merge - all fail
    def invariant_multiple_fail(x):
        return [
            (False, "test1_fail"),
            (False, "test2_fail")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_fail)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("test1_fail", "test2_fail")
    
    # Test case 6: Invariant with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x greater than y")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(15, y=10)
    assert result == (True, "x greater than y")
    
    result = wrapped(5, y=10)
    assert result == (False, "x greater than y")
    
    # Test case 7: Empty results list
    def invariant_empty_results(x):
        return []
    
    wrapped = wrap_invariant(invariant_empty_results)
    result = wrapped(5)
    assert result[0] is True
    assert result[1] == ()


# LLM-generated content at query #13
#--------------------------

def test_maybe_parse_user_type():
    # Test with a single type
    result = maybe_parse_user_type(int)
    assert result == [int]
    
    # Test with a string
    result = maybe_parse_user_type('int')
    assert result == ['int']
    
    # Test with Enum (preserved iterable type)
    class TestEnum(Enum):
        A = 1
        B = 2
    
    result = maybe_parse_user_type(TestEnum)
    assert result == [TestEnum]
    
    # Test with a list of types
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)
    
    # Test with a tuple of types
    result = maybe_parse_user_type((int, float, bool))
    assert result == (int, float, bool)
    
    # Test with nested iterables
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)
    
    # Test with mixed types and strings
    result = maybe_parse_user_type([int, 'str', float])
    assert result == (int, 'str', float)
    
    # Test with deeply nested iterables
    result = maybe_parse_user_type([int, [str, [float, bool]]])
    assert result == (int, str, float, bool)
    
    # Test with invalid input (non-type, non-string, non-iterable)
    import pytest
    with pytest.raises(TypeError) as exc_info:
        maybe_parse_user_type(123)
    assert 'Type specifications must be types or strings' in str(exc_info.value)
    
    # Test with invalid input (None)
    with pytest.raises(TypeError) as exc_info:
        maybe_parse_user_type(None)
    assert 'Type specifications must be types or strings' in str(exc_info.value)
    
    # Test with empty list
    result = maybe_parse_user_type([])
    assert result == ()
    
    # Test with list containing Enum
    result = maybe_parse_user_type([int, TestEnum])
    assert result == (int, TestEnum)


# LLM-generated content at query #14
#--------------------------

```python
def test_get_type():
    # Test with actual type object
    assert get_type(int) == int
    assert get_type(str) == str
    assert get_type(list) == list
    assert get_type(dict) == dict
    
    # Test with string type reference
    assert get_type('builtins.int') == int
    assert get_type('builtins.str') == str
    assert get_type('builtins.list') == list
    assert get_type('builtins.dict') == dict
    
    # Test with custom classes
    assert get_type(CheckedType) == CheckedType
    assert get_type('abc.ABCMeta') == ABCMeta
    
    # Test with string reference to class in this module
    assert get_type('collections.abc.Iterable') == Iterable


# LLM-generated content at query #15
#--------------------------

```python
def test_get_type():
    # Test with a type object - should return the type itself
    assert get_type(int) is int
    assert get_type(str) is str
    assert get_type(list) is list
    assert get_type(CheckedType) is CheckedType
    
    # Test with a string representing a built-in type
    assert get_type('builtins.int') is int
    assert get_type('builtins.str') is str
    assert get_type('builtins.dict') is dict
    
    # Test with a string representing a class from this module
    assert get_type('pyrsistent._pmap.PMap') is PMap
    assert get_type('pyrsistent._pset.PSet') is PSet
    assert get_type('pyrsistent._pvector.PythonPVector') is PythonPVector
    
    # Test with exception classes
    assert get_type(InvariantException) is InvariantException
    assert get_type(CheckedTypeError) is CheckedTypeError
    
    # Test with string representation of exception class
    assert get_type('collections.abc.Iterable') is Iterable


# LLM-generated content at query #16
#--------------------------

```python
def test_CheckedPMap___new__():
    """Test CheckedPMap.__new__ method"""
    
    # Test 1: Create CheckedPMap with empty initial dict
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = SimpleMap()
    assert isinstance(result, SimpleMap)
    assert len(result) == 0
    
    # Test 2: Create CheckedPMap with initial data
    initial_data = {1: 'a', 2: 'b', 3: 'c'}
    result = SimpleMap(initial_data)
    assert isinstance(result, SimpleMap)
    assert len(result) == 3
    assert result[1] == 'a'
    assert result[2] == 'b'
    assert result[3] == 'c'
    
    # Test 3: Create CheckedPMap with size parameter
    class SizedMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    data = {'x': 10, 'y': 20}
    result = SizedMap(data, size=2)
    assert isinstance(result, SizedMap)
    assert len(result) == 2
    
    # Test 4: Create CheckedPMap with type checking
    class TypedMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = TypedMap({1: 1.5, 2: 2.5})
    assert result[1] == 1.5
    assert result[2] == 2.5
    
    # Test 5: Create CheckedPMap with invalid type should raise error
    class StrictMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    with pytest.raises(CheckedValueTypeError):
        StrictMap({1: 123})  # 123 is int, not str
    
    # Test 6: Create CheckedPMap with invalid key type should raise error
    with pytest.raises(CheckedKeyTypeError):
        StrictMap({'key': 'value'})  # 'key' is str, not int
    
    # Test 7: Create CheckedPMap with multiple key-value pairs
    class MultiMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    data = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    result = MultiMap(data)
    assert len(result) == 5
    for k, v in data.items():
        assert result[k] == v
    
    # Test 8: Create CheckedPMap with no explicit size parameter
    class DefaultMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = DefaultMap({10: 'ten', 20: 'twenty'})
    assert result[10] == 'ten'
    assert result[20] == 'twenty'


# LLM-generated content at query #17
#--------------------------

def test_store_invariants():
    # Test basic invariant storage
    def invariant1(x):
        return True, "valid"
    
    def invariant2(x):
        return True, "valid"
    
    dct = {'_invariants': invariant1}
    bases = []
    
    store_invariants(dct, bases, '_stored_invariants', '_invariants')
    
    assert '_stored_invariants' in dct
    assert len(dct['_stored_invariants']) == 1
    assert callable(dct['_stored_invariants'][0])


def test_store_invariants_inherited():
    # Test invariant inheritance from base classes
    def base_invariant(x):
        return True, "base_valid"
    
    def derived_invariant(x):
        return True, "derived_valid"
    
    base_dict = {'_invariants': base_invariant}
    
    class BaseClass:
        __dict__ = base_dict
        __bases__ = ()
    
    dct = {'_invariants': derived_invariant}
    bases = [BaseClass]
    
    store_invariants(dct, bases, '_stored_invariants', '_invariants')
    
    assert '_stored_invariants' in dct
    assert len(dct['_stored_invariants']) == 2


def test_store_invariants_non_callable_raises():
    # Test that non-callable invariants raise TypeError
    dct = {'_invariants': "not_callable"}
    bases = []
    
    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants(dct, bases, '_stored_invariants', '_invariants')


def test_store_invariants_no_invariants():
    # Test with no invariants defined
    dct = {}
    bases = []
    
    store_invariants(dct, bases, '_stored_invariants', '_invariants')
    
    assert '_stored_invariants' in dct
    assert dct['_stored_invariants'] == ()


def test_store_invariants_multiple_inheritance():
    # Test invariants with multiple inheritance
    def inv1(x):
        return True, "inv1"
    
    def inv2(x):
        return True, "inv2"
    
    def inv3(x):
        return True, "inv3"
    
    class Base1:
        __dict__ = {'_invariants': inv1}
        __bases__ = ()
    
    class Base2:
        __dict__ = {'_invariants': inv2}
        __bases__ = ()
    
    dct = {'_invariants': inv3}
    bases = [Base1, Base2]
    
    store_invariants(dct, bases, '_stored_invariants', '_invariants')
    
    assert '_stored_invariants' in dct
    assert len(dct['_stored_invariants']) >= 2


def test_store_invariants_wraps_invariants():
    # Test that invariants are wrapped
    def invariant(x):
        return True, "valid"
    
    dct = {'_invariants': invariant}
    bases = []
    
    store_invariants(dct, bases, '_stored_invariants', '_invariants')
    
    stored = dct['_stored_invariants'][0]
    result = stored(None)
    assert result[0] is True
    assert result[1] == "valid"


# LLM-generated content at query #18
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Invariant that returns a single (bool, data) tuple
    def simple_invariant(x):
        return (x > 0, "x must be positive")
    
    wrapped = wrap_invariant(simple_invariant)
    verdict, data = wrapped(5)
    assert verdict is True
    assert data == "x must be positive"
    
    # Test case 2: Invariant that returns False
    verdict, data = wrapped(-5)
    assert verdict is False
    assert data == "x must be positive"
    
    # Test case 3: Invariant that returns multiple results to merge
    def multi_invariant(x, y):
        return [
            (x > 0, "x must be positive"),
            (y > 0, "y must be positive"),
            (x + y < 100, "sum must be less than 100")
        ]
    
    wrapped_multi = wrap_invariant(multi_invariant)
    
    # All checks pass
    verdict, data = wrapped_multi(10, 20)
    assert verdict is True
    assert data == ()
    
    # Some checks fail
    verdict, data = wrapped_multi(-10, 20)
    assert verdict is False
    assert data == ("x must be positive",)
    
    # Multiple checks fail
    verdict, data = wrapped_multi(-10, -20)
    assert verdict is False
    assert ("x must be positive",) == data or ("y must be positive",) in data
    assert len(data) >= 1
    
    # Test case 4: Invariant with all failures
    verdict, data = wrapped_multi(-5, -5)
    assert verdict is False
    assert len(data) >= 2
    
    # Test case 5: Invariant that returns tuple with non-bool first element triggers merge
    def complex_invariant():
        return [
            (True, "check1"),
            (False, "check2"),
            (True, "check3")
        ]
    
    wrapped_complex = wrap_invariant(complex_invariant)
    verdict, data = wrapped_complex()
    assert verdict is False
    assert data == ("check2",)
    
    # Test case 6: All passing complex invariant
    def all_pass_invariant():
        return [
            (True, "check1"),
            (True, "check2"),
            (True, "check3")
        ]
    
    wrapped_pass = wrap_invariant(all_pass_invariant)
    verdict, data = wrapped_pass()
    assert verdict is True
    assert data == ()


# LLM-generated content at query #19
#--------------------------

def test_wrap_invariant():
    # Test with single boolean result
    def invariant_single(x):
        return (True, "valid")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "valid")
    
    # Test with single boolean False result
    def invariant_single_fail(x):
        return (False, "invalid")
    
    wrapped_single_fail = wrap_invariant(invariant_single_fail)
    result = wrapped_single_fail(5)
    assert result == (False, "invalid")
    
    # Test with multiple results to merge
    def invariant_multiple(x):
        return [
            (True, "check1"),
            (True, "check2"),
            (False, "check3"),
        ]
    
    wrapped_multiple = wrap_invariant(invariant_multiple)
    result = wrapped_multiple(5)
    assert result[0] is False
    assert result[1] == ("check3",)
    
    # Test with all passing multiple results
    def invariant_multiple_pass(x):
        return [
            (True, "check1"),
            (True, "check2"),
            (True, "check3"),
        ]
    
    wrapped_multiple_pass = wrap_invariant(invariant_multiple_pass)
    result = wrapped_multiple_pass(5)
    assert result[0] is True
    assert result[1] == ()
    
    # Test with mixed results - multiple failures
    def invariant_multiple_fails(x):
        return [
            (False, "error1"),
            (True, "check2"),
            (False, "error2"),
        ]
    
    wrapped_multiple_fails = wrap_invariant(invariant_multiple_fails)
    result = wrapped_multiple_fails(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x must be greater than y")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_kwargs(15, y=5)
    assert result == (True, "x must be greater than y")
    
    result = wrapped_kwargs(3, y=5)
    assert result == (False, "x must be greater than y")


# LLM-generated content at query #20
#--------------------------

```python
def test_store_invariants():
    # Test basic invariant storage
    dct = {}
    bases = []
    
    def invariant1(obj):
        return True, "invariant1_passed"
    
    dct['__invariants__'] = invariant1
    store_invariants(dct, bases, '_invariants', '__invariants__')
    
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 1
    assert callable(dct['_invariants'][0])


def test_store_invariants_inherited():
    # Test invariant inheritance from base classes
    def invariant1(obj):
        return True, "invariant1"
    
    def invariant2(obj):
        return True, "invariant2"
    
    base_dct = {'__invariants__': invariant1}
    base_class = type('BaseClass', (), base_dct)
    
    dct = {'__invariants__': invariant2}
    store_invariants(dct, [base_class], '_invariants', '__invariants__')
    
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 2


def test_store_invariants_no_invariants():
    # Test when no invariants are defined
    dct = {}
    bases = []
    
    store_invariants(dct, bases, '_invariants', '__invariants__')
    
    assert '_invariants' in dct
    assert dct['_invariants'] == ()


def test_store_invariants_non_callable_raises():
    # Test that non-callable invariants raise TypeError
    dct = {'__invariants__': "not_callable"}
    bases = []
    
    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants(dct, bases, '_invariants', '__invariants__')


def test_store_invariants_multiple_results():
    # Test invariant that returns multiple results
    def invariant_multi(obj):
        return [(True, "test1"), (False, "test2")]
    
    dct = {'__invariants__': invariant_multi}
    bases = []
    
    store_invariants(dct, bases, '_invariants', '__invariants__')
    
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 1
    result = dct['_invariants'][0]({})
    assert result[0] is False
    assert len(result[1]) == 1


def test_store_invariants_wrapped_callable():
    # Test that invariants are wrapped
    def invariant(obj):
        return True, "passed"
    
    dct = {'__invariants__': invariant}
    bases = []
    
    store_invariants(dct, bases, '_invariants', '__invariants__')
    
    wrapped = dct['_invariants'][0]
    assert callable(wrapped)
    result = wrapped({})
    assert result == (True, "passed")


def test_store_invariants_deep_inheritance():
    # Test inheritance chain with multiple levels
    def inv1(obj):
        return True, "inv1"
    
    def inv2(obj):
        return True, "inv2"
    
    def inv3(obj):
        return True, "inv3"
    
    base1_dct = {'__invariants__': inv1}
    base1_class = type('Base1', (), base1_dct)
    
    base2_dct = {'__invariants__': inv2}
    base2_class = type('Base2', (base1_class,), base2_dct)
    
    dct = {'__invariants__': inv3}
    store_invariants(dct, [base2_class], '_invariants', '__invariants__')
    
    assert len(dct['_invariants']) == 3


def test_store_invariants_empty_bases():
    # Test with empty bases list
    def invariant(obj):
        return True, "test"
    
    dct = {'__invariants__': invariant}
    store_invariants(dct, [], '_invariants', '__invariants__')
    
    assert len(dct['_invariants']) == 1


def test_store_invariants_custom_names():
    # Test with custom destination and source names
    def invariant(obj):
        return True, "custom"
    
    dct = {'custom_source': invariant}
    bases = []
    
    store_invariants(dct, bases, 'custom_dest', 'custom_source')
    
    assert 'custom_dest' in dct
    assert len(dct['custom_dest']) == 1


# LLM-generated content at query #21
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test basic initialization with empty dict
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    m = SimpleMap()
    assert len(m) == 0
    assert isinstance(m, CheckedPMap)
    assert isinstance(m, SimpleMap)
    
    # Test initialization with initial data
    m = SimpleMap({1: 10, 2: 20})
    assert len(m) == 2
    assert m[1] == 10
    assert m[2] == 20
    
    # Test initialization with multiple key-value pairs
    m = SimpleMap({1: 100, 2: 200, 3: 300})
    assert len(m) == 3
    assert m[1] == 100
    assert m[2] == 200
    assert m[3] == 300
    
    # Test that result is immutable (PMap behavior)
    m = SimpleMap({1: 10})
    m2 = m.set(2, 20)
    assert len(m) == 1
    assert len(m2) == 2
    assert m[1] == 10
    assert m2[1] == 10
    assert m2[2] == 20
    
    # Test type checking on initialization
    class StrictMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = float
    
    m = StrictMap({'a': 1.5, 'b': 2.5})
    assert m['a'] == 1.5
    assert m['b'] == 2.5
    
    # Test initialization with wrong key type raises error
    try:
        StrictMap({1: 1.5})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass
    
    # Test initialization with wrong value type raises error
    try:
        StrictMap({'a': 'not_a_float'})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass
    
    # Test initialization from existing CheckedPMap returns same instance
    m1 = SimpleMap({1: 10})
    m2 = SimpleMap(m1)
    assert m1 is m2
    
    # Test with invariant
    class InvariantMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > k, 'Value must be greater than key')
    
    m = InvariantMap({1: 10, 2: 20})
    assert m[1] == 10
    assert m[2] == 20
    
    # Test invariant violation on initialization
    try:
        InvariantMap({10: 5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass
    
    # Test optional types
    class OptionalMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = optional(int, str)
    
    m = OptionalMap({1: 10, 2: 'hello'})
    assert m[1] == 10
    assert m[2] == 'hello'
    
    # Test with None value when optional
    m = OptionalMap({1: None})
    assert m[1] is None


# LLM-generated content at query #22
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test basic initialization with empty dict
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    m1 = SimpleMap()
    assert len(m1) == 0
    assert isinstance(m1, CheckedPMap)
    assert isinstance(m1, SimpleMap)
    
    # Test initialization with initial data
    m2 = SimpleMap({1: 'a', 2: 'b'})
    assert len(m2) == 2
    assert m2[1] == 'a'
    assert m2[2] == 'b'
    
    # Test initialization with multiple items
    m3 = SimpleMap({10: 'x', 20: 'y', 30: 'z'})
    assert len(m3) == 3
    assert m3[10] == 'x'
    assert m3[20] == 'y'
    assert m3[30] == 'z'
    
    # Test that invalid key type raises error
    with pytest.raises(CheckedKeyTypeError):
        SimpleMap({'invalid': 'value'})
    
    # Test that invalid value type raises error
    with pytest.raises(CheckedValueTypeError):
        SimpleMap({1: 123})
    
    # Test with type checking disabled (empty type specs)
    class NoTypeCheckMap(CheckedPMap):
        pass
    
    m4 = NoTypeCheckMap({1: 'a', 'key': 123})
    assert len(m4) == 2
    assert m4[1] == 'a'
    assert m4['key'] == 123
    
    # Test initialization preserves immutability
    m5 = SimpleMap({1: 'a'})
    m5_modified = m5.set(2, 'b')
    assert len(m5) == 1
    assert len(m5_modified) == 2
    assert m5 is not m5_modified
    
    # Test with invariant
    class ValidatedMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (k > 0 and v > 0, 'Both key and value must be positive')
    
    m6 = ValidatedMap({1: 1, 2: 2})
    assert len(m6) == 2
    
    with pytest.raises(InvariantException):
        ValidatedMap({-1: 1})
    
    with pytest.raises(InvariantException):
        ValidatedMap({1: -1})
    
    # Test that size parameter works correctly
    m7 = SimpleMap.__new__(SimpleMap, {1: 'a', 2: 'b'}, size=2)
    assert isinstance(m7, SimpleMap)


# LLM-generated content at query #23
#--------------------------

```python
import pytest


def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean result
    def simple_invariant(x):
        return (True, "valid")
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "valid")
    
    # Test with a simple invariant that returns False
    def failing_invariant(x):
        return (False, "invalid")
    
    wrapped = wrap_invariant(failing_invariant)
    result = wrapped(5)
    assert result == (False, "invalid")
    
    # Test with an invariant that returns multiple results to be merged
    def multi_result_invariant(x):
        return [
            (True, "check1_passed"),
            (True, "check2_passed"),
            (False, "check3_failed")
        ]
    
    wrapped = wrap_invariant(multi_result_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert "check3_failed" in result[1]
    assert "check1_passed" in result[1]
    assert "check2_passed" in result[1]
    
    # Test with all passing multiple results
    def all_passing_invariant(x):
        return [
            (True, "check1_passed"),
            (True, "check2_passed")
        ]
    
    wrapped = wrap_invariant(all_passing_invariant)
    result = wrapped(5)
    assert result[0] is True
    assert result[1] == ("check1_passed", "check2_passed")
    
    # Test with all failing multiple results
    def all_failing_invariant(x):
        return [
            (False, "error1"),
            (False, "error2"),
            (False, "error3")
        ]
    
    wrapped = wrap_invariant(all_failing_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert len(result[1]) == 3
    assert "error1" in result[1]
    assert "error2" in result[1]
    assert "error3" in result[1]
    
    # Test that wrapped function accepts keyword arguments
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x_greater_than_y")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(5, y=3)
    assert result == (True, "x_greater_than_y")
    
    result = wrapped(2, y=5)
    assert result == (False, "x_greater_than_y")


# LLM-generated content at query #24
#--------------------------

def test_wrap_invariant():
    # Test case 1: Invariant returns a single boolean result
    def invariant_single_bool(value):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single_bool)
    result = wrapped(42)
    assert result == (True, "success")
    
    # Test case 2: Invariant returns multiple results that need merging
    def invariant_multiple_results(value):
        return (
            (True, "check1_passed"),
            (False, "check2_failed"),
            (True, "check3_passed"),
        )
    
    wrapped = wrap_invariant(invariant_multiple_results)
    result = wrapped(42)
    assert result[0] is False
    assert result[1] == ("check2_failed",)
    
    # Test case 3: All multiple results pass
    def invariant_all_pass(value):
        return (
            (True, "check1"),
            (True, "check2"),
            (True, "check3"),
        )
    
    wrapped = wrap_invariant(invariant_all_pass)
    result = wrapped(42)
    assert result[0] is True
    assert result[1] == ()
    
    # Test case 4: Multiple results with multiple failures
    def invariant_multiple_failures(value):
        return (
            (False, "error1"),
            (True, "ok"),
            (False, "error2"),
        )
    
    wrapped = wrap_invariant(invariant_multiple_failures)
    result = wrapped(42)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test case 5: Invariant with kwargs
    def invariant_with_kwargs(value, threshold=10):
        return (value > threshold, "value_check")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(15, threshold=10)
    assert result == (True, "value_check")
    
    # Test case 6: Invariant with False result
    def invariant_false_result(value):
        return (False, "failed")
    
    wrapped = wrap_invariant(invariant_false_result)
    result = wrapped(42)
    assert result == (False, "failed")


# LLM-generated content at query #25
#--------------------------

```python
def test_store_invariants():
    # Test 1: Basic invariant storage
    dct = {}
    bases = []
    
    def invariant1(obj):
        return True, "valid"
    
    dct['__invariants__'] = invariant1
    store_invariants(dct, bases, '_invariants', '__invariants__')
    
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 1
    assert callable(dct['_invariants'][0])
    
    # Test 2: Invariants inherited from base classes
    dct = {}
    
    def base_invariant(obj):
        return True, "base_valid"
    
    class BaseClass:
        __dict__ = {'__invariants__': base_invariant}
    
    def derived_invariant(obj):
        return True, "derived_valid"
    
    dct['__invariants__'] = derived_invariant
    store_invariants(dct, [BaseClass], '_invariants', '__invariants__')
    
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 2
    
    # Test 3: No invariants defined
    dct = {}
    bases = []
    store_invariants(dct, bases, '_invariants', '__invariants__')
    
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 0
    
    # Test 4: Non-callable invariant raises TypeError
    dct = {'__invariants__': "not_callable"}
    bases = []
    
    try:
        store_invariants(dct, bases, '_invariants', '__invariants__')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert 'Invariants must be callable' in str(e)
    
    # Test 5: Multiple invariants from multiple base classes
    dct = {}
    
    def inv1(obj):
        return True, "inv1"
    
    def inv2(obj):
        return True, "inv2"
    
    class Base1:
        __dict__ = {'__invariants__': inv1}
    
    class Base2:
        __dict__ = {'__invariants__': inv2}
    
    def inv3(obj):
        return True, "inv3"
    
    dct['__invariants__'] = inv3
    store_invariants(dct, [Base1, Base2], '_invariants', '__invariants__')
    
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 3
    
    # Test 6: Invariants are wrapped
    dct = {}
    
    def test_invariant(obj):
        return True, "test"
    
    dct['__invariants__'] = test_invariant
    store_invariants(dct, [], '_invariants', '__invariants__')
    
    wrapped = dct['_invariants'][0]
    result = wrapped(None)
    assert result == (True, "test")
    
    # Test 7: Wrapped invariant handles multiple results
    dct = {}
    
    def multi_result_invariant(obj):
        return [(True, "test1"), (False, "test2")]
    
    dct['__invariants__'] = multi_result_invariant
    store_invariants(dct, [], '_invariants', '__invariants__')
    
    wrapped = dct['_invariants'][0]
    result = wrapped(None)
    assert result[0] is False
    assert len(result[1]) == 1


# LLM-generated content at query #26
#--------------------------

def test_maybe_parse_user_type():
    # Test with a single type
    result = maybe_parse_user_type(int)
    assert result == [int]
    
    result = maybe_parse_user_type(str)
    assert result == [str]
    
    # Test with a string
    result = maybe_parse_user_type("int")
    assert result == ["int"]
    
    result = maybe_parse_user_type("CustomType")
    assert result == ["CustomType"]
    
    # Test with Enum (preserved iterable type)
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    
    result = maybe_parse_user_type(Color)
    assert result == [Color]
    
    # Test with a list of types
    result = maybe_parse_user_type([int, str])
    assert set(result) == {int, str}
    
    result = maybe_parse_user_type([int, str, float])
    assert set(result) == {int, str, float}
    
    # Test with a tuple of types
    result = maybe_parse_user_type((int, str))
    assert set(result) == {int, str}
    
    # Test with nested iterables
    result = maybe_parse_user_type([int, [str, float]])
    assert set(result) == {int, str, float}
    
    result = maybe_parse_user_type((int, (str, float)))
    assert set(result) == {int, str, float}
    
    # Test with mixed strings and types in iterable
    result = maybe_parse_user_type([int, "CustomType"])
    assert set(result) == {int, "CustomType"}
    
    # Test with deeply nested iterables
    result = maybe_parse_user_type([int, [str, [float, bool]]])
    assert set(result) == {int, str, float, bool}
    
    # Test error cases
    import pytest
    
    # Invalid type - neither type nor string nor iterable
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(3.14)
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)
    
    # Invalid type in iterable
    with pytest.raises(TypeError):
        maybe_parse_user_type([int, 123])
    
    with pytest.raises(TypeError):
        maybe_parse_user_type([int, [str, 3.14]])


# LLM-generated content at query #27
#--------------------------

def test_wrap_invariant():
    # Test with single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "success")
    
    # Test with single False result
    def invariant_false(x):
        return (False, "error")
    
    wrapped_false = wrap_invariant(invariant_false)
    result = wrapped_false(5)
    assert result == (False, "error")
    
    # Test with multiple results that need merging
    def invariant_multiple(x):
        return [
            (True, "check1_passed"),
            (False, "check2_failed"),
            (True, "check3_passed"),
            (False, "check4_failed")
        ]
    
    wrapped_multiple = wrap_invariant(invariant_multiple)
    result = wrapped_multiple(5)
    assert result[0] is False
    assert result[1] == ("check2_failed", "check4_failed")
    
    # Test with all passing multiple results
    def invariant_all_pass(x):
        return [
            (True, "check1"),
            (True, "check2"),
            (True, "check3")
        ]
    
    wrapped_all_pass = wrap_invariant(invariant_all_pass)
    result = wrapped_all_pass(5)
    assert result[0] is True
    assert result[1] == ()
    
    # Test with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x must be greater than y")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_kwargs(15, y=10)
    assert result == (True, "x must be greater than y")
    
    result = wrapped_kwargs(5, y=10)
    assert result == (False, "x must be greater than y")


# LLM-generated content at query #28
#--------------------------

```python
def test_wrap_invariant():
    # Test with a single boolean result
    def invariant_single_bool(x):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single_bool)
    result = wrapped(42)
    assert result == (True, "success")
    
    # Test with a single boolean result (False case)
    def invariant_single_bool_false(x):
        return (False, "failure")
    
    wrapped = wrap_invariant(invariant_single_bool_false)
    result = wrapped(42)
    assert result == (False, "failure")
    
    # Test with multiple test results that need merging
    def invariant_multiple_results(x):
        return [
            (True, "test1_passed"),
            (False, "test2_failed"),
            (True, "test3_passed"),
            (False, "test4_failed"),
        ]
    
    wrapped = wrap_invariant(invariant_multiple_results)
    result = wrapped(42)
    assert result[0] is False  # Overall verdict should be False
    assert result[1] == ("test2_failed", "test4_failed")  # Only failures
    
    # Test with all passing multiple results
    def invariant_multiple_all_pass(x):
        return [
            (True, "test1_passed"),
            (True, "test2_passed"),
        ]
    
    wrapped = wrap_invariant(invariant_multiple_all_pass)
    result = wrapped(42)
    assert result[0] is True
    assert result[1] == ()
    
    # Test with all failing multiple results
    def invariant_multiple_all_fail(x):
        return [
            (False, "test1_failed"),
            (False, "test2_failed"),
        ]
    
    wrapped = wrap_invariant(invariant_multiple_all_fail)
    result = wrapped(42)
    assert result[0] is False
    assert result[1] == ("test1_failed", "test2_failed")
    
    # Test that wrapped function preserves function arguments
    def invariant_with_kwargs(x, y=10):
        return (x + y > 0, f"sum_{x + y}")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(5, y=15)
    assert result == (True, "sum_20")


# LLM-generated content at query #29
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test with empty initial dict
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = SimpleMap()
    assert isinstance(result, CheckedPMap)
    assert len(result) == 0
    
    # Test with initial dict
    initial_data = {1: "a", 2: "b", 3: "c"}
    result = SimpleMap(initial_data)
    assert len(result) == 3
    assert result[1] == "a"
    assert result[2] == "b"
    assert result[3] == "c"
    
    # Test with explicit size parameter
    pmap_instance = pmap({4: "d", 5: "e"})
    result = SimpleMap(pmap_instance, size=2)
    assert isinstance(result, CheckedPMap)
    assert len(result) == 2
    
    # Test type checking during initialization
    class StrictMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = float
    
    result = StrictMap({1: 1.5, 2: 2.5})
    assert result[1] == 1.5
    assert result[2] == 2.5
    
    # Test that invalid types raise CheckedKeyTypeError or CheckedValueTypeError
    with pytest.raises(CheckedKeyTypeError):
        StrictMap({"invalid": 1.5})
    
    with pytest.raises(CheckedValueTypeError):
        StrictMap({1: "invalid"})
    
    # Test with invariant
    class MapWithInvariant(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, 'Value must be positive')
    
    result = MapWithInvariant({1: 10, 2: 20})
    assert result[1] == 10
    
    with pytest.raises(InvariantException):
        MapWithInvariant({1: -5})
    
    # Test with multiple key/value types
    class MultiTypeMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = (int, float)
    
    result = MultiTypeMap({1: 1.5, "key": 10})
    assert result[1] == 1.5
    assert result["key"] == 10


# LLM-generated content at query #30
#--------------------------

```python
def test_store_invariants():
    # Test basic invariant storage
    dct = {'__invariants__': lambda self: (True, None)}
    bases = []
    store_invariants(dct, bases, '__invariants__', '__invariants__')
    assert '__invariants__' in dct
    assert len(dct['__invariants__']) == 1
    assert callable(dct['__invariants__'][0])

    # Test invariant inheritance from base classes
    base_invariant = lambda self: (True, None)
    base_dct = {'__invariants__': base_invariant}
    
    class BaseClass:
        pass
    BaseClass.__dict__ = base_dct
    
    child_dct = {}
    child_invariant = lambda self: (True, None)
    child_dct['__invariants__'] = child_invariant
    
    store_invariants(child_dct, [BaseClass], '__invariants__', '__invariants__')
    assert '__invariants__' in child_dct
    assert len(child_dct['__invariants__']) >= 1

    # Test with no invariants
    empty_dct = {}
    empty_bases = []
    store_invariants(empty_dct, empty_bases, '__invariants__', '__invariants__')
    assert '__invariants__' in empty_dct
    assert empty_dct['__invariants__'] == ()

    # Test with multiple invariants in inheritance chain
    class Base1:
        pass
    Base1.__dict__ = {'__invariants__': lambda self: (True, None)}
    
    class Base2:
        pass
    Base2.__dict__ = {'__invariants__': lambda self: (True, None)}
    
    multi_dct = {'__invariants__': lambda self: (True, None)}
    store_invariants(multi_dct, [Base1, Base2], '__invariants__', '__invariants__')
    assert '__invariants__' in multi_dct
    assert len(multi_dct['__invariants__']) >= 1

    # Test that non-callable invariants raise TypeError
    import pytest
    bad_dct = {'__invariants__': 'not_callable'}
    bad_bases = []
    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants(bad_dct, bad_bases, '__invariants__', '__invariants__')

    # Test with multiple invariants in same dict
    inv1 = lambda self: (True, None)
    inv2 = lambda self: (True, None)
    multi_inv_dct = {}
    multi_inv_bases = []
    
    class MultiBase:
        pass
    MultiBase.__dict__ = {'__invariants__': inv1}
    
    multi_inv_dct['__invariants__'] = inv2
    store_invariants(multi_inv_dct, [MultiBase], '__invariants__', '__invariants__')
    assert len(multi_inv_dct['__invariants__']) >= 1
    assert all(callable(inv) for inv in multi_inv_dct['__invariants__'])

    # Test that invariants are wrapped
    def test_invariant(self):
        return (True, None)
    
    wrap_dct = {'__invariants__': test_invariant}
    wrap_bases = []
    store_invariants(wrap_dct, wrap_bases, '__invariants__', '__invariants__')
    # The invariant should be wrapped by wrap_invariant
    assert len(wrap_dct['__invariants__']) == 1
    assert callable(wrap_dct['__invariants__'][0])


# LLM-generated content at query #31
#--------------------------

def test_wrap_invariant():
    # Test case 1: Single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test case 2: Single boolean result with failure
    def invariant_single_fail(x):
        return (False, "failed")
    
    wrapped = wrap_invariant(invariant_single_fail)
    result = wrapped(5)
    assert result == (False, "failed")
    
    # Test case 3: Multiple results that need merging - all pass
    def invariant_multiple_pass(x):
        return [
            (True, "check1"),
            (True, "check2"),
            (True, "check3")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_pass)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test case 4: Multiple results with one failure
    def invariant_multiple_fail(x):
        return [
            (True, "check1"),
            (False, "check2_failed"),
            (True, "check3")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_fail)
    result = wrapped(5)
    assert result == (False, ("check2_failed",))
    
    # Test case 5: Multiple results with multiple failures
    def invariant_multiple_failures(x):
        return [
            (False, "error1"),
            (False, "error2"),
            (True, "check3")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_failures)
    result = wrapped(5)
    assert result == (False, ("error1", "error2"))
    
    # Test case 6: With kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x must be greater than y")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(15, y=10)
    assert result == (True, "x must be greater than y")
    
    result = wrapped(5, y=10)
    assert result == (False, "x must be greater than y")
    
    # Test case 7: Empty result list
    def invariant_empty(x):
        return []
    
    wrapped = wrap_invariant(invariant_empty)
    result = wrapped(5)
    assert result == (True, ())


# LLM-generated content at query #32
#--------------------------

def test_wrap_invariant():
    # Test with single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "success")
    
    # Test with single boolean False result
    def invariant_single_fail(x):
        return (False, "failure")
    
    wrapped_single_fail = wrap_invariant(invariant_single_fail)
    result = wrapped_single_fail(5)
    assert result == (False, "failure")
    
    # Test with multiple results that need merging
    def invariant_multiple(x):
        return [
            (True, "check1_passed"),
            (True, "check2_passed"),
            (False, "check3_failed")
        ]
    
    wrapped_multiple = wrap_invariant(invariant_multiple)
    result = wrapped_multiple(5)
    assert result[0] is False
    assert result[1] == ("check3_failed",)
    
    # Test with all passing multiple results
    def invariant_multiple_pass(x):
        return [
            (True, "check1_passed"),
            (True, "check2_passed"),
            (True, "check3_passed")
        ]
    
    wrapped_multiple_pass = wrap_invariant(invariant_multiple_pass)
    result = wrapped_multiple_pass(5)
    assert result[0] is True
    assert result[1] == ()
    
    # Test with multiple failures
    def invariant_multiple_fails(x):
        return [
            (False, "error1"),
            (False, "error2"),
            (True, "check_passed")
        ]
    
    wrapped_multiple_fails = wrap_invariant(invariant_multiple_fails)
    result = wrapped_multiple_fails(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test that wrapped function preserves kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "value_check")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result1 = wrapped_kwargs(5, y=10)
    assert result1 == (False, "value_check")
    
    result2 = wrapped_kwargs(15, y=10)
    assert result2 == (True, "value_check")


# LLM-generated content at query #33
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a single boolean result
    def simple_invariant(x):
        return (x > 0, "x must be positive")
    
    wrapped_simple = wrap_invariant(simple_invariant)
    verdict, data = wrapped_simple(5)
    assert verdict is True
    assert data == "x must be positive"
    
    # Test with a simple invariant that returns False
    verdict, data = wrapped_simple(-5)
    assert verdict is False
    assert data == "x must be positive"
    
    # Test with an invariant that returns multiple test results
    def multi_invariant(x, y):
        return [
            (x > 0, "x must be positive"),
            (y > 0, "y must be positive"),
            (x + y < 100, "sum must be less than 100")
        ]
    
    wrapped_multi = wrap_invariant(multi_invariant)
    
    # All tests pass
    verdict, data = wrapped_multi(5, 10)
    assert verdict is True
    assert data == ("x must be positive", "y must be positive", "x + y < 100")
    
    # One test fails
    verdict, data = wrapped_multi(-5, 10)
    assert verdict is False
    assert "x must be positive" in data
    assert len(data) == 1
    
    # Multiple tests fail
    verdict, data = wrapped_multi(-5, -10)
    assert verdict is False
    assert "x must be positive" in data
    assert "y must be positive" in data
    assert len(data) == 2
    
    # Test with all tests failing
    verdict, data = wrapped_multi(-5, -10)
    assert verdict is False
    assert len(data) == 2
    
    # Test with tuple of results (already in correct format)
    def tuple_invariant(x):
        return (True, "test")
    
    wrapped_tuple = wrap_invariant(tuple_invariant)
    verdict, data = wrapped_tuple(5)
    assert verdict is True
    assert data == "test"
    
    # Test with multiple results where some pass and some fail
    def partial_fail_invariant(x):
        return [
            (True, "passed"),
            (False, "failed"),
            (True, "passed again")
        ]
    
    wrapped_partial = wrap_invariant(partial_fail_invariant)
    verdict, data = wrapped_partial(5)
    assert verdict is False
    assert data == ("failed",)


# LLM-generated content at query #34
#--------------------------

def test_maybe_parse_user_type():
    # Test with a single type
    result = maybe_parse_user_type(int)
    assert result == [int]
    
    # Test with a string
    result = maybe_parse_user_type("int")
    assert result == ["int"]
    
    # Test with an Enum (preserved iterable type)
    class MyEnum(Enum):
        A = 1
        B = 2
    
    result = maybe_parse_user_type(MyEnum)
    assert result == [MyEnum]
    
    # Test with a list of types
    result = maybe_parse_user_type([int, str])
    assert result == (int, str)
    
    # Test with a tuple of types
    result = maybe_parse_user_type((int, str, float))
    assert result == (int, str, float)
    
    # Test with nested iterables
    result = maybe_parse_user_type([int, [str, float]])
    assert result == (int, str, float)
    
    # Test with list containing strings
    result = maybe_parse_user_type(["int", "str"])
    assert result == ("int", "str")
    
    # Test with mixed list of types and strings
    result = maybe_parse_user_type([int, "str", float])
    assert result == (int, "str", float)
    
    # Test with invalid input (non-type, non-string, non-iterable)
    with pytest.raises(TypeError) as exc_info:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(exc_info.value)
    
    # Test with invalid input (dict - iterable but should fail on contained non-types)
    with pytest.raises(TypeError):
        maybe_parse_user_type({1: 2})
    
    # Test with complex nested structure
    result = maybe_parse_user_type([int, (str, [float, "bool"])])
    assert result == (int, str, float, "bool")


# LLM-generated content at query #35
#--------------------------

def test_wrap_invariant():
    # Test case 1: Single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test case 2: Single boolean result with False
    def invariant_single_fail(x):
        return (False, "failure")
    
    wrapped = wrap_invariant(invariant_single_fail)
    result = wrapped(5)
    assert result == (False, "failure")
    
    # Test case 3: Multiple results to merge - all passing
    def invariant_multiple_pass(x):
        return [(True, "check1"), (True, "check2"), (True, "check3")]
    
    wrapped = wrap_invariant(invariant_multiple_pass)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test case 4: Multiple results to merge - some failing
    def invariant_multiple_fail(x):
        return [(True, "check1"), (False, "error1"), (True, "check3"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_multiple_fail)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test case 5: Multiple results to merge - all failing
    def invariant_multiple_all_fail(x):
        return [(False, "fail1"), (False, "fail2")]
    
    wrapped = wrap_invariant(invariant_multiple_all_fail)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("fail1", "fail2")
    
    # Test case 6: Invariant with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, f"x={x}, y={y}")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(5, y=3)
    assert result == (True, "x=5, y=3")
    
    # Test case 7: Single result that is not a boolean tuple
    def invariant_edge_case(x):
        return (True, "data")
    
    wrapped = wrap_invariant(invariant_edge_case)
    result = wrapped(10)
    assert result == (True, "data")


# LLM-generated content at query #36
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: invariant returns single boolean result
    def invariant_single(value):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single)
    result = wrapped(42)
    assert result == (True, "success")
    
    # Test case 2: invariant returns multiple boolean results that need merging
    def invariant_multiple(value):
        return [
            (True, "check1_passed"),
            (False, "check2_failed"),
            (True, "check3_passed"),
            (False, "check4_failed")
        ]
    
    wrapped = wrap_invariant(invariant_multiple)
    result = wrapped(42)
    assert result[0] is False  # verdict should be False since some checks failed
    assert result[1] == ("check2_failed", "check4_failed")  # only failures in data
    
    # Test case 3: invariant returns multiple results all passing
    def invariant_all_pass(value):
        return [
            (True, "check1_passed"),
            (True, "check2_passed"),
            (True, "check3_passed")
        ]
    
    wrapped = wrap_invariant(invariant_all_pass)
    result = wrapped(42)
    assert result[0] is True
    assert result[1] == ()
    
    # Test case 4: invariant returns multiple results all failing
    def invariant_all_fail(value):
        return [
            (False, "error1"),
            (False, "error2")
        ]
    
    wrapped = wrap_invariant(invariant_all_fail)
    result = wrapped(42)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test case 5: invariant with kwargs
    def invariant_with_kwargs(value, threshold=10):
        return (value > threshold, f"value_check_{value}")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(15, threshold=10)
    assert result == (True, f"value_check_15")
    
    # Test case 6: single result (not list) should be returned as-is
    def invariant_false_result(value):
        return (False, "single_error")
    
    wrapped = wrap_invariant(invariant_false_result)
    result = wrapped(42)
    assert result == (False, "single_error")


# LLM-generated content at query #37
#--------------------------

def test_store_invariants():
    # Test basic invariant storage
    dct = {'_invariants': lambda self: (True, None)}
    bases = []
    store_invariants(dct, bases, '_stored_invariants', '_invariants')
    assert '_stored_invariants' in dct
    assert len(dct['_stored_invariants']) == 1
    assert callable(dct['_stored_invariants'][0])

    # Test inherited invariants from base classes
    base_dct = {'_invariants': lambda self: (True, "base")}
    base_class = type('BaseClass', (), base_dct)
    
    dct = {'_invariants': lambda self: (True, "derived")}
    bases = [base_class]
    store_invariants(dct, bases, '_stored_invariants', '_invariants')
    assert '_stored_invariants' in dct
    assert len(dct['_stored_invariants']) == 2

    # Test multiple inherited invariants
    base1_dct = {'_invariants': lambda self: (True, "base1")}
    base1_class = type('Base1Class', (), base1_dct)
    
    base2_dct = {'_invariants': lambda self: (True, "base2")}
    base2_class = type('Base2Class', (), base2_dct)
    
    dct = {}
    bases = [base1_class, base2_class]
    store_invariants(dct, bases, '_stored_invariants', '_invariants')
    assert '_stored_invariants' in dct
    assert len(dct['_stored_invariants']) == 2

    # Test no invariants defined
    dct = {}
    bases = []
    store_invariants(dct, bases, '_stored_invariants', '_invariants')
    assert '_stored_invariants' in dct
    assert len(dct['_stored_invariants']) == 0

    # Test non-callable invariant raises TypeError
    dct = {'_invariants': "not_callable"}
    bases = []
    try:
        store_invariants(dct, bases, '_stored_invariants', '_invariants')
        assert False, "Should raise TypeError"
    except TypeError as e:
        assert 'Invariants must be callable' in str(e)

    # Test wrapped invariants handle merged results
    def multi_result_invariant(self):
        return [(True, "check1"), (False, "check2")]
    
    dct = {'_invariants': multi_result_invariant}
    bases = []
    store_invariants(dct, bases, '_stored_invariants', '_invariants')
    assert '_stored_invariants' in dct
    wrapped_inv = dct['_stored_invariants'][0]
    result = wrapped_inv(None)
    assert result[0] is False
    assert "check2" in result[1]

    # Test invariant wrapping preserves single boolean results
    def single_result_invariant(self):
        return (True, "passed")
    
    dct = {'_invariants': single_result_invariant}
    bases = []
    store_invariants(dct, bases, '_stored_invariants', '_invariants')
    wrapped_inv = dct['_stored_invariants'][0]
    result = wrapped_inv(None)
    assert result == (True, "passed")


# LLM-generated content at query #38
#--------------------------

```python
def test_wrap_invariant():
    # Test with single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "success")
    
    # Test with multiple results that need merging
    def invariant_multiple(x):
        return ((True, "check1"), (False, "check2"), (True, "check3"))
    
    wrapped_multiple = wrap_invariant(invariant_multiple)
    result = wrapped_multiple(5)
    assert result[0] is False
    assert result[1] == ("check2",)
    
    # Test with all passing multiple results
    def invariant_all_pass(x):
        return ((True, "check1"), (True, "check2"))
    
    wrapped_all_pass = wrap_invariant(invariant_all_pass)
    result = wrapped_all_pass(5)
    assert result[0] is True
    assert result[1] == ()
    
    # Test with all failing multiple results
    def invariant_all_fail(x):
        return ((False, "error1"), (False, "error2"), (False, "error3"))
    
    wrapped_all_fail = wrap_invariant(invariant_all_fail)
    result = wrapped_all_fail(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2", "error3")
    
    # Test with kwargs
    def invariant_with_kwargs(x, y=None):
        return (True, f"result with y={y}")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_kwargs(5, y=10)
    assert result == (True, "result with y=10")
    
    # Test with callable error data
    def invariant_callable_error(x):
        error_func = lambda: "lazy error message"
        return ((True, "ok"), (False, error_func))
    
    wrapped_callable = wrap_invariant(invariant_callable_error)
    result = wrapped_callable(5)
    assert result[0] is False
    assert result[1] == (error_func,)


# LLM-generated content at query #39
#--------------------------

```python
def test_wrap_invariant():
    # Test with a simple invariant that returns a tuple of (bool, data)
    def simple_invariant(x):
        return (x > 0, "Value must be positive")
    
    wrapped = wrap_invariant(simple_invariant)
    verdict, data = wrapped(5)
    assert verdict is True
    assert data == ("Value must be positive",)
    
    verdict, data = wrapped(-5)
    assert verdict is False
    assert data == ("Value must be positive",)
    
    # Test with an invariant that returns multiple test results
    def multi_invariant(x):
        return [
            (x > 0, "Must be positive"),
            (x < 100, "Must be less than 100"),
            (x % 2 == 0, "Must be even")
        ]
    
    wrapped_multi = wrap_invariant(multi_invariant)
    
    # All tests pass
    verdict, data = wrapped_multi(50)
    assert verdict is True
    assert data == ()
    
    # One test fails
    verdict, data = wrapped_multi(-5)
    assert verdict is False
    assert "Must be positive" in data
    
    # Multiple tests fail
    verdict, data = wrapped_multi(101)
    assert verdict is False
    assert "Must be less than 100" in data
    assert "Must be even" in data
    
    # Test with invariant returning single bool result
    def bool_invariant(x):
        return (isinstance(x, int), "Must be an integer")
    
    wrapped_bool = wrap_invariant(bool_invariant)
    verdict, data = wrapped_bool(42)
    assert verdict is True
    assert data == ("Must be an integer",)
    
    verdict, data = wrapped_bool("not an int")
    assert verdict is False
    assert data == ("Must be an integer",)
    
    # Test with kwargs
    def kwargs_invariant(x, threshold=10):
        return (x > threshold, "Value too small")
    
    wrapped_kwargs = wrap_invariant(kwargs_invariant)
    verdict, data = wrapped_kwargs(15, threshold=10)
    assert verdict is True
    
    verdict, data = wrapped_kwargs(5, threshold=10)
    assert verdict is False


# LLM-generated content at query #40
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "success")
    
    # Test case 2: Single boolean result with False
    def invariant_single_fail(x):
        return (False, "failure")
    
    wrapped_single_fail = wrap_invariant(invariant_single_fail)
    result = wrapped_single_fail(5)
    assert result == (False, "failure")
    
    # Test case 3: Multiple boolean results to merge - all pass
    def invariant_multiple_pass(x):
        return [
            (True, "check1"),
            (True, "check2"),
            (True, "check3")
        ]
    
    wrapped_multiple_pass = wrap_invariant(invariant_multiple_pass)
    result = wrapped_multiple_pass(5)
    assert result == (True, ())
    
    # Test case 4: Multiple boolean results to merge - some fail
    def invariant_multiple_fail(x):
        return [
            (True, "check1"),
            (False, "check2_failed"),
            (True, "check3"),
            (False, "check4_failed")
        ]
    
    wrapped_multiple_fail = wrap_invariant(invariant_multiple_fail)
    result = wrapped_multiple_fail(5)
    assert result == (False, ("check2_failed", "check4_failed"))
    
    # Test case 5: Multiple boolean results - all fail
    def invariant_multiple_all_fail(x):
        return [
            (False, "check1_failed"),
            (False, "check2_failed")
        ]
    
    wrapped_multiple_all_fail = wrap_invariant(invariant_multiple_all_fail)
    result = wrapped_multiple_all_fail(5)
    assert result == (False, ("check1_failed", "check2_failed"))
    
    # Test case 6: Invariant with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x should be greater than y")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_kwargs(15, y=10)
    assert result == (True, "x should be greater than y")
    
    result = wrapped_kwargs(5, y=10)
    assert result == (False, "x should be greater than y")
    
    # Test case 7: Empty results list
    def invariant_empty(x):
        return []
    
    wrapped_empty = wrap_invariant(invariant_empty)
    result = wrapped_empty(5)
    assert result == (True, ())


# LLM-generated content at query #41
#--------------------------

def test_wrap_invariant():
    # Test with single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test with single boolean result (failure)
    def invariant_single_fail(x):
        return (False, "failure")
    
    wrapped_fail = wrap_invariant(invariant_single_fail)
    result_fail = wrapped_fail(5)
    assert result_fail == (False, "failure")
    
    # Test with multiple boolean results (all pass)
    def invariant_multi_pass(x):
        return [(True, "check1"), (True, "check2"), (True, "check3")]
    
    wrapped_multi_pass = wrap_invariant(invariant_multi_pass)
    result_multi_pass = wrapped_multi_pass(5)
    assert result_multi_pass[0] is True
    assert result_multi_pass[1] == ("check1", "check2", "check3")
    
    # Test with multiple boolean results (some fail)
    def invariant_multi_fail(x):
        return [(True, "check1"), (False, "error1"), (False, "error2")]
    
    wrapped_multi_fail = wrap_invariant(invariant_multi_fail)
    result_multi_fail = wrapped_multi_fail(5)
    assert result_multi_fail[0] is False
    assert result_multi_fail[1] == ("error1", "error2")
    
    # Test with multiple boolean results (all fail)
    def invariant_all_fail(x):
        return [(False, "fail1"), (False, "fail2")]
    
    wrapped_all_fail = wrap_invariant(invariant_all_fail)
    result_all_fail = wrapped_all_fail(5)
    assert result_all_fail[0] is False
    assert result_all_fail[1] == ("fail1", "fail2")
    
    # Test with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x greater than y")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result_kwargs = wrapped_kwargs(5, y=3)
    assert result_kwargs == (True, "x greater than y")
    
    # Test with multiple results returning empty errors (all pass)
    def invariant_empty_errors(x):
        return [(True, "ok1"), (True, "ok2")]
    
    wrapped_empty = wrap_invariant(invariant_empty_errors)
    result_empty = wrapped_empty(5)
    assert result_empty[0] is True
    assert result_empty[1] == ("ok1", "ok2")


# LLM-generated content at query #42
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Invariant returns a single boolean result
    def invariant_single_bool(value):
        return (True, "success")
    
    wrapped = wrap_invariant(invariant_single_bool)
    result = wrapped(42)
    assert result == (True, "success")
    
    # Test case 2: Invariant returns multiple results that need merging
    def invariant_multiple_results(value):
        return [
            (True, "check1_passed"),
            (False, "check2_failed"),
            (True, "check3_passed"),
        ]
    
    wrapped = wrap_invariant(invariant_multiple_results)
    result = wrapped(42)
    verdict, data = result
    assert verdict is False
    assert data == ("check2_failed",)
    
    # Test case 3: Invariant returns multiple results, all passing
    def invariant_all_pass(value):
        return [
            (True, "check1_passed"),
            (True, "check2_passed"),
            (True, "check3_passed"),
        ]
    
    wrapped = wrap_invariant(invariant_all_pass)
    result = wrapped(42)
    verdict, data = result
    assert verdict is True
    assert data == ()
    
    # Test case 4: Invariant returns multiple results, all failing
    def invariant_all_fail(value):
        return [
            (False, "check1_failed"),
            (False, "check2_failed"),
        ]
    
    wrapped = wrap_invariant(invariant_all_fail)
    result = wrapped(42)
    verdict, data = result
    assert verdict is False
    assert data == ("check1_failed", "check2_failed")
    
    # Test case 5: Invariant with arguments and kwargs
    def invariant_with_args(a, b, c=None):
        return (a > b, f"a={a}, b={b}, c={c}")
    
    wrapped = wrap_invariant(invariant_with_args)
    result = wrapped(10, 5, c=100)
    assert result == (True, "a=10, b=5, c=100")
    
    # Test case 6: Single boolean result in list format (edge case)
    def invariant_single_in_list(value):
        return [(True, "single_result")]
    
    wrapped = wrap_invariant(invariant_single_in_list)
    result = wrapped(42)
    verdict, data = result
    assert verdict is True
    assert data == ()


# LLM-generated content at query #43
#--------------------------

def test_wrap_invariant():
    # Test with a single boolean result
    def invariant_single(x):
        return True, "single result"
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "single result")
    
    # Test with multiple boolean results that need merging
    def invariant_multiple(x):
        return [(True, "first"), (True, "second"), (False, "third")]
    
    wrapped_multiple = wrap_invariant(invariant_multiple)
    result = wrapped_multiple(5)
    assert result[0] == False
    assert result[1] == ("third",)
    
    # Test with all passing multiple results
    def invariant_all_pass(x):
        return [(True, "first"), (True, "second")]
    
    wrapped_all_pass = wrap_invariant(invariant_all_pass)
    result = wrapped_all_pass(5)
    assert result[0] == True
    assert result[1] == ()
    
    # Test with single result in list format
    def invariant_single_list(x):
        return [(False, "error")]
    
    wrapped_single_list = wrap_invariant(invariant_single_list)
    result = wrapped_single_list(5)
    assert result[0] == False
    assert result[1] == ("error",)
    
    # Test with multiple failing results
    def invariant_multi_fail(x):
        return [(False, "error1"), (False, "error2"), (True, "ok")]
    
    wrapped_multi_fail = wrap_invariant(invariant_multi_fail)
    result = wrapped_multi_fail(5)
    assert result[0] == False
    assert result[1] == ("error1", "error2")
    
    # Test that wrapped function preserves kwargs
    def invariant_with_kwargs(x, y=None):
        if y:
            return False, "with y"
        return True, "without y"
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result1 = wrapped_kwargs(5)
    assert result1 == (True, "without y")
    result2 = wrapped_kwargs(5, y=True)
    assert result2 == (False, "with y")


# LLM-generated content at query #44
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Invariant returning a single boolean result
    def single_result_invariant(x):
        return (True, "success")
    
    wrapped = wrap_invariant(single_result_invariant)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test case 2: Invariant returning multiple results that need merging
    def multiple_results_invariant(x):
        return [
            (True, "check1_passed"),
            (True, "check2_passed"),
            (False, "check3_failed")
        ]
    
    wrapped = wrap_invariant(multiple_results_invariant)
    verdict, data = wrapped(5)
    assert verdict is False
    assert data == ("check3_failed",)
    
    # Test case 3: Invariant with all passing checks
    def all_passing_invariant(x):
        return [
            (True, "check1_passed"),
            (True, "check2_passed"),
            (True, "check3_passed")
        ]
    
    wrapped = wrap_invariant(all_passing_invariant)
    verdict, data = wrapped(5)
    assert verdict is True
    assert data == ()
    
    # Test case 4: Invariant with all failing checks
    def all_failing_invariant(x):
        return [
            (False, "check1_failed"),
            (False, "check2_failed"),
            (False, "check3_failed")
        ]
    
    wrapped = wrap_invariant(all_failing_invariant)
    verdict, data = wrapped(5)
    assert verdict is False
    assert data == ("check1_failed", "check2_failed", "check3_failed")
    
    # Test case 5: Invariant with mixed results
    def mixed_results_invariant(x):
        return [
            (True, "check1_passed"),
            (False, "check2_failed"),
            (True, "check3_passed"),
            (False, "check4_failed")
        ]
    
    wrapped = wrap_invariant(mixed_results_invariant)
    verdict, data = wrapped(10)
    assert verdict is False
    assert data == ("check2_failed", "check4_failed")
    
    # Test case 6: Single result with False
    def single_false_invariant(x):
        return (False, "failed_check")
    
    wrapped = wrap_invariant(single_false_invariant)
    result = wrapped(5)
    assert result == (False, "failed_check")


# LLM-generated content at query #45
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Single boolean result
    def invariant_single_bool(x):
        return (True, "valid")
    
    wrapped = wrap_invariant(invariant_single_bool)
    result = wrapped(5)
    assert result == (True, "valid")
    
    # Test 2: Single boolean result with failure
    def invariant_single_bool_fail(x):
        return (False, "invalid")
    
    wrapped = wrap_invariant(invariant_single_bool_fail)
    result = wrapped(5)
    assert result == (False, "invalid")
    
    # Test 3: Multiple test results (tuple of tuples)
    def invariant_multiple(x):
        return ((True, "test1"), (True, "test2"), (False, "test3"))
    
    wrapped = wrap_invariant(invariant_multiple)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("test3",)
    
    # Test 4: Multiple results all passing
    def invariant_multiple_pass(x):
        return ((True, "pass1"), (True, "pass2"))
    
    wrapped = wrap_invariant(invariant_multiple_pass)
    result = wrapped(5)
    assert result[0] is True
    assert result[1] == ()
    
    # Test 5: Multiple results with multiple failures
    def invariant_multiple_fails(x):
        return ((False, "fail1"), (True, "pass"), (False, "fail2"))
    
    wrapped = wrap_invariant(invariant_multiple_fails)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("fail1", "fail2")
    
    # Test 6: Wrapped function with kwargs
    def invariant_with_kwargs(x, y=10):
        return (x > y, "x should be greater than y")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(15, y=10)
    assert result == (True, "x should be greater than y")
    
    result = wrapped(5, y=10)
    assert result == (False, "x should be greater than y")
    
    # Test 7: Multiple results with single failure among many passes
    def invariant_one_fail_many_pass(x):
        return ((True, "a"), (True, "b"), (True, "c"), (False, "d"), (True, "e"))
    
    wrapped = wrap_invariant(invariant_one_fail_many_pass)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("d",)


# LLM-generated content at query #46
#--------------------------

def test_wrap_invariant():
    # Test with single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "success")
    
    # Test with single boolean result returning False
    def invariant_single_fail(x):
        return (False, "error")
    
    wrapped_single_fail = wrap_invariant(invariant_single_fail)
    result = wrapped_single_fail(5)
    assert result == (False, "error")
    
    # Test with multiple results to merge (all passing)
    def invariant_multiple_pass(x):
        return [(True, "check1"), (True, "check2"), (True, "check3")]
    
    wrapped_multiple_pass = wrap_invariant(invariant_multiple_pass)
    result = wrapped_multiple_pass(5)
    assert result == (True, ())
    
    # Test with multiple results to merge (some failing)
    def invariant_multiple_fail(x):
        return [(True, "check1"), (False, "error1"), (True, "check3"), (False, "error2")]
    
    wrapped_multiple_fail = wrap_invariant(invariant_multiple_fail)
    result = wrapped_multiple_fail(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test with multiple results all failing
    def invariant_multiple_all_fail(x):
        return [(False, "error1"), (False, "error2")]
    
    wrapped_multiple_all_fail = wrap_invariant(invariant_multiple_all_fail)
    result = wrapped_multiple_all_fail(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test with kwargs
    def invariant_with_kwargs(x, y=None):
        return (True, "success")
    
    wrapped_with_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_with_kwargs(5, y=10)
    assert result == (True, "success")
    
    # Test with single tuple result (not list)
    def invariant_tuple_result(x):
        return (True, "single")
    
    wrapped_tuple = wrap_invariant(invariant_tuple_result)
    result = wrapped_tuple(5)
    assert result == (True, "single")


# LLM-generated content at query #47
#--------------------------

def test_wrap_invariant():
    # Test with a simple invariant that returns a single (bool, data) tuple
    def simple_invariant(x):
        return (True, "success")
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "success")
    
    # Test with a failing invariant
    def failing_invariant(x):
        return (False, "error_message")
    
    wrapped = wrap_invariant(failing_invariant)
    result = wrapped(5)
    assert result == (False, "error_message")
    
    # Test with multiple results that need merging
    def multi_invariant(x):
        return [
            (True, "check1_passed"),
            (False, "check2_failed"),
            (True, "check3_passed"),
            (False, "check4_failed")
        ]
    
    wrapped = wrap_invariant(multi_invariant)
    result = wrapped(5)
    assert result[0] is False  # verdict should be False since some checks failed
    assert result[1] == ("check2_failed", "check4_failed")  # only failures collected
    
    # Test with all passing multiple results
    def all_pass_invariant(x):
        return [
            (True, "check1_passed"),
            (True, "check2_passed"),
            (True, "check3_passed")
        ]
    
    wrapped = wrap_invariant(all_pass_invariant)
    result = wrapped(5)
    assert result[0] is True
    assert result[1] == ()
    
    # Test with kwargs
    def kwargs_invariant(x, y=10):
        return (x > y, "x_greater_than_y")
    
    wrapped = wrap_invariant(kwargs_invariant)
    result = wrapped(15, y=10)
    assert result == (True, "x_greater_than_y")
    
    result = wrapped(5, y=10)
    assert result == (False, "x_greater_than_y")
    
    # Test with multiple results all failing
    def all_fail_invariant(x):
        return [
            (False, "error1"),
            (False, "error2"),
            (False, "error3")
        ]
    
    wrapped = wrap_invariant(all_fail_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2", "error3")


# LLM-generated content at query #48
#--------------------------

def test_wrap_invariant():
    # Test with a single boolean result
    def invariant_single(x):
        return (True, "success")
    
    wrapped_single = wrap_invariant(invariant_single)
    result = wrapped_single(5)
    assert result == (True, "success")
    
    # Test with multiple results that need merging
    def invariant_multiple(x):
        return [(True, "check1"), (True, "check2")]
    
    wrapped_multiple = wrap_invariant(invariant_multiple)
    result = wrapped_multiple(5)
    assert result == (True, ("check1", "check2"))
    
    # Test with mixed results (some pass, some fail)
    def invariant_mixed(x):
        return [(True, "check1"), (False, "check2"), (True, "check3")]
    
    wrapped_mixed = wrap_invariant(invariant_mixed)
    result = wrapped_mixed(5)
    assert result == (False, ("check2",))
    
    # Test with all failing results
    def invariant_all_fail(x):
        return [(False, "error1"), (False, "error2")]
    
    wrapped_fail = wrap_invariant(invariant_all_fail)
    result = wrapped_fail(5)
    assert result == (False, ("error1", "error2"))
    
    # Test with single item list (should be treated as multiple results)
    def invariant_single_list(x):
        return [(True, "single_check")]
    
    wrapped_single_list = wrap_invariant(invariant_single_list)
    result = wrapped_single_list(5)
    assert result == (True, ("single_check",))
    
    # Test with keyword arguments
    def invariant_with_kwargs(x, y=10):
        return (True, f"x={x}, y={y}")
    
    wrapped_kwargs = wrap_invariant(invariant_with_kwargs)
    result = wrapped_kwargs(5, y=20)
    assert result == (True, "x=5, y=20")


# LLM-generated content at query #49
#--------------------------

def test_wrap_invariant():
    # Test with a simple invariant that returns a single (bool, data) tuple
    def simple_invariant(x):
        return (True, "valid")
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "valid")
    
    # Test with a failing simple invariant
    def failing_invariant(x):
        return (False, "invalid")
    
    wrapped = wrap_invariant(failing_invariant)
    result = wrapped(5)
    assert result == (False, "invalid")
    
    # Test with multiple invariant results that all pass
    def multi_pass_invariant(x):
        return ((True, "check1"), (True, "check2"), (True, "check3"))
    
    wrapped = wrap_invariant(multi_pass_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test with multiple invariant results where some fail
    def multi_fail_invariant(x):
        return ((True, "check1"), (False, "error1"), (True, "check3"), (False, "error2"))
    
    wrapped = wrap_invariant(multi_fail_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test with all failing multiple invariant results
    def all_fail_invariant(x):
        return ((False, "error1"), (False, "error2"))
    
    wrapped = wrap_invariant(all_fail_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test with single failing result in multiple results
    def single_fail_invariant(x):
        return ((True, "pass"), (False, "fail"))
    
    wrapped = wrap_invariant(single_fail_invariant)
    result = wrapped(5)
    assert result[0] is False
    assert result[1] == ("fail",)
    
    # Test that wrapped function passes through arguments correctly
    def arg_invariant(x, y, z=None):
        return (x > 0 and y > 0, "args_valid" if x > 0 and y > 0 else "args_invalid")
    
    wrapped = wrap_invariant(arg_invariant)
    result = wrapped(5, 10, z=20)
    assert result == (True, "args_valid")
    
    result = wrapped(-5, 10)
    assert result == (False, "args_invalid")


# LLM-generated content at query #50
#--------------------------

```python
def test_wrap_invariant():
    # Test case 1: Invariant returning a single boolean result
    def invariant_single(value):
        return (True, "valid")
    
    wrapped = wrap_invariant(invariant_single)
    result = wrapped(42)
    assert result == (True, "valid")
    
    # Test case 2: Invariant returning False with error data
    def invariant_false(value):
        return (False, "error_code")
    
    wrapped = wrap_invariant(invariant_false)
    result = wrapped(42)
    assert result == (False, "error_code")
    
    # Test case 3: Invariant returning multiple test results that need merging
    def invariant_multiple(value):
        return [
            (True, "test1_passed"),
            (True, "test2_passed"),
            (False, "test3_failed")
        ]
    
    wrapped = wrap_invariant(invariant_multiple)
    result = wrapped(42)
    assert result[0] is False
    assert result[1] == ("test3_failed",)
    
    # Test case 4: Invariant with all passing multiple tests
    def invariant_all_pass(value):
        return [
            (True, "test1"),
            (True, "test2"),
            (True, "test3")
        ]
    
    wrapped = wrap_invariant(invariant_all_pass)
    result = wrapped(42)
    assert result[0] is True
    assert result[1] == ()
    
    # Test case 5: Invariant with multiple failures
    def invariant_multiple_failures(value):
        return [
            (False, "error1"),
            (True, "success"),
            (False, "error2")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_failures)
    result = wrapped(42)
    assert result[0] is False
    assert result[1] == ("error1", "error2")
    
    # Test case 6: Invariant with kwargs
    def invariant_with_kwargs(value, threshold=10):
        return (value > threshold, f"value must be > {threshold}")
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(15, threshold=10)
    assert result == (True, f"value must be > 10")
    
    # Test case 7: Invariant with multiple args
    def invariant_multiple_args(a, b):
        return (a < b, "a must be less than b")
    
    wrapped = wrap_invariant(invariant_multiple_args)
    result = wrapped(5, 10)
    assert result == (True, "a must be less than b")


