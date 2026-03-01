####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def invariant_single_true(x):
        return True, "ok"
    
    wrapped = wrap_invariant(invariant_single_true)
    result = wrapped(5)
    assert result == (True, "ok")
    
    # Test 2: Invariant returns single boolean false result
    def invariant_single_false(x):
        return False, "error"
    
    wrapped = wrap_invariant(invariant_single_false)
    result = wrapped(5)
    assert result == (False, "error")
    
    # Test 3: Invariant returns multiple results that need merging (all true)
    def invariant_multiple_true(x):
        return [(True, "ok1"), (True, "ok2")]
    
    wrapped = wrap_invariant(invariant_multiple_true)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 4: Invariant returns multiple results with some false
    def invariant_multiple_mixed(x):
        return [(True, "ok1"), (False, "error1"), (True, "ok2"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_multiple_mixed)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "error1" in result[1]
    assert "error2" in result[1]
    
    # Test 5: Invariant returns multiple results all false
    def invariant_multiple_false(x):
        return [(False, "error1"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_multiple_false)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "error1" in result[1]
    assert "error2" in result[1]
    
    # Test 6: Invariant with multiple arguments
    def invariant_multi_args(x, y, z=0):
        if x > y and z == 0:
            return True, "valid"
        return False, "invalid"
    
    wrapped = wrap_invariant(invariant_multi_args)
    result = wrapped(10, 5, z=0)
    assert result == (True, "valid")
    
    result = wrapped(5, 10, z=0)
    assert result == (False, "invalid")
    
    # Test 7: Invariant with keyword arguments
    def invariant_kwargs(**kwargs):
        if kwargs.get('valid', False):
            return True, "ok"
        return False, "not ok"
    
    wrapped = wrap_invariant(invariant_kwargs)
    result = wrapped(valid=True)
    assert result == (True, "ok")
    
    result = wrapped(valid=False)
    assert result == (False, "not ok")


# LLM-generated content at query #2
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test with string
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("custom_type") == ["custom_type"]
    
    # Test with Enum subclass (preserved type)
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    
    assert maybe_parse_user_type(Color) == [Color]
    
    # Test with iterable of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    assert maybe_parse_user_type((float, bool)) == (float, bool)
    
    # Test with nested iterables
    assert maybe_parse_user_type([int, [str, float]]) == (int, str, float)
    assert maybe_parse_user_type([[int, str], float]) == (int, str, float)
    
    # Test with mixed types and strings
    assert maybe_parse_user_type([int, "custom", str]) == (int, "custom", str)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    assert maybe_parse_user_type(()) == ()
    
    # Test that non-type, non-string, non-iterable raises TypeError
    try:
        maybe_parse_user_type(42)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    try:
        maybe_parse_user_type(None)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test with callable that's not a type
    def some_func():
        pass
    
    try:
        maybe_parse_user_type(some_func)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test with iterable containing invalid element
    try:
        maybe_parse_user_type([int, 42])
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_CheckedPSet___new__():
    # Test 1: Create empty CheckedPSet
    class IntSet(CheckedPSet):
        __type__ = int
    
    result = IntSet()
    assert isinstance(result, IntSet)
    assert len(result) == 0
    
    # Test 2: Create from iterable with correct types
    result = IntSet([1, 2, 3])
    assert isinstance(result, IntSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    
    # Test 3: Create from existing PMap (internal representation)
    from pyrsistent import pmap
    internal_map = pmap({1: True, 2: True})
    result = IntSet(internal_map)
    assert isinstance(result, IntSet)
    assert len(result) == 2
    assert 1 in result
    assert 2 in result
    
    # Test 4: Type checking - should raise CheckedValueTypeError
    class StringSet(CheckedPSet):
        __type__ = str
    
    try:
        StringSet([1, 2, 3])
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.source_class == StringSet
        assert e.actual_type == int
    
    # Test 5: Invariant checking - should raise InvariantException
    class PositiveSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda x: (x > 0, "Must be positive")
    
    try:
        PositiveSet([1, -2, 3])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) == 1
        assert "Must be positive" in str(e.invariant_errors[0])
    
    # Test 6: Multiple types allowed
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    result = MultiTypeSet([1, "hello", 3])
    assert isinstance(result, MultiTypeSet)
    assert len(result) == 3
    assert 1 in result
    assert "hello" in result
    assert 3 in result
    
    # Test 7: Duplicate elements should be deduplicated
    result = IntSet([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    
    # Test 8: Create from another CheckedPSet of same type
    original = IntSet([1, 2, 3])
    result = IntSet(original)
    assert isinstance(result, IntSet)
    assert len(result) == 3
    assert result == original
    
    # Test 9: Create with optional type including None
    class OptionalSet(CheckedPSet):
        __type__ = optional(int)
    
    result = OptionalSet([1, None, 3])
    assert isinstance(result, OptionalSet)
    assert len(result) == 3
    assert 1 in result
    assert None in result
    assert 3 in result
    
    # Test 10: Empty iterable
    result = IntSet([])
    assert isinstance(result, IntSet)
    assert len(result) == 0


# LLM-generated content at query #4
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def invariant_single_true(x):
        return True, "All good"
    
    wrapped = wrap_invariant(invariant_single_true)
    result = wrapped(5)
    assert result == (True, "All good")
    
    # Test 2: Invariant returns single false result
    def invariant_single_false(x):
        return False, "Something wrong"
    
    wrapped = wrap_invariant(invariant_single_false)
    result = wrapped(5)
    assert result == (False, "Something wrong")
    
    # Test 3: Invariant returns multiple results that need merging (all true)
    def invariant_multiple_true(x):
        return [(True, "First good"), (True, "Second good")]
    
    wrapped = wrap_invariant(invariant_multiple_true)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 4: Invariant returns multiple results with some false
    def invariant_multiple_mixed(x):
        return [(True, "First good"), (False, "Second bad"), (True, "Third good")]
    
    wrapped = wrap_invariant(invariant_multiple_mixed)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 1
    assert result[1][0] == "Second bad"
    
    # Test 5: Invariant returns multiple results with all false
    def invariant_multiple_all_false(x):
        return [(False, "First bad"), (False, "Second bad")]
    
    wrapped = wrap_invariant(invariant_multiple_all_false)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 2
    assert result[1][0] == "First bad"
    assert result[1][1] == "Second bad"
    
    # Test 6: Invariant with no arguments
    def invariant_no_args():
        return True, "No args"
    
    wrapped = wrap_invariant(invariant_no_args)
    result = wrapped()
    assert result == (True, "No args")
    
    # Test 7: Invariant with keyword arguments
    def invariant_kwargs(**kwargs):
        return kwargs.get('check', False), "Keyword check"
    
    wrapped = wrap_invariant(invariant_kwargs)
    result = wrapped(check=True)
    assert result == (True, "Keyword check")


# LLM-generated content at query #5
#--------------------------

```python
def test_CheckedPSet_serialize():
    # Test 1: Basic serialization with default serializer
    class SimpleSet(CheckedPSet):
        __type__ = int
    
    s = SimpleSet([1, 2, 3])
    result = s.serialize()
    assert result == {1, 2, 3}
    assert isinstance(result, set)
    
    # Test 2: Serialization with nested CheckedType objects
    class InnerSet(CheckedPSet):
        __type__ = int
    
    class OuterSet(CheckedPSet):
        __type__ = InnerSet
    
    inner = InnerSet([1, 2])
    outer = OuterSet([inner])
    result = outer.serialize()
    assert len(result) == 1
    assert list(result)[0] == {1, 2}
    
    # Test 3: Serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return value * 2
    
    s = CustomSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {2, 4, 6}
    
    # Test 4: Serialization with format parameter
    class FormatSet(CheckedPSet):
        __type__ = str
        
        def __serializer__(self, format, value):
            if format == 'uppercase':
                return value.upper()
            return value
    
    s = FormatSet(['a', 'b', 'c'])
    result = s.serialize(format='uppercase')
    assert result == {'A', 'B', 'C'}
    
    # Test 5: Serialization with empty set
    class EmptySet(CheckedPSet):
        __type__ = int
    
    s = EmptySet()
    result = s.serialize()
    assert result == set()
    
    # Test 6: Serialization preserves set semantics (no duplicates)
    class NoDupSet(CheckedPSet):
        __type__ = int
    
    s = NoDupSet([1, 2, 2, 3, 3, 3])
    result = s.serialize()
    assert len(result) == 3
    assert result == {1, 2, 3}
    
    # Test 7: Serialization with multiple types
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    s = MultiTypeSet([1, 'a', 2, 'b'])
    result = s.serialize()
    assert result == {1, 'a', 2, 'b'}
    
    # Test 8: Verify that serialize returns a new set, not the internal representation
    class IntSet(CheckedPSet):
        __type__ = int
    
    s = IntSet([1, 2, 3])
    result = s.serialize()
    result.add(4)  # Should not affect the original CheckedPSet
    assert 4 not in s
    assert len(s) == 3


# LLM-generated content at query #6
#--------------------------

```python
def test_CheckedPSet_serialize():
    # Test 1: Basic serialization with primitive types
    class IntSet(CheckedPSet):
        __type__ = int
    
    int_set = IntSet([1, 2, 3])
    result = int_set.serialize()
    assert result == {1, 2, 3}
    assert isinstance(result, set)
    
    # Test 2: Serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        __serializer__ = lambda self, _, value: value * 2
    
    custom_set = CustomSerializerSet([1, 2, 3])
    result = custom_set.serialize()
    assert result == {2, 4, 6}
    
    # Test 3: Serialization with nested CheckedType objects
    class InnerSet(CheckedPSet):
        __type__ = int
    
    class OuterSet(CheckedPSet):
        __type__ = InnerSet
        __serializer__ = lambda self, _, value: value.serialize()
    
    inner = InnerSet([1, 2])
    outer = OuterSet([inner])
    result = outer.serialize()
    assert result == {{1, 2}}
    
    # Test 4: Serialization with format parameter (should be ignored by default serializer)
    class FormatSet(CheckedPSet):
        __type__ = str
    
    str_set = FormatSet(['a', 'b', 'c'])
    result = str_set.serialize(format='json')
    assert result == {'a', 'b', 'c'}
    
    # Test 5: Serialization with multiple types
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    multi_set = MultiTypeSet([1, 'a', 2, 'b'])
    result = multi_set.serialize()
    assert result == {1, 'a', 2, 'b'}
    
    # Test 6: Serialization of empty set
    class EmptySet(CheckedPSet):
        __type__ = int
    
    empty = EmptySet()
    result = empty.serialize()
    assert result == set()
    
    # Test 7: Serialization preserves set semantics (no duplicates)
    class NoDupSet(CheckedPSet):
        __type__ = int
    
    dup_set = NoDupSet([1, 2, 2, 3, 3, 3])
    result = dup_set.serialize()
    assert len(result) == 3
    assert result == {1, 2, 3}
    
    # Test 8: Default serializer handles CheckedType objects
    class InnerChecked(CheckedPSet):
        __type__ = int
    
    class OuterChecked(CheckedPSet):
        __type__ = InnerChecked
    
    inner_checked = InnerChecked([10, 20])
    outer_checked = OuterChecked([inner_checked])
    result = outer_checked.serialize()
    # Default serializer should call serialize() on inner CheckedType
    assert result == {{10, 20}}


# LLM-generated content at query #7
#--------------------------

```python
def test_CheckedPSet_serialize():
    # Test 1: Basic serialization with default serializer
    class SimpleSet(CheckedPSet):
        __type__ = int
    
    s = SimpleSet([1, 2, 3])
    result = s.serialize()
    assert result == {1, 2, 3}
    assert isinstance(result, set)
    
    # Test 2: Serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return value * 2
    
    s = CustomSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {2, 4, 6}
    
    # Test 3: Serialization with nested CheckedType objects
    class InnerSet(CheckedPSet):
        __type__ = int
    
    class OuterSet(CheckedPSet):
        __type__ = InnerSet
    
    inner = InnerSet([1, 2])
    outer = OuterSet([inner])
    result = outer.serialize()
    assert result == {frozenset({1, 2})}
    
    # Test 4: Serialization with format parameter (should be ignored by default serializer)
    class FormatSet(CheckedPSet):
        __type__ = str
    
    s = FormatSet(['a', 'b', 'c'])
    result = s.serialize(format='json')
    assert result == {'a', 'b', 'c'}
    
    # Test 5: Serialization with custom serializer using format parameter
    class FormatAwareSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            if format == 'string':
                return str(value)
            return value
    
    s = FormatAwareSet([1, 2, 3])
    result = s.serialize(format='string')
    assert result == {'1', '2', '3'}
    
    # Test 6: Empty set serialization
    class EmptySet(CheckedPSet):
        __type__ = int
    
    s = EmptySet()
    result = s.serialize()
    assert result == set()
    
    # Test 7: Serialization preserves set semantics (no duplicates)
    class NoDupSet(CheckedPSet):
        __type__ = int
    
    s = NoDupSet([1, 2, 2, 3, 3, 3])
    result = s.serialize()
    assert len(result) == 3
    assert result == {1, 2, 3}
    
    # Test 8: Serialization with multiple allowed types
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    s = MultiTypeSet([1, 'a', 2, 'b'])
    result = s.serialize()
    assert result == {1, 'a', 2, 'b'}


# LLM-generated content at query #8
#--------------------------

```python
def test_CheckedPSet_serialize():
    # Test 1: Basic serialization with default serializer
    class IntSet(CheckedPSet):
        __type__ = int
    
    s = IntSet([1, 2, 3])
    result = s.serialize()
    assert result == {1, 2, 3}
    assert isinstance(result, set)
    
    # Test 2: Serialization with nested CheckedType objects
    class InnerType(CheckedPSet):
        __type__ = int
    
    class OuterType(CheckedPSet):
        __type__ = InnerType
    
    inner = InnerType([1, 2])
    outer = OuterType([inner])
    result = outer.serialize()
    assert result == {[1, 2]}
    assert isinstance(result, set)
    assert all(isinstance(item, list) for item in result)
    
    # Test 3: Serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return str(value) + "_serialized"
    
    s = CustomSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {"1_serialized", "2_serialized", "3_serialized"}
    
    # Test 4: Serialization with format parameter
    class FormatAwareSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            if format == "json":
                return str(value)
            return value
    
    s = FormatAwareSet([1, 2, 3])
    result = s.serialize(format="json")
    assert result == {"1", "2", "3"}
    
    # Test 5: Empty set serialization
    class EmptySet(CheckedPSet):
        __type__ = int
    
    s = EmptySet()
    result = s.serialize()
    assert result == set()
    
    # Test 6: Serialization preserves set semantics (no duplicates)
    class NoDupSet(CheckedPSet):
        __type__ = int
    
    s = NoDupSet([1, 1, 2, 2, 3])
    result = s.serialize()
    assert len(result) == 3
    assert result == {1, 2, 3}
    
    # Test 7: Serialization with mixed types (using optional)
    class MixedSet(CheckedPSet):
        __type__ = optional(int, str)
    
    s = MixedSet([1, "hello", None, 2])
    result = s.serialize()
    assert result == {1, "hello", None, 2}
    
    # Test 8: Verify serializer is called for each element
    class CountingSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            # Transform and track that we were called
            return value * 2
    
    s = CountingSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {2, 4, 6}


# LLM-generated content at query #9
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test with string
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("custom_type") == ["custom_type"]
    
    # Test with Enum subclass (preserved type)
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    
    assert maybe_parse_user_type(Color) == [Color]
    
    # Test with iterable of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    assert maybe_parse_user_type((float, bool)) == (float, bool)
    
    # Test with nested iterables
    assert maybe_parse_user_type([int, [str, float]]) == (int, str, float)
    assert maybe_parse_user_type([[int, str], float]) == (int, str, float)
    
    # Test with mixed types and strings
    assert maybe_parse_user_type([int, "custom", str]) == (int, "custom", str)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    assert maybe_parse_user_type(()) == ()
    
    # Test that regular iterable types are not preserved
    class CustomIterable(Iterable):
        def __iter__(self):
            return iter([])
    
    # This should return the type itself, not preserve it
    assert maybe_parse_user_type(CustomIterable) == [CustomIterable]
    
    # Test error cases
    import pytest
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)  # Not a type or string
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)  # Not a type or string
    
    with pytest.raises(TypeError):
        maybe_parse_user_type({"key": "value"})  # Dict is iterable but not valid
    
    # Test with complex nested structure
    assert maybe_parse_user_type([int, [str, [float, bool]]]) == (int, str, float, bool)


# LLM-generated content at query #10
#--------------------------

```python
def test_store_invariants():
    # Test 1: Basic functionality - stores invariants from dict
    def invariant1(obj):
        return True, None
    
    dct = {'_invariant': invariant1}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert '_checked_invariants' in dct
    assert len(dct['_checked_invariants']) == 1
    assert callable(dct['_checked_invariants'][0])
    
    # Test 2: Multiple invariants in dict
    def invariant2(obj):
        return False, "error2"
    
    dct = {'_invariant': [invariant1, invariant2]}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2
    assert dct['_checked_invariants'][0].__name__ == 'f'
    assert dct['_checked_invariants'][1].__name__ == 'f'
    
    # Test 3: Inheritance from single base class
    class Base:
        _invariant = invariant1
    
    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 1
    assert dct['_checked_invariants'][0].__name__ == 'f'
    
    # Test 4: Inheritance from multiple base classes
    class Base1:
        _invariant = invariant1
    
    class Base2:
        _invariant = invariant2
    
    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2
    
    # Test 5: Combination of local and inherited invariants
    def invariant3(obj):
        return True, None
    
    dct = {'_invariant': invariant3}
    bases = (Base1,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2
    
    # Test 6: Deep inheritance chain
    class GrandParent:
        _invariant = invariant1
    
    class Parent(GrandParent):
        pass
    
    class Child(Parent):
        _invariant = invariant2
    
    dct = {}
    bases = (Child,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2
    
    # Test 7: Invariants are wrapped with wrap_invariant
    def invariant_returns_tuple(obj):
        return True, "data"
    
    def invariant_returns_iterable(obj):
        return [(True, "data1"), (False, "data2")]
    
    dct = {'_invariant': [invariant_returns_tuple, invariant_returns_iterable]}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    # Both should be wrapped
    assert len(dct['_checked_invariants']) == 2
    assert all(callable(inv) for inv in dct['_checked_invariants'])
    
    # Test 8: Non-callable invariant raises TypeError
    dct = {'_invariant': "not callable"}
    bases = ()
    
    try:
        store_invariants(dct, bases, '_checked_invariants', '_invariant')
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test 9: Empty invariants
    dct = {}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert '_checked_invariants' in dct
    assert dct['_checked_invariants'] == ()
    
    # Test 10: Diamond inheritance - ensure no duplicates
    class CommonBase:
        _invariant = invariant1
    
    class Left(CommonBase):
        pass
    
    class Right(CommonBase):
        pass
    
    class Bottom(Left, Right):
        pass
    
    dct = {}
    bases = (Bottom,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    # Should only have one copy of invariant1 despite diamond inheritance
    assert len(dct['_checked_invariants']) == 1


# LLM-generated content at query #11
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test with string
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("custom_type") == ["custom_type"]
    
    # Test with Enum (preserved type)
    class Color(Enum):
        RED = 1
        GREEN = 2
    
    assert maybe_parse_user_type(Color) == [Color]
    
    # Test with iterable of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    assert maybe_parse_user_type((float, bool)) == (float, bool)
    
    # Test with nested iterables
    assert maybe_parse_user_type([int, [str, float]]) == (int, str, float)
    assert maybe_parse_user_type([[int, str], float]) == (int, str, float)
    
    # Test with mixed types and strings
    assert maybe_parse_user_type([int, "custom", str]) == (int, "custom", str)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    assert maybe_parse_user_type(()) == ()
    
    # Test that non-type, non-string, non-iterable raises TypeError
    try:
        maybe_parse_user_type(42)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    try:
        maybe_parse_user_type(None)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test with complex nested structure
    class MyEnum(Enum):
        VALUE = 1
    
    result = maybe_parse_user_type([int, [MyEnum, "test"], float])
    assert result == (int, MyEnum, "test", float)


# LLM-generated content at query #12
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def simple_invariant(x):
        return x > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant returns multiple results that need merging
    def complex_invariant(x):
        return [
            (x > 0, "Value must be positive"),
            (x < 10, "Value must be less than 10"),
            (x % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # All conditions pass
    result = wrapped(4)
    assert result == (True, ())
    
    # One condition fails
    result = wrapped(-2)
    assert result[0] == False
    assert len(result[1]) == 1
    assert "Value must be positive" in result[1][0]
    
    # Multiple conditions fail
    result = wrapped(11)
    assert result[0] == False
    assert len(result[1]) == 2
    error_messages = [str(e) for e in result[1]]
    assert any("less than 10" in msg for msg in error_messages)
    assert any("even" in msg for msg in error_messages)
    
    # Test 3: Mixed results
    result = wrapped(3)
    assert result[0] == False
    assert len(result[1]) == 1
    assert "even" in result[1][0]
    
    # Test 4: Empty result list (edge case)
    def empty_invariant(x):
        return []
    
    wrapped = wrap_invariant(empty_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 5: Single element list (edge case)
    def single_list_invariant(x):
        return [(x > 0, "Positive")]
    
    wrapped = wrap_invariant(single_list_invariant)
    result = wrapped(5)
    assert result == (True, ())
    result = wrapped(-5)
    assert result == (False, ("Positive",))


# LLM-generated content at query #13
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def invariant_single_true(x):
        return True, "OK"
    
    wrapped = wrap_invariant(invariant_single_true)
    result = wrapped(5)
    assert result == (True, "OK")
    
    # Test 2: Invariant returns single boolean false result
    def invariant_single_false(x):
        return False, "Error"
    
    wrapped = wrap_invariant(invariant_single_false)
    result = wrapped(5)
    assert result == (False, "Error")
    
    # Test 3: Invariant returns multiple results that need merging (all true)
    def invariant_multiple_true(x):
        return [(True, "OK1"), (True, "OK2")]
    
    wrapped = wrap_invariant(invariant_multiple_true)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 4: Invariant returns multiple results with some false
    def invariant_multiple_mixed(x):
        return [(True, "OK1"), (False, "Error1"), (True, "OK2"), (False, "Error2")]
    
    wrapped = wrap_invariant(invariant_multiple_mixed)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Error1" in result[1]
    assert "Error2" in result[1]
    
    # Test 5: Invariant returns multiple results all false
    def invariant_multiple_false(x):
        return [(False, "Error1"), (False, "Error2")]
    
    wrapped = wrap_invariant(invariant_multiple_false)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Error1" in result[1]
    assert "Error2" in result[1]
    
    # Test 6: Invariant with no arguments
    def invariant_no_args():
        return True, "OK"
    
    wrapped = wrap_invariant(invariant_no_args)
    result = wrapped()
    assert result == (True, "OK")
    
    # Test 7: Invariant with keyword arguments
    def invariant_kwargs(**kwargs):
        return kwargs.get("check", False), "Result"
    
    wrapped = wrap_invariant(invariant_kwargs)
    result = wrapped(check=True)
    assert result == (True, "Result")


# LLM-generated content at query #14
#--------------------------

```python
def test_store_invariants():
    # Test 1: Basic functionality - invariants are stored correctly
    class Base:
        def invariant1(self):
            return True, "invariant1"
    
    class Child(Base):
        def invariant2(self):
            return True, "invariant2"
    
    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, "dest", "invariant1")
    assert "dest" in dct
    assert len(dct["dest"]) == 1
    assert dct["dest"][0].__name__ == "f"
    
    # Test 2: Multiple inheritance - invariants from all bases are collected
    class GrandParent:
        def gp_invariant(self):
            return True, "gp"
    
    class Parent(GrandParent):
        def p_invariant(self):
            return True, "p"
    
    class MultipleChild(Parent):
        def c_invariant(self):
            return True, "c"
    
    dct2 = {}
    bases2 = (Parent,)
    store_invariants(dct2, bases2, "dest2", "gp_invariant")
    assert len(dct2["dest2"]) == 1
    
    # Test 3: Invariants in dct take precedence
    dct3 = {"invariant": lambda self: (True, "dct_invariant")}
    bases3 = (Base,)
    store_invariants(dct3, bases3, "dest3", "invariant")
    assert len(dct3["dest3"]) == 2
    
    # Test 4: Wrap invariant function handles tuple returns
    def complex_invariant(self):
        return [(True, "test1"), (False, "test2")]
    
    dct4 = {"invariant": complex_invariant}
    bases4 = ()
    store_invariants(dct4, bases4, "dest4", "invariant")
    wrapped = dct4["dest4"][0]
    result = wrapped(None)
    assert result == (False, ("test2",))
    
    # Test 5: Non-callable invariant raises TypeError
    class BadBase:
        invariant = "not callable"
    
    dct5 = {}
    bases5 = (BadBase,)
    try:
        store_invariants(dct5, bases5, "dest5", "invariant")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test 6: Empty invariants
    dct6 = {}
    bases6 = ()
    store_invariants(dct6, bases6, "dest6", "nonexistent")
    assert len(dct6["dest6"]) == 0
    
    # Test 7: Diamond inheritance - ensure no duplicates
    class A:
        def common_invariant(self):
            return True, "common"
    
    class B(A):
        pass
    
    class C(A):
        pass
    
    class D(B, C):
        pass
    
    dct7 = {}
    bases7 = (B, C)
    store_invariants(dct7, bases7, "dest7", "common_invariant")
    assert len(dct7["dest7"]) == 1
    
    # Test 8: Invariant that returns boolean directly
    def simple_invariant(self):
        return True
    
    dct8 = {"invariant": simple_invariant}
    bases8 = ()
    store_invariants(dct8, bases8, "dest8", "invariant")
    wrapped_simple = dct8["dest8"][0]
    result = wrapped_simple(None)
    assert result == (True, ())


# LLM-generated content at query #15
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test with string
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("custom_type") == ["custom_type"]
    
    # Test with Enum subclass (preserved type)
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    
    assert maybe_parse_user_type(Color) == [Color]
    
    # Test with iterable of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    assert maybe_parse_user_type((float, bool)) == (float, bool)
    
    # Test with nested iterables
    assert maybe_parse_user_type([int, [str, float]]) == (int, str, float)
    assert maybe_parse_user_type([[int, str], float]) == (int, str, float)
    
    # Test with mixed types and strings
    assert maybe_parse_user_type([int, "custom", str]) == (int, "custom", str)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    assert maybe_parse_user_type(()) == ()
    
    # Test with complex nested structure
    assert maybe_parse_user_type([int, [str, [float, bool]]]) == (int, str, float, bool)
    
    # Test that invalid input raises TypeError
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)  # Not a type or string
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)  # Not a type or string
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(object())  # Not a type or string
    
    # Test that non-preserved iterable types are not preserved
    class CustomIterable(Iterable):
        def __iter__(self):
            return iter([])
    
    # This should be treated as an iterable, not a preserved type
    result = maybe_parse_user_type(CustomIterable)
    assert len(result) == 1
    assert result[0] == CustomIterable


# LLM-generated content at query #16
#--------------------------

```python
def test_store_invariants():
    # Test 1: Basic functionality - stores invariants from dict
    def invariant1(obj):
        return True, ()
    
    dct = {'_invariant': invariant1}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert '_checked_invariants' in dct
    assert len(dct['_checked_invariants']) == 1
    assert callable(dct['_checked_invariants'][0])
    
    # Test 2: Multiple invariants in dict
    def invariant2(obj):
        return False, "error2"
    
    dct = {'_invariant': [invariant1, invariant2]}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2
    assert dct['_checked_invariants'][0].__name__ == 'f'
    assert dct['_checked_invariants'][1].__name__ == 'f'
    
    # Test 3: Inheritance from single base class
    class Base:
        _invariant = invariant1
    
    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 1
    assert callable(dct['_checked_invariants'][0])
    
    # Test 4: Inheritance from multiple base classes
    class Base1:
        _invariant = invariant1
    
    class Base2:
        _invariant = invariant2
    
    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2
    
    # Test 5: Override invariants in subclass
    def invariant3(obj):
        return True, ()
    
    class Base:
        _invariant = invariant1
    
    dct = {'_invariant': invariant3}
    bases = (Base,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2  # Both base and subclass invariants
    # Order should be subclass first, then base
    assert dct['_checked_invariants'][0].__name__ == 'f'
    assert dct['_checked_invariants'][1].__name__ == 'f'
    
    # Test 6: Inheritance chain with multiple levels
    class GrandParent:
        _invariant = invariant1
    
    class Parent(GrandParent):
        _invariant = invariant2
    
    dct = {}
    bases = (Parent,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    # Should get invariants from both Parent and GrandParent
    assert len(dct['_checked_invariants']) == 2
    
    # Test 7: Diamond inheritance - ensure no duplicates
    class BaseA:
        _invariant = invariant1
    
    class BaseB(BaseA):
        pass
    
    class BaseC(BaseA):
        pass
    
    class Derived(BaseB, BaseC):
        _invariant = invariant2
    
    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    # Should have invariant2 from Derived and invariant1 from BaseA (only once)
    assert len(dct['_checked_invariants']) == 2
    
    # Test 8: Non-callable invariant raises TypeError
    dct = {'_invariant': "not callable"}
    bases = ()
    
    try:
        store_invariants(dct, bases, '_checked_invariants', '_invariant')
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test 9: Mixed callable and non-callable in list raises TypeError
    dct = {'_invariant': [invariant1, "not callable"]}
    bases = ()
    
    try:
        store_invariants(dct, bases, '_checked_invariants', '_invariant')
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test 10: Empty invariants
    dct = {}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert '_checked_invariants' in dct
    assert dct['_checked_invariants'] == ()
    
    # Test 11: Invariants are wrapped with wrap_invariant
    def invariant_returns_tuple(obj):
        return (True, "data1"), (False, "data2")
    
    dct = {'_invariant': invariant_returns_tuple}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    wrapped_invariant = dct['_checked_invariants'][0]
    result = wrapped_invariant(None)
    
    # Should be merged by wrap_invariant
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] is False  # One of the sub-results was False
    assert isinstance(result[1], tuple)
    assert len(result[1]) == 1  # Only the failing test data
    
    # Test 12: Invariant that already returns bool, data
    def simple_invariant(obj):
        return True, ()
    
    dct = {'_invariant': simple_invariant}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    wrapped_invariant = dct['_checked_invariants'][0]
    result = wrapped_invariant(None)
    
    assert result == (True, ())


# LLM-generated content at query #17
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test with string
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("custom_type") == ["custom_type"]
    
    # Test with Enum (preserved type)
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    
    assert maybe_parse_user_type(Color) == [Color]
    
    # Test with iterable of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    assert maybe_parse_user_type((float, bool)) == (float, bool)
    
    # Test with nested iterables
    assert maybe_parse_user_type([int, [str, float]]) == (int, str, float)
    assert maybe_parse_user_type([[int, str], float]) == (int, str, float)
    
    # Test with mixed types and strings
    assert maybe_parse_user_type([int, "custom", str]) == (int, "custom", str)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    assert maybe_parse_user_type(()) == ()
    
    # Test that invalid input raises TypeError
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)
    
    with pytest.raises(TypeError):
        maybe_parse_user_type({"key": "value"})
    
    # Test with complex nested structure
    class MyEnum(Enum):
        VALUE = 1
    
    result = maybe_parse_user_type([int, [str, MyEnum], "test"])
    assert result == (int, str, MyEnum, "test")


# LLM-generated content at query #18
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test with string
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("custom_type") == ["custom_type"]
    
    # Test with Enum (preserved type)
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    
    assert maybe_parse_user_type(Color) == [Color]
    
    # Test with iterable of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    assert maybe_parse_user_type((float, bool)) == (float, bool)
    
    # Test with nested iterables
    assert maybe_parse_user_type([int, [str, float]]) == (int, str, float)
    assert maybe_parse_user_type([[int, str], float]) == (int, str, float)
    
    # Test with mixed types and strings
    assert maybe_parse_user_type([int, "custom", str]) == (int, "custom", str)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    assert maybe_parse_user_type(()) == ()
    
    # Test that non-type, non-string, non-iterable raises TypeError
    try:
        maybe_parse_user_type(42)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    try:
        maybe_parse_user_type(None)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test with complex nested structure
    class MyEnum(Enum):
        VALUE = 1
    
    result = maybe_parse_user_type([int, [str, MyEnum], "test"])
    assert result == (int, str, MyEnum, "test")
    
    # Test that regular iterable types are not preserved
    class MyIterable(Iterable):
        def __iter__(self):
            return iter([])
    
    # MyIterable is a type but also Iterable, so it should be treated as iterable
    # and its contents (empty) should be parsed
    assert maybe_parse_user_type(MyIterable) == ()


# LLM-generated content at query #19
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns a single boolean result
    def single_bool_invariant(x):
        return x > 0, "Value must be positive"
    
    wrapped = wrap_invariant(single_bool_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant returns multiple results that need merging
    def multi_result_invariant(x):
        return [(x > 0, "positive"), (x < 10, "less than 10")]
    
    wrapped = wrap_invariant(multi_result_invariant)
    
    # All conditions satisfied
    result = wrapped(5)
    assert result == (True, ())
    
    # One condition failed
    result = wrapped(-1)
    assert result == (False, ("Value must be positive",))
    
    # Multiple conditions failed
    result = wrapped(15)
    assert result == (False, ("Value must be positive", "Value must be less than 10"))
    
    # Test 3: Invariant with no arguments
    def no_arg_invariant():
        return True, "Always true"
    
    wrapped = wrap_invariant(no_arg_invariant)
    result = wrapped()
    assert result == (True, "Always true")
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(value, threshold=0):
        return value > threshold, f"Value must be greater than {threshold}"
    
    wrapped = wrap_invariant(kwarg_invariant)
    result = wrapped(5, threshold=3)
    assert result == (True, "Value must be greater than 3")
    
    result = wrapped(2, threshold=3)
    assert result == (False, "Value must be greater than 3")
    
    # Test 5: Invariant returns empty list
    def empty_list_invariant(x):
        return []
    
    wrapped = wrap_invariant(empty_list_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 6: Invariant returns tuple of tuples
    def tuple_of_tuples_invariant(x):
        return ((x > 0, "positive"), (x % 2 == 0, "even"))
    
    wrapped = wrap_invariant(tuple_of_tuples_invariant)
    result = wrapped(4)
    assert result == (True, ())
    
    result = wrapped(-2)
    assert result == (False, ("Value must be positive",))
    
    result = wrapped(3)
    assert result == (False, ("Value must be even",))


# LLM-generated content at query #20
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a single boolean result
    def single_bool_invariant(x):
        return x > 0, "Value must be positive"
    
    wrapped = wrap_invariant(single_bool_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results (list of tuples)
    def multi_result_invariant(x):
        return [(x > 0, "positive"), (x < 10, "less than 10")]
    
    wrapped = wrap_invariant(multi_result_invariant)
    
    # All conditions met
    result = wrapped(5)
    assert result == (True, ())
    
    # One condition failed
    result = wrapped(15)
    assert result == (False, ("less than 10",))
    
    # Both conditions failed
    result = wrapped(-5)
    assert result == (False, ("positive", "less than 10"))
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(x, y):
        return [(x > 0, "x positive"), (y > 0, "y positive")]
    
    wrapped = wrap_invariant(multi_arg_invariant)
    result = wrapped(5, 3)
    assert result == (True, ())
    
    result = wrapped(-1, 3)
    assert result == (False, ("x positive",))
    
    result = wrapped(-1, -2)
    assert result == (False, ("x positive", "y positive"))
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(x, threshold=0):
        return x > threshold, f"x > {threshold}"
    
    wrapped = wrap_invariant(kwarg_invariant)
    result = wrapped(5, threshold=3)
    assert result == (True, "x > 3")
    
    result = wrapped(2, threshold=3)
    assert result == (False, "x > 3")
    
    # Test 5: Invariant that returns empty tuple for success
    def empty_success_invariant(x):
        if x > 0:
            return True, ()
        else:
            return False, "failed"
    
    wrapped = wrap_invariant(empty_success_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    result = wrapped(-1)
    assert result == (False, "failed")


# LLM-generated content at query #21
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a simple boolean result
    def simple_invariant(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results to be merged
    def complex_invariant(value):
        return [
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # All conditions pass
    result = wrapped(4)
    assert result == (True, ())
    
    # One condition fails
    result = wrapped(11)
    assert result[0] == False
    assert "Value must be less than 10" in result[1]
    
    # Multiple conditions fail
    result = wrapped(-2)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be positive" in result[1]
    assert "Value must be even" in result[1]
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(a, b):
        return [
            (a > 0, "A must be positive"),
            (b > 0, "B must be positive"),
            (a < b, "A must be less than B")
        ]
    
    wrapped = wrap_invariant(multi_arg_invariant)
    
    result = wrapped(2, 5)
    assert result == (True, ())
    
    result = wrapped(5, 2)
    assert result[0] == False
    assert "A must be less than B" in result[1]
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(value, threshold=10):
        return value < threshold, f"Value must be less than {threshold}"
    
    wrapped = wrap_invariant(kwarg_invariant)
    
    result = wrapped(5, threshold=10)
    assert result == (True, "Value must be less than 10")
    
    result = wrapped(15, threshold=10)
    assert result == (False, "Value must be less than 10")
    
    # Test 5: Empty result list (edge case)
    def empty_invariant(value):
        return []
    
    wrapped = wrap_invariant(empty_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 6: Single tuple in list
    def single_tuple_invariant(value):
        return [(value > 0, "Positive")]
    
    wrapped = wrap_invariant(single_tuple_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    result = wrapped(-5)
    assert result == (False, ("Positive",))


# LLM-generated content at query #22
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a simple boolean result
    def simple_invariant(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results that need merging
    def complex_invariant(value):
        return [
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # Test case where all conditions pass
    result = wrapped(4)
    assert result == (True, ())
    
    # Test case where some conditions fail
    result = wrapped(11)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be less than 10" in result[1]
    assert "Value must be even" in result[1]
    
    # Test case where all conditions fail
    result = wrapped(-1)
    assert result[0] == False
    assert len(result[1]) == 3
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(a, b):
        return [
            (a > 0, "a must be positive"),
            (b > 0, "b must be positive"),
            (a + b < 100, "sum must be less than 100")
        ]
    
    wrapped = wrap_invariant(multi_arg_invariant)
    
    result = wrapped(10, 20)
    assert result == (True, ())
    
    result = wrapped(-10, 200)
    assert result[0] == False
    assert len(result[1]) == 2
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(value, threshold=10):
        return [
            (value > 0, "Value must be positive"),
            (value < threshold, f"Value must be less than {threshold}")
        ]
    
    wrapped = wrap_invariant(kwarg_invariant)
    
    result = wrapped(5, threshold=10)
    assert result == (True, ())
    
    result = wrapped(15, threshold=10)
    assert result[0] == False
    assert len(result[1]) == 1
    
    # Test 5: Empty result list (edge case)
    def empty_invariant(value):
        return []
    
    wrapped = wrap_invariant(empty_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 6: Single tuple in list (edge case)
    def single_invariant(value):
        return [(value > 0, "Positive")]
    
    wrapped = wrap_invariant(single_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    result = wrapped(-5)
    assert result == (False, ("Positive",))


# LLM-generated content at query #23
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def invariant_single_true(x):
        return True, "ok"
    
    wrapped = wrap_invariant(invariant_single_true)
    result = wrapped(5)
    assert result == (True, "ok")
    
    # Test 2: Invariant returns single boolean false result
    def invariant_single_false(x):
        return False, "error"
    
    wrapped = wrap_invariant(invariant_single_false)
    result = wrapped(5)
    assert result == (False, "error")
    
    # Test 3: Invariant returns multiple results that need merging (all true)
    def invariant_multiple_true(x):
        return [(True, "ok1"), (True, "ok2")]
    
    wrapped = wrap_invariant(invariant_multiple_true)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 4: Invariant returns multiple results with some false
    def invariant_multiple_mixed(x):
        return [(True, "ok1"), (False, "error1"), (True, "ok2"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_multiple_mixed)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "error1" in result[1]
    assert "error2" in result[1]
    
    # Test 5: Invariant returns multiple results all false
    def invariant_multiple_false(x):
        return [(False, "error1"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_multiple_false)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "error1" in result[1]
    assert "error2" in result[1]
    
    # Test 6: Invariant with multiple arguments
    def invariant_multi_args(x, y, z=0):
        if x > y + z:
            return True, "x is greater"
        return False, "x is not greater"
    
    wrapped = wrap_invariant(invariant_multi_args)
    result = wrapped(10, 5, z=2)
    assert result == (True, "x is greater")
    
    result = wrapped(5, 10)
    assert result == (False, "x is not greater")
    
    # Test 7: Invariant with keyword arguments
    def invariant_kwargs(**kwargs):
        if kwargs.get('valid', False):
            return True, "valid"
        return False, "invalid"
    
    wrapped = wrap_invariant(invariant_kwargs)
    result = wrapped(valid=True)
    assert result == (True, "valid")
    
    result = wrapped(valid=False)
    assert result == (False, "invalid")


# LLM-generated content at query #24
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a single boolean result
    def simple_invariant(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results to be merged
    def complex_invariant(value):
        return [
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # All conditions pass
    result = wrapped(4)
    assert result == (True, ())
    
    # One condition fails
    result = wrapped(11)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be less than 10" in result[1]
    assert "Value must be even" in result[1]
    
    # Multiple conditions fail
    result = wrapped(-2)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be positive" in result[1]
    assert "Value must be even" in result[1]
    
    # Test 3: Invariant with no arguments
    def no_arg_invariant():
        return True, "Always passes"
    
    wrapped = wrap_invariant(no_arg_invariant)
    result = wrapped()
    assert result == (True, "Always passes")
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(value, threshold=5):
        return value > threshold, f"Value must be greater than {threshold}"
    
    wrapped = wrap_invariant(kwarg_invariant)
    result = wrapped(10, threshold=8)
    assert result == (True, "Value must be greater than 8")
    
    result = wrapped(3, threshold=1)
    assert result == (True, "Value must be greater than 1")
    
    result = wrapped(2, threshold=5)
    assert result == (False, "Value must be greater than 5")
    
    # Test 5: Invariant that returns empty tuple for success
    def empty_success_invariant(value):
        if value > 0:
            return True, ()
        else:
            return False, "Failed"
    
    wrapped = wrap_invariant(empty_success_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    result = wrapped(-1)
    assert result == (False, "Failed")


# LLM-generated content at query #25
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def invariant_single_true(x):
        return True, "All good"
    
    wrapped = wrap_invariant(invariant_single_true)
    result = wrapped(5)
    assert result == (True, "All good")
    
    # Test 2: Invariant returns single false result
    def invariant_single_false(x):
        return False, "Something wrong"
    
    wrapped = wrap_invariant(invariant_single_false)
    result = wrapped(5)
    assert result == (False, "Something wrong")
    
    # Test 3: Invariant returns multiple results that need merging (all true)
    def invariant_multiple_true(x):
        return [(True, "Check1 passed"), (True, "Check2 passed")]
    
    wrapped = wrap_invariant(invariant_multiple_true)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 4: Invariant returns multiple results with some false
    def invariant_multiple_mixed(x):
        return [(True, "Check1 passed"), (False, "Check2 failed"), (True, "Check3 passed")]
    
    wrapped = wrap_invariant(invariant_multiple_mixed)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 1
    assert result[1][0] == "Check2 failed"
    
    # Test 5: Invariant returns multiple results with all false
    def invariant_multiple_all_false(x):
        return [(False, "Check1 failed"), (False, "Check2 failed")]
    
    wrapped = wrap_invariant(invariant_multiple_all_false)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 2
    assert result[1][0] == "Check1 failed"
    assert result[1][1] == "Check2 failed"
    
    # Test 6: Invariant with multiple arguments
    def invariant_with_args(x, y, z=10):
        return x > y and x < z, "Range check"
    
    wrapped = wrap_invariant(invariant_with_args)
    result = wrapped(5, 3, z=10)
    assert result == (True, "Range check")
    
    result = wrapped(5, 3, z=6)
    assert result == (False, "Range check")
    
    # Test 7: Invariant with keyword arguments
    def invariant_with_kwargs(**kwargs):
        return kwargs.get('valid', False), "Validity check"
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(valid=True)
    assert result == (True, "Validity check")
    
    result = wrapped(valid=False)
    assert result == (False, "Validity check")


# LLM-generated content at query #26
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def invariant_single_true(*args, **kwargs):
        return True, "success"
    
    wrapped = wrap_invariant(invariant_single_true)
    result = wrapped()
    assert result == (True, "success")
    
    # Test 2: Invariant returns single boolean false result
    def invariant_single_false(*args, **kwargs):
        return False, "failure"
    
    wrapped = wrap_invariant(invariant_single_false)
    result = wrapped()
    assert result == (False, "failure")
    
    # Test 3: Invariant returns multiple results that need merging (all true)
    def invariant_multiple_true(*args, **kwargs):
        return [(True, "ok1"), (True, "ok2"), (True, "ok3")]
    
    wrapped = wrap_invariant(invariant_multiple_true)
    result = wrapped()
    assert result == (True, ())
    
    # Test 4: Invariant returns multiple results with some false
    def invariant_multiple_mixed(*args, **kwargs):
        return [(True, "ok1"), (False, "error1"), (True, "ok2"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_multiple_mixed)
    result = wrapped()
    assert result[0] == False
    assert len(result[1]) == 2
    assert "error1" in result[1]
    assert "error2" in result[1]
    
    # Test 5: Invariant returns multiple results all false
    def invariant_multiple_false(*args, **kwargs):
        return [(False, "err1"), (False, "err2")]
    
    wrapped = wrap_invariant(invariant_multiple_false)
    result = wrapped()
    assert result[0] == False
    assert len(result[1]) == 2
    assert "err1" in result[1]
    assert "err2" in result[1]
    
    # Test 6: Invariant with arguments
    def invariant_with_args(x, y):
        if x > y:
            return True, f"{x} > {y}"
        else:
            return False, f"{x} <= {y}"
    
    wrapped = wrap_invariant(invariant_with_args)
    result = wrapped(5, 3)
    assert result == (True, "5 > 3")
    
    result = wrapped(2, 4)
    assert result == (False, "2 <= 4")
    
    # Test 7: Invariant with keyword arguments
    def invariant_with_kwargs(value, threshold=10):
        if value > threshold:
            return True, f"{value} > {threshold}"
        else:
            return False, f"{value} <= {threshold}"
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(15, threshold=10)
    assert result == (True, "15 > 10")
    
    result = wrapped(5, threshold=10)
    assert result == (False, "5 <= 10")
    
    # Test 8: Complex nested structure that needs merging
    def invariant_complex(*args, **kwargs):
        return [
            (True, "level1_ok"),
            [(True, "level2_ok1"), (False, "level2_err1")],
            (False, "level1_err")
        ]
    
    wrapped = wrap_invariant(invariant_complex)
    result = wrapped()
    assert result[0] == False
    assert len(result[1]) == 2
    assert "level2_err1" in result[1]
    assert "level1_err" in result[1]


# LLM-generated content at query #27
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a single boolean result
    def simple_invariant(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    
    # Test with positive value
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    # Test with negative value
    result = wrapped(-5)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results that need merging
    def complex_invariant(value):
        return [
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # Test with value that satisfies all conditions
    result = wrapped(4)
    assert result == (True, ())
    
    # Test with value that fails one condition
    result = wrapped(-2)
    assert result[0] == False
    assert len(result[1]) == 1
    assert "Value must be positive" in str(result[1][0])
    
    # Test with value that fails multiple conditions
    result = wrapped(11)
    assert result[0] == False
    assert len(result[1]) == 2
    errors = [str(e) for e in result[1]]
    assert any("less than 10" in e for e in errors)
    assert any("must be even" in e for e in errors)
    
    # Test 3: Invariant that returns empty list
    def empty_invariant(value):
        return []
    
    wrapped = wrap_invariant(empty_invariant)
    result = wrapped(42)
    assert result == (True, ())
    
    # Test 4: Invariant with no arguments
    def no_arg_invariant():
        return True, "Always valid"
    
    wrapped = wrap_invariant(no_arg_invariant)
    result = wrapped()
    assert result == (True, "Always valid")
    
    # Test 5: Invariant with keyword arguments
    def kwarg_invariant(value, threshold=5):
        return value > threshold, f"Value must be greater than {threshold}"
    
    wrapped = wrap_invariant(kwarg_invariant)
    result = wrapped(10, threshold=8)
    assert result == (True, "Value must be greater than 8")
    
    result = wrapped(10, threshold=12)
    assert result == (False, "Value must be greater than 12")


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CheckedType_serialize():
    # Test 1: Check that CheckedType is abstract and cannot be instantiated
    try:
        CheckedType()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass

    # Test 2: Create a concrete subclass and test serialize method
    class ConcreteCheckedType(CheckedType):
        def __init__(self, data):
            self.data = data
        
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls(source_data)
        
        def serialize(self, format=None):
            if format == "json":
                return f'{{"data": {self.data}}}'
            return str(self.data)

    # Test 3: Test serialize with default format
    obj = ConcreteCheckedType(42)
    result = obj.serialize()
    assert result == "42"
    
    # Test 4: Test serialize with specific format
    obj = ConcreteCheckedType("test")
    result = obj.serialize(format="json")
    assert result == '{"data": test}'
    
    # Test 5: Test serialize with different data types
    obj = ConcreteCheckedType([1, 2, 3])
    result = obj.serialize()
    assert result == "[1, 2, 3]"
    
    # Test 6: Test that serialize method can be overridden properly
    class AnotherCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
        
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls(source_data)
        
        def serialize(self, format=None):
            return {"value": self.value, "format": format}

    obj = AnotherCheckedType("test_value")
    result = obj.serialize(format="custom")
    assert result == {"value": "test_value", "format": "custom"}
    
    # Test 7: Test serialize with None format
    obj = ConcreteCheckedType(None)
    result = obj.serialize(format=None)
    assert result == "None"


# LLM-generated content at query #2
#--------------------------

```python
def test_CheckedType_serialize():
    # Test 1: Check that CheckedType is an abstract class that cannot be instantiated
    try:
        obj = CheckedType()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError as e:
        assert "abstract" in str(e).lower()
    
    # Test 2: Create a concrete subclass and test serialize method
    class ConcreteCheckedType(CheckedType):
        def __init__(self, data):
            self.data = data
        
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls(source_data)
        
        def serialize(self, format=None):
            return {"data": self.data, "format": format}
    
    # Test 3: Test serialize method with no format specified
    obj = ConcreteCheckedType("test_data")
    result = obj.serialize()
    assert result == {"data": "test_data", "format": None}
    
    # Test 4: Test serialize method with format specified
    obj = ConcreteCheckedType(123)
    result = obj.serialize(format="json")
    assert result == {"data": 123, "format": "json"}
    
    # Test 5: Test serialize method with complex data
    obj = ConcreteCheckedType({"key": "value", "list": [1, 2, 3]})
    result = obj.serialize(format="yaml")
    assert result == {"data": {"key": "value", "list": [1, 2, 3]}, "format": "yaml"}
    
    # Test 6: Test that serialize method can be overridden with different behavior
    class AnotherCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
        
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls(source_data)
        
        def serialize(self, format=None):
            if format == "string":
                return str(self.value)
            return self.value
    
    obj = AnotherCheckedType(42)
    assert obj.serialize() == 42
    assert obj.serialize(format="string") == "42"


# LLM-generated content at query #3
#--------------------------

```python
def test_CheckedType_serialize():
    # Test 1: Check that CheckedType is abstract and cannot be instantiated
    try:
        CheckedType()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass

    # Test 2: Create a concrete subclass and test serialize method
    class ConcreteCheckedType(CheckedType):
        def __init__(self, data):
            self.data = data
        
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls(source_data)
        
        def serialize(self, format=None):
            if format == "json":
                return {"data": self.data}
            return self.data

    # Test 3: Test serialize with default format
    obj = ConcreteCheckedType("test_data")
    result = obj.serialize()
    assert result == "test_data"

    # Test 4: Test serialize with specific format
    obj = ConcreteCheckedType({"key": "value"})
    result = obj.serialize(format="json")
    assert result == {"data": {"key": "value"}}

    # Test 5: Test serialize with different data types
    obj = ConcreteCheckedType(123)
    result = obj.serialize()
    assert result == 123

    obj = ConcreteCheckedType([1, 2, 3])
    result = obj.serialize()
    assert result == [1, 2, 3]

    # Test 6: Test that serialize method can be overridden properly
    class AnotherCheckedType(CheckedType):
        def __init__(self, value):
            self.value = value
        
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls(source_data)
        
        def serialize(self, format=None):
            return f"Serialized: {self.value}"

    obj = AnotherCheckedType("test")
    result = obj.serialize()
    assert result == "Serialized: test"

    # Test 7: Test that abstract methods are properly defined
    assert hasattr(CheckedType, 'create')
    assert hasattr(CheckedType, 'serialize')
    assert callable(CheckedType.create)
    assert callable(CheckedType.serialize)


# LLM-generated content at query #4
#--------------------------

```python
def test_store_invariants():
    # Test 1: Basic functionality - stores invariants from dict
    def invariant1(obj):
        return True, ()
    
    dct = {'_invariant': invariant1}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert '_checked_invariants' in dct
    assert len(dct['_checked_invariants']) == 1
    assert callable(dct['_checked_invariants'][0])
    
    # Test 2: Multiple invariants in dict
    def invariant2(obj):
        return False, "error2"
    
    dct = {'_invariant': [invariant1, invariant2]}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2
    assert all(callable(inv) for inv in dct['_checked_invariants'])
    
    # Test 3: Inheritance from single base class
    class Base:
        _invariant = invariant1
    
    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 1
    assert dct['_checked_invariants'][0].__name__ == 'f'  # Wrapped invariant
    
    # Test 4: Inheritance from multiple base classes
    class Base1:
        _invariant = invariant1
    
    class Base2:
        _invariant = invariant2
    
    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2
    
    # Test 5: Override invariants in subclass
    def invariant3(obj):
        return True, "custom"
    
    class Base:
        _invariant = invariant1
    
    dct = {'_invariant': invariant3}
    bases = (Base,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2  # Both base and subclass invariants
    # Subclass invariant should come first
    result = dct['_checked_invariants'][0](None)
    assert result == (True, ())  # invariant3 wrapped
    
    # Test 6: Deep inheritance chain
    class GrandParent:
        _invariant = invariant1
    
    class Parent(GrandParent):
        _invariant = invariant2
    
    class Child(Parent):
        pass
    
    dct = {}
    bases = (Child,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    # Should get invariants from all ancestors
    assert len(dct['_checked_invariants']) == 2
    
    # Test 7: Diamond inheritance - ensure no duplicates
    class BaseA:
        _invariant = invariant1
    
    class BaseB(BaseA):
        pass
    
    class BaseC(BaseA):
        pass
    
    class Derived(BaseB, BaseC):
        pass
    
    dct = {}
    bases = (Derived,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    # Should only get invariant1 once
    assert len(dct['_checked_invariants']) == 1
    
    # Test 8: Non-callable invariant raises TypeError
    dct = {'_invariant': "not a callable"}
    bases = ()
    
    try:
        store_invariants(dct, bases, '_checked_invariants', '_invariant')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Invariants must be callable" in str(e)
    
    # Test 9: Mixed callable and non-callable in inheritance chain
    class BaseGood:
        _invariant = invariant1
    
    class BaseBad:
        _invariant = "not callable"
    
    dct = {}
    bases = (BaseGood, BaseBad)
    
    try:
        store_invariants(dct, bases, '_checked_invariants', '_invariant')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Invariants must be callable" in str(e)
    
    # Test 10: Wrap invariant that returns multiple results
    def multi_invariant(obj):
        return [(True, "ok1"), (False, "error1"), (True, "ok2")]
    
    dct = {'_invariant': multi_invariant}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    wrapped_inv = dct['_checked_invariants'][0]
    result = wrapped_inv(None)
    assert result == (False, ("error1",))  # Merged result
    
    # Test 11: Wrap invariant that already returns bool, tuple
    def simple_invariant(obj):
        return False, "simple error"
    
    dct = {'_invariant': simple_invariant}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    wrapped_inv = dct['_checked_invariants'][0]
    result = wrapped_inv(None)
    assert result == (False, ("simple error",))
    
    # Test 12: Empty invariants
    dct = {}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert '_checked_invariants' in dct
    assert dct['_checked_invariants'] == ()
    
    # Test 13: Invariant with no return value (should fail when called)
    def no_return_invariant(obj):
        pass
    
    dct = {'_invariant': no_return_invariant}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    # The invariant is stored but will fail when called
    assert len(dct['_checked_invariants']) == 1


# LLM-generated content at query #5
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a simple boolean result
    def simple_invariant(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results that need merging
    def complex_invariant(value):
        return [
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # Test case where all conditions pass
    result = wrapped(4)
    assert result == (True, ())
    
    # Test case where some conditions fail
    result = wrapped(11)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be less than 10" in result[1]
    assert "Value must be even" in result[1]
    
    # Test case where all conditions fail
    result = wrapped(-3)
    assert result[0] == False
    assert len(result[1]) == 3
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(a, b):
        return [
            (a > 0, "a must be positive"),
            (b > 0, "b must be positive"),
            (a + b < 20, "sum must be less than 20")
        ]
    
    wrapped = wrap_invariant(multi_arg_invariant)
    result = wrapped(5, 10)
    assert result == (True, ())
    
    result = wrapped(15, 10)
    assert result[0] == False
    assert "sum must be less than 20" in result[1]
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(value, threshold=5):
        return value > threshold, f"Value must be greater than {threshold}"
    
    wrapped = wrap_invariant(kwarg_invariant)
    result = wrapped(10, threshold=5)
    assert result == (True, "Value must be greater than 5")
    
    result = wrapped(3, threshold=5)
    assert result == (False, "Value must be greater than 5")
    
    # Test 5: Invariant that returns empty list
    def empty_invariant(value):
        return []
    
    wrapped = wrap_invariant(empty_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 6: Invariant that returns single tuple in list
    def single_tuple_invariant(value):
        return [(value > 0, "Positive")]
    
    wrapped = wrap_invariant(single_tuple_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    result = wrapped(-5)
    assert result == (False, ("Positive",))


# LLM-generated content at query #6
#--------------------------

```python
def test_store_invariants():
    # Test 1: Basic functionality - store invariants from dict
    def invariant1(obj):
        return True, ()
    
    def invariant2(obj):
        return False, "error2"
    
    dct = {'_invariant': invariant1}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert '_checked_invariants' in dct
    assert len(dct['_checked_invariants']) == 1
    assert callable(dct['_checked_invariants'][0])
    
    # Test 2: Multiple invariants in dict
    dct = {'_invariant': [invariant1, invariant2]}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2
    assert all(callable(inv) for inv in dct['_checked_invariants'])
    
    # Test 3: Inheritance from single base class
    class Base:
        _invariant = invariant1
    
    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 1
    assert dct['_checked_invariants'][0](None) == (True, ())
    
    # Test 4: Inheritance from multiple base classes
    class Base1:
        _invariant = invariant1
    
    class Base2:
        _invariant = invariant2
    
    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2
    
    # Test 5: Override invariants in subclass
    def invariant3(obj):
        return True, "custom"
    
    class Base:
        _invariant = invariant1
    
    dct = {'_invariant': invariant3}
    bases = (Base,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert len(dct['_checked_invariants']) == 2
    # Order should be: subclass first, then base class
    assert dct['_checked_invariants'][0](None) == (True, ())
    
    # Test 6: Wrap invariant that returns multiple results
    def multi_invariant(obj):
        return [(True, "ok1"), (False, "error1"), (True, "ok2")]
    
    dct = {'_invariant': multi_invariant}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    wrapped_invariant = dct['_checked_invariants'][0]
    result = wrapped_invariant(None)
    assert result == (False, ("error1",))
    
    # Test 7: Wrap invariant that already returns bool, tuple
    dct = {'_invariant': invariant2}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    wrapped_invariant = dct['_checked_invariants'][0]
    result = wrapped_invariant(None)
    assert result == (False, "error2")
    
    # Test 8: Diamond inheritance - ensure no duplicates
    class Root:
        _invariant = invariant1
    
    class Middle1(Root):
        pass
    
    class Middle2(Root):
        pass
    
    class Leaf(Middle1, Middle2):
        pass
    
    dct = {}
    bases = (Leaf,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    # Should only have one invariant, not duplicate from Root
    assert len(dct['_checked_invariants']) == 1
    
    # Test 9: Non-callable invariant should raise TypeError
    dct = {'_invariant': "not a callable"}
    bases = ()
    
    try:
        store_invariants(dct, bases, '_checked_invariants', '_invariant')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Invariants must be callable" in str(e)
    
    # Test 10: Empty invariants
    dct = {}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    assert '_checked_invariants' in dct
    assert dct['_checked_invariants'] == ()
    
    # Test 11: Complex inheritance chain
    class A:
        _invariant = lambda self: (True, "A")
    
    class B(A):
        _invariant = lambda self: (True, "B")
    
    class C(A):
        pass
    
    class D(B, C):
        _invariant = lambda self: (True, "D")
    
    dct = {}
    bases = (D,)
    store_invariants(dct, bases, '_checked_invariants', '_invariant')
    
    # Should have invariants from D, B, A (C has none)
    assert len(dct['_checked_invariants']) == 3


# LLM-generated content at query #7
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns simple boolean result
    def simple_invariant(x):
        return x > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    assert wrapped(5) == (True, "Value must be positive")
    assert wrapped(-1) == (False, "Value must be positive")
    
    # Test 2: Invariant returns multiple results that need merging
    def complex_invariant(x):
        return [
            (x > 0, "Value must be positive"),
            (x < 10, "Value must be less than 10"),
            (x % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # All conditions satisfied
    result = wrapped(4)
    assert result == (True, ())
    
    # One condition failed
    result = wrapped(-2)
    assert result[0] == False
    assert len(result[1]) == 1
    assert "Value must be positive" in result[1][0]
    
    # Multiple conditions failed
    result = wrapped(11)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be less than 10" in result[1][0] or "Value must be less than 10" in result[1][1]
    assert "Value must be even" in result[1][0] or "Value must be even" in result[1][1]
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(x, y):
        return [
            (x > 0, "x must be positive"),
            (y > 0, "y must be positive"),
            (x + y < 20, "sum must be less than 20")
        ]
    
    wrapped = wrap_invariant(multi_arg_invariant)
    
    # All conditions satisfied
    result = wrapped(5, 10)
    assert result == (True, ())
    
    # Some conditions failed
    result = wrapped(-5, 25)
    assert result[0] == False
    assert len(result[1]) == 2
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(x, threshold=10):
        return x < threshold, f"Value must be less than {threshold}"
    
    wrapped = wrap_invariant(kwarg_invariant)
    assert wrapped(5, threshold=10) == (True, "Value must be less than 10")
    assert wrapped(15, threshold=10) == (False, "Value must be less than 10")
    
    # Test 5: Empty result list (edge case)
    def empty_invariant(x):
        return []
    
    wrapped = wrap_invariant(empty_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 6: Single tuple in list (edge case)
    def single_invariant(x):
        return [(x > 0, "positive")]
    
    wrapped = wrap_invariant(single_invariant)
    assert wrapped(5) == (True, ())
    assert wrapped(-5) == (False, ("positive",))


# LLM-generated content at query #8
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a single boolean result
    def simple_invariant(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    
    # Test with positive value
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    # Test with negative value
    result = wrapped(-3)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results (iterable of tuples)
    def complex_invariant(value):
        return [
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # Test with value that satisfies all conditions
    result = wrapped(4)
    assert result == (True, ())
    
    # Test with value that fails some conditions
    result = wrapped(11)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be less than 10" in result[1]
    assert "Value must be even" in result[1]
    
    # Test 3: Invariant that returns empty iterable
    def empty_invariant(value):
        return []
    
    wrapped = wrap_invariant(empty_invariant)
    result = wrapped(42)
    assert result == (True, ())
    
    # Test 4: Invariant with no arguments
    def no_arg_invariant():
        return True, "Always valid"
    
    wrapped = wrap_invariant(no_arg_invariant)
    result = wrapped()
    assert result == (True, "Always valid")
    
    # Test 5: Invariant with keyword arguments
    def kwarg_invariant(value, threshold=5):
        return value > threshold, f"Value must be greater than {threshold}"
    
    wrapped = wrap_invariant(kwarg_invariant)
    result = wrapped(10, threshold=8)
    assert result == (True, "Value must be greater than 8")
    
    result = wrapped(3, threshold=1)
    assert result == (True, "Value must be greater than 1")
    
    # Test 6: Nested complex results
    def nested_invariant(value):
        return [
            (value > 0, "Positive"),
            [
                (value < 10, "Less than 10"),
                (value % 3 == 0, "Divisible by 3")
            ]
        ]
    
    wrapped = wrap_invariant(nested_invariant)
    result = wrapped(9)
    assert result == (True, ())
    
    result = wrapped(15)
    assert result[0] == False
    assert len(result[1]) == 1  # Only "Less than 10" fails


# LLM-generated content at query #9
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test with string
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("custom_type") == ["custom_type"]
    
    # Test with Enum subclass (preserved type)
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    
    assert maybe_parse_user_type(Color) == [Color]
    
    # Test with iterable of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    assert maybe_parse_user_type((float, bool)) == (float, bool)
    
    # Test with nested iterables
    assert maybe_parse_user_type([int, [str, float]]) == (int, str, float)
    assert maybe_parse_user_type([[int, str], float]) == (int, str, float)
    
    # Test with mixed types and strings
    assert maybe_parse_user_type([int, "custom", str]) == (int, "custom", str)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    
    # Test that non-type, non-string, non-iterable raises TypeError
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(42)
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)
    with pytest.raises(TypeError):
        maybe_parse_user_type(object())
    
    # Test that iterable types that are not preserved get flattened
    class CustomIterable(Iterable):
        def __iter__(self):
            return iter([1, 2, 3])
    
    # CustomIterable is a type and iterable, but not preserved
    # It should be treated as an iterable and its contents should be parsed
    # But since it's not instantiated, it will be treated as an empty iterable
    assert maybe_parse_user_type(CustomIterable) == ()


# LLM-generated content at query #10
#--------------------------

```python
def test_CheckedPSet_serialize():
    # Test 1: Basic serialization with default serializer
    class SimpleSet(CheckedPSet):
        __type__ = int
    
    s = SimpleSet([1, 2, 3])
    result = s.serialize()
    assert result == {1, 2, 3}
    assert isinstance(result, set)
    
    # Test 2: Serialization with nested CheckedType objects
    class InnerType(CheckedPSet):
        __type__ = int
    
    class OuterSet(CheckedPSet):
        __type__ = InnerType
    
    inner = InnerType([1, 2])
    outer = OuterSet([inner])
    result = outer.serialize()
    assert result == {[1, 2]}
    assert isinstance(result, set)
    assert all(isinstance(item, list) for item in result)
    
    # Test 3: Serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return str(value) + "_serialized"
    
    s = CustomSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {"1_serialized", "2_serialized", "3_serialized"}
    
    # Test 4: Serialization with format parameter
    class FormatAwareSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            if format == "double":
                return value * 2
            return value
    
    s = FormatAwareSet([1, 2, 3])
    result = s.serialize(format="double")
    assert result == {2, 4, 6}
    
    # Test 5: Serialization of empty set
    class EmptySet(CheckedPSet):
        __type__ = int
    
    s = EmptySet()
    result = s.serialize()
    assert result == set()
    
    # Test 6: Serialization preserves set semantics (no duplicates)
    class NoDupSet(CheckedPSet):
        __type__ = int
    
    s = NoDupSet([1, 1, 2, 2, 3])
    result = s.serialize()
    assert len(result) == 3
    assert result == {1, 2, 3}
    
    # Test 7: Serialization with multiple allowed types
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    s = MultiTypeSet([1, "hello", 2])
    result = s.serialize()
    assert result == {1, "hello", 2}
    
    # Test 8: Verify serializer is called for each element
    class CountingSerializerSet(CheckedPSet):
        __type__ = int
        call_count = 0
        
        def __serializer__(self, format, value):
            CountingSerializerSet.call_count += 1
            return value
    
    s = CountingSerializerSet([1, 2, 3, 4, 5])
    CountingSerializerSet.call_count = 0
    result = s.serialize()
    assert CountingSerializerSet.call_count == 5
    assert result == {1, 2, 3, 4, 5}


# LLM-generated content at query #11
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a simple boolean result
    def simple_invariant(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    assert wrapped(5) == (True, "Value must be positive")
    assert wrapped(-1) == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results that need merging
    def complex_invariant(value):
        return [
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # Test case where all conditions pass
    result = wrapped(4)
    assert result == (True, ())
    
    # Test case where some conditions fail
    result = wrapped(11)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be less than 10" in result[1]
    assert "Value must be even" in result[1]
    
    # Test case where all conditions fail
    result = wrapped(-3)
    assert result[0] == False
    assert len(result[1]) == 3
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(a, b):
        return [
            (a > 0, "a must be positive"),
            (b > 0, "b must be positive"),
            (a + b < 20, "sum must be less than 20")
        ]
    
    wrapped = wrap_invariant(multi_arg_invariant)
    assert wrapped(5, 10) == (True, ())
    result = wrapped(-5, 25)
    assert result[0] == False
    assert len(result[1]) == 2
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(value, threshold=10):
        return [
            (value > 0, "Value must be positive"),
            (value < threshold, f"Value must be less than {threshold}")
        ]
    
    wrapped = wrap_invariant(kwarg_invariant)
    assert wrapped(5) == (True, ())
    assert wrapped(15, threshold=20) == (True, ())
    result = wrapped(15, threshold=10)
    assert result[0] == False
    assert len(result[1]) == 1
    
    # Test 5: Empty result list (edge case)
    def empty_invariant(value):
        return []
    
    wrapped = wrap_invariant(empty_invariant)
    assert wrapped(5) == (True, ())
    
    # Test 6: Single tuple in list (edge case)
    def single_item_invariant(value):
        return [(value > 0, "Positive")]
    
    wrapped = wrap_invariant(single_item_invariant)
    assert wrapped(5) == (True, ())
    result = wrapped(-5)
    assert result == (False, ("Positive",))


# LLM-generated content at query #12
#--------------------------

```python
def test_CheckedPSet_serialize():
    # Test basic serialization with default serializer
    class SimpleSet(CheckedPSet):
        __type__ = int
    
    s = SimpleSet([1, 2, 3])
    result = s.serialize()
    assert result == {1, 2, 3}
    
    # Test serialization with nested CheckedType objects
    class InnerType(CheckedPSet):
        __type__ = str
    
    class OuterSet(CheckedPSet):
        __type__ = InnerType
    
    inner = InnerType(['a', 'b'])
    outer = OuterSet([inner])
    result = outer.serialize()
    assert len(result) == 1
    assert isinstance(list(result)[0], set)
    assert list(result)[0] == {'a', 'b'}
    
    # Test serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return value * 2
    
    s = CustomSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {2, 4, 6}
    
    # Test serialization with format parameter
    class FormatSet(CheckedPSet):
        __type__ = str
        
        def __serializer__(self, format, value):
            if format == 'uppercase':
                return value.upper()
            return value
    
    s = FormatSet(['a', 'b', 'c'])
    result = s.serialize('uppercase')
    assert result == {'A', 'B', 'C'}
    
    # Test serialization with empty set
    class EmptySet(CheckedPSet):
        __type__ = int
    
    s = EmptySet()
    result = s.serialize()
    assert result == set()
    
    # Test serialization preserves set semantics (no duplicates)
    class NoDupSet(CheckedPSet):
        __type__ = int
    
    s = NoDupSet([1, 1, 2, 2, 3])
    result = s.serialize()
    assert len(result) == 3
    assert result == {1, 2, 3}
    
    # Test serialization with mixed types through optional
    class OptionalSet(CheckedPSet):
        __type__ = optional(int, str)
    
    s = OptionalSet([1, 'a', None, 2])
    result = s.serialize()
    assert result == {1, 'a', None, 2}


# LLM-generated content at query #13
#--------------------------

```python
def test_CheckedPSet_serialize():
    # Test basic serialization with default serializer
    class SimpleSet(CheckedPSet):
        __type__ = int
    
    s = SimpleSet([1, 2, 3])
    result = s.serialize()
    assert result == {1, 2, 3}
    assert isinstance(result, set)
    
    # Test serialization with nested CheckedType objects
    class InnerType(CheckedPSet):
        __type__ = int
    
    class OuterSet(CheckedPSet):
        __type__ = InnerType
    
    inner = InnerSet([1, 2])
    outer = OuterSet([inner])
    result = outer.serialize()
    assert len(result) == 1
    assert isinstance(next(iter(result)), set)
    assert next(iter(result)) == {1, 2}
    
    # Test serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return str(value) + "_serialized"
    
    s = CustomSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {"1_serialized", "2_serialized", "3_serialized"}
    
    # Test serialization with format parameter
    class FormatAwareSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            if format == "json":
                return str(value)
            return value
    
    s = FormatAwareSet([1, 2, 3])
    result = s.serialize(format="json")
    assert result == {"1", "2", "3"}
    
    # Test serialization of empty set
    class EmptySet(CheckedPSet):
        __type__ = int
    
    s = EmptySet()
    result = s.serialize()
    assert result == set()
    
    # Test serialization with multiple types
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    s = MultiTypeSet([1, "hello", 3])
    result = s.serialize()
    assert result == {1, "hello", 3}
    
    # Test that serializer is called for each element
    class CountingSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return value * 2
    
    s = CountingSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {2, 4, 6}
    
    # Test inheritance of serializer
    class BaseSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return value + 10
    
    class DerivedSet(BaseSet):
        pass
    
    s = DerivedSet([1, 2, 3])
    result = s.serialize()
    assert result == {11, 12, 13}
    
    # Test that non-CheckedType objects are passed through by default serializer
    class MixedSet(CheckedPSet):
        __type__ = (int, list)
    
    s = MixedSet([1, [2, 3], 4])
    result = s.serialize()
    assert 1 in result
    assert [2, 3] in result
    assert 4 in result
    
    # Test with None values when allowed
    class OptionalSet(CheckedPSet):
        __type__ = optional(int)
    
    s = OptionalSet([1, None, 3])
    result = s.serialize()
    assert result == {1, None, 3}


# LLM-generated content at query #14
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def invariant_single_true(x):
        return True, "All good"
    
    wrapped = wrap_invariant(invariant_single_true)
    result = wrapped(5)
    assert result == (True, "All good")
    
    # Test 2: Invariant returns single false result
    def invariant_single_false(x):
        return False, "Something wrong"
    
    wrapped = wrap_invariant(invariant_single_false)
    result = wrapped(5)
    assert result == (False, "Something wrong")
    
    # Test 3: Invariant returns multiple results that need merging (all true)
    def invariant_multiple_all_true(x):
        return [(True, "Check1 passed"), (True, "Check2 passed")]
    
    wrapped = wrap_invariant(invariant_multiple_all_true)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 4: Invariant returns multiple results with some false
    def invariant_multiple_some_false(x):
        return [(True, "Check1 passed"), (False, "Check2 failed"), (True, "Check3 passed")]
    
    wrapped = wrap_invariant(invariant_multiple_some_false)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 1
    assert result[1][0] == "Check2 failed"
    
    # Test 5: Invariant returns multiple results with multiple false
    def invariant_multiple_many_false(x):
        return [(False, "Check1 failed"), (True, "Check2 passed"), (False, "Check3 failed")]
    
    wrapped = wrap_invariant(invariant_multiple_many_false)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Check1 failed" in result[1]
    assert "Check3 failed" in result[1]
    
    # Test 6: Invariant with no arguments
    def invariant_no_args():
        return True, "No args works"
    
    wrapped = wrap_invariant(invariant_no_args)
    result = wrapped()
    assert result == (True, "No args works")
    
    # Test 7: Invariant with keyword arguments
    def invariant_with_kwargs(x, y=10):
        return x > y, f"{x} > {y}"
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(15, y=5)
    assert result == (True, "15 > 5")
    
    result = wrapped(3, y=5)
    assert result == (False, "3 > 5")


# LLM-generated content at query #15
#--------------------------

```python
def test_CheckedPSet_serialize():
    # Test 1: Basic serialization with default serializer
    class SimpleSet(CheckedPSet):
        __type__ = int
    
    s = SimpleSet([1, 2, 3])
    result = s.serialize()
    assert result == {1, 2, 3}
    assert isinstance(result, set)
    
    # Test 2: Serialization with nested CheckedType objects
    class InnerType(CheckedPSet):
        __type__ = int
    
    class OuterSet(CheckedPSet):
        __type__ = InnerType
    
    inner = InnerType([1, 2])
    outer = OuterSet([inner])
    result = outer.serialize()
    assert result == {frozenset({1, 2})}
    
    # Test 3: Serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return value * 2
    
    s = CustomSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {2, 4, 6}
    
    # Test 4: Serialization with format parameter
    class FormatAwareSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            if format == 'double':
                return value * 2
            return value
    
    s = FormatAwareSet([1, 2, 3])
    result = s.serialize('double')
    assert result == {2, 4, 6}
    
    # Test 5: Serialization of empty set
    class EmptySet(CheckedPSet):
        __type__ = int
    
    s = EmptySet()
    result = s.serialize()
    assert result == set()
    
    # Test 6: Serialization with multiple types
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    s = MultiTypeSet([1, 'a', 2])
    result = s.serialize()
    assert result == {1, 'a', 2}
    
    # Test 7: Serialization preserves set semantics (no duplicates)
    class NoDupSet(CheckedPSet):
        __type__ = int
    
    s = NoDupSet([1, 2, 2, 3, 3, 3])
    result = s.serialize()
    assert len(result) == 3
    assert result == {1, 2, 3}


# LLM-generated content at query #16
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a single boolean result
    def simple_invariant(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    
    # Test with positive value
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    # Test with negative value
    result = wrapped(-5)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results
    def multi_invariant(value):
        return [
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(multi_invariant)
    
    # Test with value that passes all checks
    result = wrapped(4)
    assert result == (True, ())
    
    # Test with value that fails one check
    result = wrapped(-2)
    assert result[0] == False
    assert len(result[1]) == 1
    assert "Value must be positive" in result[1][0]
    
    # Test with value that fails multiple checks
    result = wrapped(15)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be less than 10" in result[1][0] or "Value must be less than 10" in result[1][1]
    assert "Value must be even" in result[1][0] or "Value must be even" in result[1][1]
    
    # Test 3: Invariant with multiple arguments
    def complex_invariant(a, b):
        return [
            (a > 0, "a must be positive"),
            (b > 0, "b must be positive"),
            (a + b < 100, "sum must be less than 100")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # Test passing case
    result = wrapped(10, 20)
    assert result == (True, ())
    
    # Test failing case
    result = wrapped(-5, 200)
    assert result[0] == False
    assert len(result[1]) == 2
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(value, threshold=10):
        return value < threshold, f"Value must be less than {threshold}"
    
    wrapped = wrap_invariant(kwarg_invariant)
    
    result = wrapped(5, threshold=10)
    assert result == (True, "Value must be less than 10")
    
    result = wrapped(15, threshold=10)
    assert result == (False, "Value must be less than 10")
    
    # Test 5: Invariant that already returns merged result
    def already_merged_invariant(value):
        if value > 0 and value < 10:
            return True, ()
        else:
            errors = []
            if value <= 0:
                errors.append("Value must be positive")
            if value >= 10:
                errors.append("Value must be less than 10")
            return False, tuple(errors)
    
    wrapped = wrap_invariant(already_merged_invariant)
    
    result = wrapped(5)
    assert result == (True, ())
    
    result = wrapped(-5)
    assert result == (False, ("Value must be positive",))
    
    result = wrapped(15)
    assert result == (False, ("Value must be less than 10",))


# LLM-generated content at query #17
#--------------------------

```python
def test_store_invariants():
    # Test 1: Basic invariant storage
    class BaseClass:
        def base_invariant(self):
            return True, ()
    
    dct = {}
    bases = (BaseClass,)
    store_invariants(dct, bases, '_invariants', '__invariant__')
    
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 1
    assert callable(dct['_invariants'][0])
    
    # Test 2: Multiple invariants from multiple classes
    class GrandParent:
        def grandparent_invariant(self):
            return True, ()
    
    class Parent(GrandParent):
        def parent_invariant(self):
            return False, ("error1",)
    
    class Child(Parent):
        def child_invariant(self):
            return True, ()
    
    dct = {'__invariant__': Child.child_invariant}
    bases = (Parent,)
    store_invariants(dct, bases, '_invariants', '__invariant__')
    
    assert len(dct['_invariants']) == 3
    assert all(callable(inv) for inv in dct['_invariants'])
    
    # Test 3: Invariant wrapping for tuple results
    class TestClass:
        def multi_invariant(self):
            return [(True, ()), (False, "error1"), (False, "error2")]
    
    dct = {'__invariant__': TestClass.multi_invariant}
    bases = ()
    store_invariants(dct, bases, '_invariants', '__invariant__')
    
    wrapped_invariant = dct['_invariants'][0]
    result = wrapped_invariant(None)
    assert result == (False, ("error1", "error2"))
    
    # Test 4: Simple boolean invariant remains unchanged
    class SimpleClass:
        def simple_invariant(self):
            return False, "simple_error"
    
    dct = {'__invariant__': SimpleClass.simple_invariant}
    bases = ()
    store_invariants(dct, bases, '_invariants', '__invariant__')
    
    wrapped_invariant = dct['_invariants'][0]
    result = wrapped_invariant(None)
    assert result == (False, "simple_error")
    
    # Test 5: Non-callable invariant raises TypeError
    class BadClass:
        __invariant__ = "not a callable"
    
    dct = {'__invariant__': "not a callable"}
    bases = ()
    
    try:
        store_invariants(dct, bases, '_invariants', '__invariant__')
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test 6: Empty invariants
    dct = {}
    bases = ()
    store_invariants(dct, bases, '_invariants', '__invariant__')
    assert dct['_invariants'] == ()
    
    # Test 7: Diamond inheritance - ensure no duplicates
    class A:
        def a_invariant(self):
            return True, ()
    
    class B(A):
        def b_invariant(self):
            return True, ()
    
    class C(A):
        def c_invariant(self):
            return True, ()
    
    class D(B, C):
        def d_invariant(self):
            return True, ()
    
    dct = {'__invariant__': D.d_invariant}
    bases = (B, C)
    store_invariants(dct, bases, '_invariants', '__invariant__')
    
    # Should have 4 invariants: D, B, A, C (A appears only once despite being parent of both B and C)
    assert len(dct['_invariants']) == 4
    
    # Test 8: Invariant with complex return structure
    class ComplexClass:
        def complex_invariant(self):
            return [
                (True, ()),
                [(False, "nested_error1"), (True, ())],
                (False, "error2")
            ]
    
    dct = {'__invariant__': ComplexClass.complex_invariant}
    bases = ()
    store_invariants(dct, bases, '_invariants', '__invariant__')
    
    wrapped_invariant = dct['_invariants'][0]
    result = wrapped_invariant(None)
    assert result == (False, ("nested_error1", "error2"))


# LLM-generated content at query #18
#--------------------------

```python
def test_CheckedPSet_serialize():
    # Test basic serialization with default serializer
    class SimpleSet(CheckedPSet):
        __type__ = int
    
    s = SimpleSet([1, 2, 3])
    result = s.serialize()
    assert result == {1, 2, 3}
    assert isinstance(result, set)
    
    # Test serialization with nested CheckedType objects
    class InnerType(CheckedPSet):
        __type__ = int
    
    class OuterSet(CheckedPSet):
        __type__ = InnerType
    
    inner = InnerType([1, 2])
    outer = OuterSet([inner])
    result = outer.serialize()
    assert result == {[1, 2]}
    assert isinstance(result, set)
    assert all(isinstance(item, list) for item in result)
    
    # Test serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return str(value)
    
    s = CustomSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {'1', '2', '3'}
    
    # Test serialization with format parameter
    class FormatSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            if format == 'hex':
                return hex(value)
            return value
    
    s = FormatSet([10, 20, 30])
    result = s.serialize('hex')
    assert result == {'0xa', '0x14', '0x1e'}
    
    # Test serialization with empty set
    class EmptySet(CheckedPSet):
        __type__ = int
    
    s = EmptySet()
    result = s.serialize()
    assert result == set()
    
    # Test serialization preserves order (though sets are unordered)
    class OrderedTestSet(CheckedPSet):
        __type__ = str
    
    s = OrderedTestSet(['a', 'b', 'c'])
    result = s.serialize()
    assert len(result) == 3
    assert 'a' in result
    assert 'b' in result
    assert 'c' in result
    
    # Test serialization with multiple types
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    s = MultiTypeSet([1, 'a', 2, 'b'])
    result = s.serialize()
    assert result == {1, 'a', 2, 'b'}
    
    # Test that serialize returns a new set, not the original
    class OriginalSet(CheckedPSet):
        __type__ = int
    
    s = OriginalSet([1, 2, 3])
    result = s.serialize()
    result.add(4)
    assert 4 not in s
    assert len(s) == 3


# LLM-generated content at query #19
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a single boolean result
    def simple_invariant(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results to be merged
    def complex_invariant(value):
        return [
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # All conditions pass
    result = wrapped(4)
    assert result == (True, ())
    
    # One condition fails
    result = wrapped(11)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be less than 10" in result[1]
    assert "Value must be even" in result[1]
    
    # Multiple conditions fail
    result = wrapped(-2)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be positive" in result[1]
    assert "Value must be even" in result[1]
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(a, b):
        return [
            (a > 0, "a must be positive"),
            (b > 0, "b must be positive"),
            (a + b < 100, "sum must be less than 100")
        ]
    
    wrapped = wrap_invariant(multi_arg_invariant)
    
    result = wrapped(10, 20)
    assert result == (True, ())
    
    result = wrapped(-10, 120)
    assert result[0] == False
    assert len(result[1]) == 2
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(value, threshold=10):
        return [
            (value > 0, "value must be positive"),
            (value < threshold, f"value must be less than {threshold}")
        ]
    
    wrapped = wrap_invariant(kwarg_invariant)
    
    result = wrapped(5, threshold=10)
    assert result == (True, ())
    
    result = wrapped(15, threshold=10)
    assert result[0] == False
    assert "value must be less than 10" in result[1]
    
    # Test 5: Empty result list (edge case)
    def empty_invariant(value):
        return []
    
    wrapped = wrap_invariant(empty_invariant)
    result = wrapped(5)
    assert result == (True, ())


# LLM-generated content at query #20
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test with string
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("custom_type") == ["custom_type"]
    
    # Test with Enum subclass (preserved type)
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    
    assert maybe_parse_user_type(Color) == [Color]
    
    # Test with iterable of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    assert maybe_parse_user_type((float, bool)) == (float, bool)
    
    # Test with nested iterables
    assert maybe_parse_user_type([int, [str, float]]) == (int, str, float)
    assert maybe_parse_user_type([[int, str], float]) == (int, str, float)
    
    # Test with mixed types and strings
    assert maybe_parse_user_type([int, "str", float]) == (int, "str", float)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    assert maybe_parse_user_type(()) == ()
    
    # Test that regular iterable types are not preserved
    class CustomIterable(Iterable):
        def __iter__(self):
            return iter([])
    
    result = maybe_parse_user_type(CustomIterable)
    assert len(result) == 1
    assert result[0] == CustomIterable
    
    # Test error cases
    try:
        maybe_parse_user_type(123)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    try:
        maybe_parse_user_type({"key": "value"})
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    try:
        maybe_parse_user_type(object())
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_maybe_parse_user_type():
    # Test with single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test with string
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("custom_type") == ["custom_type"]
    
    # Test with Enum subclass (preserved type)
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    
    assert maybe_parse_user_type(Color) == [Color]
    
    # Test with iterable of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    assert maybe_parse_user_type((float, bool)) == (float, bool)
    
    # Test with nested iterables
    assert maybe_parse_user_type([int, [str, float]]) == (int, str, float)
    assert maybe_parse_user_type([[int, str], float]) == (int, str, float)
    
    # Test with mixed types and strings
    assert maybe_parse_user_type([int, "custom", str]) == (int, "custom", str)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    
    # Test with single element iterable
    assert maybe_parse_user_type([int]) == (int,)
    
    # Test that non-type, non-string, non-iterable raises TypeError
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(42)
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)
    with pytest.raises(TypeError):
        maybe_parse_user_type(object())
    
    # Test that iterable types that are not preserved are treated as iterables
    class CustomIterable(Iterable):
        def __iter__(self):
            return iter([1, 2, 3])
    
    # This should be treated as an iterable, not a type
    with pytest.raises(TypeError):
        maybe_parse_user_type(CustomIterable)


# LLM-generated content at query #22
#--------------------------

```python
def test_CheckedPSet_serialize():
    # Test 1: Basic serialization with default serializer
    class SimpleSet(CheckedPSet):
        __type__ = int
    
    s = SimpleSet([1, 2, 3])
    result = s.serialize()
    assert result == {1, 2, 3}
    assert isinstance(result, set)
    
    # Test 2: Serialization with nested CheckedType objects
    class InnerType(CheckedPSet):
        __type__ = int
    
    class OuterSet(CheckedPSet):
        __type__ = InnerType
    
    inner = InnerType([1, 2])
    outer = OuterSet([inner])
    result = outer.serialize()
    assert len(result) == 1
    inner_serialized = next(iter(result))
    assert inner_serialized == {1, 2}
    
    # Test 3: Serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return value * 2
    
    s = CustomSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {2, 4, 6}
    
    # Test 4: Serialization with format parameter
    class FormatAwareSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            if format == 'double':
                return value * 2
            return value
    
    s = FormatAwareSet([1, 2, 3])
    result = s.serialize('double')
    assert result == {2, 4, 6}
    
    # Test 5: Serialization with empty set
    class EmptySet(CheckedPSet):
        __type__ = int
    
    s = EmptySet()
    result = s.serialize()
    assert result == set()
    
    # Test 6: Serialization preserves set semantics (no duplicates)
    class NoDupSet(CheckedPSet):
        __type__ = int
    
    s = NoDupSet([1, 1, 2, 2, 3])
    result = s.serialize()
    assert result == {1, 2, 3}
    assert len(result) == 3
    
    # Test 7: Serialization with multiple types
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    s = MultiTypeSet([1, "hello", 2])
    result = s.serialize()
    assert result == {1, "hello", 2}
    
    # Test 8: Serialization with None values (using optional)
    class OptionalSet(CheckedPSet):
        __type__ = optional(int)
    
    s = OptionalSet([1, None, 2])
    result = s.serialize()
    assert result == {1, None, 2}
    
    # Test 9: Serialization doesn't modify original object
    class ImmutableSet(CheckedPSet):
        __type__ = int
    
    s = ImmutableSet([1, 2, 3])
    original = set(s)
    result = s.serialize()
    assert result == original
    assert s == ImmutableSet([1, 2, 3])
    
    # Test 10: Serialization with complex nested structure
    class VectorType(CheckedPVector):
        __type__ = int
    
    class ComplexSet(CheckedPSet):
        __type__ = VectorType
    
    vec1 = VectorType([1, 2])
    vec2 = VectorType([3, 4])
    s = ComplexSet([vec1, vec2])
    result = s.serialize()
    assert len(result) == 2
    assert all(isinstance(v, list) for v in result)
    assert {1, 2} in result
    assert {3, 4} in result


# LLM-generated content at query #23
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a single boolean result
    def simple_invariant(x):
        return x > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    
    # Test with positive value
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    # Test with negative value
    result = wrapped(-3)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results (tuple of tuples)
    def complex_invariant(x):
        return ((x > 0, "Positive"), (x < 10, "Less than 10"))
    
    wrapped = wrap_invariant(complex_invariant)
    
    # Test with value that satisfies both conditions
    result = wrapped(5)
    assert result == (True, ())
    
    # Test with value that fails first condition
    result = wrapped(-2)
    assert result == (False, ("Value must be positive",))
    
    # Test with value that fails second condition
    result = wrapped(15)
    assert result == (False, ("Less than 10",))
    
    # Test with value that fails both conditions
    result = wrapped(-5)
    assert result == (False, ("Value must be positive", "Less than 10"))
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(x, y):
        return ((x > 0, "x positive"), (y > 0, "y positive"))
    
    wrapped = wrap_invariant(multi_arg_invariant)
    
    result = wrapped(5, 3)
    assert result == (True, ())
    
    result = wrapped(-1, 3)
    assert result == (False, ("x positive",))
    
    result = wrapped(-1, -2)
    assert result == (False, ("x positive", "y positive"))
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(x, threshold=0):
        return x > threshold, f"Value must be greater than {threshold}"
    
    wrapped = wrap_invariant(kwarg_invariant)
    
    result = wrapped(5, threshold=3)
    assert result == (True, "Value must be greater than 3")
    
    result = wrapped(2, threshold=3)
    assert result == (False, "Value must be greater than 3")
    
    # Test 5: Invariant that returns empty tuple for errors
    def empty_error_invariant(x):
        if x > 0:
            return True, ()
        else:
            return False, ()
    
    wrapped = wrap_invariant(empty_error_invariant)
    
    result = wrapped(5)
    assert result == (True, ())
    
    result = wrapped(-3)
    assert result == (False, ())


# LLM-generated content at query #24
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def simple_invariant(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant returns multiple results that need merging
    def complex_invariant(value):
        return [
            (value > 0, "positive"),
            (value < 10, "less than 10"),
            (value % 2 == 0, "even")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # All conditions pass
    result = wrapped(4)
    assert result == (True, ())
    
    # Some conditions fail
    result = wrapped(15)
    assert result[0] == False
    assert "positive" in result[1]
    assert "less than 10" in result[1]
    assert len(result[1]) == 2
    
    # All conditions fail
    result = wrapped(-2)
    assert result[0] == False
    assert "positive" in result[1]
    assert "less than 10" in result[1]
    assert "even" in result[1]
    assert len(result[1]) == 3
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(a, b):
        return [
            (a > 0, "a positive"),
            (b > 0, "b positive"),
            (a + b < 20, "sum less than 20")
        ]
    
    wrapped = wrap_invariant(multi_arg_invariant)
    
    result = wrapped(5, 10)
    assert result == (True, ())
    
    result = wrapped(-5, 25)
    assert result[0] == False
    assert "a positive" in result[1]
    assert "sum less than 20" in result[1]
    assert len(result[1]) == 2
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(value, threshold=10):
        return [
            (value > 0, "positive"),
            (value < threshold, f"less than {threshold}")
        ]
    
    wrapped = wrap_invariant(kwarg_invariant)
    
    result = wrapped(5, threshold=10)
    assert result == (True, ())
    
    result = wrapped(15, threshold=10)
    assert result[0] == False
    assert "less than 10" in result[1]
    
    # Test 5: Empty result list (edge case)
    def empty_invariant(value):
        return []
    
    wrapped = wrap_invariant(empty_invariant)
    result = wrapped(5)
    assert result == (True, ())


# LLM-generated content at query #25
#--------------------------

```python
def test_store_invariants():
    # Test 1: Basic invariant storage
    class BaseClass:
        def base_invariant(self, value):
            return value > 0, "Base invariant failed"
    
    dct = {}
    bases = (BaseClass,)
    store_invariants(dct, bases, '_invariants', '_invariants')
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 1
    assert callable(dct['_invariants'][0])
    
    # Test 2: Multiple invariants from multiple sources
    class Parent1:
        def parent1_invariant(self, value):
            return value < 100, "Parent1 invariant failed"
    
    class Parent2:
        def parent2_invariant(self, value):
            return value % 2 == 0, "Parent2 invariant failed"
    
    class Child(Parent1, Parent2):
        def child_invariant(self, value):
            return value != 50, "Child invariant failed"
    
    dct = {'_invariants': Child.child_invariant}
    bases = Child.__bases__
    store_invariants(dct, bases, '_invariants', '_invariants')
    assert len(dct['_invariants']) == 3
    
    # Test 3: Invariant wrapping for multiple results
    class MultiResultInvariant:
        def complex_invariant(self, value):
            return [
                (value > 0, "Must be positive"),
                (value < 10, "Must be less than 10"),
                (value != 5, "Cannot be 5")
            ]
    
    dct = {'_invariants': MultiResultInvariant.complex_invariant}
    bases = (MultiResultInvariant,)
    store_invariants(dct, bases, '_invariants', '_invariants')
    assert len(dct['_invariants']) == 1
    wrapped_invariant = dct['_invariants'][0]
    
    # Test the wrapped invariant
    result = wrapped_invariant(MultiResultInvariant(), 3)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] is True
    assert result[1] == ()
    
    result = wrapped_invariant(MultiResultInvariant(), 5)
    assert result[0] is False
    assert len(result[1]) == 1
    assert "Cannot be 5" in str(result[1][0])
    
    result = wrapped_invariant(MultiResultInvariant(), 15)
    assert result[0] is False
    assert len(result[1]) == 1
    assert "Must be less than 10" in str(result[1][0])
    
    # Test 4: Non-callable invariant should raise TypeError
    class BadInvariant:
        _invariants = "not a callable"
    
    dct = {}
    bases = (BadInvariant,)
    try:
        store_invariants(dct, bases, '_invariants', '_invariants')
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test 5: Diamond inheritance - ensure no duplicates
    class A:
        def a_invariant(self, value):
            return True, "A OK"
    
    class B(A):
        pass
    
    class C(A):
        pass
    
    class D(B, C):
        def d_invariant(self, value):
            return True, "D OK"
    
    dct = {'_invariants': D.d_invariant}
    bases = D.__bases__
    store_invariants(dct, bases, '_invariants', '_invariants')
    
    # Should have A's invariant only once, not once from B and once from C
    invariant_names = [inv.__name__ for inv in dct['_invariants']]
    assert invariant_names.count('a_invariant') == 1
    assert 'd_invariant' in invariant_names
    
    # Test 6: Simple boolean return invariant (should not be wrapped differently)
    class SimpleInvariant:
        def simple_invariant(self, value):
            return value != 0
    
    dct = {'_invariants': SimpleInvariant.simple_invariant}
    bases = (SimpleInvariant,)
    store_invariants(dct, bases, '_invariants', '_invariants')
    
    wrapped_invariant = dct['_invariants'][0]
    result = wrapped_invariant(SimpleInvariant(), 1)
    assert result == (True, ())
    
    result = wrapped_invariant(SimpleInvariant(), 0)
    assert result == (False, ())
    
    # Test 7: Empty invariants
    class NoInvariants:
        pass
    
    dct = {}
    bases = (NoInvariants,)
    store_invariants(dct, bases, '_invariants', '_invariants')
    assert dct['_invariants'] == ()


