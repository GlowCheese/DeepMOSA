####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_CheckedPSet___new__():
    # Test 1: Create empty CheckedPSet
    class SimpleSet(CheckedPSet):
        __type__ = int
    
    result = SimpleSet()
    assert isinstance(result, SimpleSet)
    assert len(result) == 0

    # Test 2: Create CheckedPSet with initial values
    result = SimpleSet([1, 2, 3])
    assert isinstance(result, SimpleSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

    # Test 3: Create CheckedPSet with duplicate values (should deduplicate)
    result = SimpleSet([1, 2, 2, 3, 3, 3])
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result

    # Test 4: Type checking - valid types
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    result = MultiTypeSet([1, "hello", 2, "world"])
    assert isinstance(result, MultiTypeSet)
    assert len(result) == 4

    # Test 5: Type checking - invalid types should raise CheckedValueTypeError
    class IntSet(CheckedPSet):
        __type__ = int
    
    try:
        IntSet([1, 2, "invalid"])
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.source_class == IntSet
        assert e.actual_type == str
        assert e.actual_value == "invalid"

    # Test 6: Invariant checking - valid values
    class PositiveSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda n: (n > 0, "Not positive")
    
    result = PositiveSet([1, 2, 3])
    assert isinstance(result, PositiveSet)
    assert len(result) == 3

    # Test 7: Invariant checking - invalid values should raise InvariantException
    try:
        PositiveSet([1, -2, 3])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0
        assert any("Not positive" in str(err) for err in e.invariant_errors)

    # Test 8: Create from existing PMap (internal structure)
    from pyrsistent import pmap
    internal_map = pmap({1: True, 2: True})
    result = SimpleSet(internal_map)
    assert isinstance(result, SimpleSet)
    assert len(result) == 2
    assert 1 in result
    assert 2 in result

    # Test 9: Create from existing CheckedPSet of same type
    original = SimpleSet([1, 2, 3])
    result = SimpleSet(original)
    assert result is original  # Should return same instance

    # Test 10: Create from existing CheckedPSet of different type
    class AnotherSet(CheckedPSet):
        __type__ = int
    
    result = AnotherSet(original)
    assert isinstance(result, AnotherSet)
    assert len(result) == 3
    assert result != original  # Different class instances

    # Test 11: Multiple invariants
    class ComplexSet(CheckedPSet):
        __type__ = int
        __invariant__ = [
            lambda n: (n > 0, "Not positive"),
            lambda n: (n < 100, "Too large")
        ]
    
    result = ComplexSet([1, 50, 99])
    assert isinstance(result, ComplexSet)
    assert len(result) == 3

    # Test 12: Multiple invariants failure
    try:
        ComplexSet([1, 150, 99])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0

    # Test 13: String type specification
    class StringTypeSet(CheckedPSet):
        __type__ = "builtins.int"
    
    result = StringTypeSet([1, 2, 3])
    assert isinstance(result, StringTypeSet)
    assert len(result) == 3

    # Test 14: Optional types
    class OptionalSet(CheckedPSet):
        __type__ = optional(int, str)
    
    result = OptionalSet([1, "hello", None, 2])
    assert isinstance(result, OptionalSet)
    assert len(result) == 4
    assert None in result

    # Test 15: Empty initial with type checking
    result = IntSet()
    assert isinstance(result, IntSet)
    assert len(result) == 0

    # Test 16: Inheritance of type and invariants
    class BaseSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda n: (n > 0, "Not positive")
    
    class DerivedSet(BaseSet):
        pass
    
    result = DerivedSet([1, 2, 3])
    assert isinstance(result, DerivedSet)
    assert len(result) == 3
    
    try:
        DerivedSet([-1])
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test 17: Enum type preservation
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3
    
    class EnumSet(CheckedPSet):
        __type__ = Color
    
    result = EnumSet([Color.RED, Color.GREEN])
    assert isinstance(result, EnumSet)
    assert len(result) == 2
    assert Color.RED in result
    assert Color.GREEN in result

    # Test 18: Iterable type specification
    class IterableTypeSet(CheckedPSet):
        __type__ = [int, str]
    
    result = IterableTypeSet([1, "test"])
    assert isinstance(result, IterableTypeSet)
    assert len(result) == 2


# LLM-generated content at query #2
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test 1: Create empty CheckedPMap
    class EmptyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    empty_map = EmptyMap()
    assert isinstance(empty_map, EmptyMap)
    assert len(empty_map) == 0
    
    # Test 2: Create with initial dictionary
    class IntStrMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    initial_map = IntStrMap({1: "a", 2: "b"})
    assert isinstance(initial_map, IntStrMap)
    assert len(initial_map) == 2
    assert initial_map[1] == "a"
    assert initial_map[2] == "b"
    
    # Test 3: Create with existing CheckedPMap instance
    existing_map = IntStrMap({3: "c"})
    new_map = IntStrMap(existing_map)
    assert isinstance(new_map, IntStrMap)
    assert len(new_map) == 1
    assert new_map[3] == "c"
    
    # Test 4: Type checking on creation
    class StrictMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    # Should work with correct types
    strict_map = StrictMap({1: "test", 5: "value"})
    assert len(strict_map) == 2
    
    # Test 5: Check that size parameter works (internal use)
    internal_map = StrictMap(pmap({10: "internal"}), size=1)
    assert isinstance(internal_map, StrictMap)
    assert len(internal_map) == 1
    assert internal_map[10] == "internal"
    
    # Test 6: Multiple type specifications
    class MultiTypeMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = (int, float)
    
    multi_map = MultiTypeMap({1: 1.5, "key": 2})
    assert len(multi_map) == 2
    assert multi_map[1] == 1.5
    assert multi_map["key"] == 2
    
    # Test 7: Inheritance of type specifications
    class BaseMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    class DerivedMap(BaseMap):
        pass
    
    derived_map = DerivedMap({1: "inherited"})
    assert isinstance(derived_map, DerivedMap)
    assert derived_map[1] == "inherited"
    
    # Test 8: Empty initial with size parameter
    empty_with_size = IntStrMap(pmap(), size=0)
    assert isinstance(empty_with_size, IntStrMap)
    assert len(empty_with_size) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_get_type():
    # Test with built-in type
    assert get_type(int) == int
    
    # Test with standard library type
    import collections
    assert get_type(collections.OrderedDict) == collections.OrderedDict
    
    # Test with string representation of built-in type
    assert get_type('int') == int
    
    # Test with string representation of standard library type
    assert get_type('collections.OrderedDict') == collections.OrderedDict
    
    # Test with string representation of custom class
    class CustomClass:
        pass
    
    import sys
    module_name = sys.modules[__name__].__name__
    type_name = f"{module_name}.CustomClass"
    assert get_type(type_name) == CustomClass
    
    # Test with nested module class
    import pyrsistent._pvector
    assert get_type('pyrsistent._pvector.PythonPVector') == pyrsistent._pvector.PythonPVector


# LLM-generated content at query #4
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
    assert maybe_parse_user_type([[int], [str], [float]]) == (int, str, float)
    
    # Test with mixed types and strings
    assert maybe_parse_user_type([int, "str", float]) == (int, "str", float)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    
    # Test that regular iterable types are not preserved
    class CustomIterable(Iterable):
        def __iter__(self):
            return iter([])
    
    # This should be treated as iterable, not as preserved type
    result = maybe_parse_user_type(CustomIterable)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0] == CustomIterable
    
    # Test error cases
    import pytest
    
    # Non-type, non-string, non-iterable should raise TypeError
    with pytest.raises(TypeError):
        maybe_parse_user_type(42)
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(object())
    
    # Test with complex nested structure
    assert maybe_parse_user_type([int, [str, [float, bool]]]) == (int, str, float, bool)
    
    # Test that string type specification is preserved as string
    assert maybe_parse_user_type("List[int]") == ["List[int]"]


# LLM-generated content at query #5
#--------------------------

```python
def test_InvariantException___str__():
    # Test with no errors or missing fields
    exc = InvariantException()
    assert str(exc) == ", invariant_errors=[], missing_fields=[]"
    
    # Test with only invariant errors
    exc = InvariantException(error_codes=["error1", "error2"])
    assert str(exc) == ", invariant_errors=[error1, error2], missing_fields=[]"
    
    # Test with only missing fields
    exc = InvariantException(missing_fields=["field1", "field2"])
    assert str(exc) == ", invariant_errors=[], missing_fields=[field1, field2]"
    
    # Test with both invariant errors and missing fields
    exc = InvariantException(error_codes=["err1", "err2"], missing_fields=["f1", "f2", "f3"])
    assert str(exc) == ", invariant_errors=[err1, err2], missing_fields=[f1, f2, f3]"
    
    # Test with callable error codes
    def error_func():
        return "dynamic_error"
    
    exc = InvariantException(error_codes=[error_func, "static_error"])
    assert str(exc) == ", invariant_errors=[dynamic_error, static_error], missing_fields=[]"
    
    # Test with empty lists
    exc = InvariantException(error_codes=[], missing_fields=[])
    assert str(exc) == ", invariant_errors=[], missing_fields=[]"
    
    # Test with single item
    exc = InvariantException(error_codes=["single_error"], missing_fields=["single_field"])
    assert str(exc) == ", invariant_errors=[single_error], missing_fields=[single_field]"


# LLM-generated content at query #6
#--------------------------

```python
def test_CheckedPSet___new__():
    # Test 1: Create empty CheckedPSet
    class IntSet(CheckedPSet):
        __type__ = int
    
    result = IntSet()
    assert isinstance(result, IntSet)
    assert len(result) == 0
    
    # Test 2: Create CheckedPSet with initial values
    result = IntSet([1, 2, 3])
    assert isinstance(result, IntSet)
    assert len(result) == 3
    assert 1 in result
    assert 2 in result
    assert 3 in result
    
    # Test 3: Type checking works
    class StringSet(CheckedPSet):
        __type__ = str
    
    result = StringSet(["a", "b", "c"])
    assert isinstance(result, StringSet)
    assert "a" in result
    
    # Test 4: Type violation raises CheckedValueTypeError
    class IntOnlySet(CheckedPSet):
        __type__ = int
    
    try:
        IntOnlySet([1, "invalid", 3])
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.source_class == IntOnlySet
        assert e.actual_type == str
    
    # Test 5: Invariant checking works
    class PositiveSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda n: (n > 0, "Not positive")
    
    result = PositiveSet([1, 2, 3])
    assert len(result) == 3
    
    # Test 6: Invariant violation raises InvariantException
    try:
        PositiveSet([1, -2, 3])
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert len(e.invariant_errors) > 0
    
    # Test 7: Multiple types allowed
    class MultiTypeSet(CheckedPSet):
        __type__ = (int, str)
    
    result = MultiTypeSet([1, "two", 3])
    assert len(result) == 3
    assert 1 in result
    assert "two" in result
    
    # Test 8: Create from existing CheckedPSet (same type)
    original = IntSet([1, 2, 3])
    result = IntSet(original)
    assert isinstance(result, IntSet)
    assert len(result) == 3
    
    # Test 9: Create from PMap (internal representation)
    from pyrsistent import pmap
    pmap_data = pmap({1: True, 2: True})
    result = IntSet(pmap_data)
    assert isinstance(result, IntSet)
    assert len(result) == 2
    
    # Test 10: Duplicate values are handled
    result = IntSet([1, 2, 2, 3, 1])
    assert len(result) == 3
    
    # Test 11: Optional types work
    class OptionalSet(CheckedPSet):
        __type__ = optional(int)
    
    result = OptionalSet([1, None, 3])
    assert len(result) == 3
    assert None in result
    
    # Test 12: String type specifications work
    class StringTypeSet(CheckedPSet):
        __type__ = "int"
    
    result = StringTypeSet([1, 2, 3])
    assert isinstance(result, StringTypeSet)
    
    # Test 13: Inheritance of type constraints
    class BaseSet(CheckedPSet):
        __type__ = int
    
    class DerivedSet(BaseSet):
        pass
    
    result = DerivedSet([1, 2, 3])
    assert isinstance(result, DerivedSet)
    
    try:
        DerivedSet(["invalid"])
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass
    
    # Test 14: Inheritance of invariants
    class BaseInvariantSet(CheckedPSet):
        __type__ = int
        __invariant__ = lambda n: (n > 0, "Not positive")
    
    class DerivedInvariantSet(BaseInvariantSet):
        pass
    
    result = DerivedInvariantSet([1, 2, 3])
    assert len(result) == 3
    
    try:
        DerivedInvariantSet([-1])
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_get_type():
    # Test with built-in type
    assert get_type(int) is int
    
    # Test with standard library type
    import collections
    assert get_type(collections.OrderedDict) is collections.OrderedDict
    
    # Test with string representation of built-in type
    assert get_type('builtins.int') is int
    
    # Test with string representation of standard library type
    assert get_type('collections.OrderedDict') is collections.OrderedDict
    
    # Test with string representation of custom module type
    import sys
    from types import ModuleType
    
    # Create a dummy module for testing
    dummy_module = ModuleType('dummy_module')
    dummy_module.SomeClass = type('SomeClass', (), {})
    sys.modules['dummy_module'] = dummy_module
    
    assert get_type('dummy_module.SomeClass') is dummy_module.SomeClass
    
    # Clean up
    del sys.modules['dummy_module']
    
    # Test with nested module
    import os.path
    assert get_type('os.path') is os.path
    
    # Test that it returns the same object for type input
    class CustomClass:
        pass
    
    assert get_type(CustomClass) is CustomClass


# LLM-generated content at query #8
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test basic instantiation with empty dict
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    m = TestMap()
    assert isinstance(m, TestMap)
    assert len(m) == 0
    
    # Test instantiation with initial data
    m = TestMap({1: "a", 2: "b"})
    assert m[1] == "a"
    assert m[2] == "b"
    assert isinstance(m, TestMap)
    
    # Test instantiation with existing PMap (size parameter)
    from pyrsistent import pmap
    existing_pmap = pmap({3: "c", 4: "d"})
    m = TestMap(existing_pmap, size=2)
    assert m[3] == "c"
    assert m[4] == "d"
    assert isinstance(m, TestMap)
    
    # Test type checking on keys
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    try:
        IntMap({"invalid": "value"})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError as e:
        assert e.source_class == IntMap
        assert e.actual_type == str
    
    # Test type checking on values
    class StrValueMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    try:
        StrValueMap({1: 123})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.source_class == StrValueMap
        assert e.actual_type == int
    
    # Test invariant checking
    class PositiveMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, "Value must be positive")
    
    try:
        PositiveMap({1: -1})
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Value must be positive" in str(e.error_codes)
    
    # Test multiple invariants
    class MultiInvariantMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, "Positive")
        __invariant__ = lambda k, v: (v < 10, "Less than 10")
    
    m = MultiInvariantMap({1: 5})
    assert m[1] == 5
    
    try:
        MultiInvariantMap({1: 15})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass
    
    # Test inheritance of type constraints
    class BaseMap(CheckedPMap):
        __key_type__ = int
    
    class DerivedMap(BaseMap):
        __value_type__ = str
    
    m = DerivedMap({1: "test"})
    assert isinstance(m, DerivedMap)
    
    try:
        DerivedMap({"invalid": "test"})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass
    
    # Test with optional types
    class OptionalMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = optional(str, type(None))
    
    m1 = OptionalMap({1: "test"})
    m2 = OptionalMap({1: None})
    assert m1[1] == "test"
    assert m2[1] is None
    
    # Test evolver integration
    m = TestMap({1: "a"})
    evolver = m.evolver()
    evolver.set(2, "b")
    m2 = evolver.persistent()
    assert isinstance(m2, TestMap)
    assert m2[2] == "b"
    
    # Test that same instance is returned if already correct type
    m = TestMap({1: "a"})
    m2 = TestMap(m)
    assert m is m2
    
    # Test with string type specifications
    class StringTypeMap(CheckedPMap):
        __key_type__ = "int"
        __value_type__ = "str"
    
    m = StringTypeMap({1: "test"})
    assert isinstance(m, StringTypeMap)
    assert m[1] == "test"


# LLM-generated content at query #9
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
    
    # All conditions pass
    result = wrapped(4)
    assert result == (True, ())
    
    # Some conditions fail
    result = wrapped(11)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "Value must be less than 10" in result[1]
    assert "Value must be even" in result[1]
    
    # All conditions fail
    result = wrapped(-1)
    assert result[0] == False
    assert len(result[1]) == 3
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(x, y):
        return [
            (x > 0, "x must be positive"),
            (y > 0, "y must be positive"),
            (x + y < 20, "sum must be less than 20")
        ]
    
    wrapped = wrap_invariant(multi_arg_invariant)
    assert wrapped(5, 5) == (True, ())
    
    result = wrapped(-1, 25)
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
    assert wrapped(5) == (True, ())
    
    # Test 6: Single tuple in list
    def single_tuple_invariant(x):
        return [(x > 0, "positive")]
    
    wrapped = wrap_invariant(single_tuple_invariant)
    assert wrapped(5) == (True, ())
    assert wrapped(-5) == (False, ("positive",))


# LLM-generated content at query #10
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
    def invariant_multiple_all_true(x):
        return [(True, "ok1"), (True, "ok2"), (True, "ok3")]
    
    wrapped = wrap_invariant(invariant_multiple_all_true)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 4: Invariant returns multiple results with some false
    def invariant_multiple_some_false(x):
        return [(True, "ok1"), (False, "error1"), (True, "ok2"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_multiple_some_false)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "error1" in result[1]
    assert "error2" in result[1]
    
    # Test 5: Invariant returns multiple results all false
    def invariant_multiple_all_false(x):
        return [(False, "error1"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_multiple_all_false)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "error1" in result[1]
    assert "error2" in result[1]
    
    # Test 6: Invariant with multiple arguments
    def invariant_multi_args(x, y, z=10):
        if x + y > z:
            return False, "sum too large"
        return True, "ok"
    
    wrapped = wrap_invariant(invariant_multi_args)
    result = wrapped(3, 4, z=10)
    assert result == (True, "ok")
    
    result = wrapped(6, 5, z=10)
    assert result == (False, "sum too large")
    
    # Test 7: Invariant returns empty list
    def invariant_empty_list(x):
        return []
    
    wrapped = wrap_invariant(invariant_empty_list)
    result = wrapped(5)
    assert result == (True, ())


# LLM-generated content at query #11
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
        return [(True, "success1"), (True, "success2")]
    
    wrapped = wrap_invariant(invariant_multiple_true)
    result = wrapped()
    assert result == (True, ())
    
    # Test 4: Invariant returns multiple results with some false
    def invariant_multiple_mixed(*args, **kwargs):
        return [(True, "success"), (False, "error1"), (True, "success2"), (False, "error2")]
    
    wrapped = wrap_invariant(invariant_multiple_mixed)
    result = wrapped()
    assert result[0] == False
    assert len(result[1]) == 2
    assert "error1" in result[1]
    assert "error2" in result[1]
    
    # Test 5: Invariant returns multiple results all false
    def invariant_multiple_false(*args, **kwargs):
        return [(False, "error1"), (False, "error2"), (False, "error3")]
    
    wrapped = wrap_invariant(invariant_multiple_false)
    result = wrapped()
    assert result[0] == False
    assert len(result[1]) == 3
    assert all(e in result[1] for e in ("error1", "error2", "error3"))
    
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
            return True, f"value {value} > threshold {threshold}"
        else:
            return False, f"value {value} <= threshold {threshold}"
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(15, threshold=10)
    assert result == (True, "value 15 > threshold 10")
    
    result = wrapped(5, threshold=10)
    assert result == (False, "value 5 <= threshold 10")


# LLM-generated content at query #12
#--------------------------

```python
def test_wrap_invariant():
    def invariant_single_bool(value):
        return value > 0, "Value must be positive"

    def invariant_multiple_results(value):
        return [(value > 0, "positive"), (value < 10, "less than 10")]

    wrapped_single = wrap_invariant(invariant_single_bool)
    wrapped_multiple = wrap_invariant(invariant_multiple_results)

    result = wrapped_single(5)
    assert result == (True, "Value must be positive")

    result = wrapped_single(-1)
    assert result == (False, "Value must be positive")

    result = wrapped_multiple(5)
    assert result == (True, ())

    result = wrapped_multiple(15)
    assert result == (False, ("less than 10",))

    result = wrapped_multiple(-5)
    assert result == (False, ("positive", "less than 10"))

    def invariant_nested_results(value):
        return [
            (value > 0, "positive"),
            [(value % 2 == 0, "even"), (value < 100, "small")]
        ]

    wrapped_nested = wrap_invariant(invariant_nested_results)
    result = wrapped_nested(50)
    assert result == (True, ())

    result = wrapped_nested(150)
    assert result == (False, ("small",))

    result = wrapped_nested(-3)
    assert result == (False, ("positive", "even", "small"))


# LLM-generated content at query #13
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def single_bool_invariant(x):
        return x > 0, "Value must be positive"
    
    wrapped = wrap_invariant(single_bool_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant returns multiple results that need merging
    def multi_result_invariant(x):
        return [
            (x > 0, "positive"),
            (x < 10, "less than 10"),
            (x % 2 == 0, "even")
        ]
    
    wrapped = wrap_invariant(multi_result_invariant)
    
    # All conditions pass
    result = wrapped(4)
    assert result == (True, ())
    
    # Some conditions fail
    result = wrapped(11)
    assert result[0] == False
    assert "positive" in result[1]
    assert "less than 10" not in result[1]  # This one failed
    assert "even" not in result[1]  # This one failed
    
    # Test 3: Invariant with multiple failing conditions
    result = wrapped(-2)
    assert result[0] == False
    assert len(result[1]) == 2  # positive and less than 10 failed
    assert "positive" in result[1]
    assert "less than 10" in result[1]
    
    # Test 4: Invariant with no arguments
    def no_arg_invariant():
        return True, "Always true"
    
    wrapped = wrap_invariant(no_arg_invariant)
    result = wrapped()
    assert result == (True, "Always true")
    
    # Test 5: Invariant with keyword arguments
    def kwarg_invariant(**kwargs):
        return kwargs.get('valid', False), "Validity check"
    
    wrapped = wrap_invariant(kwarg_invariant)
    result = wrapped(valid=True)
    assert result == (True, "Validity check")
    
    result = wrapped(valid=False)
    assert result == (False, "Validity check")


# LLM-generated content at query #14
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test basic instantiation with empty dict
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    m1 = SimpleMap()
    assert isinstance(m1, SimpleMap)
    assert len(m1) == 0

    # Test instantiation with initial data
    m2 = SimpleMap({1: "one", 2: "two"})
    assert len(m2) == 2
    assert m2[1] == "one"
    assert m2[2] == "two"

    # Test that type checking works during instantiation
    class StringIntMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int

    m3 = StringIntMap({"a": 1, "b": 2})
    assert m3["a"] == 1

    # Test type checking failure for keys
    try:
        StringIntMap({1: "one"})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

    # Test type checking failure for values
    try:
        StringIntMap({"a": "not_an_int"})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass

    # Test with invariants
    class PositiveMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, "Value must be positive")

    m4 = PositiveMap({1: 5, 2: 10})
    assert m4[1] == 5

    # Test invariant violation during instantiation
    try:
        PositiveMap({1: -5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass

    # Test that existing CheckedPMap instance returns itself
    m5 = SimpleMap({1: "a"})
    m6 = SimpleMap(m5)
    assert m5 is m6

    # Test with multiple invariants
    class MultiInvariantMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, "Positive")
        __invariant__ = lambda k, v: (v < 100, "Less than 100")

    m7 = MultiInvariantMap({1: 50})
    assert m7[1] == 50

    # Test inheritance of type constraints
    class BaseMap(CheckedPMap):
        __key_type__ = str

    class DerivedMap(BaseMap):
        __value_type__ = int

    m8 = DerivedMap({"x": 1})
    assert m8["x"] == 1

    # Test that key_type is inherited
    try:
        DerivedMap({1: 1})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass

    # Test with optional types
    class OptionalMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = optional(int, type(None))

    m9 = OptionalMap({"a": 1, "b": None})
    assert m9["a"] == 1
    assert m9["b"] is None

    # Test internal size parameter (private API)
    internal_pmap = pmap({1: "a", 2: "b"})
    m10 = SimpleMap(internal_pmap, size=2)
    assert len(m10) == 2
    assert m10[1] == "a"


# LLM-generated content at query #15
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def invariant_single_bool(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(invariant_single_bool)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant returns multiple results that need merging
    def invariant_multiple_results(value):
        return [
            (value > 0, "positive"),
            (value < 10, "less than 10"),
            (value % 2 == 0, "even")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_results)
    
    # All conditions true
    result = wrapped(4)
    assert result == (True, ())
    
    # Some conditions false
    result = wrapped(15)
    assert result == (False, ("positive", "less than 10"))
    
    # All conditions false
    result = wrapped(-3)
    assert result == (False, ("positive", "less than 10", "even"))
    
    # Test 3: Invariant with no arguments
    def invariant_no_args():
        return True, "No args test"
    
    wrapped = wrap_invariant(invariant_no_args)
    result = wrapped()
    assert result == (True, "No args test")
    
    # Test 4: Invariant with keyword arguments
    def invariant_with_kwargs(value, threshold=5):
        return value > threshold, f"Value must be greater than {threshold}"
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(10, threshold=7)
    assert result == (True, "Value must be greater than 7")
    
    result = wrapped(3, threshold=1)
    assert result == (True, "Value must be greater than 1")
    
    # Test 5: Invariant returns empty tuple for data when all true
    def invariant_all_true(value):
        return [
            (True, "always true 1"),
            (True, "always true 2")
        ]
    
    wrapped = wrap_invariant(invariant_all_true)
    result = wrapped(42)
    assert result == (True, ())
    
    # Test 6: Invariant returns mixed results
    def invariant_mixed(value):
        return [
            (value > 0, "positive"),
            (True, "always true"),
            (value < 100, "less than 100")
        ]
    
    wrapped = wrap_invariant(invariant_mixed)
    result = wrapped(50)
    assert result == (True, ())
    
    result = wrapped(150)
    assert result == (False, ("less than 100",))
    
    # Test 7: Verify wrap_invariant doesn't modify single boolean result invariants
    def invariant_simple(value):
        return False, "Always false"
    
    wrapped = wrap_invariant(invariant_simple)
    result = wrapped(999)
    assert result == (False, "Always false")


# LLM-generated content at query #16
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def invariant_single_bool(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(invariant_single_bool)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant returns multiple results that need merging
    def invariant_multiple_results(value):
        return [
            (value > 0, "positive"),
            (value < 10, "less than 10"),
            (value % 2 == 0, "even")
        ]
    
    wrapped = wrap_invariant(invariant_multiple_results)
    
    # All conditions true
    result = wrapped(6)
    assert result == (True, ())
    
    # One condition false
    result = wrapped(11)
    assert result == (False, ("less than 10",))
    
    # Multiple conditions false
    result = wrapped(-2)
    assert result == (False, ("positive", "less than 10"))
    
    # Test 3: Invariant with no arguments
    def invariant_no_args():
        return True, "No args test"
    
    wrapped = wrap_invariant(invariant_no_args)
    result = wrapped()
    assert result == (True, "No args test")
    
    # Test 4: Invariant with keyword arguments
    def invariant_with_kwargs(value, threshold=5):
        return value > threshold, f"Value must be greater than {threshold}"
    
    wrapped = wrap_invariant(invariant_with_kwargs)
    result = wrapped(10, threshold=7)
    assert result == (True, "Value must be greater than 7")
    
    result = wrapped(3, threshold=1)
    assert result == (True, "Value must be greater than 1")
    
    # Test 5: Invariant returns tuple with boolean first
    def invariant_tuple_bool(value):
        return (value != 0, "Value cannot be zero")
    
    wrapped = wrap_invariant(invariant_tuple_bool)
    result = wrapped(5)
    assert result == (True, "Value cannot be zero")
    
    result = wrapped(0)
    assert result == (False, "Value cannot be zero")
    
    # Test 6: Nested structure that needs merging
    def invariant_nested(value):
        return [
            (value > 0, "positive"),
            [
                (value < 100, "less than 100"),
                (value % 3 == 0, "divisible by 3")
            ]
        ]
    
    wrapped = wrap_invariant(invariant_nested)
    result = wrapped(99)
    assert result == (True, ())
    
    result = wrapped(101)
    assert result == (False, ("less than 100",))
    
    # Test 7: Empty result list
    def invariant_empty_list(value):
        return []
    
    wrapped = wrap_invariant(invariant_empty_list)
    result = wrapped(5)
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
    assert maybe_parse_user_type([int, "str", float]) == (int, "str", float)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    assert maybe_parse_user_type(()) == ()
    
    # Test with complex nested structure
    assert maybe_parse_user_type([int, [str, [float, bool]]]) == (int, str, float, bool)
    
    # Test that non-type, non-string, non-iterable raises TypeError
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(42)
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(object())
    
    # Test that iterable types (except preserved ones) are treated as iterables
    # List is iterable but not in _preserved_iterable_types
    assert maybe_parse_user_type([int]) == (int,)
    
    # Test with multiple Enums
    class Status(Enum):
        ACTIVE = 1
        INACTIVE = 2
    
    assert maybe_parse_user_type([Color, Status]) == (Color, Status)


# LLM-generated content at query #18
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def single_bool_invariant(x):
        return x > 0, "Value must be positive"
    
    wrapped = wrap_invariant(single_bool_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-5)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant returns multiple results that need merging
    def multi_result_invariant(x):
        return [(x > 0, "positive"), (x < 10, "less than 10")]
    
    wrapped = wrap_invariant(multi_result_invariant)
    
    # All conditions satisfied
    result = wrapped(5)
    assert result == (True, ())
    
    # One condition failed
    result = wrapped(-5)
    assert result == (False, ("Value must be positive",))
    
    # Multiple conditions failed
    result = wrapped(15)
    assert result == (False, ("Value must be less than 10",))
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(x, y):
        return [(x > 0, "x positive"), (y > 0, "y positive")]
    
    wrapped = wrap_invariant(multi_arg_invariant)
    
    result = wrapped(5, 10)
    assert result == (True, ())
    
    result = wrapped(-5, 10)
    assert result == (False, ("x must be positive",))
    
    result = wrapped(-5, -10)
    assert result == (False, ("x must be positive", "y must be positive"))
    
    # Test 4: Invariant with keyword arguments
    def kwarg_invariant(x, threshold=0):
        return x > threshold, f"Value must be greater than {threshold}"
    
    wrapped = wrap_invariant(kwarg_invariant)
    
    result = wrapped(5, threshold=0)
    assert result == (True, "Value must be greater than 0")
    
    result = wrapped(5, threshold=10)
    assert result == (False, "Value must be greater than 10")
    
    # Test 5: Empty result list (edge case)
    def empty_result_invariant(x):
        return []
    
    wrapped = wrap_invariant(empty_result_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 6: Single tuple result (should not be merged)
    def single_tuple_invariant(x):
        return (x > 0, "positive")
    
    wrapped = wrap_invariant(single_tuple_invariant)
    result = wrapped(5)
    assert result == (True, "positive")
    
    # Test 7: Nested structure that shouldn't be merged
    def complex_result_invariant(x):
        # Returns a tuple where first element is not a list/tuple
        return ([(x > 0, "pos"), (x < 10, "small")], "additional info")
    
    wrapped = wrap_invariant(complex_result_invariant)
    result = wrapped(5)
    # Should return the result as-is since first element is not a bool
    assert isinstance(result, tuple)
    assert len(result) == 2


# LLM-generated content at query #19
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns single boolean result
    def invariant_single_true(x):
        return True, "All good"
    
    wrapped = wrap_invariant(invariant_single_true)
    result = wrapped(5)
    assert result == (True, "All good")
    
    # Test 2: Invariant returns single boolean result (False)
    def invariant_single_false(x):
        return False, "Something wrong"
    
    wrapped = wrap_invariant(invariant_single_false)
    result = wrapped(5)
    assert result == (False, "Something wrong")
    
    # Test 3: Invariant returns multiple results that need merging (all True)
    def invariant_multiple_true(x):
        return [(True, "Check1 passed"), (True, "Check2 passed")]
    
    wrapped = wrap_invariant(invariant_multiple_true)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 4: Invariant returns multiple results with one False
    def invariant_multiple_mixed(x):
        return [(True, "Check1 passed"), (False, "Check2 failed"), (True, "Check3 passed")]
    
    wrapped = wrap_invariant(invariant_multiple_mixed)
    result = wrapped(5)
    assert result[0] == False
    assert len(result[1]) == 1
    assert result[1][0] == "Check2 failed"
    
    # Test 5: Invariant returns multiple results with multiple False
    def invariant_multiple_false(x):
        return [(False, "Check1 failed"), (True, "Check2 passed"), (False, "Check3 failed")]
    
    wrapped = wrap_invariant(invariant_multiple_false)
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
    def invariant_kwargs(**kwargs):
        return kwargs.get('check', False), "Keyword check"
    
    wrapped = wrap_invariant(invariant_kwargs)
    result = wrapped(check=True)
    assert result == (True, "Keyword check")
    
    # Test 8: Empty result list (edge case)
    def invariant_empty_list(x):
        return []
    
    wrapped = wrap_invariant(invariant_empty_list)
    result = wrapped(5)
    assert result == (True, ())


# LLM-generated content at query #20
#--------------------------

```python
def test_store_invariants():
    # Test 1: Basic functionality - stores invariants from dict
    def invariant1(obj):
        return True, ()
    
    def invariant2(obj):
        return False, ("error",)
    
    dct = {'_invariants': invariant1}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariants')
    assert '_checked_invariants' in dct
    assert len(dct['_checked_invariants']) == 1
    assert callable(dct['_checked_invariants'][0])
    
    # Test 2: Multiple invariants in dict
    dct = {'_invariants': [invariant1, invariant2]}
    store_invariants(dct, bases, '_checked_invariants', '_invariants')
    assert len(dct['_checked_invariants']) == 2
    
    # Test 3: Inheritance from single base class
    class Base:
        _invariants = invariant1
    
    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, '_checked_invariants', '_invariants')
    assert len(dct['_checked_invariants']) == 1
    
    # Test 4: Inheritance from multiple base classes
    class Base1:
        _invariants = invariant1
    
    class Base2:
        _invariants = invariant2
    
    dct = {}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_checked_invariants', '_invariants')
    assert len(dct['_checked_invariants']) == 2
    
    # Test 5: Local dict overrides inheritance
    dct = {'_invariants': invariant2}
    bases = (Base1,)
    store_invariants(dct, bases, '_checked_invariants', '_invariants')
    assert len(dct['_checked_invariants']) == 2  # Both local and inherited
    
    # Test 6: Deep inheritance hierarchy
    class GrandParent:
        _invariants = invariant1
    
    class Parent(GrandParent):
        _invariants = invariant2
    
    class Child(Parent):
        pass
    
    dct = {}
    bases = (Child,)
    store_invariants(dct, bases, '_checked_invariants', '_invariants')
    assert len(dct['_checked_invariants']) == 2  # From both GrandParent and Parent
    
    # Test 7: Diamond inheritance (should avoid duplicates)
    class A:
        _invariants = invariant1
    
    class B(A):
        pass
    
    class C(A):
        pass
    
    class D(B, C):
        pass
    
    dct = {}
    bases = (D,)
    store_invariants(dct, bases, '_checked_invariants', '_invariants')
    assert len(dct['_checked_invariants']) == 1  # Only one copy of invariant1
    
    # Test 8: Invariants are wrapped
    def multi_invariant(obj):
        return [(True, ()), (False, ("error1",)), (False, ("error2",))]
    
    dct = {'_invariants': multi_invariant}
    store_invariants(dct, bases, '_checked_invariants', '_invariants')
    wrapped = dct['_checked_invariants'][0]
    result = wrapped(None)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] is False
    assert len(result[1]) == 2
    
    # Test 9: Non-callable invariant raises TypeError
    dct = {'_invariants': "not a callable"}
    try:
        store_invariants(dct, bases, '_checked_invariants', '_invariants')
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test 10: Empty invariants
    dct = {}
    bases = ()
    store_invariants(dct, bases, '_checked_invariants', '_invariants')
    assert dct['_checked_invariants'] == ()
    
    # Test 11: Invariant returning simple boolean
    def simple_invariant(obj):
        return True
    
    dct = {'_invariants': simple_invariant}
    store_invariants(dct, bases, '_checked_invariants', '_invariants')
    wrapped = dct['_checked_invariants'][0]
    result = wrapped(None)
    assert result == (True, ())


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__CheckedMapTypeMeta___new__():
    # Test basic class creation with key and value types
    class TestMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
    
    assert TestMap._checked_key_types == (str,)
    assert TestMap._checked_value_types == (int,)
    assert TestMap._checked_invariants == ()
    assert hasattr(TestMap, '__serializer__')
    
    # Test inheritance of type specifications
    class ParentMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
    
    class ChildMap(ParentMap):
        pass
    
    assert ChildMap._checked_key_types == (str,)
    assert ChildMap._checked_value_types == (int,)
    
    # Test overriding type specifications
    class OverrideMap(ParentMap):
        __key_type__ = int
        __value_type__ = str
    
    assert OverrideMap._checked_key_types == (int,)
    assert OverrideMap._checked_value_types == (str,)
    
    # Test multiple types
    class MultiTypeMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = (str, int)
        __value_type__ = (int, float, type(None))
    
    assert set(MultiTypeMap._checked_key_types) == {str, int}
    assert set(MultiTypeMap._checked_value_types) == {int, float, type(None)}
    
    # Test type specifications as strings
    class StringTypeMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = "builtins.str"
        __value_type__ = "builtins.int"
    
    assert StringTypeMap._checked_key_types == ("builtins.str",)
    assert StringTypeMap._checked_value_types == ("builtins.int",)
    
    # Test invariants inheritance
    def invariant1(x):
        return True, None
    
    def invariant2(x):
        return False, "error"
    
    class ParentWithInvariants(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
        __invariant__ = invariant1
    
    class ChildWithInvariants(ParentWithInvariants):
        __invariant__ = invariant2
    
    assert len(ChildWithInvariants._checked_invariants) == 2
    assert ChildWithInvariants._checked_invariants[0].__wrapped__ == invariant1
    assert ChildWithInvariants._checked_invariants[1].__wrapped__ == invariant2
    
    # Test that invariants are wrapped
    for inv in ChildWithInvariants._checked_invariants:
        assert hasattr(inv, '__wrapped__')
    
    # Test default serializer
    class DefaultSerializerMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
    
    # Check that default serializer is set
    assert hasattr(DefaultSerializerMap, '__serializer__')
    
    # Test slots are set
    assert DefaultSerializerMap.__slots__ == ()
    
    # Test with Enum as key type (should be preserved)
    class TestEnum(Enum):
        A = 1
        B = 2
    
    class EnumKeyMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = TestEnum
        __value_type__ = int
    
    assert EnumKeyMap._checked_key_types == (TestEnum,)
    
    # Test with iterable type specifications
    class IterableTypesMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = [str, "builtins.int"]
        __value_type__ = (float, [int, "builtins.str"])
    
    # All types should be flattened
    assert len(IterableTypesMap._checked_key_types) == 2
    assert len(IterableTypesMap._checked_value_types) == 3
    
    # Test that non-callable invariants raise TypeError
    try:
        class BadInvariantMap(metaclass=_CheckedMapTypeMeta):
            __key_type__ = str
            __value_type__ = int
            __invariant__ = "not callable"
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant returns simple boolean result
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
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
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
    
    # Test 3: Invariant with multiple arguments
    def multi_arg_invariant(a, b):
        return [
            (a > 0, "A must be positive"),
            (b > 0, "B must be positive"),
            (a + b < 100, "Sum must be less than 100")
        ]
    
    wrapped = wrap_invariant(multi_arg_invariant)
    
    result = wrapped(10, 20)
    assert result == (True, ())
    
    result = wrapped(-10, 200)
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


# LLM-generated content at query #3
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
    
    # Test that iterable types (except preserved ones) are treated as iterables
    class CustomIterable(Iterable):
        def __iter__(self):
            return iter([])
    
    # CustomIterable should be treated as an iterable, not a type
    try:
        maybe_parse_user_type(CustomIterable)
        assert False, "Should have raised TypeError for non-preserved iterable type"
    except TypeError:
        pass
    
    # Test with complex nested structure
    assert maybe_parse_user_type([int, [str, [float, "custom"]]]) == (int, str, float, "custom")
    
    # Test that preserved types in iterables are preserved
    assert maybe_parse_user_type([Color, int]) == (Color, int)


# LLM-generated content at query #4
#--------------------------

```python
def test_get_type():
    # Test with built-in type
    assert get_type(int) is int
    
    # Test with standard library type
    import collections
    assert get_type(collections.OrderedDict) is collections.OrderedDict
    
    # Test with string type specification
    assert get_type('collections.OrderedDict') is collections.OrderedDict
    
    # Test with nested module
    import collections.abc
    assert get_type('collections.abc.Iterable') is collections.abc.Iterable
    
    # Test with local module type
    from enum import Enum
    assert get_type('enum.Enum') is Enum
    
    # Test that same type object is returned
    type_obj = get_type(int)
    assert type_obj is int
    
    # Test with custom class (simulate)
    class CustomClass:
        pass
    
    # Note: Can't test string path for CustomClass without proper module path
    # But we can test that type object passes through
    assert get_type(CustomClass) is CustomClass


# LLM-generated content at query #5
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
            return iter([])
    
    # This should be treated as an iterable, not a type
    with pytest.raises(TypeError):
        maybe_parse_user_type(CustomIterable)


# LLM-generated content at query #6
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
    assert maybe_parse_user_type([[int], [str, [float]]]) == (int, str, float)
    
    # Test with mixed types and strings
    assert maybe_parse_user_type([int, "str", float]) == (int, "str", float)
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    assert maybe_parse_user_type(()) == ()
    
    # Test that invalid inputs raise TypeError
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)
    
    with pytest.raises(TypeError):
        maybe_parse_user_type({"key": "value"})
    
    # Test with custom class (non-iterable type)
    class CustomClass:
        pass
    
    assert maybe_parse_user_type(CustomClass) == [CustomClass]
    
    # Test with iterable type that's not preserved
    class IterableClass(Iterable):
        def __iter__(self):
            return iter([])
    
    # This should be treated as an iterable, not a type
    with pytest.raises(TypeError):
        maybe_parse_user_type(IterableClass)


# LLM-generated content at query #7
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
    result = maybe_parse_user_type([int, "custom", str])
    assert len(result) == 3
    assert int in result
    assert str in result
    assert "custom" in result
    
    # Test with complex nested structure
    result = maybe_parse_user_type([int, [str, [float, "custom"]], bool])
    assert len(result) == 5
    assert int in result
    assert str in result
    assert float in result
    assert bool in result
    assert "custom" in result
    
    # Test that invalid input raises TypeError
    import pytest
    with pytest.raises(TypeError):
        maybe_parse_user_type(123)  # Not a type or string
    
    with pytest.raises(TypeError):
        maybe_parse_user_type({"key": "value"})  # Dict is iterable but not valid
    
    # Test with empty iterable
    assert maybe_parse_user_type([]) == ()
    assert maybe_parse_user_type(()) == ()
    
    # Test with single element iterable
    assert maybe_parse_user_type([int]) == (int,)
    
    # Test that regular iterable types are not preserved
    class CustomIterable(Iterable):
        def __iter__(self):
            return iter([])
    
    # CustomIterable is a type but also iterable, so it should be treated as iterable
    # and its contents should be parsed (empty in this case)
    assert maybe_parse_user_type(CustomIterable) == ()


# LLM-generated content at query #8
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test 1: Create empty CheckedPMap
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    result = TestMap()
    assert isinstance(result, TestMap)
    assert len(result) == 0
    
    # Test 2: Create with initial dict
    result = TestMap({1: "a", 2: "b"})
    assert isinstance(result, TestMap)
    assert result[1] == "a"
    assert result[2] == "b"
    
    # Test 3: Create with existing CheckedPMap (should return same instance)
    original = TestMap({1: "a"})
    result = TestMap(original)
    assert result is original
    
    # Test 4: Type checking on keys
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    # Valid key type
    result = IntMap({1: "valid"})
    assert result[1] == "valid"
    
    # Invalid key type should raise CheckedKeyTypeError
    try:
        IntMap({"invalid": "value"})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass
    
    # Test 5: Type checking on values
    class StrValueMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    # Valid value type
    result = StrValueMap({1: "valid"})
    assert result[1] == "valid"
    
    # Invalid value type should raise CheckedValueTypeError
    try:
        StrValueMap({1: 123})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass
    
    # Test 6: Invariant checking
    class PositiveMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, "Value must be positive")
    
    # Valid invariant
    result = PositiveMap({1: 5})
    assert result[1] == 5
    
    # Invalid invariant should raise InvariantException
    try:
        PositiveMap({1: -5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass
    
    # Test 7: Multiple types allowed
    class MultiTypeMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = (int, float)
    
    result1 = MultiTypeMap({1: 3.14})
    result2 = MultiTypeMap({"key": 42})
    assert result1[1] == 3.14
    assert result2["key"] == 42
    
    # Test 8: Inheritance of type constraints
    class BaseMap(CheckedPMap):
        __key_type__ = int
    
    class DerivedMap(BaseMap):
        __value_type__ = str
    
    result = DerivedMap({1: "test"})
    assert isinstance(result, DerivedMap)
    assert result[1] == "test"
    
    # Test 9: CheckedPMap with CheckedType values
    class InnerChecked(CheckedPVector):
        __type__ = int
    
    class OuterMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = InnerChecked
    
    inner = InnerChecked([1, 2, 3])
    result = OuterMap({"key": inner})
    assert result["key"] == inner
    
    # Test 10: Direct instantiation with size parameter (internal use)
    internal_map = pmap({1: "a", 2: "b"})
    result = TestMap(internal_map, size=2)
    assert isinstance(result, TestMap)
    assert len(result) == 2


# LLM-generated content at query #9
#--------------------------

```python
def test_get_type():
    # Test with built-in type
    assert get_type(int) == int
    
    # Test with standard library type
    import collections
    assert get_type(collections.OrderedDict) == collections.OrderedDict
    
    # Test with string representation of built-in type
    assert get_type('builtins.int') == int
    
    # Test with string representation of standard library type
    assert get_type('collections.OrderedDict') == collections.OrderedDict
    
    # Test with string representation of custom class
    class CustomClass:
        pass
    
    import sys
    module_name = sys.modules[__name__].__name__
    type_name = f'{module_name}.CustomClass'
    assert get_type(type_name) == CustomClass
    
    # Test with nested module
    import os.path
    assert get_type('os.path') == os.path


# LLM-generated content at query #10
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
    assert len(result) == 1
    assert isinstance(next(iter(result)), set)
    assert next(iter(result)) == {1, 2}
    
    # Test serialization with custom serializer
    class CustomSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return value * 2
    
    s = CustomSerializerSet([1, 2, 3])
    result = s.serialize()
    assert result == {2, 4, 6}
    
    # Test serialization with format parameter
    class FormatAwareSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            if format == 'double':
                return value * 2
            return value
    
    s = FormatAwareSet([1, 2, 3])
    result = s.serialize(format='double')
    assert result == {2, 4, 6}
    
    # Test serialization with empty set
    class EmptySet(CheckedPSet):
        __type__ = int
    
    s = EmptySet()
    result = s.serialize()
    assert result == set()
    
    # Test serialization with mixed types using optional
    class OptionalSet(CheckedPSet):
        __type__ = optional(int, str)
    
    s = OptionalSet([1, "hello", None])
    result = s.serialize()
    assert result == {1, "hello", None}
    
    # Test that serializer is called for each element
    class CountingSerializerSet(CheckedPSet):
        __type__ = int
        
        def __serializer__(self, format, value):
            return f"value_{value}"
    
    s = CountingSerializerSet([1, 2])
    result = s.serialize()
    assert result == {"value_1", "value_2"}


# LLM-generated content at query #11
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test 1: Create empty CheckedPMap
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert len(result) == 0

    # Test 2: Create from dict with correct types
    result = TestMap({1: "a", 2: "b"})
    assert isinstance(result, TestMap)
    assert result[1] == "a"
    assert result[2] == "b"
    assert len(result) == 2

    # Test 3: Create from existing CheckedPMap instance
    original = TestMap({1: "a"})
    result = TestMap(original)
    assert isinstance(result, TestMap)
    assert result[1] == "a"
    assert result is not original

    # Test 4: Type checking for keys
    class StringKeyMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int

    result = StringKeyMap({"a": 1, "b": 2})
    assert result["a"] == 1
    assert result["b"] == 2

    # Test 5: Type checking for values
    class IntValueMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int

    result = IntValueMap({"a": 1, "b": 2})
    assert result["a"] == 1
    assert result["b"] == 2

    # Test 6: Multiple allowed types
    class MultiTypeMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = (int, float)

    result = MultiTypeMap({1: 1.5, "a": 2})
    assert result[1] == 1.5
    assert result["a"] == 2

    # Test 7: With invariants
    class PositiveMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, "Value must be positive")

    result = PositiveMap({1: 5, 2: 10})
    assert result[1] == 5
    assert result[2] == 10

    # Test 8: Using evolver pattern
    evolver = TestMap().evolver()
    evolver.set(1, "test")
    result = evolver.persistent()
    assert isinstance(result, TestMap)
    assert result[1] == "test"

    # Test 9: Check that internal size parameter works
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    # This tests the internal path when size is provided
    internal_map = pmap({1: "a", 2: "b"})
    result = SimpleMap(internal_map, size=2)
    assert isinstance(result, SimpleMap)
    assert result[1] == "a"
    assert result[2] == "b"

    # Test 10: Empty dict initialization
    result = TestMap({})
    assert isinstance(result, TestMap)
    assert len(result) == 0


# LLM-generated content at query #12
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
    assert dct['_invariants'][0].__name__ == 'f'
    
    # Test 2: Multiple invariants from multiple inheritance
    class Parent1:
        def parent1_invariant(self):
            return True, ()
    
    class Parent2:
        def parent2_invariant(self):
            return False, ("error",)
    
    dct = {'__invariant__': lambda self: (True, ())}
    bases = (Parent1, Parent2)
    store_invariants(dct, bases, '_invariants', '__invariant__')
    
    assert len(dct['_invariants']) == 3
    
    # Test 3: Invariants are wrapped properly
    def test_invariant(self):
        return [(True, ()), (False, ("error1",)), (True, ())]
    
    dct = {'__invariant__': test_invariant}
    bases = ()
    store_invariants(dct, bases, '_invariants', '__invariant__')
    
    wrapped = dct['_invariants'][0]
    result = wrapped(None)
    assert result == (False, ("error1",))
    
    # Test 4: Non-callable invariant raises TypeError
    dct = {'__invariant__': "not a callable"}
    bases = ()
    
    try:
        store_invariants(dct, bases, '_invariants', '__invariant__')
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test 5: Diamond inheritance - ensure no duplicates
    class GrandParent:
        def grandparent_invariant(self):
            return True, ()
    
    class ParentA(GrandParent):
        def parent_a_invariant(self):
            return True, ()
    
    class ParentB(GrandParent):
        def parent_b_invariant(self):
            return True, ()
    
    class Child(ParentA, ParentB):
        def child_invariant(self):
            return True, ()
    
    dct = {}
    bases = (Child,)
    store_invariants(dct, bases, '_invariants', '__invariant__')
    
    # Should have 4 unique invariants (GrandParent, ParentA, ParentB, Child)
    assert len(dct['_invariants']) == 4
    
    # Test 6: Empty invariants
    dct = {}
    bases = ()
    store_invariants(dct, bases, '_invariants', '__invariant__')
    assert dct['_invariants'] == ()
    
    # Test 7: Invariant that returns boolean directly
    def simple_invariant(self):
        return True
    
    dct = {'__invariant__': simple_invariant}
    bases = ()
    store_invariants(dct, bases, '_invariants', '__invariant__')
    
    wrapped = dct['_invariants'][0]
    result = wrapped(None)
    assert result == True
    
    # Test 8: Invariant that returns tuple with boolean
    def tuple_invariant(self):
        return (False, ("error",))
    
    dct = {'__invariant__': tuple_invariant}
    bases = ()
    store_invariants(dct, bases, '_invariants', '__invariant__')
    
    wrapped = dct['_invariants'][0]
    result = wrapped(None)
    assert result == (False, ("error",))


# LLM-generated content at query #13
#--------------------------

```python
def test_get_type():
    # Test with built-in type
    assert get_type(int) is int
    
    # Test with standard library type
    import collections
    assert get_type(collections.OrderedDict) is collections.OrderedDict
    
    # Test with string representation of built-in type
    assert get_type('builtins.int') is int
    
    # Test with string representation of standard library type
    assert get_type('collections.OrderedDict') is collections.OrderedDict
    
    # Test with string representation of custom type
    class CustomType:
        pass
    
    import sys
    current_module = sys.modules[__name__]
    type_name = f'{current_module.__name__}.CustomType'
    assert get_type(type_name) is CustomType
    
    # Test with nested module type
    import os.path
    type_name = 'os.path.join'
    # This should import os.path and return the join function
    result = get_type(type_name)
    assert result is os.path.join


# LLM-generated content at query #14
#--------------------------

```python
def test_wrap_invariant():
    # Test 1: Invariant that returns a simple boolean verdict
    def simple_invariant(value):
        return value > 0, "Value must be positive"
    
    wrapped = wrap_invariant(simple_invariant)
    result = wrapped(5)
    assert result == (True, "Value must be positive")
    
    result = wrapped(-1)
    assert result == (False, "Value must be positive")
    
    # Test 2: Invariant that returns multiple results to be merged
    def multi_invariant(value):
        return [
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(multi_invariant)
    
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
    def complex_invariant(a, b, c=0):
        return [
            (a > b, "a must be greater than b"),
            (b > c, "b must be greater than c"),
            (a + b + c == 10, "Sum must be 10")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # All conditions pass
    result = wrapped(5, 3, 2)
    assert result == (True, ())
    
    # Some conditions fail
    result = wrapped(3, 5, 2)
    assert result[0] == False
    assert len(result[1]) == 2
    assert "a must be greater than b" in result[1]
    assert "Sum must be 10" in result[1]
    
    # Test 4: Invariant with keyword arguments
    result = wrapped(8, 2, c=0)
    assert result[0] == False
    assert "Sum must be 10" in result[1]
    
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


# LLM-generated content at query #15
#--------------------------

```python
def test_get_type():
    # Test with built-in type
    assert get_type(int) is int
    
    # Test with standard library type
    import collections
    assert get_type(collections.OrderedDict) is collections.OrderedDict
    
    # Test with string reference to built-in type
    assert get_type('builtins.int') is int
    
    # Test with string reference to standard library type
    assert get_type('collections.OrderedDict') is collections.OrderedDict
    
    # Test with string reference to local module type
    from enum import Enum
    assert get_type('enum.Enum') is Enum
    
    # Test with custom class
    class CustomClass:
        pass
    
    import sys
    module_name = __name__
    sys.modules[module_name].CustomClass = CustomClass
    assert get_type(f'{module_name}.CustomClass') is CustomClass


# LLM-generated content at query #16
#--------------------------

```python
def test_store_invariants():
    # Test 1: Basic invariant storage
    class BaseClass:
        def base_invariant(self):
            return True, "base_ok"
    
    dct = {}
    bases = (BaseClass,)
    store_invariants(dct, bases, '_invariants', '_invariant')
    
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 0
    
    # Test 2: Invariant in dct
    def test_invariant(obj):
        return True, "test_ok"
    
    dct = {'_invariant': test_invariant}
    bases = ()
    store_invariants(dct, bases, '_invariants', '_invariant')
    
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 1
    assert callable(dct['_invariants'][0])
    
    # Test 3: Invariant inheritance
    class BaseWithInvariant:
        def base_invariant(self):
            return True, "base"
    
    class Derived:
        def derived_invariant(self):
            return False, "derived"
    
    dct = {'_invariant': Derived.derived_invariant}
    bases = (BaseWithInvariant,)
    store_invariants(dct, bases, '_invariants', '_invariant')
    
    assert len(dct['_invariants']) == 1
    assert dct['_invariants'][0].__name__ == 'f'
    
    # Test 4: Multiple inheritance
    class GrandParent:
        def grandparent_invariant(self):
            return True, "grandparent"
    
    class Parent(GrandParent):
        def parent_invariant(self):
            return True, "parent"
    
    class Child(Parent):
        def child_invariant(self):
            return True, "child"
    
    dct = {'_invariant': Child.child_invariant}
    bases = (Parent,)
    store_invariants(dct, bases, '_invariants', '_invariant')
    
    assert len(dct['_invariants']) == 1
    
    # Test 5: Diamond inheritance (should avoid duplicates)
    class A:
        def a_invariant(self):
            return True, "a"
    
    class B(A):
        pass
    
    class C(A):
        pass
    
    class D(B, C):
        def d_invariant(self):
            return True, "d"
    
    dct = {'_invariant': D.d_invariant}
    bases = (B, C)
    store_invariants(dct, bases, '_invariants', '_invariant')
    
    assert len(dct['_invariants']) == 1
    
    # Test 6: Non-callable invariant raises TypeError
    dct = {'_invariant': "not a callable"}
    bases = ()
    
    try:
        store_invariants(dct, bases, '_invariants', '_invariant')
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Invariants must be callable" in str(e)
    
    # Test 7: Multiple invariants from different sources
    class Source1:
        def inv1(self):
            return True, "inv1"
    
    class Source2:
        def inv2(self):
            return True, "inv2"
    
    dct = {'_invariant': lambda self: (True, "dct_inv")}
    bases = (Source1, Source2)
    
    # Add invariants to base classes
    Source1._invariant = Source1.inv1
    Source2._invariant = Source2.inv2
    
    store_invariants(dct, bases, '_invariants', '_invariant')
    
    assert len(dct['_invariants']) == 3
    
    # Test 8: Wrap invariant handles tuple returns
    def multi_result_invariant(obj):
        return [(True, "result1"), (False, "result2")]
    
    dct = {'_invariant': multi_result_invariant}
    bases = ()
    store_invariants(dct, bases, '_invariants', '_invariant')
    
    wrapped_invariant = dct['_invariants'][0]
    result = wrapped_invariant(None)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] is False
    assert isinstance(result[1], tuple)
    assert "result2" in result[1]
    
    # Test 9: Wrap invariant passes through bool results
    def simple_invariant(obj):
        return True, "simple"
    
    dct = {'_invariant': simple_invariant}
    bases = ()
    store_invariants(dct, bases, '_invariants', '_invariant')
    
    wrapped_invariant = dct['_invariants'][0]
    result = wrapped_invariant(None)
    
    assert result == (True, "simple")
    
    # Test 10: Empty invariants
    dct = {}
    bases = ()
    store_invariants(dct, bases, '_invariants', '_invariant')
    
    assert '_invariants' in dct
    assert dct['_invariants'] == ()


# LLM-generated content at query #17
#--------------------------

```python
def test_store_invariants():
    # Test 1: Basic invariant storage
    class Base:
        def invariant1(self):
            return True, ()
    
    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, '_invariants', 'invariant')
    
    assert '_invariants' in dct
    assert len(dct['_invariants']) == 1
    assert callable(dct['_invariants'][0])
    
    # Test 2: Multiple invariants from multiple sources
    class Base1:
        def invariant1(self):
            return True, ()
    
    class Base2:
        def invariant2(self):
            return False, ("error",)
    
    dct = {'invariant3': lambda self: (True, ())}
    bases = (Base1, Base2)
    store_invariants(dct, bases, '_invariants', 'invariant')
    
    assert len(dct['_invariants']) == 3
    assert all(callable(inv) for inv in dct['_invariants'])
    
    # Test 3: Inheritance hierarchy
    class GrandParent:
        def grandparent_invariant(self):
            return True, ()
    
    class Parent(GrandParent):
        def parent_invariant(self):
            return True, ()
    
    class Child(Parent):
        def child_invariant(self):
            return True, ()
    
    dct = {}
    bases = (Child,)
    store_invariants(dct, bases, '_invariants', 'invariant')
    
    # Should get invariants from all levels
    invariant_names = [inv.__name__ for inv in dct['_invariants']]
    assert 'grandparent_invariant' in invariant_names
    assert 'parent_invariant' in invariant_names
    assert 'child_invariant' in invariant_names
    
    # Test 4: Wrap_invariant functionality with tuple return
    def complex_invariant(self):
        return ((True, ()), (False, ("error1",)), (True, ()))
    
    dct = {'invariant': complex_invariant}
    bases = ()
    store_invariants(dct, bases, '_invariants', 'invariant')
    
    # The wrapped invariant should merge results
    wrapped_inv = dct['_invariants'][0]
    result = wrapped_inv(None)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] is False  # One False makes overall False
    assert len(result[1]) == 1  # Only one error
    
    # Test 5: Wrap_invariant functionality with simple bool return
    def simple_invariant(self):
        return True, ()
    
    dct = {'invariant': simple_invariant}
    bases = ()
    store_invariants(dct, bases, '_invariants', 'invariant')
    
    wrapped_inv = dct['_invariants'][0]
    result = wrapped_inv(None)
    assert result == (True, ())
    
    # Test 6: Non-callable invariant should raise TypeError
    class BadBase:
        invariant = "not callable"
    
    dct = {}
    bases = (BadBase,)
    
    try:
        store_invariants(dct, bases, '_invariants', 'invariant')
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test 7: Diamond inheritance - avoid duplicates
    class A:
        def invariant_a(self):
            return True, ()
    
    class B(A):
        pass
    
    class C(A):
        pass
    
    class D(B, C):
        def invariant_d(self):
            return True, ()
    
    dct = {}
    bases = (D,)
    store_invariants(dct, bases, '_invariants', 'invariant')
    
    # Should have invariant_a once and invariant_d once
    invariant_names = [inv.__name__ for inv in dct['_invariants']]
    assert invariant_names.count('invariant_a') == 1
    assert invariant_names.count('invariant_d') == 1
    
    # Test 8: Empty case
    dct = {}
    bases = ()
    store_invariants(dct, bases, '_invariants', 'invariant')
    
    assert '_invariants' in dct
    assert dct['_invariants'] == ()


# LLM-generated content at query #18
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
    
    # Test 2: Invariant that returns multiple results (list of tuples)
    def multi_invariant(value):
        return [
            (value > 0, "Value must be positive"),
            (value < 10, "Value must be less than 10"),
            (value % 2 == 0, "Value must be even")
        ]
    
    wrapped = wrap_invariant(multi_invariant)
    
    # Test with value that passes all checks
    result = wrapped(6)
    assert result == (True, ())
    
    # Test with value that fails one check
    result = wrapped(11)
    assert result[0] == False
    assert len(result[1]) == 2  # Should have 2 error messages
    assert "Value must be less than 10" in str(result[1])
    assert "Value must be even" in str(result[1])
    
    # Test with value that fails multiple checks
    result = wrapped(-2)
    assert result[0] == False
    assert len(result[1]) == 2  # Should have 2 error messages
    assert "Value must be positive" in str(result[1])
    assert "Value must be even" in str(result[1])
    
    # Test 3: Invariant with multiple arguments
    def complex_invariant(a, b, c=0):
        return [
            (a > b, "a must be greater than b"),
            (b > c, "b must be greater than c"),
            (a + b + c > 0, "Sum must be positive")
        ]
    
    wrapped = wrap_invariant(complex_invariant)
    
    # Test passing case
    result = wrapped(3, 2, 1)
    assert result == (True, ())
    
    # Test failing case
    result = wrapped(1, 2, 3)
    assert result[0] == False
    assert len(result[1]) == 3  # All three checks should fail
    
    # Test 4: Invariant with keyword arguments
    result = wrapped(2, 1, c=0)
    assert result == (True, ())
    
    # Test 5: Empty result list (edge case)
    def empty_invariant(value):
        return []
    
    wrapped = wrap_invariant(empty_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    # Test 6: Single tuple in list (edge case)
    def single_tuple_invariant(value):
        return [(value > 0, "Positive required")]
    
    wrapped = wrap_invariant(single_tuple_invariant)
    result = wrapped(5)
    assert result == (True, ())
    
    result = wrapped(-5)
    assert result == (False, ("Positive required",))


# LLM-generated content at query #19
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test 1: Create empty CheckedPMap
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    m1 = TestMap()
    assert isinstance(m1, TestMap)
    assert len(m1) == 0
    
    # Test 2: Create with initial dict
    m2 = TestMap({1: "a", 2: "b"})
    assert isinstance(m2, TestMap)
    assert len(m2) == 2
    assert m2[1] == "a"
    assert m2[2] == "b"
    
    # Test 3: Type checking on creation
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
    
    # Should work with correct types
    m3 = IntMap({1: 10, 2: 20})
    assert m3[1] == 10
    
    # Should raise error with wrong key type
    try:
        IntMap({"wrong": 10})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError as e:
        assert e.source_class == IntMap
        assert str in e.expected_types
    
    # Should raise error with wrong value type
    try:
        IntMap({1: "wrong"})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError as e:
        assert e.source_class == IntMap
        assert int in e.expected_types
    
    # Test 4: Invariant checking on creation
    class PositiveMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, "Value must be positive")
    
    # Should work with positive values
    m4 = PositiveMap({1: 10, 2: 20})
    assert m4[1] == 10
    
    # Should raise InvariantException with negative values
    try:
        PositiveMap({1: -10})
        assert False, "Should have raised InvariantException"
    except InvariantException as e:
        assert "Value must be positive" in str(e.error_codes)
    
    # Test 5: Create from existing CheckedPMap (should return same instance)
    m5 = TestMap({1: "a"})
    m5_copy = TestMap(m5)
    assert m5_copy is m5  # Should be same instance
    
    # Test 6: Create with size parameter (internal use)
    m6 = TestMap(pmap({1: "a", 2: "b"}), size=2)
    assert isinstance(m6, TestMap)
    assert len(m6) == 2
    
    # Test 7: Multiple key/value types
    class MultiTypeMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = (int, float, type(None))
    
    m7 = MultiTypeMap({1: 10, "a": 3.14, 2: None})
    assert len(m7) == 3
    assert m7["a"] == 3.14
    
    # Test 8: CheckedType inheritance in values
    class InnerType(CheckedPVector):
        __type__ = int
    
    class OuterMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = InnerType
    
    inner = InnerType([1, 2, 3])
    m8 = OuterMap({"test": inner})
    assert isinstance(m8["test"], InnerType)
    assert list(m8["test"]) == [1, 2, 3]
    
    # Test 9: Serializer default
    m9 = TestMap({1: "test"})
    serialized = m9.serialize()
    assert serialized == {1: "test"}
    
    # Test 10: Evolver integration
    evolver = m2.evolver()
    evolver.set(3, "c")
    m10 = evolver.persistent()
    assert isinstance(m10, TestMap)
    assert len(m10) == 3
    assert m10[3] == "c"


# LLM-generated content at query #20
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test 1: Create empty CheckedPMap
    class TestMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    result = TestMap()
    assert isinstance(result, TestMap)
    assert len(result) == 0

    # Test 2: Create with initial dictionary
    result = TestMap({1: "a", 2: "b"})
    assert isinstance(result, TestMap)
    assert result[1] == "a"
    assert result[2] == "b"
    assert len(result) == 2

    # Test 3: Create with existing CheckedPMap instance
    original = TestMap({1: "a"})
    result = TestMap(original)
    assert isinstance(result, TestMap)
    assert result[1] == "a"
    assert result is not original

    # Test 4: Type checking for keys
    class IntMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    with pytest.raises(CheckedKeyTypeError):
        IntMap({"invalid": "value"})

    # Test 5: Type checking for values
    class StrValueMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    with pytest.raises(CheckedValueTypeError):
        StrValueMap({1: 123})

    # Test 6: Invariant checking
    class PositiveMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, "Value must be positive")

    with pytest.raises(InvariantException):
        PositiveMap({1: -1})

    # Test 7: Multiple invariants
    class MultiInvariantMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, "Positive")
        __invariant__ = lambda k, v: (v < 10, "Less than 10")

    with pytest.raises(InvariantException):
        MultiInvariantMap({1: 15})

    # Test 8: Inheritance of types and invariants
    class BaseMap(CheckedPMap):
        __key_type__ = int
        __invariant__ = lambda k, v: (k > 0, "Key positive")

    class DerivedMap(BaseMap):
        __value_type__ = str

    result = DerivedMap({1: "a"})
    assert isinstance(result, DerivedMap)
    
    with pytest.raises(InvariantException):
        DerivedMap({-1: "a"})

    # Test 9: Create with size parameter (internal use)
    internal_pmap = pmap({1: "a", 2: "b"})
    result = TestMap(internal_pmap, size=2)
    assert isinstance(result, TestMap)
    assert result[1] == "a"
    assert result[2] == "b"

    # Test 10: Check evolver integration
    test_map = TestMap({1: "a"})
    evolver = test_map.evolver()
    evolver.set(2, "b")
    result = evolver.persistent()
    assert isinstance(result, TestMap)
    assert result[1] == "a"
    assert result[2] == "b"


# LLM-generated content at query #21
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test basic instantiation with empty dict
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    m = SimpleMap()
    assert isinstance(m, SimpleMap)
    assert len(m) == 0
    
    # Test instantiation with initial data
    m = SimpleMap({1: "one", 2: "two"})
    assert m[1] == "one"
    assert m[2] == "two"
    
    # Test type checking on keys
    class StringMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    m = StringMap({"a": 1, "b": 2})
    assert m["a"] == 1
    
    # Test type checking violation on keys
    try:
        StringMap({1: 1})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError as e:
        assert e.source_class == StringMap
    
    # Test type checking violation on values
    try:
        StringMap({"a": "not_an_int"})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass
    
    # Test with multiple allowed types
    class MultiTypeMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = (int, float)
    
    m = MultiTypeMap({1: 1.5, "two": 2})
    assert m[1] == 1.5
    assert m["two"] == 2
    
    # Test invariant checking
    class PositiveMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, "Value must be positive")
    
    m = PositiveMap({1: 5, 2: 10})
    assert m[1] == 5
    
    try:
        PositiveMap({1: -5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass
    
    # Test that existing CheckedPMap instance is returned unchanged
    m1 = SimpleMap({1: "one"})
    m2 = SimpleMap(m1)
    assert m1 is m2
    
    # Test with internal size parameter (private API)
    internal_map = SimpleMap({1: "one"})._map
    m = SimpleMap(internal_map, size=1)
    assert m[1] == "one"
    
    # Test with custom serializer
    class CustomSerializerMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
        __serializer__ = lambda self, _, k, v: (str(k), v.upper())
    
    m = CustomSerializerMap({1: "hello"})
    serialized = m.serialize()
    assert serialized == {"1": "HELLO"}
    
    # Test inheritance of type definitions
    class BaseMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    class DerivedMap(BaseMap):
        pass
    
    m = DerivedMap({1: "test"})
    assert isinstance(m, DerivedMap)
    assert m[1] == "test"
    
    # Test that non-dict initializer raises appropriate errors
    try:
        SimpleMap([(1, "one")])  # Should work since it's iterable
    except Exception:
        pass
    
    # Test with None values using optional
    class OptionalMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = optional(int)
    
    m = OptionalMap({"a": 1, "b": None})
    assert m["a"] == 1
    assert m["b"] is None


# LLM-generated content at query #22
#--------------------------

```python
def test_CheckedPMap___new__():
    # Test basic creation with empty dict
    class SimpleMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    m = SimpleMap()
    assert isinstance(m, SimpleMap)
    assert len(m) == 0
    
    # Test creation with initial data
    m = SimpleMap({1: "one", 2: "two"})
    assert m[1] == "one"
    assert m[2] == "two"
    assert len(m) == 2
    
    # Test creation with existing CheckedPMap instance
    m2 = SimpleMap(m)
    assert m2[1] == "one"
    assert m2[2] == "two"
    
    # Test type checking on keys
    class StringMap(CheckedPMap):
        __key_type__ = str
        __value_type__ = int
    
    m = StringMap({"a": 1, "b": 2})
    assert m["a"] == 1
    
    # Test type checking raises error for wrong key type
    try:
        StringMap({1: 1})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass
    
    # Test type checking raises error for wrong value type
    try:
        StringMap({"a": "not_int"})
        assert False, "Should have raised CheckedValueTypeError"
    except CheckedValueTypeError:
        pass
    
    # Test with multiple allowed types
    class MultiTypeMap(CheckedPMap):
        __key_type__ = (int, str)
        __value_type__ = (int, float)
    
    m = MultiTypeMap({1: 1.5, "two": 2})
    assert m[1] == 1.5
    assert m["two"] == 2
    
    # Test invariant checking
    class PositiveMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = int
        __invariant__ = lambda k, v: (v > 0, "Value must be positive")
    
    m = PositiveMap({1: 5, 2: 10})
    assert m[1] == 5
    
    # Test invariant violation raises error
    try:
        PositiveMap({1: -5})
        assert False, "Should have raised InvariantException"
    except InvariantException:
        pass
    
    # Test with size parameter (internal use)
    internal_pmap = pmap({1: "a", 2: "b"})
    m = SimpleMap(internal_pmap, size=2)
    assert m[1] == "a"
    assert m[2] == "b"
    
    # Test inheritance of type constraints
    class BaseMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
    
    class DerivedMap(BaseMap):
        pass
    
    m = DerivedMap({1: "test"})
    assert isinstance(m, DerivedMap)
    assert m[1] == "test"
    
    # Test that wrong type in derived class still raises error
    try:
        DerivedMap({"string_key": "value"})
        assert False, "Should have raised CheckedKeyTypeError"
    except CheckedKeyTypeError:
        pass
    
    # Test with optional types
    class OptionalMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = optional(str, type(None))
    
    m = OptionalMap({1: "hello", 2: None})
    assert m[1] == "hello"
    assert m[2] is None
    
    # Test evolver integration
    m = SimpleMap({1: "a"})
    evolver = m.evolver()
    evolver.set(2, "b")
    m2 = evolver.persistent()
    assert m2[2] == "b"
    assert isinstance(m2, SimpleMap)


