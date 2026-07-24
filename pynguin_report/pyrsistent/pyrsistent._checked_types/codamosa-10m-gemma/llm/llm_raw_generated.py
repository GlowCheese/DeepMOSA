####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    def invariant_true(x):
        return True, "success"
    
    wrapped_true = wrap_invariant(invariant_true)
    assert wrapped_true(1) == (True, "success")

    # Case 2: Invariant returns a simple boolean (False)
    def invariant_false(x):
        return False, "failure"
    
    wrapped_false = wrap_invariant(invariant_false)
    assert wrapped_false(1) == (False, "failure")

    # Case 3: Invariant returns a list of results (all True)
    def invariant_all_pass(x):
        return [(True, "val1"), (True, "val2")]
    
    wrapped_all_pass = wrap_invariant(invariant_all_pass)
    assert wrapped_all_pass(1) == (True, ("val1", "val2"))

    # Case 4: Invariant returns a list of results (some False)
    def invariant_some_fail(x):
        return [(True, "val1"), (False, "err1"), (True, "val2"), (False, "err2")]
    
    wrapped_some_fail = wrap_invariant(invariant_some_fail)
    # Should return False and a tuple of only the error data
    assert wrapped_some_fail(1) == (False, ("err1", "err2"))

    # Case 5: Invariant returns a list of results (all False)
    def invariant_all_fail(x):
        return [(False, "err1"), (False, "err2")]
    
    wrapped_all_fail = wrap_invariant(invariant_all_fail)
    assert wrapped_all_fail(1) == (False, ("err1", "err2"))

    # Case 6: Verifying that the wrapper passes arguments correctly
    def invariant_check_args(a, b, c=3):
        return (a == b == c), "matches"
    
    wrapped_args = wrap_invariant(invariant_check_args)
    assert wrapped_args(5, 5, c=5) == (True, "matches")
    assert wrapped_args(5, 5, c=10) == (False, "") # Error data is empty string in this mock
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test__CheckedMapTypeMeta___new__():
    """
    Tests the __new__ method of _CheckedMapTypeMeta.
    It verifies that:
    1. __key_type__ is correctly extracted from the class dict.
    2. __value_type__ is correctly extracted from the class dict.
    3. __invariant__ is correctly extracted from the class dict.
    4. __serializer__ is set with a default implementation.
    5. Inheritance works (types and invariants are inherited from bases).
    6. __slots__ is set to an empty tuple.
    """
    
    def sample_invariant(dct, key, value):
        return True

    class BaseMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = int
        __value_type__ = str
        __invariant__ = sample_invariant

    class ChildMap(BaseMap):
        __key_type__ = str
        # __value_type__ and __invariant__ are inherited
        __slots__ = ('extra',) # To test that __slots__ is overwritten by metaclass

    # 1. Test type extraction and inheritance for ChildMap
    assert ChildMap._checked_key_types == [str]
    assert ChildMap._checked_value_types == [str]
    assert ChildMap._checked_invariants == (wrap_invariant(sample_invariant),)

    # 2. Test BaseMap extraction
    assert BaseMap._checked_key_types == [int]
    assert BaseMap._checked_value_types == [str]
    assert BaseMap._checked_invariants == (wrap_invariant(sample_invariant),)

    # 3. Test __serializer__ default implementation
    # The default serializer should handle CheckedType serialization
    class MockCheckedType(CheckedType):
        def serialize(self, format=None):
            return "serialized"
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=None):
            return cls()

    mock_key = MockCheckedType()
    mock_val = MockCheckedType()
    
    # Test the default serializer logic inside the metaclass
    # Expected behavior: if value is CheckedType, call .serialize()
    # We use a dummy instance of ChildMap to access the __serializer__ method
    dummy_instance = ChildMap()
    
    # Check serialization of CheckedType objects
    serialized_key, serialized_val = ChildMap.__serializer__(dummy_instance, mock_key, mock_val)
    assert serialized_key == "serialized"
    assert serialized_val == "serialized"

    # Check serialization of standard objects
    key_std, val_std = ChildMap.__serializer__(dummy_instance, 1, "hello")
    assert key_std == 1
    assert val_std == "hello"

    # 4. Test __slots__ enforcement
    # The metaclass must set __slots__ = () regardless of what is in the class dict
    assert ChildMap.__slots__ == ()
    assert ChildMap.__dict__['__slots__'] == ()

    # 5. Test __invariant__ wrapping
    # Ensure that the invariant is wrapped by wrap_invariant (which handles the result merging)
    # Since our sample_invariant returns True, wrap_invariant returns (True,)
    # However, _CheckedMapTypeMeta stores the result of wrap_invariant(inv)
    # We verify the result is a tuple of wrapped functions.
    assert isinstance(ChildMap._checked_invariants, tuple)
    # Calling the wrapped invariant should return (True,)
    assert ChildMap._checked_invariants[0](None, None, None) == (True,)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test__CheckedMapTypeMeta___new__():
    """
    Tests the __new__ method of _CheckedMapTypeMeta, ensuring it correctly:
    1. Inherits/stores key types from __key_type__.
    2. Inherits/stores value types from __value_type__.
    3. Inherits/stores invariants from __invariant__.
    4. Sets a default __serializer__.
    5. Sets __slots__ to an empty tuple.
    """

    # Define a base class with specifications
    class BaseMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = (int,)
        __value_type__ = (str,)
        __invariant__ = lambda self, k, v: (True, None)

    # Define a subclass that overrides some specs and adds new ones
    class SubMap(BaseMap):
        __key_type__ = (str,)  # Overrides int
        __value_type__ = (int, float)  # Overrides str
        __invariant__ = lambda self, k, v: (v > 0, "Must be positive")

    # 1. Test key type storage (should prioritize subclass)
    assert SubMap._checked_key_types == (str,)
    # Verify inheritance of types not present in subclass (if we had any)
    # In this specific implementation, _store_types replaces/merges via the list comprehension
    
    # 2. Test value type storage
    assert SubMap._checked_value_types == (int, float)

    # 3. Test invariant storage (should be a tuple of wrapped callables)
    # The metaclass wraps invariants using wrap_invariant
    assert len(SubMap._checked_invariants) == 1
    # Check if the invariant logic works (it should return (bool, data))
    res_bool, res_data = SubMap._checked_invariants[0](None, "key", 10)
    assert res_bool is True
    assert res_data is None

    # 4. Test default serializer
    # The serializer should handle CheckedType objects by calling .serialize()
    class MockCheckedType(CheckedType):
        def serialize(self, format=None):
            return "serialized"
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=None):
            return cls()

    instance = SubMap()
    mock_ct = MockCheckedType()
    
    # Test serialization of a CheckedType value
    key, val = SubMap.__serializer__(instance, mock_ct, "value")
    assert key == "serialized"
    assert val == "value"

    # Test serialization of a normal value
    key2, val2 = SubMap.__serializer__(instance, "key", mock_ct)
    assert key2 == "key"
    assert val2 == "serialized"

    # 5. Test __slots__
    assert SubMap.__slots__ == ()
    assert BaseMap.__slots__ == ()

def test__CheckedMapTypeMeta___new__ inheritance_logic():
    """
    Verifies that _store_types correctly aggregates types from the hierarchy.
    """
    class GrandParent(metaclass=_CheckedMapTypeMeta):
        __key_type__ = (int,)

    class Parent(GrandParent):
        __value_type__ = (str,)

    class Child(Parent):
        pass

    # Child should have both key_type from GrandParent and value_type from Parent
    # Note: The implementation of _store_types uses a list comprehension over [dct] + bases
    # so it collects all defined types in the MRO.
    assert int in Child._checked_key_types
    assert str in Child._checked_value_types
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_store_invariants():
    # Mock invariant functions
    def inv1(data):
        return True, "ok"

    def inv2(data):
        return False, "error"

    def inv3(data):
        return (True, "sub_ok"), (False, "sub_error")

    # Test Case 1: Basic functionality - storing a single invariant
    class Base1:
        pass

    class Derived1(Base1):
        my_invariant = inv1

    assert len(Derived1.my_invariants) == 1
    assert Derived1.my_invariants[0](None) == (True, "ok")

    # Test Case 2: Inheritance - invariants should be inherited from bases
    class Base2:
        base_inv = inv1

    class Derived2(Base2):
        derived_inv = inv2

    # Note: store_invariants is used by a metaclass (implied), 
    # here we simulate the logic applied to the class dict
    class MockMeta(type):
        def __new__(cls, name, bases, dct):
            store_invariants(dct, bases, 'my_invariants', 'some_source_key')
            return super().__new__(cls, name, bases, dct)

    # We need to simulate the way store_invariants is called by a metaclass
    # Since we can't easily trigger __new__ without a real metaclass, 
    # we manually call it on a dict as the function does.
    
    dct = {'some_source_key': inv1}
    bases = (Base2,)
    store_invariants(dct, bases, 'my_invariants', 'some_source_key')
    
    # It should find inv1 from dct and inv1 from Base2 (if it were there)
    # In our setup, Base2 has 'base_inv', not 'some_source_key'
    # So it should only find the one in dct.
    assert len(dct['my_invariants']) == 1

    # Test Case 3: Multiple invariants and wrapped results
    # Testing the wrap_invariant logic inside store_invariants
    dct_multi = {'some_source_key': inv3} # inv3 returns a tuple of results
    store_invariants(dct_multi, (), 'my_invariants', 'some_source_key')
    
    # The result of inv3 is (True, "sub_ok"), (False, "sub_error")
    # wrap_invariant should merge this to (False, ("sub_error",))
    verdict, errors = dct_multi['my_invariants'][0](None)
    assert verdict is False
    assert errors == ("sub_error",)

    # Test Case 4: TypeError when invariant is not callable
    dct_invalid = {'some_source_key': "not_callable"}
    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants(dct_invalid, (), 'my_invariants', 'some_source_key')

    # Test Case 5: Inheritance of multiple keys
    class Base3:
        some_source_key = inv1

    class Derived3(Base3):
        pass

    # Manual simulation of the metadata population
    dct_inheritance = {'some_source_key': inv2}
    # Base3 has 'some_source_key'
    store_invariants(dct_inheritance, (Base3,), 'my_invariants', 'some_source_key')
    
    # Should contain both inv2 (from dct) and inv1 (from Base3)
    # The order depends on implementation, but both should be present
    assert len(dct_inheritance['my_invariants']) == 2
    
    # Verify that the wrapped functions work
    results = [f(None) for f in dct_inheritance['my_invariants']]
    # One should be True, one should be False
    verdicts = [r[0] for r in results]
    assert True in verdicts
    assert False in verdicts
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test__CheckedMapTypeMeta___new__():
    # Define a base class with type specifications and invariants
    class BaseMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = (str,)
        __value_type__ = (int,)
        __invariant__ = lambda self: (True, "No error")

    # Define a subclass that overrides or adds to the specifications
    class SubMap(BaseMap):
        __key_type__ = (int, str)
        __invariant__ = lambda self: (True, "Sub error")

    # Test that __new__ correctly aggregated and stored types in _checked_key_types
    # It should contain the union/list of types from both classes
    assert SubMap._checked_key_types == [int, str]
    
    # Test that __new__ correctly aggregated and stored types in _checked_value_types
    assert SubMap._checked_value_types == [int]

    # Test that __new__ correctly aggregated invariants
    # The wrap_invariant decorator is applied by store_invariants via the metaclass
    # We check the count of invariants (BaseMap's + SubMap's)
    assert len(SubMap._checked_invariants) == 2

    # Test that __new__ sets the default __serializer__
    assert callable(SubMap.__serializer__)
    
    # Test the functionality of the default serializer created by __new__
    # Create a dummy CheckedType for serialization testing
    class MockCheckedType(CheckedType):
        def __init__(self, val):
            self.val = val
        def serialize(self, format=None):
            return f"serialized_{self.val}"
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls(source_data)

    instance = SubMap()
    key = MockCheckedType("key")
    val = MockCheckedType(123)
    
    # The serializer should call .serialize() on CheckedType objects
    serialized_key, serialized_val = SubMap.__serializer__(instance, key, val)
    assert serialized_key == "serialized_key"
    assert serialized_val == "serialized_123"

    # Test that __new__ sets __slots__ to an empty tuple
    assert SubMap.__slots__ == ()
    assert BaseMap.__slots__ == ()

    # Test that the serializer handles non-CheckedType values normally
    sk, sv = SubMap.__serializer__(instance, "plain_key", 456)
    assert sk == "plain_key"
    assert sv == 456
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Test case 1: Invariant returns a single boolean result
    @wrap_invariant(lambda x: (x > 0, "too small") if x <= 0 else (True, ()))
    def simple_invariant(x):
        return (x > 0, "too small") if x <= 0 else (True, ())

    assert simple_invariant(5) == (True, ())
    assert simple_invariant(-1) == (False, ("too small",))

    # Test case 2: Invariant returns a tuple of (bool, data) pairs
    # This simulates the logic where the wrapper merges multiple results
    def multi_result_invariant(x):
        # Returns a list/tuple of (verdict, error_data)
        return [
            (x > 0, "must be positive"),
            (x < 10, "must be less than 10"),
            (x % 2 == 0, "must be even")
        ]
    
    wrapped_multi = wrap_invariant(multi_result_invariant)

    # Case: All pass
    # Note: Since we can't easily pass 2, 4, 6 etc through the specific 
    # multi_result_invariant logic without it failing one, 
    # let's use a controlled version.
    
    def controlled_multi(x):
        return [
            (x > 0, "pos"),
            (x < 10, "small")
        ]
    
    wrapped_controlled = wrap_invariant(controlled_multi)
    
    # All pass
    assert wrapped_controlled(5) == (True, ())
    
    # One fails
    assert wrapped_controlled(-1) == (False, ("pos",))
    
    # Multiple fail
    assert wrapped_controlled(15) == (False, ("small",))
    
    # Both fail
    assert wrapped_controlled(-5) == (False, ("pos", "small"))

    # Test case 3: Invariant returns a simple boolean directly
    @wrap_invariant(lambda x: x == 1)
    def direct_bool_invariant(x):
        return x == 1

    assert direct_bool_invariant(1) is True
    assert direct_bool_invariant(2) is False

    # Test case 4: Verifying the merge logic for empty results
    @wrap_invariant(lambda: [(True, ""), (True, "")])
    def empty_fail_list():
        return [(True, ""), (True, "")]
    
    assert empty_fail_list() == (True, ())
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple (bool, data) tuple
    def simple_invariant(x):
        return (x > 0, "error_msg")

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(10) == (True, ())
    assert wrapped_simple(-1) == (False, ("error_msg",))

    # Case 2: Invariant returns a boolean directly
    def bool_only_invariant(x):
        return x == 0

    wrapped_bool = wrap_invariant(bool_only_invariant)
    assert wrapped_bool(0) == True
    assert wrapped_bool(5) == False

    # Case 3: Invariant returns a list of (bool, data) tuples (multiple results)
    def multi_result_invariant(x):
        return [
            (x > 0, "pos_error"),
            (x < 10, "range_error"),
            (x % 2 == 0, "even_error")
        ]

    wrapped_multi = wrap_invariant(multi_result_invariant)
    
    # Test passing all checks
    assert wrapped_multi(4) == (True, ())
    
    # Test failing one check
    assert wrapped_multi(-1) == (False, ("pos_error", "even_error"))
    
    # Test failing multiple checks
    assert wrapped_multi(11) == (False, ("range_error"))
    
    # Test failing all checks
    assert wrapped_multi(-2) == (False, ("pos_error", "range_error", "even_error"))

    # Case 4: Invariant returns a single (bool, data) tuple where result[0] is bool
    # The wrapper should return it as-is without attempting to merge
    def single_tuple_invariant(x):
        return (x < 5, "not_a_list_of_tuples")

    wrapped_single_tuple = wrap_invariant(single_tuple_invariant)
    assert wrapped_single_tuple(2) == (True, "not_a_list_of_tuples")
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_store_invariants():
    # Test case 1: Basic functionality - storing a single invariant
    class Base:
        def my_invariant(self, x):
            return x > 0

    class Derived(Base):
        pass

    # Check that the invariant is correctly stored in the class dict
    # We need to manually trigger the descriptor-like behavior or 
    # simulate what the decorator/metaclass would do.
    # Since store_invariants is designed to be used in a metaclass __new__ or __init__,
    # we call it directly on a dictionary.
    
    class_dict = {}
    store_invariants(class_dict, (Base,), "invariants", "my_invariant")
    
    assert "invariants" in class_dict
    assert len(class_dict["invariants"]) == 1
    
    # Test case 2: Inheritance - invariants should be inherited from bases
    class GrandBase:
        def inv1(self, x): return True
        def inv2(self, x): return True

    class Parent(GrandBase):
        def inv3(self, x): return True

    class Child(Parent):
        pass

    child_dict = {}
    store_invariants(child_dict, (Parent, GrandBase), "invariants", "inv1") 
    # Note: the implementation looks for source_name in the dicts.
    # Let's test a more realistic scenario where we provide a source_name that exists.
    
    class MockMeta(type):
        def __new__(cls, name, bases, dct):
            store_invariants(dct, bases, "invariants", "check")
            return super().__new__(cls, name, bases, dct)

    class Root:
        def check(self, val):
            return val == "ok"

    class Sub(Root, metaclass=MockMeta):
        pass

    assert "invariants" in Sub.__dict__
    # The wrapped invariant should return (True, ()) for a successful check
    assert Sub.invariants[0]("ok") == (True, ())
    assert Sub.invariants[0]("fail") == (False, ("fail",))

    # Test case 3: Error handling - non-callable invariant
    class BadBase:
        check = "not a callable"

    bad_dict = {}
    with pytest.raises(TypeError, match="Invariants must be callable"):
        store_invariants(bad_dict, (BadBase,), "invariants", "check")

    # Test case 4: Multiple invariants from different levels of hierarchy
    class Level1:
        def check(self, x): return x > 0
    
    class Level2(Level1):
        def check(self, x): return x < 10

    # We simulate the metaclass logic for Level2
    level2_dict = {"check": lambda x: x < 10}
    store_invariants(level2_dict, (Level1,), "invariants", "check")
    
    # It should have wrapped both Level1.check and Level2.check
    # Based on the implementation: it collects all 'check' from dct and bases
    assert len(level2_dict["invariants"]) == 2
    
    # Test the wrapped execution logic
    # One invariant returns a bool, the other returns a tuple (simulating wrap_invariant)
    # We need to verify if the logic handles both.
    # Let's mock the invariants to return different structures.
    
    class ComplexBase:
        def check(self, x): return True # returns bool
        
    class ComplexDerived(ComplexBase):
        def check(self, x): return (False, "error") # returns tuple
        
    comp_dict = {"check": lambda x: (False, "error")}
    store_invariants(comp_dict, (ComplexBase,), "invariants", "check")
    
    # The result should be a tuple of wrapped invariants
    # The wrapper 'f' handles both boolean and tuple returns.
    # If result[0] is bool, it returns result.
    # If result[0] is not bool (it's a tuple), it merges.
    
    # Let's test the resulting function manually
    inv_funcs = comp_dict["invariants"]
    # Test the wrapped version of the tuple-returning one
    # The implementation of wrap_invariant calls the original.
    # If original is (False, "error"), result[0] is False (bool).
    # So it returns (False, "error") directly.
    assert inv_funcs[0]("anything") == (False, "error")

    # Test case 5: No invariants found
    empty_dict = {}
    store_invariants(empty_dict, (object,), "invariants", "non_existent")
    assert empty_dict["invariants"] == ()
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_store_invariants():
    # Mock invariants
    def invariant_1(obj):
        return True, "ok"

    def invariant_2(obj):
        return False, "fail"

    def invalid_invariant(obj):
        return "not a boolean or tuple"

    # Case 1: Basic functionality - storing a single invariant
    class Base:
        pass

    class Child(Base):
        val = invariant_1

    assert Child.val == (wrap_invariant(invariant_1),)
    # Verify wrap_invariant logic: if it returns a tuple (bool, data), it merges
    # In this case, invariant_1 returns (True, "ok"), wrap_invariant returns (True, ("ok",))
    assert Child.val[0] == (True, ("ok",))

    # Case 2: Inheritance - collecting invariants from base classes
    class GrandParent:
        inv_gp = invariant_1

    class Parent(GrandParent):
        inv_p = invariant_2

    class GrandChild(Parent):
        pass

    # GrandChild should inherit gp and p invariants
    # Note: wrap_invariant wraps each one individually in the tuple
    # The resulting tuple contains the wrapped functions
    assert len(GrandChild.val) == 2
    
    # Case 3: Verification of wrapped execution
    # Let's check if the wrapped function behaves as expected when called
    # wrap_invariant(invariant_1) -> returns (True, ('ok',))
    # wrap_invariant(invariant_2) -> returns (False, ('fail',))
    
    # We need to find the functions in the class dict manually to test them
    # because store_invariants puts the wrapped functions into the dict.
    # We use the actual functions from the class.
    
    # Case 4: Error handling - Non-callable invariants
    class BadClass:
        bad_inv = "not callable"

    with pytest.raises(TypeError, match="Invariants must be callable"):
        store_invariants(BadClass.__dict__, (BadClass,), 'dest', 'bad_inv')

    # Case 5: Multiple inheritance
    class Mixin:
        mixin_inv = invariant_1

    class MultiChild(Parent, Mixin):
        pass

    # Should have gp, p, and mixin invariants
    assert len(MultiChild.val) == 3

    # Case 6: No invariants found
    class EmptyClass:
        pass

    assert EmptyClass.val == ()

    # Case 7: Testing the merge logic of wrap_invariant via a mock
    def multi_result_invariant(obj):
        return [(True, "a"), (False, "b"), (True, "c")]
    
    wrapped_multi = wrap_invariant(multi_result_invariant)
    # Should merge to (False, ("b",))
    assert wrapped_multi(None) == (False, ("b",))

    def single_result_invariant(obj):
        return (True, "a")
    
    wrapped_single = wrap_invariant(single_result_invariant)
    # Should return as is if first element is bool
    assert wrapped_single(None) == (True, "a")
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Test case 1: Invariant returns a simple (bool, data) tuple
    def simple_invariant(val):
        return (val > 0, "positive")

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(10) == (True, "")
    assert wrapped_simple(-1) == (False, "positive")

    # Test case 2: Invariant returns a list of (bool, data) tuples (merging logic)
    def multi_invariant(val):
        return [
            (val > 0, "must be positive"),
            (val % 2 == 0, "must be even"),
            (val < 100, "must be small")
        ]

    wrapped_multi = wrap_invariant(multi_invariant)
    
    # All pass: result should be (True, ())
    assert wrapped_multi(10) == (True, ())
    
    # One fails: result should be (False, ('must be even',))
    assert wrapped_multi(11) == (False, ("must be even",))
    
    # Two fail: result should be (False, ('must be even', 'must be small'))
    # (Note: 101 is not even and not small)
    assert wrapped_multi(101) == (False, ("must be even", "must be small"))
    
    # All fail
    assert wrapped_multi(-2) == (False, ("must be positive", "must be even"))

    # Test case 3: Invariant returns a single bool (should return as is)
    def bool_only_invariant(val):
        return val == "ok"

    wrapped_bool = wrap_invariant(bool_only_invariant)
    assert wrapped_bool("ok") is True
    assert wrapped_bool("fail") is False

    # Test case 4: Verifying complex merging with different types of data
    def complex_invariant(val):
        return [
            (isinstance(val, int), "not an int"),
            (val > 5, "too small")
        ]
    
    wrapped_complex = wrap_invariant(complex_invariant)
    assert wrapped_complex(10) == (True, ())
    assert wrapped_complex(3) == (False, ("too small",))
    assert wrapped_complex("string") == (False, ("not an int", "too small"))
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    def invariant_true(data):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    assert wrapped_true({"x": 1}) == (True, [])

    # Case 2: Invariant returns a simple boolean (False)
    def invariant_false(data):
        return False, ["error"]
    
    wrapped_false = wrap_invariant(invariant_false)
    assert wrapped_false({"x": 1}) == (False, ["error"])

    # Case 3: Invariant returns a tuple of results (multiple checks)
    # Result format: (bool, data)
    def invariant_multiple(data):
        return (
            (True, "pass1"),
            (False, "fail1"),
            (True, "pass2"),
            (False, "fail2")
        )
    
    wrapped_multiple = wrap_invariant(invariant_multiple)
    # Should merge: verdict is False because one failed, data contains only failed data
    assert wrapped_multiple({"x": 1}) == (False, ("fail1", "fail2"))

    # Case 4: Invariant returns all passing checks
    def invariant_all_pass(data):
        return (
            (True, "pass1"),
            (True, "pass2"),
        )
    
    wrapped_all_pass = wrap_invariant(invariant_all_pass)
    assert wrapped_all_pass({"x": 1}) == (True, ())

    # Case 5: Invariant returns a single tuple result (bool, data)
    def invariant_single_tuple(data):
        return (True, "only_one")
    
    wrapped_single = wrap_invariant(invariant_single_tuple)
    # If the first element is bool, it returns the tuple directly
    assert wrapped_single({"x": 1}) == (True, "only_one")
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Test case 1: Invariant returns a single boolean (True)
    def invariant_true(data):
        return (True, "all good")
    
    wrapped_true = wrap_invariant(invariant_true)
    assert wrapped_true({"x": 1}) == (True, "all good")

    # Test case 2: Invariant returns a single boolean (False)
    def invariant_false(data):
        return (False, "failed")
    
    wrapped_false = wrap_invariant(invariant_false)
    assert wrapped_false({"x": 1}) == (False, "failed")

    # Test case 3: Invariant returns a list of (bool, data) tuples, all True
    def invariant_all_pass(data):
        return [(True, "val1"), (True, "val2")]
    
    wrapped_pass = wrap_invariant(invariant_all_pass)
    assert wrapped_pass({"x": 1}) == (True, ("val1", "val2"))

    # Test case 4: Invariant returns a list of (bool, data) tuples, some False
    def invariant_mixed(data):
        return [(True, "val1"), (False, "error1"), (True, "val2"), (False, "error2")]
    
    wrapped_mixed = wrap_invariant(invariant_mixed)
    # Should return False and only the data from the failed tests
    assert wrapped_mixed({"x": 1}) == (False, ("error1", "error2"))

    # Test case 5: Invariant returns a list of (bool, data) tuples, all False
    def invariant_all_fail(data):
        return [(False, "err1"), (False, "err2")]
    
    wrapped_fail = wrap_invariant(invariant_all_fail)
    assert wrapped_fail({"x": 1}) == (False, ("err1", "err2"))
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple (bool, data) tuple
    def simple_invariant(x):
        return (x > 0, "positive")
    
    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, ())
    assert wrapped_simple(-5) == (False, ("positive",))

    # Case 2: Invariant returns a single boolean (no data)
    def boolean_only_invariant(x):
        return x == 10
    
    wrapped_bool = wrap_invariant(boolean_only_invariant)
    assert wrapped_bool(10) == True
    assert wrapped_bool(5) == False

    # Case 3: Invariant returns a list of (bool, data) tuples (multiple results)
    def multi_result_invariant(x):
        return [
            (x > 0, "must be positive"),
            (x < 10, "must be less than 10"),
            (x % 2 == 0, "must be even")
        ]
    
    wrapped_multi = wrap_invariant(multi_result_invariant)
    
    # All pass
    # Note: Since 2 is not < 10 logic is fine, but we need a number that satisfies all
    # Let's use 2: 2 > 0 (T), 2 < 10 (T), 2 % 2 == 0 (T)
    # Wait, the return of wrap_invariant for multi results is (verdict, tuple_of_errors)
    # If all are True, verdict is True, data is empty tuple.
    assert wrapped_multi(2) == (True, ())

    # One fails
    # 11: 11 > 0 (T), 11 < 10 (F), 11 % 2 == 0 (F)
    # Errors: "must be less than 10", "must be even"
    assert wrapped_multi(11) == (False, ("must be less than 10", "must be even"))

    # All fail
    # -1: -1 > 0 (F), -1 < 10 (T), -1 % 2 == 0 (F)
    # Errors: "must be positive", "must be even"
    assert wrapped_multi(-1) == (False, ("must be positive", "must be even"))

    # Case 4: Invariant returns a single (bool, data) tuple but logic is wrapped
    # Verifying the isinstance(result[0], bool) check
    def single_tuple_invariant(x):
        return (x == 0, "zero error")
    
    wrapped_single_tuple = wrap_invariant(single_tuple_invariant)
    assert wrapped_single_tuple(0) == (True, ())
    assert wrapped_single_tuple(1) == (False, ("zero error",))
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_CheckedPMap___new__():
    # Test case 1: Initializing with a dictionary (standard usage)
    initial_data = {1: "a", 2: "b"}
    mapping = CheckedPMap(initial_data)
    assert isinstance(mapping, CheckedPMap)
    assert mapping[1] == "a"
    assert mapping[2] == "b"
    assert len(mapping) == 2

    # Test case 2: Initializing with a size (using the internal size-specific constructor)
    # Note: _UNDEFINED_CHECKED_PMAP_SIZE is used to trigger the super().__new__ path
    # We simulate this by passing a specific size.
    size_val = 5
    mapping_with_size = CheckedPMask_new_internal_sim(size_val, initial_data)
    assert mapping_with_size[1] == "a"
    assert len(mapping_with_size) == 2
    
    # Test case 3: Initializing with empty dictionary
    empty_mapping = CheckedPMap({})
    assert len(empty_mapping) == 0

    # Test case 4: Verifying that type-checking/invariants are applied during construction
    class IntStringMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str
        __invariant__ = lambda k, v: (len(str(v)) > 0, "Empty string not allowed")

    # This should succeed
    valid_map = IntStringMap({1: "valid"})
    assert valid_map[1] == "valid"

    # This should raise InvariantException during .persistent() call inside __new__
    # because the Evolver collects errors and raises them on persistent()
    with pytest.raises(InvariantException):
        IntStringMap({1: ""})

def CheckedPMask_new_internal_sim(size, initial):
    """
    Helper to simulate the specific branch of __new__ where size is provided.
    Since we cannot easily access the private _UNDEFINED_CHECKED_PMAP_SIZE 
    from outside without imports, we mimic the logic of the branch:
    if size is not _UNDEFINED_CHECKED_PMAP_SIZE: return super().__new__(...)
    """
    # We bypass the 'is _UNDEFINED' check by providing a real integer
    # This triggers the branch: return super(CheckedPMap, cls).__new__(cls, size, initial)
    # We use a subclass to avoid polluting global state
    class MockMap(CheckedPMap):
        pass
    
    # We simulate the logic of the 'else' branch manually for the test
    evolver = MockMap.Evolver(MockMap, pmap())
    for k, v in initial.items():
        evolver.set(k, v)
    return evolver.persistent()
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Test case 1: Invariant returns a simple boolean (True)
    def invariant_true(data):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    assert wrapped_true({"x": 1}) == (True, [])

    # Test case 2: Invariant returns a simple boolean (False)
    def invariant_false(data):
        return False, ["error"]
    
    wrapped_false = wrap_invariant(invariant_false)
    assert wrapped_false({"x": 1}) == (False, ["error"])

    # Test case 3: Invariant returns a tuple of results (all passing)
    def invariant_all_pass(data):
        return [(True, None), (True, "nothing")]
    
    wrapped_pass = wrap_invariant(invariant_all_pass)
    assert wrapped_pass({"x": 1}) == (True, ())

    # Test case 4: Invariant returns a tuple of results (one failing)
    def invariant_one_fails(data):
        return [(True, "good"), (False, "bad_error"), (True, "ok")]
    
    wrapped_fail = wrap_invariant(invariant_one_fails)
    assert wrapped_fail({"x": 1}) == (False, ("bad_error",))

    # Test case 5: Invariant returns multiple failures
    def invariant_multiple_fails(data):
        return [(False, "err1"), (False, "err2"), (True, "ignore")]
    
    wrapped_multi_fail = wrap_invariant(invariant_multiple_fails)
    assert wrapped_multi_fail({"x": 1}) == (False, ("err1", "err2"))

    # Test case 6: Invariant returns a result where the first element is boolean 
    # (should bypass merging logic)
    def invariant_bool_first(data):
        return (True, ["should not be merged"])
    
    wrapped_bool_first = wrap_invariant(invariant_bool_first)
    assert wrapped_bool_first({"x": 1}) == (True, ["should not be merged"])
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test__CheckedMapTypeMeta___new__():
    # Define a base class using the metaclass
    class BaseMap(metaclass=_CheckedMapTypeMeta):
        __key_type__ = str
        __value_type__ = int
        __invariant__ = lambda self, k, v: (v > 0, "Must be positive")

    # Define a subclass that overrides/adds types and invariants
    class SubMap(BaseMap):
        __key_type__ = (str, int)
        # __value_type__ is inherited from BaseMap
        __invariant__ = lambda self, k, v: (isinstance(k, str), "Key must be string")

    # Test 1: Verify type storage for keys and values
    assert SubMap._checked_key_types == (str, int)
    assert SubMap._checked_value_types == (int,)

    # Test 2: Verify invariant inheritance and wrapping
    # SubMap should have both invariants, wrapped by wrap_invariant
    assert len(SubMap._checked_invariants) == 2
    
    # Check if the first invariant is the one from SubMap
    # The implementation of wrap_invariant returns the result of the function
    # We simulate calling the wrapped invariant
    sample_map = SubMap()
    
    # Test SubMap's specific invariant
    # Note: The way wrap_invariant is implemented, it calls the original function
    # We check if the result is the expected boolean/tuple from the original function
    res1 = SubMap._checked_invariants[0](sample_map, "key", 10)
    assert res1 == (True, "Key must be string") # This actually depends on the logic in the code
    # Looking at the code: wrap_invariant(invariant) returns a function that calls invariant
    # and returns the result directly if it's a bool, or merges if it's a list of results.
    
    # Test 3: Verify __slots__ is set to empty tuple
    assert SubMap.__slots__ == ()

    # Test 4: Verify default serializer logic
    # The serializer should handle CheckedType serialization
    class MockCheckedType(CheckedType):
        def serialize(self, format=None):
            return "serialized"
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()

    mock_key = MockCheckedType()
    mock_val = MockCheckedType()
    
    # The serializer is stored in __serializer__
    # Signature: (self, format, key, value)
    serializer = SubMap.__serializer__
    
    # Test serialization of CheckedTypes
    k_ser, v_ser = serializer(None, mock_key, mock_val)
    assert k_ser == "serialized"
    assert v_ser == "serialized"

    # Test serialization of primitive types
    k_prim, v_prim = serializer(None, "plain_key", 123)
    assert k_prim == "plain_key"
    assert v_prim == 123

    # Test 5: Verify __slots__ is empty for the created class
    assert hasattr(SubMap, '__slots__')
    assert SubMap.__slots__ == ()
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Test Case 1: Invariant returns a simple boolean (True)
    def invariant_true(x):
        return True, "no errors"
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict, errors = wrapped_true(1)
    assert verdict is True
    assert errors == "no errors"

    # Test Case 2: Invariant returns a simple boolean (False)
    def invariant_false(x):
        return False, "error occurred"
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, errors = wrapped_false(1)
    assert verdict is False
    assert errors == "error occurred"

    # Test Case 3: Invariant returns a list of (bool, data) tuples (All True)
    def invariant_all_pass(x):
        return [(True, "ok"), (True, "fine")]
    
    wrapped_pass = wrap_invariant(invariant_all_pass)
    verdict, errors = wrapped_pass(1)
    assert verdict is True
    assert errors == ()

    # Test Case 4: Invariant returns a list of (bool, data) tuples (Some False)
    def invariant_mixed(x):
        return [(True, "ok"), (False, "error1"), (True, "fine"), (False, "error2")]
    
    wrapped_mixed = wrap_invariant(invariant_mixed)
    verdict, errors = wrapped_mixed(1)
    assert verdict is False
    assert errors == ("error1", "error2")

    # Test Case 5: Invariant returns a list of (bool, data) tuples (All False)
    def invariant_all_fail(x):
        return [(False, "err1"), (False, "err2")]
    
    wrapped_fail = wrap_invariant(invariant_all_fail)
    verdict, errors = wrapped_fail(1)
    assert verdict is False
    assert errors == ("err1", "err2")
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from enum import Enum

def test_maybe_parse_user_type():
    # Test single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test single string
    assert maybe_parse_user_type("int") == ["int"]
    
    # Test preserved iterable type (Enum)
    class MyEnum(Enum):
        A = 1
    assert maybe_parse_user_type(MyEnum) == [MyEnum]
    
    # Test list of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    
    # Test nested iterable of types
    assert maybe_parse_user_type([[int], [str, float]]) == (int, str, float)
    
    # Test list of strings
    assert maybe_parse_user_type(["int", "str"]) == ("int", "str")
    
    # Test nested iterable of strings
    assert maybe_parse_user_type([["int"], ["str"]]) == ("int", "str")

    # Test invalid input (e.g., an integer instead of a type or string)
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    # Test invalid input (e.g., a dictionary where keys are not types/strings)
    # Note: dict is iterable, so it iterates keys.
    with pytest.raises(TypeError):
        maybe_parse_user_type({1: "int"})
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_get_type():
    # Test with a direct type object
    assert get_type(int) == int
    assert get_type(str) == str
    assert get_type(list) == list
    
    # Test with a string representing a built-in type
    # Note: __import__ is used in get_type, so we use 'builtins' prefix
    assert get_type('builtins.int') == int
    assert get_type('builtins.str') == str
    assert get_type('builtins.list') == list

    # Test with a string representing a type from a standard module
    # Using os.path (which is a module/type depending on platform, but here we check class resolution)
    # We'll use a more reliable one: datetime.datetime
    assert get_type('datetime.datetime') == __import__('datetime').datetime

    # Test that it raises ImportError/AttributeError for invalid strings
    with pytest.raises((ImportError, AttributeError, ValueError)):
        get_type('non_existent_module.NonExistentClass')

    with pytest.raises((ImportError, AttributeError, ValueError)):
        # Case where module exists but class does not
        get_type('builtins.DoesNotExist')

    # Test that it raises ValueError if the string format is incorrect (no dot)
    with pytest.raises(ValueError):
        get_type('int')
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from enum import Enum

def test_maybe_parse_user_type():
    # Test simple types
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test string type specifications
    assert maybe_parse_user_type("int") == ["int"]
    
    # Test preserved iterable types (Enum)
    class MyEnum(Enum):
        A = 1
    assert maybe_parse_user(MyEnum) == [MyEnum]
    
    # Test list of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    
    # Test nested iterables
    assert maybe_parse_user_type([[int], str]) == (int, str)
    assert maybe_parse_user_type([(int, [str])]) == (int, str)
    
    # Test tuple of types
    assert maybe_parse_user_type((float, )) == (float,)
    
    # Test error case: non-type, non-string, non-iterable
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    with pytest.raises(TypeError):
        maybe_parse_user_type(None)

    # Test complex nested structure
    complex_type = [int, [str, (float,)], "bool"]
    assert maybe_parse_user_type(complex_type) == (int, str, float, "bool")
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test__CheckedMapTypeMeta___new__():
    # Define a base class using the metaclass
    class Base(metaclass=_CheckedMapTypeMeta):
        __key_type__ = (str,)
        __value_type__ = (int,)
        __invariant__ = lambda self, k, v: (v >= 0, "Negative Value")

    # Define a subclass that overrides/extends types and invariants
    class Sub(Base):
        __key_type__ = (int,)
        # __value_type__ is inherited from Base
        __invariant__ = lambda self, k, v: (k > 0, "Non-positive key")

    # 1. Test Type Storage: Check if types are correctly merged and stored in __dict__
    # Note: _store_types pulls from the class and its bases
    assert Sub._checked_key_types == (int,)
    assert Sub._checked_value_types == (int,)
    
    # 2. Test Invariant Storage: Check if invariants are inherited and wrapped
    # The metaclass calls store_invariants which wraps them with wrap_invariant
    assert len(Sub._checked_invariants) == 2
    
    # Test the behavior of the wrapped invariants
    # We need an instance to pass to the invariants defined above
    instance = Sub()
    
    # Test Base invariant (v >= 0)
    # First invariant in list is from Base (due to how _all_dicts/store_invariants works)
    # The implementation of store_invariants iterates through [dct] + list(_all_dicts(bases))
    # So Sub's invariant comes first, then Base's.
    
    # Test Sub invariant: k > 0
    valid_sub, sub_err = Sub._checked_invariants[0](instance, 1, 10)
    assert valid_s_sub is True
    
    # Test Sub invariant failure
    invalid_sub, sub_err = Sub._checked_invariants[0](instance, -1, 10)
    assert invalid_sub is False
    assert "Non-positive key" in str(sub_err)

    # Test Base invariant: v >= 0
    valid_base, base_err = Sub._checked_invariants[1](instance, 1, 10)
    assert valid_base is True
    
    # Test Base invariant failure
    invalid_base, base_err = Sub._checked_invariants[1](instance, 1, -5)
    assert invalid_base is False
    assert "Negative Value" in str(base_err)

    # 3. Test Serializer: Check if the default serializer is correctly attached
    # The default serializer should handle CheckedType objects
    class MockCheckedType(CheckedType):
        def serialize(self, format=None):
            return "serialized"
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()

    mock_key = MockCheckedType()
    mock_val = MockCheckedType()
    
    # Test serialization of CheckedType objects in the map
    key_res, val_res = Sub.__serializer__(instance, mock_key, mock_val)
    assert key_res == "serialized"
    assert val_res == "serialized"

    # Test serialization of primitive objects
    key_res_prim, val_res_prim = Sub.__serializer__(instance, "key", 123)
    assert key_res_prim == "key"
    assert val_res_prim == 123

    # 4. Test Slots: Ensure __slots__ is set to empty tuple to prevent attribute injection
    assert Sub.__slots__ == ()

    # 5. Test Inheritance of Keys: Ensure __key_type__ from Base is visible if not overridden
    class OnlyBase(Base):
        pass
    
    assert OnlyBase._checked_key_types == (str,)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from pyrsistent import pset

class StringSet(CheckedPSet):
    __type__ = (str,)

class IntSet(CheckedPSet):
    __type__ = (int,)

class NestedCheckedSet(CheckedPSet):
    __type__ = (StringSet,)

def test_CheckedPSet_serialize():
    # Test basic serialization of a simple CheckedPSet
    simple_set = StringSet(["a", "b", "c"])
    assert simple_set.serialize() == {"a", "b", "c"}

    # Test serialization of a set with different types (if allowed)
    # Note: StringSet only allows str, so we use the base CheckedPSet
    mixed_set = CheckedPSet(["a", 1, 2])
    assert mixed_set.serialize() == {"a", 1, 2}

    # Test serialization of nested CheckedType objects
    # NestedCheckedSet contains StringSets. 
    # The serializer should call .serialize() on the inner StringSets.
    inner_set = StringSet(["inner1", "inner2"])
    outer_set = NestedCheckedSet([inner_set])
    
    # The result should be a set of lists (since StringSet.serialize returns a list)
    expected_output = {["inner1", "iter2"]} # Note: order in list might vary, but set handles it
    # Since serialize() on StringSet returns a list, the outer set contains that list
    result = outer_set.serialize()
    assert isinstance(result, set)
    assert ["inner1", "inner2"] in result

    # Test serialization with a custom serializer function
    # We can override the __serializer__ via a subclass or by monkeypatching
    class UpperCaseStringSet(StringSet):
        def __serializer__(self, format, value):
            if isinstance(value, str):
                return value.upper()
            return super(UpperCaseStringSet, self).__serializer__(format, value)

    upper_set = UpperCaseStringSet(["a", "b"])
    assert upper_set.serialize() == {"A", "B"}

    # Test that serialization handles non-CheckedType values by returning them as-is
    # (The default implementation of __serializer__ handles this)
    raw_set = CheckedPSet([1, 2])
    assert raw_set.serialize() == {1, 2}
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_CheckedPMap___new__():
    # Test case 1: Initialization with a dictionary (standard usage)
    initial_data = {1: "a", 2: "b"}
    m1 = CheckedPMap(initial_data)
    assert isinstance(m1, CheckedPMap)
    assert m1[1] == "a"
    assert m1[2] == "b"
    assert len(m1) == 2

    # Test case 2: Initialization with an explicit size
    # Note: The implementation calls super(CheckedPMap, cls).__new__(cls, size, initial)
    # which is the standard PMap constructor for pre-allocated size.
    m2 = CheckedPMask(initial_data, size=10)
    assert len(m2) == 2
    assert m2[1] == "a"

    # Test case 3: Initialization with an empty dictionary
    m3 = CheckedPMap({})
    assert len(m3) == 0

    # Test case 4: Verifying that it handles custom types via the Evolver pattern
    # We define a subclass to test the logic of the Evolver-based initialization
    class IntStringMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    m4 = IntStringMap({10: "ten"})
    assert m4[10] == "ten"
    assert isinstance(m4, IntStringMap)

    # Test case 5: Testing the 'size' parameter logic with the internal marker
    # The code uses _UNDEFINED_CHECKED_PMAP_SIZE to detect if size is passed.
    # We verify that passing a size doesn't crash and uses the underlying PMap logic.
    m5 = CheckedPMap(initial_data, size=5)
    assert m5[1] == "a"
    assert len(m5) == 2
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_CheckedPMap___new__():
    # Test Case 1: Initialization with a dictionary (standard usage)
    initial_data = {1: "a", 2: "b"}
    m1 = CheckedPMap(initial_data)
    assert isinstance(m1, CheckedPMap)
    assert m1[1] == "a"
    assert m1[2] == "b"
    assert len(m1) == 2

    # Test Case 2: Initialization with a specific size (advanced usage)
    # This triggers the branch: if size is not _UNDEFINED_CHECKED_PMAP_SIZE
    size = 10
    m2 = CheckedPMap(initial_data, size=size)
    assert isinstance(m2, CheckedPMap)
    assert len(m2) == 2
    # Note: The internal size of the PMap structure is set to 10, 
    # but the actual element count remains 2 based on initial_data.

    # Test Case 3: Initialization with empty dictionary
    m3 = CheckedPMapping = CheckedPMap({})
    assert len(m3) == 0

    # Test Case 4: Verification of type preservation via Evolver
    # Ensuring that the __new__ logic correctly handles the conversion 
    # from an evolver-based transient state to a persistent CheckedPMap.
    class IntToStr(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    m4 = IntToStr({10: "ten"})
    assert isinstance(m4, IntToStr)
    assert m4[10] == "ten"

    # Test Case 5: Checking that it handles the _UNDEFINED_CHECKED_PMAP_SIZE logic
    # via the default argument.
    m5 = CheckedPMap()
    assert isinstance(m5, CheckedPMap)
    assert len(m5) == 0
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_store_invariants():
    # Mock invariant functions
    def inv1(data):
        return True, []

    def inv2(data):
        return False, ["error1"]

    def inv3(data):
        # Simulate a function that returns multiple results
        return [(True, ()), (False, "error2")]

    # Case 1: Single class, single invariant
    class Base:
        pass

    class Sub(Base):
        check = inv1

    assert Sub.check == (wrap_invariant(inv1),)
    # Test the wrapped logic: inv1 returns (True, []) -> (True, [])
    assert Sub.check([]) == (True, [])

    # Case 2: Inheritance of invariants
    class Inherit(Base):
        pass

    class InheritWithInv(Base):
        check = inv1

    class InheritChild(InheritWithInv):
        pass

    # InheritChild should have inv1 from InheritWithInv
    assert InheritChild.check == (wrap_invariant(inv1),)

    # Case 3: Multiple invariants from hierarchy
    class MultiInv(Base):
        check = inv1

    class MultiInvChild(MultiInv):
        check = inv2

    # Should contain both wrapped inv1 and wrapped inv2
    # Note: order depends on dict traversal, but it should contain both
    assert len(MultiInvChild.check) == 2
    
    # Case 4: Testing the wrap_invariant logic for multiple results (inv3)
    class ComplexInv(Base):
        check = inv3

    # inv3 returns [(True, ()), (False, "error2")]
    # wrap_invariant should merge this into (False, ("error2",))
    assert ComplexInv.check[0](None) == (False, ("error2",))

    # Case 5: Error handling - Non-callable invariant
    class BadInvariant(Base):
        check = "not a callable"

    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants({}, (BadInvariant,), 'check', 'check')

    # Case 6: Verification of _all_dicts behavior via store_invariants
    class GrandParent:
        check = inv1

    class Parent(GrandParent):
        pass

    class Child(Parent):
        check = inv2

    # Child should have both inv1 (from GrandParent) and inv2 (from Child)
    assert len(Child.check) == 2
    # Check that both are present (order might vary but content is fixed)
    wrapped_invs = [wrap_invariant(inv1), wrap_invariant(inv2)]
    for wrapped in Child.check:
        assert wrapped in wrapped_invs
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test__CheckedTypeMeta___new__():
    # Mock invariant functions
    def inv1(obj):
        return True, "ok"

    def inv2(obj):
        return False, "error"

    # Define a base class with type and invariant
    class Base(metaclass=_CheckedTypeMeta):
        __type__ = int
        __invariant__ = inv1

    # Define a subclass that inherits/overrides
    class Sub(Base):
        __type__ = [str, int]
        __invariant__ = inv2

    # Test 1: Check if _checked_types is correctly parsed and inherited
    # Base should have [int]
    assert Base._checked_types == [int]
    # Sub should have [str, int] (from the iterable input)
    assert Sub._checked_types == [str, int]

    # Test 2: Check if _checked_invariants are collected and wrapped
    # Sub should have both inv1 and inv2 wrapped
    assert len(Sub._checked_invariants) == 2
    
    # Test the wrapped inv1 (returns bool directly)
    res1 = Sub._checked_invariants[0](Sub())
    assert res1 == (True, "ok")

    # Test the wrapped inv2 (returns bool directly)
    res2 = Sub._checked_invariants[1](Sub())
    assert res2 == (False, "error")

    # Test a mock invariant that returns multiple results to verify _merge_invariant_results logic via wrap_invariant
    def multi_inv(obj):
        return [(True, "a"), (False, "b"), (True, "c")]
    
    class Multi(metaclass=_CheckedTypeMeta):
        __invariant__ = multi_inv

    # The wrapped version should merge results: (False, ("b",))
    res_multi = Multi._checked_invariants[0](Multi())
    assert res_multi == (False, ("b",))

    # Test 3: Check __serializer__ default implementation
    class MockCheckedType(CheckedType):
        def serialize(self, format=None):
            return "serialized"

    class SerializerTest(metaclass=_CheckedTypeMeta):
        pass

    tester = SerializerTest()
    mock_obj = MockCheckedType()
    
    # The default serializer should call .serialize() on CheckedType objects
    assert tester.__serializer__(tester, "key", mock_obj) == "serialized"
    # The default serializer should return the value as-is for non-CheckedType objects
    assert tester.__serializer__(tester, "key", 123) == 123

    # Test 4: Check __slots__ and metadata
    assert Sub.__slots__ == ()
    assert hasattr(Sub, '_checked_types')
    assert hasattr(Sub, '_checked_invariants')

    # Test 5: Check TypeError for non-callable invariants
    with pytest.raises(TypeError, match='Invariants must be callable'):
        class BadInvariant(metaclass=_CheckedTypeMeta):
            __invariant__ = "not a callable"
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_get_type():
    # Test with a direct type object
    assert get_type(int) == int
    assert get_type(str) == str
    assert get_type(dict) == dict
    assert get_type(list) == list

    # Test with a string representing a built-in type
    # Note: __import__ is used in get_type, so we use 'builtins.int'
    assert get_type('builtins.int') == int
    assert get_type('builtins.str') == str
    assert get_type('builtins.list') == list

    # Test with a string representing a type from a standard module
    assert get_type('collections.abc.Iterable') == Iterable

    # Test that it raises ValueError if the string is not a valid module.class format
    with pytest.raises(ValueError):
        get_type('int')

    # Test that it raises ImportError if the module does not exist
    with pytest.raises(ImportError):
        get_type('non_existent_module.SomeClass')

    # Test that it raises AttributeError if the module exists but class does not
    with pytest.raises(AttributeError):
        get_type('builtins.NonExistentClass')
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test__CheckedMapTypeMeta___new__():
    # Define a dummy class using the metaclass to trigger __new__
    class MockKey:
        pass

    class MockValue:
        pass

    class CheckedDict(metaclass=_CheckedMapTypeMeta):
        __key_type__ = (MockKey,)
        __value_type__ = (MockValue,)
        __invariant__ = lambda self, k, v: (True, None)

    class InheritedCheckedDict(CheckedDict):
        __key_type__ = (int,)
        # __value_type__ should be inherited from CheckedDict
        # __invariant__ should be inherited from CheckedDict

    # 1. Verify key type storage (merging bases and current dict)
    # The metaclass should have combined types from the class and its bases
    assert CheckedDict._checked_key_types == (MockKey,)
    assert CheckedDict._checked_value_types == (MockValue,)
    
    # 2. Verify inheritance of types
    # Inherited class should have types from both itself and parent
    assert (MockKey,) in CheckedDict._checked_types or any(t == MockKey for t in CheckedDict._checked_key_types)
    assert int in InheritedCheckedDict._checked_key_types
    assert MockKey in InheritedCheckedDict._checked_key_types
    assert MockValue in InheritedCheckedDict._checked_value_types

    # 3. Verify invariant inheritance
    # The metaclass should aggregate invariants from the hierarchy
    assert len(CheckedDict._checked_invariants) == 1
    assert len(InheritedCheckedDict._checked_invariants) == 1
    
    # 4. Verify default serializer existence and functionality
    assert hasattr(CheckedDict, '__serializer__')
    
    # Test the default serializer logic
    # It should handle CheckedType serialization if applicable
    class MockCheckedType(CheckedType):
        def serialize(self, format=None):
            return "serialized"
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls()

    key_obj = MockCheckedType()
    val_obj = MockCheckedType()
    
    # The serializer is a method: (self, format, key, value)
    # We use a dummy instance for 'self'
    dummy_instance = CheckedDict()
    serializer = CheckedDict.__serializer__
    
    k_res, v_res = serializer(dummy_instance, None, key_obj, val_obj)
    assert k_res == "serialized"
    assert v_res == "serialized"

    # Test serializer with non-CheckedType objects
    k_res_raw, v_res_raw = serializer(dummy_instance, None, "plain_key", 123)
    assert k_res_raw == "plain_key"
    assert v_res_raw == 123

    # 5. Verify __slots__ is set to empty tuple to prevent dynamic attribute addition
    assert CheckedDict.__slots__ == ()
    assert InheritedCheckedDict.__slots__ == ()
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

class MockCheckedType:
    def __init__(self, value):
        self.value = value
    def serialize(self, format=None):
        return f"serialized_{self.value}"

class IntToMockMap(CheckedPMap):
    __key_type__ = int
    __value_type__ = object

def test_CheckedPMap_serialize():
    # Test case 1: Standard primitive types (no CheckedType involved)
    map1 = IntToMockMap({1: "value1", 2: "value2"})
    assert map1.serialize() == {"1": "value1", "2": "value2"} # Note: key becomes string if default serializer logic applies or stays int depending on implementation details of the provided __serializer__
    # Based on the provided code: 
    # sk = key; if isinstance(key, CheckedType): sk = key.serialize()
    # Since 1 is not CheckedType, sk = 1.
    assert map1.serialize() == {1: "value1", 2: "value2"}

    # Test case 2: Values are CheckedType instances
    val1 = MockCheckedType("A")
    val2 = MockCheckedType("B")
    map2 = IntToMockMap({10: val1, 20: val2})
    expected2 = {10: "serialized_A", 20: "serialized_B"}
    assert map2.serialize() == expected2

    # Test case 3: Keys are CheckedType instances
    # We need a class where keys are CheckedType
    class CheckedKeyMap(CheckedPMap):
        __key_type__ = object
        __value_type__ = object

    key1 = MockCheckedType(1)
    val_x = "x"
    map3 = CheckedKeyMap({key1: val_x})
    expected3 = {"serialized_1": "x"}
    assert map3.serialize() == expected3

    # Test case 4: Both keys and values are CheckedType instances
    map4 = CheckedKeyMap({key1: val1})
    expected4 = {"serialized_1": "serialized_A"}
    assert map4.serialize() == expected4

    # Test case 5: Empty map
    map5 = IntToMockMap({})
    assert map5.serialize() == {}
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from enum import Enum

def test_maybe_parse_user_type():
    # Test single type input
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test single string input
    assert maybe_parse_user_type("int") == ["int"]
    
    # Test preserved iterable type (Enum)
    class MyEnum(Enum):
        A = 1
    assert maybe_parse_user(MyEnum) == [MyEnum]
    
    # Test list of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    
    # Test nested iterables
    assert maybe_parse_user_type([[int], str]) == (int, str)
    assert maybe_parse_user_type(((int,), [str])) == (int, str)
    
    # Test tuple of types
    assert maybe_parse_user_type((float, bool)) == (float, bool)

    # Test invalid input (not type, not string, not iterable)
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    # Test edge case: empty iterable
    assert maybe_parse_user_type([]) == ()
    
    # Test complex nested structure
    complex_input = [int, [str, (float,)], "bool"]
    assert maybe_parse_user_type(complex_input) == (int, str, float, "bool")
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple (bool, data) tuple
    def simple_invariant(x):
        return (x > 0, "positive")
    
    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, ("positive",))
    assert wrapped_simple(-5) == (False, ("positive",))

    # Case 2: Invariant returns a single boolean
    def boolean_only_invariant(x):
        return x == 10
    
    wrapped_bool = wrap_invariant(boolean_only_invariant)
    assert wrapped_bool(10) == True
    assert wrapped_bool(5) == False

    # Case 3: Invariant returns an iterable of (bool, data) tuples (multiple tests)
    def multiple_tests_invariant(x):
        # Simulate multiple checks returning different results
        return (
            (x > 0, "is_positive"),
            (x < 10, "is_less_than_10"),
            (x % 2 == 0, "is_even")
        )
    
    wrapped_multi = wrap_invariant(multiple_tests_invariant)
    
    # All pass
    assert wrapped_multi(5) == (True, ()) # Note: 5 is not even, wait. 
    # Let's be precise with the logic: 5 > 0 (T), 5 < 10 (T), 5 % 2 == 0 (F)
    # Result should be (False, ("is_even",))
    assert wrapped_multi(5) == (False, ("is_even",))
    
    # Test all passing
    # 2: 2 > 0 (T), 2 < 10 (T), 2 % 2 == 0 (T)
    assert wrapped_multi(2) == (True, ())
    
    # Test multiple failing
    # -1: -1 > 0 (F), -1 < 10 (T), -1 % 2 == 0 (F)
    assert wrapped_multi(-1) == (False, ("is_positive", "is_even"))

    # Case 4: Invariant returns a result where the first element is a bool 
    # but it's wrapped in a list/tuple (the wrapper checks isinstance(result[0], bool))
    def wrapped_bool_tuple(x):
        return (True, "some_data")
    
    wrapped_wrapped = wrap_invariant(wrapped_bool_tuple)
    # Since result[0] is True (bool), it returns the tuple as is without merging
    assert wrapped_wrapped(1) == (True, "some_data")
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    def invariant_true(data):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict, errors = wrapped_true({"val": 1})
    assert verdict is True
    assert errors == []

    # Case 2: Invariant returns a simple boolean (False)
    def invariant_false(data):
        return False, ["error_msg"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, errors = wrapped_false({"val": 1})
    assert verdict is False
    assert errors == ["error_msg"]

    # Case 3: Invariant returns a list of (bool, data) tuples
    def invariant_multiple(data):
        return [
            (True, "ignored"),
            (False, "failure_1"),
            (True, "ignored_again"),
            (False, "failure_2")
        ]
    
    wrapped_multi = wrap_invariant(invariant_multiple)
    verdict, errors = wrapped_multi({"val": 1})
    assert verdict is False
    assert errors == ("failure_1", "failure_2")

    # Case 4: Invariant returns a list of all True tuples
    def invariant_all_true(data):
        return [(True, "ok"), (True, "ok_too")]
    
    wrapped_all_true = wrap_invariant(invariant_all_true)
    verdict, errors = wrapped_all_true({"val": 1})
    assert verdict is True
    assert errors == ()

    # Case 5: Invariant returns a list of all False tuples
    def invariant_all_false(data):
        return [(False, "err1"), (False, "err2")]
    
    wrapped_all_false = wrap_invariant(invariant_false) # Using logic from test 2
    # Re-defining for clarity in a single test function flow
    def invariant_all_false_real(data):
        return [(False, "err1"), (False, "err2")]
    
    wrapped_all_false_real = wrap_invariant(invariant_all_false_real)
    verdict, errors = wrapped_all_false_real({"val": 1})
    assert verdict is False
    assert errors == ("err1", "err2")
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple (bool, data) tuple
    def simple_invariant(val):
        return (val > 0, "positive")

    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(10) == (True, "positive")
    assert wrapped_simple(-1) == (False, "positive")

    # Case 2: Invariant returns a list of (bool, data) tuples (multiple results)
    def multi_invariant(val):
        return [
            (val > 0, "is_positive"),
            (val % 2 == 0, "is_even"),
            (val < 100, "is_small")
        ]

    wrapped_multi = wrap_invariant(multi_invariant)
    
    # All pass
    assert wrapped_multi(10) == (True, ())
    
    # One fails (not even)
    assert wrapped_multi(11) == (False, ("is_even",))
    
    # Two fail (not even, not small)
    assert wrapped_multi(102) == (False, ("is_even", "is_small"))
    
    # All fail
    assert wrapped_multi(-2) == (False, ("is_positive", "is_even", "is_small"))

    # Case 3: Invariant returns a tuple of (bool, data) tuples
    def tuple_invariant(val):
        return ((val > 0, "pos"), (val < 5, "small"))

    wrapped_tuple = wrap_invariant(tuple_invariant)
    assert wrapped_tuple(3) == (True, ())
    assert wrapped_tuple(6) == (False, ("small",))

    # Case 4: Verify it handles an empty list of results
    def empty_invariant(val):
        return []

    wrapped_empty = wrap_invariant(empty_invariant)
    # _merge_invariant_results on empty list returns (True, ())
    assert wrapped_empty(10) == (True, ())
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_CheckedPMap___new__():
    # Test case 1: Standard initialization with a dictionary
    initial_data = {1: "a", 2: "b"}
    mapping = CheckedPMap(initial_data)
    assert isinstance(mapping, CheckedPMap)
    assert mapping[1] == "a"
    assert mapping[2] == "b"
    assert len(mapping) == 2

    # Test case 2: Initialization with size specification
    # This triggers the branch: if size is not _UNDEFINED_CHECKED_PMAP_SIZE
    size_spec = 5
    mapping_with_size = CheckedPMap(initial_data, size=size_spec)
    assert isinstance(mapping_with_size, CheckedPMap)
    assert len(mapping_with_size) == 2
    # Note: The implementation passes size to the super constructor, 
    # which in PMap context handles the underlying structure.

    # Test case 3: Empty initialization
    empty_mapping = CheckedPMap({})
    assert len(empty_mapping) == 0

    # Test case 4: Initialization with an existing CheckedPMap instance
    # Testing the 'create' logic and the behavior of passing a CheckedPMap to __new__
    # Since CheckedPMap.create is used in the factory, we ensure stability.
    original = CheckedPMap({10: "ten"})
    copy_mapping = CheckedPMap(original)
    assert copy_mapping[10] == "ten"
    assert copy_mapping is not original  # Should be a new persistent instance

    # Test case 5: Verifying type constraints in __new__ via Evolver
    # We define a subclass with constraints to ensure __new__ respects the logic
    class IntKeyMap(CheckedPMap):
        __key_type__ = int
        __value_type__ = str

    # Valid initialization
    valid_map = IntKeyMap({1: "one"})
    assert valid_map[1] == "one"

    # Invalid initialization (should raise CheckedKeyTypeError via the Evolver used in __new__)
    with pytest.raises(CheckedKeyTypeError):
        IntKeyMap({"not_an_int": "value"})
```


