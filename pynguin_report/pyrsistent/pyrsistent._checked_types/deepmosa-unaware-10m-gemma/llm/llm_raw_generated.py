####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pyrsistent import pset

class MockCheckedType(CheckedPSet):
    __type__ = (int, str)
    
    @classmethod
    def create(cls, source_data, _factory_fields=None, ignore_extra=False):
        return cls(source_data)

    def serialize(self, format=None):
        # Custom serializer for testing: converts all elements to strings
        return set(str(v) for v in self)

def test_CheckedPSet_serialize():
    # Test Case 1: Standard serialization (default behavior via __serializer__)
    # Since CheckedPSet.serialize uses the class's __serializer__, 
    # and default_serializer returns value if not a CheckedType.
    base_set = CheckedPSet([1, "a", 2])
    assert base_set.serialize() == {1, "a", 2}

    # Test Case 2: Custom serialization implementation in a subclass
    # Testing the logic inside the overridden serialize method
    custom_set = MockCheckedType([10, "hello", True])
    # The custom serialize converts everything to string
    expected_output = {"10", "hello", "True"}
    assert custom_set.serialize() == expected_output

    # Test Case 3: Serialization with Nested CheckedTypes
    # Creating a set containing another CheckedType
    inner_set = CheckedPSet([1, 2])
    outer_set = CheckedPSet([inner_set, "data"])
    
    # The default __serializer__ calls .serialize() on CheckedType instances
    # result should be { {1, 2}, 'data' } (as a set of list/set depending on implementation)
    serialized = outer_set.serialize()
    assert [1, 2] in serialized or {1, 2} in serialized
    assert "data" in serialized

    # Test Case 4: Empty set serialization
    empty_set = CheckedPSet([])
    assert empty_set.serialize() == set()
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from enum import Enum

def test_maybe_parse_user_type():
    # Test single type (int)
    assert maybe_parse_user_type(int) == [int]

    # Test single string
    assert maybe_parse_user_type("str") == ["str"]

    # Test preserved iterable type (Enum)
    class MyEnum(Enum):
        A = 1
    assert maybe_parse_user_type(MyEnum) == [MyEnum]

    # Test list of types
    assert maybe_parse_user_type([int, str]) == (int, str)

    # Test nested iterable of types
    assert maybe_parse_user_type([[int], str]) == (int, str)

    # Test tuple of types
    assert maybe_parse_user_type((float, bool)) == (float, bool)

    # Test deeply nested structures
    assert maybe_parse_user_type([ [ [int] ], "str" ]) == (int, "str")

    # Test invalid input (non-type, non-string, non-iterable)
    # Note: 1 is an int, but not a type or string. However, 1 is not iterable.
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(1)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    # Test invalid input (None)
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)

    # Test mixed valid types in list
    assert maybe_parse_user_type([int, MyEnum, "float"]) == (int, MyEnum, "float")
```


# LLM-generated content at query #3
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
    
    # Test preserved type (Enum)
    class MyEnum(Enum):
        A = 1
    assert maybe_parse_user_type(MyEnum) == [MyEnum]
    
    # Test list of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    
    # Test nested iterable/list
    assert maybe_parse_user_type([[int], str]) == (int, str)
    
    # Test tuple of types
    assert maybe_parse_user_type((float, list)) == (float, list)
    
    # Test deep nesting
    assert maybe_parse_user_type([ [ [int] ], "str" ]) == (int, "str")

    # Test invalid input (non-type, non-string, non-iterable object)
    class NotIterable:
        pass
    
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(NotIterable())
    assert "Type specifications must be types or strings" in str(excinfo.value)

    # Test error case for non-serializable/invalid formatting in recursion
    # Using an object that fails when passed to format/string conversion if needed,
    # but the current implementation relies on TypeError from the logic itself.
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    def invariant_true(x):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict, data = wrapped_true(1)
    assert verdict is True
    assert data == []

    # Case 2: Invariant returns a simple boolean (False)
    def invariant_false(x):
        return False, ["error"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, data = wrapped_false(1)
    assert verdict is False
    assert data == ["error"]

    # Case 3: Invariant returns a tuple of (bool, data) pairs
    def invariant_multi(x):
        return (
            (True, None),
            (False, "fail_1"),
            (True, "ignored"),
            (False, "fail_2")
        )
    
    wrapped_multi = wrap_invariant(invariant_multi)
    verdict, data = wrapped_multi(1)
    assert verdict is False
    assert data == ("fail_1", "fail_2")

    # Case 4: Invariant returns a tuple of (bool, data) pairs where all are True
    def invariant_all_true(x):
        return (
            (True, "val1"),
            (True, "val2")
        )
    
    wrapped_all_true = wrap_invariant(invariant_all_true)
    verdict, data = wrapped_all_true(1)
    assert verdict is True
    assert data == ()

    # Case 5: Invariant returns a single (bool, data) tuple (not a list of results)
    def invariant_single_tuple(x):
        return (True, "success")
    
    wrapped_single = wrap_invariant(invariant_single_tuple)
    verdict, data = wrapped_single(1)
    assert verdict is True
    assert data == () # Result of _merge_invariant_results on a single-item loop where verd is True
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple (bool, data) tuple
    @wrap_invariant(lambda x: (x > 0, "positive"))
    def simple_invariant(x):
        pass

    assert simple_invariant(5) == (True, "positive")
    assert simple_invariant(-1) == (False, "positive")

    # Case 2: Invariant returns a list of results [(bool, data), ...]
    @wrap_import_helper
    def multi_result_invariant(x):
        return [
            (x > 0, "is_pos"),
            (x < 10, "is_small"),
            (x % 2 == 0, "is_even")
        ]

    # All pass
    assert multi_result_invariant(4) == (True, ())
    
    # One fails (not small)
    assert multi_result_invariant(12) == (False, ("is_small",))
    
    # Two fail (not small and not even)
    assert multi_result_invariant(11) == (False, ("is_small", "is_even"))
    
    # All fail
    assert multi_result_invariant(-2) == (False, ("is_pos", "is_small", "is_even"))

    # Case 3: Invariant returns a single boolean (should be returned as is)
    @wrap_invariant(lambda x: x == "magic")
    def bool_only_invariant(x):
        pass

    assert bool_only_invariant("magic") is True
    assert bool_only_invariant("not magic") is False

# Helper to allow the test logic above to function since wrap_invariant 
# implementation relies on a specific structure in the provided code.
def wrap_import_helper(func):
    return wrap_invariant(func)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple (bool, data) tuple
    simple_invariant = lambda x: (x > 0, "positive")
    wrapped_simple = wrap_invariant(simple_invariant)
    assert wrapped_simple(5) == (True, "positive")
    assert wrapped_simple(-5) == (False, "") # Note: data is 'positive' in logic but depends on implementation of simple_invariant return value. 
                                            # Here we test the Boolean part.

    # Case 2: Invariant returns a list of (bool, data) tuples (multiple results)
    def multi_result_invariant(x):
        return [
            (x > 0, "is_positive"),
            (x % 2 == 0, "is_even")
        ]
    
    wrapped_multi = wrap_invariant(multi_result_invariant)
    
    # Test all pass
    assert wrapped_multi(2) == (True, ())
    
    # Test one fails
    assert wrapped_multi(1) == (False, ("is_even",))
    
    # Test both fail
    assert wrapped_multi(-1) == (False, ("is_positive", "is_even"))

    # Case 3: Invariant returns a single boolean (no tuple/iterable result)
    # The wrapper checks isinstance(result[0], bool). If it is, it returns the result directly.
    single_bool_invariant = lambda x: x == True
    wrapped_bool = wrap_invariant(single_bool_invariant)
    assert wrapped_bool(True) is True
    assert wrapped_bool(False) is False

    # Case 4: Invariant returns a tuple where the first element is not a bool (e.g., nested structure)
    # The wrapper will attempt to iterate and merge.
    def complex_invariant(x):
        return [
            (True, "ok"),
            (False, "error_msg")
        ]
    wrapped_complex = wrap_invariant(complex_invariant)
    assert wrapped_complex(None) == (False, ("error_msg",))

    # Case 5: Testing the identity behavior for simple boolean return
    def bool_only(x):
        return True
    assert wrap_invariant(bool_only)(None) is True
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    def invariant_true(data):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict, errors = wrapped_true({"x": 1})
    assert verdict is True
    assert errors == []

    # Case 2: Invariant returns a simple boolean (False)
    def invariant_false(data):
        return False, ["error"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, errors = wrapped_false({"x": 1})
    assert verdict is False
    assert errors == ["error"]

    # Case 3: Invariant returns a list of results (merging required)
    def invariant_multi_result(data):
        # Format: [(bool, error_msg), ...]
        return [
            (True, None),
            (False, "first_fail"),
            (True, None),
            (False, "second_fail")
        ]

    wrapped_multi = wrap_invariant(invariant_multi_result)
    verdict, errors = wrapped_multi({"x": 1})
    assert verdict is False
    assert errors == ("first_fail", "second_fail")

    # Case 4: Invariant returns a list of all True results
    def invariant_all_true(data):
        return [
            (True, None),
            (True, "ignored_msg")
        ]

    wrapped_all_true = wrap_invariant(invariant_all_true)
    verdict, errors = wrapped_all_true({"x": 1})
    assert verdict is True
    assert errors == ()

    # Case 5: Invariant returns a single result tuple (not a list of results)
    def invariant_single_tuple(data):
        return False, "single_error"

    wrapped_single = wrap_invariant(invariant_single_tuple)
    verdict, errors = wrapped_single({"x": 1})
    assert verdict is False
    assert errors == ("single_error",)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

class MockEnum(Enum):
    A = 1

class SimpleType:
    pass

def test_maybe_parse_user_type():
    # Test single type
    assert maybe_parse_user_type(SimpleType) == [SimpleType]
    
    # Test single string
    assert maybe_parse_user_type("int") == ["int"]
    
    # Test preserved iterable (Enum)
    assert maybe_parse_user_type(MockEnum) == [MockEnum]
    
    # Test list of types
    assert maybe_parse_user_type([SimpleType, "str"]) == (SimpleType, "str")
    
    # Test nested iterables (tuple of lists)
    assert maybe_parse_user_type(([SimpleType], ["str"])) == (SimpleType, "str")
    
    # Test single element tuple
    assert maybe_parse_user_type((SimpleType,)) == (SimpleType,)

    # Test invalid input (non-iterable, non-string, non-type)
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    # Test deeply nested structures
    nested = [[SimpleType], [["str"], SimpleType]]
    assert maybe_parse_user_type(nested) == (SimpleType, "str", SimpleType)

    # Test empty iterable
    assert maybe_parse_user_type([]) == ()
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Test case 1: Invariant returns a simple boolean (True)
    def invariant_true(x):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict, data = wrapped_true(10)
    assert verdict is True
    assert data == []

    # Test case 2: Invariant returns a simple boolean (False)
    def invariant_false(x):
        return False, ["error"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, data = wrapped_false(10)
    assert verdict is False
    assert data == ["error"]

    # Test case 3: Invariant returns a list of results (Multiple checks)
    # Result format: [(bool, error_msg), (bool, error_msg), ...]
    def invariant_multiple(x):
        return [
            (True, None),
            (False, "first_error"),
            (True, None),
            (False, "second_error")
        ]
    
    wrapped_multi = wrap_invariant(invariant_multiple)
    verdict, data = wrapped_multi(10)
    assert verdict is False
    assert data == ("first_error", "second_error")

    # Test case 4: Invariant returns a list of results (All True)
    def invariant_all_true(x):
        return [
            (True, None),
            (True, "not an error")
        ]
    
    wrapped_all_true = wrap_invariant(invariant_all_true)
    verdict, data = wrapped_all_true(10)
    assert verdict is True
    assert data == ()

    # Test case 5: Invariant returns a list of results (Empty list)
    def invariant_empty(x):
        return []
    
    wrapped_empty = wrap_invariant(invariant_empty)
    verdict, data = wrapped_empty(10)
    assert verdict is True
    assert data == ()

    # Test case 6: Invariant returns a single tuple (Boolean style as input)
    # The wrapper checks if result[0] is bool. If so, it returns the result directly.
    def invariant_direct(x):
        return False, ["direct_error"]
    
    wrapped_direct = wrap_with_logic = wrap_invariant(invariant_direct)
    verdict, data = wrapped_direct(10)
    assert verdict is False
    assert data == ["direct_error"]
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_store_invariants():
    # Test Case 1: Basic functionality - storing a single invariant in a class
    class Base:
        pass

    class Derived(Base):
        def my_invariant(self, x):
            return x > 0
        my_invariant = store_invariants(lambda d, b, dest, src: None, (Base,), 'target', 'source')
        # Note: The decorator logic in the provided code is actually used via metaclass-like patterns.
        # Since we are testing the function directly, we simulate its behavior on a dict.

    def mock_invariant(val):
        return val > 0

    dct = {}
    bases = (object,)
    # We need to test the logic of the function provided in the snippet
    # which takes dct, bases, destination_name, source_name
    
    # Case: Single invariant present
    dct['src'] = mock_invariant
    store_invariants(dct, (object,), 'dest', 'src')
    assert 'dest' in dct
    assert len(dct['dest']) == 1
    assert callable(dct['dest'][0])

    # Case: Inheritance of invariants
    class Parent:
        def inv_p(self): return True
    
    class Child(Parent):
        def inv_c(self): return True

    dct_child = {}
    # Simulate the process of store_invariants being called on Child's dict
    # with Parent as a base. 
    # We manually provide the source attribute in the dicts to simulate what the function looks for.
    class MockParent:
        def p_inv(self): return True

    class MockChild(MockParent):
        def c_inv(self): return True

    # The function expects 'source_name' to be present in dct or bases.__dict__
    dct_test = {'c_inv': MockChild.c_inv}
    bases_test = (MockParent,)
    store_invariants(dct_test, bases_test, 'all_invs', 'p_inv') # This won't work as expected because p_inv is in Parent.__dict__
    
    # Let's re-implement the logic test properly:
    # The function iterates over [dct] + _all_dicts(bases) and looks for source_name.
    
    class Alpha:
        def alpha_check(self): return True

    class Beta(Alpha):
        def beta_check(self): return True

    target_dict = {'beta_check': Beta.beta_check}
    # We use the actual class structure to ensure _all_dicts works
    store_invariants(target_dict, (Alpha,), 'collected', 'alpha_check')
    
    assert 'collected' in target_dict
    assert len(target_dict['collected']) == 1
    assert target_dict['collected'][0](None) is True

    # Case: Multiple invariants from different levels
    class Gamma:
        def gamma_inv(self): return True
    
    class Delta(Gamma):
        def delta_inv(self): return True

    target_dict_2 = {'delta_inv': Delta.delta_inv}
    store_invariants(target_dict_2, (Gamma,), 'merged', 'gamma_inv')
    # Note: store_invariants looks for source_name in the dicts. 
    # If we look for 'gamma_inv', it finds it in Gamma's dict.
    assert len(target_dict_2['merged']) == 1

    # Case: Error handling - non-callable invariant
    target_dict_err = {'bad_inv': "not a callable"}
    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants(target_dict_err, (object,), 'dest', 'bad_inv')

    # Case: Verification of wrap_invariant logic via store_invariants
    # If an invariant returns a tuple (bool, data), wrap_invariant should flatten it.
    def multi_result_invariant(x):
        return (False, "error1"), (True, "ignored")

    target_dict_multi = {'multi': multi_result_invariant}
    store_invariants(target_dict_multi, (object,), 'wrapped', 'multi')
    
    # The wrapped function should return (False, ('error1',)) 
    # because _merge_invariant_results processes the result of the inner call.
    result = target_dict_multi['wrapped']('test')
    assert result == (False, ('error1',))

    # Case: Empty search
    target_dict_empty = {}
    store_invariants(target_dict_empty, (object,), 'dest', 'non_existent')
    assert 'dest' not in target_dict_empty
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Test Case 1: Invariant returns a simple boolean (True)
    def invariant_true(data):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict, errors = wrapped_true({"key": "val"})
    assert verdict is True
    assert errors == []

    # Test Case 2: Invariant returns a simple boolean (False)
    def invariant_false(data):
        return False, ["error"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, errors = wrapped_false({"key": "val"})
    assert verdict is False
    assert errors == ["error"]

    # Test Case 3: Invariant returns a list of (bool, error_data) tuples
    def invariant_complex(data):
        return [
            (True, None),
            (False, "error_1"),
            (True, "ignored_error"),
            (False, "error_2")
        ]
    
    wrapped_complex = wrap_invariant(invariant_complex)
    verdict, errors = wrapped_complex({"key": "val"})
    assert verdict is False
    # Should only contain error messages from failed tests
    assert errors == ("error_1", "error_2")

    # Test Case 4: Invariant returns all True results in a list
    def invariant_all_true(data):
        return [
            (True, None),
            (True, "not_an_error")
        ]
    
    wrapped_all_true = wrap_invariant(invariant_all_true)
    verdict, errors = wrapped_all_true({"key": "val"})
    assert verdict is True
    assert errors == ()

    # Test Case 5: Invariant returns an empty list of results
    def invariant_empty(data):
        return []
    
    wrapped_empty = wrap_invariant(invariant_empty)
    verdict, errors = wrapped_empty({"key": "val"})
    assert verdict is True
    assert errors == ()
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_store_invariants():
    # Test 1: Basic functionality - storing a single invariant
    class Base:
        def my_invariant(self, x):
            return x > 0

    class Derived(Base):
        pass

    # Check if the decorator/function correctly populates the destination_name
    # We simulate the metaclass-like behavior manually for testing
    dct = {}
    bases = (Base,)
    store_invariants(dct, bases, 'invariants', 'my_invariants')
    
    # Note: store_invariants looks for source_name in dct and bases
    # In our setup, we need to ensure the source_name exists in the dictionaries being scanned.
    # Let's redefine the test case more precisely.

    class MockBase:
        def check_val(self, x):
            return x == 1

    class MockDerived(MockBase):
        pass

    # We need to manually trigger the logic as it would be in a metaclass
    # The function looks for source_name in dct and bases.__dict__
    
    # Case A: Single invariant in base class
    dct_a = {}
    store_invariants(dct_a, (MockBase,), 'invariants', 'check_val')
    assert 'invariants' in dct_a
    assert len(dct_a['invariants']) == 1
    # The wrapped function should return a boolean or tuple
    res = dct_a['invariants'][0](1)
    assert res is True
    res_fail = dct_a['invariants'][0](2)
    assert res_fail is False

    # Case B: Inheritance of invariants
    class MockGrandParent:
        def inv1(self, x): return True
        def inv2(self, x): return x > 0

    class MockParent(MockGrandParent):
        def inv2(self, x): return x < 10

    dct_b = {}
    store_invariants(dct_b, (MockParent,), 'all_invs', 'inv2')
    # It should find inv2 from Parent and inv2 from GrandParent is overridden? 
    # No, the code iterates through all dicts and appends.
    # Actually, it collects ALL instances of source_name found in the hierarchy.
    assert len(dct_b['all_invs']) == 1 # Only 'inv2' was searched for

    # Case C: Multiple different invariants being collected via a single source name search is not what this does.
    # The function searches for one specific `source_name` across the hierarchy.
    
    class MultiInv:
        def check(self, x): return x > 0
        def check(self, x): return x < 5 # This is just a single attr in dict

    dct_c = {}
    # If we have multiple classes that all define 'check'
    class Level1:
        def check(self, x): return True
    class Level2(Level1):
        def check(self, x): return False

    store_invariants(dct_c, (Level2,), 'collected', 'check')
    # It should find 'check' in Level2 and 'check' in Level1
    assert len(dct_c['collected']) == 2

    # Case D: TypeError when invariant is not callable
    class BadBase:
        check = "not a function"

    dct_d = {}
    with pytest.raises(TypeError, match="Invariants must be callable"):
        store_invariants(dct_d, (BadBase,), 'invariants', 'check')

    # Case E: Check if wrap_invariant works for functions returning tuples
    def multi_result_inv(x):
        return [(True, "ok"), (False, "fail")]

    class MultiResultClass:
        def check(self, x):
            return multi_result_inv(x)

    dct_e = {}
    store_invariants(dct_e, (MultiResultClass,), 'wrapped', 'check')
    # The wrapped function should return the merged result: (False, ("fail",))
    verdict, errors = dct_e['wrapped'][0](1)
    assert verdict is False
    assert errors == ("fail",)

    # Case F: Check if wrap_invariant works for functions returning simple bools
    def simple_bool_inv(x):
        return x > 0

    class SimpleClass:
        def check(self, x):
            return simple_bool_inv(x)

    dct_f = {}
    store_invariants(dct_f, (SimpleClass,), 'wrapped', 'check')
    # Should return the bool directly
    assert dct_f['wrapped'][0](5) is True
    assert dct_f['wrapped'][0](-5) is False
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    @wrap_invariant(lambda x: (True, "all good"))
    def invariant_bool_true(x):
        return True, "all good"

    assert invariant_bool_true(None) == (True, "all good")

    # Case 2: Invariant returns a simple boolean (False)
    @wrap_invariant(lambda x: (False, "failed"))
    def invariant_bool_false(x):
        return False, "failed"

    assert invariant_bool_false(None) == (False, "failed")

    # Case 3: Invariant returns a list of results to be merged
    # Result format: [(verdict, data), (verdict, data), ...]
    @wrap_invariant(lambda x: [
        (True, "pass1"),
        (False, "fail1"),
        (True, "pass2"),
        (False, "fail2")
    ])
    def invariant_multiple_results(x):
        return [(True, "pass1"), (False, "fail1"), (True, "pass2"), (False, "fail2")]

    # Expected: verdict is False because at least one failed. 
    # Data should only contain the 'data' from failed results.
    assert invariant_multiple_results(None) == (False, ("fail1", "fail2"))

    # Case 4: All items in the list are True
    @wrap_invariant(lambda x: [(True, "p1"), (True, "p2")])
    def invariant_all_true(x):
        return [(True, "p1"), (True, "p2")]

    assert invariant_all_true(None) == (True, ())

    # Case 5: Single item in list that is True
    @wrap_invariant(lambda x: [(True, "single")])
    def invariant_single_true(x):
        return [(True, "single")]

    assert invariant_single_true(None) == (True, ())

    # Case 6: Single item in list that is False
    @wrap_invariant(lambda x: [(False, "only_fail")])
    def invariant_single_false(x):
        return [(False, "only_fail")]

    assert invariant_single_false(None) == (False, ("only_fail",))
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_store_invariants():
    # Test case 1: Basic functionality - storing a single invariant
    class Base:
        def my_invariant(self, x):
            return x > 0

    class Child(Base):
        pass

    # We need to simulate the decorator behavior on a class dictionary
    # Since store_invariants modifies dct in place
    class DummyClass:
        pass

    # Mocking the logic of storing an invariant named 'check' from source 'check'
    # into destination 'validators'
    
    # Test 1: Single level, valid callable
    class Simple:
        def validator(self, val):
            return True
    
    dct = {}
    bases = (Simple,)
    store_invars = [lambda x: True] # Mocking what the function finds in ns[source_name]
    
    # Because we can't easily intercept the internal dict lookup of store_invariants 
    # without actual classes, we use real class definitions.

    class InvariantSource:
        def check(self, x):
            return x == 1

    class Target(InvariantSource):
        pass

    # We call the function on Target's dict manually as if it were a decorator
    store_invariants(Target.__dict__, (InvariantSource,), 'validators', 'check')
    
    assert 'validators' in Target.__dict__
    assert len(Target.validators) == 1
    # The wrapped invariant should return (True, ()) or (False, (...))
    # Since check returns True/False directly via wrap_invariant logic if it doesn't return a tuple
    assert Target.validators[0](1) is True
    assert Target.validators[0](2) is False

    # Test 2: Inheritance of invariants
    class GrandParent:
        def primary(self, x):
            return True
        def secondary(self, x):
            return False

    class Parent(GrandParent):
        def tertiary(self, x):
            return True

    class Final(Parent):
        pass

    store_invariants(Final.__dict__, (Parent,), 'all_checks', 'primary') 
    # Note: store_invariants searches for source_name in the dicts. 
    # If we look for 'primary' in Final, it finds it in GrandParent via _all_dicts.

    # Resetting logic to test inheritance specifically
    class Root:
        def inc(self, x): return True
    class Sub(Root):
        def inc(self, x): return False # This will override/shadow but store_invariants collects all
        def extra(self, x): return True

    # Let's test the accumulation logic
    class Accumulator:
        pass

    # We use a manual approach to trigger the 'all_dicts' scanning
    # by defining classes that have the attributes.
    class SourceA:
        def rule1(self, x): return True
    class SourceB(SourceA):
        def rule2(self, x): return False

    store_invariants(Accumulator.__dict__, (SourceB,), 'rules', 'rule1')
    # It should find rule1 in SourceA and rule1 in SourceB if they were present.
    # But store_invariants looks for source_name specifically.
    
    # Corrected Test 2: Verify multiple invariants are collected from the hierarchy
    class Provable:
        def check_a(self, x): return True
    class Derived(Provable):
        def check_b(self, x): return False

    class Testing(Derived):
        pass

    # We want to find 'check_a' in the hierarchy and put it in 'checks'
    store_invariants(Testing.__dict__, (Derived,), 'checks', 'check_a')
    assert len(Testing.checks) == 1 # Only check_a is being searched for via source_name='check_a'

    # Test 3: Error handling - Non-callable invariant
    class BadSource:
        not_a_callable = "I am a string"

    with pytest.raises(TypeError, match="Invariants must be callable"):
        store_invariants(Accumulator.__dict__, (BadSource,), 'error_test', 'not_a_callable')

    # Test 4: Verify wrap_invariant logic for multiple return values (the tuple case)
    class MultiReturnSource:
        def multi(self, x):
            return [(True, "ok"), (False, "fail")]

    class MultiTarget(MultiReturnSource):
        pass

    store_invariants(MultiTarget.__dict__, (MultiReturnSource,), 'multi_wrapped', 'multi')
    # wrap_invariant should merge the list of tuples into (verdict, data)
    result = MultiTarget.multi_wrapped[0](None)
    assert result == (False, ("fail",))

    # Test 5: Verify it handles empty/missing source names gracefully
    class NoSource:
        pass
    
    store_invariants(Accumulator.__dict__, (NoSource,), 'empty', 'non_existent')
    assert 'empty' in Accumulator.__dict__
    assert Accumulator.empty == []
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

class MockEnum(Enum):
    A = 1

class MockClass:
    pass

def test_maybe_parse_user_type():
    # Test single type input
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(MockClass) == [MockClass]
    
    # Test string input
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("") == [""]

    # Test preserved iterable types (Enum)
    assert maybe_parse_user_type(MockEnum) == [MockEnum]
    
    # Test list/tuple of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    assert maybe_parse_user_type((float, MockClass)) == (float, MockClass)
    
    # Test nested iterables
    assert maybe_parse_user_type([[int], str]) == (int, str)
    assert maybe_parse_user_type([(MockEnum,), "int"]) == (MockEnum, "int")

    # Test error case: non-type/non-string/non-iterable input
    # Note: 1 is an int, but we want to test something that isn't a type or string.
    # Since we can't easily pass an object that is not iterable, type, or str without 
    # more complex mocking, we test the behavior with an object that should fail logic.
    with pytest.raises(TypeError):
        # Using an object that doesn't satisfy any condition (though most objects are 
        # technically part of some hierarchy, in Python almost everything is an instance)
        # We use a custom object that specifically fails the checks.
        class BadInput:
            def __iter__(self):
                raise TypeError("Iterating failed")
        
        maybe_parse_user_type(BadInput())

    # Test complex nested structure
    complex_input = [MockEnum, [int, "str"], (float,)]
    assert maybe_parse_user_type(complex_input) == (MockEnum, int, "str", float)
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

class MockEnum(Enum):
    A = 1

class SimpleClass:
    pass

def test_maybe_parse_user_type():
    # Test single type input
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    assert maybe_parse_user_type(SimpleClass) == [SimpleClass]

    # Test preserved types (Enums should return as a list containing the type, not members)
    assert maybe_parse_user_type(MockEnum) == [MockEnum]

    # Test string input
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type(["str", int]) == ("str", int)

    # Test nested iterables (tuple/list of types)
    assert maybe_parse_user_type((int, float)) == (int, float)
    assert maybe_parse_user_type([str, [int, float]]) == (str, int, float)
    assert maybe_parse_user_type([[MockEnum], "Type"]) == (MockEnum, "Type")

    # Test invalid input
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    with pytest.raises(TypeError):
        maybe_parse_user_type(None)

    # Test complex nested structure
    complex_input = (int, [str, (float, MockEnum)])
    expected_output = (int, str, float, MockEnum)
    assert maybe_parse_user_type(complex_input) == expected_output
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
    verdict, errors = wrapped_true({"key": "val"})
    assert verdict is True
    assert errors == []

    # Case 2: Invariant returns a simple boolean (False)
    def invariant_false(data):
        return False, ["error_msg"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, errors = wrapped_false({"key": "val"})
    assert verdict is False
    assert errors == ["error_msg"]

    # Case 3: Invariant returns a list of (bool, data) tuples (Multiple tests)
    def invariant_multi(data):
        return [
            (True, "pass1"),
            (False, "fail1"),
            (True, "pass2"),
            (False, "fail2")
        ]
    
    wrapped_multi = wrap_invariant(invariant_multi)
    verdict, errors = wrapped_multi({"key": "val"})
    assert verdict is False
    assert errors == ("fail1", "fail2")

    # Case 4: Invariant returns a list of (bool, data) tuples (All passing)
    def invariant_all_pass(data):
        return [
            (True, "pass1"),
            (True, "pass2")
        ]
    
    wrapped_all_pass = wrap_invariant(invariant_all_pass)
    verdict, errors = wrapped_all_pass({"key": "val"})
    assert verdict is True
    assert errors == ()

    # Case 5: Invariant returns a list of (bool, data) tuples (All failing)
    def invariant_all_fail(data):
        return [
            (False, "err1"),
            (False, "err2")
        ]
    
    wrapped_all_fail = wrap_invariant(invariant_all_fail)
    verdict, errors = wrapped_all_fail({"key": "val"})
    assert verdict is False
    assert errors == ("err1", "err2")
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from enum import Enum

class MockEnum(Enum):
    A = 1

class SimpleClass:
    pass

def test_maybe_parse_user_type():
    # Test single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(SimpleClass) == [SimpleClass]

    # Test string input
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("") == [""]

    # Test preserved iterable types (Enums should be wrapped in a list, not exploded)
    assert maybe_parse_user_type(MockEnum) == [MockEnum]

    # Test flat iterable of types/strings
    assert maybe_parse_user_type([int, str, SimpleClass]) == (int, str, SimpleClass)
    assert maybe_parse_user_type((float, "bool")) == (float, "bool")

    # Test nested iterables
    assert maybe_parse_user_type([[int], [str, [float]]]) == (int, str, float)
    assert maybe_parse_user_type([MockEnum, [int]]) == (MockEnum, int)

    # Test invalid input types
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    with pytest.raises(TypeError):
        maybe_parse_user_type(None)

    # Test complex nested structure with mixed valid types
    complex_input = [int, [str, (MockEnum, ["float"])], "list_of_types"]
    expected_output = (int, str, MockEnum, float, "list_of_types")
    assert maybe_parse_user_type(complex_input) == expected_output
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

class MockEnum(Enum):
    A = 1

class MockClass:
    pass

def test_maybe_parse_user_type():
    # Test single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(MockClass) == [MockClass]

    # Test string representation
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("MyCustomType") == ["MyCustomType"]

    # Test preserved iterable types (Enum)
    assert maybe_parse_user_type(MockEnum) == [MockEnum]
    assert maybe_parse_user_type((MockEnum, int)) == (MockEnum, int)

    # Test nested iterables
    assert maybe_parse_user_type([int, str]) == (int, str)
    assert maybe_parse_user_type([[int], str]) == (int, str)
    assert maybe_parse_user_type((int, [str, float])) == (int, str, float)

    # Test complex nested structure with strings and types
    complex_input = ([int, "string"], MockEnum)
    expected_output = (int, "string", MockEnum)
    assert maybe_parse_user_type(complex_input) == expected_output

    # Test error case: non-type/non-string/non-iterable input
    # Note: 123 is an int, but we want to test something that isn't a type or string.
    # However, the function checks if it's an instance of type, str, or Iterable.
    # Most basic objects are not iterable, so they hit the TypeError.
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(None) # None is not a type/str/iterable in this logic context
    assert "Type specifications must be types or strings" in str(excinfo.value)

    # Test error case with an object that cannot be formatted (as per docstring)
    class Unformattable:
        def __str__(self):
            raise ValueError("Cannot format")
    
    with pytest.raises(ValueError, match="Cannot format"):
        maybe_parse_user_type([Unformattable])
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Test case 1: Invariant returns a simple boolean (True)
    def invariant_true(data):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict_t, errors_t = wrapped_true({"x": 1})
    assert verdict_t is True
    assert errors_t == []

    # Test case 2: Invariant returns a simple boolean (False)
    def invariant_false(data):
        return False, ["error message"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict_f, errors_f = wrapped_false({"x": 1})
    assert verdict_f is False
    assert errors_f == ["error message"]

    # Test case 3: Invariant returns a tuple of results (Multiple tests)
    # Result format: [(bool, data), (bool, data), ...]
    def invariant_multi(data):
        return [
            (True, "all good"),
            (False, "failed test 1"),
            (True, "another good one"),
            (False, "failed test 2")
        ]

    wrapped_multi = wrap_invariant(invariant_multi)
    verdict_m, errors_m = wrapped_multi({"x": 1})
    
    # The merged verdict should be False because at least one failed
    assert verdict_m is False
    # Only the data from the failed tests should be collected
    assert errors_m == ("failed test 1", "failed test 2")

    # Test case 4: Invariant returns a tuple of results (All passed)
    def invariant_all_pass(data):
        return [
            (True, "ok"),
            (True, "fine")
        ]

    wrapped_all_pass = wrap_invariant(invariant_all_pass)
    verdict_p, errors_p = wrapped_all_pass({"x": 1})
    assert verdict_p is True
    assert errors_p == ()

    # Test case 5: Invariant returns a tuple of results (Single pass)
    def invariant_single_pass(data):
        return [(True, "only one")]

    wrapped_single = wrap_invariant(invariant_single_pass)
    verdict_s, errors_s = wrapped_single({"x": 1})
    assert verdict_s is True
    assert errors_s == ()
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_store_invariants():
    # Mock invariant functions
    def inv1(data):
        return True, []

    def inv2(data):
        return False, ["error1"]

    # Case 1: Basic functionality - storing a single invariant in a new class
    class BaseClass:
        pass

    class DerivedClass(BaseClass):
        my_invariant = inv1

    assert DerivedClass.stored_inv == (wrap_invariant(inv1),)

    # Case 2: Inheritance - invariants from base classes should be collected
    class Parent:
        parent_inv = inv1

    class Child(Parent):
        child_inv = inv2

    # The function gathers all invariants from the class and its bases
    # It wraps them with wrap_invariant logic
    assert len(Child.stored_invariants) == 2
    # Check that both invariants are present in the tuple
    results = Child.stored_invariants
    # We check if they return the expected boolean results when called
    assert all(res(None)[0] for res in results)

    # Case 3: Type Error - If an invariant is not callable
    class BadClass:
        bad_inv = "not a callable"

    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants(BadClass.__dict__, (BadClass,), 'stored', 'bad_inv')

    # Case 4: Multiple inheritance
    class GrandParent:
        gp_inv = inv1

    class Parent2:
        p2_inv = inv1

    class MultiChild(GrandParent, Parent2):
        mc_inv = inv1

    # Should aggregate all unique invariants from the hierarchy
    assert len(MultiChild.all_invariants) == 3

    # Case 5: No invariants present in the hierarchy
    class EmptyClass:
        pass

    class NoInvClass(EmptyClass):
        pass

    assert NoInvClass.no_invariants == ()

def test_store_invariants_edge_cases():
    # Test with a decorator-like usage simulation
    def mock_inv(x):
        return True, []

    class TestStore:
        @staticmethod
        def setup_func(dct, bases, dest, src):
            store_invariants(dct, bases, dest, src)

    # Simulate the metaclass/decorator behavior
    class Target:
        target_inv = mock_inv

    # Manually trigger the logic as a decorator would
    store_invariants(Target.__dict__, (Target,), 'stored_inv', 'target_inv')
    assert len(Target.stored_inv) == 1
    assert Target.stored_inv[0](None) == (True, ())

```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_store_invariants():
    # Mock invariant functions
    def inv1(obj):
        return True, "msg1"

    def inv2(obj):
        return False, "error2"

    def inv3(obj):
        return (True, "nested_ok"), (False, "nested_fail")

    # Test Case 1: Basic functionality and inheritance of invariants
    class Base:
        source_invariant = inv1

    class Derived(Base):
        # This should inherit source_invariant and add its own
        source_invariant_new = inv2

    # We need to manually trigger the descriptor-like behavior 
    # because store_invariants is designed to be used in a metaclass __new__ or similar.
    # Here we simulate the logic of what the decorator/metaclass would do.
    
    class MockMeta(type):
        def __new__(mcs, name, bases, attrs):
            # Simulate storing invariants from bases into 'destination_invariants'
            store_invariants(attrs, bases, 'destination_invariants', 'source_invariant')
            return super().__new__(mcs, name, bases, attrs)

    class SimpleClass(metaclass=MockMeta):
        source_invariant = inv1

    assert len(SimpleClass.destination_invariants) == 1
    assert SimpleClass.destination_invariants[0](None) == (True, ())

    # Test Case 2: Multiple invariants and wrapping logic
    class MultiInvariantClass(metaclass=MockMeta):
        source_invariant = inv3 # Complex return type
        # Note: In a real scenario, store_invariants would look at all bases.
        # To test the accumulation, we'll use a more direct approach.

    # Test Case 3: Testing the accumulation and wrapping of multiple sources
    class Accumulator(metaclass=MockMeta):
        pass

    # Manually simulate what happens to 'Accumulator' if it had its own
    attrs = {'source_invariant': inv2}
    store_invariants(attrs, (Base,), 'destination_invariants', 'source_invariant')
    
    # The resulting destination_invariants should contain wrapped versions of 
    # both Base.source_invariant and Accumulator.source_invariant
    # Wrapped inv1: returns (True, ()) or result from inv1
    # Wrapped inv2: returns (False, ('error2',)) or result from inv2
    
    # Check if the first element is the wrapped inv1 (from Base)
    # Note: store_invariants appends in order of [dct] + bases
    assert len(attrs['destination_invariants']) == 2
    
    # Test Case 4: Error handling - Non-callable invariants
    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants({'source_invariant': 'not_a_callable'}, (), 'dest', 'source_invariant')

    # Test Case 5: Verifying the wrap_invariant logic via store_invariants
    # If an invariant returns a tuple of (bool, data), it should be merged.
    class ComplexInvClass(metaclass=MockMeta):
        source_invariant = inv3 # returns ((True, 'msg'), (False, 'err'))

    # The wrapped version of inv3 should return (False, ('err',))
    # because _merge_invariant_results aggregates all False results.
    result = ComplexInvClass.destination_invariants[0](None)
    assert result == (False, ('err',))

    # Test Case 6: Verifying simple boolean return from invariant
    class SimpleBoolClass(metaclass=MockMeta):
        source_invariant = lambda x: True
    
    result = SimpleBoolClass.destination_invariants[0](None)
    assert result == (True,) # The wrap_invariant returns the result directly if it's a bool
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

class MockEnum(Enum):
    A = 1

def test_maybe_parse_user_type():
    # Test single type input
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test single string input
    assert maybe_parse_user_type("int") == ["int"]
    
    # Test preserved iterable types (Enums)
    assert maybe_parse_user_type(MockEnum) == [MockEnum]
    
    # Test list of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    
    # Test nested iterables
    assert maybe_parse_user_type([[int], str]) == (int, str)
    
    # Test tuple of types
    assert maybe_parse_user_type((float,))] == (float,)
    
    # Test error case: non-type, non-string, non-iterable input
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    # Test error case: complex object that is not iterable/type/str
    class NonIterable:
        pass
    
    with pytest.raises(TypeError):
        maybe_parse_user_type(NonIterable())

    # Test deep nesting recursion
    assert maybe_parse_user_type([["str"], [int]]) == (str, int)
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_store_invariants():
    # Helper invariant functions
    def inv1(obj):
        return True, "inv1_ok"

    def inv2(obj):
        return False, "inv2_fail"

    def inv3(obj):
        return True, "inv3_ok"

    def inv_multi(obj):
        # Testing the wrap_invariant logic via return type
        return [(True, "part1"), (False, "part2")]

    # 1. Test basic storage in a single class
    class Base:
        pass

    class Derived(Base):
        some_invariant = inv1

    assert Derived.stored_invariants == (wrap_invariant(inv1),)

    # 2. Test inheritance of invariants
    class Inherited(Derived):
        pass

    # Should contain both inv1 and its wrapped version
    assert len(Inherally.__dict__['stored_invariants']) == 1 # Wait, logic check:
    # The code uses _all_dicts which yields bases. 
    # In 'store_invariants', it collects from dct and all bases.

    class MultiInvar(Base):
        invar_a = inv1
        invar_b = inv2

    assert len(MultiInvar.stored_invariants) == 2
    # Check if wrap_invariant is applied (it transforms multi-result returns)
    # and check identity of functions in the tuple
    funcs = MultiInvar.stored_invariants
    assert any(f.__wrapped__ == inv1 or f == inv1 for f in funcs)

    # 3. Test complex wrap_invariant (merging results)
    class ComplexInv(Base):
        complex_inv = inv_multi

    # The wrapper should reduce [(True, "part1"), (False, "part2")] to (False, ("part2",))
    # We need to check the actual behavior of the wrapped function call
    result_func = ComplexInv.stored_invariants[0]
    verdict, errors = result_func(None)
    assert verdict is False
    assert errors == ("part2",)

    # 4. Test TypeError when invariant is not callable
    class BadInvar(Base):
        not_an_inv = "not_callable"

    with pytest.raises(TypeError, match="Invariants must be callable"):
        store_invariants(BadInvar.__dict__, (Base,), "stored_invariants", "not_an_inv")

    # 5. Test inheritance chain accumulation
    class GrandParent:
        gp_inv = inv1

    class Parent(GrandParent):
        p_inv = inv2

    class Child(Parent):
        c_inv = inv3

    # The class 'Child' should have all three
    assert len(Child.stored_invariants) == 3
    # Check if the functions are correctly wrapped
    # Note: wrap_invariant returns a function 'f'
    
    # 6. Test with no invariants present in chain
    class Empty(Base):
        pass
    
    assert "stored_invariants" not in Empty.__dict__

    # 7. Test that it handles non-existent source_name gracefully (skips)
    class NoSource(Base):
        def __init__(self): pass

    # This should not raise error, just result in empty tuple if nothing found
    store_invariants(NoSource.__dict__, (Base,), "stored_invariants", "non_existent")
    assert NoSource.stored_invariants == ()
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_CheckedType_serialize():
    # Create a concrete implementation of CheckedType for testing purposes
    class ConcreteCheckedType(CheckedType):
        def __init__(self, data):
            self.data = data

        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls(source_data)

        def serialize(self, format=None):
            if format == 'json':
                return {"data": self.data}
            elif format == 'text':
                return str(self.data)
            return self.data

    # Test Case 1: Default serialization (returns raw data)
    obj_default = ConcreteCheckedType("some_value")
    assert obj_default.serialize() == "some_value"

    # Test Case 2: Serialization with 'json' format
    obj_json = ConcreteCheckedType({"key": "value"})
    assert obj_json.serialize(format='json') == {"data": {"key": "value"}}

    # Test Case 3: Serialization with 'text' format
    obj_text = ConcreteCheckedType(123)
    assert obj_text.serialize(format='text') == "123"

    # Test Case 4: Verify abstract method behavior (NotImplementedError)
    class AbstractOnly(CheckedType):
        def serialize(self, format=None):
            super().serialize(format)

    with pytest.raises(TypeError):
        # Cannot instantiate abstract class
        AbstractOnly()

    # Test Case 5: Mocking to ensure the method is actually called
    mock_obj = MagicMock(spec=CheckedType)
    mock_obj.serialize.return_value = "mocked_output"
    assert mock_obj.serialize(format='xml') == "mocked_output"
    mock_obj.serialize.assert_called_once_with(format='xml')
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

class MockCheckedType(CheckedType):
    def __init__(self, data):
        self.data = data

    def serialize(self, format=None):
        if format == 'json':
            return str(self.data)
        elif format == 'text':
            return f"Value: {self.data}"
        return self.data

def test_CheckedType_serialize():
    # Test default behavior (no format)
    obj_default = MockCheckedType({"key": "value"})
    assert obj_default.serialize() == {"key": "value"}

    # Test specific format 'json'
    obj_json = MockCheckedType({"id": 1})
    assert obj_json.serialize(format='json') == "{'id': 1}"

    # Test specific format 'text'
    obj_text = MockCheckedType("hello")
    assert obj_text.serialize(format='text') == "Value: hello"

    # Test with different data types
    obj_int = MockCheckedType(42)
    assert obj_int.serialize(format='json') == "42"

def test_CheckedType_serialize_abstract_error():
    # Test that the abstract method raises NotImplementedError if called on base class
    class AbstractImplementation(CheckedType):
        pass
    
    # Since serialize is @abstractmethod, calling it on a class that 
    # doesn't override it (and isn't instantiated) is handled by Python.
    # We test the behavior of an un-implemented subclass.
    with pytest.raises(TypeError):
        # Attempting to instantiate a class with abstract methods
        AbstractImplementation()
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_CheckedType_serialize():
    class MockCheckedType(CheckedType):
        def __init__(self, data):
            self.data = data

        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls(source_data)

        def serialize(self, format=None):
            if format == 'json':
                return {"data": self.data}
            elif format == 'text':
                return str(self.data)
            return self.data

    # Test basic serialization (default format)
    obj = MockCheckedType("test_value")
    assert obj.serialize() == "test_value"

    # Test specific format: json
    assert obj.serialize(format='json') == {"data": "test_value"}

    # Test specific format: text
    assert obj.serialize(format='text') == "test_value"

    # Test with complex data structure
    complex_obj = MockCheckedType({"key": [1, 2, 3]})
    assert complex_obj.serialize(format='json') == {"data": {"key": [1, 2, 3]}}

    # Verify abstract method behavior via subclassing error if attempted on base class
    with pytest.raises(TypeError):
        # ABCMeta prevents instantiation of classes with unimplemented abstract methods
        class UnimplementedType(CheckedType):
            pass
        UnimplementedType()

    # Test that serialize is called as expected using a Mock
    mock_instance = MagicMock(spec=CheckedType)
    mock_instance.serialize.return_value = "mocked"
    assert mock_instance.serialize("any_format") == "mocked"
    mock_instance.serialize.assert_called_once_with("any_format")
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_store_invariants():
    # Mock invariant functions
    def inv1(obj):
        return True, []

    def inv2(obj):
        return False, ["error1"]

    def inv3(obj):
        # Test case where result is a list of results (to be merged by wrap_invariant)
        return [(True, []), (False, "error2")]

    class Base:
        pass

    class Derived(Base):
        pass

    # 1. Test basic storage in a single class
    dct = {}
    store_invariants(dct, (Base,), 'dest', 'src')
    assert 'dest' not in dct  # Source doesn't exist in Base or Derived

    # 2. Test inheritance of invariants
    class Parent:
        src_inv = inv1

    class Child(Parent):
        pass

    dct_child = {}
    store_invariants(dct_child, (Parent,), 'dest', 'src_inv')
    assert len(dct_child['dest']) == 1
    # Check that it's wrapped and returns the correct type/value
    # wrap_invariant should return (True, ()) for inv1
    assert dct_child['dest'][0](None) == (True, ())

    # 3. Test inheritance of multiple invariants
    class GrandParent:
        src_inv = inv1

    class Parent2(GrandParent):
        src_inv = inv2

    class Child2(Parent2):
        pass

    dct_child2 = {}
    store_invariants(dct_child2, (Parent2,), 'dest', 'src_inv')
    # Should have both inv1 and inv2 wrapped
    assert len(dct_child2['dest']) == 2
    results = [f(None) for f in dct_child2['dest']]
    assert (True, ()) in results
    assert (False, ("error1",)) in results

    # 4. Test merging logic via wrap_invariant (complex return value)
    class ComplexParent:
        src_inv = inv3

    dct_complex = {}
    store_invariants(dct_complex, (ComplexParent,), 'dest', 'src_inv')
    # wrap_invariant should merge [(True, []), (False, "error2")] -> (False, ("error2",))
    assert dct_complex['dest'][0](None) == (False, ("error2",))

    # 5. Test TypeError when invariant is not callable
    class BadParent:
        src_inv = "not a function"

    dct_bad = {}
    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants(dct_bad, (BadParent,), 'dest', 'src_inv')

    # 6. Test that it doesn't crash if source_name is missing in some bases
    class OnlyOneHasIt:
        src_inv = inv1

    class OtherDoesNot:
        pass

    dct_mixed = {}
    store_invariants(dct_mixed, (OnlyOneHasIt, OtherDoesNot), 'dest', 'src_inv')
    assert len(dct_mixed['dest']) == 1
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    @wrap_invariant(lambda x: (True, "msg"))
    def invariant_bool_true(x):
        return True, "msg"

    assert invariant_bool_true(None) == (True, "msg")

    # Case 2: Invariant returns a simple boolean (False)
    @wrap_invariant(lambda x: (False, "error"))
    def invariant_bool_false(x):
        return False, "error"

    assert invariant_bool_false(None) == (False, "error")

    # Case 3: Invariant returns a list of tuples (Multiple results)
    # The wrapper should merge them.
    @wrap_invariant(lambda x: [(True, "a"), (False, "b"), (True, "c"), (False, "d")])
    def invariant_multiple(x):
        return [(True, "a"), (False, "b"), (True, "c"), (False, "d")]

    # Expected: verdict False, data contains only the failed error messages ('b', 'd')
    verdict, errors = invariant_multiple(None)
    assert verdict is False
    assert errors == ("b", "d")

    # Case 4: Invariant returns a list of tuples where all are True
    @wrap_invariant(lambda x: [(True, "a"), (True, "b")])
    def invariant_all_true(x):
        return [(True, "a"), (True, "b")]

    verdict, errors = invariant_all_true(None)
    assert verdict is True
    assert errors == ()

    # Case 5: Test with arguments passing through
    @wrap_invariant(lambda x, y: (x == y, "mismatch"))
    def invariant_with_args(x, y):
        return (x == y, "mismatch")

    assert invariant_with_args(10, 10) == (True, "mismatch")
    assert invariant_with_args(10, 20) == (False, "mismatch")
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_get_type():
    # Test case 1: Passing an actual type object should return the same type object
    assert get_type(int) == int
    assert get_type(str) == str
    assert get_type(list) == list

    # Test case 2: Passing a string representation of a built-in type
    # Note: 'builtins' is the standard module for basic types in Python 3
    assert get_type("builtins.int") == int
    assert get_type("builtins.str") == str

    # Test case 3: Passing a string representation of a class from a known module
    # Since we can't easily mock __import__ without complexity, we use a standard library type
    assert get_type("collections.abc.Iterable") == Iterable

    # Test case 4: Test with an error for invalid string format
    with pytest.raises(ValueError):
        # This will fail because rsplit('.', 1) won't find a '.' to split on
        get_type("int")

    # Test case 5: Test with an error for non-existent module/class
    with pytest.raises(ImportError):
        get_type("non_existent_module.SomeClass")

    # Test case 6: Test with a valid class in the current scope (if applicable)
    # Since CheckedType is in the same module, we can reference it
    assert get_type("builtins.Exception") == Exception
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    def invariant_true(data):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict, errors = wrapped_true({"x": 1})
    assert verdict is True
    assert errors == []

    # Case 2: Invariant returns a simple boolean (False)
    def invariant_false(data):
        return False, ["error_msg"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, errors = wrapped_false({"x": 1})
    assert verdict is False
    assert errors == ["error_msg"]

    # Case 3: Invariant returns a list of results (multiple tests)
    # All passing
    def invariant_all_pass(data):
        return [(True, None), (True, "extra")]
    
    wrapped_pass = wrap_invariant(invariant_all_pass)
    verdict, errors = wrapped_pass({"x": 1})
    assert verdict is True
    assert errors == ()

    # Case 4: Invariant returns a list of results (one failing)
    def invariant_one_fail(data):
        return [(True, None), (False, "failure_1"), (True, "extra")]
    
    wrapped_fail = wrap_invariant(invariant_one_fail)
    verdict, errors = wrapped_fail({"x": 1})
    assert verdict is False
    assert errors == ("failure_1",)

    # Case 5: Invariant returns a list of results (multiple failing)
    def invariant_multi_fail(data):
        return [(False, "err1"), (True, None), (False, "err2")]
    
    wrapped_multi_fail = wrap_invariant(invariant_multi_fail)
    verdict, errors = wrapped_multi_fail({"x": 1})
    assert verdict is False
    assert errors == ("err1", "err2")

    # Case 6: Invariant returns a single boolean (direct return)
    def invariant_direct_bool(data):
        return True
    
    wrapped_direct = wrap_invariant(invariant_direct_bool)
    verdict, errors = wrapped_direct({"x": 1})
    assert verdict is True
    assert errors == []
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple (bool, data) tuple
    @wrap_invariant(lambda x: (x > 0, "error message"))
    def invariant_simple(x):
        return (x > 0, "error message")

    assert invariant_simple(5) == (True, ())
    assert invariant_simple(-1) == (False, ("error message",))

    # Case 2: Invariant returns a single boolean
    @wrap_invariant(lambda x: x == 10)
    def invariant_bool_only(x):
        return x == 10

    assert invariant_simple(5) == (True, ()) # Note: reusing logic from Case 1 for simplicity in testing the wrapper's branch coverage
    assert invariant_simple(-1) == (False, ("error message",))
    # Re-test specifically with bool return type to hit the 'isinstance(result[0], bool)' branch
    @wrap_invariant(lambda x: x == 10)
    def invariant_pure_bool(x):
        return x == 10
    assert invariant_pure_bool(10) == True
    assert invariant_pure_bool(5) == False

    # Case 3: Invariant returns a list/tuple of multiple results (the merging logic)
    @wrap_invariant(lambda x: [
        (x > 0, "positive error"),
        (x < 10, "too large error"),
        (x % 2 == 0, "must be even")
    ])
    def invariant_multiple(x):
        # This simulates the logic of returning an iterable of (bool, data)
        return [
            (x > 0, "positive error"),
            (x < 10, "too large error"),
            (x % 2 == 0, "must be even")
        ]

    # All pass: x=4 -> True, []
    assert invariant_multiple(4) == (True, ())
    
    # One fails: x=11 -> False, ("too large error",)
    assert invariant_multiple(11) == (False, ("too large error",))
    
    # Two fail: x=-1 -> False, ("positive error", "must be even")
    # Note: in the provided code, if x=-1, 
    # (x > 0) is False -> "positive error"
    # (x < 10) is True
    # (x % 2 == 0) is False -> "must be even"
    assert invariant_multiple(-1) == (False, ("positive error", "must be even"))

    # Case 4: Invariant returns a single bool inside a tuple (simulating the same as Case 1)
    @wrap_invariant(lambda x: (True, "won't see this"))
    def invariant_single_tuple(x):
        return (True, "won't see this")
    assert invariant_single_tuple(None) == (True, ())

    # Case 5: Invariant returns multiple failures
    @wrap_invariant(lambda x: [(False, "err1"), (False, "err2")])
    def invariant_all_fail(x):
        return [(False, "err1"), (False, "err2")]
    assert invariant_all_fail(None) == (False, ("err1", "err2"))
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

class TestEnum(Enum):
    A = 1

class MockType:
    pass

def test_maybe_parse_user_type():
    # Test single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(MockType) == [MockType]

    # Test string type
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type("") == [""]

    # Test preserved iterable types (Enum)
    assert maybe_parse_user_type(TestEnum) == [TestEnum]
    assert maybe_parse_user_type((TestEnum,)) == [TestEnum]

    # Test nested iterables (tuple of types/strings)
    assert maybe_parse_user_type((int, str)) == (int, str)
    assert maybe_parse_user_type([int, [str, MockType]]) == (int, str, MockType)
    assert maybe_parse_user_type((TestEnum, "string")) == (TestEnum, "string")

    # Test deep nesting with mixed types
    complex_input = (int, (str, [MockType]))
    assert maybe_parse_user_type(complex_input) == (int, str, MockType)

    # Test invalid input (raises TypeError)
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    with pytest.raises(TypeError):
        maybe_parse_user_type([None]) # None is not a type, string, or preserved iterable
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from pyrsistent import pset

class StringSet(CheckedPSet):
    __type__ = (str,)

class IntSet(CheckedPSet):
    __type__ = (int,)

def test_CheckedPSet_serialize():
    # Test serialization of a basic CheckedPSet with strings
    initial_data = {"apple", "banana", "cherry"}
    string_set = StringSet(initial_data)
    serialized_string_set = string_set.serialize()
    
    assert isinstance(serialized_string_set, set)
    assert serialized_string_set == initial_data

    # Test serialization with custom serializer (e.g., upper case)
    def upper_serializer(fmt, value):
        return value.upper()
    
    serialized_upper = string_set.serialize(format="upper")
    assert serialized_upper == {"APPLE", "BANANA", "CHERRY"}

    # Test serialization of an IntSet
    int_set = IntSet([1, 2, 3])
    serialized_int_set = int_set.serialize()
    assert serialized_int_set == {1, 2, 3}

    # Test serialization of a CheckedPSet containing another CheckedType (if applicable)
    # Note: serialize calls the __serializer__ which defaults to checking if value is CheckedType
    class NestedSet(CheckedPSet):
        __type__ = (object,)
    
    nested_set = NestedSet([string_set])
    serialized_nested = nestedly_set.serialize()
    # The default serializer calls .serialize() on the inner CheckedType
    assert serialized_nested == [{"apple", "banana", "cherry"}]

def test_CheckedPSet_serialize_empty():
    empty_set = StringSet([])
    assert empty_set.serialize() == set()
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a single boolean (True)
    def invariant_true(x):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    assert wrapped_true(1) == (True, [])

    # Case 2: Invariant returns a single boolean (False)
    def invariant_false(x):
        return False, ["error"]
    
    wrapped_false = wrap_invariant(invariant_false)
    assert wrapped_false(1) == (False, ["error"])

    # Case 3: Invariant returns a list of results (All True)
    def invariant_all_pass(x):
        return [(True, None), (True, "some_data")]
    
    wrapped_pass = wrap_invariant(invariant_all_pass)
    assert wrapped_pass(1) == (True, ())

    # Case 4: Invariant returns a list of results (One False)
    def invariant_one_fail(x):
        return [(True, "good"), (False, "bad_data"), (True, "ok")]
    
    wrapped_fail = wrap_invariant(invariant_one_fail)
    assert wrapped_fail(1) == (False, ("bad_data",))

    # Case 5: Invariant returns a list of results (Multiple False)
    def invariant_multiple_fail(x):
        return [(False, "err1"), (True, "ok"), (False, "err2")]
    
    wrapped_multi_fail = wrap_invariant(invariant_multiple_fail)
    assert wrapped_multi_fail(1) == (False, ("err1", "err2"))

    # Case 6: Invariant returns a simple boolean directly (not in tuple)
    def invariant_simple_bool(x):
        return True
    
    wrapped_simple = wrap_invariant(invariant_simple_bool)
    assert wrapped_simple(1) == True
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    def invariant_true(x):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict, data = wrapped_true(10)
    assert verdict is True
    assert data == []

    # Case 2: Invariant returns a simple boolean (False)
    def invariant_false(x):
        return False, ["error"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, data = wrapped_false(10)
    assert verdict is False
    assert data == ["error"]

    # Case 3: Invariant returns a list of (bool, data) tuples (All True)
    def invariant_all_pass(x):
        return [(True, "msg1"), (True, "msg2")]
    
    wrapped_pass = wrap_invariant(invariant_all_pass)
    verdict, data = wrapped_pass(10)
    assert verdict is True
    assert data == ()

    # Case 4: Invariant returns a list of (bool, data) tuples (Some False)
    def invariant_some_fail(x):
        return [(True, "msg1"), (False, "error1"), (True, "msg2"), (False, "error2")]
    
    wrapped_fail = wrap_invariant(invariant_some_fail)
    verdict, data = wrapped_fail(10)
    assert verdict is False
    assert data == ("error1", "error2")

    # Case 5: Invariant returns a list of (bool, data) tuples (All False)
    def invariant_all_fail(x):
        return [(False, "err1"), (False, "err2")]
    
    wrapped_all_fail = wrap_invariant(invariant_all_fail)
    verdict, data = wrapped_all_fail(10)
    assert verdict is False
    assert data == ("err1", "err2")

    # Case 6: Invariant returns a single boolean without tuple (Directly handled by wrapper logic)
    def invariant_direct_bool(x):
        return True, [] # The check 'isinstance(result[0], bool)' handles this
    
    wrapped_direct = wrap_invariant(invariant_direct_bool)
    verdict, data = wrapped_direct(10)
    assert verdict is True
    assert data == []
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    @wrap_invariant(lambda x: (True, "success"))
    def invariant_bool_true(x):
        return True, "success"

    assert invariant_bool_true(None) == (True, "success")

    # Case 2: Invariant returns a simple boolean (False)
    @wrap_invariant(lambda x: (False, "failure"))
    def invariant_bool_false(x):
        return False, "failure"

    assert invariant_bool_false(None) == (False, "failure")

    # Case 3: Invariant returns a list/tuple of results (All True)
    @wrap_invariant(lambda x: [(True, "msg1"), (True, "msg2")])
    def invariant_all_true(x):
        return [(True, "msg1"), (True, "msg2")]

    assert invariant_all_true(None) == (True, ("msg1", "msg2"))

    # Case 4: Invariant returns a list/tuple of results (Mixed results)
    @wrap_invariant(lambda x: [(True, "ok"), (False, "error1"), (True, "fine"), (False, "error2")])
    def invariant_mixed(x):
        return [(True, "ok"), (False, "error1"), (True, "fine"), (False, "error2")]

    assert invariant_mixed(None) == (False, ("error1", "error2"))

    # Case 5: Invariant returns a list/tuple of results (All False)
    @wrap_invariant(lambda x: [(False, "err1"), (False, "err2")])
    def invariant_all_false(x):
        return [(False, "err1"), (False, "err2")]

    assert invariant_all_false(None) == (False, ("err1", "err2"))

    # Case 6: Verify the wrapper passes arguments through correctly
    @wrap_invariant(lambda x, y: (x == y, "matched" if x == y else "mismatch"))
    def invariant_with_args(x, y):
        return (x == y, "matched" if x == y else "mismatch")

    assert invariant_with_args(1, 1) == (True, ("matched",))
    assert invariant_with_args(1, 2) == (False, ("mismatch",))
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    def invariant_true(data):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict, errors = wrapped_true({"a": 1})
    assert verdict is True
    assert errors == []

    # Case 2: Invariant returns a simple boolean (False)
    def invariant_false(data):
        return False, ["error"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, errors = wrapped_false({"a": 1})
    assert verdict is False
    assert errors == ["error"]

    # Case 3: Invariant returns a tuple of results (merging required)
    # Result format: [(bool, error_data), ...]
    def invariant_multi(data):
        return (
            (True, None),
            (False, "first_error"),
            (True, "ignored_data"),
            (False, "second_error")
        )
    
    wrapped_multi = wrap_invariant(invariant_multi)
    verdict, errors = wrapped_multi({"a": 1})
    assert verdict is False
    assert errors == ("first_error", "second_error")

    # Case 4: Invariant returns a tuple of results where all are True
    def invariant_all_true(data):
        return (
            (True, None),
            (True, "some_meta"),
        )
    
    wrapped_all_true = wrap_invariant(invariant_all_true)
    verdict, errors = wrapped_all_true({"a": 1})
    assert verdict is True
    assert errors == ()

    # Case 5: Invariant returns a simple boolean (already handled by logic branch)
    def invariant_direct_bool(data):
        return False, ["direct_error"]
    
    wrapped_direct = wrap_invariant(invariant_direct_bool)
    verdict, errors = wrapped_direct({"a": 1})
    assert verdict is False
    assert errors == ["direct_error"]
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from pyrsistent import pset

class MockCheckedType(CheckedPSet):
    __type__ = (int,)

def test_CheckedPSet_serialize():
    # Test 1: Basic serialization of a simple CheckedPSet with integers
    initial_data = {1, 2, 3}
    p_set = MockCheckedType(initial_data)
    serialized = p_set.serialize()
    
    assert isinstance(serialized, set)
    assert serialized == {1, 2, 3}

    # Test 2: Serialization with a custom serializer for CheckedType objects
    class NestedCheckedType(CheckedType):
        def __init__(self, val):
            self.val = val
        def serialize(self, format=None):
            return f"value_{self.val}"
        @classmethod
        def create(cls, source_data, _factory_fields=None, ignore_extra=False):
            return cls(source_data)

    nested_obj = NestedCheckedType(10)
    p_set_complex = MockCheckedType({nested_obj, 5})
    
    # The default __serializer__ in CheckedTypeMeta calls value.serialize() 
    # if the value is an instance of CheckedType
    serialized_complex = p_set_complex.serialize()
    assert "value_10" in serialized_complex
    assert 5 in serialized_complex
    assert isinstance(serialized_complex, set)

    # Test 3: Serialization when elements are not CheckedTypes (standard behavior)
    p_set_strings = MockCheckedType({"a", "b"})
    # Note: In the provided code, MockCheckedType has __type__ = (int,)
    # So we must ensure we don't trigger a TypeError during creation.
    # We use a class without strict type constraints for this specific sub-test
    class UnconstrainedSet(CheckedPSet):
        __slots__ = ()
    
    p_set_unconstrained = UnconstrainedSet({"apple", "banana"})
    serialized_unconstrained = p_set_unconstrained.serialize()
    assert serialized_unconstrained == {"apple", "banana"}

    # Test 4: Verify serialization maintains set properties (no duplicates)
    p_set_dupes = MockCheckedType([1, 1, 2])
    serialized_dupes = p_set_dupes.serialize()
    assert len(serialized_dupes) == 2
    assert serialized_dupes == {1, 2}
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    def invariant_true(x):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict, errors = wrapped_true(10)
    assert verdict is True
    assert errors == []

    # Case 2: Invariant returns a simple boolean (False)
    def invariant_false(x):
        return False, ["error"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, errors = wrapped_false(10)
    assert verdict is False
    assert errors == ["error"]

    # Case 3: Invariant returns a tuple of (bool, data) results (All True)
    def invariant_all_true(x):
        return [(True, "ok"), (True, "fine")]
    
    wrapped_all_true = wrap_invariant(invariant_all_true)
    verdict, errors = wrapped_all_true(10)
    assert verdict is True
    assert errors == ()

    # Case 4: Invariant returns a tuple of (bool, data) results (One False)
    def invariant_mixed(x):
        return [(True, "ok"), (False, "bad_data"), (True, "good")]
    
    wrapped_mixed = wrap_invariant(invariant_mixed)
    verdict, errors = wrapped_mixed(10)
    assert verdict is False
    assert errors == ("bad_data",)

    # Case 5: Invariant returns a tuple of (bool, data) results (Multiple Fails)
    def invariant_many_fails(x):
        return [(False, "err1"), (False, "err2"), (True, "ignored")]
    
    wrapped_fails = wrap_invariant(invariant_many_fails)
    verdict, errors = wrapped_fails(10)
    assert verdict is False
    assert errors == ("err1", "err2")

    # Case 6: Invariant returns a simple boolean (True/False) without the tuple structure
    # The wrapper logic checks `isinstance(result[0], bool)`
    def invariant_simple_bool(x):
        return True
    
    wrapped_simple = wrap_invariant(invariant_simple_bool)
    verdict, errors = wrapped_simple(10)
    assert verdict is True
    assert errors == []

    def invariant_simple_fail(x):
        return False
    
    # Note: In the original code's implementation of wrap_invariant, 
    # if result[0] is bool, it returns result as-is.
    # If input was (False,), verdict becomes False, errors becomes [].
    wrapped_simple_fail = wrap_invariant(invariant_simple_fail)
    result = wrapped_simple_fail(10)
    assert result == False 
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_store_invariants():
    # Mock invariant functions
    def inv1(obj):
        return True, "error1"

    def inv2(obj):
        return False, "fail1"

    def inv3(obj):
        return (True, "msg1"), (False, "msg2")

    # Case 1: Single class with one invariant
    class Base:
        pass

    class Sub(Base):
        check = inv1

    assert len(Sub.check) == 1
    # wrap_invariant turns result into (verdict, data)
    # inv1 returns (True, "error1"), so wrapped is (True, ("error1",))
    assert Sub.check[0](None) == (True, ("errorjack" if False else "error1",))

    # Case 2: Inheritance of invariants
    class Parent:
        rule = inv1

    class Child(Parent):
        pass

    # Child should inherit rule from Parent
    assert len(Child.rule) == 1
    assert Child.rule[0](None) == (True, ("error1",))

    # Case 3: Multiple invariants from different levels of hierarchy
    class GrandParent:
        rule_a = inv1

    class Parent2(GrandParent):
        rule_b = inv2

    class Child2(Parent2):
        pass

    assert len(Child2.rule_a) == 1 # Inherited from GP
    # We need to test the logic of store_invariants specifically regarding destination_name/source_name
    
    # Manually triggering the descriptor-like behavior for testing
    class TestClass:
        pass

    # Mocking the dictionary and bases as the decorator would
    dct = {}
    bases = (Parent2,)
    
    # We simulate what happens when 'check' is stored from 'rule_a' and 'rule_b'
    # Note: store_invariants looks for source_name in dct and bases
    class MockBase:
        src = inv1

    class MockSub(MockBase):
        pass

    # Testing the actual function implementation logic
    # Since we can't easily use decorators in a pure unit test without defining classes,
    # we simulate the call to store_invariants manually.
    
    target_dct = {}
    # Simulate: @store_invariants(target_dct, (MockBase,), 'dest', 'src')
    store_invariants(target_dct, (MockBase,), 'dest', 'src')
    
    assert 'dest' in target_dict := target_dct
    assert len(target_dict['dest']) == 1
    # Check if it wrapped correctly
    result = target_dict['dest'][0](None)
    assert result == (True, ("error1",))

    # Case 4: Error handling - Non-callable invariant
    class BadClass:
        not_a_func = "I am a string"

    with pytest.raises(TypeError, match='Invariants must be callable'):
        store_invariants({}, (BadClass,), 'dest', 'not_a_func')

    # Case 5: Merging multiple results from one invariant function (via wrap_invariant)
    class MultiResultClass:
        pass
    
    target_dct_multi = {}
    # We use an invariant that returns a tuple of results
    def multi_inv(obj):
        return [(True, "ok"), (False, "bad")]

    # Manually inject to test the wrapping logic inside store_invariants
    class Source:
        src = multi_inv

    store_invariants(target_dct_multi, (Source,), 'dest', 'src')
    # wrap_invariant should merge (True, "ok") and (False, "bad") into (False, ("bad",))
    verdict, errors = target_dct_multi['dest'][0](None)
    assert verdict is False
    assert errors == ("bad",)

    # Case 6: Inheritance of multiple different source names
    class MultiSource:
        src1 = inv1
        src2 = inv2

    target_dct_multi_src = {}
    store_invariants(target_dct_multi_src, (MultiSource,), 'dest', 'src1') 
    # This only looks for 'src1'. Let's test if it finds both if we define them.
    
    class MultiSrcBase:
        src = inv1
    
    class MultiSrcSub(MultiSrcBase):
        src = inv2

    target_dct_test = {}
    store_invariants(target_dct_test, (MultiSrcSub,), 'dest', 'src')
    # It should find 'src' in Sub and 'src' in Base.
    assert len(target_dct_test['dest']) == 2

    # Case 7: Verification of _all_dicts utility via store_invariants
    class DeepBase:
        src = inv1
    class MidBase(DeepBase):
        pass
    class TopBase(MidBase):
        src = inv2

    target_dct_deep = {}
    store_invariants(target_dct_deep, (TopBase,), 'dest', 'src')
    # Should find src from TopBase and DeepBase
    assert len(target_dct_deep['dest']) == 2
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (single result)
    @wrap_invariant(lambda x: (True, "success"))
    def invariant_bool(x):
        return True, "data"

    assert invariant_bool(None) == (True, "data")

    @wrap_invariant(lambda x: (False, "failure"))
    def invariant_fail_bool(x):
        return False, "error"

    assert invariant_fail_bool(None) == (False, "error")

    # Case 2: Invariant returns a list of results (multiple tests)
    # Success scenario: all tests pass
    @wrap_invariant(lambda x: [(True, "ok"), (True, "good")])
    def invariant_all_pass(x):
        return [(True, "ok"), (True, "good")]

    assert invariant_all_pass(None) == (True, ("ok", "good"))

    # Failure scenario: one test fails
    @wrap_invariant(lambda x: [(True, "ok"), (False, "bad_error")])
    def invariant_one_fails(x):
        return [(True, "ok"), (False, "bad_error")]

    assert invariant_one_fails(None) == (False, ("bad_error",))

    # Failure scenario: multiple tests fail
    @wrap_invariant(lambda x: [(False, "err1"), (True, "ignore_me"), (False, "err2")])
    def invariant_multiple_fail(x):
        return [(False, "err1"), (True, "ignore_me"), (False, "err2")]

    assert invariant_multiple_fail(None) == (False, ("err1", "err2"))

    # Case 3: Verifying the logic with dynamic inputs
    def complex_invariant(x):
        # Simulate a function that returns multiple truth values/messages
        results = []
        if x > 0:
            results.append((True, "positive"))
        else:
            results.append((False, "not_positive"))
        return results

    wrapped = wrap_invariant(complex_invariant)
    
    assert wrapped(10) == (True, ("positive",))
    assert wrapped(-5) == (False, ("not_positive",))
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from enum import Enum

class MockEnum(Enum):
    A = 1

class MockClass:
    pass

def test_maybe_parse_user_type():
    # Test single type
    assert maybe_parse_user_type(MockClass) == [MockClass]
    
    # Test string input
    assert maybe_parse_user_type("int") == ["int"]
    
    # Test preserved iterable type (Enum)
    assert maybe_parse_user_type(MockEnum) == [MockEnum]
    
    # Test list of types
    assert maybe_parse_user_type([MockClass, "str"]) == (MockClass, "str")
    
    # Test nested iterable
    assert maybe_parse_user_type([[MockClass], ["str"]]) == (MockClass, "str")
    
    # Test tuple of types
    assert maybe_parse_user_type((MockClass, MockEnum)) == (MockClass, MockEnum)

    # Test invalid input (non-type, non-string, non-iterable)
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    # Test InvariantException structure
    error_msg = "Error occurred"
    exc = InvariantException(error_codes=["err1", lambda: "err2"], missing_fields=("field1",))
    assert exc.invariant_errors == ("err1", "err2")
    assert exc.missing_fields == ("field1",)
    assert "invariant_errors=[err1, err2], missing_fields=[field1]" in str(exc)

def test_invariant_exception_formatting():
    # Test with no errors or fields
    exc = InvariantException()
    assert "invariant_errors=[], missing_fields=[]" in str(exc)
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

class MockCheckedType(CheckedType):
    @classmethod
    def create(cls, source_data, _factory_fields=None, ignore_extra=False):
        return cls(source_data)

    def serialize(self, format=None):
        return f"serialized_{self.value}"

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if isinstance(other, MockCheckedType):
            return self.value == other.value
        return False

class StringSet(CheckedPSet):
    __type__ = (str,)

class ComplexSet(CheckedPSet):
    __type__ = (MockCheckedType,)

def test_CheckedPSet_serialize():
    # Test 1: Basic serialization of a set with primitive types
    simple_set = StringSet(["a", "b", "c"])
    assert simple_set.serialize() == {"a", "b", "c"}

    # Test 2: Serialization of CheckedType objects within the set
    obj1 = MockCheckedType("data1")
    obj2 = MockCheckedType("data2")
    complex_set = ComplexSet([obj1, obj2])
    
    expected_serialization = {"serialized_data1", "serialized_data2"}
    assert complex_set.serialize() == expected_serialization

    # Test 3: Serialization with an empty set
    empty_set = StringSet([])
    assert empty_set.serialize() == set()

    # Test 4: Verify that the serializer uses the instance's __serializer__ logic
    # (By default, it should call .serialize() on CheckedType instances)
    class CustomSerializerSet(CheckedPSet):
        def __serializer__(self, format, value):
            return f"custom_{value}"

    custom_set = CustomSerializerSet([1, 2])
    assert custom_set.serialize() == {"custom_1", "custom_2"}

    # Test 5: Verify that non-CheckedType elements are returned as-is by the default serializer
    # (This is implicitly tested in Test 1, but we ensure it works for mixed types if allowed)
    class MixedSet(CheckedPSet):
        __type__ = (object,)

    mixed_set = MixedSet([obj1, "plain_string"])
    assert mixed_set.serialize() == {"serialized_data1", "plain_string"}
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Test case 1: Invariant returns a simple boolean (single success)
    def invariant_success(x):
        return True, []
    
    wrapped_success = wrap_invariant(invariant_success)
    verdict, data = wrapped_success(10)
    assert verdict is True
    assert data == []

    # Test case 2: Invariant returns a simple boolean (single failure)
    def invariant_failure(x):
        return False, ["error"]
    
    wrapped_failure = wrap_invariant(invariant_failure)
    verdict, data = wrapped_failure(10)
    assert verdict is False
    assert data == ["error"]

    # Test case 3: Invariant returns a list of (bool, data) tuples (merging logic)
    def invariant_mixed(x):
        return [
            (True, "passed_1"),
            (False, "failed_a"),
            (True, "passed_2"),
            (False, "failed_b")
        ]
    
    wrapped_mixed = wrap_invariant(invariant_mixed)
    verdict, data = wrapped_mixed(10)
    # Should be False because at least one failed
    assert verdict is False
    # Should only contain the data from the failed tests
    assert data == ("failed_a", "failed_b")

    # Test case 4: Invariant returns all successes in a list
    def invariant_all_pass(x):
        return [
            (True, "msg1"),
            (True, "msg2")
        ]
    
    wrapped_all_pass = wrap_invariant(invariant_all_pass)
    verdict, data = wrapped_all_pass(10)
    assert verdict is True
    assert data == ()

    # Test case 5: Invariant returns a simple boolean (no tuple/list provided)
    def invariant_raw_bool(x):
        return True
    
    wrapped_raw = wrap_invariant(invariant_raw_bool)
    verdict, data = wrapped_raw(10)
    assert verdict is True
    # Note: The implementation returns result directly if first element is bool. 
    # If the function returned just 'True', result[0] would raise TypeError for non-subscriptable type.
    # However, based on code: `if isinstance(result[0], bool): return result`
    # This implies the input invariant MUST return a subscriptable object (like a tuple) 
    # where the first element is a bool. Let's test that specific boundary.

    def invariant_tuple_bool(x):
        return (True, "data")
    
    wrapped_tuple_bool = wrap_invariant(invariant_tuple_bool)
    verdict, data = wrapped_tuple_bool(10)
    assert verdict is True
    assert data == "data" # Since it returned result directly, and result was (True, "data")
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

class MockEnum(Enum):
    VAL = 1

class SimpleType:
    pass

def test_maybe_parse_user_type():
    # Test single type
    assert maybe_parse_user_type(SimpleType) == [SimpleType]
    
    # Test single string
    assert maybe_parse_user_type("int") == ["int"]
    
    # Test preserved iterable type (Enum)
    assert maybe_parse_user_type(MockEnum) == [MockEnum]
    
    # Test list of types
    assert maybe_parse_user_type([SimpleType, "str"]) == (SimpleType, "str")
    
    # Test nested iterables
    assert maybe_parse_user_type([[SimpleType], ["str"]]) == (SimpleType, "str")
    
    # Test tuple of types
    assert maybe_parse_user_type((SimpleType,)) == (SimpleType,)
    
    # Test invalid input (not type/string/iterable)
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    # Test deep nesting with mixed valid types
    complex_input = [MockEnum, ["int", SimpleType]]
    assert maybe_parse_user_type(complex_input) == (MockEnum, "int", SimpleType)
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

class MockEnum(Enum):
    A = 1

def test_maybe_parse_user_type():
    # Test single type input
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(str) == [str]
    
    # Test preserved types (Enum)
    assert maybe_parse_user_type(MockEnum) == [MockEnum]
    
    # Test string input
    assert maybe_parse_user_type("int") == ["int"]
    
    # Test simple iterable of types
    assert maybe_parse_user_type([int, str]) == (int, str)
    
    # Test nested iterable of types
    assert maybe_parse_user_type([[int], [str]]) == (int, str)
    
    # Test mixed list containing strings and types
    assert maybe_parse_user_type([int, "float", MockEnum]) == (int, "float", MockEnum)

    # Test nested complex structure
    assert maybe_parse_user_type(([int], ["str"])) == (int, "str")

    # Test invalid input (non-type, non-string, non-iterable)
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    # Test invalid input (None)
    with pytest.raises(TypeError):
        maybe_parse_user_type(None)
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_wrap_invariant():
    # Case 1: Invariant returns a simple boolean (True)
    def invariant_true(x):
        return True, []
    
    wrapped_true = wrap_invariant(invariant_true)
    verdict, data = wrapped_true(10)
    assert verdict is True
    assert data == []

    # Case 2: Invariant returns a simple boolean (False)
    def invariant_false(x):
        return False, ["error"]
    
    wrapped_false = wrap_invariant(invariant_false)
    verdict, data = wrapped_false(10)
    assert verdict is False
    assert data == ["error"]

    # Case 3: Invariant returns a sequence of (bool, data) tuples (All True)
    def invariant_all_true(x):
        return [(True, "a"), (True, "b")]
    
    wrapped_all_true = wrap_invariant(invariant_all_true)
    verdict, data = wrapped_all_true(10)
    assert verdict is True
    assert data == ("a", "b")

    # Case 4: Invariant returns a sequence of (bool, data) tuples (Mixed results)
    def invariant_mixed(x):
        return [(True, "good"), (False, "bad1"), (True, "ok"), (False, "bad2")]
    
    wrapped_mixed = wrap_invariant(invariant_mixed)
    verdict, data = wrapped_mixed(10)
    assert verdict is False
    # Should only contain data from the False entries
    assert data == ("bad1", "bad2")

    # Case 5: Invariant returns a sequence of (bool, data) tuples (All False)
    def invariant_all_false(x):
        return [(False, "err1"), (False, "err2")]
    
    wrapped_all_false = wrap_invariant(invariant_all_false)
    verdict, data = wrapped_all_false(10)
    assert verdict is False
    assert data == ("err1", "err2")
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

class MockEnum(Enum):
    A = 1

class NonIterableType:
    pass

def test_maybe_parse_user_type():
    # Test single type
    assert maybe_parse_user_type(int) == [int]
    assert maybe_parse_user_type(NonIterableType) == [NonIterableType]

    # Test string input
    assert maybe_parse_user_type("int") == ["int"]
    assert maybe_parse_user_type(["str", int]) == ("str", int)

    # Test preserved types (Enum)
    assert maybe_parse_user_type(MockEnum) == [MockEnum]
    assert maybe_parse_user_type([MockEnum, int]) == (MockEnum, int)

    # Test nested iterables
    assert maybe_parse_user_type([[int], "str"]) == (int, "str")
    assert maybe_parse_user_type((int, [str, float])) == (int, str, float)

    # Test error case: non-type/non-string/non-iterable input
    with pytest.raises(TypeError) as excinfo:
        maybe_parse_user_type(123)
    assert "Type specifications must be types or strings" in str(excinfo.value)

    # Test error case: invalid nested element
    with pytest.raises(TypeError):
        maybe_parse_user_type([None]) # type: ignore (None is not a type/str/iterable in this context)

def test_invariant_exception_formatting():
    # Test InvariantException string representation
    error_codes = [lambda: "Error1", "Error2"]
    missing_fields = ("field1", "field2")
    exc = InvariantException(error_codes=error_codes, missing_fields=missing_fields, msg="Base error")
    
    error_str = str(exc)
    assert "Base error" in error_str
    assert "Error1, Error2" in error_str
    assert "field1, field2" in error_str
    assert exc.invariant_errors == ("Error1", "Error2")
    assert exc.missing_fields == ("field1", "function_name_or_val") # checking tuple structure
```


