####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_PMapValues___eq__():
    # Setup a mock PMap-like object that supports itervalues
    class MockPMap:
        def __init__(self, data):
            self.data = data
        def itervalues(self):
            return iter(self.data.values())

    m1 = MockPMap({'a': 1, 'b': 2})
    m2 = MockPMap({'a': 1, 'b': 2})
    m3 = MockPMap({'a': 3, 'b': 4})
    
    view1 = PMapValues(m1)
    view2 = PMapValues(m2)
    view3 = PMapValues(m3)

    # Test identity: should return True
    assert view1 == view1
    
    # Test inequality with different object (even if content is same)
    # Based on the implementation: if x is self: return True else: return False
    assert view1 != view2
    
    # Test inequality with different content
    assert view1 != view3

    # Test equality with other types (should return False per implementation)
    assert view1 != [1, 2]
    assert view1 != {'a': 1, 'b': 2}.values()
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap
    # Note: Assuming m() is a factory function that returns a PMap instance 
    # as implied by the docstrings in the provided code.
    from pyrsistent import m 
    from operator import add

    m1 = m(a=1, b=2)
    m_extra = m(c=3)

    # Test Case 1: Basic update (using rightmost value wins, similar to dict.update)
    # Using a lambda that mimics standard update behavior: lambda l, r: r
    m2 = m1.update_with(lambda l, r: r, m_extra, m(a=10))
    assert m2 == {'a': 10, 'b': 2, 'c': 3}

    # Test Case 2: Using a merge function (e.g., addition)
    # This tests the core logic of update_with where values are combined
    m3 = m1.update_with(add, m(a=2))
    assert m3 == {'a': 3, 'b': 2}

    # Test Case 3: Leftmost element wins (inverse behavior)
    # Testing the specific docstring example for reverse behavior
    m4 = m(a=1).update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m4 == {'a': 1}

    # Test Case 4: Multiple maps passed at once
    # Mappings are processed left to right
    m5 = m1.update_with(add, m(a=10, b=20), m(b=30, c=40))
    # a: 1 + 10 = 11
    # b: 2 + 20 + 30 = 52
    # c: 40 (new)
    assert m5 == {'a': 11, 'b': 52, 'c': 40}

    # Test Case 5: Update with a standard dict
    m6 = m1.update_with(add, {'a': 5, 'd': 9})
    assert m6 == {'a': 6, 'b': 2, 'd': 9}

    # Test Case 6: No changes when no new maps are provided (or empty maps)
    m7 = m1.update_with(add)
    assert m7 == m1
    
    m8 = m1.update_with(add, m())
    assert m8 == m1

    # Test Case 7: Ensure original PMap remains immutable (Persistent property)
    m9 = m(x=1)
    m9.update_with(add, m(x=2))
    assert m9 == {'x': 1}
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap
    # Note: Using 'm' as a factory function assuming it exists in the context 
    # (as per docstring examples). If not, we use PMap constructor with manual buckets.
    # For testing logic, we assume m(a=1) works as described.
    from pyrsistent import m
    from operator import add

    m1 = m(a=1, b=2)

    # Test 1: Standard update (rightmost takes precedence)
    # mimics the behavior of 'update' when using a lambda that returns right value
    m2 = m1.update_with(lambda l, r: r, m(a=2, c=3), {'a': 17, 'd': 35})
    expected_m2 = {'a': 17, 'b': 2, 'c': 3, 'd': 35}
    assert dict(m2.items()) == expected_m2

    # Test 2: Custom merge function (e.g., addition)
    m3 = m1.update_with(add, m(a=2))
    expected_m3 = {'a': 3, 'b': 2}
    assert dict(m3.items()) == expected_m3

    # Test 3: Leftmost takes precedence (inverse behavior)
    m4 = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    expected_m4 = {'a': 1, 'b': 2}
    assert dict(m4.items()) == expected_m4

    # Test 4: Multiple maps with complex merging
    # m5 starts with a=10, b=20
    # map_a adds 5 to existing keys or inserts new ones
    # map_b sets values or inserts new ones
    m5 = m(a=10, b=20)
    map_a = m(a=5, c=30) # a becomes 15, c is 30
    map_b = {'b': 25, 'd': 40} # b becomes 25, d is 40
    
    # We use add for map_a to merge, and lambda l, r: r for map_b to overwrite
    # Since update_with iterates through *maps sequentially:
    # Step 1 (map_a): a: 10+5=15, b: 20, c: 30
    # Step 2 (map_b): a: 15, b: 25, c: 30, d: 40
    m6 = m5.update_with(add, map_a, map_b)
    expected_m6 = {'a': 15, 'b': 25, 'c': 30, 'd': 40}
    assert dict(m6.items()) == expected_m6

    # Test 5: Empty updates should return a copy/same structure
    m7 = m1.update_with(add)
    assert m7 == m1

    # Test 6: Update with an empty map
    m8 = m1.update_with(add, m())
    assert m8 == m1

    # Test 7: Ensure original PMap is not mutated (Immutability check)
    m9 = m(x=1)
    m9.update_with(add, m(x=2))
    assert m9['x'] == 1
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a sample PMap
    data = {'a': 1, 'b': 2, 'c': 3}
    m = pmap(data)
    items_view = PMapItems(m)

    # Test case 1: Existing item (key and value match)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test case 2: Key exists but value is different
    assert ('a', 99) not in items_view

    # Test case 3: Key does not exist in the map
    assert ('z', 1) not in items_view

    # Test case 4: Input is not a tuple/iterable (should return False via exception handling)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test case 5: Input is a tuple of wrong length
    assert ('a', 1, 'extra') not in items_view

    # Test case 6: Empty map behavior
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapValues():
    # Test initialization with a PMap
    data = {'a': 1, 'b': 2}
    p_map = pmap(data)
    values_view = PMapValues(p_map)
    
    assert len(values_view) == 2
    assert list(values_view) == [1, 2] or list(values_view) == [2, 1]
    assert 1 in values_view
    assert 3 not in values_view

    # Test initialization with a standard dict (should convert to PMap)
    dict_data = {'x': 10, 'y': 20}
    values_view_from_dict = PMapValues(dict_data)
    assert len(values_view_from_dict) == 2
    assert 10 in values_view_from_dict

    # Test initialization with an invalid type (not a Mapping)
    with pytest.raises(TypeError, match="PViewMap requires a Mapping object"):
        PMapValues([1, 2, 3])

    # Test immutability of the view via __setattr__
    with pytest.raises(TypeError, match="<class 'pyrsistent.PMapValues'> is immutable"):
        values_view.new_attr = "error"

    # Test __reversed__ restriction
    with pytest.raises(TypeError, match="Persistent maps are not reversible"):
        reversed(values_view)

    # Test string representation
    expected_str = f"pmap_values({list(values_view)})"
    assert str(values_view) == expected_str
    assert repr(values_view) == expected_str

    # Test equality behavior (as defined in the class: only True if same object)
    assert values_view == values_view
    assert values_view != PMapValues(p_map)
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_PMapValues___eq__():
    # Setup a mock PMap-like object that implements itervalues
    class MockPMap:
        def __init__(self, data):
            self.data = data
        def itervalues(self):
            return iter(self.data.values())

    m1_data = {'a': 1, 'b': 2}
    m2_data = {'c': 1, 'd': 2}
    
    m1 = MockPMap(m1_data)
    m2 = MockPMap(m2_data)
    
    values1 = PMapValues(m1)
    values2 = PMapValues(m2)

    # Test identity equality (should be True)
    assert values1 == values1
    
    # Test inequality for different objects even if they represent same values
    # Based on the implementation: if x is self: return True else: return False
    assert values1 != values2
    
    # Test inequality with other types
    assert values1 != [1, 2]
    assert values1 != None
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup: Create initial maps using a mock-like approach 
    # Since we don't have the factory 'm', we use PMap directly with required internal structure.
    # Note: This assumes the existence of PVector/pvector as implied by the code.
    from pyrsistent import pmap, pvector

    # Initial map m1: {'a': 1, 'b': 2}
    m1 = pmap({'a': 1, 'b': 2})
    
    # Test Case 1: Basic update (rightmost value wins)
    # m2 = m1.update_with(lambda l, r: r, m(a=2, c=3), {'a': 17, 'd': 35})
    m_extra1 = pmap({'a': 2, 'c': 3})
    m_extra2 = pmap({'a': 17, 'd': 35})
    result1 = m1.update_with(lambda l, r: r, m_extra1, m_extra2)
    assert result1 == {'a': 17, 'b': 2, 'c': 3, 'd': 35}

    # Test Case 2: Custom merge function (leftmost value wins)
    # m1.update_with(lambda l, r: l, m(a=2), {'a':3}) -> {'a': 1}
    m_extra3 = pmap({'a': 2})
    m_extra4 = pmap({'a': 3})
    result2 = m1.update_with(lambda l, r: l, m_extra3, m_extra4)
    assert result2['a'] == 1
    assert result2['b'] == 2

    # Test Case 3: Using operator.add as merge function
    from operator import add
    m_increment = pmap({'a': 2})
    result3 = m1.update_with(add, m_increment)
    assert result3['a'] == 3
    assert result3['b'] == 2

    # Test Case 4: Updating with an empty map should not change the original map
    result4 = m1.update_with(add, pmap({}))
    assert result4 == m1
    assert result4 is not m1 # If a new evolver was used, it might be a different object but same content

    # Test Case 5: Ensure the original map remains immutable
    m1_copy = pmap({'a': 1, 'b': 2})
    m1_copy.update_with(add, pmap({'a': 10}))
    assert m1_copy['a'] == 1

    # Test Case 6: Multiple maps in sequence
    # m1 + m2 + m3 where all use 'r' (rightmost)
    m3 = pmap({'b': 99, 'e': 5})
    result5 = m1.update_with(lambda l, r: r, m_extra1, m3)
    # Step 1: m1 + m_extra1 -> {'a': 2, 'b': 2, 'c': 3}
    # Step 2: result + m3 -> {'a': 2, 'b': 99, 'c': 3, 'e': 5}
    assert result5 == {'a': 2, 'b': 99, 'c': 3, 'e': 5}

    # Test Case 7: Key exists in evolver but not in the incoming map (should stay same)
    # This is handled by the logic: `update_fn(evolver[key], value) if key in evolver else value`
    m_new_key = pmap({'z': 100})
    result6 = m1.update_with(add, m_new_key)
    assert result6['z'] == 100
    assert result6['a'] == 1
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test valid items present in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test items with existing key but wrong value
    assert ('a', 99) not in items_view
    
    # Test items with non-existent key
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test invalid argument types (not a tuple/iterable of length 2)
    assert 'a' not in items_view
    assert [('a', 1)] not in items_view
    assert 123 not in items_view
    assert None not in items_view
    assert (1, 2, 3) not in items_view

    # Test edge case: Empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test valid items present in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test valid key with wrong value
    assert ('a', 99) not in items_view

    # Test invalid key
    assert ('z', 1) not in items_view

    # Test item format that is not a tuple/pair (should return False via exception handling)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [] not in items_view

    # Test item format that is a tuple but length is incorrect
    assert ('a', 1, 'extra') not in items_view
```


# LLM-generated content at query #10
#--------------------------

```python
def test_PMapValues___eq__():
    # Setup a mock PMap object since we don't have the full class definition
    class MockPMap:
        def __init__(self, data):
            self.data = data
        def itervalues(self):
            return iter(self.data.values())

    data = {'a': 1, 'b': 2}
    mock_map = MockPMap(data)
    view = PMapValues(mock_map)
    
    # Test identity equality (x is self)
    assert view == view
    
    # Test inequality with different object (even if content matches)
    other_view = PMapValues(mock_map)
    assert view != other_view
    
    # Test inequality with a list/different type
    assert view != [1, 2]
    
    # Test inequality with None
    assert view != None
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test valid item exists
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    
    # Test valid key exists but with wrong value
    assert ('a', 99) not in items_view
    
    # Test key does not exist in map
    assert ('z', 1) not in items_view
    
    # Test non-iterable argument (should return False via exception handling)
    assert 123 not in items_view
    
    # Test iterable that is not a pair (tuple of length != 2)
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view
    
    # Test empty input/non-pair container
    assert [] not in items_view
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_PMapItems___contains__():
    # Setup a mock mapping that behaves like PMap
    # We need to satisfy the requirements of PMapItems.__contains__:
    # 1. It accesses self._map[k]
    # 2. It checks k in self._map
    class MockPMap(dict):
        def iteritems(self):
            return iter(self.items())

    data = {'a': 1, 'b': 2, 'c': 3}
    mock_map = MockPMap(data)
    view = PMapItems(mock_map)

    # Case 1: Valid item exists in the map
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Case 2: Key exists but value is different
    assert ('a', 99) not in view
    assert ('c', 0) not in view

    # Case 3: Key does not exist in the map
    assert ('z', 1) not in view

    # Case 4: Argument is not a tuple/iterable (triggers Exception in try block)
    assert 1 not in view
    assert "not_a_tuple" not in view
    assert None not in view

    # Case 5: Argument is an iterable but not a valid (k, v) pair for the map
    assert ('a',) not in view  # Too short
    assert [('a', 1)] not in view # Not a tuple/pair compatible with unpacking logic
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test valid items present in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test valid key but incorrect value
    assert ('a', 99) not in items_view
    
    # Test key not present in the map
    assert ('z', 1) not in items_view
    
    # Test non-iterable input (should return False via try-except block)
    assert 123 not in items_view
    assert None not in items_view
    
    # Test iterable that is not a pair (cannot be unpacked)
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view
    
    # Test non-tuple/list input that might raise error on unpacking
    assert "not a tuple" not in items_view
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_PMap___eq__():
    # We need a mock or real instance of PMap. 
    # Since we cannot import, we assume 'm' is the factory function available in the scope.
    # For the sake of this test, I will use the logic provided in the snippet to construct instances.
    
    # Mocking the required dependencies for a minimal working PMap instance
    # In a real scenario, these would be imported from pyrsistent.
    from types import MappingProxyType
    from collections.abc import Mapping

    # Setup: Create base maps using the provided logic/factory
    # Assuming 'm' is available as per docstrings
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m3 = m(a=1, b=3)
    m4 = m(a=1, b=2, c=3)
    
    dict_version = {'a': 1, 'b': 2}
    other_mapping = MappingProxyType({'a': 1, 'bo': 2}) # Not equal due to key mismatch
    
    # 1. Test Identity: self is other
    assert m1 == m1
    
    # 2. Test Equality with same content PMap
    assert m1 == m2
    
    # 3. Test Inequality with different content PMap
    assert m1 != m3
    
    # 4. Test Equality with standard dict
    assert m1 == {'a': 1, 'b': 2}
    
    # 5. Test Inequality with standard dict (different values)
    assert m1 != {'a': 1, 'b': 3}
    
    # 6. Test Equality with standard dict (different keys)
    assert m1 != {'a': 1, 'c': 2}

    # 7. Test Inequality with different size
    assert m1 != m4
    
    # 8. Test Equality with other Mapping types (e.g., MappingProxyType)
    # The implementation uses dict(self.iteritems()) == other.items() for non-PMap mappings
    m_proxy = MappingProxyType({'a': 1, 'b': 2})
    assert m1 == m_proxy

    # 9. Test Inequality with incompatible types (NotImplemented handling)
    # Comparing PMap to an integer should return False via __ne__ or NotImplemented logic
    assert m1 != 123
    
    # 10. Test Case: Different bucket structure but same content
    # This tests the fallback 'dict(self.iteritems()) == dict(other.iteritems())'
    # We create a manual PMap with different bucket distribution if possible, 
    # but since we use factory 'm', we rely on the fact that __eq__ handles structural differences.
    
    # 11. Test against an empty map
    empty_map = m()
    assert m1 != empty_map
    assert empty_map == {}

    # 12. Test with different hash-based collision potential
    # If two maps have same elements but different internal bucket layouts (simulated by manual construction)
    # we verify the dict comparison logic.
    class MockMapping(Mapping):
        def __getitem__(self, key): return dict_version[key]
        def __len__(self): return len(dict_version)
        def __iter__(self): return iter(dict_version)

    assert m1 == MockMapping()
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_PMapItems___contains__():
    # Setup a mock PMap-like object that behaves like the expected dependency
    # Since we cannot import pmap, we create a class that mimics the interface
    class MockPMap:
        def __init__(self, data):
            self.data = data
        def __contains__(self, key):
            return key in self.data
        def iteritems(self):
            return iter(self.data.items())

    # Test Data
    mapping_data = {'a': 1, 'b': 2, 'c': 3}
    mock_map = MockPMap(mapping_data)
    view = PMapItems(mock_map)

    # Case 1: Existing key-value pair
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Case 2: Existing key, but wrong value
    assert ('a', 99) not in view

    # Case 3: Non-existing key
    assert ('z', 1) not in view

    # Case 4: Input is not a tuple/pair (triggers the Exception block)
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Case 5: Input is a tuple but has wrong length (triggers the Exception block)
    assert ('a', 1, 'extra') not in view
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap
    # Using a mock or real instance depending on environment availability.
    # Assuming m is the factory function available in the context of PMap.
    from pyrsistent import m
    from operator import add

    m1 = m(a=1, b=2)
    m2 = m(a=2, c=3)
    m3 = {'a': 17, 'd': 35}

    # Test Case 1: Standard update (rightmost value wins)
    # This mimics the behavior of PMap.update() which uses lambda l, r: r
    result_update = m1.update(m2, m3)
    expected_update = {'a': 17, 'b': 2, 'c': 3, 'd': 35}
    assert result_update == expected_update
    # Verify immutability of original
    assert m1 == {'a': 1, 'b': 2}

    # Test Case 2: Using a custom merge function (e.g., addition)
    # Key 'a' exists in m1 and m2, so new value = m1['a'] + m2['a'] = 1 + 2 = 3
    result_add = m1.update_with(add, m2)
    expected_add = {'a': 3, 'b': 2, 'c': 3}
    assert result_add == expected_add

    # Test Case 3: Using a custom merge function that keeps the leftmost element
    # Key 'a' exists in m1 and m2, so new value = m1['a'] = 1
    result_leftmost = m1.update_with(lambda l, r: l, m2, m3)
    expected_leftmost = {'a': 1, 'b': 2, 'c': 3, 'd': 35}
    assert result_leftmost == expected_leftmost

    # Test Case 4: Update with multiple mappings in sequence
    # m1: {a:1, b:2}, m2: {a:2}, m3: {a:3} -> applying add sequentially
    # Step 1 (m1 + m2): a becomes 1+2=3. Result: {a:3, b:2}
    # Step 2 (prev + m3): a becomes 3+3=6. Result: {a:6, b:2}
    result_chain = m1.update_with(add, m2, m3)
    assert result_chain['a'] == 6

    # Test Case 5: Update with an empty mapping
    result_empty = m1.update_with(add, m())
    assert result_empty == m1

    # Test Case 6: Ensure it works with standard dicts as input via items()
    result_dict_input = m1.update_with(add, {'b': 10})
    assert result_dict_input['b'] == 12
    assert result_dict_input['a'] == 1

    # Test Case 7: Key not present in evolver (should just insert the value)
    result_new_key = m1.update_with(add, {'z': 100})
    assert result_new_key['z'] == 100
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test case: Exact match with tuple (key, value) exists
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test case: Key exists but value is incorrect
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test case: Key does not exist in the map
    assert ('d', 1) not in items_view
    assert ('z', 3) not in items_view

    # Test case: Input is a tuple with wrong length (not an item pair)
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view

    # Test case: Input is not a tuple/iterable (triggers exception handling in __contains__)
    assert 1 not in items_view
    assert None not in items_view
    assert "not_a_tuple" not in items_view

    # Test case: Input is an empty tuple
    assert () not in items_view
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test valid item exists
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test key exists but value is different
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test key does not exist
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test non-iterable input (should return False via try-except)
    assert 123 not in items_view
    assert None not in items_view

    # Test iterable that is not a pair (should return False via try-except/unpack error)
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view

    # Test empty map behavior
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test valid item exists in view
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test key exists but value is different
    assert ('a', 99) not in items_view

    # Test key does not exist in map
    assert ('z', 1) not in items_view

    # Test item is not a tuple/pair (should return False via Exception handling)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [('a', 1)] not in items_view  # List of tuple is iterable but unpacks differently/fails logic

    # Test empty view behavior with valid key/value pair
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view

    # Test edge case: item is a tuple with wrong length
    assert ('a', 1, 'extra') not in items_view
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_PMap___eq__():
    # Mocking dependencies required for PMap instantiation/behavior
    # Since we cannot import, we assume a setup where m() or PMap can be instantiated.
    # For the purpose of this test, we will use a mock-like structure if necessary, 
    # but following the prompt's logic to test the provided __eq__ implementation.

    # Setup: Creating instances for testing
    # We assume 'm' is the factory function mentioned in docstrings.
    # For testing purposes, we simulate the internal state of PMap.
    
    def create_mock_pmap(data_dict):
        # This simulates a valid PMap instance with required internals
        # In a real test environment, you would use: m1 = m(a=1)
        # Here we manually construct to ensure __eq__ logic is isolated.
        buckets = [None] * 10
        for k, v in data_dict.items():
            idx = hash(k) % 10
            if buckets[idx] is None:
                buckets[idx] = []
            buckets[idx].append((k, v))
        
        instance = PMap(len(data_dict), buckets)
        # Simulate cached hash if needed for the specific branch in __eq__
        instance._cached_hash = hash(frozenset(data_dict.items()))
        return instance

    m1 = create_mock_pmap({'a': 1, 'b': 2})
    m2 = create_mock_pmap({'a': 1, 'b': 2})
    m3 = create_mock_pmap({'a': 1, 'c': 3})
    m4 = create_mock_pmap({'b': 2, 'c': 3})
    dict_m1 = {'a': 1, 'b': 2}
    other_type = [('a', 1), ('b', 2)]

    # Test Case 1: Identity (self is other)
    assert m1 == m1

    # Test Case 2: Equality with another PMap of same content
    assert m1 == m2

    # Test Case 3: Inequality with different PMap content
    assert m1 != m3
    assert m1 != m4

    # Test Case 4: Equality with a standard dict (same content)
    assert m1 == dict_m1

    # Test Case 5: Inequality with a standard dict (different content)
    assert m1 != {'a': 1, 'b': 3}

    # Test Case 6: Inequality with different types (Mapping vs non-Mapping)
    # Note: PMap inherits from Mapping. list of tuples is not a Mapping.
    with pytest.raises(TypeError):
        # The __eq__ implementation returns NotImplemented for non-Mapping, 
        # which usually results in a TypeError or False when compared to primitives.
        assert m1 == other_type

    # Test Case 7: Length mismatch
    assert m1 != create_mock_pmap({'a': 1, 'b': 2, 'c': 3})

    # Test Case 8: Testing the _cached_hash optimization branch
    # We force a scenario where hashes differ but content is same (if possible)
    # or specifically test that if hashes are different, it checks contents.
    m2._cached_hash = hash(999) 
    assert m1 == m2 # Should fall back to dict comparison
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)

    # Test valid items present in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test items with correct key but wrong value
    assert ('a', 2) not in items_view
    assert ('b', 99) not in items_view

    # Test keys not present in the map
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test non-iterable argument (should return False via try-except)
    assert 123 not in items_view
    assert None not in items_view

    # Test iterable that is not a pair (e.g., single element tuple)
    assert ('a',) not in items_view

    # Test empty map behavior
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test successful containment of existing key-value pairs
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test non-existent key-value pair (correct key, wrong value)
    assert ('a', 99) not in items_view

    # Test non-existent key (key does not exist in map)
    assert ('z', 1) not in items_view

    # Test invalid input format (not a tuple/iterable) - should return False via try-except
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test invalid tuple length/structure (e.g., single element tuple)
    assert ('a',) not in items_view
    assert (1, 2, 3) not in items_view
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a PMap for testing
    data = {'a': 1, 'b': 2, 'c': 3}
    m = pmap(data)
    items_view = PMapItems(m)

    # Test case 1: Existing key-value pair
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test case 2: Existing key with wrong value
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test case 3: Non-existent key
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test case 4: Input is not a tuple (should return False via try/except)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [('a', 1)] not in items_view  # List is not unpackable to k, v

    # Test case 5: Input is a tuple of wrong length
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view

    # Test case 6: Empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial maps using a mock-like approach since we cannot instantiate PMap directly 
    # as per documentation (requires factory functions like m() or pmap()).
    # However, for the purpose of unit testing the logic of update_with:
    from operator import add

    # We assume 'm' is the factory function available in the scope where PMap is defined.
    # Since we cannot call external factories not provided in the snippet, 
    # we will use the logic that simulates the behavior described in the docstrings.
    
    # Mocking the structure of a PMap for testing update_with logic.
    # We need a way to create a functional PMap instance. 
    # In a real scenario, you'd import 'm' from the module.
    try:
        from . import m
    except ImportError:
        # Fallback for standalone test execution if 'm' is not in the same package
        # This assumes 'm' and 'PMap' are available via the environment setup.
        pass

    # Test Case 1: Basic update_with using 'add' as merge function (as seen in docstring)
    # m1 = m(a=1, b=2)
    # Result should be {'a': 3, 'b': 2} when updating with m(a=2)
    m1 = m(a=1, b=2)
    m_update = m(a=2)
    result1 = m1.update_with(add, m_update)
    assert result1 == {'a': 3, 'b': 2}
    assert m1 == {'a': 1, 'b': 2} # Ensure original is immutable

    # Test Case 2: Leftmost element preference (as seen in docstring)
    # m1 = m(a=1)
    # Update with m(a=2) and {'a': 3} should result in {'a': 1}
    m2 = m(a=1)
    m_update2 = m(a=2)
    m_dict_update = {'a': 3}
    result2 = m2.update_with(lambda l, r: l, m_update2, m_dict_update)
    assert result2 == {'a': 1}

    # Test Case 3: Multiple maps update
    # m1 = m(a=1, b=2)
    # Update with m(c=3) and m(d=4)
    m3 = m(a=1, b=2)
    m_extra1 = m(c=3)
    m_extra2 = m(d=4)
    result3 = m3.update_with(lambda l, r: r, m_extra1, m_extra2)
    assert result3 == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    # Test Case 4: Testing with dicts as inputs (since PMap.update handles Mappings)
    m4 = m(a=10)
    dict_input = {'a': 5, 'b': 20}
    # Using add: existing 10 + incoming 5 = 15. New key 'b' = 20.
    result4 = m4.update_with(add, dict_input)
    assert result4 == {'a': 15, 'b': 20}

    # Test Case 5: Key exists in multiple updates, verifying right-to-left precedence for the merge function
    # update_with iterates through maps from left to right.
    # m1(a=1) + m2(a=2) + m3(a=3) with add -> (1+2)+3 = 6
    m5 = m(a=1)
    m6 = m(a=2)
    m7 = m(a=3)
    result5 = m5.update_with(add, m6, m7)
    assert result5['a'] == 6

    # Test Case 6: Empty update
    m8 = m(a=1)
    result6 = m8.update_with(add)
    assert result6 == {'a': 1}
    assert result6 is m8 # Should return same object if no changes (based on evolver logic)

    # Test Case 7: Verifying that it handles keys not present in the current map correctly
    m9 = m(a=1)
    m_new = m(b=5)
    result7 = m9.update_with(lambda l, r: l + r, m_new)
    assert result7 == {'a': 1, 'b': 5} # 'b' is not in evolver yet, so it just takes value
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test valid items present in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test valid item structure but wrong value
    assert ('a', 99) not in items_view
    
    # Test valid item structure but wrong key
    assert ('z', 1) not in items_view
    
    # Test non-tuple/non-iterable input (should return False via exception handling)
    assert 1 not in items_view
    assert 'a' not in items_view
    
    # Test tuple with wrong length (should return False via exception handling)
    assert ('a', 1, 'extra') not in items_view
    
    # Test empty mapping
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest

def test_PMap___eq__():
    # Mocking necessary components since we don't have the full environment
    # We assume 'm' is a factory function that returns PMap instances
    # and that PMap follows the logic provided in the snippet.
    
    # Setup: Create a base PMap with some data
    # Note: In a real test, you would use the actual factory functions (e.g., from pyrsistent)
    m1 = m(a=1, b=2, c=3)
    
    # 1. Identity test: A map must equal itself
    assert m1 == m1
    
    # 2. Equality with same content (different instance)
    m2 = m(a=1, b=2, c=3)
    assert m1 == m2
    
    # 3. Equality with dict containing same keys/values
    dict_content = {'a': 1, 'b': 2, 'c': 3}
    assert m1 == dict_content
    
    # 4. Equality with different content (different values)
    m3 = m(a=1, b=99, c=3)
    assert m1 != m3
    
    # 5. Equality with different keys
    m4 = m(a=1, b=2, d=4)
    assert m1 != m4
    
    # 6. Inequality with different sizes
    m5 = m(a=1, b=2)
    assert m1 != m5
    
    # 7. Equality with another PMap instance via bucket comparison (if applicable)
    # This tests the branch: if self._buckets == other._buckets: return True
    # We simulate this by creating a structure that shares the same underlying buckets
    m6 = m(a=1, b=2, c=3) 
    # Depending on implementation of 'm', m1 and m6 might share buckets if created via set()
    if hasattr(m1, '_buckets') and hasattr(m6, '_buckets'):
        assert m1 == m6

    # 8. Inequality with non-mapping types (should return NotImplemented/False)
    assert m1 != [1, 2, 3]
    assert m1 != "not a map"
    assert m1 != None

    # 9. Testing __ne__ (which is Mapping.__ne__)
    assert m1 != m(a=5)
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial maps
    m1 = m(a=1, b=2)
    m2 = m(a=2, c=3)
    m3 = {'a': 17, 'd': 35}

    # Test case 1: Standard update (rightmost wins)
    # This is equivalent to the .update() method behavior
    result_update = m1.update(m2, m3)
    assert result_update == {'a': 17, 'b': 2, 'c': 3, 'd': 35}

    # Test case 2: update_with using an additive function (merging values)
    from operator import add
    result_add = m1.update_with(add, m2)
    # a: 1 + 2 = 3, b: 2 (remains), c: 3 (new)
    assert result_add == {'a': 3, 'b': 2, 'c': 3}

    # Test case 3: update_with using a function that keeps the leftmost value
    # m1 has a=1, m2 has a=2. Leftmost (m1) wins.
    result_leftmost = m1.update_with(lambda l, r: l, m2)
    assert result_leftmost == {'a': 1, 'b': 2, 'c': 3}

    # Test case 4: update_with with multiple maps and custom logic
    # We will use a function that takes the maximum of existing and new values
    result_max = m1.update_with(max, m2, m3)
    # key 'a': max(1, 2) -> 2; then max(2, 17) -> 17
    # key 'b': 2 (no change)
    # key 'c': 3 (from m2)
    # key 'd': 35 (from m3)
    assert result_max == {'a': 17, 'b': 2, 'c': 3, 'd': 35}

    # Test case 5: update_with with a map that has no overlapping keys
    m4 = m(e=5)
    result_no_overlap = m1.update_with(add, m4)
    assert result_no_overlap == {'a': 1, 'b': 2, 'e': 5}

    # Test case 6: update_with on an empty map
    empty_m = m()
    result_from_empty = empty_m.update_with(add, m1)
    assert result_from_empty == {'a': 1, 'b': 2}

    # Test case 7: Ensure the original PMap remains immutable (Persistence check)
    m1_original = m(a=1, b=2)
    m1_original.update_with(add, m2)
    assert m1_original == {'a': 1, 'b': 2}

    # Test case 8: Verifying the logic for keys present in evolver but not in incoming map
    # The implementation says: update_fn(evolver[key], value) if key in evolver else value
    # If we use a function that fails when 'value' is passed alone, it would crash.
    # We test the "else value" branch by ensuring keys only in the new map are inserted as-is.
    m_new_only = m(z=100)
    result_branch_check = m1.update_with(add, m_new_only)
    assert result_branch_check['z'] == 100
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a persistent map for testing
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)

    # Test case 1: Existing item (key and value match)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test case 2: Key exists but value is different
    assert ('a', 99) not in items_view

    # Test case 3: Key does not exist in the map
    assert ('z', 1) not in items_view

    # Test case 4: Input is a tuple but doesn't match any item
    assert ('d', 4) not in items_view

    # Test case 5: Input is not a tuple (should return False via try-except)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test case 6: Input is an empty tuple
    assert () not in items_view

    # Test case 7: Input is a tuple of wrong length
    assert ('a', 1, 'extra') not in items_view
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test with valid tuple (key, value) that exists in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test with valid tuple (key, value) where key exists but value is wrong
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test with key that does not exist in the map
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test with non-tuple/non-iterable argument (should return False via try-except)
    assert "not a tuple" not in items_view
    assert 123 not in items_view
    assert None not in items_view

    # Test with an iterable that is not a pair (e.g., a single element tuple)
    assert ('a',) not in items_view

    # Test with an empty tuple
    assert () not in items_view
```


# LLM-generated content at query #30
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test valid item exists in view
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    
    # Test key exists but value is different
    assert ('a', 99) not in items_view
    
    # Test key does not exist in map
    assert ('z', 1) not in items_view
    
    # Test non-iterable input (should return False via Exception handling)
    assert 123 not in items_view
    assert None not in items_view
    
    # Test tuple with wrong length (not a pair)
    assert ('a', 1, 'extra') not in items_view
    
    # Test empty map behavior
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_PMapView___setattr__():
    # Setup a mock PMap (assuming pmap is available in the scope)
    m = pmap({'a': 1, 'b': 2})
    view = PMapView(m)
    
    # Verify that setting an attribute raises TypeError
    with pytest.raises(TypeError) as excinfo:
        view.new_attr = "value"
    
    assert "is immutable" in str(excinfo.value)
    
    # Verify that setting a standard attribute also fails
    with pytest.raises(TypeError):
        view._map = pmap({'c': 3})
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_PMapView___setattr__():
    # Create a dummy mapping to satisfy the constructor requirements
    class MockMapping(dict):
        pass
    
    m = MockMapping({'a': 1})
    view = PMapView(m)
    
    # Attempting to set an attribute should raise TypeError
    with pytest.raises(TypeError) as excinfo:
        view.new_attr = "value"
    
    assert "<class '...PMapView'>" in str(excinfo.value) or "is immutable" in str(excinfo.value)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Mocking dependency requirements for the test environment
    # Since we cannot import, we assume pmap/m factory exists as per docstrings.
    # We use a helper to create PMap instances if 'm' is not in scope.
    from pyrsistent import m

    # 1. Test basic update_with with addition (mimicking standard update)
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    # Using a lambda that adds values
    result = m1.update_with(lambda l, r: l + r, m2)
    assert result == {'a': 1, 'b': 2, 'c': 3}

    # 2. Test update_with with logic to merge existing keys
    m3 = m(a=10, b=20)
    m4 = m(a=5, c=30)
    # Logic: sum the values if key exists, otherwise keep new value
    result2 = m3.update_with(lambda l, r: l + r, m4)
    assert result2['a'] == 15
    assert result2['b'] == 20
    assert result2['c'] == 30

    # 3. Test the "Reverse behaviour" mentioned in docstring
    # Keep the leftmost element instead of the rightmost
    m5 = m(a=1)
    m6 = m(a=2)
    m7 = {'a': 3}
    result3 = m5.update_with(lambda l, r: l, m6, m7)
    assert result3['a'] == 1

    # 4. Test with multiple mappings at once
    m8 = m(a=1, b=1)
    m9 = m(b=2, c=2)
    m10 = {'c': 3, 'd': 4}
    # Logic: multiply existing by new
    result4 = m8.update_with(lambda l, r: l * r, m9, m10)
    # a: remains 1 (not in m9/m10)
    # b: 1 * 2 = 2
    # c: 2 * 3 = 6
    # d: comes from m10 = 4
    assert result4 == {'a': 1, 'b': 2, 'c': 6, 'd': 4}

    # 5. Test with an empty map
    m_empty = m()
    m_full = m(x=100)
    result5 = m_empty.update_with(lambda l, r: r, m_full)
    assert result5 == {'x': 100}

    # 6. Test with an update function that handles missing keys via the logic in code:
    # `update_fn(evolver[key], value) if key in evolver else value`
    m_base = m(a=1)
    m_new = m(a=5, b=10)
    # The function is only called for existing keys. For new keys, it just takes 'value'.
    result6 = m_base.update_with(lambda l, r: l + r, m_new)
    assert result6['a'] == 6 # 1 + 5
    assert result6['b'] == 10 # just 10
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapValues___str__():
    # Test with empty map
    m1 = pmap({})
    v1 = PMapValues(m1)
    assert str(v1) == "pmap_values([])"
    assert repr(v1) == "pmap_values([])"

    # Test with populated map
    m2 = pmap({'a': 1, 'b': 2, 'c': 3})
    v2 = PMapValues(m2)
    # Note: pmap order is insertion order
    assert str(v2) == "pmap_values([1, 2, 3])"
    assert repr(v2) == "pmap_values([1, 2, 3])"

    # Test with different types of values
    m3 = pmap({'x': 'hello', 'y': None})
    v3 = PMapValues(m3)
    assert str(v3) == "pmap_values(['hello', None])"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapValues___str__():
    # Setup a PMap with specific values
    data = {'a': 1, 'b': 2, 'c': 3}
    m = pmap(data)
    values_view = PMapValues(m)
    
    # The __str__ implementation uses list(iter(self)) inside f"pmap_values(...)"
    # Since pmap preserves order in modern Python/pyrsistent, we expect the list of values
    expected_str = "pmap_values([1, 2, 3])"
    
    assert str(values_view) == expected_str
    
    # Test with an empty PMap
    empty_view = PMapValues(pmap({}))
    assert str(empty_view) == "pmap_values([])"
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_PMapItems___contains__():
    # Setup a mock PMap-like object that behaves like a PMap for testing purposes
    # Since we cannot rely on the actual PMap implementation being present in the environment,
    # we create a minimal compatible object.
    class MockPMap:
        def __init__(self, data):
            self._data = data
        def __contains__(self, key):
            return key in self._data
        def __getitem__(self, key):
            return self._data[key]
        def iteritems(self):
            return iter(self._data.items())

    data = {'a': 1, 'b': 2, 'c': 3}
    mock_map = MockPMap(data)
    view = PMapItems(mock_map)

    # Test case 1: Existing key-value pair
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test case 2: Existing key, but wrong value
    assert ('a', 99) not in view
    assert ('b', 0) not in view

    # Test case 3: Non-existing key
    assert ('z', 1) not in view
    assert ('d', 3) not in view

    # Test case 4: Input is not a tuple (should return False via exception handling)
    assert 'a' not in view
    assert 1 not in view
    assert None not in view
    assert [('a', 1)] not in view  # List is not unpackable as k, v

    # Test case 5: Input is a tuple with wrong length (should return False via exception handling)
    assert ('a', 1, 'extra') not in view
    assert ('a',) not in view

    # Test case 6: Empty mapping
    empty_view = PMapItems(MockPMap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test valid items present in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test items with correct key but wrong value
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test keys that do not exist in the map
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test non-iterable input (should return False via try/except)
    assert 123 not in items_view
    assert None not in items_view

    # Test iterable that is not a tuple of length 2 (should return False via try/except)
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view
    assert [] not in items_view

    # Test edge case: empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a persistent map
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)

    # Test successful contains with existing (key, value) pair
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test contains with non-existent key but correct value
    assert ('d', 1) not in items_view

    # Test contains with existing key but incorrect value
    assert ('a', 99) not in items_view

    # Test contains with an item that is not a tuple/pair (should return False via Exception catch)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test contains with a tuple of wrong length
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view

    # Test contains with an unhashable/invalid type that might trigger error in unpacking
    assert [('a', 1)] not in items_view
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a sample PMap
    data = {'a': 1, 'b': 2, 'c': 3}
    m = pmap(data)
    items_view = PMapItems(m)

    # Case 1: Valid item exists in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Case 2: Key exists but value is different
    assert ('a', 99) not in items_view

    # Case 3: Key does not exist in the map
    assert ('z', 1) not in items_view

    # Case 4: Input is not a tuple (should return False via Exception handling)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Case 5: Input is a tuple with wrong length (should return False via Exception handling)
    assert ('a', 1, 'extra') not in items_view
    assert () not in items_view

    # Case 6: Input is an unhashable type that cannot be unpacked
    assert [('a', 1)] not in items_view
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial map
    m1 = m(a=1, b=2)
    
    # Test Case 1: Basic update (rightmost value wins)
    # Using a dict as the second argument to simulate another mapping
    m2 = m1.update_with(lambda l, r: r, m(a=10, c=3), {'b': 20})
    assert m2 == {'a': 10, 'b': 20, 'c': 3}
    # Verify immutability of original
    assert m1 == {'a': 1, 'b': 2}

    # Test Case 2: Custom merge function (addition)
    from operator import add
    m3 = m1.update_with(add, m(a=5), {'b': 10})
    assert m3 == {'a': 6, 'b': 12}

    # Test Case 3: Custom merge function (keep leftmost)
    # When the key exists in both, it picks the value from the original/left side
    m4 = m1.update_with(lambda l, r: l, m(a=99), {'b': 99})
    assert m4 == {'a': 1, 'b': 2}

    # Test Case 4: Multiple maps in sequence
    # Order should be: m1 -> map_a -> map_b
    map_a = m(a=2, c=3)
    map_b = {'d': 4}
    m5 = m1.update_with(lambda l, r: r, map_a, map_b)
    assert m5 == {'a': 2, 'b': 2, 'c': 3, 'd': 4}

    # Test Case 5: Update with no changes (empty maps)
    m6 = m1.update_with(add, m(), {})
    assert m6 == m1

    # Test Case 6: Update where keys do not exist in original (simple insertion)
    m7 = m1.update_with(lambda l, r: r, m(z=100))
    assert m7['z'] == 100
    assert m7['a'] == 1

    # Test Case 7: Verifying that it handles the logic of "key in evolver" correctly
    # If key is not in evolver, it just sets the value (no lambda called)
    m8 = m1.update_with(lambda l, r: l + r, m(new_key=5))
    assert m8['new_key'] == 5
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a persistent map for testing
    data = {'a': 1, 'b': 2, 'c': 3}
    m = pmap(data)
    items_view = PMapItems(m)

    # Test Case 1: Valid item exists in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test Case 2: Key exists but value is different
    assert ('a', 99) not in items_view

    # Test Case 3: Key does not exist in the map
    assert ('z', 1) not in items_view

    # Test Case 4: Input is not a tuple/iterable (should return False via Exception catch)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test Case 5: Input is an iterable but not a pair of correct value
    assert ('a', 1, 'extra') not in items_view
    
    # Test Case 6: Empty map behavior
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {1: 'a', 2: 'b', 3: 'c'}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test valid items present in the map
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view

    # Test items with correct key but incorrect value
    assert (1, 'z') not in items_view
    assert (2, 'not_b') not in items_view

    # Test keys that do not exist in the map
    assert (4, 'd') not in items_view
    assert (0, 'a') not in items_view

    # Test non-iterable arguments (should return False via try-except)
    assert 1 not in items_view
    assert None not in items_view
    assert "not a tuple" not in items_view

    # Test tuples of incorrect length (should return False via try-except/logic)
    assert (1,) not in items_view
    assert (1, 'a', 'extra') not in items_view

    # Test empty map case
    empty_view = PMapItems(pmap())
    assert (1, 'a') not in empty_view
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap
    # Note: Since we cannot instantiate PMap directly, 
    # we assume 'm' is the factory function available in the scope.
    m1 = m(a=1, b=2)
    
    # Test Case 1: Basic update with a single map (rightmost wins)
    # m2 should have 'a' updated to 2 and 'c' added as 3
    m2 = m1.update_with(lambda l, r: r, m(a=2, c=3))
    assert m2 == {'a': 2, 'b': 2, 'c': 3}
    # Ensure original is unchanged (immutability)
    assert m1 == {'a': 1, 'b': 2}

    # Test Case 2: update_with using a custom merge function (addition)
    # a: 1 + 2 = 3; b remains 2; c added as 3
    m3 = m1.update_with(lambda l, r: l + r, m(a=2, c=3))
    assert m3 == {'a': 3, 'b': 2, 'c': 3}

    # Test Case 3: update_with using a custom merge function (leftmost wins)
    # a: keeps original 1; b remains 2; c added as 3
    m4 = m1.update_with(lambda l, r: l, m(a=2, c=3))
    assert m4 == {'a': 1, 'b': 2, 'c': 3}

    # Test Case 4: Multiple maps provided in a sequence
    # m5 starts with {a:1, b:2}
    # Update with {a:2} -> {a:2, b:2}
    # Update with {'a':17, 'd':35} -> {a:17, b:2, d:35}
    m5 = m1.update_with(m(a=2), {'a': 17, 'd': 35})
    assert m5 == {'a': 17, 'b': 2, 'd': 35}

    # Test Case 5: Update with an empty map
    m6 = m1.update_with(m())
    assert m6 == m1

    # Test Case 6: Updating a key that doesn't exist in the original (handled by logic)
    # The code says: update_fn(evolver[key], value) if key in evolver else value
    # So for 'z', it should just take the value from the source map.
    m7 = m1.update_with(lambda l, r: l + r, m(z=10))
    assert m7['z'] == 10
    assert m7['a'] == 1 # 'a' was not in the update map, so it remains unchanged

    # Test Case 7: Complex chain of updates
    m8 = m1.update_with(
        lambda l, r: l + r, 
        m(a=10),           # a becomes 1 + 10 = 11
        {'b': 20},         # b becomes 2 + 20 = 22
        m(c=30)            # c is new = 30
    )
    assert m8 == {'a': 11, 'b': 22, 'c': 30}
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {1: 'a', 2: 'b', 3: 'c'}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test valid item presence (key and value match)
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view

    # Test key exists but value is different
    assert (1, 'z') not in items_view
    assert (2, 'a') not in items_view

    # Test key does not exist in map
    assert (4, 'd') not in items_view
    assert (99, 'a') not in items_view

    # Test non-iterable input (should catch exception and return False)
    assert 1 not in items_view
    assert None not in items_view

    # Test iterable that is not a pair (e.g., single element tuple)
    assert (1,) not in items_view

    # Test empty map
    empty_view = PMapItems(pmap({}))
    assert (1, 'a') not in empty_view
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap using a mock factory/constructor approach 
    # Since we cannot instantiate PMap directly easily without the full environment,
    # we assume 'm' or the constructor is available as per docstring.
    # For testing purposes, we simulate the behavior of PMap via its interface.
    
    # Helper to create a PMap instance for testing
    # In a real test environment, you would use: from pyrsistent import m
    def create_pmap(data):
        # This assumes the existence of the logic provided in the snippet
        # We simulate the behavior described in the docstrings.
        from pyrsistent import m 
        return m(data)

    try:
        m1 = create_pmap({'a': 1, 'b': 2})
        
        # Test Case 1: Simple update with another PMap (Rightmost wins)
        # m1.update_with(lambda l, r: r, m(a=2), {'a': 17, 'd': 35}) == {'a': 17, 'b': 2, 'c': 3, 'd': 35}
        m2 = m1.update_with(lambda l, r: r, create_pmap({'a': 2}), create_pmap({'a': 17, 'd': 35}))
        assert m2['a'] == 17
        assert m2['b'] == 2
        assert m2['s'] is not hasattr(m2, 's') # Safety check
        assert m2['d'] == 35

        # Test Case 2: Update with a merge function (Addition)
        # m1.update_with(add, m(a=2)) == {'a': 3, 'b': 2}
        from operator import add
        m3 = m1.update_with(add, create_pmap({'a': 2}))
        assert m3['a'] == 3
        assert m3['b'] == 2

        # Test Case 3: Leftmost wins (Reverse behavior)
        # m1.update_with(lambda l, r: l, m(a=2), {'a':3}) -> {'a': 1}
        m4 = m1.update_with(lambda l, r: l, create_pmap({'a': 2}), create_pmap({'a': 3}))
        assert m4['a'] == 1

        # Test Case 4: Update with standard dicts
        m5 = m1.update_with(add, {'b': 10})
        assert m5['b'] == 12
        assert m5['a'] == 1

        # Test Case 5: No changes when no new keys are provided/merging same values
        m6 = m1.update_with(lambda l, r: r, create_pmap({'b': 2}))
        assert m6 == m1
        
    except ImportError:
        pytest.skip("pyrsistent not installed; cannot run integration test for PMap")

```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap
    # Note: Using a mock-like approach for dependencies since we can't import pmap/m
    # But assuming the environment has access to the classes provided in the snippet.
    # We use 'm' as it is the standard factory function implied by the docstrings.
    from pyrsistent import m, pvector

    # Test Case 1: Basic update (rightmost value wins)
    m1 = m(a=1, b=2)
    m2 = m(a=2, c=3)
    m3 = {'a': 17, 'd': 35}
    # The logic in the code: update_with uses a lambda l, r: r (standard update)
    result = m1.update_with(lambda l, r: r, m2, m3)
    assert result == {'a': 17, 'b': 2, 'c': 3, 'd': 35}

    # Test Case 2: Custom merge function (leftmost value wins)
    # Logic: update_fn(evolver[key], value) if key in evolver else value
    m4 = m(a=1)
    m5 = m(a=2)
    m6 = {'a': 3}
    # Using lambda l, r: l means we keep the existing (leftmost) value
    result_leftmost = m4.update_with(lambda l, r: l, m5, m6)
    assert result_leftmost['a'] == 1

    # Test Case 3: Arithmetic merge (summing values)
    from operator import add
    m7 = m(a=1, b=2)
    m8 = m(a=2)
    result_add = m7.update_with(add, m8)
    assert result_add['a'] == 3
    assert result_add['b'] == 2

    # Test Case 4: Update with an empty map
    m9 = m(a=1)
    result_empty = m9.update_with(add, m())
    assert result_empty == {'a': 1}

    # Test Case 5: Verifying that the original PMap remains immutable
    m10 = m(x=10)
    m10.update_with(add, m(x=5))
    assert m10['x'] == 10

    # Test Case 6: Multiple maps with overlapping keys
    # Order of operations matters: m1 -> m2 (wins) -> m3 (wins)
    m_base = m(k=0)
    m_first = m(k=1, y=1)
    m_second = {'k': 2, 'z': 2}
    result_chain = m_base.update_with(lambda l, r: r, m_first, m_second)
    assert result_chain['k'] == 2
    assert result_chain['y'] == 1
    assert result_chain['z'] == 2

    # Test Case 7: Key exists in new map but not in evolver
    m_only_new = m(a=1)
    m_incoming = {'b': 2}
    result_new_key = m_only_new.update_with(lambda l, r: r, m_incoming)
    assert 'b' in result_new_key
    assert result_new_key['b'] == 2
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap using a mock-like structure or factory if available.
    # Since we don't have the factory 'm', we use the class constructor directly.
    # We need to simulate the internal buckets structure for a minimal working instance.
    
    def create_minimal_pmap(data_dict):
        # Calculate size and buckets manually for testing purposes
        size = len(data_dict)
        num_buckets = 10  # Arbitrary number of buckets
        buckets = [None] * num_buckets
        for k, v in data_dict.items():
            idx = hash(k) % num_buckets
            if buckets[idx] is None:
                buckets[idx] = []
            buckets[idx].append((k, v))
        return PMap(size, pvector(buckets))

    # We assume pvector and m/pmap are available in the environment as per context
    # If not, we rely on the provided class definitions.
    
    # Test Case 1: Standard merge (rightmost wins)
    m1 = create_minimal_pass_through({'a': 1, 'b': 2})
    m2 = create_minimal_pass_through({'a': 10, 'c': 3})
    
    # Using update as it is a wrapper for update_with with lambda l, r: r
    result_update = m1.update(m2)
    assert result_update['a'] == 10
    assert result_update['b'] == 2
    assert result_update['c'] == 3

    # Test Case 2: Custom update function (leftmost wins)
    # Using lambda l, r: l
    result_leftmost = m1.update_with(lambda l, r: l, m2)
    assert result_leftmost['a'] == 1
    assert result_leftmost['b'] == 2
    assert result_leftmost['c'] == 3

    # Test Case 3: Custom update function (additive)
    from operator import add
    m3 = create_minimal_pass_through({'a': 5, 'b': 10})
    m4 = create_minimal_pass_through({'a': 2, 'c': 20})
    
    result_add = m3.update_with(add, m4)
    assert result_add['a'] == 7  # 5 + 2
    assert result_add['b'] == 10
    assert result_add['c'] == 20

    # Test Case 4: Multiple maps in update_with
    m5 = create_minimal_pass_through({'a': 1})
    m6 = create_minimal_pass_through({'a': 2, 'b': 2})
    m7 = create_minimal_pass_through({'a': 3, 'c': 3})
    
    # Sequence: m5 -> update with m6 (1+2=3) -> update with m7 (3+3=6)
    result_multi = m5.update_with(add, m6, m7)
    assert result_multi['a'] == 6
    assert result_multi['b'] == 2
    assert result_multi['c'] == 3

    # Test Case 5: Update with non-existent key in evolver (should just insert value)
    m8 = create_minimal_pass_through({'x': 10})
    m9 = create_minimal_pass_through({'y': 20})
    result_new_key = m8.update_with(add, m9)
    assert result_new_key['y'] == 20
    assert 'x' in result_new_key

def create_minimal_pass_through(d):
    """Helper to construct PMap for testing without the full factory logic."""
    # This assumes pvector is available as used in the class implementation
    size = len(d)
    num_buckets = 1024 
    buckets = [None] * num_buckets
    for k, v in d.items():
        idx = hash(k) % num_buckets
        if buckets[idx] is None:
            buckets[idx] = []
        buckets[idx].append((k, v))
    # We use a pvector to wrap the list as PMap expects it
    return PMap(size, pvector(buckets))
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a persistent map
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)

    # Test containing an existing item (key and value match)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test containing an existing key but wrong value
    assert ('a', 99) not in items_view

    # Test containing a non-existent key
    assert ('z', 1) not in items_view

    # Test containing something that is not a tuple (should return False via try/except)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test containing an invalid tuple structure (wrong length)
    assert ('a', 1, 'extra') not in items_view

    # Test containing an empty tuple
    assert () not in items_view

    # Verify it works with a mapping that is converted to pmap internally
    dict_m = {'a': 1}
    items_view_from_dict = PMapItems(dict_m)
    assert ('a', 1) in items_view_from_dict

    # Test TypeError for non-mapping initialization (edge case of __init__)
    with pytest.raises(TypeError, match="PViewMap requires a Mapping object"):
        PMapItems([1, 2, 3])
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap
    # Note: Since we cannot instantiate PMap directly due to its design, 
    # we assume 'm' is the factory function available in the scope.
    # We use a helper to simulate the creation of PMap for testing purposes.
    from operator import add

    def create_test_map(data):
        # Using the provided logic: m1 = m(a=1, b=2) 
        # In a real test environment, 'm' is imported from pyrsistent
        # Here we mock the behavior based on the docstring examples.
        e = PMap(0, []).evolver()
        for k, v in data.items():
            e.set(k, v)
        return e.persistent()

    m1 = create_test_map({'a': 1, 'b': 2})
    m2 = create_test_map({'a': 2})
    m3 = {'a': 17, 'd': 35} # Standard dict
    
    # Test Case 1: Standard merge (using update which uses lambda l, r: r)
    # m1.update(m2) should result in {'a': 2, 'b': 2}
    res_update = m1.update(m2)
    assert res_update == {'a': 2, 'b': 2}
    assert res_update['a'] == 2
    assert res_update['b'] == 2

    # Test Case 2: update_with using addition as merge function
    # m1.update_with(add, m2) -> 'a' becomes 1 + 2 = 3
    res_add = m1.update_with(add, m2)
    assert res_add == {'a': 3, 'b': 2}

    # Test Case 3: update_with using a custom function (leftmost preference)
    # m1.update_with(lambda l, r: l, m2) -> 'a' stays 1
    res_left = m1.update_with(lambda l, r: l, m2)
    assert res_left == {'a': 1, 'b': 2}

    # Test Case 4: update_with with multiple maps and a dict
    # m1.update_with(add, m(a=2), {'a': 3, 'd': 35})
    # Step 1: m1 + m2 -> {'a': 3, 'b': 2}
    # Step 2: result + {'a': 3, 'd': 35} -> {'a': 6, 'b': 2, 'd': 35}
    m_extra = create_test_map({'a': 2})
    res_multi = m1.update_with(add, m_extra, {'a': 3, 'd': 35})
    assert res_multi == {'a': 6, 'b': 2, 'd': 35}

    # Test Case 5: update_with with a key that does not exist in the base map (it should just insert)
    m4 = create_test_map({'z': 10})
    res_new_key = m1.update_with(add, m4)
    assert res_new_key == {'a': 1, 'b': 2, 'z': 10}

    # Test Case 6: Immutability check
    # Ensure original map m1 is not modified
    assert m1 == {'a': 1, 'b': 2}

    # Test Case 7: update_with with no additional maps (should return self)
    res_self = m1.update_with(add)
    assert res_self == m1
    assert res_self is m1
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_PMap___eq__():
    # We need a way to create PMap instances. 
    # Since the implementation uses an internal factory/constructor logic 
    # and depends on PVector/PSet, we assume 'm' is available as per docstrings.
    # For the purpose of this test, we will mock the creation or use a known working state.
    
    # Setup: Create base map instances
    # Note: Using dict-based initialization if m() is available in the environment
    from pyrsistent import m 
    
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m3 = m(a=1, b=3)
    m4 = m(a=1, b=2, c=3)
    dict_equiv = {'a': 1, 'b': 2}
    other_type = [('a', 1), ('b', 2)]
    
    # Test Case 1: Identity (Self equality)
    assert m1 == m1
    
    # Test Case 2: Equality with same content (PMap vs PMap)
    assert m1 == m2
    
    # Test Case 3: Inequality with different content (PMap vs PMap)
    assert m1 != m3
    assert m1 != m4
    
    # Test Case 4: Equality with dict
    assert m1 == dict_equiv
    
    # Test Case 5: Inequality with dict (different size)
    assert m1 != {'a': 1, 'b': 2, 'c': 3}
    
    # Test Case 6: Equality with dict (different order - dicts are order-agnostic for ==)
    assert m1 == {'b': 2, 'a': 1}
    
    # Test Case 7: Equality with other Mapping types (e.g., another PMap instance via items)
    # The code uses dict(other.items()) for non-PMap mappings
    class MockMapping:
        def __len__(self): return 2
        def items(self): return [('a', 1), ('b', 2)]
        def __getitem__(self, k):
            if k == 'a': return 1
            if k == 'b': return 2
            raise KeyError()

    assert m1 == MockMapping()
    
    # Test Case 8: Inequality with non-mapping types
    assert m1 != [('a', 1), ('b', 2)]
    assert m1 != "not a map"
    assert m1 != None

    # Test Case 9: Testing the _cached_hash optimization logic 
    # (If we can trigger a hash mismatch, but that's hard without internal access)
    # We verify that if contents are same, it returns True even if hashes aren't computed yet.
    m5 = m(a=1, b=2)
    assert m1 == m5
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a PMap for testing
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)

    # Test Case 1: Existing item (key and value match)
    assert ('a', 1) in items_key_value := items_view
    assert ('b', 2) in items_key_value
    
    # Test Case 2: Key exists but value is different
    assert ('a', 99) not in items_key_value
    
    # Test Case 3: Key does not exist in the map
    assert ('z', 1) not in items_key_value
    
    # Test Case 4: Providing an item that is not a tuple/iterable (should return False via exception handling)
    assert 'a' not in items_key_value
    assert 1 not in items_key_value
    assert None not in items_key_value
    
    # Test Case 5: Providing an iterable that isn't a pair (e.g., a single element tuple)
    assert ('a',) not in items_key_value

    # Test Case 6: Providing more than two elements in the tuple
    assert ('a', 1, 'extra') not in items_key_value

    # Test Case 7: Empty map behavior
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap
    # Note: Using m() factory as implied by docstrings in provided code
    m1 = m(a=1, b=2)
    
    # Test Case 1: Basic update with rightmost values (standard behavior)
    # Similar to .update(), should overwrite with latest value
    m2 = m1.update_with(lambda l, r: r, m(a=2, c=3), {'a': 17, 'd': 35})
    assert m2 == {'a': 17, 'b': 2, 'c': 3, 'd': 35}

    # Test Case 2: Using a custom merge function (additive)
    from operator import add
    m3 = m1.update_with(add, m(a=2))
    assert m3 == {'a': 3, 'b': 2}

    # Test Case 3: Reverse behavior (keeping leftmost element)
    m4 = m(a=1)
    m5 = m4.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m5 == {'a': 1}

    # Test Case 4: Multiple maps with complex logic
    # Start with {a: 10}, add {a: 5} then {a: 2}, using addition at each step
    m6 = m(a=10)
    m7 = m6.update_with(add, m(a=5), {'a': 2})
    assert m7['a'] == 17

    # Test Case 5: Updating with an empty map
    m8 = m1.update_with(add, m())
    assert m8 == m1
    assert m8 is not m1 # Should be a new instance if changes occurred or via evolver

    # Test Case 6: Verifying that the original map remains immutable
    m9 = m(x=1)
    m9.update_with(add, m(x=2))
    assert m9['x'] == 1

    # Test Case 7: Updating with a standard dict (Mapping protocol)
    m10 = m(k=1)
    m11 = m10.update_with(lambda l, r: r, {'k': 5, 'y': 10})
    assert m11 == {'k': 5, 'y': 10}

    # Test Case 8: Key exists in evolver but not in provided map (checking logic)
    # The implementation uses: update_fn(evolver[key], value) if key in evolver else value
    m12 = m(z=100)
    m13 = m12.update_with(lambda l, r: l + r, m(a=1)) 
    # 'a' is not in evolver yet, so it should just take the value from the map (1)
    assert m13['a'] == 1
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a persistent map for testing
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test Case 1: Valid item (key and value exist in map)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test Case 2: Key exists but value is different
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test Case 3: Key does not exist in map
    assert ('z', 1) not in items_view
    assert ('d', 4) not in items_view

    # Test Case 4: Input is not a tuple/pair (should return False via except block)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [] not in items_view

    # Test Case 5: Input is a tuple but has wrong length (should return False via except block)
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view

    # Test Case 6: Input is a tuple that is not hashable (e.g., containing a list)
    # This tests the robustness of the try-except block for unpacking/lookup
    assert (('a',), 1) not in items_view
```


# LLM-generated content at query #24
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap using a factory-like approach 
    # (Assuming 'm' is the factory function described in docstrings)
    # Since we can't call m() directly without its definition, 
    # we use the logic provided: PMap(size, buckets) via an evolver.
    
    def create_pmap(data_dict):
        evolver = PMap(0, pvector()).evolver()
        for k, v in data_dict.items():
            evolver.set(k, v)
        return evolver.persistent()

    # Mocking dependencies for the test environment
    # In a real scenario, these would be imported from pyrsistent
    from operator import add

    m1 = create_pmap({'a': 1, 'name': 'test'})
    m2 = create_pmap({'a': 2, 'b': 3})
    m3 = {'a': 10, 'c': 20}  # A standard dict

    # Test Case 1: Basic update (Rightmost wins)
    # m1.update(m2, m3) -> {'a': 10, 'name': 'test', 'b': 3, 'c': 20}
    updated = m1.update(m2, m3)
    assert updated['a'] == 10
    assert updated['name'] == 'test'
    assert updated['b'] == 3
    assert updated['c'] == 20
    assert len(updated) == 4

    # Test Case 2: update_with using a merge function (Addition)
    # m1.update_with(add, m2) -> {'a': 1 + 2, 'name': 'test', 'b': 3}
    added = m1.update_with(add, m2)
    assert added['a'] == 3
    assert added['name'] == 'test'
    assert added['b'] == 3

    # Test Case 3: update_with using a merge function (Leftmost wins)
    # We want the first value encountered to persist.
    # m1.update_with(lambda l, r: l, m2, m3) -> {'a': 1, 'name': 'test'}
    leftmost = m1.update_with(lambda l, r: l, m2, m3)
    assert leftmost['a'] == 1
    # Note: keys in m2 and m3 that aren't in m1 are still added as 'value' 
    # because the logic says: if key not in evolver else update_fn
    assert leftmost['b'] == 3
    assert leftmost['c'] == 20

    # Test Case 4: update_with with multiple maps and complex logic
    # m1 = {'a': 1}
    # m2 = {'a': 5, 'x': 10}
    # m3 = {'a': 10, 'y': 20}
    # Logic: 
    # Start with m1. 
    # Process m2: key 'a' is in evolver, so evolver['a'] = add(1, 5) -> 6. Key 'x' is new -> 10.
    # Process m3: key 'a' is in evolver, so evolver['a'] = add(6, 10) -> 16. Key 'y' is new -> 20.
    m_base = create_pmap({'a': 1})
    m_extra1 = create_pmap({'a': 5, 'x': 10})
    m_extra2 = {'a': 10, 'y': 20}
    
    result = m_base.update_with(add, m_extra1, m_extra2)
    assert result['a'] == 16
    assert result['x'] == 10
    assert result['y'] == 20

    # Test Case 5: Immutability check
    # Ensure the original map is not modified
    original_len = len(m1)
    m1.update_with(add, m2)
    assert len(m1) == original_len
```


# LLM-generated content at query #25
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap (Assuming m is a factory function for PMap)
    # Since we cannot use imports, we assume 'm' and 'PMap' are available in the namespace.
    # For the sake of this test, we will simulate the existence of the factory 'm'.
    
    # Helper to create a minimal working PMap structure if m isn't provided
    # In a real scenario, m = pmap or similar from pyrsistent
    try:
        from pyrsistent import m
    except ImportError:
        # Fallback for standalone testing environment logic
        class MockPMap(PMap):
            def __init__(self, data):
                # This is a hack to allow the test to run without the full pyrsistent library
                # by injecting data into the internal structure.
                super().__new__(PMap, len(data), [None] * 10)
                self._size = len(data)
                # Simplified bucket logic for testing purposes
                buckets = [None] * 10
                for k, v in data.items():
                    idx = hash(k) % 10
                    if buckets[idx] is None: buckets[idx] = []
                    buckets[idx].append((k, v))
                self._buckets = buckets

        m = lambda **kwargs: MockPMap(kwargs)

    # Test Case 1: Basic update (rightmost value wins)
    m1 = m(a=1, b=2)
    m2 = m(a=2, c=3)
    m_dict = {'a': 1, 'b': 2}
    # Using the logic provided in the docstring: m1.update(m2, {'a': 17, 'd': 35})
    result1 = m1.update(m2, {'a': 17, 'd': 35})
    assert result1 == {'a': 17, 'b': 2, 'c': 3, 'd': 35}

    # Test Case 2: update_with with addition (merging values)
    from operator import add
    m3 = m(a=1, b=2)
    m4 = m(a=2)
    result2 = m3.update_with(add, m4)
    assert result2 == {'a': 3, 'b': 2}

    # Test Case 3: update_with with multiple maps and additive logic
    m5 = m(a=10, b=20)
    m6 = m(a=1, c=3)
    m7 = {'a': 5, 'd': 4} # Testing with dict as well
    result3 = m5.update_with(add, m6, m7)
    # Step 1: a: 10 + 1 = 11, b: 20, c: 3
    # Step 2: a: 11 + 5 = 16, b: 20, c: 3, d: 4
    assert result3 == {'a': 16, 'b': 20, 'c': 3, 'd': 4}

    # Test Case 4: update_with with "leftmost" logic (lambda l, r: l)
    m8 = m(a=1)
    m9 = m(a=2)
    m10 = {'a': 3}
    result4 = m8.update_with(lambda l, r: l, m9, m10)
    assert result4 == {'a': 1}

    # Test Case 5: update_with with no changes (identity)
    m11 = m(a=1)
    result5 = m11.update_with(lambda l, r: r, m11)
    assert result5 == {'a': 1}

    # Test Case 6: update_with with keys that don't exist in the original (standard insertion)
    m12 = m(x=1)
    m13 = m(y=2)
    result6 = m12.update_with(add, m13)
    assert result6 == {'x': 1, 'y': 2}

    # Test Case 7: verify immutability of the original map
    m_orig = m(a=1)
    m_orig.update_with(add, m(a=2))
    assert m_orig['a'] == 1
```


# LLM-generated content at query #26
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a PMap for testing
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)

    # Test case: exact match of key and value
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test case: key exists but value is different
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test case: key does not exist
    assert ('z', 1) not in items_view
    assert (None, 1) not in items_view

    # Test case: input is not a tuple/pair (triggers the try-except block)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [] not in items_view

    # Test case: input is a tuple but not a pair (unpacks incorrectly)
    assert ('a', 1, 'extra') not in items_view

    # Test case: input is an empty tuple
    assert () not in items_view
```


# LLM-generated content at query #27
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test exact match existing item
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test key exists but value is different
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test key does not exist in map
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test input is not a tuple (should return False via exception handling)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test input is a tuple with wrong length
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view

    # Test empty map behavior
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #28
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test with valid existing item (key and value match)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test with existing key but wrong value
    assert ('a', 99) not in items_view

    # Test with non-existent key
    assert ('z', 1) not in items_view

    # Test with an item that is not a tuple/pair (should return False via except block)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test with a tuple that doesn't match the mapping content
    assert ('d', 4) not in items_view

    # Test with an empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #29
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial maps
    m1 = m(a=1, b=2)
    m2 = m(a=2, c=3)
    m3 = {'a': 17, 'd': 35}
    
    # Test Case 1: Basic update (using lambda for replacement/rightmost behavior)
    # Similar to the docstring example: m1.update_with(lambda l, r: r, m2, m3)
    # Expected: a=17, b=2, c=3, d=35
    res1 = m1.update_with(lambda l, r: r, m2, m3)
    assert res1 == {'a': 17, 'b': 2, 'c': 3, 'd': 35}
    assert len(res1) == 4

    # Test Case 2: Using an additive function (merging values)
    # m1 has a=1, b=2. m2 has a=2, c=3.
    # update_with(add, m2) -> for key 'a', 1 + 2 = 3. For key 'c', 3 is inserted.
    from operator import add
    res2 = m1.update_with(add, m2)
    assert res2 == {'a': 3, 'b': 2, 'c': 3}

    # Test Case 3: Leftmost behavior (keeping original value on conflict)
    # m1 has a=1. m2 has a=2. m3 has a=3.
    # update_with(lambda l, r: l, m2, m3) -> 'a' stays 1.
    m_base = m(a=1)
    res3 = m_base.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert res3 == {'a': 1}

    # Test Case 4: Immutability check
    # Ensure the original map m1 is not mutated by update_with
    m1_clone = m(a=1, b=2)
    m1.update_with(add, m2)
    assert m1 == m1_clone
    assert m1['a'] == 1

    # Test Case 5: Update with a single map (no conflict)
    res5 = m1.update_with(lambda l, r: r, m(z=99))
    assert res5 == {'a': 1, 'b': 2, 'z': 99}

    # Test Case 6: Update with empty map
    res6 = m1.update_with(add, m())
    assert res6 == {'a': 1, 'b': 2}

    # Test Case 7: Complex update with multiple maps and different types
    # m1={a:1}, m2={a:10, b:20}, m3={b:5, c:30}
    # fn = lambda l, r: l + r
    # a: 1 + 10 = 11
    # b: 20 + 5 = 25
    # c: 30 (new)
    m_start = m(a=1)
    res7 = m_start.update_with(lambda l, r: l + r, m(a=10, b=20), {'b': 5, 'c': 30})
    assert res7 == {'a': 11, 'b': 25, 'c': 30}

    # Test Case 8: Ensure it handles standard dicts as input (as per docstring)
    res8 = m1.update_with(lambda l, r: r, {'b': 99})
    assert res8['b'] == 99
```


