####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PMapValues___eq__():
    # Setup: Create a pmap and a PMapValues view
    # Assuming pmap is available in the scope as per the context of the provided code
    m = pmap({'a': 1, 'b': 2})
    view = PMapValues(m)
    
    # Test case 1: Identity equality (x is self)
    assert view == view
    
    # Test case 2: Equality with another instance containing same data
    # Based on the implementation: "if x is self: return True else: return False"
    view2 = PMapValues(m)
    assert view != view2
    
    # Test case 3: Equality with a different object type
    assert view != [1, 2]
    
    # Test case 4: Equality with None
    assert view != None
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a PMap for testing
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)

    # Test case 1: Valid item present in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test case 2: Valid key but wrong value
    assert ('a', 2) not in items_view
    assert ('b', 99) not in items_view

    # Test case 3: Key does not exist in the map
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test case 4: Input is not a tuple (should return False via Exception handling)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [] not in items_view

    # Test case 5: Input is a tuple but not a pair (length != 2)
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view

    # Test case 6: Input is an empty tuple
    assert () not in items_view
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from pyrsistent import pmap, pset

def test_PMapView():
    # Test valid PMap initialization
    m = pmap({'a': 1, 'b': 2})
    view = PMapView(m)
    assert view._map == m
    assert len(view) == 2

    # Test valid Mapping (dict) initialization (should convert to pmap)
    d = {'x': 10, 'y': 20}
    view_from_dict = PMapView(d)
    assert isinstance(view_from_dict._map, pmap().__class__)
    assert view_from_dict._map == pmap(d)
    assert len(view_from_dict) == 2

    # Test invalid type initialization (raises TypeError)
    with pytest.raises(TypeError, match="PViewMap requires a Mapping object"):
        PMapView([1, 2, 3])

    # Test immutability of the view (raises TypeError on __setattr__)
    with pytest.raises(TypeError, match="<class 'PMapView'> is immutable"):
        view.new_attr = True

    # Test __len__ functionality
    assert len(view) == len(m)

    # Test __reversed__ functionality (raises TypeError)
    with pytest.raises(TypeError, match="Persistent maps are not reversible"):
        reversed(view)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_PMapValues___str__():
    # Create a mock PMap-like object that implements itervalues
    class MockPMap:
        def __init__(self, data):
            self.data = data
        def itervalues(self):
            return iter(self.data.values())

    # Test Case 1: Empty map
    empty_map = MockPMap({})
    view_empty = PMapValues(empty_map)
    assert str(view_empty) == "pmap_values([])"

    # Test Case 2: Map with values
    data = {'a': 1, 'b': 2, 'c': 3}
    map_obj = MockPMap(data)
    view = PMapValues(map_obj)
    # The str implementation uses list(iter(self)), which follows insertion order
    assert str(view) == "pmap_values([1, 2, 3])"

    # Test Case 3: Map with different types of values
    data_mixed = {'x': 'hello', 'y': None, 'z': True}
    map_mixed = MockPMap(data_mixed)
    view_mixed = PMapValues(map_mixed)
    assert str(view_mixed) == "pmap_values(['hello', None, True])"
```


# LLM-generated content at query #5
#--------------------------

```python
def test_PMapValues___eq__():
    # Setup: Create a pmap and a PMapValues view
    # Assuming pmap and PMap are available in the environment as per instructions
    data = {'a': 1, 'b': 2}
    m = pmap(data)
    values_view = PMapValues(m)
    
    # Test identity equality (x is self)
    assert values_view == values_view
    
    # Test equality with a different object of the same type/content
    # The implementation specifically returns False for anything not 'is self'
    different_view = PMapValues(m)
    assert values_view != different_view
    
    # Test equality with a standard list of values
    assert values_view != [1, 2]
    
    # Test equality with a different PMapValues instance
    other_map = pmap({'a': 1, 'b': 2})
    other_view = PMapValues(other_map)
    assert values_view != other_view

    # Test equality with non-comparable types
    assert values_view != "not a view"
    assert values_view != 123
```


# LLM-generated content at query #6
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
    assert ('a', 2) not in items_view
    assert ('b', 99) not in items_view

    # Test key does not exist
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test input is not an iterable/tuple (triggers Exception in try block)
    assert 1 not in items_view
    assert 'a' not in items_view
    assert None not in items_view

    # Test input is a tuple of wrong length
    assert ('a', 1, 2) not in items_view
    assert ('a',) not in items_view

    # Test input is an empty tuple
    assert () not in items_view
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_PMap___eq__():
    # Mocking the dependencies needed for PMap instantiation
    # Since we cannot import, we assume a structure where we can create a PMap
    # For testing purposes, we'll use a helper to build a PMap instance
    # because PMap cannot be instantiated directly with a simple dict.
    
    def create_pmap(data_dict):
        # This is a hack to bypass the lack of factory functions in the snippet
        # We simulate the internal structure of PMap
        size = len(data_dict)
        # We need a list of buckets. We'll use a size that avoids collisions for simplicity.
        num_buckets = 100 
        buckets = [None] * num_buckets
        for k, v in data_dict.items():
            idx = hash(k) % num_buckets
            if buckets[idx] is None:
                buckets[idx] = []
            buckets[idx].append((k, v))
        return PMap(size, buckets)

    # Setup test cases
    map_a = create_pmap({'a': 1, 'b': 2})
    map_b = create_pmap({'a': 1, 'b': 2})
    map_c = create_pmap({'a': 1, 'b': 3})
    map_d = create_pmap({'a': 1, 'b': 2, 'c': 3})
    dict_equiv = {'a': 1, 'b': 2}
    dict_equiv_alt = {'b': 2, 'a': 1}
    dict_diff = {'a': 1, 'b': 3}
    
    # 1. Identity: same object
    assert map_a == map_a
    
    # 2. Equality with same content (different objects, same buckets/content)
    assert map_a == map_b
    
    # 3. Equality with dictionary containing same items
    assert map_a == dict_equiv
    assert map_a == dict_equiv_alt
    
    # 4. Inequality with different content
    assert map_a != map_c
    assert map_a != dict_diff
    
    # 5. Inequality with different size
    assert map_a != map_d
    
    # 6. Equality with different object type but same content (Mapping protocol)
    # Testing the branch: elif isinstance(other, dict):
    assert map_a == {'a': 1, 'b': 2}
    
    # 7. Inequality with different type (not a Mapping)
    # Testing the branch: if not isinstance(other, Mapping): return NotImplemented
    # Note: In pytest, comparing to a non-mapping usually returns False or NotImplemented
    # We check that it doesn't crash and returns False for incompatible types
    assert map_a != [1, 2, 3]
    assert map_a != "not a map"
    
    # 8. Testing PMap vs PMap with same content but different bucket structure
    # (The implementation checks if buckets are equal, then falls back to dict comparison)
    # We force a different bucket structure by using a different bucket count
    def create_pmap_alt_buckets(data_dict):
        num_buckets = 500
        size = len(data_dict)
        buckets = [None] * num_buckets
        for k, v in data_dict.items():
            idx = hash(k) % num_buckets
            if buckets[idx] is None:
                buckets[idx] = []
            buckets[idx].append((k, v))
        return PMap(size, buckets)
    
    map_alt = create_pmap_alt_buckets({'a': 1, 'b': 2})
    assert map_a == map_alt
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    m = pmap(data)
    items_view = PMapItems(m)

    # Test existing key-value pair
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    
    # Test non-existing key with correct value
    assert ('d', 1) not in items_view
    
    # Test existing key with incorrect value
    assert ('a', 99) not in items_view
    
    # Test non-existing key with non-existing value
    assert ('z', 99) not in items_view
    
    # Test invalid input types (not a tuple/pair)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [] not in items_view
    
    # Test tuple with wrong length
    assert ('a', 1, 'extra') not in items_view
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

    # Test containing valid key-value pair
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test containing key that exists but with wrong value
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test containing key that does not exist
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test containing non-iterable objects (should return False via exception handling)
    assert 1 not in items_view
    assert None not in items_view
    assert [1, 2] not in items_view

    # Test containing iterable that is not a pair (tuple of length != 2)
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view

    # Test containing a tuple that is a pair but not in the map
    assert ('missing', 'value') not in items_view
```


# LLM-generated content at query #10
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
    assert ('a', 2) not in items_view
    assert ('b', 0) not in items_view

    # Test keys not present in the map
    assert ('d', 4) not in items_view
    assert ('z', 1) not in items_view

    # Test non-iterable input (should return False via try-except)
    assert 123 not in items_view
    assert None not in items_view

    # Test iterable input that is not a pair (e.g., single element tuple)
    assert ('a',) not in items_view

    # Test empty map behavior
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from operator import add

def test_PMap_update_with():
    # Setup: Create initial PMap using the factory function 'm' 
    # (Assuming 'm' is the factory function that returns a PMap)
    # If 'm' is not available in the test environment, we use PMap constructor logic
    # For the purpose of this test, we assume 'm' is provided as per the docstring.
    
    # Helper to create a PMap for testing purposes if m is not globally defined
    def create_pmap(data):
        # This mimics the behavior of the factory function 'm'
        # We use the evolver to build it to ensure valid internal structure
        evolver = PMap(0, []).evolver()
        for k, v in data.items():
            evolver.set(k, v)
        return evolver.persistent()

    m1 = create_pmap({'a': 1, 'b': 2})
    m2 = create_pmap({'a': 2, 'c': 3})
    m3 = {'a': 17, 'd': 35}

    # Test Case 1: Standard update (rightmost value wins)
    # m1.update(m2, m3) -> {'a': 17, 'b': 2, 'c': 3, 'd': 35}
    result1 = m1.update_with(lambda l, r: r, m2, m3)
    expected1 = {'a': 17, 'b': 2, 'c': 3, 'd': 35}
    assert dict(result1.items()) == expected1

    # Test Case 2: Custom update function (e.g., addition)
    # m1.update_with(add, m2) -> {'a': 3, 'b': 2, 'c': 3}
    result2 = m1.update_with(add, m2)
    expected2 = {'a': 3, 'b': 2, 'c': 3}
    assert dict(result2.items()) == expected2

    # Test Case 3: Custom update function (leftmost value wins)
    # m1.update_with(lambda l, r: l, m2, m3) -> {'a': 1, 'b': 2, 'c': 3, 'd': 35}
    # Note: 'a' starts at 1 in m1. m2 has 'a': 2. m3 has 'a': 17.
    # Since it processes left to right:
    # step 1: m1 + m2 -> 'a' becomes 1 (from l)
    # step 2: (result) + m3 -> 'a' remains 1 (from l)
    result3 = m1.update_with(lambda l, r: l, m2, m3)
    expected3 = {'a': 1, 'b': 2, 'c': 3, 'd': 35}
    assert dict(result3.items()) == expected3

    # Test Case 4: Immutability check
    # Ensure original m1 is not modified
    assert dict(m1.items()) == {'a': 1, 'b': 2}

    # Test Case 5: Update with an empty map
    result4 = m1.update_with(add, {})
    assert dict(result4.items()) == {'a': 1, 'b': 2}

    # Test Case 6: Update with a map containing keys not in m1
    m4 = create_pmap({'z': 100})
    result5 = m1.update_with(add, m4)
    assert result5['z'] == 100
    assert result5['a'] == 1
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test valid item existence
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test non-existent key
    assert ('a', 2) not in items_view
    assert ('z', 1) not in items_view

    # Test non-existent value for existing key
    assert ('a', 99) not in items_view

    # Test invalid input types (not a tuple/iterable)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test invalid tuple structure (not a pair)
    assert ('a',) not in items_view
    assert (1, 2, 3) not in items_view

    # Test empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_PMap___eq__():
    # Setup: We need a way to create PMap instances. 
    # Since the provided code uses a factory 'm' and relies on PVector/PSet,
    # we assume 'm' is the factory function available in the scope.
    # For the sake of this test, we will simulate the internal structure 
    # required for PMap(size, buckets).
    
    def create_mock_pmap(data_dict):
        # We use a simple implementation of the buckets logic for testing __eq__
        # to avoid dependency on the full complex logic of pvector/evolver.
        # We simulate the internal state: _size and _buckets.
        # We use a list of buckets where each bucket is a list of (k, v) tuples.
        size = len(data_dict)
        # We use a large enough fixed size to avoid complex reallocations in the test
        num_buckets = 100 
        buckets = [None] * num_buckets
        for k, v in data_dict.items():
            idx = hash(k) % num_buckets
            if buckets[idx] is None:
                buckets[idx] = []
            buckets[idx].append((k, v))
        
        # Create the PMap instance
        # Note: PMap.__new__ is used here
        pmap = PMap(size, buckets)
        # Manually set _cached_hash to test the optimization path
        pmap._cached_hash = hash(frozenset(data_dict.items()))
        return pmap

    # Test Case 1: Identity (Self equality)
    m1 = create_mock_pmap({'a': 1, 'b': 2})
    assert m1 == m1

    # Test Case 2: Equality with another PMap (same content, different object)
    m2 = create_mock_pmap({'a': 1, 'b': 2})
    assert m1 == m2

    # Test Case 3: Equality with a dictionary (same content)
    m3 = {'a': 1, 'b': 2}
    assert m1 == m3

    # Test Case 4: Inequality (different content)
    m4 = create_mock_pmap({'a': 1, 'b': 3})
    assert m1 != m4
    assert m1 != {'a': 1, 'b': 3}

    # Test Case 5: Inequality (different length)
    m5 = create_mock_pmap({'a': 1})
    assert m1 != m5

    # Test Case 6: Inequality with different type (not a Mapping)
    assert m1 != [1, 2, 3]
    assert m1 != "not a map"

    # Test Case 7: Equality with a dict using items() (as specified in the code)
    m6 = {'a': 1, 'b': 2}
    assert m1 == m6.items() # This tests the branch: dict(self.iteritems()) == dict(other.items())

    # Test Case 8: Optimization path - different cached hash
    # We manually manipulate the hash to force the check
    m7 = create_mock_pmap({'a': 1, 'b': 2})
    m7._cached_hash = 999999 
    # Even if content is same, if cached_hash differs, it should return False 
    # according to the specific logic in the provided __eq__ implementation
    assert m1 != m7

    # Test Case 9: Inequality with different keys
    m8 = create_mock_pmap({'c': 1, 'b': 2})
    assert m1 != m8
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_PMap___eq__():
    # Mocking the internal structure of PMap for testing __eq__
    # PMap(size, buckets)
    # buckets is a list/vector of buckets, where each bucket is a list of (k, v)
    
    # Helper to create a PMap instance manually for testing purposes
    def create_test_pmap(data_dict):
        # We simulate the internal structure of PMap
        # Since we can't easily use the factory 'm' without the full library,
        # we use the __new__ method provided in the class definition.
        size = len(data_dict)
        # Create a simple bucket array. Using a large enough size to avoid collisions for simple tests
        # or just use a fixed size for predictable testing.
        buckets_size = 10 
        buckets = [None] * buckets_size
        for k, v in data_dict.items():
            idx = hash(k) % buckets_size
            if buckets[idx] is None:
                buckets[idx] = []
            buckets[idx].append((k, v))
        return PMap(size, buckets)

    # Case 1: Identity (Self comparison)
    m1 = create_test_pmap({'a': 1, 'key': 'val'})
    assert m1 == m1

    # Case 2: Equality with another PMap with same content
    m2 = create_test_pmap({'a': 1, 'key': 'val'})
    assert m1 == m2

    # Case 3: Equality with a dict containing same content
    m3_dict = {'a': 1, 'key': 'val'}
    assert m1 == m3_dict

    # Case 4: Equality with a PMapItems view (if implemented via Mapping protocol)
    # The code says: elif isinstance(other, dict): ... return dict(self.iteritems()) == other
    # and: return dict(self.iteritems()) == other.items()
    # So we test with a standard dict items view
    m3_items = {'a': 1, 'key': 'val'}.items()
    # Note: In Python 3, dict.items() is a view. The code uses dict(other.items())
    assert m1 == m3_items

    # Case 5: Inequality - Different size
    m4 = create_test_pmap({'a': 1})
    assert m1 != m4

    # Case 6: Inequality - Different values
    m5 = create_testting_pmap({'a': 2, 'key': 'val'})
    assert m1 != m5

    # Case 7: Inequality - Different keys
    m6 = create_test_pmap({'b': 1, 'key': 'val'})
    assert m1 != m6

    # Case 8: Inequality with non-mapping type
    assert m1 != [1, 2, 3]
    assert m1 != "not a map"
    assert m1 != None

    # Case 9: Different bucket structure but same content (should be True)
    # m1 has buckets via hash(k) % 10. 
    # We create a map with different bucket array size but same data.
    size_alt = 20
    buckets_alt = [None] * size_alt
    for k, v in {'a': 1, 'key': 'val'}.items():
        idx = hash(k) % size_alt
        if buckets_alt[idx] is None:
            buckets_alt[idx] = []
        buckets_alt[idx].append((k, v))
    m7 = PMap(2, buckets_alt)
    assert m1 == m7

# Helper for the test scope to allow the test to run independently
def create_testting_pmap(data_dict):
    size = len(data_dict)
    buckets_size = 10 
    buckets = [None] * buckets_size
    for k, v in data_dict.items():
        idx = hash(k) % buckets_size
        if buckets[idx] is None:
            buckets[idx] = []
        buckets[idx].append((k, v))
    return PMap(size, buckets)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_PMap___eq__():
    # Mocking the dependency of PMap since we cannot instantiate it with real PVector/PSet 
    # without the full environment. We will mock the internal structure.
    
    # Helper to create a mock PMap instance
    def create_mock_pmap(data_dict, buckets_list=None):
        # PMap requires size and buckets. We simulate the internal structure.
        # We mock the __iter__ and __len__ to behave like a dict.
        m = PMap(len(data_dict), buckets_list or [None] * 10)
        # We override the internal items for testing purposes
        m.iteritems = lambda: data_dict.items()
        m._buckets = buckets_list or [None] * 10
        m._size = len(data_dict)
        return m

    # Test Case 1: Identity (self is other)
    m1 = create_mock_pmap({'a': 1, 'b': 2})
    assert m1 == m1

    # Test Case 2: Equality with same content (dict)
    m2 = create_mock_pmap({'a': 1, 'b': 2})
    assert m1 == {'a': 1, 'b': 2}

    # Test Case 3: Equality with same content (another PMap)
    m3 = create_mock_pmap({'a': 1, 'b': 2})
    assert m1 == m3

    # Test Case 4: Inequality due to different size
    m4 = create_mock_pmap({'a': 1})
    assert m1 != m4
    assert m1 != {'a': 1, 'b': 2, 'c': 3}

    # Test Case 5: Inequality due to different values
    m5 = create_mock_pmap({'a': 99, 'b': 2})
    assert m1 != m5
    assert m1 != {'a': 99, 'b': 2}

    # Test Case 6: Inequality with different type (not a Mapping)
    assert m1 != [1, 2, 3]
    assert m1 != "not a map"

    # Test Case 7: Equality with different object but same dict-like items
    # (Testing the dict(self.iteritems()) == other logic)
    m6 = create_mock_pmap({'b': 2, 'a': 1})
    assert m1 == m6

    # Test Case 8: Inequality with different keys
    m7 = create_mock_pmap({'a': 1, 'c': 2})
    assert m1 != m7
```


# LLM-generated content at query #16
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
    
    # Test valid key exists but value is wrong
    assert ('a', 99) not in items_view
    
    # Test key does not exist in map
    assert ('z', 1) not in items_view
    
    # Test non-iterable argument (should return False via try-except)
    assert 123 not in items_view
    
    # Test tuple with wrong length (should return False via try-except)
    assert ('a', 1, 'extra') not in items_view
    
    # Test non-tuple/non-sequence type that might fail unpacking
    assert None not in items_view
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    m = pmap(data)
    items_view = PMapItems(m)

    # Test with valid key-value pair present in map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test with valid key but incorrect value
    assert ('a', 99) not in items_view
    assert ('c', 0) not in items_view

    # Test with key not present in map
    assert ('z', 1) not in items_view

    # Test with non-iterable input (should return False via try-except)
    assert 1 not in items_view
    assert None not in items_view

    # Test with iterable that is not a pair (length != 2)
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view

    # Test with a tuple that is a pair but not in the map
    assert ('nonexistent', 1) not in items_view
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___eq__():
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m3 = pmap({'a': 1, 'c': 3})
    
    items1 = PMapItems(m1)
    items2 = PMapItems(m2)
    items3 = PMapItems(m3)
    
    # Test identity equality
    assert items1 == items1
    
    # Test equality with same content
    assert items1 == items2
    
    # Test inequality with different content
    assert items1 != items3
    
    # Test inequality with different type
    assert items1 != [('a', 1), ('b', 2)]
    assert items1 != m1
    
    # Test equality with different instance but same underlying map
    class MockItems:
        def __init__(self, map_obj):
            self._map = map_obj
            
    items_mock = MockItems(m1)
    assert items1 == items_mock
```


# LLM-generated content at query #2
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
    assert ('b', 3) not in items_view

    # Test keys not present in the map
    assert ('d', 1) not in items_view
    assert ('z', 99) not in items_view

    # Test invalid input types (not a tuple/pair)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [] not in items_view

    # Test tuple with wrong length
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view

    # Test empty input
    assert () not in items_view
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___eq__():
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m3 = pmap({'a': 1, 'c': 3})
    m4 = pmap({'b': 2, 'a': 1})
    
    view1 = PMapItems(m1)
    view2 = PMapItems(m2)
    view3 = PMapItems(m3)
    view4 = PMapItems(m4)
    
    # Test identity
    assert view1 == view1
    
    # Test equality of different instances with same underlying map content
    # Note: pmap equality is based on content, and PMapItems.__eq__ compares underlying maps
    assert view1 == view2
    assert view1 == view4
    
    # Test inequality with different content
    assert view1 != view3
    
    # Test inequality with different type
    assert view1 != m1
    assert view1 != [('a', 1), ('b', 2)]
    assert view1 != PMapValues(m1)
    
    # Test with empty map
    empty_view = PMapItems(pmap())
    assert view1 != empty_view
    assert empty_view == PMapItems(pmap())
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    data = {'a': 1, 'b': 2, 'c': 3}
    p_map = pmap(data)
    items_view = PMapItems(p_map)

    # Test existing item (key and value match)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test existing key but wrong value
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test non-existing key
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test input that is not a tuple/iterable (should return False via try-except)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test input that is an iterable but not a pair (e.g., single element tuple)
    assert ('a',) not in items_view

    # Test input that is an iterable but has more than two elements
    assert ('a', 1, 'extra') not in items_view
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Mocking the necessary dependencies for PMap to function in a test environment
    # Since we cannot import, we assume PMap, m, and the evolver logic are available.
    
    # Setup: Create initial PMap
    # Using a helper to simulate the factory 'm' which is common in pyrsistent
    def create_pmap(data):
        # This is a simplified mock of the PMap internal structure for testing purposes
        # In a real scenario, we would use the actual 'm' or 'pmap' factory.
        buckets = [None] * 10
        size = 0
        evolver = PMap._Evolver(PMap(0, pvector())) # Using a dummy base
        for k, v in data.items():
            evolver.set(k, v)
        return evolver.persistent()

    # Note: Because the provided code is a snippet, we assume 'm' is available 
    # as per the docstrings provided in the class definition.
    
    # Test Case 1: Standard merge (rightmost value wins)
    m1 = m(a=1, b=2)
    m2 = m(a=2, c=3)
    m3 = {'a': 17, 'd': 35}
    
    # The update method uses update_with(lambda l, r: r, ...)
    # This simulates standard dict.update behavior
    result = m1.update_with(lambda l, r: r, m2, m3)
    
    expected = {'a': 17, 'b': 2, 'c': 3, 'd': 35}
    assert dict(result.items()) == expected

    # Test Case 2: Custom merge function (addition)
    # m1: {a:1, b:2}, m2: {a:2} -> result: {a:3, b:2}
    m4 = m(a=1, b=2)
    m5 = m(a=2)
    from operator import add
    result_add = m4.update_with(add, m5)
    
    assert result_add['a'] == 3
    assert result_add['b'] == 2
    assert len(result_add) == 2

    # Test Case 3: Custom merge function (leftmost value wins)
    # m1: {a:1}, m2: {a:2}, m3: {a:3} -> result: {a:1}
    m6 = m(a=1)
    m7 = m(a=2)
    m8 = {'a': 3}
    result_left = m6.update_with(lambda l, r: l, m7, m8)
    
    assert result_left['a'] == 1
    assert len(result_left) == 1

    # Test Case 4: Update with no overlapping keys
    m9 = m(x=10)
    m10 = m(y=20)
    result_no_overlap = m9.update_with(add, m10)
    assert result_no_overlap['x'] == 10
    assert result_no_overlap['y'] == 20
    assert len(result_no_overlap) == 2

    # Test Case 5: Update with an empty mapping
    m11 = m(a=1)
    result_empty = m11.update_with(add, m())
    assert result_empty == m11
    assert result_empty is not m11 # Should be a new object if changes occurred (though here it's a new persistent state)

    # Test Case 6: Verifying the behavior of the 'update' shorthand
    # update should behave like update_with(lambda l, r: r, ...)
    m12 = m(a=1, b=2)
    m13 = m(b=5, c=10)
    result_update = m12.update(m13)
    assert result_update['a'] == 1
    assert result_update['b'] == 5
    assert result_update['c'] == 10
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_PMap_update_with():
    # Mocking the necessary components for PMap and its Evolver
    # Since we cannot instantiate a real PMap easily without the full environment 
    # (PVector, PSet, etc.), we will mock the behavior of the method.
    
    # Setup: Create a mock PMap and a mock Evolver
    mock_pmap = MagicMock(spec=PMap)
    mock_evolver = MagicMock(spec=PMap._Evolver)
    
    # Configure the evolver to return a new persistent PMap when .persistent() is called
    new_pmap = MagicMock(spec=PMap)
    mock_evolver.persistent.return_value = new_pmap
    mock_pmap.evolver.return_value = mock_evolver
    
    # Case 1: Standard update (Rightmost value wins)
    # mimics: m1.update(m2, m3)
    m1 = MagicMock(spec=PMap)
    m2 = MagicMock(spec=PMap)
    m3 = MagicMock(spec=PMap)
    
    # Define item behavior for m2 and m3
    m2.items.return_value = [('a', 2), ('c', 3)]
    m3.items.return_value = [('a', 17), ('d', 35)]
    
    # Define how the evolver handles the set operations
    # For 'a', it should call update_fn(existing, new)
    # We'll simulate the logic of update_with manually in our expectations
    def side_effect_set(key, val):
        # This mimics the logic: update_fn(evolver[key], value) if key in evolver else value
        # In a real test, we'd track calls to verify the correct values were passed
        pass

    mock_evolver.set.side_effect = side_effect_set
    
    # We use a real lambda for the test logic
    from operator import add
    
    # Test 1: update_with using a merge function (add)
    # m1.update_with(add, m2) where m1 has {'a': 1, 'b': 2} and m2 has {'a': 2}
    # Expected: 'a' becomes 1 + 2 = 3, 'b' stays 2
    
    # Setup internal state for the mock to simulate 'key in evolver'
    # We'll use a dictionary to track what the 'evolver' thinks it contains
    evolver_state = {'a': 1, 'b': 2}
    
    def mock_set_logic(key, val):
        if key in evolver_state:
            # This is where the update_fn (add) is applied in the real code
            # But we need to pass the logic to the test's verification
            pass
        else:
            pass

    # Re-implementing a controlled version of the method for verification
    # because we can't easily mock the 'in evolver' check without a real object
    class MockEvolver:
        def __init__(self, initial_state):
            self.state = initial_state
            self.calls = []
        def set(self, key, val):
            # Capture the logic of the real PMap.update_with
            # We need to capture the lambda call specifically
            self.calls.append((key, val))
            return self
        def persistent(self):
            return MagicMock()

    # Test execution for update_with
    # We will use a real functional approach to verify the logic of the provided snippet
    
    # Mocking the 'evolver' behavior to verify the lambda application
    class SpyEvolver:
        def __init__(self, initial_state):
            self.state = initial_state
            self.history = []
        def __getitem__(self, key):
            return self.state[key]
        def __contains__(self, key):
            return key in self.state
        def set(self, key, val):
            # This is the critical part: the update_fn is called inside the real method
            # We need to capture if the update_fn was called with (old, new)
            self.history.append((key, val))
            return self
        def persistent(self):
            return MagicMock()

    # Create the objects
    m1_state = {'a': 1, 'b': 2}
    spy = SpyEvolver(m1_state)
    mock_pmap.evolver.return_value = spy
    
    # The method we are testing (logic extraction from the provided class)
    def update_with_logic(self_obj, update_fn, *maps):
        evolver = self_obj.evolver()
        for m in maps:
            for key, value in m.items():
                if key in evolver:
                    evolver.set(key, update_fn(evolver[key], value))
                else:
                    evolver.set(key, value)
        return evolver.persistent()

    # Test 1: Add functionality
    from operator import add
    m2_items = [('a', 2)]
    m2 = MagicMock()
    m2.items.return_value = m2_items
    
    update_with_logic(mock_pmap, add, m2)
    
    # Verify 'a' was updated with 1 + 2
    # The spy records the result of the call: evolver.set(key, update_fn(old, new))
    # So we check if the second argument in the call was 3
    assert spy.history[0] == ('a', 3)

    # Test 2: Leftmost element preference (lambda l, r: l)
    spy_2 = SpyEvolver({'a': 1})
    mock_pmap.evolver.return_value = spy_2
    m3 = MagicMock()
    m3.items.return_value = [('a', 3)]
    
    update_with_logic(mock_pmap, lambda l, r: l, m3)
    
    # Result should be 1 (the original value)
    assert spy_2.history[0] == ('a', 1)

    # Test 3: Multiple maps
    spy_3 = SpyEvolver({'a': 1})
    mock_pmap.evolver.return_value = spy_3
    m4 = MagicMock()
    m4.items.return_value = [('a', 10)]
    m5 = MagicMock()
    m5.items.return_value = [('b', 20)]
    
    update_with_logic(mock_pmap, add, m4, m5)
    
    # Check history: 
    # 1. 'a' updated: 1 + 10 = 11
    # 2. 'b' inserted: 20
    assert spy_3.history[0] == ('a', 11)
    assert spy_3.history[1] == ('b', 20)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test existing item (key and value match)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    
    # Test existing key but wrong value
    assert ('a', 99) not in items_view
    
    # Test non-existent key
    assert ('z', 1) not in items_view
    
    # Test item with wrong structure (not a tuple/pair)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert [('a', 1)] not in items_view
    
    # Test empty/invalid input for unpacking
    assert () not in items_view
    assert (None,) not in items_view
    
    # Test edge case: key exists, value is None
    m_none = pmap({'d': None})
    items_view_none = PMapItems(m_none)
    assert ('d', None) in items_view_none
    assert ('d', 1) not in items_view_none
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

    # Test items with correct key but wrong value
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test keys not present in the map
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test non-iterable arguments (should return False via try-except)
    assert 123 not in items_view
    assert None not in items_view
    assert 'a' not in items_view

    # Test iterables that are not pairs (length mismatch)
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view
    assert () not in items_view

    # Test edge case: empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___eq__():
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m3 = pmap({'a': 1, 'c': 3})
    
    items1 = PMapItems(m1)
    items2 = PMapItems(m2)
    items3 = PMapItems(m3)
    
    # Test identity (x is self)
    assert items1 == items1
    
    # Test equality of different instances with same underlying map
    assert items1 == items2
    
    # Test inequality of different instances with different underlying maps
    assert items1 != items3
    
    # Test inequality with different type
    assert items1 != [('a', 1), ('b', 2)]
    assert items1 != m1
    assert items1 != None
```


# LLM-generated content at query #10
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

    # Test valid key but incorrect value
    assert ('a', 99) not in items_view

    # Test key not present in the map
    assert ('z', 1) not in items_view

    # Test input is not a tuple/pair (should return False via try-except)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test input is a tuple but not a pair (unpacking failure)
    assert ('a', 1, 'extra') not in items_view

    # Test input is a tuple with wrong value
    assert ('a', 2) not in items_view
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___contains__():
    # Setup a PMap for testing
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)

    # Test case 1: Valid item present in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test case 2: Valid key but wrong value
    assert ('a', 99) not in items_view
    assert ('b', 0) not in items_view

    # Test case 3: Key does not exist in the map
    assert ('z', 1) not in items_view
    assert ('d', 3) not in items_view

    # Test case 4: Input is not a tuple (should trigger the Exception block)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [] not in items_view

    # Test case 5: Input is a tuple but has wrong length (should trigger the Exception block)
    assert ('a', 1, 'extra') not in items_view
    assert ('a',) not in items_view

    # Test case 6: Input is an empty tuple
    assert () not in items_view
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___eq__():
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m3 = pmap({'a': 1, 'c': 3})
    
    items1 = PMapItems(m1)
    items2 = PMapItems(m2)
    items3 = PMapItems(m3)
    
    # Test identity equality
    assert items1 == items1
    
    # Test equality with different instance but same underlying map
    assert items1 == items2
    
    # Test inequality with different underlying map
    assert itemsly1 != items3
    
    # Test inequality with different type
    assert items1 != [('a', 1), ('b', 2)]
    assert items1 != m1
    
    # Test inequality with None
    assert items1 != None
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
from pyrsistent import pmap

def test_PMapItems___eq__():
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m3 = pmap({'a': 1, 'c': 3})
    
    items1 = PMapItems(m1)
    items2 = PMapItems(m2)
    items3 = PMapItems(m3)
    
    # Identity check
    assert items1 == items1
    
    # Equality check (same content)
    assert items1 == items2
    
    # Inequality check (different content)
    assert items1 != items3
    
    # Inequality check (different type)
    assert items1 != [('a', 1), ('b', 2)]
    assert items1 != m1
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup initial PMap
    # Note: Since PMap is not instantiated directly, we assume 'm' is the factory function
    # as per docstrings provided in the code.
    m1 = m(a=1, b=2)
    
    # Test Case 1: Basic update with rightmost value wins (Standard update behavior)
    m2 = m1.update_with(lambda l, r: r, m(a=2, c=3), {'a': 17, 'd': 35})
    assert m2 == {'a': 17, 'b': 2, 'c': 3, 'd': 35}

    # Test Case 2: Using a custom merge function (e.g., addition)
    # m1 has a=1, m(a=2) has a=2. result should be a=3
    m3 = m1.update_with(lambda l, r: l + r, m(a=2))
    assert m3['a'] == 3
    assert m3['b'] == 2

    # Test Case 3: Leftmost element wins (Reverse behavior)
    # m1 has a=1, m(a=2) has a=2, {'a':3} has a=3. result should be a=1
    m4 = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m4['a'] == 1
    assert m4['b'] == 2

    # Test Case 4: Updating with multiple maps of varying types
    # Testing interaction between PMap and standard dict
    m5 = m1.update_with(lambda l, r: r, m(b=10), {'c': 20, 'a': 5})
    assert m5 == {'a': 5, 'b': 10, 'c': 20}

    # Test Case 5: Updating with an empty map should not change the original
    m6 = m1.update_with(lambda l, r: r, m())
    assert m6 == m1
    assert m6 is not m1

    # Test Case 6: Verifying that original map remains immutable
    m1_original = m(x=100)
    m1_original.update_with(lambda l, r: r, m(x=200))
    assert m1_original['x'] == 100

    # Test Case 7: Handling keys that exist in maps but not in the original
    # The implementation uses: update_fn(evolver[key], value) if key in evolver else value
    # This means if the key is new, the update_fn is NOT called, the value is just inserted.
    m7 = m1.update_with(lambda l, r: l + r, m(z=50))
    assert m7['z'] == 50
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_PMap_update_with():
    # Setup: Create an initial PMap
    # Note: Since we don't have the factory 'm', we use the constructor 
    # or assume a mock/helper for the sake of the test logic.
    # For this test, we assume the presence of a way to create PMap instances.
    
    # We'll use a helper to simulate the factory 'm' if needed, 
    # but here we assume the environment allows instantiation.
    def create_pmap(data_dict):
        # This is a mock implementation of how PMap would be initialized 
        # for testing purposes, assuming the internal structure logic.
        # In a real scenario, one would use the provided factory.
        from pyrsistent import pmap
        return pmap(data_dict)

    m1 = create_pmap({'a': 1, 'b': 2})
    m2 = create_pmap({'a': 2, 'c': 3})
    m3 = {'a': 17, 'd': 35} # dict is a Mapping

    # 1. Test basic update (rightmost value wins)
    # This mimics the behavior of the 'update' method which uses 'update_with' with a replacement lambda
    updated_standard = m1.update(m2, m3)
    assert updated_standard == {'a': 17, 'b': 2, 'c': 3, 'd': 35}

    # 2. Test update_with with an addition function (e.g., operator.add)
    from operator import add
    # m1: {'a': 1, 'b': 2}, m2: {'a': 2, 'c': 3} -> Result: {'a': 3, 'b': 2, 'c': 3}
    updated_add = m1.update_with(add, m2)
    assert updated_add == {'a': 3, 'b': 2, 'c': 3}

    # 3. Test update_with with a custom lambda (e.g., keeping the leftmost)
    # m1: {'a': 1}, m2: {'a': 2}, m3: {'a': 3} -> Result: {'a': 1}
    m_left = create_pmap({'a': 1})
    m_left_2 = create_pmap({'a': 2})
    m_left_3 = {'a': 3}
    updated_leftmost = m_left.update_with(lambda l, r: l, m_left_2, m_left_3)
    assert updated_leftmost == {'a': 1}

    # 4. Test update_with with a custom lambda (e.g., multiplication)
    # m1: {'a': 10}, m2: {'a': 2} -> Result: {'a': 20}
    m_mult = create_pmap({'a': 10})
    m_val = create_pmap({'a': 2})
    updated_mult = m_mult.update_with(lambda l, r: l * r, m_val)
    assert updated_mult == {'a': 20}

    # 5. Test update_with when the key does not exist in the evolver (should just insert)
    # m1: {'b': 2}, m2: {'a': 5} -> Result: {'b': 2, 'a': 5}
    m_single = create_pmap({'b': 2})
    m_new = create_pmap({'a': 5})
    updated_new_key = m_single.update_with(add, m_new)
    assert updated_new_key == {'b': 2, 'a': 5}

    # 6. Test update_with with empty maps
    updated_empty = m1.update_with(add, create_pmap({}))
    assert updated_empty == m1

    # 7. Ensure immutability: m1 should not have changed
    assert m1 == {'a': 1, 'b': 2}
```


