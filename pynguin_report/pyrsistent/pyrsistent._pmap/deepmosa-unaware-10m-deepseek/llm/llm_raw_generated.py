####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PMapView():
    # Test with valid PMap input
    valid_pmap = pmap({"a": 1, "b": 2})
    view = PMapView(valid_pmap)
    assert len(view) == 2
    assert view._map == valid_pmap

    # Test with regular dict (should be converted to pmap)
    regular_dict = {"x": 10, "y": 20}
    view = PMapView(regular_dict)
    assert isinstance(view._map, PMap)
    assert len(view) == 2
    assert view._map["x"] == 10
    assert view._map["y"] == 20

    # Test with other Mapping types
    from collections import OrderedDict
    ordered_dict = OrderedDict([("key1", "value1"), ("key2", "value2")])
    view = PMapView(ordered_dict)
    assert isinstance(view._map, PMap)
    assert len(view) == 2
    assert view._map["key1"] == "value1"

    # Test with empty mapping
    empty_pmap = pmap({})
    view = PMapView(empty_pmap)
    assert len(view) == 0

    # Test that __setattr__ raises TypeError
    view = PMapView(pmap({"a": 1}))
    try:
        view._map = pmap({"b": 2})
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test with non-Mapping type (should raise TypeError)
    try:
        PMapView([1, 2, 3])
        assert False, "Should have raised TypeError for non-Mapping"
    except TypeError:
        pass

    try:
        PMapView("not a mapping")
        assert False, "Should have raised TypeError for string"
    except TypeError:
        pass

    try:
        PMapView(123)
        assert False, "Should have raised TypeError for integer"
    except TypeError:
        pass

    # Test that __reversed__ raises TypeError
    view = PMapView(pmap({"a": 1, "b": 2}))
    try:
        reversed(view)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_PMapView():
    # Test with valid PMap input
    m = pmap({'a': 1, 'b': 2})
    view = PMapView(m)
    assert view._map == m
    assert len(view) == 2

    # Test with regular dict (should be converted to pmap)
    regular_dict = {'x': 10, 'y': 20}
    view = PMapView(regular_dict)
    assert isinstance(view._map, PMap)
    assert view._map == pmap(regular_dict)
    assert len(view) == 2

    # Test with Mapping that's not a dict (should be converted to pmap)
    class CustomMapping(Mapping):
        def __init__(self, data):
            self._data = data
        
        def __getitem__(self, key):
            return self._data[key]
        
        def __iter__(self):
            return iter(self._data)
        
        def __len__(self):
            return len(self._data)
    
    custom_mapping = CustomMapping({'p': 100, 'q': 200})
    view = PMapView(custom_mapping)
    assert isinstance(view._map, PMap)
    assert view._map == pmap({'p': 100, 'q': 200})
    assert len(view) == 2

    # Test immutability
    view = PMapView(pmap({'a': 1}))
    try:
        view._map = pmap({'b': 2})
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test with non-Mapping input (should raise TypeError)
    try:
        PMapView([1, 2, 3])
        assert False, "Should have raised TypeError for non-Mapping input"
    except TypeError:
        pass

    # Test with string (should raise TypeError since strings are not Mappings in this context)
    try:
        PMapView("not a mapping")
        assert False, "Should have raised TypeError for string input"
    except TypeError:
        pass

    # Test with integer (should raise TypeError)
    try:
        PMapView(42)
        assert False, "Should have raised TypeError for integer input"
    except TypeError:
        pass

    # Test __setattr__ raises TypeError
    view = PMapView(pmap({'a': 1}))
    try:
        view.some_attribute = 'value'
        assert False, "__setattr__ should have raised TypeError"
    except TypeError as e:
        assert "immutable" in str(e)

    # Test __reversed__ raises TypeError
    view = PMapView(pmap({'a': 1, 'b': 2}))
    try:
        reversed(view)
        assert False, "__reversed__ should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ("key", "value") not in empty_items
    assert (1, 2) not in empty_items
    
    # Test with non-empty pmap
    test_pmap = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(test_pmap)
    
    # Test existing items
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Test non-existing keys
    assert ("d", 4) not in items_view
    assert ("a", 2) not in items_view  # Wrong value for existing key
    assert ("b", 1) not in items_view  # Wrong value for existing key
    
    # Test wrong value types
    assert ("a", "1") not in items_view
    assert ("b", None) not in items_view
    
    # Test with non-tuple argument (should return False)
    assert "a" not in items_view
    assert 1 not in items_view
    assert ["a", 1] not in items_view
    assert None not in items_view
    
    # Test with wrong tuple length
    assert () not in items_view
    assert ("a",) not in items_view
    assert ("a", 1, "extra") not in items_view
    
    # Test with tuple containing non-hashable key
    assert ([1, 2], 3) not in items_view
    
    # Test that it works with different value types
    complex_pmap = pmap({
        "int": 42,
        "str": "hello",
        "list": [1, 2, 3],
        "none": None,
        "dict": {"nested": "value"}
    })
    complex_items = PMapItems(complex_pmap)
    
    assert ("int", 42) in complex_items
    assert ("str", "hello") in complex_items
    assert ("list", [1, 2, 3]) in complex_items
    assert ("none", None) in complex_items
    assert ("dict", {"nested": "value"}) in complex_items
    
    # Test with wrong values for complex types
    assert ("list", [1, 2]) not in complex_items
    assert ("dict", {"wrong": "value"}) not in complex_items


# LLM-generated content at query #4
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    m = pmap({})
    items_view = PMapItems(m)
    assert (1, 2) not in items_view
    assert () not in items_view
    assert (1,) not in items_view
    
    # Test with non-empty PMap
    m = pmap({1: 'a', 2: 'b', 3: 'c'})
    items_view = PMapItems(m)
    
    # Test existing items
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view
    
    # Test non-existent keys with correct value
    assert (4, 'a') not in items_view
    
    # Test existing keys with wrong values
    assert (1, 'b') not in items_view
    assert (2, 'c') not in items_view
    assert (3, 'a') not in items_view
    
    # Test with wrong argument types
    assert 1 not in items_view
    assert 'a' not in items_view
    assert [1, 'a'] not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with tuple of wrong length
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with non-tuple that can't be unpacked
    class NotUnpackable:
        pass
    
    assert NotUnpackable() not in items_view
    
    # Test with PMap containing None values
    m = pmap({1: None, 2: 0})
    items_view = PMapItems(m)
    assert (1, None) in items_view
    assert (2, 0) in items_view
    assert (1, 0) not in items_view
    
    # Test with nested PMap
    m = pmap({1: pmap({2: 3})})
    items_view = PMapItems(m)
    assert (1, pmap({2: 3})) in items_view
    assert (1, pmap({2: 4})) not in items_view
    
    # Test that it works with regular dict as argument
    m = pmap({1: {'a': 1}, 2: {'b': 2}})
    items_view = PMapItems(m)
    assert (1, {'a': 1}) in items_view
    assert (2, {'b': 2}) in items_view
    assert (1, {'a': 2}) not in items_view


# LLM-generated content at query #5
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert ("key", "value") not in empty_items
    
    # Test with non-empty pmap
    test_pmap = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(test_pmap)
    
    # Test existing items
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Test non-existing keys
    assert ("d", 4) not in items_view
    assert ("a", 2) not in items_view  # Wrong value for existing key
    
    # Test wrong value types
    assert ("a", "1") not in items_view
    assert ("b", None) not in items_view
    
    # Test with wrong argument format (not a tuple)
    assert "not_a_tuple" not in items_view
    assert 123 not in items_view
    assert ["a", 1] not in items_view
    assert {"a": 1} not in items_view
    
    # Test with tuple of wrong length
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 2, 3) not in items_view
    
    # Test with nested pmap
    nested_pmap = pmap({"x": pmap({"y": 5})})
    nested_items = PMapItems(nested_pmap)
    assert ("x", pmap({"y": 5})) in nested_items
    
    # Test that it handles exceptions gracefully
    class BadEq:
        def __eq__(self, other):
            raise ValueError("Comparison failed")
    
    bad_pmap = pmap({"key": BadEq()})
    bad_items = PMapItems(bad_pmap)
    # Should return False without raising when comparison fails
    assert ("key", "wrong_value") not in bad_items


# LLM-generated content at query #6
#--------------------------

```python
def test_PMap___eq__():
    from pyrsistent import pmap, m
    
    # Test equality with self
    pm1 = pmap({'a': 1, 'b': 2})
    assert pm1 == pm1
    assert not (pm1 != pm1)
    
    # Test equality with identical PMap
    pm2 = pmap({'a': 1, 'b': 2})
    assert pm1 == pm2
    assert not (pm1 != pm2)
    
    # Test equality with different PMap (different values)
    pm3 = pmap({'a': 1, 'b': 3})
    assert not (pm1 == pm3)
    assert pm1 != pm3
    
    # Test equality with different PMap (different keys)
    pm4 = pmap({'a': 1, 'c': 2})
    assert not (pm1 == pm4)
    assert pm1 != pm4
    
    # Test equality with regular dict
    d1 = {'a': 1, 'b': 2}
    assert pm1 == d1
    assert not (pm1 != d1)
    
    # Test equality with different dict
    d2 = {'a': 1, 'b': 3}
    assert not (pm1 == d2)
    assert pm1 != d2
    
    # Test equality with different sized mapping
    d3 = {'a': 1, 'b': 2, 'c': 3}
    assert not (pm1 == d3)
    assert pm1 != d3
    
    # Test equality with other Mapping type
    from collections.abc import Mapping
    class TestMapping(Mapping):
        def __init__(self, data):
            self._data = data
        def __getitem__(self, key):
            return self._data[key]
        def __iter__(self):
            return iter(self._data)
        def __len__(self):
            return len(self._data)
    
    tm1 = TestMapping({'a': 1, 'b': 2})
    assert pm1 == tm1
    assert not (pm1 != tm1)
    
    # Test equality with non-mapping returns NotImplemented
    result = pm1.__eq__(123)
    assert result is NotImplemented
    
    # Test that different PMaps with same content but different bucket structure compare equal
    # Create two PMaps that should have different bucket structures
    pm5 = pmap({i: i for i in range(100)})
    pm6 = pmap({i: i for i in range(100)})
    assert pm5 == pm6
    
    # Test cached hash affects equality check
    pm7 = pmap({'x': 1, 'y': 2})
    pm8 = pmap({'x': 1, 'y': 2})
    hash(pm7)  # Cache hash
    hash(pm8)  # Cache hash
    assert pm7 == pm8
    
    # Modify one and test inequality
    pm9 = pm7.set('z', 3)
    assert not (pm7 == pm9)
    assert pm7 != pm9
    
    # Test with m() factory function
    pm10 = m(a=1, b=2)
    assert pm1 == pm10
    assert not (pm1 != pm10)
    
    # Test empty PMaps
    pm_empty1 = pmap({})
    pm_empty2 = pmap({})
    d_empty = {}
    assert pm_empty1 == pm_empty2
    assert pm_empty1 == d_empty
    assert not (pm_empty1 != d_empty)


# LLM-generated content at query #7
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = empty_pmap.items()
    assert (1, 2) not in empty_items
    assert () not in empty_items
    assert (1,) not in empty_items
    
    # Test with non-empty pmap
    test_pmap = pmap({1: 'a', 2: 'b', 3: 'c'})
    items_view = test_pmap.items()
    
    # Test existing items
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view
    
    # Test non-existing items with wrong value
    assert (1, 'b') not in items_view
    assert (2, 'c') not in items_view
    assert (3, 'a') not in items_view
    
    # Test non-existing key
    assert (4, 'd') not in items_view
    assert (0, 'a') not in items_view
    
    # Test with wrong argument types
    assert 1 not in items_view
    assert 'a' not in items_view
    assert [1, 'a'] not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with tuple of wrong length
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with non-tuple argument that can't be unpacked
    assert None not in items_view
    assert 42 not in items_view
    assert "string" not in items_view
    
    # Test equality with same items from different pmap
    other_pmap = pmap({1: 'a', 2: 'b', 3: 'c'})
    other_items = other_pmap.items()
    assert (1, 'a') in other_items
    
    # Test that view reflects pmap changes through new view
    new_pmap = test_pmap.set(4, 'd')
    new_items = new_pmap.items()
    assert (4, 'd') in new_items
    assert (4, 'd') not in items_view  # Original view unchanged


# LLM-generated content at query #8
#--------------------------

```python
def test_PMap___eq__():
    from pyrsistent import pmap, m
    
    # Test equality with self
    pm1 = pmap({'a': 1, 'b': 2})
    assert pm1 == pm1
    assert not (pm1 != pm1)
    
    # Test equality with identical PMap
    pm2 = pmap({'a': 1, 'b': 2})
    assert pm1 == pm2
    assert not (pm1 != pm2)
    
    # Test inequality with different values
    pm3 = pmap({'a': 1, 'b': 3})
    assert not (pm1 == pm3)
    assert pm1 != pm3
    
    # Test inequality with different keys
    pm4 = pmap({'a': 1, 'c': 2})
    assert not (pm1 == pm4)
    assert pm1 != pm4
    
    # Test equality with regular dict
    d1 = {'a': 1, 'b': 2}
    assert pm1 == d1
    assert not (pm1 != d1)
    
    # Test inequality with different dict
    d2 = {'a': 1, 'b': 3}
    assert not (pm1 == d2)
    assert pm1 != d2
    
    # Test with different size dict
    d3 = {'a': 1, 'b': 2, 'c': 3}
    assert not (pm1 == d3)
    assert pm1 != d3
    
    # Test with empty PMap
    pm_empty1 = pmap({})
    pm_empty2 = pmap({})
    assert pm_empty1 == pm_empty2
    assert pm_empty1 == {}
    
    # Test with non-mapping type
    assert not (pm1 == [('a', 1), ('b', 2)])
    assert pm1 != [('a', 1), ('b', 2)]
    
    # Test with None
    assert not (pm1 == None)
    assert pm1 != None
    
    # Test with subclass of Mapping
    from collections.abc import Mapping
    class TestMapping(Mapping):
        def __init__(self, data):
            self._data = data
        def __getitem__(self, key):
            return self._data[key]
        def __iter__(self):
            return iter(self._data)
        def __len__(self):
            return len(self._data)
    
    tm = TestMapping({'a': 1, 'b': 2})
    assert pm1 == tm
    assert not (pm1 != tm)
    
    # Test with different PMap but same content (different bucket structure)
    pm_large = pmap({i: i for i in range(100)})
    pm_large2 = pmap({i: i for i in range(100)})
    assert pm_large == pm_large2
    
    # Test cached hash affects equality check
    pm5 = pmap({'a': 1, 'b': 2})
    pm6 = pmap({'a': 1, 'b': 2})
    hash(pm5)  # Cache hash
    hash(pm6)  # Cache hash
    assert pm5 == pm6
    
    # Test with different cached hash values
    pm7 = pmap({'a': 1, 'b': 2})
    pm8 = pmap({'a': 1, 'b': 3})
    hash(pm7)
    hash(pm8)
    assert not (pm7 == pm8)
    
    # Test with PMapItems view
    pm9 = pmap({'a': 1, 'b': 2})
    items_view = pm9.items()
    assert isinstance(items_view, type(pm9.items()))
    
    # Test equality between two PMaps with same buckets
    pm10 = pmap({'x': 10, 'y': 20})
    pm11 = pmap({'x': 10, 'y': 20})
    if pm10._buckets == pm11._buckets:
        assert pm10 == pm11


# LLM-generated content at query #9
#--------------------------

```python
def test_PMap_update_with():
    from operator import add, sub
    
    # Test basic update_with functionality
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}))
    assert result == {'a': 3, 'b': 2}
    assert isinstance(result, PMap)
    
    # Test with multiple maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}), pmap({'a': 3, 'c': 1}))
    assert result == {'a': 6, 'b': 2, 'c': 1}
    
    # Test with empty maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({}), pmap({}))
    assert result == m1
    assert result is m1  # Should return same instance when no changes
    
    # Test with different merge function (keep leftmost)
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: l, pmap({'a': 2}), pmap({'a': 3}))
    assert result == {'a': 1}
    
    # Test with keep rightmost (simulating regular update)
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: r, pmap({'a': 2}), pmap({'a': 3}))
    assert result == {'a': 3}
    
    # Test with new keys
    m1 = pmap({'a': 1})
    result = m1.update_with(add, pmap({'b': 2}), pmap({'c': 3}))
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with subtraction
    m1 = pmap({'a': 10, 'b': 5})
    result = m1.update_with(sub, pmap({'a': 3, 'b': 2}))
    assert result == {'a': 7, 'b': 3}
    
    # Test with complex merge function
    m1 = pmap({'a': [1], 'b': 2})
    result = m1.update_with(lambda l, r: l + r, pmap({'a': [2, 3]}))
    assert result == {'a': [1, 2, 3], 'b': 2}
    
    # Test with regular dict as argument
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, {'a': 2, 'c': 3})
    assert result == {'a': 3, 'b': 2, 'c': 3}
    
    # Test immutability - original should not change
    m1 = pmap({'a': 1, 'b': 2})
    original_hash = hash(m1)
    result = m1.update_with(add, pmap({'a': 2}))
    assert m1 == {'a': 1, 'b': 2}
    assert hash(m1) == original_hash
    
    # Test with no maps provided
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add)
    assert result == m1
    assert result is m1
    
    # Test with nested updates
    m1 = pmap({'a': pmap({'x': 1}), 'b': 2})
    result = m1.update_with(
        lambda l, r: l.update_with(add, r) if isinstance(l, PMap) else r,
        pmap({'a': pmap({'x': 2, 'y': 3})})
    )
    assert result['a'] == {'x': 3, 'y': 3}
    assert result['b'] == 2


# LLM-generated content at query #10
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with empty PMap
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert (1, 2) not in empty_items
    assert () not in empty_items
    assert (1,) not in empty_items
    
    # Test with non-empty PMap
    test_map = pmap({1: 'a', 2: 'b', 3: 'c'})
    items_view = PMapItems(test_map)
    
    # Test existing items
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view
    
    # Test non-existing items with wrong value
    assert (1, 'b') not in items_view
    assert (2, 'c') not in items_view
    assert (3, 'a') not in items_view
    
    # Test non-existing key
    assert (4, 'd') not in items_view
    assert (0, 'a') not in items_view
    
    # Test with wrong argument types
    assert 1 not in items_view
    assert 'a' not in items_view
    assert [1, 'a'] not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with tuple of wrong length
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with non-tuple that can't be unpacked
    class NotUnpackable:
        pass
    
    assert NotUnpackable() not in items_view
    
    # Test with PMap containing different value types
    complex_map = pmap({'key': [1, 2, 3], 42: {'nested': 'value'}})
    complex_items = PMapItems(complex_map)
    
    assert ('key', [1, 2, 3]) in complex_items
    assert (42, {'nested': 'value'}) in complex_items
    assert ('key', [1, 2]) not in complex_items
    assert (42, {'nested': 'wrong'}) not in complex_items
    
    # Test that it works with regular dict conversion
    regular_dict = {1: 'a', 2: 'b'}
    regular_pmap = pmap(regular_dict)
    regular_items = PMapItems(regular_pmap)
    
    for k, v in regular_dict.items():
        assert (k, v) in regular_items


# LLM-generated content at query #11
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert ("key", "value") not in empty_items
    
    # Test with non-empty pmap
    test_pmap = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(test_pmap)
    
    # Test existing items
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Test non-existing keys
    assert ("d", 4) not in items_view
    assert ("a", 2) not in items_view  # Wrong value for existing key
    assert ("b", 1) not in items_view  # Wrong value for existing key
    
    # Test with wrong value type
    assert ("a", "1") not in items_view
    
    # Test with non-tuple argument (should return False)
    assert "a" not in items_view
    assert 1 not in items_view
    assert ["a", 1] not in items_view
    assert {"a": 1} not in items_view
    
    # Test with malformed tuple (wrong length)
    assert (1, 2, 3) not in items_view
    assert (1,) not in items_view
    
    # Test with nested structures
    complex_pmap = pmap({"x": [1, 2], "y": {"z": 3}})
    complex_items = PMapItems(complex_pmap)
    assert ("x", [1, 2]) in complex_items
    assert ("y", {"z": 3}) in complex_items
    assert ("x", [1]) not in complex_items  # Different list
    
    # Test equality comparison (not identity)
    another_pmap = pmap({"a": 1, "b": 2, "c": 3})
    another_items = PMapItems(another_pmap)
    assert ("a", 1) in another_items  # Same content, different object
    
    # Test that it works with PMapView base class initialization
    view_as_base = PMapView(test_pmap)
    items_from_base = PMapItems(view_as_base._map)
    assert ("a", 1) in items_from_base


# LLM-generated content at query #12
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert ("key", "value") not in empty_items
    assert (None, None) not in empty_items

    # Test with non-empty PMap
    test_pmap = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(test_pmap)
    
    # Items that exist
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Items with wrong value
    assert ("a", 2) not in items_view
    assert ("b", 1) not in items_view
    assert ("c", 4) not in items_view
    
    # Items with non-existent key
    assert ("d", 1) not in items_view
    assert ("x", 100) not in items_view
    
    # Test with non-tuple argument (should return False)
    assert "not a tuple" not in items_view
    assert 42 not in items_view
    assert None not in items_view
    assert ["a", 1] not in items_view
    
    # Test with wrong-sized tuple
    assert (1,) not in items_view
    assert (1, 2, 3) not in items_view
    
    # Test with PMap containing None values
    pmap_with_none = pmap({"x": None, "y": 0})
    items_with_none = PMapItems(pmap_with_none)
    assert ("x", None) in items_with_none
    assert ("y", 0) in items_with_none
    assert ("x", 0) not in items_with_none
    assert ("y", None) not in items_with_none
    
    # Test with nested PMap
    nested_pmap = pmap({"outer": pmap({"inner": "value"})})
    nested_items = PMapItems(nested_pmap)
    assert ("outer", pmap({"inner": "value"})) in nested_items
    assert ("outer", pmap({"inner": "wrong"})) not in nested_items
    
    # Test that it works with regular dict conversion
    regular_dict = {"key": "value"}
    regular_pmap = pmap(regular_dict)
    regular_items = PMapItems(regular_pmap)
    assert ("key", "value") in regular_items


# LLM-generated content at query #13
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    m = pmap({})
    items_view = PMapItems(m)
    assert (1, 2) not in items_view
    assert () not in items_view
    assert (1,) not in items_view
    
    # Test with single item PMap
    m = pmap({1: 'a'})
    items_view = PMapItems(m)
    assert (1, 'a') in items_view
    assert (1, 'b') not in items_view
    assert (2, 'a') not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with multiple items PMap
    m = pmap({1: 'a', 2: 'b', 3: 'c'})
    items_view = PMapItems(m)
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view
    assert (1, 'b') not in items_view
    assert (4, 'd') not in items_view
    
    # Test with non-tuple argument
    m = pmap({1: 'a'})
    items_view = PMapItems(m)
    assert 1 not in items_view
    assert 'a' not in items_view
    assert [1, 'a'] not in items_view
    assert {'key': 1, 'value': 'a'} not in items_view
    
    # Test with wrong length tuple
    m = pmap({1: 'a'})
    items_view = PMapItems(m)
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with tuple that can't unpack
    m = pmap({1: 'a'})
    items_view = PMapItems(m)
    assert None not in items_view
    assert 42 not in items_view
    assert "not_a_tuple" not in items_view
    
    # Test equality comparison (not identity)
    m1 = pmap({1: 'a', 2: 'b'})
    m2 = pmap({1: 'a', 2: 'b'})
    items_view1 = PMapItems(m1)
    items_view2 = PMapItems(m2)
    assert (1, 'a') in items_view1
    assert (1, 'a') in items_view2
    assert (2, 'b') in items_view1
    assert (2, 'b') in items_view2


# LLM-generated content at query #14
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with empty PMap
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert (1, 2) not in empty_items
    assert () not in empty_items
    assert ("key", "value") not in empty_items
    
    # Test with non-empty PMap
    test_map = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(test_map)
    
    # Test existing items
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Test non-existing keys
    assert ("d", 4) not in items_view
    assert ("a", 2) not in items_view  # Wrong value for existing key
    assert ("b", 1) not in items_view  # Wrong value for existing key
    
    # Test wrong value types
    assert ("a", "1") not in items_view
    assert ("b", None) not in items_view
    
    # Test with wrong argument format (not a 2-tuple)
    assert "not_a_tuple" not in items_view
    assert 123 not in items_view
    assert None not in items_view
    assert ("a", 1, "extra") not in items_view  # 3-tuple
    assert ("a") not in items_view  # 1-tuple
    assert [] not in items_view
    
    # Test with PMap containing different value types
    complex_map = pmap({"x": [1, 2], "y": {"nested": "value"}, "z": None})
    complex_items = PMapItems(complex_map)
    
    assert ("x", [1, 2]) in complex_items
    assert ("y", {"nested": "value"}) in complex_items
    assert ("z", None) in complex_items
    assert ("x", [1]) not in complex_items  # Different list
    assert ("y", "wrong") not in complex_items
    
    # Test that it works with regular dict conversion
    regular_dict = {"key": "value"}
    regular_pmap = pmap(regular_dict)
    regular_items = PMapItems(regular_pmap)
    assert ("key", "value") in regular_items
    assert ("key", "wrong") not in regular_items
    
    # Test with nested PMap
    nested_map = pmap({"outer": pmap({"inner": 42})})
    nested_items = PMapItems(nested_map)
    assert ("outer", pmap({"inner": 42})) in nested_items
    assert ("outer", {"inner": 42}) not in nested_items  # Different type
    
    # Test that __contains__ doesn't modify the original map
    original_size = len(test_map)
    _ = ("a", 1) in items_view
    _ = ("x", 99) in items_view
    assert len(test_map) == original_size
    assert test_map["a"] == 1


# LLM-generated content at query #15
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with functionality
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}))
    assert result == {'a': 3, 'b': 2}
    
    # Test with multiple maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}), {'a': 3, 'c': 1})
    assert result == {'a': 6, 'b': 2, 'c': 1}
    
    # Test with lambda that keeps leftmost value
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: l, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 1}
    
    # Test with lambda that keeps rightmost value (default update behavior)
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: r, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 3}
    
    # Test with new keys
    m1 = pmap({'a': 1})
    result = m1.update_with(add, pmap({'b': 2}), {'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with empty maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, {}, pmap({}))
    assert result == {'a': 1, 'b': 2}
    assert result is m1  # Should return same instance when no changes
    
    # Test with complex merge function
    m1 = pmap({'a': [1], 'b': [2]})
    result = m1.update_with(lambda l, r: l + r, pmap({'a': [3], 'c': [4]}))
    assert result == {'a': [1, 3], 'b': [2], 'c': [4]}
    
    # Test that original map is unchanged
    m1 = pmap({'a': 1, 'b': 2})
    original_hash = hash(m1)
    m2 = m1.update_with(add, pmap({'a': 2}))
    assert m1 == {'a': 1, 'b': 2}
    assert hash(m1) == original_hash
    assert m2 == {'a': 3, 'b': 2}
    
    # Test with non-PMap mappings
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, {'a': 2}, pmap({'a': 3}))
    assert result == {'a': 6, 'b': 2}
    
    # Test with update_fn that returns different type
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(lambda l, r: str(l) + str(r), pmap({'a': 2}))
    assert result == {'a': '12', 'b': 2}


# LLM-generated content at query #16
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert () not in empty_items
    assert (1,) not in empty_items
    
    # Test with non-empty PMap
    test_pmap = pmap({1: 'a', 2: 'b', 3: 'c'})
    items_view = PMapItems(test_pmap)
    
    # Test existing items
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view
    
    # Test non-existing items with wrong value
    assert (1, 'b') not in items_view
    assert (2, 'a') not in items_view
    assert (3, 'x') not in items_view
    
    # Test non-existing key
    assert (4, 'd') not in items_view
    assert (0, 'a') not in items_view
    
    # Test with wrong argument type (not a tuple)
    assert 1 not in items_view
    assert 'a' not in items_view
    assert [1, 'a'] not in items_view
    assert {1: 'a'} not in items_view
    
    # Test with tuple of wrong length
    assert (1,) not in items_view
    assert (1, 'a', 'extra') not in items_view
    assert () not in items_view
    
    # Test with non-hashable key in tuple
    assert ([1], 'a') not in items_view
    
    # Test that it works with different value types
    complex_pmap = pmap({'key': [1, 2, 3], 42: {'nested': 'dict'}})
    complex_items = PMapItems(complex_pmap)
    assert ('key', [1, 2, 3]) in complex_items
    assert (42, {'nested': 'dict'}) in complex_items
    assert ('key', [1, 2]) not in complex_items
    
    # Test that it uses value equality, not identity
    same_list = [1, 2, 3]
    assert ('key', same_list) in complex_items
    
    # Test with PMap containing None values
    none_pmap = pmap({1: None, 2: None})
    none_items = PMapItems(none_pmap)
    assert (1, None) in none_items
    assert (2, None) in none_items
    assert (1, 0) not in none_items


# LLM-generated content at query #17
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with valid key-value pair that exists
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test with valid key but wrong value
    assert ('a', 2) not in items_view
    assert ('b', 1) not in items_view
    
    # Test with non-existent key
    assert ('d', 4) not in items_view
    assert ('x', 1) not in items_view
    
    # Test with wrong argument type (not a tuple)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert ['a', 1] not in items_view
    assert {'a': 1} not in items_view
    
    # Test with tuple of wrong length
    assert () not in items_view
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    
    # Test with nested structures
    nested_map = pmap({'x': [1, 2], 'y': {'z': 3}})
    nested_items = PMapItems(nested_map)
    assert ('x', [1, 2]) in nested_items
    assert ('y', {'z': 3}) in nested_items
    assert ('x', [1]) not in nested_items
    
    # Test that it works with the same value but different identity
    lst = [1, 2]
    map_with_list = pmap({'key': lst})
    items_with_list = PMapItems(map_with_list)
    assert ('key', [1, 2]) in items_with_list


# LLM-generated content at query #18
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with functionality
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}))
    assert result == {'a': 3, 'b': 2}
    
    # Test with multiple maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}), pmap({'a': 3, 'c': 1}))
    assert result == {'a': 6, 'b': 2, 'c': 1}
    
    # Test with lambda that keeps leftmost value
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: l, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 1}
    
    # Test with lambda that keeps rightmost value (default behavior)
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: r, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 3}
    
    # Test with empty map
    m1 = pmap()
    result = m1.update_with(add, pmap({'a': 1, 'b': 2}))
    assert result == {'a': 1, 'b': 2}
    
    # Test with no additional maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add)
    assert result == m1
    assert result is m1
    
    # Test with new keys
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: l + r, pmap({'b': 2}))
    assert result == {'a': 1, 'b': 2}
    
    # Test with complex update function
    m1 = pmap({'a': [1], 'b': 2})
    result = m1.update_with(lambda l, r: l + r, pmap({'a': [2, 3]}))
    assert result == {'a': [1, 2, 3], 'b': 2}
    
    # Test that original map is unchanged
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 2})
    result = m1.update_with(add, m2)
    assert m1 == {'a': 1, 'b': 2}
    assert m2 == {'a': 2}
    assert result == {'a': 3, 'b': 2}
    
    # Test with different map types
    m1 = pmap({'a': 1})
    result = m1.update_with(add, {'a': 2}, pmap({'a': 3}))
    assert result == {'a': 6}
    
    # Test with update function that returns None
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: None, pmap({'a': 2}))
    assert result == {'a': None}


# LLM-generated content at query #19
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    m = pmap({})
    items_view = PMapItems(m)
    assert (1, 2) not in items_view
    assert () not in items_view
    assert ("key", "value") not in items_view

    # Test with non-empty PMap
    m = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(m)
    
    # Items that are present
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Items with wrong value
    assert ("a", 2) not in items_view
    assert ("b", 1) not in items_view
    
    # Items with non-existent key
    assert ("d", 4) not in items_view
    assert ("x", 1) not in items_view
    
    # Test with wrong argument type (not a tuple)
    assert "not_a_tuple" not in items_view
    assert 42 not in items_view
    assert ["a", 1] not in items_view
    assert {"a": 1} not in items_view
    
    # Test with tuple of wrong length
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 2, 3) not in items_view
    
    # Test with nested PMap
    m = pmap({"x": {"nested": "value"}, "y": [1, 2, 3]})
    items_view = PMapItems(m)
    assert ("x", {"nested": "value"}) in items_view
    assert ("y", [1, 2, 3]) in items_view
    assert ("x", "wrong_value") not in items_view
    
    # Test that it works with the same semantics as dict.items()
    regular_dict = {"a": 1, "b": 2}
    pmap_dict = pmap(regular_dict)
    regular_items = regular_dict.items()
    pmap_items = PMapItems(pmap_dict)
    
    # Same items should be in both
    for item in regular_items:
        assert item in pmap_items
    
    # Non-items should not be in either
    assert ("a", 2) not in regular_items
    assert ("a", 2) not in pmap_items


# LLM-generated content at query #20
#--------------------------

```python
def test_PMap_update_with():
    from operator import add, mul
    from pyrsistent import m

    # Test basic update_with functionality
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}
    assert isinstance(result, PMap)

    # Test with multiple maps
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2), {'a': 3, 'c': 5})
    assert result == {'a': 6, 'b': 2, 'c': 5}

    # Test with lambda that keeps leftmost value
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1, 'b': 2}

    # Test with lambda that keeps rightmost value (default update behavior)
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert result == {'a': 3, 'b': 2}

    # Test with multiplication operator
    m1 = m(a=2, b=3)
    result = m1.update_with(mul, m(a=3, c=4))
    assert result == {'a': 6, 'b': 3, 'c': 4}

    # Test with empty maps
    m1 = m(a=1, b=2)
    result = m1.update_with(add, {}, m())
    assert result == {'a': 1, 'b': 2}
    assert result is m1  # Should return same instance when no changes

    # Test with new keys
    m1 = m(a=1)
    result = m1.update_with(add, m(b=2), {'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}

    # Test with complex update function
    m1 = m(a=10, b=20)
    def complex_update(left, right):
        return left + right if left > right else right - left
    result = m1.update_with(complex_update, m(a=5, b=30, c=40))
    assert result == {'a': 15, 'b': 10, 'c': 40}

    # Test that original map is unchanged
    m1 = m(a=1, b=2)
    original_hash = hash(m1)
    result = m1.update_with(add, m(a=2))
    assert m1 == {'a': 1, 'b': 2}
    assert hash(m1) == original_hash
    assert result != m1

    # Test with dict and PMap mixed
    m1 = m(a=1)
    result = m1.update_with(add, {'a': 2}, m(b=3))
    assert result == {'a': 3, 'b': 3}

    # Test with None values
    m1 = m(a=None, b=2)
    def replace_none(left, right):
        return right if left is None else left
    result = m1.update_with(replace_none, m(a=5, b=None, c=10))
    assert result == {'a': 5, 'b': 2, 'c': 10}

    # Test that update_fn is called with correct arguments
    call_log = []
    def logging_update(left, right):
        call_log.append((left, right))
        return right
    m1 = m(a=1, b=2)
    result = m1.update_with(logging_update, m(a=10, c=30))
    assert call_log == [(1, 10), (2, 30)]


# LLM-generated content at query #21
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert ("key", "value") not in empty_items
    assert (None, None) not in empty_items

    # Test with non-empty PMap
    test_pmap = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(test_pmap)
    
    # Items that exist
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Items with wrong value
    assert ("a", 2) not in items_view
    assert ("b", 1) not in items_view
    assert ("c", 4) not in items_view
    
    # Keys that don't exist
    assert ("d", 1) not in items_view
    assert ("x", 100) not in items_view
    
    # Test with non-tuple argument (should return False)
    assert "a" not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert ["a", 1] not in items_view
    
    # Test with wrong-sized tuple
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 2, 3) not in items_view
    
    # Test with PMap containing None values
    pmap_with_none = pmap({"x": None, "y": 0})
    items_none = PMapItems(pmap_with_none)
    assert ("x", None) in items_none
    assert ("y", 0) in items_none
    assert ("x", 0) not in items_none
    assert ("y", None) not in items_none
    
    # Test with nested PMap
    nested_pmap = pmap({"outer": pmap({"inner": "value"})})
    nested_items = PMapItems(nested_pmap)
    assert ("outer", pmap({"inner": "value"})) in nested_items
    assert ("outer", pmap({"inner": "wrong"})) not in nested_items
    
    # Test that __contains__ works with same instance
    assert items_view is items_view


# LLM-generated content at query #22
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert ("key", "value") not in empty_items
    
    # Test with non-empty pmap
    test_pmap = pmap({"a": 1, "b": 2, "c": 3})
    items = PMapItems(test_pmap)
    
    # Test existing items
    assert ("a", 1) in items
    assert ("b", 2) in items
    assert ("c", 3) in items
    
    # Test non-existing keys
    assert ("d", 4) not in items
    assert ("a", 4) not in items  # Wrong value for existing key
    
    # Test wrong value type
    assert ("a", "1") not in items
    
    # Test with non-tuple argument (should return False)
    assert "a" not in items
    assert 1 not in items
    assert ["a", 1] not in items
    assert {"a": 1} not in items
    
    # Test with malformed tuple (wrong length)
    assert (1, 2, 3) not in items
    assert (1,) not in items
    
    # Test with nested structures
    nested_pmap = pmap({"x": [1, 2], "y": {"z": 3}})
    nested_items = PMapItems(nested_pmap)
    assert ("x", [1, 2]) in nested_items
    assert ("y", {"z": 3}) in nested_items
    
    # Test equality comparison (not identity)
    assert ("a", 1) in items  # Same value
    assert not (("a", 1.0) in items)  # Different type
    
    # Test that it works with PMapView initialization
    view = PMapView(test_pmap)
    items_from_view = PMapItems(view._map)
    assert ("a", 1) in items_from_view


# LLM-generated content at query #23
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    m = pmap({})
    items_view = PMapItems(m)
    assert (1, 2) not in items_view
    assert () not in items_view
    assert (1,) not in items_view

    # Test with single item PMap
    m = pmap({1: 'a'})
    items_view = PMapItems(m)
    assert (1, 'a') in items_view
    assert (1, 'b') not in items_view
    assert (2, 'a') not in items_view
    assert (1, 'a', 'extra') not in items_view

    # Test with multiple items PMap
    m = pmap({1: 'a', 2: 'b', 3: 'c'})
    items_view = PMapItems(m)
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view
    assert (1, 'b') not in items_view
    assert (4, 'd') not in items_view

    # Test with non-tuple argument
    m = pmap({1: 'a'})
    items_view = PMapItems(m)
    assert 1 not in items_view
    assert 'a' not in items_view
    assert [1, 'a'] not in items_view
    assert {'key': 1, 'value': 'a'} not in items_view

    # Test with tuple of wrong length
    m = pmap({1: 'a'})
    items_view = PMapItems(m)
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 'a', 'extra') not in items_view

    # Test with correct key but wrong value
    m = pmap({1: 'a', 2: 'b'})
    items_view = PMapItems(m)
    assert (1, 'b') not in items_view
    assert (2, 'a') not in items_view

    # Test that it works with different value types
    m = pmap({'a': 1, 'b': [1, 2], 'c': {'nested': 'dict'}})
    items_view = PMapItems(m)
    assert ('a', 1) in items_view
    assert ('b', [1, 2]) in items_view
    assert ('c', {'nested': 'dict'}) in items_view
    assert ('b', [1]) not in items_view

    # Test that PMapItems view works after PMap modification
    m1 = pmap({1: 'a'})
    items_view1 = PMapItems(m1)
    m2 = m1.set(2, 'b')
    items_view2 = PMapItems(m2)
    assert (1, 'a') in items_view1
    assert (2, 'b') not in items_view1
    assert (1, 'a') in items_view2
    assert (2, 'b') in items_view2


# LLM-generated content at query #24
#--------------------------

```python
def test_PMap___eq__():
    from pyrsistent import pmap, m
    
    # Test equality with self
    pm1 = pmap({'a': 1, 'b': 2})
    assert pm1 == pm1
    assert not (pm1 != pm1)
    
    # Test equality with identical PMap
    pm2 = pmap({'a': 1, 'b': 2})
    assert pm1 == pm2
    assert not (pm1 != pm2)
    
    # Test inequality with different PMap
    pm3 = pmap({'a': 1, 'b': 3})
    assert pm1 != pm3
    assert not (pm1 == pm3)
    
    # Test equality with regular dict
    d = {'a': 1, 'b': 2}
    assert pm1 == d
    assert not (pm1 != d)
    
    # Test inequality with different dict
    d2 = {'a': 1, 'b': 3}
    assert pm1 != d2
    assert not (pm1 == d2)
    
    # Test with different size dict
    d3 = {'a': 1, 'b': 2, 'c': 3}
    assert pm1 != d3
    assert not (pm1 == d3)
    
    # Test with empty PMap
    pm_empty1 = pmap({})
    pm_empty2 = pmap({})
    assert pm_empty1 == pm_empty2
    assert pm_empty1 == {}
    
    # Test with nested structures
    pm_nested1 = pmap({'a': pmap({'b': 1})})
    pm_nested2 = pmap({'a': pmap({'b': 1})})
    assert pm_nested1 == pm_nested2
    
    # Test with different types (should return NotImplemented)
    result = pm1.__eq__(123)
    assert result is NotImplemented
    
    # Test with other Mapping types
    from collections.abc import Mapping
    class CustomMapping(Mapping):
        def __init__(self, data):
            self._data = data
        def __getitem__(self, key):
            return self._data[key]
        def __iter__(self):
            return iter(self._data)
        def __len__(self):
            return len(self._data)
    
    custom_map = CustomMapping({'a': 1, 'b': 2})
    assert pm1 == custom_map
    
    # Test PMaps with same content but different bucket structure
    # (after rehashing/evolution)
    pm4 = pmap({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6})
    pm5 = pm4.discard('f').set('f', 6)
    assert pm4 == pm5
    
    # Test cached hash affects equality check
    pm6 = pmap({'x': 10, 'y': 20})
    pm7 = pmap({'x': 10, 'y': 20})
    hash(pm6)
    hash(pm7)
    assert pm6 == pm7
    
    # Test with PMapItems view
    pm8 = pmap({'a': 1, 'b': 2})
    items_view = pm8.items()
    assert items_view == items_view
    
    # Test with different PMapItems (should compare underlying maps)
    pm9 = pmap({'a': 1, 'b': 2})
    items_view1 = pm8.items()
    items_view2 = pm9.items()
    assert items_view1 == items_view2


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    m = pmap({})
    items_view = PMapItems(m)
    assert (1, 2) not in items_view
    assert () not in items_view
    assert ("key", "value") not in items_view

    # Test with non-empty PMap
    m = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(m)
    
    # Items that exist
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Items with wrong value
    assert ("a", 2) not in items_view
    assert ("b", 1) not in items_view
    assert ("c", 4) not in items_view
    
    # Items with non-existent key
    assert ("d", 1) not in items_view
    assert ("x", 100) not in items_view
    
    # Test with wrong argument type (not a tuple)
    assert "not a tuple" not in items_view
    assert 42 not in items_view
    assert None not in items_view
    assert ["a", 1] not in items_view
    
    # Test with wrong tuple length
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 2, 3) not in items_view
    
    # Test with PMap containing different value types
    m = pmap({"int": 42, "str": "hello", "list": [1, 2], "none": None})
    items_view = PMapItems(m)
    
    assert ("int", 42) in items_view
    assert ("str", "hello") in items_view
    assert ("list", [1, 2]) in items_view
    assert ("none", None) in items_view
    
    # Test with nested PMap
    inner = pmap({"x": 10})
    outer = pmap({"inner": inner})
    items_view = PMapItems(outer)
    assert ("inner", inner) in items_view
    assert ("inner", pmap({"x": 10})) in items_view
    assert ("inner", pmap({"x": 20})) not in items_view
    
    # Test that __contains__ works with the same instance
    assert items_view is items_view


# LLM-generated content at query #2
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with functionality
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}))
    assert result == {'a': 3, 'b': 2}
    
    # Test with multiple maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}), pmap({'a': 3, 'c': 1}))
    assert result == {'a': 6, 'b': 2, 'c': 1}
    
    # Test with lambda that keeps leftmost value
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: l, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 1}
    
    # Test with lambda that keeps rightmost value (default behavior)
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: r, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 3}
    
    # Test with empty map
    m1 = pmap()
    result = m1.update_with(add, {'a': 1, 'b': 2})
    assert result == {'a': 1, 'b': 2}
    
    # Test with no additional maps (should return unchanged)
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add)
    assert result == m1
    assert result is m1
    
    # Test with new keys
    m1 = pmap({'a': 1})
    result = m1.update_with(add, {'b': 2, 'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with complex update function
    m1 = pmap({'a': 10, 'b': 20})
    def multiply_then_add(left, right):
        return left * 2 + right
    result = m1.update_with(multiply_then_add, {'a': 5, 'b': 10})
    assert result == {'a': 25, 'b': 50}  # a: 10*2 + 5 = 25, b: 20*2 + 10 = 50
    
    # Test that original map is unchanged
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 2})
    result = m1.update_with(add, m2)
    assert m1 == {'a': 1, 'b': 2}
    assert m2 == {'a': 2}
    assert result == {'a': 3, 'b': 2}
    
    # Test with different map types mixed
    m1 = pmap({'a': 1})
    result = m1.update_with(add, {'a': 2}, pmap({'a': 3}), dict(a=4))
    assert result == {'a': 10}
    
    # Test with update function that returns None (edge case)
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: None, {'a': 2})
    assert result == {'a': None}


# LLM-generated content at query #3
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert () not in empty_items
    assert (1,) not in empty_items

    # Test with non-empty PMap
    test_pmap = pmap({1: 'a', 2: 'b', 3: 'c'})
    items_view = PMapItems(test_pmap)
    
    # Test existing items
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view
    
    # Test non-existing items with wrong value
    assert (1, 'b') not in items_view
    assert (2, 'a') not in items_view
    assert (3, 'd') not in items_view
    
    # Test non-existing key
    assert (4, 'a') not in items_view
    assert (0, 'z') not in items_view
    
    # Test with wrong argument types
    assert 1 not in items_view
    assert 'a' not in items_view
    assert [1, 'a'] not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with tuple of wrong length
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with non-tuple argument that can't be unpacked
    assert None not in items_view
    assert 42 not in items_view
    assert "string" not in items_view
    
    # Test that it works with different value types
    complex_pmap = pmap({'x': [1, 2], 'y': {'nested': 'dict'}, 'z': None})
    complex_items = PMapItems(complex_pmap)
    assert ('x', [1, 2]) in complex_items
    assert ('y', {'nested': 'dict'}) in complex_items
    assert ('z', None) in complex_items
    assert ('x', [1]) not in complex_items
    assert ('y', {'nested': 'wrong'}) not in complex_items
    
    # Test that it's not affected by mutations to original data
    mutable_list = [1, 2]
    mutating_pmap = pmap({'key': mutable_list})
    mutating_items = PMapItems(mutating_pmap)
    assert ('key', [1, 2]) in mutating_items
    mutable_list.append(3)
    # The PMap should still contain the original list
    assert ('key', [1, 2]) in mutating_items
    assert ('key', [1, 2, 3]) not in mutating_items


# LLM-generated content at query #4
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert () not in empty_items
    assert (1,) not in empty_items
    
    # Test with non-empty PMap
    test_pmap = pmap({1: 'a', 2: 'b', 3: 'c'})
    items_view = PMapItems(test_pmap)
    
    # Test existing items
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view
    
    # Test non-existing items with wrong value
    assert (1, 'b') not in items_view
    assert (2, 'a') not in items_view
    assert (3, 'x') not in items_view
    
    # Test non-existing key
    assert (4, 'd') not in items_view
    assert (0, 'a') not in items_view
    
    # Test with wrong argument types
    assert 1 not in items_view
    assert 'a' not in items_view
    assert [1, 'a'] not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with tuple of wrong length
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with non-tuple that can't be unpacked
    assert None not in items_view
    assert 42 not in items_view
    assert "string" not in items_view
    
    # Test that it works with different value types
    complex_pmap = pmap({'key': [1, 2, 3], 42: {'nested': 'value'}})
    complex_items = PMapItems(complex_pmap)
    assert ('key', [1, 2, 3]) in complex_items
    assert (42, {'nested': 'value'}) in complex_items
    assert ('key', [1, 2]) not in complex_items
    
    # Test that view reflects changes to underlying PMap
    base_map = pmap({1: 'original'})
    view = PMapItems(base_map)
    assert (1, 'original') in view
    updated_map = base_map.set(1, 'updated')
    new_view = PMapItems(updated_map)
    assert (1, 'updated') in new_view
    assert (1, 'original') not in new_view


# LLM-generated content at query #5
#--------------------------

```python
def test_PMap___eq__():
    from pyrsistent import pmap, m
    from collections.abc import Mapping

    # Test equality with self
    m1 = pmap({"a": 1, "b": 2})
    assert m1 == m1
    assert not (m1 != m1)

    # Test equality with identical PMap
    m2 = pmap({"a": 1, "b": 2})
    assert m1 == m2
    assert not (m1 != m2)

    # Test inequality with different PMap
    m3 = pmap({"a": 1, "b": 3})
    assert m1 != m3
    assert not (m1 == m3)

    # Test equality with dict
    d1 = {"a": 1, "b": 2}
    assert m1 == d1
    assert not (m1 != d1)

    # Test inequality with dict
    d2 = {"a": 1, "b": 3}
    assert m1 != d2
    assert not (m1 == d2)

    # Test equality with other Mapping type
    class CustomMapping(Mapping):
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    cm1 = CustomMapping({"a": 1, "b": 2})
    assert m1 == cm1
    assert not (m1 != cm1)

    # Test inequality with different length
    m4 = pmap({"a": 1, "b": 2, "c": 3})
    assert m1 != m4
    assert not (m1 == m4)

    # Test equality with empty PMap
    m5 = pmap({})
    m6 = pmap({})
    assert m5 == m6
    assert m5 == {}
    assert {} == m5

    # Test inequality with non-Mapping
    assert m1 != "not a mapping"
    assert m1 != 123
    assert m1 != [1, 2, 3]

    # Test NotImplemented for non-Mapping
    result = m1.__eq__("not a mapping")
    assert result is NotImplemented

    # Test hash cache optimization
    m7 = pmap({"x": 10, "y": 20})
    m8 = pmap({"x": 10, "y": 20})
    hash(m7)
    hash(m8)
    assert m7 == m8

    # Test with different hash cache values
    m9 = pmap({"x": 10, "y": 20})
    m10 = pmap({"x": 10, "y": 21})
    hash(m9)
    hash(m10)
    assert m9 != m10

    # Test equality with same buckets
    m11 = pmap({"a": 1})
    m12 = pmap({"a": 1})
    assert m11._buckets == m12._buckets
    assert m11 == m12

    # Test equality with different buckets but same content
    m13 = pmap({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    m14 = pmap({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    assert m13 == m14

    # Test using m() factory function
    m15 = m(a=1, b=2)
    m16 = m(a=1, b=2)
    assert m15 == m16
    assert m15 == {"a": 1, "b": 2}


# LLM-generated content at query #6
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert ("key", "value") not in empty_items
    assert ((),) not in empty_items

    # Test with non-empty PMap
    test_pmap = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(test_pmap)
    
    # Test existing items
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Test non-existing keys
    assert ("d", 4) not in items_view
    assert ("a", 2) not in items_view  # Wrong value for existing key
    assert ("b", 1) not in items_view  # Wrong value for existing key
    
    # Test wrong value types
    assert ("a", "1") not in items_view
    assert ("b", None) not in items_view
    
    # Test with non-tuple argument (should return False)
    assert "a" not in items_view
    assert 1 not in items_view
    assert ["a", 1] not in items_view
    assert {"a": 1} not in items_view
    
    # Test with wrong tuple length
    assert ("a",) not in items_view
    assert ("a", 1, "extra") not in items_view
    assert () not in items_view
    
    # Test with PMap containing complex values
    complex_pmap = pmap({"x": [1, 2], "y": {"nested": "value"}})
    complex_items = PMapItems(complex_pmap)
    assert ("x", [1, 2]) in complex_items
    assert ("y", {"nested": "value"}) in complex_items
    assert ("x", [1]) not in complex_items  # Different list
    assert ("y", {"nested": "other"}) not in complex_items
    
    # Test that it works with converted mapping
    regular_dict = {"key1": "val1", "key2": "val2"}
    converted_items = PMapItems(regular_dict)
    assert ("key1", "val1") in converted_items
    assert ("key2", "val2") in converted_items
    assert ("key3", "val3") not in converted_items


# LLM-generated content at query #7
#--------------------------

```python
def test_PMap_update_with():
    from operator import add, sub, mul
    from pyrsistent import m

    # Test basic update_with functionality
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}
    assert isinstance(result, type(m1))

    # Test with multiple maps
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2), {'a': 3, 'c': 5})
    assert result == {'a': 6, 'b': 2, 'c': 5}

    # Test with custom merge function (keep leftmost)
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1, 'b': 2}

    # Test with custom merge function (keep rightmost - same as regular update)
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert result == {'a': 3, 'b': 2}

    # Test with empty maps
    m1 = m(a=1, b=2)
    result = m1.update_with(add, {}, m())
    assert result == {'a': 1, 'b': 2}
    assert result is m1  # Should return same instance when no changes

    # Test with new keys
    m1 = m(a=1)
    result = m1.update_with(add, {'b': 2, 'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}

    # Test with complex merge function
    m1 = m(a=10, b=20)
    result = m1.update_with(lambda x, y: x * y, {'a': 2, 'b': 3})
    assert result == {'a': 20, 'b': 60}

    # Test that original map is unchanged
    m1 = m(a=1, b=2)
    original_hash = hash(m1)
    result = m1.update_with(add, m(a=2))
    assert m1 == {'a': 1, 'b': 2}
    assert hash(m1) == original_hash

    # Test with no maps provided
    m1 = m(a=1, b=2)
    result = m1.update_with(add)
    assert result == {'a': 1, 'b': 2}
    assert result is m1

    # Test with different value types
    m1 = m(a='hello', b='world')
    result = m1.update_with(lambda x, y: x + ' ' + y, {'a': 'there'})
    assert result == {'a': 'hello there', 'b': 'world'}

    # Test with list concatenation
    m1 = m(a=[1, 2], b=[3, 4])
    result = m1.update_with(lambda x, y: x + y, {'a': [3, 4]})
    assert result == {'a': [1, 2, 3, 4], 'b': [3, 4]}

    # Test that it works with any Mapping type, not just PMap
    from collections import OrderedDict
    m1 = m(a=1, b=2)
    od = OrderedDict([('a', 3), ('c', 4)])
    result = m1.update_with(add, od)
    assert result == {'a': 4, 'b': 2, 'c': 4}


# LLM-generated content at query #8
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert ("key", "value") not in empty_items
    
    # Test with non-empty pmap
    test_pmap = pmap({"a": 1, "b": 2, "c": 3})
    items = PMapItems(test_pmap)
    
    # Test existing items
    assert ("a", 1) in items
    assert ("b", 2) in items
    assert ("c", 3) in items
    
    # Test non-existing keys
    assert ("d", 4) not in items
    assert ("a", 2) not in items  # Wrong value for existing key
    assert ("x", 1) not in items  # Wrong key for existing value
    
    # Test with wrong value type
    assert ("a", "1") not in items
    
    # Test with non-tuple argument (should return False)
    assert "a" not in items
    assert 1 not in items
    assert ["a", 1] not in items
    assert {"a": 1} not in items
    
    # Test with tuple of wrong length
    assert (1, 2, 3) not in items
    assert (1,) not in items
    
    # Test with nested pmap
    nested_pmap = pmap({"x": pmap({"y": 5})})
    nested_items = PMapItems(nested_pmap)
    assert ("x", pmap({"y": 5})) in nested_items
    
    # Test equality comparison with same pmap
    same_pmap = pmap({"a": 1, "b": 2})
    items1 = PMapItems(same_pmap)
    items2 = PMapItems(same_pmap)
    assert ("a", 1) in items1
    assert ("a", 1) in items2
    
    # Test that it works with any mapping, not just pmap
    class TestMapping(Mapping):
        def __init__(self, data):
            self._data = dict(data)
        
        def __getitem__(self, key):
            return self._data[key]
        
        def __iter__(self):
            return iter(self._data)
        
        def __len__(self):
            return len(self._data)
    
    test_mapping = TestMapping({"k1": "v1", "k2": "v2"})
    mapping_items = PMapItems(test_mapping)
    assert ("k1", "v1") in mapping_items
    assert ("k2", "v2") in mapping_items
    assert ("k3", "v3") not in mapping_items


# LLM-generated content at query #9
#--------------------------

```python
def test_PMap___eq__():
    from pyrsistent import pmap, m
    
    # Test equality with self
    pm1 = pmap({'a': 1, 'b': 2})
    assert pm1 == pm1
    assert not (pm1 != pm1)
    
    # Test equality with identical PMap
    pm2 = pmap({'a': 1, 'b': 2})
    assert pm1 == pm2
    assert not (pm1 != pm2)
    
    # Test equality with different PMap (different values)
    pm3 = pmap({'a': 1, 'b': 3})
    assert not (pm1 == pm3)
    assert pm1 != pm3
    
    # Test equality with different PMap (different keys)
    pm4 = pmap({'a': 1, 'c': 2})
    assert not (pm1 == pm4)
    assert pm1 != pm4
    
    # Test equality with regular dict
    assert pm1 == {'a': 1, 'b': 2}
    assert not (pm1 == {'a': 1, 'b': 3})
    assert pm1 != {'a': 1, 'b': 3}
    
    # Test equality with different sized mapping
    pm5 = pmap({'a': 1, 'b': 2, 'c': 3})
    assert not (pm1 == pm5)
    assert pm1 != pm5
    
    # Test equality with non-mapping type
    assert not (pm1 == [('a', 1), ('b', 2)])
    assert pm1 != [('a', 1), ('b', 2)]
    
    # Test equality with PMap that has same items but different bucket structure
    # Create two PMaps that should have same content but potentially different internal structure
    pm6 = pmap({}).set('x', 10).set('y', 20)
    pm7 = pmap({'x': 10, 'y': 20})
    assert pm6 == pm7
    assert not (pm6 != pm7)
    
    # Test equality with empty PMaps
    pm_empty1 = pmap({})
    pm_empty2 = pmap({})
    assert pm_empty1 == pm_empty2
    assert pm_empty1 == {}
    
    # Test equality with subclass of Mapping
    from collections.abc import Mapping
    class TestMapping(Mapping):
        def __init__(self, data):
            self._data = data
        def __getitem__(self, key):
            return self._data[key]
        def __iter__(self):
            return iter(self._data)
        def __len__(self):
            return len(self._data)
    
    test_map = TestMapping({'a': 1, 'b': 2})
    assert pm1 == test_map
    assert not (pm1 != test_map)
    
    # Test cached hash affects equality check
    pm8 = pmap({'x': 1, 'y': 2})
    pm9 = pmap({'x': 1, 'y': 2})
    hash(pm8)  # Cache hash
    hash(pm9)  # Cache hash
    assert pm8 == pm9
    
    # Test with different cached hash values
    pm10 = pmap({'x': 1, 'y': 2})
    pm11 = pmap({'x': 1, 'y': 3})
    hash(pm10)
    hash(pm11)
    assert not (pm10 == pm11)
    
    # Test PMap equality with dict_items-like object
    assert not (pm1 == pm1.items())
    assert pm1 != pm1.items()


# LLM-generated content at query #10
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    m = pmap({})
    items_view = PMapItems(m)
    assert (1, 2) not in items_view
    assert () not in items_view
    assert ("key", "value") not in items_view

    # Test with non-empty PMap
    m = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(m)
    
    # Items that are present
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Items with wrong value
    assert ("a", 2) not in items_view
    assert ("b", 1) not in items_view
    assert ("c", 4) not in items_view
    
    # Keys not in map
    assert ("d", 1) not in items_view
    assert ("x", 100) not in items_view
    
    # Test with wrong argument types (should return False)
    assert 42 not in items_view
    assert "string" not in items_view
    assert ["a", 1] not in items_view
    assert ("a", 1, "extra") not in items_view
    assert ("a",) not in items_view
    
    # Test with PMap containing different value types
    m = pmap({1: "one", 2.5: [1, 2, 3], (1, 2): {"nested": "dict"}})
    items_view = PMapItems(m)
    
    assert (1, "one") in items_view
    assert (2.5, [1, 2, 3]) in items_view
    assert ((1, 2), {"nested": "dict"}) in items_view
    
    # Wrong values for existing keys
    assert (1, "two") not in items_view
    assert (2.5, [1, 2]) not in items_view
    assert ((1, 2), {"nested": "dictionary"}) not in items_view
    
    # Test that it works with PMapView base class constraints
    m = pmap({"x": 10, "y": 20})
    items_view = PMapItems(m)
    
    # Verify it handles the actual PMapItems iteration correctly
    for item in items_view:
        assert item in items_view
    
    # Test with non-tuple argument that can't be unpacked
    class BadContainer:
        def __iter__(self):
            raise ValueError("Cannot iterate")
    
    bad = BadContainer()
    assert bad not in items_view


# LLM-generated content at query #11
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    m = pmap({})
    items_view = PMapItems(m)
    assert (1, 2) not in items_view
    assert () not in items_view
    assert ("key", "value") not in items_view

    # Test with single item PMap
    m = pmap({"a": 1})
    items_view = PMapItems(m)
    assert ("a", 1) in items_view
    assert ("a", 2) not in items_view
    assert ("b", 1) not in items_view
    assert ("a",) not in items_view
    assert "a" not in items_view
    assert 1 not in items_view

    # Test with multiple items PMap
    m = pmap({"x": 10, "y": 20, "z": 30})
    items_view = PMapItems(m)
    assert ("x", 10) in items_view
    assert ("y", 20) in items_view
    assert ("z", 30) in items_view
    assert ("x", 20) not in items_view
    assert ("w", 10) not in items_view
    assert (10, "x") not in items_view

    # Test with non-tuple argument
    m = pmap({"key": "value"})
    items_view = PMapItems(m)
    assert "key" not in items_view
    assert "value" not in items_view
    assert ["key", "value"] not in items_view
    assert {"key": "value"} not in items_view

    # Test with tuple of wrong length
    m = pmap({"a": 1, "b": 2})
    items_view = PMapItems(m)
    assert () not in items_view
    assert (1, 2, 3) not in items_view
    assert ("a",) not in items_view

    # Test with correct value but wrong type comparison
    m = pmap({"num": 5})
    items_view = PMapItems(m)
    assert ("num", 5.0) not in items_view
    assert ("num", "5") not in items_view

    # Test that __contains__ works after PMap modification
    m1 = pmap({"old": "data"})
    items_view1 = PMapItems(m1)
    m2 = m1.set("new", "item")
    items_view2 = PMapItems(m2)
    assert ("old", "data") in items_view1
    assert ("old", "data") in items_view2
    assert ("new", "item") not in items_view1
    assert ("new", "item") in items_view2


# LLM-generated content at query #12
#--------------------------

```python
def test_PMap_update_with():
    from operator import add, sub
    
    # Test basic update_with functionality
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}))
    assert result == {'a': 3, 'b': 2}
    
    # Test with multiple maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}), {'b': 3, 'c': 1})
    assert result == {'a': 3, 'b': 5, 'c': 1}
    
    # Test with lambda that keeps leftmost value
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(lambda l, r: l, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 1, 'b': 2}
    
    # Test with lambda that keeps rightmost value (default update behavior)
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(lambda l, r: r, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 3, 'b': 2}
    
    # Test with subtraction operator
    m1 = pmap({'a': 10, 'b': 5})
    result = m1.update_with(sub, pmap({'a': 3, 'b': 2}))
    assert result == {'a': 7, 'b': 3}
    
    # Test with new keys
    m1 = pmap({'a': 1})
    result = m1.update_with(add, pmap({'b': 2, 'c': 3}))
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with empty maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({}), {})
    assert result == {'a': 1, 'b': 2}
    assert result is m1  # Should return same instance when no changes
    
    # Test with complex merge function
    m1 = pmap({'a': [1], 'b': [2]})
    result = m1.update_with(lambda l, r: l + r, pmap({'a': [3], 'b': [4]}))
    assert result == {'a': [1, 3], 'b': [2, 4]}
    
    # Test that original map is unchanged
    m1 = pmap({'a': 1, 'b': 2})
    original_hash = hash(m1)
    m2 = m1.update_with(add, pmap({'a': 2}))
    assert m1 == {'a': 1, 'b': 2}
    assert hash(m1) == original_hash
    assert m2 == {'a': 3, 'b': 2}
    
    # Test with non-PMap mappings
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, {'a': 2}, dict(c=3))
    assert result == {'a': 3, 'b': 2, 'c': 3}
    
    # Test with no additional maps (should return self)
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add)
    assert result is m1
    
    # Test with update_fn that returns None
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(lambda l, r: None, pmap({'a': 2}))
    assert result == {'a': None, 'b': 2}


# LLM-generated content at query #13
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with functionality
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}))
    assert result == {'a': 3, 'b': 2}
    
    # Test update_with with multiple maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}), pmap({'a': 3, 'c': 1}))
    assert result == {'a': 6, 'b': 2, 'c': 1}
    
    # Test update_with with leftmost preference
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: l, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 1}
    
    # Test update_with with rightmost preference (like regular update)
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: r, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 3}
    
    # Test update_with with new keys
    m1 = pmap({'a': 1})
    result = m1.update_with(add, pmap({'b': 2}), {'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test update_with with empty map
    m1 = pmap()
    result = m1.update_with(add, {'a': 1, 'b': 2})
    assert result == {'a': 1, 'b': 2}
    
    # Test update_with with no additional maps returns same instance
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add)
    assert result is m1
    
    # Test update_with with complex merge function
    m1 = pmap({'a': [1], 'b': [2]})
    result = m1.update_with(lambda l, r: l + r, {'a': [3], 'c': [4]})
    assert result == {'a': [1, 3], 'b': [2], 'c': [4]}
    
    # Test update_with preserves immutability of original
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 3, 'c': 4})
    result = m1.update_with(add, m2)
    assert m1 == {'a': 1, 'b': 2}
    assert m2 == {'a': 3, 'c': 4}
    assert result == {'a': 4, 'b': 2, 'c': 4}
    
    # Test update_with with dict and PMap mixed
    m1 = pmap({'a': 1})
    result = m1.update_with(add, {'a': 2}, pmap({'a': 3}))
    assert result == {'a': 6}


# LLM-generated content at query #14
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with empty pmap
    m = pmap({})
    items_view = PMapItems(m)
    assert (1, 2) not in items_view
    assert () not in items_view
    assert (1,) not in items_view
    
    # Test with single item pmap
    m = pmap({1: 'a'})
    items_view = PMapItems(m)
    assert (1, 'a') in items_view
    assert (1, 'b') not in items_view
    assert (2, 'a') not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with multiple items
    m = pmap({1: 'a', 2: 'b', 3: 'c'})
    items_view = PMapItems(m)
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view
    assert (1, 'b') not in items_view
    assert (4, 'd') not in items_view
    
    # Test with non-tuple argument (should return False)
    assert 1 not in items_view
    assert 'a' not in items_view
    assert [1, 'a'] not in items_view
    assert {'key': 1, 'value': 'a'} not in items_view
    
    # Test with tuple of wrong length
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test with correct key but wrong value
    assert (1, 'wrong') not in items_view
    assert (2, 'not_b') not in items_view
    
    # Test with nested structures
    m = pmap({1: [1, 2], 2: {'a': 1}})
    items_view = PMapItems(m)
    assert (1, [1, 2]) in items_view
    assert (2, {'a': 1}) in items_view
    assert (1, [1, 2, 3]) not in items_view
    assert (2, {'a': 2}) not in items_view
    
    # Test that it handles exceptions gracefully
    class BadEq:
        def __eq__(self, other):
            raise ValueError("Comparison failed")
    
    m = pmap({1: BadEq()})
    items_view = PMapItems(m)
    # This should return False, not raise an exception
    assert (1, BadEq()) not in items_view
    assert (1, "something") not in items_view


# LLM-generated content at query #15
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert ("key", "value") not in empty_items
    
    # Test with non-empty pmap
    test_pmap = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(test_pmap)
    
    # Test existing items
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Test non-existing keys
    assert ("d", 4) not in items_view
    assert ("a", 2) not in items_view  # Wrong value for existing key
    assert ("b", 1) not in items_view  # Wrong value for existing key
    
    # Test with wrong value type
    assert ("a", "1") not in items_view
    
    # Test with non-tuple argument (should return False)
    assert "a" not in items_view
    assert 1 not in items_view
    assert ["a", 1] not in items_view
    assert {"a": 1} not in items_view
    
    # Test with malformed tuple (wrong length)
    assert (1, 2, 3) not in items_view
    assert (1,) not in items_view
    
    # Test with nested structures
    complex_pmap = pmap({"x": [1, 2], "y": {"z": 3}})
    complex_items = PMapItems(complex_pmap)
    assert ("x", [1, 2]) in complex_items
    assert ("y", {"z": 3}) in complex_items
    assert ("x", [1]) not in complex_items  # Different list
    
    # Test that it works with PMapItems from different pmaps
    other_pmap = pmap({"a": 1, "b": 2})
    other_items = PMapItems(other_pmap)
    assert ("a", 1) in other_items
    assert ("c", 3) not in other_items
    
    # Test that view reflects changes to original pmap
    original = pmap({"initial": "value"})
    view = PMapItems(original)
    assert ("initial", "value") in view
    
    # Test with None values
    none_pmap = pmap({"key": None})
    none_items = PMapItems(none_pmap)
    assert ("key", None) in none_items
    assert ("key", 0) not in none_items


# LLM-generated content at query #16
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with functionality
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}
    
    # Test with multiple maps
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2), {'a': 3, 'c': 1})
    assert result == {'a': 6, 'b': 2, 'c': 1}
    
    # Test with custom update function that keeps leftmost value
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1, 'b': 2}
    
    # Test with custom update function that keeps rightmost value
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert result == {'a': 3, 'b': 2}
    
    # Test with empty maps
    m1 = m(a=1, b=2)
    result = m1.update_with(add, {}, {})
    assert result == {'a': 1, 'b': 2}
    assert result is m1
    
    # Test with new keys
    m1 = m(a=1)
    result = m1.update_with(add, {'b': 2}, {'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with complex update function
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda x, y: x * y, {'a': 3}, {'a': 2})
    assert result == {'a': 6, 'b': 2}
    
    # Test that original map is unchanged
    m1 = m(a=1, b=2)
    original_hash = hash(m1)
    m2 = m1.update_with(add, {'a': 2})
    assert m1 == {'a': 1, 'b': 2}
    assert hash(m1) == original_hash
    assert m2 == {'a': 3, 'b': 2}
    
    # Test with no maps (only self)
    m1 = m(a=1, b=2)
    result = m1.update_with(add)
    assert result == {'a': 1, 'b': 2}
    assert result is m1
    
    # Test with update function that concatenates strings
    m1 = m(a='hello', b='world')
    result = m1.update_with(lambda x, y: x + ' ' + y, {'a': 'there'})
    assert result == {'a': 'hello there', 'b': 'world'}
    
    # Test with mixed types in maps
    m1 = m(a=1, b='test')
    result = m1.update_with(lambda l, r: str(l) + str(r), {'a': 2}, {'a': 3})
    assert result == {'a': '123', 'b': 'test'}


# LLM-generated content at query #17
#--------------------------

```python
def test_PMap_update_with():
    from pyrsistent import m
    from operator import add
    
    # Test basic update_with functionality
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}
    
    # Test with multiple maps
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2), {'a': 3, 'c': 1})
    assert result == {'a': 6, 'b': 2, 'c': 1}
    
    # Test with custom merge function (keep leftmost)
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1, 'b': 2}
    
    # Test with custom merge function (keep rightmost - same as regular update)
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert result == {'a': 3, 'b': 2}
    
    # Test with empty maps
    m1 = m(a=1, b=2)
    result = m1.update_with(add, {}, {})
    assert result == m1
    assert result is m1  # Should return same instance when no changes
    
    # Test adding new keys
    m1 = m(a=1)
    result = m1.update_with(add, {'b': 2}, {'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with complex merge function
    m1 = m(a=1, b=2)
    def multiply_then_add(l, r):
        return l * 2 + r
    result = m1.update_with(multiply_then_add, {'a': 3, 'b': 4})
    assert result == {'a': 5, 'b': 10}  # a: 1*2+3=5, b: 2*2+4=10
    
    # Test that original map is unchanged
    m1 = m(a=1, b=2)
    original_hash = hash(m1)
    result = m1.update_with(add, {'a': 2})
    assert m1 == {'a': 1, 'b': 2}
    assert hash(m1) == original_hash
    
    # Test with no maps provided
    m1 = m(a=1, b=2)
    result = m1.update_with(add)
    assert result == m1
    assert result is m1
    
    # Test with different map types
    m1 = m(a=1)
    result = m1.update_with(add, dict(b=2), m(c=3))
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test merge function that returns None
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: None, {'a': 2})
    assert result == {'a': None, 'b': 2}


# LLM-generated content at query #18
#--------------------------

```python
def test_PMap_update_with():
    from pyrsistent import m
    from operator import add, mul

    # Test basic update_with with single map
    m1 = m(a=1, b=2)
    m2 = m1.update_with(add, m(a=3))
    assert m2 == {'a': 4, 'b': 2}
    assert m1 == {'a': 1, 'b': 2}  # Original unchanged

    # Test update_with with multiple maps
    m1 = m(a=1, b=2)
    m2 = m1.update_with(add, m(a=3), {'a': 5, 'c': 4})
    assert m2 == {'a': 9, 'b': 2, 'c': 4}

    # Test update_with with lambda that keeps leftmost value
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m(a=3), {'a': 5})
    assert m2 == {'a': 1, 'b': 2}

    # Test update_with with lambda that keeps rightmost value (default behavior)
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r, m(a=3), {'a': 5})
    assert m2 == {'a': 5, 'b': 2}

    # Test update_with with multiplication
    m1 = m(a=2, b=3)
    m2 = m1.update_with(mul, m(a=4, b=5))
    assert m2 == {'a': 8, 'b': 15}

    # Test update_with adding new keys
    m1 = m(a=1)
    m2 = m1.update_with(add, m(b=2, c=3))
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

    # Test update_with on empty map
    m1 = m()
    m2 = m1.update_with(add, m(a=1, b=2))
    assert m2 == {'a': 1, 'b': 2}

    # Test update_with with empty update maps
    m1 = m(a=1, b=2)
    m2 = m1.update_with(add)
    assert m2 == m1
    assert m2 is m1  # Should return same instance when no updates

    # Test update_with preserves immutability
    m1 = m(a=1)
    m2 = m1.update_with(add, m(a=2))
    m3 = m2.update_with(add, m(a=3))
    assert m1 == {'a': 1}
    assert m2 == {'a': 3}
    assert m3 == {'a': 6}

    # Test update_with with complex merge function
    def merge_dicts(left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            return {**left, **right}
        return right

    m1 = m(a={'x': 1}, b=2)
    m2 = m1.update_with(merge_dicts, m(a={'y': 2}))
    assert m2 == {'a': {'x': 1, 'y': 2}, 'b': 2}

    # Test update_with with None values
    m1 = m(a=None, b=2)
    m2 = m1.update_with(lambda l, r: r if l is None else l, m(a=1, b=None))
    assert m2 == {'a': 1, 'b': 2}


# LLM-generated content at query #19
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert ("key", "value") not in empty_items
    
    # Test with non-empty pmap
    test_pmap = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(test_pmap)
    
    # Test existing items
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Test non-existing keys
    assert ("d", 4) not in items_view
    assert ("a", 2) not in items_view  # Wrong value for existing key
    assert ("b", 1) not in items_view  # Wrong value for existing key
    
    # Test with wrong value type
    assert ("a", "1") not in items_view
    
    # Test with non-tuple argument (should return False)
    assert "a" not in items_view
    assert 1 not in items_view
    assert ["a", 1] not in items_view
    assert {"a": 1} not in items_view
    
    # Test with malformed tuple (wrong length)
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 2, 3) not in items_view
    
    # Test with tuple containing wrong types
    assert (None, None) not in items_view
    assert ([], {}) not in items_view
    
    # Test that it works with different value types
    complex_pmap = pmap({
        "int": 42,
        "str": "hello",
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "none": None
    })
    complex_items = PMapItems(complex_pmap)
    
    assert ("int", 42) in complex_items
    assert ("str", "hello") in complex_items
    assert ("list", [1, 2, 3]) in complex_items
    assert ("dict", {"nested": "value"}) in complex_items
    assert ("none", None) in complex_items
    
    # Test with modified value (should not be found)
    assert ("list", [1, 2]) not in complex_items
    assert ("dict", {"nested": "wrong"}) not in complex_items


# LLM-generated content at query #20
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert () not in empty_items
    assert (1,) not in empty_items
    
    # Test with non-empty PMap
    test_pmap = pmap({1: 'a', 2: 'b', 3: 'c'})
    items_view = PMapItems(test_pmap)
    
    # Test existing items
    assert (1, 'a') in items_view
    assert (2, 'b') in items_view
    assert (3, 'c') in items_view
    
    # Test non-existing keys
    assert (4, 'd') not in items_view
    assert (1, 'b') not in items_view  # Wrong value for existing key
    assert (2, 'a') not in items_view  # Wrong value for existing key
    
    # Test wrong value types
    assert (1, 100) not in items_view
    assert ('a', 1) not in items_view
    
    # Test with non-tuple argument
    assert 1 not in items_view
    assert 'a' not in items_view
    assert [1, 'a'] not in items_view
    
    # Test with tuple of wrong length
    assert (1,) not in items_view
    assert (1, 'a', 'extra') not in items_view
    
    # Test that it works with PMap containing different types
    complex_pmap = pmap({'key': [1, 2, 3], 42: {'nested': 'value'}})
    complex_items = PMapItems(complex_pmap)
    assert ('key', [1, 2, 3]) in complex_items
    assert (42, {'nested': 'value'}) in complex_items
    assert ('key', [1, 2]) not in complex_items  # Different list
    assert (42, 'wrong') not in complex_items
    
    # Test that __contains__ handles exceptions gracefully
    class BadComparable:
        def __eq__(self, other):
            raise ValueError("Cannot compare")
    
    bad_pmap = pmap({1: BadComparable()})
    bad_items = PMapItems(bad_pmap)
    # Should return False without raising when tuple unpacking fails
    assert BadComparable() not in bad_items
    assert (1, BadComparable()) not in bad_items  # This might raise during comparison
    
    # Test with None values
    none_pmap = pmap({1: None, 2: None})
    none_items = PMapItems(none_pmap)
    assert (1, None) in none_items
    assert (2, None) in none_items
    assert (3, None) not in none_items


# LLM-generated content at query #21
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid PMap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test that existing items are found
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test that non-existent items are not found
    assert ('a', 2) not in items_view  # Wrong value
    assert ('d', 1) not in items_view  # Wrong key
    assert ('d', 4) not in items_view  # Both wrong
    
    # Test with wrong type (should return False, not raise)
    assert 42 not in items_view
    assert ('a',) not in items_view  # Single element tuple
    assert ('a', 1, 'extra') not in items_view  # Too many elements
    assert [1, 2] not in items_view  # List instead of tuple
    assert None not in items_view
    
    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('anything', 1) not in empty_items
    
    # Test that it works with different value types
    complex_pmap = pmap({'x': [1, 2], 'y': {'nested': 'dict'}, 'z': None})
    complex_items = PMapItems(complex_pmap)
    assert ('x', [1, 2]) in complex_items
    assert ('y', {'nested': 'dict'}) in complex_items
    assert ('z', None) in complex_items
    
    # Test that key must be in map AND value must match
    m2 = pmap({'key': 'value'})
    items2 = PMapItems(m2)
    assert ('key', 'value') in items2
    assert ('key', 'wrong_value') not in items2
    assert ('wrong_key', 'value') not in items2
    
    # Test with PMap containing many items
    large_pmap = pmap({i: i*2 for i in range(100)})
    large_items = PMapItems(large_pmap)
    assert (50, 100) in large_items
    assert (99, 198) in large_items
    assert (100, 200) not in large_items  # Key doesn't exist


# LLM-generated content at query #22
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid PMap
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test that existing items are found
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test that non-existent items are not found
    assert ('a', 2) not in items_view  # Wrong value
    assert ('d', 1) not in items_view  # Wrong key
    assert ('d', 4) not in items_view  # Both wrong
    
    # Test with wrong type (not a tuple)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert ['a', 1] not in items_view
    assert ('a', 1, 'extra') not in items_view
    
    # Test with tuple that can't unpack
    assert (1,) not in items_view
    assert () not in items_view
    
    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items
    
    # Test that it works with non-PMap Mapping (gets converted)
    regular_dict = {'x': 10, 'y': 20}
    converted_items = PMapItems(regular_dict)
    assert ('x', 10) in converted_items
    assert ('y', 20) in converted_items
    assert ('z', 30) not in converted_items


# LLM-generated content at query #23
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with functionality
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}))
    assert result == {'a': 3, 'b': 2}
    assert isinstance(result, PMap)
    
    # Test with multiple maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, pmap({'a': 2}), pmap({'a': 3, 'c': 1}))
    assert result == {'a': 6, 'b': 2, 'c': 1}
    
    # Test with custom merge function (keep leftmost)
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: l, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 1}
    
    # Test with custom merge function (keep rightmost)
    m1 = pmap({'a': 1})
    result = m1.update_with(lambda l, r: r, pmap({'a': 2}), {'a': 3})
    assert result == {'a': 3}
    
    # Test with empty maps
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add, {}, pmap({}))
    assert result == {'a': 1, 'b': 2}
    assert result is m1
    
    # Test adding new keys
    m1 = pmap({'a': 1})
    result = m1.update_with(add, pmap({'b': 2}), {'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with different map types
    m1 = pmap({'a': 1})
    result = m1.update_with(add, {'a': 2}, pmap({'a': 3}))
    assert result == {'a': 6}
    
    # Test with no maps (should return original)
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.update_with(add)
    assert result == {'a': 1, 'b': 2}
    assert result is m1
    
    # Test merge function that combines values differently
    m1 = pmap({'a': [1], 'b': [2]})
    result = m1.update_with(lambda l, r: l + r, {'a': [3]}, pmap({'a': [4], 'c': [5]}))
    assert result == {'a': [1, 3, 4], 'b': [2], 'c': [5]}
    
    # Test that original is not modified
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 3, 'c': 4})
    result = m1.update_with(add, m2)
    assert m1 == {'a': 1, 'b': 2}
    assert m2 == {'a': 3, 'c': 4}
    assert result == {'a': 4, 'b': 2, 'c': 4}


# LLM-generated content at query #24
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Test with valid PMap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test that existing items are found
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test that non-existent items are not found
    assert ('a', 2) not in items_view  # Wrong value
    assert ('d', 1) not in items_view  # Wrong key
    assert ('d', 4) not in items_view  # Both wrong
    
    # Test with wrong type (not a tuple)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert ['a', 1] not in items_view
    assert ('a', 1, 'extra') not in items_view
    
    # Test with tuple that can't unpack to 2 elements
    assert () not in items_view
    assert ('a',) not in items_view
    
    # Test with empty PMap
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    
    # Test that it works with non-tuple sequences
    assert ['a', 1] not in items_view  # Should return False, not crash
    
    # Test with nested PMap
    nested = pmap({'x': pmap({'y': 5})})
    nested_items = PMapItems(nested)
    assert ('x', pmap({'y': 5})) in nested_items


# LLM-generated content at query #25
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert (1, 2) not in empty_items
    assert ("key", "value") not in empty_items
    assert () not in empty_items
    assert (1,) not in empty_items

    # Test with non-empty PMap
    test_pmap = pmap({"a": 1, "b": 2, "c": 3})
    items_view = PMapItems(test_pmap)
    
    # Test existing items
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view
    assert ("c", 3) in items_view
    
    # Test non-existing keys
    assert ("d", 4) not in items_view
    assert ("a", 2) not in items_view  # Wrong value for existing key
    assert ("b", 1) not in items_view  # Wrong value for existing key
    
    # Test with wrong value type
    assert ("a", "1") not in items_view
    
    # Test with non-tuple argument
    assert "a" not in items_view
    assert 1 not in items_view
    assert ["a", 1] not in items_view
    assert {"a": 1} not in items_view
    
    # Test with tuple of wrong length
    assert () not in items_view
    assert (1,) not in items_view
    assert (1, 2, 3) not in items_view
    
    # Test with tuple that can't unpack
    class Unpackable:
        def __iter__(self):
            raise TypeError("Cannot unpack")
    
    assert Unpackable() not in items_view
    
    # Test equality with same items but different PMap instance
    another_pmap = pmap({"a": 1, "b": 2, "c": 3})
    another_items = PMapItems(another_pmap)
    assert ("a", 1) in another_items
    
    # Test that it works with nested structures
    nested_pmap = pmap({"x": [1, 2], "y": {"z": 3}})
    nested_items = PMapItems(nested_pmap)
    assert ("x", [1, 2]) in nested_items
    assert ("y", {"z": 3}) in nested_items
    assert ("x", [1]) not in nested_items


