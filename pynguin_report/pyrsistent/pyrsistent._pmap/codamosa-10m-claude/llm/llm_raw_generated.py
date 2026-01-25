####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PMapItems___contains__():
    # Create a PMap for testing
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test with valid items that exist in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test with items that don't exist (wrong value for key)
    assert ('a', 2) not in items_view
    assert ('b', 1) not in items_view
    
    # Test with items where key doesn't exist
    assert ('d', 4) not in items_view
    assert ('x', 100) not in items_view
    
    # Test with invalid argument types that can't be unpacked
    assert 'not_a_tuple' not in items_view
    assert 42 not in items_view
    assert None not in items_view
    assert [] not in items_view
    
    # Test with tuples that have wrong length
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items_view = PMapItems(empty_map)
    assert ('a', 1) not in empty_items_view
    assert 'anything' not in empty_items_view


# LLM-generated content at query #2
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with with add operator
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}
    assert m1 == {'a': 1, 'b': 2}  # Original unchanged
    
    # Test with multiple maps
    m2 = m(a=1)
    result = m2.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1}
    assert m2 == {'a': 1}  # Original unchanged
    
    # Test with rightmost value (using lambda r: r)
    m3 = m(x=10, y=20)
    result = m3.update_with(lambda l, r: r, m(x=100, z=30))
    assert result == {'x': 100, 'y': 20, 'z': 30}
    assert m3 == {'x': 10, 'y': 20}  # Original unchanged
    
    # Test with empty map
    m4 = m()
    result = m4.update_with(add, m(a=1, b=2))
    assert result == {'a': 1, 'b': 2}
    assert m4 == {}  # Original unchanged
    
    # Test with no additional maps
    m5 = m(a=1, b=2)
    result = m5.update_with(add)
    assert result == {'a': 1, 'b': 2}
    assert result is m5  # Should return same object when no maps provided
    
    # Test with multiple overlapping keys
    m6 = m(a=1, b=2, c=3)
    result = m6.update_with(lambda l, r: l + r, m(a=10, b=20), {'b': 100, 'd': 4})
    assert result == {'a': 11, 'b': 22, 'c': 3, 'd': 4}
    assert m6 == {'a': 1, 'b': 2, 'c': 3}  # Original unchanged
    
    # Test with custom merge function (max)
    m7 = m(x=5, y=3)
    result = m7.update_with(max, m(x=2, y=10, z=8))
    assert result == {'x': 5, 'y': 10, 'z': 8}
    assert m7 == {'x': 5, 'y': 3}  # Original unchanged
    
    # Test with dict argument
    m8 = m(a=1)
    result = m8.update_with(lambda l, r: r, {'a': 99, 'b': 200})
    assert result == {'a': 99, 'b': 200}
    assert m8 == {'a': 1}  # Original unchanged
    
    # Test that evolver is properly used internally
    m9 = m(p=1, q=2)
    result = m9.update_with(add, m(p=5, q=3, r=7))
    assert result == {'p': 6, 'q': 5, 'r': 7}
    assert isinstance(result, type(m9))


# LLM-generated content at query #3
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with itself
    m1 = m(a=1, b=2)
    assert m1 == m1
    
    # Test equality with another PMap with same content
    m2 = m(a=1, b=2)
    assert m1 == m2
    
    # Test inequality with PMap with different content
    m3 = m(a=1, b=3)
    assert not (m1 == m3)
    
    # Test equality with dict with same content
    assert m1 == {'a': 1, 'b': 2}
    
    # Test inequality with dict with different content
    assert not (m1 == {'a': 1, 'b': 3})
    
    # Test inequality with dict with different keys
    assert not (m1 == {'a': 1, 'c': 2})
    
    # Test inequality with different length dict
    assert not (m1 == {'a': 1})
    
    # Test equality with empty PMap
    m_empty1 = m()
    m_empty2 = m()
    assert m_empty1 == m_empty2
    
    # Test inequality of empty with non-empty
    assert not (m_empty1 == m1)
    
    # Test equality with other Mapping types
    from collections import OrderedDict
    od = OrderedDict([('a', 1), ('b', 2)])
    assert m1 == od
    
    # Test inequality with non-Mapping types
    assert not (m1 == [('a', 1), ('b', 2)])
    assert not (m1 == "not a mapping")
    assert not (m1 == 42)
    
    # Test with cached hash - two PMaps with same content and cached hash
    m4 = m(x=10, y=20)
    m5 = m(x=10, y=20)
    hash(m4)
    hash(m5)
    assert m4 == m5
    
    # Test that different hash values make them unequal
    m6 = m(a=1, b=2)
    m7 = m(a=1, b=2, c=3)
    assert not (m6 == m7)
    
    # Test with single element
    m_single1 = m(key='value')
    m_single2 = m(key='value')
    assert m_single1 == m_single2
    
    # Test NotImplemented is returned for non-Mapping comparison
    result = m1.__eq__([1, 2, 3])
    assert result is NotImplemented


# LLM-generated content at query #4
#--------------------------

```python
def test_PMapValues___eq__():
    # Test that __eq__ returns True when comparing with itself
    m = pmap({'a': 1, 'b': 2})
    values = PMapValues(m)
    assert values == values
    assert (values == values) is True
    
    # Test that __eq__ returns False when comparing with another PMapValues instance
    m2 = pmap({'a': 1, 'b': 2})
    values2 = PMapValues(m2)
    assert not (values == values2)
    assert (values == values2) is False
    
    # Test that __eq__ returns False when comparing with different PMapValues
    m3 = pmap({'x': 10, 'y': 20})
    values3 = PMapValues(m3)
    assert not (values == values3)
    assert (values == values3) is False
    
    # Test that __eq__ returns False when comparing with other types
    assert not (values == [1, 2])
    assert not (values == {1, 2})
    assert not (values == "pmap_values([1, 2])")
    assert not (values == None)
    assert not (values == 42)
    
    # Test that __eq__ only returns True for identity (is self)
    values_same_ref = values
    assert values == values_same_ref
    assert (values == values_same_ref) is True


# LLM-generated content at query #5
#--------------------------

```python
def test_PMapValues___eq__():
    # Test that PMapValues.__eq__ returns True when comparing with itself
    m = pmap({'a': 1, 'b': 2})
    values = PMapValues(m)
    assert values == values
    assert (values == values) is True
    
    # Test that PMapValues.__eq__ returns False when comparing with a different instance
    m2 = pmap({'a': 1, 'b': 2})
    values2 = PMapValues(m2)
    assert not (values == values2)
    assert (values == values2) is False
    
    # Test that PMapValues.__eq__ returns False when comparing with other types
    assert not (values == [1, 2])
    assert (values == [1, 2]) is False
    
    assert not (values == {'a': 1, 'b': 2})
    assert (values == {'a': 1, 'b': 2}) is False
    
    assert not (values == None)
    assert (values == None) is False
    
    # Test that PMapValues.__eq__ returns False when comparing with dict_values
    dict_values = {'a': 1, 'b': 2}.values()
    assert not (values == dict_values)
    assert (values == dict_values) is False
    
    # Test with empty PMapValues
    empty_m = pmap({})
    empty_values = PMapValues(empty_m)
    assert empty_values == empty_values
    assert (empty_values == empty_values) is True
    
    # Test that different empty PMapValues instances are not equal
    empty_values2 = PMapValues(pmap({}))
    assert not (empty_values == empty_values2)
    assert (empty_values == empty_values2) is False


# LLM-generated content at query #6
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item tuple that exists in the map
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test with valid item tuple that does not exist in the map
    assert ('a', 2) not in items_view
    assert ('d', 4) not in items_view
    assert ('b', 1) not in items_view
    
    # Test with key that exists but value is different
    assert ('a', 999) not in items_view
    
    # Test with non-tuple argument
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    
    # Test with invalid tuple structure (too many elements)
    assert ('a', 1, 'extra') not in items_view
    
    # Test with empty tuple
    assert () not in items_view
    
    # Test with list instead of tuple (unpacking should still work)
    assert ['a', 1] in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items_view = PMapItems(empty_map)
    assert ('a', 1) not in empty_items_view
    assert () not in empty_items_view
    
    # Test with various value types
    m2 = pmap({'key': None, 'list': [1, 2], 'dict': {'nested': 'value'}})
    items_view2 = PMapItems(m2)
    assert ('key', None) in items_view2
    assert ('list', [1, 2]) in items_view2
    assert ('dict', {'nested': 'value'}) in items_view2


# LLM-generated content at query #7
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with with add function
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}
    assert m1 == {'a': 1, 'b': 2}  # Original unchanged
    
    # Test update_with keeping leftmost element
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1}
    
    # Test update_with with multiple maps
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(a=10, c=3), {'c': 30, 'd': 4})
    assert result == {'a': 10, 'b': 2, 'c': 30, 'd': 4}
    
    # Test update_with with no conflicting keys
    m1 = m(a=1)
    result = m1.update_with(add, m(b=2), {'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test update_with with empty map
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m())
    assert result == {'a': 1, 'b': 2}
    
    # Test update_with on empty map
    m1 = m()
    result = m1.update_with(add, m(a=1, b=2))
    assert result == {'a': 1, 'b': 2}
    
    # Test update_with with custom merge function (concatenation)
    m1 = m(a='hello')
    result = m1.update_with(lambda l, r: l + ' ' + r, m(a='world'))
    assert result == {'a': 'hello world'}
    
    # Test update_with with multiple overlapping keys
    m1 = m(x=1, y=2)
    result = m1.update_with(add, m(x=10, y=20, z=30), m(x=100, y=200))
    assert result == {'x': 111, 'y': 222, 'z': 30}
    
    # Test that update_with doesn't modify original
    m1 = m(a=1, b=2)
    m2 = m(a=5)
    result = m1.update_with(lambda l, r: l + r, m2)
    assert m1 == {'a': 1, 'b': 2}
    assert m2 == {'a': 5}
    assert result == {'a': 6, 'b': 2}
    
    # Test update_with with dict argument
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: r, {'a': 10, 'b': 20})
    assert result == {'a': 10, 'b': 20}


# LLM-generated content at query #8
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key, value) tuple that exists
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test with valid item (key, value) tuple that doesn't exist
    assert ('a', 999) not in items_view
    assert ('d', 1) not in items_view
    assert ('x', 'y') not in items_view
    
    # Test with non-tuple argument
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    
    # Test with tuple of wrong length
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view
    assert () not in items_view
    
    # Test with non-iterable argument
    assert 42 not in items_view
    assert 3.14 not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items_view = PMapItems(empty_map)
    assert ('a', 1) not in empty_items_view
    assert () not in empty_items_view
    
    # Test with tuple containing None values
    m_with_none = pmap({'key': None})
    items_view_none = PMapItems(m_with_none)
    assert ('key', None) in items_view_none
    assert ('key', 'value') not in items_view_none
    
    # Test with various types of keys and values
    m_mixed = pmap({1: 'one', 'two': 2, (3, 4): [5, 6]})
    items_view_mixed = PMapItems(m_mixed)
    assert (1, 'one') in items_view_mixed
    assert ('two', 2) in items_view_mixed
    assert ((3, 4), [5, 6]) in items_view_mixed
    assert (1, 'wrong') not in items_view_mixed


# LLM-generated content at query #9
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Create a test pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test valid items that exist in the map
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test valid items that don't exist in the map
    assert ('a', 2) not in items_view
    assert ('d', 4) not in items_view
    assert ('b', 1) not in items_view
    
    # Test invalid argument formats that should return False
    assert 'not_a_tuple' not in items_view
    assert 42 not in items_view
    assert None not in items_view
    assert [1, 2, 3] not in items_view
    
    # Test tuples with wrong length
    assert (1,) not in items_view
    assert (1, 2, 3) not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items_view = PMapItems(empty_map)
    assert ('a', 1) not in empty_items_view
    
    # Test with various value types
    m2 = pmap({'key1': None, 'key2': [1, 2], 'key3': {'nested': 'dict'}})
    items_view2 = PMapItems(m2)
    assert ('key1', None) in items_view2
    assert ('key2', [1, 2]) in items_view2
    assert ('key3', {'nested': 'dict'}) in items_view2


# LLM-generated content at query #10
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with with add function
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}
    
    # Test that original map is unchanged
    assert m1 == {'a': 1, 'b': 2}
    
    # Test update_with with lambda keeping leftmost element
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1}
    
    # Test update_with with multiple maps
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(a=10, c=3), m(a=20, d=4))
    assert result == {'a': 20, 'b': 2, 'c': 3, 'd': 4}
    
    # Test update_with with empty map
    m1 = m(a=1, b=2)
    result = m1.update_with(add, {})
    assert result == {'a': 1, 'b': 2}
    
    # Test update_with with new keys only
    m1 = m(a=1)
    result = m1.update_with(add, m(b=2, c=3))
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test update_with with custom merge function
    m1 = m(a=5, b=2)
    result = m1.update_with(lambda l, r: l + r, m(a=3, c=1))
    assert result == {'a': 8, 'b': 2, 'c': 1}
    
    # Test update_with with multiple overlapping keys
    m1 = m(x=1, y=2)
    result = m1.update_with(lambda l, r: l * r, m(x=3, z=4), {'x': 2})
    assert result == {'x': 6, 'y': 2, 'z': 4}
    
    # Test that result is a new PMap instance
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: r, m(b=2))
    assert result is not m1
    assert isinstance(result, type(m1))
    
    # Test update_with with dict argument
    m1 = m(a=1, b=2)
    result = m1.update_with(add, {'a': 5})
    assert result == {'a': 6, 'b': 2}


# LLM-generated content at query #11
#--------------------------

```python
def test_PMapItems___contains__():
    # Create a simple PMap for testing
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test valid items are found
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test invalid items are not found
    assert ('a', 2) not in items_view
    assert ('d', 4) not in items_view
    assert ('b', 1) not in items_view
    
    # Test non-tuple arguments return False
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [] not in items_view
    
    # Test single-element tuple returns False
    assert ('a',) not in items_view
    
    # Test tuple with more than 2 elements returns False
    assert ('a', 1, 'extra') not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    
    # Test with various value types
    m2 = pmap({'x': None, 'y': [], 'z': {'nested': 'dict'}})
    items_view2 = PMapItems(m2)
    assert ('x', None) in items_view2
    assert ('y', []) in items_view2
    assert ('z', {'nested': 'dict'}) in items_view2
    assert ('x', []) not in items_view2


# LLM-generated content at query #12
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with itself
    m1 = m(a=1, b=2, c=3)
    assert m1 == m1
    
    # Test equality with another PMap with same content
    m2 = m(a=1, b=2, c=3)
    assert m1 == m2
    
    # Test inequality with PMap with different content
    m3 = m(a=1, b=2, c=4)
    assert not (m1 == m3)
    
    # Test inequality with PMap with different keys
    m4 = m(a=1, b=2, d=3)
    assert not (m1 == m4)
    
    # Test inequality with PMap with different size
    m5 = m(a=1, b=2)
    assert not (m1 == m5)
    
    # Test equality with dict with same content
    d1 = {'a': 1, 'b': 2, 'c': 3}
    assert m1 == d1
    
    # Test inequality with dict with different content
    d2 = {'a': 1, 'b': 2, 'c': 4}
    assert not (m1 == d2)
    
    # Test inequality with dict with different keys
    d3 = {'a': 1, 'b': 2, 'd': 3}
    assert not (m1 == d3)
    
    # Test inequality with dict with different size
    d4 = {'a': 1, 'b': 2}
    assert not (m1 == d4)
    
    # Test inequality with non-Mapping type
    assert (m1 == "not a mapping") == NotImplemented
    assert (m1 == [1, 2, 3]) == NotImplemented
    assert (m1 == 42) == NotImplemented
    
    # Test equality with other Mapping types
    from collections import OrderedDict
    od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    assert m1 == od
    
    # Test empty pmaps
    empty1 = m()
    empty2 = m()
    assert empty1 == empty2
    assert empty1 == {}
    
    # Test with cached hash - two pmaps with same content but different cached hash
    m6 = m(x=1, y=2)
    m7 = m(x=1, y=2)
    hash(m6)  # Compute and cache hash for m6
    hash(m7)  # Compute and cache hash for m7
    assert m6 == m7
    
    # Test equality with complex values
    m8 = m(a=m(nested=1), b=[1, 2, 3])
    m9 = m(a=m(nested=1), b=[1, 2, 3])
    assert m8 == m9


# LLM-generated content at query #13
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key, value) tuple that exists
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items = PMapItems(m)
    assert ('a', 1) in items
    assert ('b', 2) in items
    assert ('c', 3) in items
    
    # Test with valid item (key, value) tuple that doesn't exist
    assert ('a', 2) not in items
    assert ('d', 4) not in items
    assert ('b', 3) not in items
    
    # Test with non-existent key
    assert ('z', 999) not in items
    
    # Test with invalid argument types (should return False, not raise)
    assert 'invalid' not in items
    assert 42 not in items
    assert None not in items
    assert [] not in items
    
    # Test with tuple of wrong length (should return False)
    assert ('a',) not in items
    assert ('a', 1, 'extra') not in items
    
    # Test with empty pmap
    empty_items = PMapItems(pmap({}))
    assert ('a', 1) not in empty_items
    
    # Test with nested values
    m_nested = pmap({'key': (1, 2), 'other': {'nested': 'value'}})
    items_nested = PMapItems(m_nested)
    assert ('key', (1, 2)) in items_nested
    assert ('other', {'nested': 'value'}) in items_nested
    assert ('key', (1, 3)) not in items_nested


# LLM-generated content at query #14
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item tuple
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items = PMapItems(m)
    
    # Test contains with existing item
    assert ('a', 1) in items
    assert ('b', 2) in items
    assert ('c', 3) in items
    
    # Test contains with non-existing item (wrong value)
    assert ('a', 2) not in items
    assert ('b', 1) not in items
    
    # Test contains with non-existing key
    assert ('d', 4) not in items
    assert ('x', 1) not in items
    
    # Test contains with invalid argument (non-tuple)
    assert 'a' not in items
    assert 1 not in items
    assert None not in items
    
    # Test contains with invalid tuple (not enough elements)
    assert () not in items
    assert ('a',) not in items
    
    # Test contains with invalid tuple (too many elements)
    assert ('a', 1, 'extra') not in items
    
    # Test contains with non-unpacking argument
    assert [1, 2, 3] not in items
    assert {'key': 'value'} not in items
    
    # Test with empty pmap
    empty_m = pmap({})
    empty_items = PMapItems(empty_m)
    assert ('a', 1) not in empty_items
    assert () not in empty_items


# LLM-generated content at query #15
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with with add function
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}
    assert m1 == {'a': 1, 'b': 2}  # Original unchanged
    
    # Test update_with keeping leftmost element
    m2 = m(a=1)
    result = m2.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1}
    
    # Test update_with with multiple maps
    m3 = m(a=1, b=2)
    result = m3.update_with(add, m(a=2, c=3), {'a': 5, 'd': 1})
    assert result == {'a': 8, 'b': 2, 'c': 3, 'd': 1}
    
    # Test update_with with empty map
    m4 = m(x=10)
    result = m4.update_with(add, m())
    assert result == {'x': 10}
    
    # Test update_with with no additional maps
    m5 = m(x=5)
    result = m5.update_with(add)
    assert result == {'x': 5}
    
    # Test update_with replacing values (rightmost wins with lambda r: r)
    m6 = m(a=1, b=2)
    result = m6.update_with(lambda l, r: r, m(a=10, c=3), {'a': 100})
    assert result == {'a': 100, 'b': 2, 'c': 3}
    
    # Test update_with with custom merge function (concatenate strings)
    m7 = m(name='John')
    result = m7.update_with(lambda l, r: l + ' ' + r, {'name': 'Doe'})
    assert result == {'name': 'John Doe'}
    
    # Test update_with with new keys only
    m8 = m(a=1)
    result = m8.update_with(add, {'b': 2, 'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test that original map is not modified
    m9 = m(x=1, y=2)
    original_id = id(m9)
    m9.update_with(lambda l, r: r, {'x': 99})
    assert id(m9) == original_id
    assert m9 == {'x': 1, 'y': 2}


# LLM-generated content at query #16
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with with add function
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}
    assert m1 == {'a': 1, 'b': 2}  # Original unchanged
    
    # Test keeping leftmost element
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1}
    
    # Test with multiple maps
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(a=10, c=3), {'a': 20, 'd': 4})
    assert result == {'a': 20, 'b': 2, 'c': 3, 'd': 4}
    
    # Test with empty map
    m1 = m(a=1, b=2)
    result = m1.update_with(add, {})
    assert result == {'a': 1, 'b': 2}
    
    # Test with new keys only
    m1 = m(a=1)
    result = m1.update_with(add, m(b=2, c=3))
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with custom merge function (concatenation)
    m1 = m(x='hello')
    result = m1.update_with(lambda l, r: l + ' ' + r, m(x='world'))
    assert result == {'x': 'hello world'}
    
    # Test with multiple overlapping keys
    m1 = m(a=1, b=2, c=3)
    result = m1.update_with(add, m(b=10, c=20), m(c=100))
    assert result == {'a': 1, 'b': 12, 'c': 123}
    
    # Test that original pmap is not modified
    m1 = m(x=5)
    m2 = m(x=3)
    result = m1.update_with(add, m2)
    assert m1 == {'x': 5}
    assert m2 == {'x': 3}
    assert result == {'x': 8}


# LLM-generated content at query #17
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key, value) tuple
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items = PMapItems(m)
    assert ('a', 1) in items
    assert ('b', 2) in items
    assert ('c', 3) in items
    
    # Test with invalid item (key exists but value doesn't match)
    assert ('a', 2) not in items
    assert ('b', 1) not in items
    
    # Test with non-existent key
    assert ('d', 4) not in items
    assert ('x', 100) not in items
    
    # Test with non-tuple argument (should return False)
    assert 'a' not in items
    assert 1 not in items
    assert None not in items
    
    # Test with invalid tuple (too many elements)
    assert ('a', 1, 'extra') not in items
    
    # Test with invalid tuple (too few elements)
    assert ('a',) not in items
    
    # Test with empty tuple
    assert () not in items
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    assert () not in empty_items
    
    # Test with non-iterable argument (should return False)
    assert 42 not in items
    assert {'a': 1} not in items


# LLM-generated content at query #18
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key-value pair)
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test with invalid item - wrong value
    assert ('a', 999) not in items_view
    assert ('b', 1) not in items_view
    
    # Test with invalid item - key doesn't exist
    assert ('z', 1) not in items_view
    assert ('nonexistent', 2) not in items_view
    
    # Test with non-tuple argument (should return False)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    
    # Test with invalid tuple structure (should return False)
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items_view = PMapItems(empty_map)
    assert ('a', 1) not in empty_items_view
    
    # Test with single element map
    single_map = pmap({'x': 'y'})
    single_items_view = PMapItems(single_map)
    assert ('x', 'y') in single_items_view
    assert ('x', 'z') not in single_items_view
    
    # Test with various value types
    complex_map = pmap({'int': 42, 'str': 'hello', 'list': [1, 2, 3], 'none': None})
    complex_items_view = PMapItems(complex_map)
    
    assert ('int', 42) in complex_items_view
    assert ('str', 'hello') in complex_items_view
    assert ('list', [1, 2, 3]) in complex_items_view
    assert ('none', None) in complex_items_view
    assert ('int', 43) not in complex_items_view


# LLM-generated content at query #19
#--------------------------

```python
def test_PMapItems___contains__():
    """Test PMapItems.__contains__ method"""
    from pyrsistent import pmap
    
    # Create a test pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test valid items are contained
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test invalid items are not contained
    assert ('a', 2) not in items_view
    assert ('d', 4) not in items_view
    assert ('b', 1) not in items_view
    
    # Test non-tuple/non-pair arguments return False
    assert 'a' not in items_view
    assert 1 not in items_view
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view
    assert None not in items_view
    assert [] not in items_view
    
    # Test with empty pmap
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    
    # Test with different value types
    m2 = pmap({'key': [1, 2, 3], 'nested': {'inner': 'value'}})
    items_view2 = PMapItems(m2)
    assert ('key', [1, 2, 3]) in items_view2
    assert ('nested', {'inner': 'value'}) in items_view2
    assert ('key', [1, 2]) not in items_view2


# LLM-generated content at query #20
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with a valid item (key, value) tuple that exists
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items = PMapItems(m)
    assert ('a', 1) in items
    assert ('b', 2) in items
    assert ('c', 3) in items
    
    # Test with a valid item (key, value) tuple that does not exist
    assert ('a', 2) not in items
    assert ('d', 4) not in items
    assert ('b', 3) not in items
    
    # Test with a key that exists but wrong value
    assert ('a', 999) not in items
    
    # Test with non-tuple arguments (should return False, not raise)
    assert 'a' not in items
    assert 1 not in items
    assert None not in items
    
    # Test with invalid tuple unpacking (non-2-element tuple)
    assert (1, 2, 3) not in items
    assert () not in items
    assert ('only_one',) not in items
    
    # Test with empty map
    empty_items = PMapItems(pmap({}))
    assert ('a', 1) not in empty_items
    assert () not in empty_items
    
    # Test with non-hashable values in tuple (should still work)
    m_list = pmap({'key': [1, 2, 3]})
    items_list = PMapItems(m_list)
    assert ('key', [1, 2, 3]) in items_list
    assert ('key', [1, 2, 4]) not in items_list
    
    # Test with None as key and value
    m_none = pmap({None: None, 'a': None})
    items_none = PMapItems(m_none)
    assert (None, None) in items_none
    assert ('a', None) in items_none
    assert (None, 'a') not in items_none


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PMapValues___eq__():
    # Test that a PMapValues object is equal to itself
    m = pmap({'a': 1, 'b': 2})
    values = PMapValues(m)
    assert values == values
    assert values.__eq__(values) is True
    
    # Test that a PMapValues object is not equal to another PMapValues object
    # even if they have the same underlying map
    m2 = pmap({'a': 1, 'b': 2})
    values2 = PMapValues(m2)
    assert not (values == values2)
    assert values.__eq__(values2) is False
    
    # Test that a PMapValues object is not equal to other types
    m3 = pmap({'a': 1, 'b': 2})
    values3 = PMapValues(m3)
    assert not (values3 == [1, 2])
    assert not (values3 == {'a': 1, 'b': 2})
    assert not (values3 == None)
    assert not (values3 == "pmap_values([1, 2])")
    
    # Test with empty pmap
    m_empty = pmap({})
    values_empty1 = PMapValues(m_empty)
    values_empty2 = PMapValues(m_empty)
    assert values_empty1 == values_empty1
    assert not (values_empty1 == values_empty2)
    
    # Test that identity check works (x is self)
    assert values.__eq__(values) is True


# LLM-generated content at query #2
#--------------------------

```python
def test_PMap___getattr__():
    # Test successful attribute access for existing keys
    m1 = m(a=1, b=2, c=3)
    assert m1.a == 1
    assert m1.b == 2
    assert m1.c == 3
    
    # Test that __getattr__ raises AttributeError for non-existent keys
    with pytest.raises(AttributeError) as exc_info:
        m1.nonexistent
    assert "PMap has no attribute 'nonexistent'" in str(exc_info.value)
    
    # Test with string keys that contain special characters
    m2 = m(**{'key-with-dash': 10})
    assert m2['key-with-dash'] == 10
    
    # Test that __getattr__ works with numeric values
    m3 = m(x=42, y=3.14, z=0)
    assert m3.x == 42
    assert m3.y == 3.14
    assert m3.z == 0
    
    # Test with None value
    m4 = m(empty=None)
    assert m4.empty is None
    
    # Test that __getattr__ works with complex values
    m5 = m(nested={'a': 1}, lst=[1, 2, 3])
    assert m5.nested == {'a': 1}
    assert m5.lst == [1, 2, 3]
    
    # Test that regular attribute access still works
    m6 = m(data=100)
    assert len(m6) == 1
    assert m6.data == 100
    
    # Test error message contains the correct class name
    with pytest.raises(AttributeError) as exc_info:
        m1.missing_key
    assert "PMap" in str(exc_info.value)


# LLM-generated content at query #3
#--------------------------

```python
def test_PMapView___reversed__():
    """Test that __reversed__ raises TypeError for PMapView."""
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapView(m)
    
    with pytest.raises(TypeError, match="Persistent maps are not reversible"):
        reversed(view)


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import pmap


def test_PMapView___setattr__():
    """Test that PMapView.__setattr__ raises TypeError for immutability."""
    m = pmap({'a': 1, 'b': 2})
    view = PMapView(m)
    
    # Test that setting an attribute raises TypeError
    with pytest.raises(TypeError, match="PMapView is immutable"):
        view.some_attr = 'value'
    
    # Test that attempting to set multiple different attributes all raise TypeError
    with pytest.raises(TypeError, match="PMapView is immutable"):
        view.x = 10
    
    with pytest.raises(TypeError, match="PMapView is immutable"):
        view._internal = 'test'
    
    # Verify the view is still functional after failed setattr attempts
    assert len(view) == 2


# LLM-generated content at query #5
#--------------------------

```python
def test_PMapValues___eq__():
    # Test that a PMapValues object is equal to itself
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    values = PMapValues(m)
    assert values == values
    
    # Test that two different PMapValues objects with same content are not equal
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    values1 = PMapValues(m1)
    values2 = PMapValues(m2)
    assert not (values1 == values2)
    
    # Test that PMapValues is not equal to other types
    m = pmap({'a': 1, 'b': 2})
    values = PMapValues(m)
    assert not (values == [1, 2])
    assert not (values == {1, 2})
    assert not (values == (1, 2))
    assert not (values == "pmap_values([1, 2])")
    assert not (values == None)
    assert not (values == 42)
    
    # Test that identity check works (is self returns True)
    m = pmap({'x': 10})
    values = PMapValues(m)
    assert values == values  # Same object reference
    
    # Test with empty pmap
    empty_map = pmap({})
    empty_values = PMapValues(empty_map)
    assert empty_values == empty_values
    assert not (empty_values == PMapValues(pmap({})))


# LLM-generated content at query #6
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Create a test pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test: item exists in the view
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test: item does not exist (wrong value)
    assert ('a', 999) not in items_view
    assert ('b', 1) not in items_view
    
    # Test: item does not exist (wrong key)
    assert ('z', 1) not in items_view
    assert ('x', 2) not in items_view
    
    # Test: invalid argument (not a tuple)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    
    # Test: invalid argument (wrong tuple length)
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view
    
    # Test: invalid argument (not unpackable)
    assert [1, 2, 3] not in items_view
    assert {'key': 'value'} not in items_view
    
    # Test: empty pmap
    empty_items_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_items_view
    
    # Test: values that are tuples themselves
    m_with_tuple_values = pmap({'x': (1, 2), 'y': (3, 4)})
    items_view_with_tuples = PMapItems(m_with_tuple_values)
    assert ('x', (1, 2)) in items_view_with_tuples
    assert ('y', (3, 4)) in items_view_with_tuples
    assert ('x', (1, 3)) not in items_view_with_tuples


# LLM-generated content at query #7
#--------------------------

```python
def test_PMap___getattr__():
    """Test __getattr__ method of PMap class"""
    # Test successful attribute access for existing keys
    m1 = m(a=1, b=2, c=3)
    assert m1.a == 1
    assert m1.b == 2
    assert m1.c == 3
    
    # Test that __getattr__ raises AttributeError for non-existent keys
    with pytest.raises(AttributeError) as excinfo:
        m1.nonexistent
    assert "PMap has no attribute 'nonexistent'" in str(excinfo.value)
    
    # Test with string keys
    m2 = m(hello='world', foo='bar')
    assert m2.hello == 'world'
    assert m2.foo == 'bar'
    
    # Test with nested values
    m3 = m(nested=m(inner=42))
    assert m3.nested == m(inner=42)
    assert m3.nested.inner == 42
    
    # Test that AttributeError is raised for empty pmap
    m_empty = m()
    with pytest.raises(AttributeError) as excinfo:
        m_empty.any_key
    assert "PMap has no attribute 'any_key'" in str(excinfo.value)
    
    # Test with various value types
    m4 = m(num=123, lst=[1, 2, 3], dct={'key': 'value'})
    assert m4.num == 123
    assert m4.lst == [1, 2, 3]
    assert m4.dct == {'key': 'value'}
    
    # Test that regular dict-style access still works alongside __getattr__
    m5 = m(x=10, y=20)
    assert m5['x'] == m5.x == 10
    assert m5['y'] == m5.y == 20
    
    # Test that __getattr__ doesn't interfere with special attributes
    m6 = m(a=1)
    assert hasattr(m6, '__class__')
    assert hasattr(m6, '__dict__') or not hasattr(m6, '__dict__')  # May or may not have __dict__ due to __slots__


# LLM-generated content at query #8
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key, value) tuple
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test with invalid item - key exists but value doesn't match
    assert ('a', 999) not in items_view
    assert ('b', 1) not in items_view
    
    # Test with invalid item - key doesn't exist
    assert ('x', 1) not in items_view
    assert ('z', 999) not in items_view
    
    # Test with non-tuple argument (should return False, not raise)
    assert 'not_a_tuple' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    
    # Test with wrong-length tuple (should return False)
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    assert 'anything' not in empty_items
    
    # Test with complex values
    m2 = pmap({'key': [1, 2, 3], 'other': {'nested': 'dict'}})
    items_view2 = PMapItems(m2)
    assert ('key', [1, 2, 3]) in items_view2
    assert ('other', {'nested': 'dict'}) in items_view2
    assert ('key', [1, 2]) not in items_view2


# LLM-generated content at query #9
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item tuple that exists
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items = PMapItems(m)
    assert ('a', 1) in items
    assert ('b', 2) in items
    assert ('c', 3) in items
    
    # Test with valid item tuple that doesn't exist
    assert ('a', 999) not in items
    assert ('z', 1) not in items
    assert ('d', 4) not in items
    
    # Test with invalid arguments that can't be unpacked
    assert 'invalid' not in items
    assert 42 not in items
    assert None not in items
    assert [] not in items
    
    # Test with tuples that can't be unpacked into (k, v)
    assert (1, 2, 3) not in items
    assert (1,) not in items
    
    # Test with empty map
    empty_items = PMapItems(pmap({}))
    assert ('a', 1) not in empty_items
    
    # Test with different value types
    m2 = pmap({'key': 'value', 'num': 42, 'list': [1, 2, 3]})
    items2 = PMapItems(m2)
    assert ('key', 'value') in items2
    assert ('num', 42) in items2
    assert ('list', [1, 2, 3]) in items2
    assert ('key', 'other') not in items2


# LLM-generated content at query #10
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key, value) tuple that exists
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items = PMapItems(m)
    assert ('a', 1) in items
    assert ('b', 2) in items
    assert ('c', 3) in items
    
    # Test with valid item tuple that doesn't exist
    assert ('a', 2) not in items
    assert ('d', 4) not in items
    assert ('b', 1) not in items
    
    # Test with non-tuple argument
    assert 'a' not in items
    assert 1 not in items
    assert None not in items
    
    # Test with empty tuple
    assert () not in items
    
    # Test with single element tuple
    assert ('a',) not in items
    
    # Test with tuple with more than 2 elements
    assert ('a', 1, 'extra') not in items
    
    # Test with non-iterable argument (should return False, not raise)
    assert 42 not in items
    
    # Test with list instead of tuple (should still work since it's unpacked)
    assert ['a', 1] not in items
    
    # Test with empty pmap
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    
    # Test with None values
    m_with_none = pmap({'key': None})
    items_with_none = PMapItems(m_with_none)
    assert ('key', None) in items_with_none
    assert ('key', 'something') not in items_with_none


# LLM-generated content at query #11
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key, value) tuple that exists
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test with valid item tuple that does not exist
    assert ('a', 2) not in items_view
    assert ('d', 4) not in items_view
    assert ('a', 1, 'extra') not in items_view
    
    # Test with non-tuple argument
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    
    # Test with empty tuple
    assert () not in items_view
    
    # Test with single element tuple
    assert ('a',) not in items_view
    
    # Test with tuple of wrong length
    assert ('a', 1, 'extra') not in items_view
    
    # Test with non-unpackable argument
    assert [1, 2] not in items_view
    assert {'key': 'value'} not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    
    # Test with various value types
    m2 = pmap({'x': None, 'y': [], 'z': {}})
    items_view2 = PMapItems(m2)
    assert ('x', None) in items_view2
    assert ('y', []) in items_view2
    assert ('z', {}) in items_view2
    assert ('x', []) not in items_view2


# LLM-generated content at query #12
#--------------------------

```python
def test_PMapItems___contains__():
    # Create a PMap with some items
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test that valid items are contained
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test that invalid items are not contained
    assert ('a', 2) not in items_view
    assert ('d', 4) not in items_view
    assert ('a',) not in items_view
    
    # Test with non-tuple arguments that can't be unpacked
    assert 'not_a_tuple' not in items_view
    assert 42 not in items_view
    assert None not in items_view
    
    # Test with invalid tuple lengths
    assert ('a', 1, 'extra') not in items_view
    assert () not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items_view = PMapItems(empty_map)
    assert ('a', 1) not in empty_items_view
    
    # Test with different value types
    m2 = pmap({'x': [1, 2, 3], 'y': {'nested': 'dict'}})
    items_view2 = PMapItems(m2)
    assert ('x', [1, 2, 3]) in items_view2
    assert ('y', {'nested': 'dict'}) in items_view2
    assert ('x', [1, 2]) not in items_view2


# LLM-generated content at query #13
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key-value pair)
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    # Test contains with existing item
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test contains with non-existing item (wrong value)
    assert ('a', 2) not in items_view
    assert ('b', 1) not in items_view
    
    # Test contains with non-existing key
    assert ('d', 4) not in items_view
    
    # Test contains with invalid argument (not a tuple)
    assert 1 not in items_view
    assert 'a' not in items_view
    assert None not in items_view
    
    # Test contains with invalid tuple format
    assert (1, 2, 3) not in items_view
    assert () not in items_view
    
    # Test contains with single element tuple
    assert ('a',) not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    
    # Test with non-hashable values
    m2 = pmap({'x': [1, 2], 'y': {'nested': 'dict'}})
    items_view2 = PMapItems(m2)
    assert ('x', [1, 2]) in items_view2
    assert ('y', {'nested': 'dict'}) in items_view2


# LLM-generated content at query #14
#--------------------------

```python
def test_PMap_update_with():
    from operator import add
    
    # Test basic update_with with add function
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}
    assert m1 == {'a': 1, 'b': 2}  # Original unchanged
    
    # Test update_with keeping leftmost element
    m2 = m(a=1)
    result = m2.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1}
    assert m2 == {'a': 1}  # Original unchanged
    
    # Test update_with with multiple maps
    m3 = m(a=1, b=2)
    result = m3.update_with(add, m(a=2, c=3), m(a=4, d=5))
    assert result == {'a': 7, 'b': 2, 'c': 3, 'd': 5}
    
    # Test update_with with empty map
    m4 = m(a=1, b=2)
    result = m4.update_with(add)
    assert result == {'a': 1, 'b': 2}
    assert result is m4  # Should return same object when no updates
    
    # Test update_with with dict
    m5 = m(a=1, b=2)
    result = m5.update_with(lambda l, r: r, {'a': 10, 'c': 3})
    assert result == {'a': 10, 'b': 2, 'c': 3}
    
    # Test update_with with custom merge function
    m6 = m(a=[1], b=[2])
    result = m6.update_with(lambda l, r: l + r, m(a=[2], c=[3]))
    assert result == {'a': [1, 2], 'b': [2], 'c': [3]}
    
    # Test update_with with non-overlapping keys
    m7 = m(a=1)
    result = m7.update_with(add, m(b=2), {'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}
    
    # Test update_with returns new PMap
    m8 = m(a=1)
    result = m8.update_with(lambda l, r: r, m(b=2))
    assert result is not m8
    assert isinstance(result, type(m8))


# LLM-generated content at query #15
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key, value) tuple
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test with invalid item - key exists but value doesn't match
    assert ('a', 999) not in items_view
    assert ('b', 1) not in items_view
    
    # Test with key that doesn't exist
    assert ('z', 1) not in items_view
    assert ('nonexistent', 'value') not in items_view
    
    # Test with non-tuple argument (should return False, not raise exception)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [] not in items_view
    assert {} not in items_view
    
    # Test with tuple of wrong length (should return False)
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view
    
    # Test with non-iterable that can't be unpacked (should return False)
    assert 42 not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items_view = PMapItems(empty_map)
    assert ('a', 1) not in empty_items_view
    assert () not in empty_items_view


# LLM-generated content at query #16
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key, value) tuple that exists
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items = PMapItems(m)
    assert ('a', 1) in items
    assert ('b', 2) in items
    assert ('c', 3) in items
    
    # Test with valid item tuple that doesn't exist
    assert ('a', 2) not in items
    assert ('d', 4) not in items
    assert ('b', 1) not in items
    
    # Test with non-existent key
    assert ('z', 100) not in items
    
    # Test with invalid arguments (not a tuple/unpacking fails)
    assert 'invalid' not in items
    assert 123 not in items
    assert None not in items
    assert [] not in items
    
    # Test with tuple of wrong length
    assert ('a',) not in items
    assert ('a', 1, 'extra') not in items
    
    # Test with empty map
    empty_items = PMapItems(pmap({}))
    assert ('a', 1) not in empty_items
    
    # Test with single item map
    single_items = PMapItems(pmap({'x': 'y'}))
    assert ('x', 'y') in single_items
    assert ('x', 'z') not in single_items
    
    # Test with complex values
    m_complex = pmap({'key1': [1, 2, 3], 'key2': {'nested': 'dict'}})
    complex_items = PMapItems(m_complex)
    assert ('key1', [1, 2, 3]) in complex_items
    assert ('key2', {'nested': 'dict'}) in complex_items
    assert ('key1', [1, 2]) not in complex_items


# LLM-generated content at query #17
#--------------------------

```python
def test_PMap___eq__():
    # Test identity comparison
    m1 = m(a=1, b=2)
    assert m1 == m1
    
    # Test equality with identical PMap
    m2 = m(a=1, b=2)
    assert m1 == m2
    
    # Test inequality with different PMap
    m3 = m(a=1, b=3)
    assert not (m1 == m3)
    
    # Test equality with dict
    assert m1 == {'a': 1, 'b': 2}
    
    # Test inequality with dict
    assert not (m1 == {'a': 1, 'b': 3})
    
    # Test inequality with dict of different size
    assert not (m1 == {'a': 1})
    
    # Test equality with different length
    m4 = m(a=1, b=2, c=3)
    assert not (m1 == m4)
    
    # Test with non-Mapping type returns NotImplemented
    assert not (m1 == "not a mapping")
    assert not (m1 == 42)
    assert not (m1 == [('a', 1), ('b', 2)])
    
    # Test with empty maps
    m_empty1 = m()
    m_empty2 = m()
    assert m_empty1 == m_empty2
    assert m_empty1 == {}
    
    # Test with dict-like objects (generic Mapping)
    from collections import OrderedDict
    od = OrderedDict([('a', 1), ('b', 2)])
    assert m1 == od
    
    # Test reflexivity
    assert m1 == m1
    
    # Test symmetry
    m5 = m(x=10, y=20)
    d = {'x': 10, 'y': 20}
    assert m5 == d
    assert d == m5
    
    # Test with cached hash
    m6 = m(p=1, q=2)
    m7 = m(p=1, q=2)
    hash(m6)
    hash(m7)
    assert m6 == m7
    
    # Test inequality when hashes differ but maps are equal
    m8 = m(foo=1, bar=2)
    m9 = m(foo=1, bar=2)
    hash(m8)
    hash(m9)
    assert m8 == m9


# LLM-generated content at query #18
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key, value) tuple
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items = PMapItems(m)
    
    assert ('a', 1) in items
    assert ('b', 2) in items
    assert ('c', 3) in items
    
    # Test with non-existent key
    assert ('d', 4) not in items
    
    # Test with existing key but wrong value
    assert ('a', 999) not in items
    
    # Test with non-tuple argument (should return False)
    assert 'not_a_tuple' not in items
    assert 42 not in items
    assert None not in items
    
    # Test with single element tuple (should return False)
    assert ('a',) not in items
    
    # Test with tuple with more than 2 elements (should return False)
    assert ('a', 1, 'extra') not in items
    
    # Test with non-unpacking argument (should return False)
    assert [] not in items
    assert {} not in items
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    
    # Test with different value types
    m2 = pmap({'key': [1, 2, 3], 'other': {'nested': 'dict'}})
    items2 = PMapItems(m2)
    assert ('key', [1, 2, 3]) in items2
    assert ('other', {'nested': 'dict'}) in items2
    assert ('key', [1, 2]) not in items2


# LLM-generated content at query #19
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item tuple that exists in map
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items = PMapItems(m)
    assert ('a', 1) in items
    assert ('b', 2) in items
    assert ('c', 3) in items
    
    # Test with valid item tuple that does not exist in map
    assert ('a', 2) not in items
    assert ('d', 4) not in items
    assert ('b', 3) not in items
    
    # Test with non-existent key
    assert ('z', 1) not in items
    
    # Test with invalid argument types that cannot be unpacked
    assert "not_a_tuple" not in items
    assert 42 not in items
    assert None not in items
    assert [] not in items
    
    # Test with tuples of wrong length
    assert ('a',) not in items
    assert ('a', 1, 'extra') not in items
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    
    # Test with various value types
    m2 = pmap({'key1': None, 'key2': [], 'key3': {'nested': 'dict'}})
    items2 = PMapItems(m2)
    assert ('key1', None) in items2
    assert ('key2', []) in items2
    assert ('key3', {'nested': 'dict'}) in items2
    assert ('key1', 'wrong_value') not in items2


# LLM-generated content at query #20
#--------------------------

```python
def test_PMapItems___contains__():
    from pyrsistent import pmap
    
    # Create a PMap and its items view
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = m.items()
    
    # Test valid items are found
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test invalid items are not found
    assert ('a', 2) not in items_view
    assert ('d', 4) not in items_view
    assert ('b', 1) not in items_view
    
    # Test non-tuple arguments return False
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [] not in items_view
    
    # Test unpacking errors return False
    assert (1, 2, 3) not in items_view  # Too many values to unpack
    assert (1,) not in items_view  # Not enough values to unpack
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = empty_map.items()
    assert ('a', 1) not in empty_items
    
    # Test with different value types
    m2 = pmap({'x': [1, 2], 'y': {'nested': 'dict'}, 'z': None})
    items_view2 = m2.items()
    assert ('x', [1, 2]) in items_view2
    assert ('y', {'nested': 'dict'}) in items_view2
    assert ('z', None) in items_view2
    assert ('x', [1, 3]) not in items_view2


# LLM-generated content at query #21
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with itself
    m1 = m(a=1, b=2)
    assert m1 == m1
    
    # Test equality with another PMap with same contents
    m2 = m(a=1, b=2)
    assert m1 == m2
    
    # Test inequality with different PMap
    m3 = m(a=1, b=3)
    assert not (m1 == m3)
    
    # Test equality with dict
    assert m1 == {'a': 1, 'b': 2}
    
    # Test inequality with dict
    assert not (m1 == {'a': 1, 'b': 3})
    
    # Test inequality with non-mapping types
    assert not (m1 == [('a', 1), ('b', 2)])
    assert not (m1 == "not a mapping")
    assert not (m1 == 42)
    
    # Test inequality when lengths differ
    m4 = m(a=1)
    assert not (m1 == m4)
    
    # Test with empty maps
    m_empty1 = m()
    m_empty2 = m()
    assert m_empty1 == m_empty2
    assert m_empty1 == {}
    
    # Test NotImplemented is returned for non-mapping types
    result = m1.__eq__([('a', 1), ('b', 2)])
    assert result is NotImplemented
    
    # Test with dict-like object
    class DictLike:
        def __init__(self, data):
            self.data = data
        def items(self):
            return self.data.items()
        def __len__(self):
            return len(self.data)
    
    dict_like = DictLike({'a': 1, 'b': 2})
    assert m1 == dict_like
    
    # Test with different keys
    m5 = m(a=1, c=2)
    assert not (m1 == m5)
    
    # Test cached hash optimization
    m6 = m(a=1, b=2)
    m7 = m(a=1, b=2)
    hash(m6)  # Cache the hash
    hash(m7)  # Cache the hash
    assert m6 == m7
    
    # Test with different values but same keys
    m8 = m(a=1, b=2)
    m9 = m(a=1, b=3)
    assert not (m8 == m9)


# LLM-generated content at query #22
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key-value pair)
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items = PMapItems(m)
    
    assert ('a', 1) in items
    assert ('b', 2) in items
    assert ('c', 3) in items
    
    # Test with invalid items
    assert ('a', 2) not in items  # Wrong value
    assert ('d', 1) not in items  # Key doesn't exist
    assert ('a',) not in items    # Single element tuple
    
    # Test with non-tuple arguments
    assert 'a' not in items
    assert 1 not in items
    assert None not in items
    
    # Test with invalid tuple structures
    assert ('a', 1, 'extra') not in items  # Too many elements
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    
    # Test with non-unpackable argument (should return False)
    assert [] not in items
    assert {} not in items


# LLM-generated content at query #23
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key, value) tuple
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test with invalid items
    assert ('a', 2) not in items_view
    assert ('d', 1) not in items_view
    assert ('x', 'y') not in items_view
    
    # Test with non-tuple argument
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    
    # Test with non-unpacking argument
    assert [1, 2, 3] not in items_view
    assert {'key': 'value'} not in items_view
    
    # Test with empty map
    empty_items = PMapItems(pmap({}))
    assert ('a', 1) not in empty_items
    
    # Test with tuple of wrong length
    assert ('a', 1, 2) not in items_view
    assert ('a',) not in items_view
    
    # Test with various value types
    m_complex = pmap({'str': 'value', 'int': 42, 'list': [1, 2], 'none': None})
    items_complex = PMapItems(m_complex)
    
    assert ('str', 'value') in items_complex
    assert ('int', 42) in items_complex
    assert ('list', [1, 2]) in items_complex
    assert ('none', None) in items_complex
    assert ('str', 'wrong') not in items_complex


# LLM-generated content at query #24
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid (key, value) tuple that exists in the map
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test with valid (key, value) tuple that doesn't exist in the map
    assert ('a', 2) not in items_view
    assert ('d', 4) not in items_view
    assert ('a', 1.0) not in items_view
    
    # Test with non-tuple arguments
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    
    # Test with invalid tuple unpacking (wrong length)
    assert (1, 2, 3) not in items_view
    assert () not in items_view
    
    # Test with non-iterable argument
    assert 42 not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    assert () not in empty_items
    
    # Test with various value types
    m2 = pmap({'x': None, 'y': [1, 2, 3], 'z': {'nested': 'dict'}})
    items_view2 = PMapItems(m2)
    assert ('x', None) in items_view2
    assert ('y', [1, 2, 3]) in items_view2
    assert ('z', {'nested': 'dict'}) in items_view2
    assert ('x', 'wrong') not in items_view2


# LLM-generated content at query #25
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with valid item (key, value) tuple
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view
    
    # Test with invalid item (key exists but value doesn't match)
    assert ('a', 999) not in items_view
    assert ('b', 1) not in items_view
    
    # Test with key that doesn't exist
    assert ('z', 1) not in items_view
    assert ('nonexistent', 'value') not in items_view
    
    # Test with non-tuple arguments (should return False, not raise)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view
    assert [] not in items_view
    assert {} not in items_view
    
    # Test with tuple of wrong length
    assert ('a',) not in items_view
    assert ('a', 1, 'extra') not in items_view
    
    # Test with empty map
    empty_map = pmap({})
    empty_items = PMapItems(empty_map)
    assert ('a', 1) not in empty_items
    assert 'anything' not in empty_items
    
    # Test with various value types
    m2 = pmap({'key': None, 'list': [1, 2, 3], 'dict': {'nested': 'value'}})
    items_view2 = PMapItems(m2)
    assert ('key', None) in items_view2
    assert ('list', [1, 2, 3]) in items_view2
    assert ('dict', {'nested': 'value'}) in items_view2


