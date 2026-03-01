####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PMapValues___eq__():
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'c': 3, 'd': 4})
    view1 = PMapValues(m1)
    view2 = PMapValues(m2)

    # Test equality with self
    assert view1 == view1
    assert view2 == view2

    # Test inequality with different views
    assert not (view1 == view2)
    assert not (view2 == view1)

    # Test inequality with non-PMapValues objects
    assert not (view1 == 1)
    assert not (view1 == [1, 2])
    assert not (view1 == {'a': 1})


# LLM-generated content at query #2
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #3
#--------------------------

```python
def test_PMap___getattr__():
    # Test accessing existing key
    m = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m.a == 1
    assert m.b == 2

    # Test accessing non-existing key raises AttributeError
    with pytest.raises(AttributeError):
        _ = m.c

    # Test that the error message is correct
    with pytest.raises(AttributeError) as excinfo:
        _ = m.c
    assert "PMap has no attribute 'c'" in str(excinfo.value)

    # Test with empty map
    empty = PMap(0, [])
    with pytest.raises(AttributeError):
        _ = empty.anything

    # Test that it works with keys that look like attributes
    m = PMap(1, [None, [('some_key', 42)]])
    assert m.some_key == 42


# LLM-generated content at query #4
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = pmap({'a': 1, 'b': 2})
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = pmap({'a': 1, 'b': 2})
    assert m1 == m2

    # Test inequality with another PMap with different content
    m3 = pmap({'a': 1, 'b': 3})
    assert m1 != m3

    # Test equality with a regular dict with same content
    d = {'a': 1, 'b': 2}
    assert m1 == d

    # Test inequality with a regular dict with different content
    d2 = {'a': 1, 'b': 3}
    assert m1 != d2

    # Test inequality with a non-Mapping object
    assert m1 != "not a map"

    # Test inequality with a Mapping of different length
    m4 = pmap({'a': 1})
    assert m1 != m4

    # Test inequality with a dict of different length
    d3 = {'a': 1}
    assert m1 != d3

    # Test equality with another PMap with same content but different hash
    m5 = pmap({'a': 1, 'b': 2})
    assert m1 == m5

    # Test inequality with another PMap with different content but same length
    m6 = pmap({'c': 1, 'd': 2})
    assert m1 != m6

    # Test inequality with a dict with different content but same length
    d4 = {'c': 1, 'd': 2}
    assert m1 != d4


# LLM-generated content at query #5
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({"a": 1, "b": 2, "c": 3})
    view = PMapItems(pmap_instance)
    assert ("a", 1) in view
    assert ("b", 2) in view
    assert ("c", 3) in view

    # Test with non-existing key-value pair
    assert ("d", 4) not in view
    assert ("a", 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert "a" not in view
    assert 1 not in view
    assert [1, 2] not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ("a", 1) not in empty_view


# LLM-generated content at query #6
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_obj)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with invalid argument (not a tuple)
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert ('a',) not in pmap_items  # Tuple with only one element

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #7
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    pmap_items = PMapItems(pmap({1: 'a', 2: 'b'}))
    assert pmap_items == pmap_items

    # Test equality with another PMapItems with the same underlying map
    pmap_items_2 = PMapItems(pmap({1: 'a', 2: 'b'}))
    assert pmap_items == pmap_items_2

    # Test inequality with another PMapItems with a different underlying map
    pmap_items_3 = PMapItems(pmap({1: 'a', 2: 'c'}))
    assert not (pmap_items == pmap_items_3)

    # Test inequality with a non-PMapItems object
    assert not (pmap_items == "not a PMapItems")
    assert not (pmap_items == {1: 'a', 2: 'b'})


# LLM-generated content at query #8
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert view == view

    # Test equality with different instance but same content
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    view1 = PMapItems(m1)
    view2 = PMapItems(m2)
    assert view1 == view2

    # Test inequality with different content
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    view1 = PMapItems(m1)
    view2 = PMapItems(m2)
    assert view1 != view2

    # Test inequality with non-PMapItems object
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert view != {'a': 1, 'b': 2}
    assert view != [('a', 1), ('b', 2)]
    assert view != "not a PMapItems"


# LLM-generated content at query #9
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    pmap_instance = pmap({"a": 1, "b": 2})
    view = PMapItems(pmap_instance)
    assert ("a", 1) in view
    assert ("b", 2) in view

    # Test with non-existing item
    assert ("c", 3) not in view
    assert ("a", 2) not in view  # Wrong value

    # Test with invalid item type
    assert "a" not in view  # Not a tuple
    assert (1, 2, 3) not in view  # Tuple too long

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ("a", 1) not in empty_view


# LLM-generated content at query #10
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #11
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_instance)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with invalid argument (not a tuple)
    assert 'a' not in view
    assert 1 not in view
    assert ('a',) not in view  # Single element tuple
    assert ('a', 1, 2) not in view  # Three element tuple

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #12
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [] not in pmap_items

    # Test with empty pmap
    empty_pmap = pmap()
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #13
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),)]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2))])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),)]), PMap(1, [(('b', 4),)]))
    assert m3 == PMap(2, [(('a', 4), ('b', 6))])

    # Test update_with with non-existent keys
    m4 = m1.update_with(lambda l, r: l * r, PMap(1, [(('c', 5),)]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2), ('c', 5))])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: l, PMap(0, []))
    assert m5 == m1

    # Test update_with with regular dict
    m6 = m1.update_with(lambda l, r: max(l, r), {'a': 5, 'c': 10})
    assert m6 == PMap(3, [(('a', 5), ('b', 2), ('c', 10))])

    # Test update_with with key only in second map
    m7 = m1.update_with(lambda l, r: l, PMap(1, [(('c', 10),)]))
    assert m7 == PMap(3, [(('a', 1), ('b', 2), ('c', 10))])

    # Test update_with with key in both maps
    m8 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 10),)]))
    assert m8 == PMap(2, [(('a', 11), ('b', 2))])

    # Test update_with with different merge function
    m9 = m1.update_with(lambda l, r: l - r, PMap(1, [(('a', 1),)]))
    assert m9 == PMap(2, [(('a', 0), ('b', 2))])

    # Test update_with with string concatenation
    m10 = PMap(1, [(('a', 'hello'),)])
    m11 = m10.update_with(lambda l, r: l + r, PMap(1, [(('a', ' world'),)]))
    assert m11 == PMap(1, [(('a', 'hello world'),)])


# LLM-generated content at query #14
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # wrong value

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #15
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    pmap1 = pmap({'a': 1, 'b': 2})
    items1 = PMapItems(pmap1)
    assert items1 == items1

    # Test inequality with different instance
    pmap2 = pmap({'a': 1, 'b': 2})
    items2 = PMapItems(pmap2)
    assert not (items1 == items2)

    # Test inequality with different type
    assert not (items1 == "not a PMapItems")

    # Test equality with same underlying map
    pmap3 = pmap({'a': 1, 'b': 2})
    items3 = PMapItems(pmap3)
    items4 = PMapItems(pmap3)
    assert items3 == items4

    # Test inequality with different underlying maps
    pmap4 = pmap({'a': 1, 'b': 2})
    pmap5 = pmap({'a': 1, 'b': 3})
    items5 = PMapItems(pmap4)
    items6 = PMapItems(pmap5)
    assert not (items5 == items6)


# LLM-generated content at query #16
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m1 = pmap({"a": 1, "b": 2})
    view1 = PMapItems(m1)
    assert view1 == view1

    # Test equality with another PMapItems with same content
    m2 = pmap({"a": 1, "b": 2})
    view2 = PMapItems(m2)
    assert view1 == view2

    # Test inequality with PMapItems with different content
    m3 = pmap({"a": 1, "b": 3})
    view3 = PMapItems(m3)
    assert not (view1 == view3)

    # Test inequality with non-PMapItems object
    assert not (view1 == {"a": 1, "b": 2})
    assert not (view1 == [("a", 1), ("b", 2)])
    assert not (view1 == "not a PMapItems")


# LLM-generated content at query #17
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view

    # Test with tuple of wrong length
    assert ('a', 1, 2) not in view


# LLM-generated content at query #18
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [(('a', 1), ('b', 2))])
    assert m1 == m2

    # Test inequality with another PMap with different content
    m3 = PMap(2, [(('a', 1), ('c', 3))])
    assert m1 != m3

    # Test equality with dict
    assert m1 == {'a': 1, 'b': 2}

    # Test inequality with dict
    assert m1 != {'a': 1, 'c': 3}

    # Test inequality with non-Mapping
    assert m1 != "not a mapping"

    # Test with different sized maps
    m4 = PMap(3, [(('a', 1), ('b', 2), ('c', 3))])
    assert m1 != m4

    # Test with cached hash
    m5 = PMap(2, [(('a', 1), ('b', 2))])
    m5._cached_hash = hash(frozenset(m5.iteritems()))
    assert m1 == m5

    # Test with different cached hash
    m6 = PMap(2, [(('a', 1), ('b', 2))])
    m6._cached_hash = 12345
    assert m1 != m6

    # Test with same buckets reference
    m7 = PMap(2, m1._buckets)
    assert m1 == m7


# LLM-generated content at query #19
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    pmap1 = pmap({'a': 1, 'b': 2})
    items1 = PMapItems(pmap1)
    assert items1 == items1

    # Test equality with another PMapItems with same content
    pmap2 = pmap({'a': 1, 'b': 2})
    items2 = PMapItems(pmap2)
    assert items1 == items2

    # Test inequality with PMapItems with different content
    pmap3 = pmap({'a': 1, 'b': 3})
    items3 = PMapItems(pmap3)
    assert not (items1 == items3)

    # Test inequality with non-PMapItems object
    assert not (items1 == "not a PMapItems")
    assert not (items1 == {'a': 1, 'b': 2})
    assert not (items1 == PMapValues(pmap1))


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m1 = pmap({'a': 1, 'b': 2})
    view1 = PMapItems(m1)
    assert view1 == view1

    # Test inequality with different PMapItems
    m2 = pmap({'a': 1, 'b': 2})
    view2 = PMapItems(m2)
    assert view1 != view2

    # Test inequality with non-PMapItems object
    assert view1 != {'a': 1, 'b': 2}
    assert view1 != [('a', 1), ('b', 2)]

    # Test equality with same content but different PMap
    m3 = pmap({'a': 1, 'b': 2})
    view3 = PMapItems(m3)
    assert view1 != view3

    # Test equality with different content
    m4 = pmap({'a': 1, 'b': 3})
    view4 = PMapItems(m4)
    assert view1 != view4


# LLM-generated content at query #2
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({"a": 1, "b": 2})
    view = PMapItems(pmap_obj)
    assert ("a", 1) in view
    assert ("b", 2) in view

    # Test with non-existing key-value pair
    assert ("c", 3) not in view
    assert ("a", 2) not in view  # Wrong value

    # Test with non-tuple argument
    assert "a" not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ("a", 1) not in empty_view


# LLM-generated content at query #3
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(m)
    assert items_view == items_view

    # Test inequality with different PMapItems instance
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items_view1 = PMapItems(m1)
    items_view2 = PMapItems(m2)
    assert items_view1 != items_view2

    # Test inequality with non-PMapItems object
    m = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(m)
    assert items_view != "not a PMapItems"

    # Test equality with same underlying map
    m = pmap({'a': 1, 'b': 2})
    items_view1 = PMapItems(m)
    items_view2 = PMapItems(m)
    assert items_view1 == items_view2

    # Test inequality with different underlying maps
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items_view1 = PMapItems(m1)
    items_view2 = PMapItems(m2)
    assert items_view1 != items_view2


# LLM-generated content at query #4
#--------------------------

```python
def test_PMapValues___eq__():
    # Test equality with self
    m = pmap({"a": 1, "b": 2})
    view = PMapValues(m)
    assert view == view

    # Test inequality with other instances
    m1 = pmap({"a": 1, "b": 2})
    m2 = pmap({"a": 1, "b": 2})
    view1 = PMapValues(m1)
    view2 = PMapValues(m2)
    assert not (view1 == view2)

    # Test inequality with non-PMapValues objects
    assert not (view1 == "not a PMapValues")
    assert not (view1 == {"a": 1, "b": 2})
    assert not (view1 == [1, 2])


# LLM-generated content at query #5
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with itself
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m2

    # Test inequality with PMap with different content
    m3 = PMap(2, [None, [('a', 1)], [('c', 3)]])
    assert m1 != m3

    # Test inequality with PMap with different size
    m4 = PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])
    assert m1 != m4

    # Test equality with dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with dict with different content
    d2 = {'a': 1, 'c': 3}
    assert m1 != d2

    # Test inequality with non-Mapping type
    assert m1 != "not a mapping"
    assert m1 != 42
    assert m1 != [1, 2, 3]

    # Test with cached hash
    m5 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m6 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m5._cached_hash = 123
    m6._cached_hash = 123
    assert m5 == m6

    m7 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m7._cached_hash = 456
    assert m5 != m7

    # Test with different bucket structure but same content
    m8 = PMap(2, [None, [('b', 2)], [('a', 1)]])
    assert m1 == m8


# LLM-generated content at query #6
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(m)
    assert items_view == items_view

    # Test inequality with different PMapItems
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items_view1 = PMapItems(m1)
    items_view2 = PMapItems(m2)
    assert items_view1 != items_view2

    # Test inequality with non-PMapItems
    m = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(m)
    assert items_view != "not a PMapItems"

    # Test equality with same PMapItems (different instance, same map)
    m = pmap({'a': 1, 'b': 2})
    items_view1 = PMapItems(m)
    items_view2 = PMapItems(m)
    assert items_view1 == items_view2

    # Test inequality with different maps
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items_view1 = PMapItems(m1)
    items_view2 = PMapItems(m2)
    assert items_view1 != items_view2


# LLM-generated content at query #7
#--------------------------

```python
def test_PMapValues___eq__():
    m1 = pmap({"a": 1, "b": 2})
    m2 = pmap({"a": 1, "b": 2})
    view1 = PMapValues(m1)
    view2 = PMapValues(m2)

    # Test equality with self
    assert view1 == view1

    # Test equality with another view with same values
    assert not (view1 == view2)

    # Test equality with non-PMapValues object
    assert not (view1 == {"a": 1, "b": 2})
    assert not (view1 == [1, 2])
    assert not (view1 == None)


# LLM-generated content at query #8
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [(('a', 1), ('b', 2))])
    assert m1 == m2

    # Test inequality with another PMap with different content
    m3 = PMap(2, [(('a', 1), ('c', 3))])
    assert m1 != m3

    # Test equality with a dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with a dict with different content
    d2 = {'a': 1, 'c': 3}
    assert m1 != d2

    # Test inequality with a non-Mapping object
    assert m1 != "not a map"

    # Test inequality with a Mapping of different length
    m4 = PMap(3, [(('a', 1), ('b', 2), ('c', 3))])
    assert m1 != m4

    # Test with cached hash
    m5 = PMap(2, [(('a', 1), ('b', 2))])
    m5._cached_hash = hash(frozenset(m5.iteritems()))
    m6 = PMap(2, [(('a', 1), ('b', 2))])
    m6._cached_hash = hash(frozenset(m6.iteritems()))
    assert m5 == m6

    # Test with different cached hash
    m7 = PMap(2, [(('a', 1), ('b', 2))])
    m7._cached_hash = 123
    m8 = PMap(2, [(('a', 1), ('b', 2))])
    m8._cached_hash = 456
    assert m7 != m8


# LLM-generated content at query #9
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items

    # Test with non-existing key-value pair
    assert ('c', 3) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [] not in pmap_items

    # Test with empty PMap
    empty_pmap_items = PMapItems(pmap({}))
    assert ('a', 1) not in empty_pmap_items


# LLM-generated content at query #10
#--------------------------

```python
def test_PMapValues___eq__():
    # Test equality with self
    m = pmap({"a": 1, "b": 2})
    view = PMapValues(m)
    assert view == view

    # Test inequality with different PMapValues
    m1 = pmap({"a": 1, "b": 2})
    m2 = pmap({"a": 1, "b": 2})
    view1 = PMapValues(m1)
    view2 = PMapValues(m2)
    assert not (view1 == view2)

    # Test inequality with non-PMapValues
    m = pmap({"a": 1, "b": 2})
    view = PMapValues(m)
    assert not (view == "not a PMapValues")
    assert not (view == {"a": 1, "b": 2})
    assert not (view == pmap({"a": 1, "b": 2}))


# LLM-generated content at query #11
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #12
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert ['a', 1] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_pmap_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_pmap_items

    # Test with tuple that can't be unpacked
    assert (('a', 1),) not in pmap_items


# LLM-generated content at query #13
#--------------------------

```python
def test_PMapValues___eq__():
    m = pmap({'a': 1, 'b': 2})
    view = PMapValues(m)

    # Test equality with self
    assert view == view

    # Test inequality with other instances
    other_view = PMapValues(pmap({'a': 1, 'b': 2}))
    assert not (view == other_view)

    # Test inequality with non-PMapValues objects
    assert not (view == {'a': 1, 'b': 2})
    assert not (view == [1, 2])
    assert not (view == None)


# LLM-generated content at query #14
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert ['a', 1] not in view

    # Test with empty PMap
    empty_pmap = pmap()
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #15
#--------------------------

```python
def test_PMapValues___eq__():
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    view1 = PMapValues(m1)
    view2 = PMapValues(m2)

    # Test equality with self
    assert view1 == view1

    # Test equality with another PMapValues instance with same values
    assert not (view1 == view2)

    # Test equality with non-PMapValues object
    assert not (view1 == {'a': 1, 'b': 2})
    assert not (view1 == [1, 2])
    assert not (view1 == None)


# LLM-generated content at query #16
#--------------------------

```python
def test_PMapValues___eq__():
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m3 = pmap({'a': 1, 'b': 3})

    view1 = PMapValues(m1)
    view2 = PMapValues(m2)
    view3 = PMapValues(m3)

    # Test equality with self
    assert view1 == view1

    # Test inequality with different instances
    assert not (view1 == view2)
    assert not (view1 == view3)

    # Test inequality with non-PMapValues objects
    assert not (view1 == 1)
    assert not (view1 == {'a': 1, 'b': 2})
    assert not (view1 == [1, 2])


# LLM-generated content at query #17
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m1

    # Test equality with different PMap with same content
    m2 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m2

    # Test inequality with different content
    m3 = PMap(2, [None, [('a', 1)], [('c', 3)]])
    assert m1 != m3

    # Test equality with dict
    d = {'a': 1, 'b': 2}
    assert m1 == d

    # Test inequality with dict with different content
    d2 = {'a': 1, 'c': 3}
    assert m1 != d2

    # Test inequality with non-Mapping type
    assert m1 != "not a mapping"
    assert m1 != 42
    assert m1 != [1, 2, 3]

    # Test with different sized maps
    m4 = PMap(1, [None, [('a', 1)]])
    assert m1 != m4

    # Test with cached hash
    m5 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m5._cached_hash = hash(frozenset(m5.iteritems()))
    m6 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m6._cached_hash = hash(frozenset(m6.iteritems()))
    assert m5 == m6

    # Test with different cached hash
    m7 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m7._cached_hash = 12345
    m8 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m8._cached_hash = 67890
    assert m7 != m8

    # Test with same buckets reference
    m9 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m10 = PMap(2, m9._buckets)
    assert m9 == m10


# LLM-generated content at query #18
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({"a": 1, "b": 2})
    items_view = PMapItems(pmap_obj)
    assert ("a", 1) in items_view
    assert ("b", 2) in items_view

    # Test with non-existing key-value pair
    assert ("c", 3) not in items_view
    assert ("a", 2) not in items_view  # Wrong value for existing key

    # Test with invalid argument type
    assert "a" not in items_view  # Not a tuple
    assert ("a",) not in items_view  # Tuple with wrong length

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items_view = PMapItems(empty_pmap)
    assert ("a", 1) not in empty_items_view


# LLM-generated content at query #19
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m = pmap({'a': 1, 'b': 2})
    items = PMapItems(m)
    assert items == items

    # Test inequality with different PMapItems
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items1 = PMapItems(m1)
    items2 = PMapItems(m2)
    assert items1 == items2

    # Test inequality with different PMapItems (different maps)
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items1 = PMapItems(m1)
    items2 = PMapItems(m2)
    assert items1 != items2

    # Test inequality with non-PMapItems
    m = pmap({'a': 1, 'b': 2})
    items = PMapItems(m)
    assert items != {'a': 1, 'b': 2}
    assert items != [('a', 1), ('b', 2)]
    assert items != "not a PMapItems"


# LLM-generated content at query #20
#--------------------------

```python
def test_PMapValues___eq__():
    m1 = pmap({"a": 1, "b": 2})
    m2 = pmap({"a": 1, "b": 2})
    m3 = pmap({"a": 1, "b": 3})

    view1 = PMapValues(m1)
    view2 = PMapValues(m2)
    view3 = PMapValues(m3)

    assert view1.__eq__(view1) is True
    assert view1.__eq__(view2) is False
    assert view1.__eq__(view3) is False
    assert view1.__eq__("not a PMapValues") is False


# LLM-generated content at query #21
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_instance)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing item
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #22
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, pvector([None, [('a', 1)], [('b', 2)]]))
    assert m1 == m1

    # Test equality with identical PMap
    m2 = PMap(2, pvector([None, [('a', 1)], [('b', 2)]]))
    assert m1 == m2

    # Test inequality with different PMap
    m3 = PMap(2, pvector([None, [('a', 1)], [('c', 3)]]))
    assert m1 != m3

    # Test equality with dict
    assert m1 == {'a': 1, 'b': 2}

    # Test inequality with dict
    assert m1 != {'a': 1, 'c': 3}

    # Test inequality with non-Mapping
    assert m1 != "not a mapping"

    # Test inequality with different size
    m4 = PMap(3, pvector([None, [('a', 1)], [('b', 2)], [('c', 3)]]))
    assert m1 != m4

    # Test with cached hash
    m5 = PMap(2, pvector([None, [('a', 1)], [('b', 2)]]))
    m5._cached_hash = hash(frozenset(m5.iteritems()))
    assert m1 == m5

    # Test with different cached hash
    m6 = PMap(2, pvector([None, [('a', 1)], [('b', 2)]]))
    m6._cached_hash = 12345
    assert m1 != m6


# LLM-generated content at query #23
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({"a": 1, "b": 2})
    view = PMapItems(pmap_instance)
    assert ("a", 1) in view
    assert ("b", 2) in view

    # Test with non-existing key-value pair
    assert ("c", 3) not in view
    assert ("a", 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert "a" not in view
    assert 1 not in view
    assert [1, 2] not in view

    # Test with empty pmap
    empty_view = PMapItems(pmap({}))
    assert ("a", 1) not in empty_view


# LLM-generated content at query #24
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m2

    # Test inequality with another PMap with different content
    m3 = PMap(2, [None, [('a', 1)], [('b', 3)]])
    assert m1 != m3

    # Test equality with a dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with a dict with different content
    d2 = {'a': 1, 'b': 3}
    assert m1 != d2

    # Test inequality with a non-Mapping object
    assert m1 != "not a mapping"

    # Test with different sizes
    m4 = PMap(1, [None, [('a', 1)]])
    assert m1 != m4

    # Test with different bucket structures but same content
    m5 = PMap(2, [None, [('b', 2)], [('a', 1)]])
    assert m1 == m5

    # Test with cached hash
    m6 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m6._cached_hash = hash(frozenset(m6.iteritems()))
    assert m1 == m6

    # Test with different cached hash
    m7 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m7._cached_hash = 12345
    assert m1 != m7


# LLM-generated content at query #25
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m = pmap({'a': 1, 'b': 2})
    assert m == m

    # Test equality with another PMap with same content
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    assert m1 == m2

    # Test inequality with another PMap with different content
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    assert not (m1 == m2)

    # Test equality with a regular dict with same content
    m = pmap({'a': 1, 'b': 2})
    d = {'a': 1, 'b': 2}
    assert m == d

    # Test inequality with a regular dict with different content
    m = pmap({'a': 1, 'b': 2})
    d = {'a': 1, 'b': 3}
    assert not (m == d)

    # Test inequality with a non-Mapping object
    m = pmap({'a': 1, 'b': 2})
    assert not (m == "not a mapping")

    # Test with different sizes
    m1 = pmap({'a': 1})
    m2 = pmap({'a': 1, 'b': 2})
    assert not (m1 == m2)

    # Test with same buckets but different cached hash
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m1._cached_hash = 1
    m2._cached_hash = 2
    assert m1 == m2

    # Test with same buckets and same cached hash
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m1._cached_hash = 1
    m2._cached_hash = 1
    assert m1 == m2

    # Test with different buckets but same content
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m1._buckets = pvector([None, [('a', 1), ('b', 2)]])
    m2._buckets = pvector([None, [('b', 2), ('a', 1)]])
    assert m1 == m2


# LLM-generated content at query #26
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert ['a', 1] not in view

    # Test with empty PMap
    empty_pmap = pmap()
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view

    # Test with different types
    mixed_pmap = pmap({1: 'one', 'two': 2, (3,): [3]})
    mixed_view = PMapItems(mixed_pmap)
    assert (1, 'one') in mixed_view
    assert ('two', 2) in mixed_view
    assert ((3,), [3]) in mixed_view
    assert (1, 'two') not in mixed_view


# LLM-generated content at query #27
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, pvector([None, [('a', 1)]]))
    m2 = m1.update_with(lambda l, r: r, PMap(1, pvector([None, [('a', 2)]])))
    assert m2['a'] == 2
    assert len(m2) == 1

    # Test update_with with multiple maps
    m1 = PMap(2, pvector([None, [('a', 1), ('b', 2)]]))
    m2 = m1.update_with(lambda l, r: r, PMap(1, pvector([None, [('a', 2)]])), PMap(1, pvector([None, [('c', 3)]])))
    assert m2['a'] == 2
    assert m2['b'] == 2
    assert m2['c'] == 3
    assert len(m2) == 3

    # Test update_with with merge function
    m1 = PMap(2, pvector([None, [('a', 1)]]))
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, pvector([None, [('a', 2)]])))
    assert m2['a'] == 3
    assert len(m2) == 1

    # Test update_with with non-existent key
    m1 = PMap(2, pvector([None, [('a', 1)]]))
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, pvector([None, [('b', 2)]])))
    assert m2['a'] == 1
    assert m2['b'] == 2
    assert len(m2) == 2

    # Test update_with with empty map
    m1 = PMap(0, pvector([]))
    m2 = m1.update_with(lambda l, r: r, PMap(1, pvector([None, [('a', 1)]])))
    assert m2['a'] == 1
    assert len(m2) == 1

    # Test update_with with no maps to update
    m1 = PMap(2, pvector([None, [('a', 1)]]))
    m2 = m1.update_with(lambda l, r: r)
    assert m2['a'] == 1
    assert len(m2) == 1


# LLM-generated content at query #28
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('a', 3)]]),
                        PMap(1, [None, [('a', 4)]]))
    assert m3 == PMap(2, [None, [('a', 8)], [('b', 2)]])

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test with different merge function
    m5 = m1.update_with(lambda l, r: l * r,
                        PMap(1, [None, [('a', 3)]]))
    assert m5 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with dict input
    m6 = m1.update_with(lambda l, r: l + r, {'a': 3})
    assert m6 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test with empty map
    m7 = PMap(0, [])
    m8 = m7.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]))
    assert m8 == PMap(1, [None, [('a', 1)]])

    # Test with no overlapping keys
    m9 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 5)]]))
    assert m9 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test with left preference
    m10 = m1.update_with(lambda l, r: l,
                         PMap(2, [None, [('a', 10)], [('b', 20)]]))
    assert m10 == PMap(2, [None, [('a', 1)], [('b', 2)]])


# LLM-generated content at query #29
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('c', 3)]]))
    assert m3 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])

    # Test update_with with merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, [None, [('a', 2)]]))
    assert m4 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with non-existent key
    m5 = m1.update_with(lambda l, r: l, PMap(1, [None, [('c', 3)]]))
    assert m5 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: r, {'a': 2, 'c': 3})
    assert m7 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])

    # Test update_with with left preference
    m8 = m1.update_with(lambda l, r: l, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m8 == PMap(2, [None, [('a', 1)], [('b', 2)]])


# LLM-generated content at query #30
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),)]))
    assert m2 == PMap(2, [(('a', 4), ('b', 2))])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),)]), PMap(1, [(('b', 4),)]))
    assert m3 == PMap(2, [(('a', 4), ('b', 6))])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 5),)]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2), ('c', 5))])

    # Test update_with with non-existent keys in original map
    m5 = m1.update_with(lambda l, r: l + r, PMap(1, [(('d', 10),)]))
    assert m5 == PMap(3, [(('a', 1), ('b', 2), ('d', 10))])

    # Test update_with with different merge function
    m6 = m1.update_with(lambda l, r: l * r, PMap(1, [(('a', 3),)]))
    assert m6 == PMap(2, [(('a', 3), ('b', 2))])

    # Test update_with with empty map
    m7 = m1.update_with(lambda l, r: l + r)
    assert m7 == m1

    # Test update_with with no overlapping keys
    m8 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 5), ('d', 6))]))
    assert m8 == PMap(4, [(('a', 1), ('b', 2), ('c', 5), ('d', 6))])

    # Test update_with with all overlapping keys
    m9 = m1.update_with(lambda l, r: l + r, PMap(2, [(('a', 3), ('b', 4))]))
    assert m9 == PMap(2, [(('a', 4), ('b', 6))])

    # Test update_with with left preference
    m10 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 100),)]))
    assert m10 == PMap(2, [(('a', 1), ('b', 2))])

    # Test update_with with right preference
    m11 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 100),)]))
    assert m11 == PMap(2, [(('a', 100), ('b', 2))])


# LLM-generated content at query #31
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({1: 'a', 2: 'b', 3: 'c'})
    view = PMapItems(pmap_obj)
    assert (1, 'a') in view
    assert (2, 'b') in view
    assert (3, 'c') in view

    # Test with non-existing key-value pair
    assert (4, 'd') not in view
    assert (1, 'b') not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 1 not in view
    assert 'a' not in view
    assert [1, 'a'] not in view

    # Test with empty pmap
    empty_view = PMapItems(pmap({}))
    assert (1, 'a') not in empty_view


# LLM-generated content at query #32
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert ['a', 1] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #33
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #34
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3))]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2))])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3)])), PMap(1, [(('c', 4))]))
    assert m3 == PMap(3, [(('a', 3), ('b', 2), ('c', 4))])

    # Test update_with with merge function
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3))]))
    assert m4 == PMap(2, [(('a', 4), ('b', 2))])

    # Test update_with with non-existent key
    m5 = m1.update_with(lambda l, r: r, PMap(1, [(('c', 3))]))
    assert m5 == PMap(3, [(('a', 1), ('b', 2), ('c', 3))])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: r, {'a': 5, 'd': 6})
    assert m7 == PMap(3, [(('a', 5), ('b', 2), ('d', 6))])


# LLM-generated content at query #35
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(0, [])
    m1 = m1.set('a', 1).set('b', 2)
    m2 = m1.update_with(lambda l, r: l + r, {'a': 3, 'b': 4})
    assert m2['a'] == 4
    assert m2['b'] == 6

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, {'a': 2}, {'b': 3})
    assert m3['a'] == 2
    assert m3['b'] == 6

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r, {'c': 5})
    assert m4['a'] == 1
    assert m4['b'] == 2
    assert m4['c'] == 5

    # Test with empty map
    m5 = PMap(0, [])
    m6 = m5.update_with(lambda l, r: l + r, {'a': 1})
    assert m6['a'] == 1

    # Test with no overlapping keys
    m7 = m1.update_with(lambda l, r: l + r, {'c': 3, 'd': 4})
    assert m7['a'] == 1
    assert m7['b'] == 2
    assert m7['c'] == 3
    assert m7['d'] == 4

    # Test with custom update function
    m8 = m1.update_with(lambda l, r: l if l > r else r, {'a': 0, 'b': 5})
    assert m8['a'] == 1
    assert m8['b'] == 5

    # Test immutability
    m9 = m1.update_with(lambda l, r: l + r, {'a': 1})
    assert m1['a'] == 1
    assert m9['a'] == 2
    assert m1 is not m9


# LLM-generated content at query #36
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(m)
    assert items_view == items_view

    # Test inequality with different PMapItems instance
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items_view1 = PMapItems(m1)
    items_view2 = PMapItems(m2)
    assert not (items_view1 == items_view2)

    # Test inequality with non-PMapItems object
    m = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(m)
    assert not (items_view == "not a PMapItems")

    # Test equality with same underlying map
    m = pmap({'a': 1, 'b': 2})
    items_view1 = PMapItems(m)
    items_view2 = PMapItems(m)
    assert items_view1 == items_view2


# LLM-generated content at query #37
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('a', 2)]]),
                        PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 3)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test with left preference
    m5 = m1.update_with(lambda l, r: l,
                        PMap(1, [None, [('a', 10)]]),
                        PMap(1, [None, [('a', 20)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test with right preference (default update behavior)
    m6 = m1.update_with(lambda l, r: r,
                        PMap(1, [None, [('a', 10)]]),
                        PMap(1, [None, [('a', 20)]]))
    assert m6 == PMap(2, [None, [('a', 20)], [('b', 2)]])

    # Test with empty map
    m7 = PMap(0, [])
    m8 = m7.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('a', 1)]]))
    assert m8 == PMap(1, [None, [('a', 1)]])

    # Test with no overlapping keys
    m9 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 3)]]))
    assert m9 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test with dict input
    m10 = m1.update_with(lambda l, r: l + r, {'a': 5})
    assert m10 == PMap(2, [None, [('a', 6)], [('b', 2)]])


# LLM-generated content at query #38
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('b', 3)]]))
    assert m3 == PMap(2, [None, [('a', 2)], [('b', 6)]])

    # Test update_with with non-existent keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test update_with with empty maps
    m5 = m1.update_with(lambda l, r: l + r)
    assert m5 == m1

    # Test update_with with different merge functions
    m6 = m1.update_with(lambda l, r: max(l, r), PMap(1, [None, [('a', 3)]]))
    assert m6 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    m7 = m1.update_with(lambda l, r: min(l, r), PMap(1, [None, [('a', 0)]]))
    assert m7 == PMap(2, [None, [('a', 0)], [('b', 2)]])

    # Test update_with with dict
    m8 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m8 == PMap(3, [None, [('a', 4)], [('b', 2)], [('c', 4)]])

    # Test update_with with mixed types
    m9 = m1.update_with(lambda l, r: str(l) + str(r), PMap(1, [None, [('a', 'x')]]), {'b': 'y'})
    assert m9 == PMap(2, [None, [('a', '1x')], [('b', '2y')]])


# LLM-generated content at query #39
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=3, c=4))
    assert m2 == {'a': 4, 'b': 2, 'c': 4}

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, m(a=2), m(b=3), m(c=5))
    assert m3 == {'a': 2, 'b': 6, 'c': 5}

    # Test with no overlapping keys
    m4 = m1.update_with(lambda l, r: l + r, m(c=3, d=4))
    assert m4 == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    # Test with empty map
    m5 = m1.update_with(lambda l, r: l + r)
    assert m5 == m1

    # Test with different update functions
    m6 = m1.update_with(lambda l, r: max(l, r), m(a=0, b=5))
    assert m6 == {'a': 1, 'b': 5}

    # Test with non-existent keys in original map
    m7 = m1.update_with(lambda l, r: l + r, m(c=10))
    assert m7 == {'a': 1, 'b': 2, 'c': 10}

    # Test immutability - original map should remain unchanged
    m1.update_with(lambda l, r: l + r, m(a=100))
    assert m1 == {'a': 1, 'b': 2}

    # Test with dict instead of PMap
    m8 = m1.update_with(lambda l, r: l + r, {'a': 5, 'c': 7})
    assert m8 == {'a': 6, 'b': 2, 'c': 7}

    # Test with complex update function
    m9 = m(a=1, b=2, c=3)
    m10 = m9.update_with(lambda l, r: l if l > r else r, m(a=0, b=5, c=1))
    assert m10 == {'a': 1, 'b': 5, 'c': 3}


# LLM-generated content at query #40
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({"a": 1, "b": 2})
    pmap_items = PMapItems(pmap_obj)
    assert ("a", 1) in pmap_items
    assert ("b", 2) in pmap_items

    # Test with non-existing key-value pair
    assert ("c", 3) not in pmap_items
    assert ("a", 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert "a" not in pmap_items
    assert 1 not in pmap_items
    assert ["a", 1] not in pmap_items

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ("a", 1) not in empty_items


# LLM-generated content at query #41
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3))]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2))])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3))]), PMap(1, [(('b', 4))]))
    assert m3 == PMap(2, [(('a', 4), ('b', 6))])

    # Test with dict
    m4 = m1.update_with(lambda l, r: l * r, {'a': 3})
    assert m4 == PMap(2, [(('a', 3), ('b', 2))])

    # Test with new key
    m5 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 5))]))
    assert m5 == PMap(3, [(('a', 1), ('b', 2)), (('c', 5),)])

    # Test with empty map
    m6 = PMap(0, [])
    m7 = m6.update_with(lambda l, r: l + r, PMap(1, [(('a', 1))]))
    assert m7 == PMap(1, [(('a', 1),)])

    # Test with no overlapping keys
    m8 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 3))]))
    assert m8 == PMap(3, [(('a', 1), ('b', 2)), (('c', 3),)])

    # Test with all overlapping keys
    m9 = m1.update_with(lambda l, r: l + r, PMap(2, [(('a', 3), ('b', 4))]))
    assert m9 == PMap(2, [(('a', 4), ('b', 6))])

    # Test with lambda that ignores left value
    m10 = m1.update_with(lambda l, r: r, PMap(2, [(('a', 10), ('b', 20))]))
    assert m10 == PMap(2, [(('a', 10), ('b', 20))])

    # Test with lambda that ignores right value
    m11 = m1.update_with(lambda l, r: l, PMap(2, [(('a', 10), ('b', 20))]))
    assert m11 == PMap(2, [(('a', 1), ('b', 2))])

    # Test with complex lambda
    m12 = m1.update_with(lambda l, r: l * r + l + r, PMap(1, [(('a', 2))]))
    assert m12 == PMap(2, [(('a', 4), ('b', 2))])


# LLM-generated content at query #42
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('a', 2)]]),
                        PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 3)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test with different update function (keep left value)
    m5 = m1.update_with(lambda l, r: l,
                        PMap(1, [None, [('a', 10)]]),
                        PMap(1, [None, [('a', 20)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test with empty map
    m6 = PMap(0, [])
    m7 = m6.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('a', 1)]]))
    assert m7 == PMap(1, [None, [('a', 1)]])

    # Test with no overlapping keys
    m8 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 3)]]))
    assert m8 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test with dict input
    m9 = m1.update_with(lambda l, r: l + r, {'a': 2, 'c': 3})
    assert m9 == PMap(3, [None, [('a', 3)], [('b', 2)], [('c', 3)]])

    # Test with mixed input types
    m10 = m1.update_with(lambda l, r: l + r,
                         PMap(1, [None, [('a', 2)]]),
                         {'b': 3, 'd': 4})
    assert m10 == PMap(3, [None, [('a', 3)], [('b', 5)], [('d', 4)]])


# LLM-generated content at query #43
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m2

    # Test inequality with another PMap with different content
    m3 = PMap(2, [None, [('a', 1)], [('c', 3)]])
    assert m1 != m3

    # Test equality with dict
    assert m1 == {'a': 1, 'b': 2}

    # Test inequality with dict
    assert m1 != {'a': 1, 'c': 3}

    # Test inequality with non-Mapping
    assert m1 != "not a mapping"

    # Test inequality with different size
    m4 = PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])
    assert m1 != m4

    # Test with cached hash
    m5 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m5._cached_hash = hash(frozenset(m5.iteritems()))
    m6 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m6._cached_hash = hash(frozenset(m6.iteritems()))
    assert m5 == m6

    # Test with different cached hash
    m7 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m7._cached_hash = 123
    m8 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m8._cached_hash = 456
    assert m7 != m8

    # Test with same buckets reference
    m9 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m10 = PMap(2, m9._buckets)
    assert m9 == m10


# LLM-generated content at query #44
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test with non-existent key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test with empty map
    m5 = m1.update_with(lambda l, r: r)
    assert m5 == m1

    # Test with dict
    m6 = m1.update_with(lambda l, r: r, {'a': 10})
    assert m6 == PMap(2, [None, [('a', 10)], [('b', 2)]])

    # Test with different types of mappings
    from collections import OrderedDict
    m7 = m1.update_with(lambda l, r: r, OrderedDict([('a', 20)]))
    assert m7 == PMap(2, [None, [('a', 20)], [('b', 2)]])

    # Test that original map is not modified
    m8 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m1 == PMap(2, [None, [('a', 1)], [('b', 2)]])


# LLM-generated content at query #45
#--------------------------

```python
def test_PMapItems___eq__():
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m3 = pmap({'a': 1, 'b': 3})

    view1 = PMapItems(m1)
    view2 = PMapItems(m2)
    view3 = PMapItems(m3)

    # Test equality with self
    assert view1 == view1

    # Test equality with another view of the same map
    assert view1 == view2

    # Test inequality with a view of a different map
    assert not (view1 == view3)

    # Test inequality with non-PMapItems object
    assert not (view1 == {'a': 1, 'b': 2})
    assert not (view1 == [('a', 1), ('b', 2)])
    assert not (view1 == None)


# LLM-generated content at query #46
#--------------------------

```python
def test_PMapItems___contains__():
    # Setup
    test_map = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(test_map)

    # Test existing item
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test non-existing item
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test invalid item type
    assert 'a' not in view  # Not a tuple
    assert ('a',) not in view  # Tuple with wrong length
    assert ('a', 1, 2) not in view  # Tuple with wrong length


# LLM-generated content at query #47
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with invalid argument (not a tuple)
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert ('a',) not in pmap_items  # Single-element tuple
    assert ('a', 1, 'extra') not in pmap_items  # Three-element tuple

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #48
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({"a": 1, "b": 2, "c": 3})
    view = PMapItems(pmap_instance)
    assert ("a", 1) in view
    assert ("b", 2) in view
    assert ("c", 3) in view

    # Test with non-existing key-value pair
    assert ("d", 4) not in view
    assert ("a", 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert "a" not in view
    assert 1 not in view
    assert ["a", 1] not in view

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ("a", 1) not in empty_view


# LLM-generated content at query #49
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('b', 3)]]))
    assert m3 == PMap(2, [None, [('a', 3)], [('b', 5)]])

    # Test with non-existing key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test with different update function (keep left value)
    m5 = m1.update_with(lambda l, r: l, PMap(1, [None, [('a', 2)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test with empty map
    m6 = PMap(0, [])
    m7 = m6.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]))
    assert m7 == PMap(1, [None, [('a', 1)]])

    # Test with dict input
    m8 = m1.update_with(lambda l, r: l + r, {'a': 2, 'b': 3})
    assert m8 == PMap(2, [None, [('a', 3)], [('b', 5)]])


# LLM-generated content at query #50
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),)]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2))])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),)]), PMap(1, [(('b', 4),)]))
    assert m3 == PMap(2, [(('a', 4), ('b', 6))])

    # Test update_with with non-existent keys
    m4 = m1.update_with(lambda l, r: l * r, PMap(1, [(('c', 5),)]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2), ('c', 5))])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: l - r)
    assert m5 == m1

    # Test update_with with different merge function
    m6 = m1.update_with(lambda l, r: l if l > r else r, PMap(2, [(('a', 0), ('b', 3))]))
    assert m6 == PMap(2, [(('a', 1), ('b', 3))])

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 10, 'c': 20})
    assert m7 == PMap(3, [(('a', 11), ('b', 2), ('c', 20))])

    # Test update_with with mixed types
    m8 = m1.update_with(lambda l, r: str(l) + str(r), PMap(1, [(('a', 'x'),)]))
    assert m8 == PMap(2, [(('a', '1x'), ('b', 2))])

    # Test that original map is not modified
    m9 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 99),)]))
    assert m1 == PMap(2, [(('a', 1), ('b', 2))])
    assert m9 == PMap(2, [(('a', 99), ('b', 2))])


# LLM-generated content at query #51
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({1: 'a', 2: 'b', 3: 'c'})
    view = PMapItems(pmap_obj)
    assert (1, 'a') in view
    assert (2, 'b') in view
    assert (3, 'c') in view

    # Test with non-existing key-value pair
    assert (4, 'd') not in view
    assert (1, 'b') not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 1 not in view
    assert 'a' not in view
    assert [1, 'a'] not in view

    # Test with empty pmap
    empty_view = PMapItems(pmap({}))
    assert (1, 'a') not in empty_view


# LLM-generated content at query #52
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 2),), None]))
    assert m2 == PMap(2, [(('a', 2), ('b', 2)), None])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 2),), None]), PMap(1, [(('b', 3),), None]))
    assert m3 == PMap(2, [(('a', 3), ('b', 5)), None])

    # Test update_with with non-existent keys
    m4 = m1.update_with(lambda l, r: l * r, PMap(1, [(('c', 3),), None]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2)), None, None, (('c', 3),)])

    # Test update_with with empty maps
    m5 = m1.update_with(lambda l, r: l, PMap(0, []))
    assert m5 == m1

    # Test update_with with different merge functions
    m6 = m1.update_with(lambda l, r: l if l > r else r, PMap(1, [(('a', 3),), None]))
    assert m6 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 2, 'b': 3})
    assert m7 == PMap(2, [(('a', 3), ('b', 5)), None])

    # Test update_with with mixed types
    m8 = m1.update_with(lambda l, r: str(l) + str(r), PMap(1, [(('a', 'x'),), None]))
    assert m8 == PMap(2, [(('a', '1x'), ('b', 2)), None])


# LLM-generated content at query #53
#--------------------------

```python
def test_PMapItems___contains__():
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(m)

    # Test existing items
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test non-existing items
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # wrong value
    assert ('b', 1) not in view  # wrong value

    # Test invalid items
    assert 'a' not in view  # not a tuple
    assert ('a',) not in view  # incomplete tuple
    assert ('a', 1, 2) not in view  # extra elements


# LLM-generated content at query #54
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # wrong value

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #55
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [('a', 1)] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #56
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(0, pvector())
    m1 = m1.set('a', 1).set('b', 2)
    m2 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 2))
    assert m2 == PMap(0, pvector()).set('a', 3).set('b', 2)

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r,
                        PMap(0, pvector()).set('a', 2),
                        PMap(0, pvector()).set('a', 3))
    assert m3 == PMap(0, pvector()).set('a', 6).set('b', 2)

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r,
                        PMap(0, pvector()).set('c', 3))
    assert m4 == PMap(0, pvector()).set('a', 1).set('b', 2).set('c', 3)

    # Test with non-existent key in left map
    m5 = m1.update_with(lambda l, r: l + r,
                        PMap(0, pvector()).set('d', 4))
    assert m5 == PMap(0, pvector()).set('a', 1).set('b', 2).set('d', 4)

    # Test with custom update function
    m6 = m1.update_with(lambda l, r: l * r,
                        PMap(0, pvector()).set('a', 2))
    assert m6 == PMap(0, pvector()).set('a', 2).set('b', 2)

    # Test with regular dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 2, 'c': 3})
    assert m7 == PMap(0, pvector()).set('a', 3).set('b', 2).set('c', 3)

    # Test with empty map
    m8 = m1.update_with(lambda l, r: l + r)
    assert m8 == m1

    # Test with leftmost preference
    m9 = m1.update_with(lambda l, r: l,
                        PMap(0, pvector()).set('a', 2),
                        PMap(0, pvector()).set('a', 3))
    assert m9 == PMap(0, pvector()).set('a', 1).set('b', 2)


# LLM-generated content at query #57
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert ['a', 1] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_pmap_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_pmap_items


# LLM-generated content at query #58
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with invalid argument (not a tuple)
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert ['a', 1] not in pmap_items

    # Test with empty PMapItems
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #59
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #60
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing item
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #61
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [(None, [('a', 1)]), (None, [('b', 2)])])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [(None, [('a', 1)]), (None, [('b', 2)])])
    assert m1 == m2

    # Test inequality with PMap with different content
    m3 = PMap(2, [(None, [('a', 1)]), (None, [('c', 3)])])
    assert m1 != m3

    # Test inequality with PMap with different size
    m4 = PMap(1, [(None, [('a', 1)])])
    assert m1 != m4

    # Test equality with dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with dict with different content
    d2 = {'a': 1, 'c': 3}
    assert m1 != d2

    # Test inequality with non-Mapping type
    assert m1 != "not a mapping"

    # Test with cached hash
    m5 = PMap(2, [(None, [('a', 1)]), (None, [('b', 2)])])
    m5._cached_hash = hash(frozenset(m5.iteritems()))
    m6 = PMap(2, [(None, [('a', 1)]), (None, [('b', 2)])])
    m6._cached_hash = hash(frozenset(m6.iteritems()))
    assert m5 == m6

    # Test with different cached hash
    m7 = PMap(2, [(None, [('a', 1)]), (None, [('b', 2)])])
    m7._cached_hash = 123
    m8 = PMap(2, [(None, [('a', 1)]), (None, [('b', 2)])])
    m8._cached_hash = 456
    assert m7 != m8


# LLM-generated content at query #62
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [('a', 1)] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #63
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({"a": 1, "b": 2, "c": 3})
    view = PMapItems(pmap_obj)
    assert ("a", 1) in view
    assert ("b", 2) in view
    assert ("c", 3) in view

    # Test with non-existing key-value pair
    assert ("d", 4) not in view
    assert ("a", 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert "a" not in view
    assert 1 not in view
    assert ["a", 1] not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ("a", 1) not in empty_view


# LLM-generated content at query #64
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('c', 3)]]))
    assert m3 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])

    # Test update_with with merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, [None, [('a', 2)]]))
    assert m4 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with left preference
    m5 = m1.update_with(lambda l, r: l, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: r)
    assert m6 == m1

    # Test update_with with non-existing keys
    m7 = m1.update_with(lambda l, r: r, PMap(1, [None, [('c', 3)]]))
    assert m7 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with dict
    m8 = m1.update_with(lambda l, r: r, {'a': 2, 'c': 3})
    assert m8 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])


# LLM-generated content at query #65
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_obj)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #66
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, pvector([None, [('a', 1)], [('b', 2)]]))
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, pvector([None, [('a', 1)], [('b', 2)]]))
    assert m1 == m2

    # Test inequality with PMap with different content
    m3 = PMap(2, pvector([None, [('a', 1)], [('b', 3)]]))
    assert m1 != m3

    # Test equality with dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with dict with different content
    d2 = {'a': 1, 'b': 3}
    assert m1 != d2

    # Test inequality with non-Mapping type
    assert m1 != "not a mapping"

    # Test with different sized PMaps
    m4 = PMap(3, pvector([None, [('a', 1)], [('b', 2)], [('c', 3)]]))
    assert m1 != m4

    # Test with empty PMaps
    m5 = PMap(0, pvector([]))
    m6 = PMap(0, pvector([]))
    assert m5 == m6
    assert m5 == {}

    # Test with cached hash
    m7 = PMap(2, pvector([None, [('a', 1)], [('b', 2)]]))
    m7._cached_hash = hash(frozenset(m7.iteritems()))
    m8 = PMap(2, pvector([None, [('a', 1)], [('b', 2)]]))
    m8._cached_hash = hash(frozenset(m8.iteritems()))
    assert m7 == m8

    # Test with different cached hashes
    m9 = PMap(2, pvector([None, [('a', 1)], [('b', 2)]]))
    m9._cached_hash = 123
    m10 = PMap(2, pvector([None, [('a', 1)], [('b', 2)]]))
    m10._cached_hash = 456
    assert m9 != m10


# LLM-generated content at query #67
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #68
#--------------------------

```python
def test_PMapItems___contains__():
    # Setup
    pmap_data = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_data)

    # Test existing item
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test non-existing item
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value
    assert ('b', 1) not in pmap_items  # Wrong value

    # Test invalid item type
    assert 'a' not in pmap_items  # Not a tuple
    assert (1, 2, 3) not in pmap_items  # Tuple with more than 2 elements
    assert [] not in pmap_items  # Not a tuple


# LLM-generated content at query #69
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),)]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2))])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),)]), PMap(1, [(('b', 4),)]))
    assert m3 == PMap(2, [(('a', 4), ('b', 6))])

    # Test update_with with non-existent keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 5),)]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2), ('c', 5))])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: r)
    assert m5 == m1

    # Test update_with with different merge function
    m6 = m1.update_with(lambda l, r: l * r, PMap(1, [(('a', 3),)]))
    assert m6 == PMap(2, [(('a', 3), ('b', 2))])

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 3, 'b': 4})
    assert m7 == PMap(2, [(('a', 4), ('b', 6))])

    # Test immutability
    m8 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),)]))
    assert m1 == PMap(2, [(('a', 1), ('b', 2))])
    assert m8 == PMap(2, [(('a', 3), ('b', 2))])


# LLM-generated content at query #70
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('a', 3)]]),
                        PMap(1, [None, [('b', 5)]]))
    assert m3 == PMap(2, [None, [('a', 4)], [('b', 7)]])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 10)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 10)]])

    # Test update_with with different merge function
    m5 = m1.update_with(lambda l, r: l * r,
                        PMap(1, [None, [('a', 3)]]))
    assert m5 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: l + r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 3, 'b': 5})
    assert m7 == PMap(2, [None, [('a', 4)], [('b', 7)]])

    # Test update_with with left preference
    m8 = m1.update_with(lambda l, r: l,
                        PMap(1, [None, [('a', 100)]]),
                        PMap(1, [None, [('a', 200)]]))
    assert m8 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with right preference (default update behavior)
    m9 = m1.update_with(lambda l, r: r,
                        PMap(1, [None, [('a', 100)]]),
                        PMap(1, [None, [('a', 200)]]))
    assert m9 == PMap(2, [None, [('a', 200)], [('b', 2)]])


# LLM-generated content at query #71
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('c', 3)]]))
    assert m3 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])

    # Test update_with with merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, [None, [('a', 2)]]))
    assert m4 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with left preference
    m5 = m1.update_with(lambda l, r: l, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with new keys
    m6 = m1.update_with(lambda l, r: r, PMap(1, [None, [('c', 3)]]))
    assert m6 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with empty map
    m7 = m1.update_with(lambda l, r: r)
    assert m7 == m1

    # Test update_with with dict
    m8 = m1.update_with(lambda l, r: r, {'a': 2, 'c': 3})
    assert m8 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])


# LLM-generated content at query #72
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('b', 3)]]))
    assert m3 == PMap(2, [None, [('a', 2)], [('b', 6)]])

    # Test with non-existent key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test with empty map
    m5 = PMap(0, [])
    m6 = m5.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]))
    assert m6 == PMap(1, [None, [('a', 1)]])

    # Test with different update functions
    m7 = m1.update_with(lambda l, r: max(l, r), PMap(1, [None, [('a', 3)]]))
    assert m7 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with dict input
    m8 = m1.update_with(lambda l, r: l + r, {'a': 3, 'b': 4})
    assert m8 == PMap(2, [None, [('a', 4)], [('b', 6)]])


# LLM-generated content at query #73
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert ['a', 1] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #74
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert ['a', 1] not in view

    # Test with empty pmap
    empty_pmap = pmap()
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view

    # Test with different types of keys and values
    mixed_pmap = pmap({1: 'one', 'two': 2, (3,): [3, 3, 3]})
    mixed_view = PMapItems(mixed_pmap)
    assert (1, 'one') in mixed_view
    assert ('two', 2) in mixed_view
    assert ((3,), [3, 3, 3]) in mixed_view
    assert (1, 'two') not in mixed_view


# LLM-generated content at query #75
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),)]))
    assert m2 == PMap(2, [(('a', 4), ('b', 2))])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [(('a', 2),)]), PMap(1, [(('b', 3),)]))
    assert m3 == PMap(2, [(('a', 2), ('b', 6))])

    # Test with non-existing key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 5),)]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2), ('c', 5))])

    # Test with empty map
    m5 = PMap(0, [])
    m6 = m5.update_with(lambda l, r: l + r, PMap(1, [(('a', 1),)]))
    assert m6 == PMap(1, [(('a', 1),)])

    # Test with different merge function
    m7 = m1.update_with(lambda l, r: l if l > r else r, PMap(1, [(('a', 5),)]))
    assert m7 == PMap(2, [(('a', 5), ('b', 2))])

    # Test with regular dict
    m8 = m1.update_with(lambda l, r: l + r, {'a': 3})
    assert m8 == PMap(2, [(('a', 4), ('b', 2))])

    # Test with mixed types
    m9 = m1.update_with(lambda l, r: str(l) + str(r), PMap(1, [(('a', '3'),)]))
    assert m9 == PMap(2, [(('a', '13'), ('b', 2))])

    # Test with no overlapping keys
    m10 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 4),)]))
    assert m10 == PMap(3, [(('a', 1), ('b', 2), ('c', 4))])

    # Test with all overlapping keys
    m11 = m1.update_with(lambda l, r: l * r, PMap(2, [(('a', 2), ('b', 3))]))
    assert m11 == PMap(2, [(('a', 2), ('b', 6))])

    # Test with left preference merge function
    m12 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 10),)]), {'a': 20})
    assert m12 == PMap(2, [(('a', 1), ('b', 2))])


# LLM-generated content at query #76
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # wrong value
    assert ('b', 1) not in view  # wrong value

    # Test with invalid argument type
    assert 'a' not in view  # not a tuple
    assert ('a',) not in view  # single element tuple
    assert ('a', 1, 2) not in view  # triple element tuple

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #77
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('b', 3)]]))
    assert m3 == PMap(2, [None, [('a', 2)], [('b', 6)]])

    # Test update_with with non-existent keys
    m4 = m1.update_with(lambda l, r: l - r, PMap(1, [None, [('c', 1)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', -1)]])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: l, PMap(0, []))
    assert m5 == m1

    # Test update_with with conflicting keys
    m6 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 10)]]), PMap(1, [None, [('a', 20)]]))
    assert m6 == PMap(2, [None, [('a', 20)], [('b', 2)]])

    # Test update_with with different types
    m7 = PMap(1, [None, [('x', [1, 2])]])
    m8 = m7.update_with(lambda l, r: l + r, PMap(1, [None, [('x', [3, 4])]]))
    assert m8 == PMap(1, [None, [('x', [1, 2, 3, 4])]])


# LLM-generated content at query #78
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [1, 2] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #79
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),), None]), PMap(1, [(('b', 4),), None]))
    assert m3 == PMap(2, [(('a', 4), ('b', 6)), None])

    # Test with non-existent key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 5),), None]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2)), None, (('c', 5),)])

    # Test with empty map
    m5 = m1.update_with(lambda l, r: r)
    assert m5 == m1

    # Test with different types of mappings
    m6 = m1.update_with(lambda l, r: r, {'a': 10, 'c': 30})
    assert m6 == PMap(3, [(('a', 10), ('b', 2)), None, (('c', 30),)])

    # Test with custom merge function
    m7 = m1.update_with(lambda l, r: l * r, PMap(1, [(('a', 3),), None]))
    assert m7 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test with left preference merge function
    m8 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 100),), None]))
    assert m8 == m1

    # Test with right preference merge function
    m9 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 100),), None]))
    assert m9 == PMap(2, [(('a', 100), ('b', 2)), None])

    # Test with complex merge function
    m10 = m1.update_with(lambda l, r: str(l) + str(r), PMap(1, [(('a', 3),), None]))
    assert m10 == PMap(2, [(('a', '13'), ('b', 2)), None])


# LLM-generated content at query #80
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert ['a', 1] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #81
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #82
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with invalid argument (not a tuple)
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #83
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items

    # Test with non-existing key-value pair
    assert ('c', 3) not in pmap_items
    assert ('a', 2) not in pmap_items

    # Test with invalid argument (not a tuple)
    assert 'a' not in pmap_items
    assert 1 not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #84
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with itself
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [(('a', 1), ('b', 2))])
    assert m1 == m2

    # Test inequality with another PMap with different content
    m3 = PMap(2, [(('a', 1), ('c', 3))])
    assert m1 != m3

    # Test inequality with another PMap with different size
    m4 = PMap(3, [(('a', 1), ('b', 2), ('c', 3))])
    assert m1 != m4

    # Test equality with a dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with a dict with different content
    d2 = {'a': 1, 'c': 3}
    assert m1 != d2

    # Test inequality with a dict with different size
    d3 = {'a': 1, 'b': 2, 'c': 3}
    assert m1 != d3

    # Test inequality with a non-Mapping object
    assert m1 != "not a mapping"

    # Test with cached hash
    m5 = PMap(2, [(('a', 1), ('b', 2))])
    m6 = PMap(2, [(('a', 1), ('b', 2))])
    m5._cached_hash = 1
    m6._cached_hash = 2
    assert m5 != m6

    # Test with same cached hash
    m7 = PMap(2, [(('a', 1), ('b', 2))])
    m8 = PMap(2, [(('a', 1), ('b', 2))])
    m7._cached_hash = 1
    m8._cached_hash = 1
    assert m7 == m8


# LLM-generated content at query #85
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = pmap(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, pmap(a=3, c=4))
    assert m2 == pmap(a=4, b=2, c=4)

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, pmap(a=2), pmap(b=3, c=5))
    assert m3 == pmap(a=2, b=6, c=5)

    # Test with non-existent keys
    m4 = m1.update_with(lambda l, r: l + r, pmap(c=10, d=20))
    assert m4 == pmap(a=1, b=2, c=10, d=20)

    # Test with empty map
    m5 = m1.update_with(lambda l, r: l + r)
    assert m5 == m1

    # Test with dict instead of PMap
    m6 = m1.update_with(lambda l, r: l + r, {'a': 5, 'd': 10})
    assert m6 == pmap(a=6, b=2, d=10)

    # Test with different merge function
    m7 = m1.update_with(lambda l, r: l if l > r else r, pmap(a=0, b=5))
    assert m7 == pmap(a=1, b=5)

    # Test immutability - original map should remain unchanged
    m8 = m1.update_with(lambda l, r: l + r, pmap(a=100))
    assert m1 == pmap(a=1, b=2)
    assert m8 == pmap(a=101, b=2)

    # Test with KeyError when accessing non-existent key in merge function
    m9 = m1.update_with(lambda l, r: l + r, pmap(c=3))
    assert m9 == pmap(a=1, b=2, c=3)


# LLM-generated content at query #86
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(0, pvector())
    m1 = m1.set('a', 1).set('b', 2)
    m2 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 2))
    assert m2 == PMap(0, pvector()).set('a', 3).set('b', 2)

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r,
                        PMap(0, pvector()).set('a', 2),
                        PMap(0, pvector()).set('a', 3))
    assert m3 == PMap(0, pvector()).set('a', 6).set('b', 2)

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r,
                        PMap(0, pvector()).set('c', 3))
    assert m4 == PMap(0, pvector()).set('a', 1).set('b', 2).set('c', 3)

    # Test with dict
    m5 = m1.update_with(lambda l, r: l + r, {'a': 2})
    assert m5 == PMap(0, pvector()).set('a', 3).set('b', 2)

    # Test with lambda that keeps left value
    m6 = m1.update_with(lambda l, r: l,
                        PMap(0, pvector()).set('a', 2),
                        PMap(0, pvector()).set('a', 3))
    assert m6 == PMap(0, pvector()).set('a', 1).set('b', 2)

    # Test with lambda that keeps right value
    m7 = m1.update_with(lambda l, r: r,
                        PMap(0, pvector()).set('a', 2),
                        PMap(0, pvector()).set('a', 3))
    assert m7 == PMap(0, pvector()).set('a', 3).set('b', 2)

    # Test with empty map
    m8 = PMap(0, pvector())
    m9 = m8.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 1))
    assert m9 == PMap(0, pvector()).set('a', 1)

    # Test with no overlapping keys
    m10 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('c', 3))
    assert m10 == PMap(0, pvector()).set('a', 1).set('b', 2).set('c', 3)

    # Test with complex merge function
    m11 = m1.update_with(lambda l, r: l * r, PMap(0, pvector()).set('a', 2))
    assert m11 == PMap(0, pvector()).set('a', 2).set('b', 2)


# LLM-generated content at query #87
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2})
    view = PMapItems(pmap_instance)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #88
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view

    # Test with empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view

    # Test with more complex values
    complex_map = pmap({'tuple': (1, 2), 'list': pvector([1, 2, 3])})
    complex_view = PMapItems(complex_map)
    assert ('tuple', (1, 2)) in complex_view
    assert ('list', pvector([1, 2, 3])) in complex_view
    assert ('tuple', [1, 2]) not in complex_view  # Different type


# LLM-generated content at query #89
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [] not in pmap_items

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items

    # Test with tuple that can't be unpacked
    assert (('a', 'b'),) not in pmap_items


# LLM-generated content at query #90
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    result = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert result == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with multiple maps
    m2 = PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])
    result = m2.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert result == PMap(3, [None, [('a', 6)], [('b', 2)], [('c', 3)]])

    # Test with non-existent keys
    m3 = PMap(1, [None, [('a', 1)]])
    result = m3.update_with(lambda l, r: l + r, PMap(1, [None, [('b', 2)]]))
    assert result == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test with empty maps
    m4 = PMap(0, [])
    result = m4.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]))
    assert result == PMap(1, [None, [('a', 1)]])

    # Test with different update functions
    m5 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    result = m5.update_with(lambda l, r: l if l > r else r, PMap(1, [None, [('a', 3)]]))
    assert result == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with dict input
    m6 = PMap(1, [None, [('a', 1)]])
    result = m6.update_with(lambda l, r: l + r, {'a': 2, 'b': 3})
    assert result == PMap(2, [None, [('a', 3)], [('b', 3)]])

    # Test with no overlapping keys
    m7 = PMap(1, [None, [('a', 1)]])
    result = m7.update_with(lambda l, r: l + r, PMap(1, [None, [('b', 2)]]))
    assert result == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test with all overlapping keys
    m8 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    result = m8.update_with(lambda l, r: l * r, PMap(2, [None, [('a', 3)], [('b', 4)]]))
    assert result == PMap(2, [None, [('a', 3)], [('b', 8)]])


# LLM-generated content at query #91
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #92
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2})
    view = PMapItems(pmap_instance)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #93
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with custom merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, [None, [('a', 2)]]))
    assert m4 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with leftmost priority
    m5 = m1.update_with(lambda l, r: l, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with new keys
    m6 = m1.update_with(lambda l, r: r, PMap(1, [None, [('c', 3)]]))
    assert m6 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with empty map
    m7 = m1.update_with(lambda l, r: r)
    assert m7 == m1

    # Test update_with with dict
    m8 = m1.update_with(lambda l, r: r, {'a': 2, 'c': 3})
    assert m8 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])

    # Test update_with with non-existent key in merge function
    m9 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m9 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])


# LLM-generated content at query #94
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(m)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test with non-existing key-value pair
    assert ('d', 4) not in items_view
    assert ('a', 2) not in items_view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in items_view
    assert 1 not in items_view
    assert ['a', 1] not in items_view

    # Test with empty map
    empty_map = pmap({})
    empty_view = PMapItems(empty_map)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #95
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]), PMap(1, [(('c', 4),), None]))
    assert m3 == PMap(3, [(('a', 3), ('b', 2)), None, (('c', 4),), None])

    # Test with merge function
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),), None]))
    assert m4 == PMap(2, [(('a', 4), ('b', 2)), None])

    # Test with non-existent key
    m5 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 5),), None]))
    assert m5 == PMap(3, [(('a', 1), ('b', 2)), None, (('c', 5),), None])

    # Test with empty map
    m6 = PMap(0, [])
    m7 = m6.update_with(lambda l, r: r, PMap(1, [(('a', 1),), None]))
    assert m7 == PMap(1, [(('a', 1),), None])

    # Test with dict
    m8 = m1.update_with(lambda l, r: r, {'a': 3, 'c': 4})
    assert m8 == PMap(3, [(('a', 3), ('b', 2)), None, (('c', 4),), None])

    # Test with KeyError in merge function
    m9 = m1.update_with(lambda l, r: l / r, PMap(1, [(('a', 0),), None]))
    try:
        # This should raise an error during iteration
        m9 = m1.update_with(lambda l, r: l / r, PMap(1, [(('a', 0),), None]))
        # If no error is raised, the test should fail
        assert False, "Expected an error but none was raised"
    except ZeroDivisionError:
        pass


# LLM-generated content at query #96
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('c', 3)]]))
    assert m3 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])

    # Test update_with with merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, [None, [('a', 2)]]))
    assert m4 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with left preference
    m5 = m1.update_with(lambda l, r: l, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: r)
    assert m6 == m1

    # Test update_with with non-existent keys
    m7 = m1.update_with(lambda l, r: r, PMap(1, [None, [('c', 3)]]))
    assert m7 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with dict
    m8 = m1.update_with(lambda l, r: r, {'a': 2, 'c': 3})
    assert m8 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])


# LLM-generated content at query #97
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_items = PMapItems(pmap({'a': 1, 'b': 2}))
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items

    # Test with non-existing key-value pair
    assert ('c', 3) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [] not in pmap_items

    # Test with empty PMapItems
    empty_pmap_items = PMapItems(pmap({}))
    assert ('a', 1) not in empty_pmap_items


# LLM-generated content at query #98
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m1

    # Test equality with identical PMap
    m2 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m2

    # Test inequality with different PMap
    m3 = PMap(2, [None, [('a', 1)], [('c', 3)]])
    assert m1 != m3

    # Test equality with dict
    assert m1 == {'a': 1, 'b': 2}

    # Test inequality with different dict
    assert m1 != {'a': 1, 'c': 3}

    # Test inequality with non-Mapping
    assert m1 != "not a mapping"

    # Test inequality with different size
    m4 = PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])
    assert m1 != m4

    # Test with cached hash
    m5 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m5._cached_hash = hash(frozenset(m5.iteritems()))
    m6 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m6._cached_hash = hash(frozenset(m6.iteritems()))
    assert m5 == m6

    # Test with different cached hash
    m7 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m7._cached_hash = 123
    m8 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m8._cached_hash = 456
    assert m7 != m8

    # Test with same buckets reference
    m9 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m10 = PMap(2, m9._buckets)
    assert m9 == m10


# LLM-generated content at query #99
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = pmap({'a': 1, 'b': 2})
    assert m1 == m1

    # Test equality with identical PMap
    m2 = pmap({'a': 1, 'b': 2})
    assert m1 == m2

    # Test inequality with different PMap
    m3 = pmap({'a': 1, 'b': 3})
    assert not (m1 == m3)

    # Test inequality with different size
    m4 = pmap({'a': 1})
    assert not (m1 == m4)

    # Test equality with equivalent dict
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with different dict
    d2 = {'a': 1, 'b': 3}
    assert not (m1 == d2)

    # Test inequality with non-Mapping type
    assert not (m1 == "not a mapping")
    assert not (m1 == 42)
    assert not (m1 == None)

    # Test with cached hash
    m5 = pmap({'x': 10, 'y': 20})
    m6 = pmap({'x': 10, 'y': 20})
    assert m5._cached_hash == m6._cached_hash
    assert m5 == m6

    # Test with different cached hash
    m7 = pmap({'x': 10, 'y': 21})
    assert m5._cached_hash != m7._cached_hash
    assert not (m5 == m7)

    # Test with different types of Mappings
    from collections import OrderedDict
    od = OrderedDict([('a', 1), ('b', 2)])
    assert m1 == od

    # Test with different content in OrderedDict
    od2 = OrderedDict([('a', 1), ('b', 3)])
    assert not (m1 == od2)


# LLM-generated content at query #100
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('c', 3)]]))
    assert m3 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])

    # Test update_with with custom merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, [None, [('a', 2)]]))
    assert m4 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with left-preferring merge
    m5 = m1.update_with(lambda l, r: l, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with non-existent keys
    m6 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m6 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with empty map
    m7 = m1.update_with(lambda l, r: r)
    assert m7 == m1

    # Test update_with with dict
    m8 = m1.update_with(lambda l, r: r, {'a': 10, 'd': 4})
    assert m8 == PMap(3, [None, [('a', 10)], [('b', 2)], [('d', 4)]])

    # Test update_with with non-PMap Mapping
    from collections import OrderedDict
    m9 = m1.update_with(lambda l, r: r, OrderedDict([('a', 10), ('e', 5)]))
    assert m9 == PMap(3, [None, [('a', 10)], [('b', 2)], [('e', 5)]])


# LLM-generated content at query #101
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #102
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = pmap({'a': 1, 'b': 2})
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = pmap({'a': 1, 'b': 2})
    assert m1 == m2

    # Test inequality with PMap with different content
    m3 = pmap({'a': 1, 'b': 3})
    assert m1 != m3

    # Test equality with dict with same content
    d = {'a': 1, 'b': 2}
    assert m1 == d

    # Test inequality with dict with different content
    d2 = {'a': 1, 'b': 3}
    assert m1 != d2

    # Test inequality with non-Mapping type
    assert m1 != "not a mapping"
    assert m1 != 42
    assert m1 != None

    # Test with different sized maps
    m4 = pmap({'a': 1})
    assert m1 != m4

    # Test with empty maps
    m5 = pmap({})
    m6 = pmap({})
    assert m5 == m6
    assert m5 == {}

    # Test with nested structures
    m7 = pmap({'a': pmap({'b': 1})})
    m8 = pmap({'a': pmap({'b': 1})})
    assert m7 == m8

    # Test with different nested structures
    m9 = pmap({'a': pmap({'b': 2})})
    assert m7 != m9


# LLM-generated content at query #103
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #104
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({"a": 1, "b": 2, "c": 3})
    view = PMapItems(pmap_obj)
    assert ("a", 1) in view
    assert ("b", 2) in view
    assert ("c", 3) in view

    # Test with non-existing key-value pair
    assert ("d", 4) not in view
    assert ("a", 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert "a" not in view
    assert 1 not in view
    assert ["a", 1] not in view

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ("a", 1) not in empty_view


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PMapValues___eq__():
    m1 = pmap({"a": 1, "b": 2})
    m2 = pmap({"a": 1, "b": 2})
    m3 = pmap({"a": 1, "b": 3})

    view1 = PMapValues(m1)
    view2 = PMapValues(m2)
    view3 = PMapValues(m3)

    # Test equality with self
    assert view1 == view1

    # Test inequality with different instances
    assert not (view1 == view2)
    assert not (view1 == view3)

    # Test inequality with non-PMapValues objects
    assert not (view1 == "not a view")
    assert not (view1 == None)


# LLM-generated content at query #2
#--------------------------

```python
def test_PMapValues___eq__():
    m = pmap({'a': 1, 'b': 2})
    view = PMapValues(m)

    # Test equality with self
    assert view == view

    # Test inequality with different PMapValues
    m2 = pmap({'a': 1, 'b': 2})
    view2 = PMapValues(m2)
    assert view != view2

    # Test inequality with non-PMapValues objects
    assert view != {'a': 1, 'b': 2}
    assert view != [1, 2]
    assert view != None


# LLM-generated content at query #3
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with itself
    m1 = pmap({'a': 1, 'b': 2})
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = pmap({'a': 1, 'b': 2})
    assert m1 == m2

    # Test inequality with PMap with different content
    m3 = pmap({'a': 1, 'b': 3})
    assert not (m1 == m3)

    # Test inequality with PMap of different size
    m4 = pmap({'a': 1})
    assert not (m1 == m4)

    # Test equality with regular dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with regular dict with different content
    d2 = {'a': 1, 'b': 3}
    assert not (m1 == d2)

    # Test inequality with non-Mapping type
    assert not (m1 == "not a mapping")
    assert not (m1 == 42)
    assert not (m1 == None)

    # Test with empty PMaps
    m_empty1 = pmap({})
    m_empty2 = pmap({})
    assert m_empty1 == m_empty2

    # Test with different hash but same content
    class CustomKey:
        def __init__(self, val):
            self.val = val
        def __hash__(self):
            return hash(self.val)
        def __eq__(self, other):
            return isinstance(other, CustomKey) and self.val == other.val

    m5 = pmap({CustomKey('a'): 1, CustomKey('b'): 2})
    m6 = pmap({CustomKey('a'): 1, CustomKey('b'): 2})
    assert m5 == m6

    # Test with cached hash
    m7 = pmap({'x': 10, 'y': 20})
    m8 = pmap({'x': 10, 'y': 20})
    assert m7._cached_hash == m8._cached_hash
    assert m7 == m8

    # Test with different cached hash
    m9 = pmap({'x': 10, 'y': 20})
    m10 = pmap({'x': 10, 'y': 21})
    assert m9._cached_hash != m10._cached_hash
    assert not (m9 == m10)


# LLM-generated content at query #4
#--------------------------

```python
def test_PMapValues___repr__():
    # Test with empty PMap
    empty_pmap = pmap({})
    empty_view = PMapValues(empty_pmap)
    assert repr(empty_view) == "pmap_values([])"

    # Test with single item PMap
    single_pmap = pmap({"a": 1})
    single_view = PMapValues(single_pmap)
    assert repr(single_view) == "pmap_values([1])"

    # Test with multiple items PMap
    multi_pmap = pmap({"a": 1, "b": 2, "c": 3})
    multi_view = PMapValues(multi_pmap)
    assert repr(multi_view) == "pmap_values([1, 2, 3])"

    # Test with different value types
    mixed_pmap = pmap({"a": "string", "b": 42, "c": 3.14})
    mixed_view = PMapValues(mixed_pmap)
    assert repr(mixed_view) == "pmap_values(['string', 42, 3.14])"


# LLM-generated content at query #5
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    pmap = pmap({"a": 1, "b": 2})
    items = PMapItems(pmap)
    assert items == items

    # Test equality with another PMapItems with same content
    pmap2 = pmap({"a": 1, "b": 2})
    items2 = PMapItems(pmap2)
    assert items == items2

    # Test inequality with another PMapItems with different content
    pmap3 = pmap({"a": 1, "b": 3})
    items3 = PMapItems(pmap3)
    assert items != items3

    # Test inequality with non-PMapItems object
    assert items != {"a": 1, "b": 2}
    assert items != [("a", 1), ("b", 2)]
    assert items != None


# LLM-generated content at query #6
#--------------------------

```python
def test_PMapValues___eq__():
    # Test equality with self
    m = pmap({"a": 1, "b": 2})
    view = PMapValues(m)
    assert view == view

    # Test inequality with other instances
    m1 = pmap({"a": 1, "b": 2})
    m2 = pmap({"a": 1, "b": 2})
    view1 = PMapValues(m1)
    view2 = PMapValues(m2)
    assert not (view1 == view2)

    # Test inequality with non-PMapValues objects
    m = pmap({"a": 1, "b": 2})
    view = PMapValues(m)
    assert not (view == "not a PMapValues")
    assert not (view == {"a": 1, "b": 2})
    assert not (view == [1, 2])


# LLM-generated content at query #7
#--------------------------

```python
def test_PMapItems___eq__():
    pmap1 = pmap({'a': 1, 'b': 2})
    pmap2 = pmap({'a': 1, 'b': 2})
    pmap3 = pmap({'a': 1, 'b': 3})

    items1 = PMapItems(pmap1)
    items2 = PMapItems(pmap2)
    items3 = PMapItems(pmap3)

    assert items1 == items1
    assert items1 == items2
    assert not (items1 == items3)
    assert not (items1 == "not a PMapItems")


# LLM-generated content at query #8
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]), PMap(1, [(('c', 4),), None]))
    assert m3 == PMap(3, [(('a', 3), ('b', 2)), None, (('c', 4),), None])

    # Test update_with with merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, [(('a', 3),), None]))
    assert m4 == PMap(2, [(('a', 4), ('b', 2)), None])

    # Test update_with with non-existent key
    m5 = m1.update_with(lambda l, r: l, PMap(1, [(('c', 3),), None]))
    assert m5 == PMap(3, [(('a', 1), ('b', 2)), None, (('c', 3),), None])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: r, {'a': 5, 'd': 6})
    assert m7 == PMap(3, [(('a', 5), ('b', 2)), None, None, (('d', 6),)])

    # Test update_with with left preference
    m8 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 7),), None]), {'a': 8})
    assert m8 == PMap(2, [(('a', 1), ('b', 2)), None])


# LLM-generated content at query #9
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    pmap = pmap({'a': 1, 'b': 2})
    items = PMapItems(pmap)
    assert items == items

    # Test inequality with different PMapItems
    other_pmap = pmap({'a': 1, 'b': 3})
    other_items = PMapItems(other_pmap)
    assert not (items == other_items)

    # Test inequality with non-PMapItems
    assert not (items == {'a': 1, 'b': 2})
    assert not (items == [('a', 1), ('b', 2)])
    assert not (items == None)

    # Test equality with same PMapItems
    same_items = PMapItems(pmap)
    assert items == same_items


# LLM-generated content at query #10
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing item
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with malformed item
    assert 'a' not in view  # Not a tuple
    assert ('a',) not in view  # Tuple with only one element
    assert ('a', 1, 2) not in view  # Tuple with more than two elements

    # Test with empty pmap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #11
#--------------------------

```python
def test_PMapValues___eq__():
    # Test equality with self
    m1 = pmap({"a": 1, "b": 2})
    view1 = PMapValues(m1)
    assert view1 == view1

    # Test inequality with other instances
    m2 = pmap({"a": 1, "b": 2})
    view2 = PMapValues(m2)
    assert not (view1 == view2)

    # Test inequality with non-PMapValues objects
    assert not (view1 == {"a": 1, "b": 2})
    assert not (view1 == [1, 2])
    assert not (view1 == None)


# LLM-generated content at query #12
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_instance)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Test with empty PMap
    empty_pmap = pmap()
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #13
#--------------------------

```python
def test_PMapValues___eq__():
    m = pmap({1: 2, 3: 4})
    view = PMapValues(m)

    # Test equality with self
    assert view == view

    # Test inequality with different PMapValues
    m2 = pmap({5: 6})
    view2 = PMapValues(m2)
    assert not (view == view2)

    # Test inequality with non-PMapValues objects
    assert not (view == "not a view")
    assert not (view == [1, 2, 3])
    assert not (view == {1: 2})


# LLM-generated content at query #14
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]), PMap(1, [(('c', 4),), None]))
    assert m3 == PMap(3, [(('a', 3), ('b', 2)), None, (('c', 4),), None])

    # Test with merge function (addition)
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),), None]))
    assert m4 == PMap(2, [(('a', 4), ('b', 2)), None])

    # Test with non-existent key
    m5 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 5),), None]))
    assert m5 == PMap(3, [(('a', 1), ('b', 2)), None, (('c', 5),), None])

    # Test with empty map
    m6 = PMap(0, [])
    m7 = m6.update_with(lambda l, r: r, PMap(1, [(('a', 1),), None]))
    assert m7 == PMap(1, [(('a', 1),), None])

    # Test with dict input
    m8 = m1.update_with(lambda l, r: r, {'a': 5, 'd': 6})
    assert m8 == PMap(3, [(('a', 5), ('b', 2)), None, None, (('d', 6),), None])

    # Test that original map is unchanged
    assert m1 == PMap(2, [(('a', 1), ('b', 2)), None])


# LLM-generated content at query #15
#--------------------------

```python
def test_PMapValues___eq__():
    m1 = pmap({"a": 1, "b": 2})
    m2 = pmap({"a": 1, "b": 2})
    view1 = PMapValues(m1)
    view2 = PMapValues(m2)

    assert view1 == view1
    assert not (view1 == view2)
    assert not (view1 == "not a PMapValues")
    assert not (view1 == None)


# LLM-generated content at query #16
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),), None]), PMap(1, [(('b', 4),), None]))
    assert m3 == PMap(2, [(('a', 4), ('b', 6)), None])

    # Test update_with with non-existent keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 5),), None]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2)), None, (('c', 5),)])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: l + r)
    assert m5 == m1

    # Test update_with with different types of mappings
    m6 = m1.update_with(lambda l, r: l + r, {'a': 3, 'b': 4})
    assert m6 == PMap(2, [(('a', 4), ('b', 6)), None])

    # Test update_with with lambda that prefers left value
    m7 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 3),), None]))
    assert m7 == PMap(2, [(('a', 1), ('b', 2)), None])

    # Test update_with with lambda that prefers right value
    m8 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]))
    assert m8 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with lambda that combines values
    m9 = m1.update_with(lambda l, r: l * r, PMap(1, [(('a', 3),), None]))
    assert m9 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with lambda that uses complex logic
    m10 = m1.update_with(lambda l, r: l if l > r else r, PMap(1, [(('a', 3),), None]))
    assert m10 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with lambda that uses complex logic (reverse)
    m11 = m1.update_with(lambda l, r: l if l < r else r, PMap(1, [(('a', 3),), None]))
    assert m11 == PMap(2, [(('a', 1), ('b', 2)), None])


# LLM-generated content at query #17
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert ['a', 1] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #18
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    pmap_instance = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(pmap_instance)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test with non-existing item
    assert ('c', 3) not in items_view
    assert ('a', 2) not in items_view  # Wrong value for existing key

    # Test with invalid item type
    assert 'a' not in items_view  # Not a tuple
    assert ('a',) not in items_view  # Tuple with only one element
    assert ('a', 1, 'extra') not in items_view  # Tuple with more than two elements

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items_view


# LLM-generated content at query #19
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({"a": 1, "b": 2, "c": 3})
    view = PMapItems(pmap_obj)
    assert ("a", 1) in view
    assert ("b", 2) in view

    # Test with non-existing key-value pair
    assert ("d", 4) not in view
    assert ("a", 2) not in view  # Wrong value

    # Test with non-tuple argument
    assert "a" not in view
    assert 1 not in view
    assert [1, 2] not in view

    # Test with empty pmap
    empty_view = PMapItems(pmap({}))
    assert ("a", 1) not in empty_view


# LLM-generated content at query #20
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with invalid argument (not a tuple)
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert ('a',) not in pmap_items  # Tuple with only one element

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_pmap_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_pmap_items


# LLM-generated content at query #21
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m2

    # Test inequality with another PMap with different content
    m3 = PMap(2, [None, [('a', 1)], [('c', 3)]])
    assert m1 != m3

    # Test equality with dict
    assert m1 == {'a': 1, 'b': 2}

    # Test inequality with dict
    assert m1 != {'a': 1, 'c': 3}

    # Test inequality with non-Mapping type
    assert m1 != "not a mapping"

    # Test with different sized PMaps
    m4 = PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])
    assert m1 != m4

    # Test with cached hash
    m5 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m5._cached_hash = hash(frozenset(m5.iteritems()))
    assert m1 == m5

    # Test with different cached hash
    m6 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m6._cached_hash = 999
    assert m1 != m6


# LLM-generated content at query #22
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(pmap_instance)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test with non-existing key-value pair
    assert ('d', 4) not in items_view
    assert ('a', 2) not in items_view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in items_view
    assert 1 not in items_view
    assert [] not in items_view

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items_view


# LLM-generated content at query #23
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, pvector([None, [('a', 1)]]))
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, pvector([None, [('a', 2)]])))
    assert m2 == PMap(1, pvector([None, [('a', 3)]]))

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r,
                        PMap(1, pvector([None, [('a', 2)]])),
                        PMap(1, pvector([None, [('a', 3)]])))
    assert m3 == PMap(1, pvector([None, [('a', 6)]]))

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r,
                        PMap(1, pvector([None, [('b', 2)]])))
    assert m4 == PMap(2, pvector([None, [('a', 1), ('b', 2)]]))

    # Test with different merge function (keep left)
    m5 = m1.update_with(lambda l, r: l,
                        PMap(1, pvector([None, [('a', 2)]])),
                        PMap(1, pvector([None, [('a', 3)]])))
    assert m5 == PMap(1, pvector([None, [('a', 1)]]))

    # Test with empty map
    m6 = PMap(0, pvector([]))
    m7 = m6.update_with(lambda l, r: l + r,
                        PMap(1, pvector([None, [('a', 1)]])))
    assert m7 == PMap(1, pvector([None, [('a', 1)]]))

    # Test with regular dict
    m8 = m1.update_with(lambda l, r: l + r, {'a': 2, 'b': 3})
    assert m8 == PMap(2, pvector([None, [('a', 3), ('b', 3)]]))

    # Test with no overlapping keys
    m9 = m1.update_with(lambda l, r: l + r,
                        PMap(1, pvector([None, [('b', 2)]])),
                        PMap(1, pvector([None, [('c', 3)]])))
    assert m9 == PMap(3, pvector([None, [('a', 1), ('b', 2), ('c', 3)]]))


# LLM-generated content at query #24
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m2

    # Test inequality with another PMap with different content
    m3 = PMap(2, [None, [('a', 1)], [('c', 3)]])
    assert m1 != m3

    # Test equality with a dict with same content
    d = {'a': 1, 'b': 2}
    assert m1 == d

    # Test inequality with a dict with different content
    d2 = {'a': 1, 'c': 3}
    assert m1 != d2

    # Test inequality with a non-Mapping object
    assert m1 != "not a map"

    # Test inequality with a Mapping of different length
    m4 = PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])
    assert m1 != m4

    # Test inequality with a dict of different length
    d3 = {'a': 1, 'b': 2, 'c': 3}
    assert m1 != d3

    # Test cached hash comparison
    m5 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m6 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m5._cached_hash = 123
    m6._cached_hash = 456
    assert m5 != m6

    # Test same buckets reference
    m7 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m8 = PMap(2, m7._buckets)
    assert m7 == m8


# LLM-generated content at query #25
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty pmap
    empty_pmap = pmap()
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #26
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = pmap({'a': 1, 'b': 2})
    assert m1 == m1

    # Test equality with identical PMap
    m2 = pmap({'a': 1, 'b': 2})
    assert m1 == m2

    # Test inequality with different PMap
    m3 = pmap({'a': 1, 'b': 3})
    assert not (m1 == m3)

    # Test inequality with different sized PMap
    m4 = pmap({'a': 1})
    assert not (m1 == m4)

    # Test equality with equivalent dict
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with different dict
    d2 = {'a': 1, 'b': 3}
    assert not (m1 == d2)

    # Test inequality with non-Mapping type
    assert not (m1 == "not a map")
    assert not (m1 == 42)
    assert not (m1 == None)

    # Test with empty maps
    m_empty1 = pmap({})
    m_empty2 = pmap({})
    assert m_empty1 == m_empty2
    assert m_empty1 == {}

    # Test with cached hash
    m5 = pmap({'x': 10, 'y': 20})
    m6 = pmap({'x': 10, 'y': 20})
    assert m5._cached_hash == m6._cached_hash
    assert m5 == m6

    # Test with different cached hash
    m7 = pmap({'x': 10, 'y': 21})
    assert m5._cached_hash != m7._cached_hash
    assert not (m5 == m7)

    # Test with different types of Mappings
    from collections import OrderedDict
    od = OrderedDict([('a', 1), ('b', 2)])
    assert m1 == od

    # Test NotImplemented return for non-Mapping
    class NonMapping:
        pass
    assert (m1 == NonMapping()) == NotImplemented


# LLM-generated content at query #27
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),), None]), PMap(1, [(('b', 4),), None]))
    assert m3 == PMap(2, [(('a', 4), ('b', 6)), None])

    # Test update_with with non-existent keys
    m4 = m1.update_with(lambda l, r: l * r, PMap(1, [(('c', 3),), None]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2)), None, None, (('c', 3),)])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: l, PMap(0, []))
    assert m5 == m1

    # Test update_with with operator functions
    from operator import add, mul
    m6 = m1.update_with(add, PMap(1, [(('a', 2),), None]))
    assert m6 == PMap(2, [(('a', 3), ('b', 2)), None])

    m7 = m1.update_with(mul, PMap(1, [(('a', 2),), None]))
    assert m7 == PMap(2, [(('a', 2), ('b', 2)), None])

    # Test update_with with leftmost priority
    m8 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 3),), None]), PMap(1, [(('a', 4),), None]))
    assert m8 == PMap(2, [(('a', 1), ('b', 2)), None])


# LLM-generated content at query #28
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert ['a', 1] not in view

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #29
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = pmap(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, pmap(a=3, c=4))
    assert m2 == pmap(a=4, b=2, c=4)

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, pmap(a=2), pmap(b=3, d=5))
    assert m3 == pmap(a=2, b=6, d=5)

    # Test with non-existent keys
    m4 = m1.update_with(lambda l, r: l - r, pmap(c=1, d=2))
    assert m4 == pmap(a=1, b=2, c=-1, d=-2)

    # Test with different merge function
    m5 = m1.update_with(lambda l, r: str(l) + str(r), pmap(a=3))
    assert m5 == pmap(a="13", b=2)

    # Test with empty map
    m6 = pmap()
    m7 = m6.update_with(lambda l, r: l + r, pmap(a=1, b=2))
    assert m7 == pmap(a=1, b=2)

    # Test with no updates
    m8 = m1.update_with(lambda l, r: l + r)
    assert m8 == m1

    # Test with dict instead of pmap
    m9 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m9 == pmap(a=4, b=2, c=4)

    # Test with mixed types
    m10 = m1.update_with(lambda l, r: l + r, pmap(a=3), {'b': 4, 'd': 5})
    assert m10 == pmap(a=4, b=6, d=5)

    # Test with left-precedence merge function
    m11 = m1.update_with(lambda l, r: l, pmap(a=10, b=20), pmap(a=100, b=200))
    assert m11 == pmap(a=1, b=2)

    # Test with right-precedence merge function
    m12 = m1.update_with(lambda l, r: r, pmap(a=10, b=20), pmap(a=100, b=200))
    assert m12 == pmap(a=100, b=200)


# LLM-generated content at query #30
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),), None]))
    assert m2 == PMap(2, [(('a', 4), ('b', 2)), None])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [(('a', 2),), None]), PMap(1, [(('b', 3),), None]))
    assert m3 == PMap(2, [(('a', 2), ('b', 6)), None])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 5),), None]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2)), None, (('c', 5),)])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: l + r)
    assert m5 == m1

    # Test update_with with non-PMap mappings
    m6 = m1.update_with(lambda l, r: l + r, {'a': 3, 'b': 4})
    assert m6 == PMap(2, [(('a', 4), ('b', 6)), None])

    # Test update_with with lambda that prefers left value
    m7 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 10),), None]))
    assert m7 == m1

    # Test update_with with lambda that prefers right value
    m8 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 10),), None]))
    assert m8 == PMap(2, [(('a', 10), ('b', 2)), None])

    # Test update_with with complex transformation
    m9 = m1.update_with(lambda l, r: str(l) + str(r), PMap(1, [(('a', 'x'),), None]))
    assert m9 == PMap(2, [(('a', '1x'), ('b', 2)), None])

    # Test that original map is not modified
    _ = m1.update_with(lambda l, r: r, PMap(1, [(('a', 100),), None]))
    assert m1 == PMap(2, [(('a', 1), ('b', 2)), None])


# LLM-generated content at query #31
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 2),), None]))
    assert m2 == PMap(2, [(('a', 2), ('b', 2)), None])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 2),), None]), PMap(1, [(('c', 3),), None]))
    assert m3 == PMap(3, [(('a', 2), ('b', 2)), None, (('c', 3),)])

    # Test update_with with merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, [(('a', 2),), None]))
    assert m4 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with leftmost element kept
    m5 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 2),), None]), PMap(1, [(('a', 3),), None]))
    assert m5 == PMap(2, [(('a', 1), ('b', 2)), None])

    # Test update_with with non-existent key
    m6 = m1.update_with(lambda l, r: l, PMap(1, [(('c', 3),), None]))
    assert m6 == PMap(3, [(('a', 1), ('b', 2)), None, (('c', 3),)])

    # Test update_with with empty map
    m7 = m1.update_with(lambda l, r: r)
    assert m7 == m1

    # Test update_with with non-PMap mapping
    m8 = m1.update_with(lambda l, r: r, {'a': 2, 'c': 3})
    assert m8 == PMap(3, [(('a', 2), ('b', 2)), None, (('c', 3),)])


# LLM-generated content at query #32
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({1: 'a', 2: 'b', 3: 'c'})
    view = PMapItems(pmap_instance)
    assert (1, 'a') in view
    assert (2, 'b') in view
    assert (3, 'c') in view

    # Test with non-existing key-value pair
    assert (4, 'd') not in view
    assert (1, 'b') not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert "not_a_tuple" not in view
    assert 123 not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert (1, 'a') not in empty_view


# LLM-generated content at query #33
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('b', 3)]]))
    assert m3 == PMap(2, [None, [('a', 2)], [('b', 6)]])

    # Test update_with with non-existent key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: l + r)
    assert m5 == m1

    # Test update_with with different types of mappings
    m6 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m6 == PMap(3, [None, [('a', 4)], [('b', 2)], [('c', 4)]])

    # Test update_with with custom function
    def custom_fn(left, right):
        if isinstance(left, int) and isinstance(right, int):
            return left * 2 + right * 3
        return right

    m7 = m1.update_with(custom_fn, {'a': 10, 'b': 20})
    assert m7 == PMap(2, [None, [('a', 26)], [('b', 62)]])

    # Test that original map is not modified
    m8 = m1.update_with(lambda l, r: l + r, {'a': 100})
    assert m1 == PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m8 == PMap(2, [None, [('a', 101)], [('b', 2)]])


# LLM-generated content at query #34
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({"a": 1, "b": 2, "c": 3})
    view = PMapItems(pmap_obj)
    assert ("a", 1) in view
    assert ("b", 2) in view
    assert ("c", 3) in view

    # Test with non-existing key-value pair
    assert ("d", 4) not in view
    assert ("a", 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert "a" not in view
    assert 1 not in view
    assert [1, 2] not in view

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ("a", 1) not in empty_view

    # Test with tuple of wrong length
    assert ("a", 1, 2) not in view


# LLM-generated content at query #35
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test with non-existent key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test with empty map
    m5 = PMap(0, [])
    m6 = m5.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]))
    assert m6 == PMap(1, [None, [('a', 1)]])

    # Test with different update functions
    m7 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]))
    assert m7 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test with dict
    m8 = m1.update_with(lambda l, r: l + r, {'a': 2})
    assert m8 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with multiple dicts
    m9 = m1.update_with(lambda l, r: l + r, {'a': 2}, {'b': 3})
    assert m9 == PMap(2, [None, [('a', 3)], [('b', 5)]])


# LLM-generated content at query #36
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing item (wrong key)
    assert ('d', 4) not in view

    # Test with non-existing item (wrong value)
    assert ('a', 2) not in view

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Test with partial tuple
    assert ('a',) not in view

    # Test with empty pmap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #37
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing item
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view

    # Test with empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #38
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_instance)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #39
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('c', 3)]]))
    assert m3 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])

    # Test update_with with custom merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, [None, [('a', 2)]]))
    assert m4 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with left preference
    m5 = m1.update_with(lambda l, r: l, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with non-existent keys
    m6 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m6 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with empty map
    m7 = m1.update_with(lambda l, r: r)
    assert m7 == m1

    # Test update_with with dict
    m8 = m1.update_with(lambda l, r: r, {'a': 2, 'c': 3})
    assert m8 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])


# LLM-generated content at query #40
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test with non-existent key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test with different update function
    m5 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]))
    assert m5 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test with empty map
    m6 = PMap(0, [])
    m7 = m6.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]))
    assert m7 == PMap(1, [None, [('a', 1)]])

    # Test with dict
    m8 = m1.update_with(lambda l, r: l + r, {'a': 2})
    assert m8 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with multiple dicts
    m9 = m1.update_with(lambda l, r: l + r, {'a': 2}, {'b': 3})
    assert m9 == PMap(2, [None, [('a', 3)], [('b', 5)]])

    # Test with left preference
    m10 = m1.update_with(lambda l, r: l, PMap(1, [None, [('a', 2)]]))
    assert m10 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test with right preference
    m11 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m11 == PMap(2, [None, [('a', 2)], [('b', 2)]])


# LLM-generated content at query #41
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # wrong value

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #42
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    m = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(m)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test with non-existing item
    assert ('c', 3) not in items_view
    assert ('a', 2) not in items_view  # Wrong value

    # Test with non-tuple argument
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test with empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #43
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #44
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(pmap)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test with non-existing key-value pair
    assert ('d', 4) not in items_view
    assert ('a', 2) not in items_view  # wrong value
    assert ('b', 1) not in items_view  # wrong value

    # Test with invalid argument (not a tuple)
    assert 'a' not in items_view
    assert 1 not in items_view
    assert None not in items_view

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items_view


# LLM-generated content at query #45
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with invalid argument type
    assert 'a' not in pmap_items  # Not a tuple
    assert ('a',) not in pmap_items  # Tuple with only one element
    assert ('a', 1, 'extra') not in pmap_items  # Tuple with more than two elements

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #46
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(0, pvector())
    m1 = m1.set('a', 1).set('b', 2)
    m2 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 2))
    assert m2['a'] == 3
    assert m2['b'] == 2

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 2), PMap(0, pvector()).set('a', 3))
    assert m3['a'] == 6
    assert m3['b'] == 2

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('c', 3))
    assert m4['a'] == 1
    assert m4['b'] == 2
    assert m4['c'] == 3

    # Test with regular dict
    m5 = m1.update_with(lambda l, r: l + r, {'a': 2})
    assert m5['a'] == 3
    assert m5['b'] == 2

    # Test with left preference
    m6 = m1.update_with(lambda l, r: l, PMap(0, pvector()).set('a', 2), {'a': 3})
    assert m6['a'] == 1
    assert m6['b'] == 2

    # Test with right preference
    m7 = m1.update_with(lambda l, r: r, PMap(0, pvector()).set('a', 2), {'a': 3})
    assert m7['a'] == 3
    assert m7['b'] == 2

    # Test with empty map
    m8 = PMap(0, pvector()).update_with(lambda l, r: l + r, m1)
    assert m8['a'] == 1
    assert m8['b'] == 2

    # Test with no overlapping keys
    m9 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('c', 3))
    assert m9['a'] == 1
    assert m9['b'] == 2
    assert m9['c'] == 3


# LLM-generated content at query #47
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert ['a', 1] not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #48
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_instance)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #49
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with PMapItems containing some items
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(m)

    # Test existing items
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test non-existing items
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # wrong value
    assert ('b', 1) not in view  # wrong value

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #50
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # wrong value

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [] not in pmap_items
    assert {} not in pmap_items

    # Test with tuple of wrong length
    assert ('a', 1, 3) not in pmap_items
    assert () not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #51
#--------------------------

```python
def test_PMapItems___contains__():
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)

    # Test existing item
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test non-existing item
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value

    # Test non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [] not in pmap_items


# LLM-generated content at query #52
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({"a": 1, "b": 2})
    pmap_items = PMapItems(pmap_obj)
    assert ("a", 1) in pmap_items
    assert ("b", 2) in pmap_items

    # Test with non-existing key-value pair
    assert ("c", 3) not in pmap_items
    assert ("a", 2) not in pmap_items  # wrong value

    # Test with non-tuple argument
    assert "a" not in pmap_items
    assert 1 not in pmap_items

    # Test with empty pmap
    empty_pmap_items = PMapItems(pmap({}))
    assert ("a", 1) not in empty_pmap_items


# LLM-generated content at query #53
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing item
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #54
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({1: 'a', 2: 'b'})
    view = PMapItems(pmap_instance)
    assert (1, 'a') in view
    assert (2, 'b') in view

    # Test with non-existing key-value pair
    assert (3, 'c') not in view
    assert (1, 'b') not in view

    # Test with invalid argument (not a tuple)
    assert "not_a_tuple" not in view
    assert 123 not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert (1, 'a') not in empty_view


# LLM-generated content at query #55
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_obj)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items

    # Test with non-existing item
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [('a', 1)] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items

    # Test with single item PMap
    single_pmap = pmap({'x': 100})
    single_items = PMapItems(single_pmap)
    assert ('x', 100) in single_items
    assert ('y', 200) not in single_items


# LLM-generated content at query #56
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with different merge function
    m5 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]))
    assert m5 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: l + r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 2, 'c': 3})
    assert m7 == PMap(3, [None, [('a', 3)], [('b', 2)], [('c', 3)]])

    # Test update_with with left preference
    m8 = m1.update_with(lambda l, r: l, PMap(2, [None, [('a', 2)], [('b', 3)]]))
    assert m8 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with right preference
    m9 = m1.update_with(lambda l, r: r, PMap(2, [None, [('a', 2)], [('b', 3)]]))
    assert m9 == PMap(2, [None, [('a', 2)], [('b', 3)]])


# LLM-generated content at query #57
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    pmap_instance = pmap({'a': 1, 'b': 2})
    view = PMapItems(pmap_instance)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing item (wrong key)
    assert ('c', 3) not in view

    # Test with non-existing item (wrong value)
    assert ('a', 2) not in view

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [] not in view

    # Test with empty pmap
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #58
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_instance)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert ['a', 1] not in view

    # Test with empty pmap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view

    # Test with non-hashable key
    pmap_with_unhashable = pmap({('a', 'b'): 1})
    view_unhashable = PMapItems(pmap_with_unhashable)
    assert (('a', 'b'), 1) in view_unhashable
    assert (('a', 'c'), 1) not in view_unhashable


# LLM-generated content at query #59
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test update_with with non-existent key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with different merge function
    m5 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]))
    assert m5 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: l + r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 2})
    assert m7 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with left preference
    m8 = m1.update_with(lambda l, r: l, PMap(2, [None, [('a', 2)], [('b', 3)]]))
    assert m8 == PMap(2, [None, [('a', 1)], [('b', 2)]])


# LLM-generated content at query #60
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('a', 3)]]),
                        PMap(1, [None, [('a', 4)]]))
    assert m3 == PMap(2, [None, [('a', 8)], [('b', 2)]])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test update_with with different merge function
    m5 = m1.update_with(lambda l, r: l * r,
                        PMap(1, [None, [('a', 3)]]))
    assert m5 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: l + r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 3, 'b': 4})
    assert m7 == PMap(2, [None, [('a', 4)], [('b', 6)]])

    # Test update_with with left preference
    m8 = m1.update_with(lambda l, r: l,
                        PMap(1, [None, [('a', 10)]]),
                        PMap(1, [None, [('a', 20)]]))
    assert m8 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with right preference (default update behavior)
    m9 = m1.update_with(lambda l, r: r,
                        PMap(1, [None, [('a', 10)]]),
                        PMap(1, [None, [('a', 20)]]))
    assert m9 == PMap(2, [None, [('a', 20)], [('b', 2)]])


# LLM-generated content at query #61
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with invalid argument type
    assert 'a' not in view  # Not a tuple
    assert ('a',) not in view  # Tuple with wrong length

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #62
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_obj)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert ['a', 1] not in pmap_items

    # Test with empty PMap
    empty_pmap_items = PMapItems(pmap({}))
    assert ('a', 1) not in empty_pmap_items


# LLM-generated content at query #63
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_obj)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert ['a', 1] not in pmap_items

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #64
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [1, 2] not in view

    # Test with empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view

    # Test with tuple of wrong length
    assert ('a', 1, 2) not in view


# LLM-generated content at query #65
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with different merge function
    m5 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]))
    assert m5 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: l + r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 2})
    assert m7 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with left preference
    m8 = m1.update_with(lambda l, r: l, PMap(2, [None, [('a', 2)], [('b', 3)]]))
    assert m8 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with right preference
    m9 = m1.update_with(lambda l, r: r, PMap(2, [None, [('a', 2)], [('b', 3)]]))
    assert m9 == PMap(2, [None, [('a', 2)], [('b', 3)]])


# LLM-generated content at query #66
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [1, 2] not in view

    # Test with empty pmap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #67
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_instance)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [1, 2] not in view

    # Test with empty pmap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view

    # Test with different types
    mixed_pmap = pmap({1: 'one', 'two': 2, (3,): [3]})
    mixed_view = PMapItems(mixed_pmap)
    assert (1, 'one') in mixed_view
    assert ('two', 2) in mixed_view
    assert ((3,), [3]) in mixed_view
    assert (1, 'two') not in mixed_view


# LLM-generated content at query #68
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert ['a', 1] not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #69
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    m = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(m)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test with non-existing item
    assert ('c', 3) not in items_view
    assert ('a', 2) not in items_view  # wrong value

    # Test with non-tuple argument
    assert 'a' not in items_view
    assert 1 not in items_view
    assert [] not in items_view

    # Test with empty map
    empty_map = pmap({})
    empty_view = PMapItems(empty_map)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #70
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view
    assert ('c', 3) in view

    # Test with non-existing key-value pair
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert [('a', 1)] not in view

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #71
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with different merge function
    m5 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]))
    assert m5 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: l + r)
    assert m6 == m1

    # Test update_with with non-existent key in first map
    m7 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 1)]]))
    assert m7 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 1)]])

    # Test update_with with dict
    m8 = m1.update_with(lambda l, r: l + r, {'a': 2, 'c': 3})
    assert m8 == PMap(3, [None, [('a', 3)], [('b', 2)], [('c', 3)]])


# LLM-generated content at query #72
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('a', 3)]]),
                        PMap(1, [None, [('a', 4)]]))
    assert m3 == PMap(2, [None, [('a', 8)], [('b', 2)]])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test update_with with left preference
    m5 = m1.update_with(lambda l, r: l,
                        PMap(1, [None, [('a', 10)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with right preference
    m6 = m1.update_with(lambda l, r: r,
                        PMap(1, [None, [('a', 10)]]))
    assert m6 == PMap(2, [None, [('a', 10)], [('b', 2)]])

    # Test update_with with empty map
    m7 = m1.update_with(lambda l, r: l + r)
    assert m7 == m1

    # Test update_with with non-existent key in left map
    m8 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('d', 6)]]))
    assert m8 == PMap(3, [None, [('a', 1)], [('b', 2)], [('d', 6)]])

    # Test update_with with complex operation
    m9 = m1.update_with(lambda l, r: l * r,
                        PMap(1, [None, [('a', 3)]]))
    assert m9 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with string concatenation
    m10 = PMap(1, [None, [('a', 'hello')]])
    m11 = m10.update_with(lambda l, r: l + r,
                          PMap(1, [None, [('a', ' world')]]))
    assert m11 == PMap(1, [None, [('a', 'hello world')]])


# LLM-generated content at query #73
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({"a": 1, "b": 2})
    pmap_items = PMapItems(pmap_obj)
    assert ("a", 1) in pmap_items
    assert ("b", 2) in pmap_items

    # Test with non-existing key-value pair
    assert ("c", 3) not in pmap_items
    assert ("a", 2) not in pmap_items

    # Test with non-tuple argument
    assert "a" not in pmap_items
    assert 1 not in pmap_items
    assert [1, 2] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ("a", 1) not in empty_items


# LLM-generated content at query #74
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('c', 3)]]))
    assert m3 == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])

    # Test with merge function that combines values
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m4 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with non-existent key
    m5 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m5 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test with empty map
    m6 = PMap(0, [])
    m7 = m6.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]))
    assert m7 == PMap(1, [None, [('a', 1)]])

    # Test with regular dict
    m8 = m1.update_with(lambda l, r: l + r, {'a': 2})
    assert m8 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with multiple regular dicts
    m9 = m1.update_with(lambda l, r: l + r, {'a': 2}, {'c': 3})
    assert m9 == PMap(3, [None, [('a', 3)], [('b', 2)], [('c', 3)]])

    # Test with leftmost value preference
    m10 = m1.update_with(lambda l, r: l, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m10 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test with complex merge function
    m11 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]))
    assert m11 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test with string concatenation
    m12 = PMap(1, [None, [('a', 'hello')]])
    m13 = m12.update_with(lambda l, r: l + r, PMap(1, [None, [('a', ' world')]]))
    assert m13 == PMap(1, [None, [('a', 'hello world')]])


# LLM-generated content at query #75
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with invalid argument (not a tuple)
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert None not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #76
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('b', 3)]]))
    assert m3 == PMap(2, [None, [('a', 2)], [('b', 6)]])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: l + r)
    assert m5 == m1

    # Test update_with with non-existent key in left map
    m6 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 10)]]))
    assert m6 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 10)]])

    # Test update_with with lambda that ignores left value
    m7 = m1.update_with(lambda l, r: r, PMap(2, [None, [('a', 100)], [('b', 200)]]))
    assert m7 == PMap(2, [None, [('a', 100)], [('b', 200)]])

    # Test update_with with lambda that ignores right value
    m8 = m1.update_with(lambda l, r: l, PMap(2, [None, [('a', 100)], [('b', 200)]]))
    assert m8 == m1

    # Test update_with with dict
    m9 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m9 == PMap(3, [None, [('a', 4)], [('b', 2)], [('c', 4)]])

    # Test update_with with mixed types
    m10 = m1.update_with(lambda l, r: str(l) + str(r), PMap(1, [None, [('a', 'x')]]), {'b': 'y'})
    assert m10 == PMap(2, [None, [('a', '1x')], [('b', '2y')]])


# LLM-generated content at query #77
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items
    assert ('c', 3) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with invalid argument (not a tuple)
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [('a', 1)] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_pmap_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_pmap_items


# LLM-generated content at query #78
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('b', 3)]]))
    assert m3 == PMap(2, [None, [('a', 2)], [('b', 6)]])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: l + r)
    assert m5 == m1

    # Test update_with with dict
    m6 = m1.update_with(lambda l, r: l + r, {'a': 3, 'b': 4})
    assert m6 == PMap(2, [None, [('a', 4)], [('b', 6)]])

    # Test update_with with left preference
    m7 = m1.update_with(lambda l, r: l, PMap(2, [None, [('a', 10)], [('b', 20)]]))
    assert m7 == m1

    # Test update_with with right preference
    m8 = m1.update_with(lambda l, r: r, PMap(2, [None, [('a', 10)], [('b', 20)]]))
    assert m8 == PMap(2, [None, [('a', 10)], [('b', 20)]])

    # Test update_with with complex function
    m9 = m1.update_with(lambda l, r: str(l) + str(r), PMap(1, [None, [('a', 2)]]))
    assert m9 == PMap(2, [None, [('a', '12')], [('b', 2)]])

    # Test update_with with KeyError for non-existent key in evolver
    m10 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 1)]]))
    assert m10 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 1)]])

    # Test update_with with multiple updates to same key
    m11 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]), PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m11 == PMap(2, [None, [('a', 7)], [('b', 2)]])


# LLM-generated content at query #79
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    result = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert result == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with multiple maps
    m2 = PMap(1, [None, [('c', 3)]])
    result = m1.update_with(lambda l, r: l * r, m2, PMap(1, [None, [('a', 2)]]))
    assert result == PMap(3, [None, [('a', 2)], [('b', 2)], [('c', 3)]])

    # Test with non-existent key
    result = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('d', 4)]]))
    assert result == PMap(3, [None, [('a', 1)], [('b', 2)], [('d', 4)]])

    # Test with empty map
    result = m1.update_with(lambda l, r: l + r)
    assert result == m1

    # Test with different merge function
    result = m1.update_with(lambda l, r: l if l > r else r, PMap(1, [None, [('a', 3)]]))
    assert result == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test with dict
    result = m1.update_with(lambda l, r: l + r, {'a': 2, 'c': 3})
    assert result == PMap(3, [None, [('a', 3)], [('b', 2)], [('c', 3)]])

    # Test that original map is not modified
    m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m1 == PMap(2, [None, [('a', 1)], [('b', 2)]])


# LLM-generated content at query #80
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m2

    # Test inequality with PMap with different content
    m3 = PMap(2, [None, [('a', 1)], [('c', 3)]])
    assert m1 != m3

    # Test equality with dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with dict with different content
    d2 = {'a': 1, 'c': 3}
    assert m1 != d2

    # Test inequality with non-Mapping type
    assert m1 != "not a mapping"
    assert m1 != 42
    assert m1 != [1, 2, 3]

    # Test with different sized maps
    m4 = PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])
    assert m1 != m4

    # Test with empty maps
    m5 = PMap(0, [])
    m6 = PMap(0, [])
    assert m5 == m6
    assert m5 == {}

    # Test with cached hash
    m7 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m8 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m7._cached_hash = 123
    m8._cached_hash = 123
    assert m7 == m8

    m9 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m9._cached_hash = 456
    assert m7 != m9

    # Test with different bucket structure but same content
    m10 = PMap(2, [None, [('b', 2)], [('a', 1)]])
    assert m1 == m10


# LLM-generated content at query #81
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('b', 3)]]))
    assert m3 == PMap(2, [None, [('a', 2)], [('b', 6)]])

    # Test with non-existent key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test with empty map
    m5 = PMap(0, [])
    m6 = m5.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]))
    assert m6 == PMap(1, [None, [('a', 1)]])

    # Test with different update function
    m7 = m1.update_with(lambda l, r: max(l, r), PMap(1, [None, [('a', 5)]]))
    assert m7 == PMap(2, [None, [('a', 5)], [('b', 2)]])

    # Test with dict input
    m8 = m1.update_with(lambda l, r: l + r, {'a': 3})
    assert m8 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test with multiple dict inputs
    m9 = m1.update_with(lambda l, r: l + r, {'a': 3}, {'b': 4})
    assert m9 == PMap(2, [None, [('a', 4)], [('b', 6)]])


# LLM-generated content at query #82
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, pvector([None, [('a', 1)], [('b', 2)]]))
    m2 = m1.update_with(lambda l, r: r, PMap(1, pvector([None, [('a', 2)]])))
    assert m2 == PMap(2, pvector([None, [('a', 2)], [('b', 2)]]))

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, pvector([None, [('a', 2)]])), PMap(1, pvector([None, [('c', 3)]])))
    assert m3 == PMap(3, pvector([None, [('a', 2)], [('b', 2)], [('c', 3)]]))

    # Test update_with with merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, pvector([None, [('a', 2)]])))
    assert m4 == PMap(2, pvector([None, [('a', 3)], [('b', 2)]]))

    # Test update_with with non-existent key
    m5 = m1.update_with(lambda l, r: l, PMap(1, pvector([None, [('c', 3)]])))
    assert m5 == PMap(3, pvector([None, [('a', 1)], [('b', 2)], [('c', 3)]]))

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: r, {'a': 2, 'c': 3})
    assert m7 == PMap(3, pvector([None, [('a', 2)], [('b', 2)], [('c', 3)]]))


# LLM-generated content at query #83
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = pmap(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, pmap(a=3, c=4))
    assert m2 == pmap(a=4, b=2, c=4)

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, pmap(a=2), pmap(b=3), pmap(c=4))
    assert m3 == pmap(a=2, b=6, c=4)

    # Test with left-biased merge
    m4 = m1.update_with(lambda l, r: l, pmap(a=10, b=20), pmap(a=100))
    assert m4 == pmap(a=1, b=2)

    # Test with right-biased merge (default update behavior)
    m5 = m1.update_with(lambda l, r: r, pmap(a=10, b=20), pmap(a=100))
    assert m5 == pmap(a=100, b=20)

    # Test with empty maps
    m6 = m1.update_with(lambda l, r: l + r)
    assert m6 == m1

    # Test with non-existent keys
    m7 = m1.update_with(lambda l, r: l + r, pmap(c=5, d=6))
    assert m7 == pmap(a=1, b=2, c=5, d=6)

    # Test with regular dict
    m8 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m8 == pmap(a=4, b=2, c=4)

    # Test immutability
    m9 = m1.update_with(lambda l, r: l + r, pmap(a=3))
    assert m1 == pmap(a=1, b=2)
    assert m9 == pmap(a=4, b=2)

    # Test with complex merge function
    m10 = m1.update_with(lambda l, r: str(l) + str(r), pmap(a=3))
    assert m10 == pmap(a='13', b=2)


# LLM-generated content at query #84
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with non-existent keys in left map
    m5 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m5 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with different merge function
    m6 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]))
    assert m6 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with empty map
    m7 = m1.update_with(lambda l, r: l + r)
    assert m7 == m1

    # Test update_with with dict
    m8 = m1.update_with(lambda l, r: l + r, {'a': 2, 'c': 3})
    assert m8 == PMap(3, [None, [('a', 3)], [('b', 2)], [('c', 3)]])

    # Test update_with with mixed types
    m9 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]), {'a': 3, 'c': 4})
    assert m9 == PMap(3, [None, [('a', 6)], [('b', 2)], [('c', 4)]])


# LLM-generated content at query #85
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(None, [('a', 2)])]))
    assert m2 == PMap(2, [(None, [('a', 2), ('b', 2)])])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [(None, [('a', 2)])]), PMap(1, [(None, [('c', 3)])]))
    assert m3 == PMap(3, [(None, [('a', 2), ('b', 2), ('c', 3)])])

    # Test update_with with merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, [(None, [('a', 2)])]))
    assert m4 == PMap(2, [(None, [('a', 3), ('b', 2)])])

    # Test update_with with left preference
    m5 = m1.update_with(lambda l, r: l, PMap(1, [(None, [('a', 2)])]), PMap(1, [(None, [('a', 3)])]))
    assert m5 == PMap(2, [(None, [('a', 1), ('b', 2)])])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: r)
    assert m6 == m1

    # Test update_with with non-existent keys
    m7 = m1.update_with(lambda l, r: r, PMap(1, [(None, [('c', 3)])]))
    assert m7 == PMap(3, [(None, [('a', 1), ('b', 2), ('c', 3)])])

    # Test update_with with dict
    m8 = m1.update_with(lambda l, r: r, {'a': 2, 'c': 3})
    assert m8 == PMap(3, [(None, [('a', 2), ('b', 2), ('c', 3)])])


# LLM-generated content at query #86
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    pmap_items = PMapItems(pmap_instance)
    assert ('a', 1) in pmap_items
    assert ('b', 2) in pmap_items

    # Test with non-existing key-value pair
    assert ('d', 4) not in pmap_items
    assert ('a', 2) not in pmap_items  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in pmap_items
    assert 1 not in pmap_items
    assert [] not in pmap_items

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #87
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    pmap_obj = pmap({'a': 1, 'b': 2})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing item
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # wrong value

    # Test with invalid item type
    assert 'a' not in view  # not a tuple
    assert (1, 2, 3) not in view  # tuple with wrong length

    # Test with empty pmap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #88
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),), None]), PMap(1, [(('b', 4),), None]))
    assert m3 == PMap(2, [(('a', 4), ('b', 6)), None])

    # Test update_with with non-existent keys
    m4 = m1.update_with(lambda l, r: l * r, PMap(1, [(('c', 5),), None]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2)), None, None, (('c', 5),)])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: l, PMap(0, []))
    assert m5 == m1

    # Test update_with with dict
    m6 = m1.update_with(lambda l, r: max(l, r), {'a': 3, 'b': 1})
    assert m6 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with KeyError handling
    def custom_merge(l, r):
        if l is None:
            return r
        return l + r

    m7 = m1.update_with(custom_merge, {'c': 10})
    assert m7 == PMap(3, [(('a', 1), ('b', 2)), None, None, (('c', 10),)])

    # Test that original map is not modified
    _ = m1.update_with(lambda l, r: r, {'a': 100})
    assert m1 == PMap(2, [(('a', 1), ('b', 2)), None])


# LLM-generated content at query #89
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]))
    assert m2 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with multiple maps
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 3),), None]), PMap(1, [(('a', 4),), None]))
    assert m2 == PMap(2, [(('a', 4), ('b', 2)), None])

    # Test update_with with merge function
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),), None]))
    assert m2 == PMap(2, [(('a', 4), ('b', 2)), None])

    # Test update_with with non-existent key
    m1 = PMap(1, [(('a', 1),), None])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [(('b', 2),), None]))
    assert m2 == PMap(2, [(('a', 1),), (('b', 2),)])

    # Test update_with with empty map
    m1 = PMap(0, [])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 1),), None]))
    assert m2 == PMap(1, [(('a', 1),), None])

    # Test update_with with no maps provided
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r)
    assert m2 == m1

    # Test update_with with dict
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, {'a': 3, 'c': 4})
    assert m2 == PMap(3, [(('a', 3), ('b', 2)), (('c', 4),)])


# LLM-generated content at query #90
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    items_view = PMapItems(pmap_obj)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view
    assert ('c', 3) in items_view

    # Test with non-existing key-value pair
    assert ('d', 4) not in items_view
    assert ('a', 2) not in items_view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in items_view
    assert 1 not in items_view
    assert [1, 2] not in items_view

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items_view

    # Test with tuple that can't be unpacked
    assert (1,) not in items_view
    assert () not in items_view


# LLM-generated content at query #91
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('b', 3)]]))
    assert m3 == PMap(2, [None, [('a', 2)], [('b', 6)]])

    # Test with non-existent key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test with empty map
    m5 = PMap(0, [])
    m6 = m5.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]))
    assert m6 == PMap(1, [None, [('a', 1)]])

    # Test with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m7 == PMap(3, [None, [('a', 4)], [('b', 2)], [('c', 4)]])

    # Test with different update functions
    from operator import add, mul
    m8 = m1.update_with(add, PMap(1, [None, [('a', 2)]]))
    assert m8 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    m9 = m1.update_with(mul, PMap(1, [None, [('a', 2)]]))
    assert m9 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test with left preference
    m10 = m1.update_with(lambda l, r: l, PMap(2, [None, [('a', 10)], [('b', 20)]]))
    assert m10 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test with right preference
    m11 = m1.update_with(lambda l, r: r, PMap(2, [None, [('a', 10)], [('b', 20)]]))
    assert m11 == PMap(2, [None, [('a', 10)], [('b', 20)]])

    # Test with no overlap
    m12 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m12 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test with multiple updates to same key
    m13 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]), PMap(1, [None, [('a', 2)]]))
    assert m13 == PMap(2, [None, [('a', 4)], [('b', 2)]])


# LLM-generated content at query #92
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('a', 3)]]),
                        PMap(1, [None, [('a', 4)]]))
    assert m3 == PMap(2, [None, [('a', 8)], [('b', 2)]])

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test with different update function (keep left)
    m5 = m1.update_with(lambda l, r: l,
                        PMap(1, [None, [('a', 10)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test with dict input
    m6 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 5})
    assert m6 == PMap(3, [None, [('a', 4)], [('b', 2)], [('c', 5)]])

    # Test with empty map
    m7 = PMap(0, [])
    m8 = m7.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]))
    assert m8 == PMap(1, [None, [('a', 1)]])

    # Test with no overlapping keys
    m9 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 3)]]))
    assert m9 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test with KeyError for non-existent key in update_fn
    m10 = m1.update_with(lambda l, r: l + r,
                         PMap(1, [None, [('d', 1)]]))
    assert m10 == PMap(3, [None, [('a', 1)], [('b', 2)], [('d', 1)]])


# LLM-generated content at query #93
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing item
    assert ('c', 3) not in view
    assert ('a', 2) not in view  # Wrong value

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view
    assert None not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #94
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=3, c=4))
    assert m2 == {'a': 4, 'b': 2, 'c': 4}

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, m(a=2), m(b=3), m(c=5))
    assert m3 == {'a': 2, 'b': 6, 'c': 5}

    # Test with non-existent keys
    m4 = m1.update_with(lambda l, r: l + r, m(c=10))
    assert m4 == {'a': 1, 'b': 2, 'c': 10}

    # Test with empty map
    m5 = m()
    m6 = m5.update_with(lambda l, r: l + r, m(a=1))
    assert m6 == {'a': 1}

    # Test with no maps to update
    m7 = m1.update_with(lambda l, r: l + r)
    assert m7 == m1

    # Test with different merge function
    m8 = m1.update_with(lambda l, r: l if l > r else r, m(a=0, b=5))
    assert m8 == {'a': 1, 'b': 5}

    # Test with dict instead of PMap
    m9 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m9 == {'a': 4, 'b': 2, 'c': 4}

    # Test with mixed PMap and dict
    m10 = m1.update_with(lambda l, r: l * r, m(a=2), {'b': 3})
    assert m10 == {'a': 2, 'b': 6}


# LLM-generated content at query #95
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(0, pvector())
    m1 = m1.set('a', 1).set('b', 2)
    m2 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 2))
    assert m2 == PMap(0, pvector()).set('a', 3).set('b', 2)

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 2), PMap(0, pvector()).set('a', 3))
    assert m3 == PMap(0, pvector()).set('a', 6).set('b', 2)

    # Test update_with with non-existing key
    m4 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('c', 3))
    assert m4 == PMap(0, pvector()).set('a', 1).set('b', 2).set('c', 3)

    # Test update_with with different merge function
    m5 = m1.update_with(lambda l, r: l * r, PMap(0, pvector()).set('a', 2))
    assert m5 == PMap(0, pvector()).set('a', 2).set('b', 2)

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: l + r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 2})
    assert m7 == PMap(0, pvector()).set('a', 3).set('b', 2)

    # Test update_with with mixed types
    m8 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 2), {'b': 3})
    assert m8 == PMap(0, pvector()).set('a', 3).set('b', 5)

    # Test update_with with no overlap
    m9 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('c', 3))
    assert m9 == PMap(0, pvector()).set('a', 1).set('b', 2).set('c', 3)

    # Test update_with with all new keys
    m10 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('c', 3).set('d', 4))
    assert m10 == PMap(0, pvector()).set('a', 1).set('b', 2).set('c', 3).set('d', 4)

    # Test update_with with left preference
    m11 = m1.update_with(lambda l, r: l, PMap(0, pvector()).set('a', 2), {'a': 3})
    assert m11 == PMap(0, pvector()).set('a', 1).set('b', 2)


# LLM-generated content at query #96
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    pmap_instance = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(pmap_instance)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test with non-existing item
    assert ('c', 3) not in items_view
    assert ('a', 2) not in items_view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 'a' not in items_view
    assert 1 not in items_view
    assert [] not in items_view

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items_view


# LLM-generated content at query #97
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]))
    assert m2 == PMap(2, [None, [('a', 3)], [('b', 2)]])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test update_with with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with non-PMap mappings
    m5 = m1.update_with(lambda l, r: l + r, {'a': 2, 'c': 3})
    assert m5 == PMap(3, [None, [('a', 3)], [('b', 2)], [('c', 3)]])

    # Test update_with with left-biased merge
    m6 = m1.update_with(lambda l, r: l, PMap(2, [None, [('a', 2)], [('b', 3)]]))
    assert m6 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test update_with with empty map
    m7 = m1.update_with(lambda l, r: l + r)
    assert m7 == m1

    # Test update_with with no overlapping keys
    m8 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 3)]]))
    assert m8 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])

    # Test update_with with complex merge function
    m9 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]))
    assert m9 == PMap(2, [None, [('a', 2)], [('b', 2)]])

    # Test update_with with string concatenation
    m10 = PMap(1, [None, [('a', 'hello')]])
    m11 = m10.update_with(lambda l, r: l + r, PMap(1, [None, [('a', ' world')]]))
    assert m11 == PMap(1, [None, [('a', 'hello world')]])


# LLM-generated content at query #98
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 2),), None]))
    assert m2 == PMap(2, [(('a', 2), ('b', 2)), None])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 2),), None]), PMap(1, [(('b', 3),), None]))
    assert m3 == PMap(2, [(('a', 3), ('b', 5)), None])

    # Test update_with with non-existent key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 3),), None]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2)), None, None, (('c', 3),)])

    # Test update_with with empty map
    m5 = m1.update_with(lambda l, r: l + r)
    assert m5 == m1

    # Test update_with with different types of mappings
    m6 = m1.update_with(lambda l, r: l + r, {'a': 2, 'b': 3})
    assert m6 == PMap(2, [(('a', 3), ('b', 5)), None])

    # Test update_with with custom function
    m7 = m1.update_with(lambda l, r: l * r, PMap(1, [(('a', 2),), None]))
    assert m7 == PMap(2, [(('a', 2), ('b', 2)), None])

    # Test update_with with left preference
    m8 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 2),), None]), {'a': 3})
    assert m8 == PMap(2, [(('a', 1), ('b', 2)), None])


# LLM-generated content at query #99
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 2),), None]))
    assert m2 == PMap(2, [(('a', 2), ('b', 2)), None])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 2),), None]), PMap(1, [(('c', 3),), None]))
    assert m3 == PMap(3, [(('a', 2), ('b', 2)), None, (('c', 3),), None])

    # Test update_with with merge function
    from operator import add
    m4 = m1.update_with(add, PMap(1, [(('a', 2),), None]))
    assert m4 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with non-existent key
    m5 = m1.update_with(lambda l, r: l, PMap(1, [(('c', 3),), None]))
    assert m5 == PMap(3, [(('a', 1), ('b', 2)), None, (('c', 3),), None])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: r, {'a': 2, 'c': 3})
    assert m7 == PMap(3, [(('a', 2), ('b', 2)), None, (('c', 3),), None])

    # Test update_with with left preference
    m8 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 2),), None]), {'a': 3})
    assert m8 == PMap(2, [(('a', 1), ('b', 2)), None])


# LLM-generated content at query #100
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [(('a', 1), ('b', 2))])
    assert m1 == m2

    # Test inequality with PMap with different content
    m3 = PMap(2, [(('a', 1), ('c', 3))])
    assert m1 != m3

    # Test equality with dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with dict with different content
    d2 = {'a': 1, 'c': 3}
    assert m1 != d2

    # Test inequality with non-Mapping type
    assert m1 != "not a mapping"

    # Test inequality with different size
    m4 = PMap(3, [(('a', 1), ('b', 2), ('c', 3))])
    assert m1 != m4

    # Test with cached hash
    m5 = PMap(2, [(('a', 1), ('b', 2))])
    m5._cached_hash = hash(frozenset(m5.iteritems()))
    m6 = PMap(2, [(('a', 1), ('b', 2))])
    m6._cached_hash = hash(frozenset(m6.iteritems()))
    assert m5 == m6

    # Test with different cached hash
    m7 = PMap(2, [(('a', 1), ('b', 2))])
    m7._cached_hash = 123
    m8 = PMap(2, [(('a', 1), ('b', 2))])
    m8._cached_hash = 456
    assert m7 != m8

    # Test with same buckets reference
    m9 = PMap(2, [(('a', 1), ('b', 2))])
    m10 = PMap(2, m9._buckets)
    assert m9 == m10


# LLM-generated content at query #101
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({1: 'a', 2: 'b', 3: 'c'})
    view = PMapItems(pmap_obj)
    assert (1, 'a') in view
    assert (2, 'b') in view
    assert (3, 'c') in view

    # Test with non-existing key-value pair
    assert (4, 'd') not in view
    assert (1, 'b') not in view  # Wrong value for existing key

    # Test with non-tuple argument
    assert 1 not in view
    assert 'a' not in view
    assert [1, 'a'] not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert (1, 'a') not in empty_view

    # Test with non-hashable key (should work as long as the key is hashable in the pmap)
    pmap_obj = pmap({(1, 2): 'a'})
    view = PMapItems(pmap_obj)
    assert ((1, 2), 'a') in view
    assert ((1, 3), 'a') not in view


