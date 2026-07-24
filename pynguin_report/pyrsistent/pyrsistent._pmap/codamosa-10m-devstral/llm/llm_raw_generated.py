####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    assert [] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #2
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    m = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(m)
    assert ('a', 1) in items_view
    assert ('b', 2) in items_view

    # Test with non-existing key-value pair
    assert ('c', 3) not in items_view
    assert ('a', 2) not in items_view  # wrong value

    # Test with non-tuple argument
    assert 'a' not in items_view
    assert 1 not in items_view
    assert [('a', 1)] not in items_view

    # Test with empty PMap
    empty_map = pmap()
    empty_view = PMapItems(empty_map)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #3
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
    assert [('a', 1)] not in view

    # Test with empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #4
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),)]))
    assert m2 == PMap(2, [(('a', 4), ('b', 2))])

    # Test with multiple maps
    m3 = PMap(1, [(('a', 1),)])
    m4 = m3.update_with(lambda l, r: l * r, PMap(1, [(('a', 2),)]), PMap(1, [(('a', 3),)]))
    assert m4 == PMap(1, [(('a', 6),)])

    # Test with non-existent key
    m5 = PMap(1, [(('a', 1),)])
    m6 = m5.update_with(lambda l, r: l + r, PMap(1, [(('b', 2),)]))
    assert m6 == PMap(2, [(('a', 1), ('b', 2))])

    # Test with empty map
    m7 = PMap(0, [])
    m8 = m7.update_with(lambda l, r: l + r, PMap(1, [(('a', 1),)]))
    assert m8 == PMap(1, [(('a', 1),)])

    # Test with different merge function
    m9 = PMap(2, [(('a', 1), ('b', 2))])
    m10 = m9.update_with(lambda l, r: l, PMap(1, [(('a', 3),)]))
    assert m10 == PMap(2, [(('a', 1), ('b', 2))])

    # Test with dict
    m11 = PMap(1, [(('a', 1),)])
    m12 = m11.update_with(lambda l, r: l + r, {'a': 2})
    assert m12 == PMap(1, [(('a', 3),)])

    # Test with mixed types
    m13 = PMap(1, [(('a', 1),)])
    m14 = m13.update_with(lambda l, r: str(l) + str(r), PMap(1, [(('a', 2),)]))
    assert m14 == PMap(1, [(('a', '12'),)])


# LLM-generated content at query #5
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
    assert None not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #6
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


# LLM-generated content at query #7
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


# LLM-generated content at query #8
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(0, pvector())
    m1 = m1.set('a', 1).set('b', 2)
    m2 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 3))
    assert m2['a'] == 4
    assert m2['b'] == 2

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(0, pvector()).set('a', 2), {'b': 3})
    assert m3['a'] == 2
    assert m3['b'] == 6

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('c', 5))
    assert m4['a'] == 1
    assert m4['b'] == 2
    assert m4['c'] == 5

    # Test with empty map
    m5 = PMap(0, pvector())
    m6 = m5.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 1))
    assert m6['a'] == 1

    # Test with no overlapping keys
    m7 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('c', 10))
    assert m7['a'] == 1
    assert m7['b'] == 2
    assert m7['c'] == 10

    # Test with left preference
    m8 = m1.update_with(lambda l, r: l, PMap(0, pvector()).set('a', 100), {'a': 200})
    assert m8['a'] == 1

    # Test with right preference
    m9 = m1.update_with(lambda l, r: r, PMap(0, pvector()).set('a', 100), {'a': 200})
    assert m9['a'] == 200

    # Test with complex update function
    m10 = m1.update_with(lambda l, r: str(l) + str(r), PMap(0, pvector()).set('a', 3))
    assert m10['a'] == '13'

    # Test immutability
    m11 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 1))
    assert m1 == PMap(0, pvector()).set('a', 1).set('b', 2)
    assert m11['a'] == 2

    # Test with dict input
    m12 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m12['a'] == 4
    assert m12['b'] == 2
    assert m12['c'] == 4


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing item
    pmap_obj = pmap({'a': 1, 'b': 2, 'c': 3})
    view = PMapItems(pmap_obj)
    assert ('a', 1) in view
    assert ('b', 2) in view

    # Test with non-existing item
    assert ('d', 4) not in view
    assert ('a', 2) not in view  # Wrong value

    # Test with non-tuple argument
    assert 'a' not in view
    assert 1 not in view

    # Test with empty map
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #11
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

    # Test with empty pmap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items

    # Test with tuple of wrong length
    assert ('a', 1, 2) not in pmap_items


# LLM-generated content at query #12
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

    # Test with empty PMapItems
    empty_pmap_items = PMapItems(pmap({}))
    assert ('a', 1) not in empty_pmap_items


# LLM-generated content at query #13
#--------------------------

```python
def test_PMapItems___contains__():
    # Test with existing key-value pair
    pmap_obj = pmap({1: 'a', 2: 'b', 3: 'c'})
    view = PMapItems(pmap_obj)
    assert (1, 'a') in view
    assert (2, 'b') in view

    # Test with non-existing key-value pair
    assert (4, 'd') not in view
    assert (1, 'b') not in view

    # Test with partial tuple (should return False)
    assert (1,) not in view

    # Test with non-tuple (should return False)
    assert 1 not in view
    assert 'a' not in view

    # Test with empty pmap
    empty_view = PMapItems(pmap({}))
    assert (1, 'a') not in empty_view


# LLM-generated content at query #14
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

    # Test inequality with non-Mapping object
    assert m1 != "not a map"

    # Test inequality with dict with different content
    assert m1 != {'a': 1, 'c': 3}

    # Test equality with dict with same content
    assert m1 == {'a': 1, 'b': 2}

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


# LLM-generated content at query #15
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

    # Test with invalid item type
    assert 'a' not in view  # Not a tuple
    assert ('a',) not in view  # Tuple too short
    assert ('a', 1, 2) not in view  # Tuple too long

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #16
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
    m3 = PMap(2, [None, [('a', 1)], [('b', 3)]])
    assert m1 != m3

    # Test inequality with PMap with different size
    m4 = PMap(1, [None, [('a', 1)]])
    assert m1 != m4

    # Test equality with dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with dict with different content
    d2 = {'a': 1, 'b': 3}
    assert m1 != d2

    # Test inequality with non-Mapping type
    assert m1 != "not a mapping"

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


# LLM-generated content at query #17
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 3)]]))
    assert m2 == PMap(2, [None, [('a', 4)], [('b', 2)]])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [None, [('a', 2)]]), PMap(1, [None, [('a', 3)]]))
    assert m3 == PMap(2, [None, [('a', 6)], [('b', 2)]])

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 5)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test with empty map
    m5 = PMap(0, [])
    m6 = m5.update_with(lambda l, r: l + r, PMap(1, [None, [('a', 1)]]))
    assert m6 == PMap(1, [None, [('a', 1)]])

    # Test with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m7 == PMap(3, [None, [('a', 4)], [('b', 2)], [('c', 4)]])

    # Test with non-existent key in left map
    m8 = m1.update_with(lambda l, r: l + r, PMap(1, [None, [('c', 10)]]))
    assert m8 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 10)]])

    # Test with function that returns different type
    m9 = m1.update_with(lambda l, r: str(l) + str(r), PMap(1, [None, [('a', 3)]]))
    assert m9 == PMap(2, [None, [('a', '13')], [('b', 2)]])


# LLM-generated content at query #18
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

    # Test with invalid argument (not a tuple)
    assert 'a' not in view
    assert 1 not in view
    assert ['a', 1] not in view

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #19
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
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with a regular dict with different content
    d2 = {'a': 1, 'b': 3}
    assert m1 != d2

    # Test inequality with a non-Mapping object
    assert m1 != "not a mapping"
    assert m1 != 42
    assert m1 != None

    # Test inequality with a Mapping of different length
    m4 = pmap({'a': 1})
    assert m1 != m4

    # Test cached hash optimization
    m5 = pmap({'a': 1, 'b': 2})
    m6 = pmap({'a': 1, 'b': 2})
    assert m5._cached_hash == m6._cached_hash
    assert m5 == m6

    # Test with different cached hashes
    m7 = pmap({'a': 1, 'b': 2})
    m8 = pmap({'a': 1, 'b': 2})
    m7._cached_hash = 123
    m8._cached_hash = 456
    assert m7 != m8

    # Test with same buckets reference
    m9 = pmap({'a': 1, 'b': 2})
    m10 = pmap({'a': 1, 'b': 2})
    m9._buckets = m10._buckets
    assert m9 == m10


# LLM-generated content at query #20
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
    assert m1 == {'a': 1, 'b': 2}

    # Test inequality with a dict with different content
    assert m1 != {'a': 1, 'c': 3}

    # Test inequality with a non-Mapping type
    assert m1 != "not a mapping"

    # Test with different sized PMaps
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


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with itself
    m1 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert m1 == m2

    # Test inequality with different PMap
    m3 = PMap(2, [None, [('a', 1)], [('c', 3)]])
    assert m1 != m3

    # Test inequality with dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with dict with different content
    d2 = {'a': 1, 'c': 3}
    assert m1 != d2

    # Test inequality with non-Mapping type
    assert m1 != "not a mapping"
    assert m1 != 42
    assert m1 != [1, 2, 3]

    # Test with different sized PMaps
    m4 = PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 3)]])
    assert m1 != m4

    # Test with empty PMaps
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
    m10 = PMap(2, [None, [('a', 1), ('b', 2)]])
    assert m1 == m10


# LLM-generated content at query #2
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m = pmap({"a": 1, "b": 2})
    view = PMapItems(m)
    assert view == view

    # Test inequality with different instance
    m1 = pmap({"a": 1, "b": 2})
    m2 = pmap({"a": 1, "b": 2})
    view1 = PMapItems(m1)
    view2 = PMapItems(m2)
    assert view1 == view2

    # Test inequality with different PMap
    m1 = pmap({"a": 1, "b": 2})
    m2 = pmap({"a": 1, "b": 3})
    view1 = PMapItems(m1)
    view2 = PMapItems(m2)
    assert view1 != view2

    # Test inequality with non-PMapItems object
    m = pmap({"a": 1, "b": 2})
    view = PMapItems(m)
    assert view != "not a PMapItems"
    assert view != PMapValues(m)


# LLM-generated content at query #3
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
    assert [('a', 1)] not in view

    # Test with empty map
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


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

    # Test inequality with another PMap with different size
    m4 = pmap({'a': 1})
    assert m1 != m4

    # Test equality with a regular dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with a regular dict with different content
    d2 = {'a': 1, 'b': 3}
    assert m1 != d2

    # Test inequality with a non-Mapping object
    assert m1 != "not a map"

    # Test with empty PMaps
    m_empty1 = pmap({})
    m_empty2 = pmap({})
    assert m_empty1 == m_empty2

    # Test with PMaps containing different types
    m_mixed1 = pmap({'a': 1, 'b': 'two', 'c': [3, 4]})
    m_mixed2 = pmap({'a': 1, 'b': 'two', 'c': [3, 4]})
    assert m_mixed1 == m_mixed2

    # Test with PMaps containing nested structures
    m_nested1 = pmap({'a': pmap({'b': 2})})
    m_nested2 = pmap({'a': pmap({'b': 2})})
    assert m_nested1 == m_nested2


# LLM-generated content at query #5
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

    # Test inequality with PMap with different size
    m4 = PMap(3, [(('a', 1), ('b', 2), ('c', 3))])
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
    assert m1 != None

    # Test with cached hash
    m5 = PMap(2, [(('a', 1), ('b', 2))])
    m6 = PMap(2, [(('a', 1), ('b', 2))])
    m5._cached_hash = 123
    m6._cached_hash = 456
    assert m5 != m6

    # Test with same cached hash
    m7 = PMap(2, [(('a', 1), ('b', 2))])
    m8 = PMap(2, [(('a', 1), ('b', 2))])
    m7._cached_hash = 123
    m8._cached_hash = 123
    assert m7 == m8

    # Test with same buckets reference
    m9 = PMap(2, [(('a', 1), ('b', 2))])
    m10 = PMap(2, m9._buckets)
    assert m9 == m10


# LLM-generated content at query #6
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m = pmap({"a": 1, "b": 2})
    view = PMapItems(m)
    assert view == view

    # Test inequality with different instance
    m1 = pmap({"a": 1, "b": 2})
    m2 = pmap({"a": 1, "b": 2})
    view1 = PMapItems(m1)
    view2 = PMapItems(m2)
    assert view1 == view2

    # Test inequality with different maps
    m1 = pmap({"a": 1, "b": 2})
    m2 = pmap({"a": 1, "b": 3})
    view1 = PMapItems(m1)
    view2 = PMapItems(m2)
    assert view1 != view2

    # Test inequality with different types
    m = pmap({"a": 1, "b": 2})
    view = PMapItems(m)
    assert view != "not a PMapItems"
    assert view != PMapValues(m)


# LLM-generated content at query #7
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
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view


# LLM-generated content at query #8
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    result = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),)]))
    assert result == PMap(2, [(('a', 4), ('b', 2))])

    # Test with multiple maps
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    result = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),)]), PMap(1, [(('b', 4),)]))
    assert result == PMap(2, [(('a', 4), ('b', 6))])

    # Test with non-existent key
    m1 = PMap(1, [(('a', 1),)])
    result = m1.update_with(lambda l, r: l + r, PMap(1, [(('b', 2),)]))
    assert result == PMap(2, [(('a', 1), ('b', 2))])

    # Test with empty map
    m1 = PMap(0, [])
    result = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 1),)]))
    assert result == PMap(1, [(('a', 1),)])

    # Test with different update function
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    result = m1.update_with(lambda l, r: l * r, PMap(1, [(('a', 3),)]))
    assert result == PMap(2, [(('a', 3), ('b', 2))])

    # Test with dict input
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    result = m1.update_with(lambda l, r: l + r, {'a': 3})
    assert result == PMap(2, [(('a', 4), ('b', 2))])

    # Test with multiple dict inputs
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    result = m1.update_with(lambda l, r: l + r, {'a': 3}, {'b': 4})
    assert result == PMap(2, [(('a', 4), ('b', 6))])

    # Test immutability
    m1 = PMap(2, [(('a', 1), ('b', 2))])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),)]))
    assert m1 == PMap(2, [(('a', 1), ('b', 2))])
    assert m2 == PMap(2, [(('a', 4), ('b', 2))])


# LLM-generated content at query #9
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = pmap({'a': 1, 'b': 2})
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = pmap({'a': 1, 'b': 2})
    assert m1 == m2

    # Test inequality with different content
    m3 = pmap({'a': 1, 'b': 3})
    assert m1 != m3

    # Test inequality with different size
    m4 = pmap({'a': 1})
    assert m1 != m4

    # Test equality with regular dict
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with regular dict with different content
    d2 = {'a': 1, 'b': 3}
    assert m1 != d2

    # Test inequality with non-Mapping type
    assert m1 != "not a map"
    assert m1 != 42
    assert m1 != [1, 2, 3]

    # Test with empty maps
    m_empty1 = pmap({})
    m_empty2 = pmap({})
    assert m_empty1 == m_empty2
    assert m_empty1 == {}

    # Test with different key types
    m_mixed1 = pmap({'a': 1, 2: 'b', (3, 4): 'tuple'})
    m_mixed2 = pmap({'a': 1, 2: 'b', (3, 4): 'tuple'})
    assert m_mixed1 == m_mixed2

    # Test with nested structures
    m_nested1 = pmap({'a': pmap({'b': 1})})
    m_nested2 = pmap({'a': pmap({'b': 1})})
    assert m_nested1 == m_nested2

    # Test with different nested structures
    m_nested3 = pmap({'a': pmap({'b': 2})})
    assert m_nested1 != m_nested3


# LLM-generated content at query #10
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
    assert ['a', 1] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_items


# LLM-generated content at query #11
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
                        PMap(1, [None, [('b', 5)]]))
    assert m3 == PMap(2, [None, [('a', 4)], [('b', 7)]])

    # Test with new keys
    m4 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 10)]]))
    assert m4 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 10)]])

    # Test with different merge function (keep left value)
    m5 = m1.update_with(lambda l, r: l,
                        PMap(1, [None, [('a', 99)]]))
    assert m5 == PMap(2, [None, [('a', 1)], [('b', 2)]])

    # Test with empty map
    m6 = PMap(0, [])
    m7 = m6.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('x', 1)]]))
    assert m7 == PMap(1, [None, [('x', 1)]])

    # Test with dict input
    m8 = m1.update_with(lambda l, r: l + r, {'a': 3, 'b': 5})
    assert m8 == PMap(2, [None, [('a', 4)], [('b', 7)]])

    # Test with non-existent key in merge
    m9 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('a', 3)]]),
                        PMap(1, [None, [('d', 20)]]))
    assert m9 == PMap(3, [None, [('a', 4)], [('b', 2)], [('d', 20)]])


# LLM-generated content at query #12
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

    # Test update_with with non-existent key in left map
    m5 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('c', 5)]]))
    assert m5 == PMap(3, [None, [('a', 1)], [('b', 2)], [('c', 5)]])

    # Test update_with with empty update_fn (should keep right value)
    m6 = m1.update_with(lambda l, r: r,
                        PMap(1, [None, [('a', 10)]]))
    assert m6 == PMap(2, [None, [('a', 10)], [('b', 2)]])

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: l * r,
                        {'a': 3, 'c': 4})
    assert m7 == PMap(3, [None, [('a', 3)], [('b', 2)], [('c', 4)]])

    # Test update_with with no updates
    m8 = m1.update_with(lambda l, r: l + r)
    assert m8 == m1

    # Test update_with with KeyError (should not raise)
    m9 = m1.update_with(lambda l, r: l + r,
                        PMap(1, [None, [('d', 1)]]))
    assert m9 == PMap(3, [None, [('a', 1)], [('b', 2)], [('d', 1)]])


# LLM-generated content at query #13
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

    # Test update_with with custom function
    m7 = m1.update_with(lambda l, r: l if l > r else r, PMap(2, [None, [('a', 5)], [('b', 1)]]))
    assert m7 == PMap(2, [None, [('a', 5)], [('b', 2)]])

    # Test update_with with dict
    m8 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m8 == PMap(3, [None, [('a', 4)], [('b', 2)], [('c', 4)]])


# LLM-generated content at query #14
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(m)
    assert items_view == items_view

    # Test inequality with different PMapItems
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items_view1 = PMapItems(m1)
    items_view2 = PMapItems(m2)
    assert not (items_view1 == items_view2)

    # Test inequality with non-PMapItems
    m = pmap({'a': 1, 'b': 2})
    items_view = PMapItems(m)
    assert not (items_view == "not a PMapItems")

    # Test equality with same content
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items_view1 = PMapItems(m1)
    items_view2 = PMapItems(m2)
    assert items_view1 == items_view2


# LLM-generated content at query #15
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
    m3 = PMap(2, [None, [('a', 1)], [('b', 3)]])
    assert m1 != m3

    # Test inequality with PMap with different size
    m4 = PMap(1, [None, [('a', 1)]])
    assert m1 != m4

    # Test equality with dict with same content
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

    # Test inequality with dict with different content
    d2 = {'a': 1, 'b': 3}
    assert m1 != d2

    # Test inequality with non-Mapping type
    assert m1 != "not a mapping"
    assert m1 != 42
    assert m1 != [1, 2, 3]

    # Test with empty PMap
    empty_pmap = PMap(0, [])
    assert empty_pmap == empty_pmap
    assert empty_pmap == {}
    assert empty_pmap != m1

    # Test with PMap containing different key types
    m5 = PMap(2, [None, [('a', 1)], [(1, 2)]])
    d3 = {'a': 1, 1: 2}
    assert m5 == d3

    # Test with PMap containing None as key
    m6 = PMap(1, [None, [(None, 1)]])
    d4 = {None: 1}
    assert m6 == d4

    # Test with PMap containing tuple as key
    m7 = PMap(1, [None, [((1, 2), 3)]])
    d5 = {(1, 2): 3}
    assert m7 == d5


# LLM-generated content at query #16
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

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_view = PMapItems(empty_pmap)
    assert ('a', 1) not in empty_view


# LLM-generated content at query #17
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m = pmap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert view == view

    # Test equality with another PMapItems with same content
    m2 = pmap({'a': 1, 'b': 2})
    view2 = PMapItems(m2)
    assert view == view2

    # Test inequality with PMapItems with different content
    m3 = pmap({'a': 1, 'b': 3})
    view3 = PMapItems(m3)
    assert view != view3

    # Test inequality with non-PMapItems object
    assert view != {'a': 1, 'b': 2}
    assert view != [('a', 1), ('b', 2)]
    assert view != None


# LLM-generated content at query #18
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    pmap = pmap({"a": 1, "b": 2})
    view = PMapItems(pmap)
    assert view == view

    # Test inequality with different type
    assert not (view == "not a PMapItems")

    # Test equality with another PMapItems with same content
    pmap2 = pmap({"a": 1, "b": 2})
    view2 = PMapItems(pmap2)
    assert view == view2

    # Test inequality with another PMapItems with different content
    pmap3 = pmap({"a": 1, "b": 3})
    view3 = PMapItems(pmap3)
    assert not (view == view3)

    # Test inequality with another PMapItems with different keys
    pmap4 = pmap({"a": 1, "c": 2})
    view4 = PMapItems(pmap4)
    assert not (view == view4)


# LLM-generated content at query #19
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

    # Test inequality with non-PMapItems
    assert view1 != "not a PMapItems"
    assert view1 != {'a': 1, 'b': 2}

    # Test equality with same content but different PMapItems
    m3 = pmap({'a': 1, 'b': 2})
    view3 = PMapItems(m3)
    assert view1 != view3

    # Test inequality with different content
    m4 = pmap({'a': 1, 'b': 3})
    view4 = PMapItems(m4)
    assert view1 != view4


# LLM-generated content at query #20
#--------------------------

```python
def test_PMapItems___eq__():
    # Test equality with self
    m1 = pmap({'a': 1, 'b': 2})
    view1 = PMapItems(m1)
    assert view1 == view1

    # Test equality with another PMapItems with same content
    m2 = pmap({'a': 1, 'b': 2})
    view2 = PMapItems(m2)
    assert view1 == view2

    # Test inequality with PMapItems with different content
    m3 = pmap({'a': 1, 'b': 3})
    view3 = PMapItems(m3)
    assert not (view1 == view3)

    # Test inequality with non-PMapItems object
    assert not (view1 == "not a PMapItems")
    assert not (view1 == {'a': 1, 'b': 2})
    assert not (view1 == PMapValues(m1))


# LLM-generated content at query #21
#--------------------------

```python
def test_PMap___eq__():
    # Test equality with self
    m1 = PMap(2, [(None, [('a', 1)]), (None, [('b', 2)])])
    assert m1 == m1

    # Test equality with another PMap with same content
    m2 = PMap(2, [(None, [('a', 1)]), (None, [('b', 2)])])
    assert m1 == m2

    # Test inequality with another PMap with different content
    m3 = PMap(2, [(None, [('a', 1)]), (None, [('c', 3)])])
    assert m1 != m3

    # Test inequality with another PMap with different size
    m4 = PMap(1, [(None, [('a', 1)])])
    assert m1 != m4

    # Test equality with dict with same content
    assert m1 == {'a': 1, 'b': 2}

    # Test inequality with dict with different content
    assert m1 != {'a': 1, 'c': 3}

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

    # Test with same buckets reference
    m9 = PMap(2, [(None, [('a', 1)]), (None, [('b', 2)])])
    m10 = PMap(2, m9._buckets)
    assert m9 == m10


# LLM-generated content at query #22
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
    assert [('a', 1)] not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap())
    assert ('a', 1) not in empty_view

    # Test with different types
    mixed_map = pmap({1: 'one', 'two': 2, (3,): [3, 3]})
    mixed_view = PMapItems(mixed_map)
    assert (1, 'one') in mixed_view
    assert ('two', 2) in mixed_view
    assert ((3,), [3, 3]) in mixed_view
    assert (1, 'two') not in mixed_view


# LLM-generated content at query #23
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 2),)]))
    assert m2 == PMap(2, [(('a', 2), ('b', 2)), None])

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: r, PMap(1, [(('a', 2),)]), PMap(1, [(('c', 3),)]))
    assert m3 == PMap(3, [(('a', 2), ('b', 2)), None, (('c', 3),)])

    # Test update_with with merge function
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 2),)]))
    assert m4 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test update_with with non-existent keys
    m5 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 3),)]))
    assert m5 == PMap(3, [(('a', 1), ('b', 2)), None, (('c', 3),)])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: r, {'a': 2, 'c': 3})
    assert m7 == PMap(3, [(('a', 2), ('b', 2)), None, (('c', 3),)])

    # Test update_with with left preference
    m8 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 2),)]), PMap(1, [(('a', 3),)]))
    assert m8 == PMap(2, [(('a', 1), ('b', 2)), None])


# LLM-generated content at query #24
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
    assert [] not in pmap_items

    # Test with empty PMap
    empty_pmap = pmap({})
    empty_items = PMapItems(empty_pmap)
    assert ("a", 1) not in empty_items


# LLM-generated content at query #25
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

    # Test inequality with non-Mapping object
    assert m1 != "not a map"

    # Test equality with dict with same content
    assert m1 == {'a': 1, 'b': 2}

    # Test inequality with dict with different content
    assert m1 != {'a': 1, 'c': 3}

    # Test inequality with dict with different size
    assert m1 != {'a': 1}

    # Test equality with another Mapping type with same content
    from collections import OrderedDict
    assert m1 == OrderedDict([('a', 1), ('b', 2)])

    # Test inequality with another Mapping type with different content
    assert m1 != OrderedDict([('a', 1), ('c', 3)])

    # Test cached hash comparison
    m4 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m4._cached_hash = 123
    m5 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m5._cached_hash = 456
    assert m4 != m5

    # Test same buckets reference
    m6 = PMap(2, [None, [('a', 1)], [('b', 2)]])
    m7 = PMap(2, m6._buckets)
    assert m6 == m7


# LLM-generated content at query #26
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
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),), None]))
    assert m4 == PMap(2, [(('a', 4), ('b', 2)), None])

    # Test update_with with non-existing key
    m5 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 3),), None]))
    assert m5 == PMap(3, [(('a', 1), ('b', 2)), None, (('c', 3),), None])

    # Test update_with with empty map
    m6 = m1.update_with(lambda l, r: r)
    assert m6 == m1

    # Test update_with with dict
    m7 = m1.update_with(lambda l, r: r, {'a': 5, 'd': 6})
    assert m7 == PMap(3, [(('a', 5), ('b', 2)), None, (('d', 6),), None])

    # Test update_with with left preference
    m8 = m1.update_with(lambda l, r: l, PMap(1, [(('a', 3),), None]), {'a': 4})
    assert m8 == PMap(2, [(('a', 1), ('b', 2)), None])


# LLM-generated content at query #27
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(0, pvector())
    m1 = m1.set('a', 1).set('b', 2)
    m2 = m1.update_with(lambda l, r: r, PMap(0, pvector()).set('a', 2))
    assert m2 == PMap(0, pvector()).set('a', 2).set('b', 2)

    # Test update_with with multiple maps
    m3 = m1.update_with(lambda l, r: l + r, PMap(0, pvector()).set('a', 2), {'b': 3})
    assert m3 == PMap(0, pvector()).set('a', 3).set('b', 5)

    # Test update_with with non-existent keys
    m4 = m1.update_with(lambda l, r: l * r, PMap(0, pvector()).set('c', 3))
    assert m4 == PMap(0, pvector()).set('a', 1).set('b', 2).set('c', 3)

    # Test update_with with empty maps
    m5 = m1.update_with(lambda l, r: l, PMap(0, pvector()))
    assert m5 == m1

    # Test update_with with dict
    m6 = m1.update_with(lambda l, r: l if l > r else r, {'a': 3, 'b': 1})
    assert m6 == PMap(0, pvector()).set('a', 3).set('b', 2)

    # Test update_with with custom function
    m7 = m1.update_with(lambda l, r: l + r if isinstance(l, int) and isinstance(r, int) else r, {'a': 'x'})
    assert m7 == PMap(0, pvector()).set('a', 'x').set('b', 2)

    # Test update_with with no maps provided
    m8 = m1.update_with(lambda l, r: l)
    assert m8 == m1


# LLM-generated content at query #28
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
    assert [('a', 1)] not in view

    # Test with empty PMap
    empty_view = PMapItems(pmap({}))
    assert ('a', 1) not in empty_view


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_PMap_update_with():
    # Test basic update_with functionality
    m1 = PMap(2, [(('a', 1), ('b', 2)), None])
    m2 = m1.update_with(lambda l, r: l + r, PMap(1, [(('a', 3),), None]))
    assert m2 == PMap(2, [(('a', 4), ('b', 2)), None])

    # Test with multiple maps
    m3 = m1.update_with(lambda l, r: l * r, PMap(1, [(('a', 2),), None]), PMap(1, [(('b', 3),), None]))
    assert m3 == PMap(2, [(('a', 2), ('b', 6)), None])

    # Test with non-existent key
    m4 = m1.update_with(lambda l, r: l + r, PMap(1, [(('c', 5),), None]))
    assert m4 == PMap(3, [(('a', 1), ('b', 2)), (('c', 5),), None])

    # Test with empty map
    m5 = m1.update_with(lambda l, r: l + r)
    assert m5 == m1

    # Test with different merge function
    m6 = m1.update_with(lambda l, r: l if l > r else r, PMap(1, [(('a', 3),), None]))
    assert m6 == PMap(2, [(('a', 3), ('b', 2)), None])

    # Test with dict
    m7 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m7 == PMap(3, [(('a', 4), ('b', 2)), (('c', 4),), None])

    # Test with KeyError in merge function (should not happen in normal usage)
    def bad_merge(l, r):
        raise KeyError("bad merge")
    m8 = m1.update_with(bad_merge, PMap(1, [(('a', 3),), None]))
    with pytest.raises(KeyError):
        m8 == PMap(2, [(('a', 4), ('b', 2)), None])


# LLM-generated content at query #31
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


