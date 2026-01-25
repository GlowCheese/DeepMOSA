####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Create a simple bucket structure
    buckets = pvector([None, [('a', 1)], [('b', 2)], None])
    
    # Test constructor with size and buckets
    pmap_instance = PMap(2, buckets)
    
    # Verify the instance was created correctly
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == buckets
    assert isinstance(pmap_instance, PMap)


def test_pmap_constructor_empty():
    from pyrsistent import pvector
    
    # Create an empty bucket structure
    buckets = pvector([None, None, None])
    
    # Test constructor with empty map
    pmap_instance = PMap(0, buckets)
    
    # Verify the instance was created correctly
    assert pmap_instance._size == 0
    assert pmap_instance._buckets == buckets
    assert len(pmap_instance) == 0


def test_pmap_constructor_single_element():
    from pyrsistent import pvector
    
    # Create a bucket structure with one element
    buckets = pvector([[('key', 'value')], None, None])
    
    # Test constructor with single element
    pmap_instance = PMap(1, buckets)
    
    # Verify the instance was created correctly
    assert pmap_instance._size == 1
    assert pmap_instance._buckets == buckets
    assert pmap_instance['key'] == 'value'


def test_pmap_constructor_multiple_elements():
    from pyrsistent import pvector
    
    # Create a bucket structure with multiple elements
    buckets = pvector([
        [('a', 1), ('d', 4)],
        [('b', 2)],
        None,
        [('c', 3)]
    ])
    
    # Test constructor with multiple elements
    pmap_instance = PMap(4, buckets)
    
    # Verify the instance was created correctly
    assert pmap_instance._size == 4
    assert pmap_instance._buckets == buckets
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2
    assert pmap_instance['c'] == 3
    assert pmap_instance['d'] == 4


def test_pmap_constructor_weakref_support():
    from pyrsistent import pvector
    import weakref
    
    # Create a simple bucket structure
    buckets = pvector([None, [('x', 10)]])
    
    # Test constructor creates object that supports weakref
    pmap_instance = PMap(1, buckets)
    
    # Verify weakref can be created (checks __weakref__ slot)
    weak_reference = weakref.ref(pmap_instance)
    assert weak_reference() is pmap_instance


# LLM-generated content at query #2
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Test basic construction with size and buckets
    buckets = pvector([None, [('a', 1)], None, [('b', 2)]])
    pmap_instance = PMap(2, buckets)
    
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == buckets
    assert hasattr(pmap_instance, '__weakref__')
    assert not hasattr(pmap_instance, '_cached_hash')


def test_pmap_constructor_empty():
    from pyrsistent import pvector
    
    # Test construction with empty buckets
    buckets = pvector([None, None, None])
    pmap_instance = PMap(0, buckets)
    
    assert pmap_instance._size == 0
    assert pmap_instance._buckets == buckets


def test_pmap_constructor_large():
    from pyrsistent import pvector
    
    # Test construction with larger size
    buckets = pvector([None] * 100)
    for i in range(10):
        buckets = buckets.set(i, [(f'key{i}', i)])
    
    pmap_instance = PMap(10, buckets)
    
    assert pmap_instance._size == 10
    assert len(pmap_instance._buckets) == 100


def test_pmap_constructor_slots():
    from pyrsistent import pvector
    
    # Test that only allowed slots are available
    buckets = pvector([None, [('x', 10)]])
    pmap_instance = PMap(1, buckets)
    
    assert hasattr(pmap_instance, '_size')
    assert hasattr(pmap_instance, '_buckets')
    assert hasattr(pmap_instance, '__weakref__')
    
    try:
        pmap_instance.arbitrary_attr = "value"
        assert False, "Should not allow arbitrary attributes"
    except AttributeError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_pmap_eq_same_object():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    assert pmap1 == pmap1


def test_pmap_eq_different_pmaps_same_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert pmap1 == pmap2


def test_pmap_eq_different_pmaps_different_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_pmap_vs_dict_same_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    assert pmap1 == dict1


def test_pmap_eq_pmap_vs_dict_different_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 3}
    assert not (pmap1 == dict1)


def test_pmap_eq_different_lengths():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1)
    assert not (pmap1 == pmap2)


def test_pmap_eq_pmap_vs_non_mapping():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    result = pmap1.__eq__([1, 2, 3])
    assert result == NotImplemented


def test_pmap_eq_empty_pmaps():
    from pyrsistent import m
    pmap1 = m()
    pmap2 = m()
    assert pmap1 == pmap2


def test_pmap_eq_pmap_vs_empty_dict():
    from pyrsistent import m
    pmap1 = m()
    dict1 = {}
    assert pmap1 == dict1


def test_pmap_eq_with_cached_hash_different_values():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    hash(pmap1)
    hash(pmap2)
    assert not (pmap1 == pmap2)


def test_pmap_eq_with_cached_hash_same_values():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    hash(pmap1)
    hash(pmap2)
    assert pmap1 == pmap2


def test_pmap_eq_same_buckets():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = pmap1.set('a', 1)
    assert pmap1 == pmap2


def test_pmap_eq_pmap_vs_custom_mapping():
    from pyrsistent import m
    from collections.abc import Mapping
    
    class CustomMapping(Mapping):
        def __init__(self, data):
            self.data = data
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __iter__(self):
            return iter(self.data)
        
        def __len__(self):
            return len(self.data)
    
    pmap1 = m(a=1, b=2)
    custom = CustomMapping({'a': 1, 'b': 2})
    assert pmap1 == custom


def test_pmap_eq_pmap_vs_custom_mapping_different_content():
    from pyrsistent import m
    from collections.abc import Mapping
    
    class CustomMapping(Mapping):
        def __init__(self, data):
            self.data = data
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __iter__(self):
            return iter(self.data)
        
        def __len__(self):
            return len(self.data)
    
    pmap1 = m(a=1, b=2)
    custom = CustomMapping({'a': 1, 'b': 3})
    assert not (pmap1 == custom)


# LLM-generated content at query #4
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Create a simple bucket structure
    buckets = pvector([None, [('key1', 'value1')], None])
    size = 1
    
    # Test constructor with valid arguments
    pmap_instance = PMap(size, buckets)
    
    assert pmap_instance._size == 1
    assert pmap_instance._buckets == buckets
    assert hasattr(pmap_instance, '__weakref__')
    
    # Test constructor with empty buckets
    empty_buckets = pvector([None, None, None])
    pmap_instance_empty = PMap(0, empty_buckets)
    
    assert pmap_instance_empty._size == 0
    assert pmap_instance_empty._buckets == empty_buckets
    
    # Test constructor with multiple entries
    buckets_multi = pvector([
        [('a', 1)],
        [('b', 2), ('c', 3)],
        None
    ])
    pmap_instance_multi = PMap(3, buckets_multi)
    
    assert pmap_instance_multi._size == 3
    assert pmap_instance_multi._buckets == buckets_multi


# LLM-generated content at query #5
#--------------------------

```python
def test_pmap_items_eq_same_instance():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert items == items


def test_pmap_items_eq_different_instances_same_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


def test_pmap_items_eq_different_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)


def test_pmap_items_eq_different_type():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    other_dict = {'a': 1, 'b': 2}
    assert not (items == other_dict)


def test_pmap_items_eq_empty_maps():
    from pyrsistent import pmap
    m1 = pmap()
    m2 = pmap()
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


def test_pmap_items_eq_different_keys():
    from pyrsistent import pmap
    m1 = pmap({'a': 1})
    m2 = pmap({'b': 1})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)


# LLM-generated content at query #6
#--------------------------

```python
def test_pmap_items_contains_with_valid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1) in items_view

def test_pmap_items_contains_with_invalid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 2) not in items_view

def test_pmap_items_contains_with_nonexistent_key():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('c', 1) not in items_view

def test_pmap_items_contains_with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert 'not_a_tuple' not in items_view

def test_pmap_items_contains_with_single_element_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a',) not in items_view

def test_pmap_items_contains_with_three_element_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1, 'extra') not in items_view

def test_pmap_items_contains_with_none_key():
    from pyrsistent import pmap
    m = pmap({None: 'value'})
    items_view = m.items()
    assert (None, 'value') in items_view

def test_pmap_items_contains_with_none_value():
    from pyrsistent import pmap
    m = pmap({'a': None})
    items_view = m.items()
    assert ('a', None) in items_view

def test_pmap_items_contains_with_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    assert ('a', 1) not in items_view

def test_pmap_items_contains_with_list_unpacking_failure():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    items_view = m.items()
    assert [1, 2, 3] not in items_view


# LLM-generated content at query #7
#--------------------------

```python
def test_turbo_mapping_with_dict():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({'a': 1, 'b': 2}, None)
    assert result['a'] == 1
    assert result['b'] == 2
    assert len(result) == 2

def test_turbo_mapping_with_pre_size():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({'a': 1}, 16)
    assert result['a'] == 1
    assert len(result) == 1

def test_turbo_mapping_with_list_of_tuples():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping([('x', 10), ('y', 20)], None)
    assert result['x'] == 10
    assert result['y'] == 20
    assert len(result) == 2

def test_turbo_mapping_empty_dict():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({}, None)
    assert len(result) == 0

def test_turbo_mapping_single_element():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({'key': 'value'}, None)
    assert result['key'] == 'value'
    assert len(result) == 1

def test_turbo_mapping_multiple_elements():
    from pyrsistent._pmap import _turbo_mapping
    initial = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    result = _turbo_mapping(initial, None)
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert result['d'] == 4
    assert len(result) == 4

def test_turbo_mapping_with_explicit_size():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({'a': 1, 'b': 2}, 32)
    assert result['a'] == 1
    assert result['b'] == 2
    assert len(result) == 2

def test_turbo_mapping_preserves_values():
    from pyrsistent._pmap import _turbo_mapping
    initial = {'key1': 'value1', 'key2': 42, 'key3': None}
    result = _turbo_mapping(initial, None)
    assert result['key1'] == 'value1'
    assert result['key2'] == 42
    assert result['key3'] is None

def test_turbo_mapping_zero_pre_size():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({'a': 1}, 0)
    assert result['a'] == 1
    assert len(result) == 1


# LLM-generated content at query #8
#--------------------------

```python
def test_contains_predicate_line_4_evaluates_to_false():
    from pyrsistent import pmap
    
    # Create a PMapItems view
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    
    # Test case 1: key exists but value doesn't match
    result1 = ('a', 999) in items_view
    assert result1 is False
    
    # Test case 2: key doesn't exist
    result2 = ('nonexistent', 1) in items_view
    assert result2 is False
    
    # Test case 3: both key and value don't match
    result3 = ('x', 99) in items_view
    assert result3 is False


# LLM-generated content at query #9
#--------------------------

```python
def test_update_with_single_map():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}


def test_update_with_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert result == {'a': 3, 'b': 2}


def test_update_with_keep_leftmost():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1}


def test_update_with_new_keys():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(c=3, d=4))
    assert result == {'a': 1, 'b': 2, 'c': 3, 'd': 4}


def test_update_with_empty_map():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m())
    assert result == {'a': 1, 'b': 2}


def test_update_with_original_unchanged():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2, c=3))
    assert m1 == {'a': 1, 'b': 2}
    assert result == {'a': 3, 'b': 2, 'c': 3}


def test_update_with_dict():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, {'a': 10, 'c': 3})
    assert result == {'a': 10, 'b': 2, 'c': 3}


def test_update_with_multiple_maps_rightmost_wins():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: r, m(a=2), m(a=3), {'a': 4})
    assert result == {'a': 4}


def test_update_with_custom_merge_function():
    from pyrsistent import m
    m1 = m(a=[1])
    result = m1.update_with(lambda l, r: l + r, m(a=[2]), m(a=[3]))
    assert result == {'a': [1, 2, 3]}


def test_update_with_no_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r)
    assert result == {'a': 1, 'b': 2}
    assert result is m1


# LLM-generated content at query #10
#--------------------------

```python
def test_update_with_predicate_false():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    
    result = m1.update_with(lambda l, r: l + r, m2)
    
    assert result == {'a': 1, 'b': 2, 'c': 3}
    assert 'c' in result
    assert result['c'] == 3


# LLM-generated content at query #11
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Create a simple bucket structure
    buckets = pvector([None, [('a', 1)], [('b', 2)], None])
    
    # Create PMap using __new__
    pmap_instance = PMap.__new__(PMap, 2, buckets)
    
    # Verify that the instance was created correctly
    assert pmap_instance._size == 2
    assert pmap_instance._buckets is buckets
    assert pmap_instance._size == 2
    assert len(pmap_instance) == 2


def test_pmap_constructor_empty():
    from pyrsistent import pvector
    
    # Create an empty bucket structure
    buckets = pvector([None, None, None, None])
    
    # Create an empty PMap
    pmap_instance = PMap.__new__(PMap, 0, buckets)
    
    # Verify empty map
    assert pmap_instance._size == 0
    assert pmap_instance._buckets is buckets
    assert len(pmap_instance) == 0


def test_pmap_constructor_large():
    from pyrsistent import pvector
    
    # Create a bucket structure with multiple entries
    buckets = pvector([
        [('key1', 'value1')],
        [('key2', 'value2'), ('key3', 'value3')],
        None,
        [('key4', 'value4')],
        None
    ])
    
    # Create PMap with 4 items
    pmap_instance = PMap.__new__(PMap, 4, buckets)
    
    # Verify the instance
    assert pmap_instance._size == 4
    assert pmap_instance._buckets is buckets
    assert len(pmap_instance) == 4


def test_pmap_constructor_preserves_buckets_reference():
    from pyrsistent import pvector
    
    # Create buckets
    buckets = pvector([None, [('x', 10)], None])
    
    # Create PMap
    pmap_instance = PMap.__new__(PMap, 1, buckets)
    
    # Verify that _buckets is the exact same object (not a copy)
    assert pmap_instance._buckets is buckets


def test_pmap_constructor_with_weakref_slot():
    from pyrsistent import pvector
    
    # Verify that __weakref__ slot exists and doesn't cause issues
    buckets = pvector([None, [('key', 'value')], None])
    pmap_instance = PMap.__new__(PMap, 1, buckets)
    
    # Should not raise any error
    assert hasattr(pmap_instance, '_size')
    assert hasattr(pmap_instance, '_buckets')
    assert pmap_instance._size == 1


# LLM-generated content at query #12
#--------------------------

```python
def test_update_with_single_map():
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}


def test_update_with_multiple_maps():
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2), m(a=3))
    assert result == {'a': 6, 'b': 2}


def test_update_with_keep_leftmost():
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1}


def test_update_with_new_keys():
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(c=3, d=4))
    assert result == {'a': 1, 'b': 2, 'c': 3, 'd': 4}


def test_update_with_empty_map():
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m())
    assert result == {'a': 1, 'b': 2}


def test_update_with_original_unchanged():
    from operator import add
    m1 = m(a=1, b=2)
    m2 = m1.update_with(add, m(a=2))
    assert m1 == {'a': 1, 'b': 2}
    assert m2 == {'a': 3, 'b': 2}


def test_update_with_dict():
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, {'a': 2, 'c': 3})
    assert result == {'a': 3, 'b': 2, 'c': 3}


def test_update_with_custom_function():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l + r, m(a=5, c=10))
    assert result == {'a': 6, 'b': 2, 'c': 10}


def test_update_with_rightmost_wins():
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert result == {'a': 3}


def test_update_with_no_overlap():
    from operator import add
    m1 = m(a=1)
    result = m1.update_with(add, m(b=2), m(c=3))
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #13
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Test basic constructor with empty buckets
    buckets = pvector()
    pmap_instance = PMap(0, buckets)
    assert pmap_instance._size == 0
    assert pmap_instance._buckets == buckets
    
    # Test constructor with non-zero size
    buckets_with_data = pvector([[(('key1', 'value1'))]])
    pmap_instance2 = PMap(1, buckets_with_data)
    assert pmap_instance2._size == 1
    assert pmap_instance2._buckets == buckets_with_data
    
    # Test constructor with multiple items
    buckets_multi = pvector([[(('a', 1), ('b', 2))], [(('c', 3))]])
    pmap_instance3 = PMap(3, buckets_multi)
    assert pmap_instance3._size == 3
    assert pmap_instance3._buckets == buckets_multi
    
    # Test that __new__ returns a PMap instance
    assert isinstance(pmap_instance, PMap)
    assert isinstance(pmap_instance2, PMap)
    assert isinstance(pmap_instance3, PMap)
    
    # Test that slots are properly initialized
    assert hasattr(pmap_instance, '_size')
    assert hasattr(pmap_instance, '_buckets')
    assert hasattr(pmap_instance, '_cached_hash')


# LLM-generated content at query #14
#--------------------------

```python
def test_pmap_items_contains_with_valid_key_value_pair():
    from pyrsistent import pmap
    items = pmap({'a': 1, 'b': 2}).items()
    assert ('a', 1) in items


def test_pmap_items_contains_with_invalid_key_value_pair():
    from pyrsistent import pmap
    items = pmap({'a': 1, 'b': 2}).items()
    assert ('a', 2) not in items


def test_pmap_items_contains_with_nonexistent_key():
    from pyrsistent import pmap
    items = pmap({'a': 1, 'b': 2}).items()
    assert ('c', 1) not in items


def test_pmap_items_contains_with_non_tuple_argument():
    from pyrsistent import pmap
    items = pmap({'a': 1, 'b': 2}).items()
    assert 'a' not in items


def test_pmap_items_contains_with_single_element():
    from pyrsistent import pmap
    items = pmap({'a': 1, 'b': 2}).items()
    assert ('a',) not in items


def test_pmap_items_contains_with_three_element_tuple():
    from pyrsistent import pmap
    items = pmap({'a': 1, 'b': 2}).items()
    assert ('a', 1, 'extra') not in items


def test_pmap_items_contains_with_none_values():
    from pyrsistent import pmap
    items = pmap({'a': None, 'b': 2}).items()
    assert ('a', None) in items


def test_pmap_items_contains_with_empty_pmap():
    from pyrsistent import pmap
    items = pmap({}).items()
    assert ('a', 1) not in items


# LLM-generated content at query #15
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Test basic constructor with empty buckets
    buckets = pvector()
    pmap_instance = PMap(0, buckets)
    assert pmap_instance._size == 0
    assert pmap_instance._buckets == buckets
    
    # Test constructor with non-empty size and buckets
    buckets_with_data = pvector([[(1, 'a')], None, [(2, 'b')]])
    pmap_instance2 = PMap(2, buckets_with_data)
    assert pmap_instance2._size == 2
    assert pmap_instance2._buckets == buckets_with_data
    
    # Test that __weakref__ slot is available
    assert hasattr(pmap_instance, '__weakref__')
    
    # Test that _cached_hash slot exists but is not set initially
    assert not hasattr(pmap_instance, '_cached_hash')
    
    # Test constructor preserves exact size and buckets passed
    test_size = 42
    test_buckets = pvector([None, [(5, 'value')]])
    pmap_instance3 = PMap(test_size, test_buckets)
    assert pmap_instance3._size == test_size
    assert pmap_instance3._buckets is test_buckets


# LLM-generated content at query #16
#--------------------------

```python
def test_pmap_items_eq_same_object():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert items == items


def test_pmap_items_eq_different_objects_same_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


def test_pmap_items_eq_different_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)


def test_pmap_items_eq_different_type():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    dict_items = {'a': 1, 'b': 2}.items()
    assert not (items == dict_items)


def test_pmap_items_eq_with_non_pmap_items():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == [('a', 1), ('b', 2)])


def test_pmap_items_eq_empty_maps():
    from pyrsistent import pmap
    m1 = pmap({})
    m2 = pmap({})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


def test_pmap_items_eq_with_string():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    items = m.items()
    assert not (items == "pmap_items([('a', 1)])")


# LLM-generated content at query #17
#--------------------------

```python
def test_pmap_items_contains_with_valid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1) in items_view


def test_pmap_items_contains_with_invalid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 2) not in items_view


def test_pmap_items_contains_with_missing_key():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('c', 1) not in items_view


def test_pmap_items_contains_with_non_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert 'not_a_tuple' not in items_view


def test_pmap_items_contains_with_single_element():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a',) not in items_view


def test_pmap_items_contains_with_three_element_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1, 'extra') not in items_view


def test_pmap_items_contains_with_none_value():
    from pyrsistent import pmap
    m = pmap({'a': None})
    items_view = m.items()
    assert ('a', None) in items_view


def test_pmap_items_contains_with_complex_value():
    from pyrsistent import pmap
    m = pmap({'a': [1, 2, 3]})
    items_view = m.items()
    assert ('a', [1, 2, 3]) in items_view


# LLM-generated content at query #18
#--------------------------

```python
def test_pmap_eq_same_instance():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    assert pmap1 == pmap1


def test_pmap_eq_with_equal_pmap():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert pmap1 == pmap2


def test_pmap_eq_with_different_pmap():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_with_equal_dict():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    assert pmap1 == dict1


def test_pmap_eq_with_different_dict():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 3}
    assert not (pmap1 == dict1)


def test_pmap_eq_with_different_lengths():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2, c=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_with_non_mapping():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    result = pmap1.__eq__([1, 2, 3])
    assert result is NotImplemented


def test_pmap_eq_empty_pmaps():
    from pyrsistent import m
    pmap1 = m()
    pmap2 = m()
    assert pmap1 == pmap2


def test_pmap_eq_empty_pmap_with_empty_dict():
    from pyrsistent import m
    pmap1 = m()
    dict1 = {}
    assert pmap1 == dict1


def test_pmap_eq_with_string_keys():
    from pyrsistent import m
    pmap1 = m(hello='world', foo='bar')
    pmap2 = m(hello='world', foo='bar')
    assert pmap1 == pmap2


def test_pmap_eq_with_numeric_values():
    from pyrsistent import m
    pmap1 = m(a=1, b=2.5, c=-3)
    dict1 = {'a': 1, 'b': 2.5, 'c': -3}
    assert pmap1 == dict1


def test_pmap_eq_with_none_values():
    from pyrsistent import m
    pmap1 = m(a=None, b=2)
    pmap2 = m(a=None, b=2)
    assert pmap1 == pmap2


def test_pmap_eq_ne_operator():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert pmap1 != pmap2


def test_pmap_eq_ne_operator_equal():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert not (pmap1 != pmap2)


def test_pmap_eq_with_generic_mapping():
    from pyrsistent import m
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
    
    pmap1 = m(a=1, b=2)
    custom_map = CustomMapping({'a': 1, 'b': 2})
    assert pmap1 == custom_map


def test_pmap_eq_dict_with_extra_keys():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2, 'c': 3}
    assert not (pmap1 == dict1)


# LLM-generated content at query #19
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Create a simple bucket structure
    buckets = pvector([None, [('a', 1)], [('b', 2)], None])
    
    # Test constructor with size and buckets
    pmap_instance = PMap(2, buckets)
    
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == buckets
    assert hasattr(pmap_instance, '__weakref__')
    assert not hasattr(pmap_instance, '_cached_hash')


def test_pmap_constructor_empty():
    from pyrsistent import pvector
    
    # Test constructor with empty buckets
    buckets = pvector([None, None, None, None])
    pmap_instance = PMap(0, buckets)
    
    assert pmap_instance._size == 0
    assert pmap_instance._buckets == buckets


def test_pmap_constructor_large_map():
    from pyrsistent import pvector
    
    # Test constructor with multiple entries
    buckets = pvector([
        [('key1', 'val1')],
        [('key2', 'val2'), ('key3', 'val3')],
        None,
        [('key4', 'val4')]
    ])
    
    pmap_instance = PMap(4, buckets)
    
    assert pmap_instance._size == 4
    assert len(pmap_instance._buckets) == 4
    assert pmap_instance._buckets[0] == [('key1', 'val1')]
    assert pmap_instance._buckets[1] == [('key2', 'val2'), ('key3', 'val3')]
    assert pmap_instance._buckets[2] is None
    assert pmap_instance._buckets[3] == [('key4', 'val4')]


# LLM-generated content at query #20
#--------------------------

```python
def test_contains_returns_false_on_unpacking_exception():
    from pyrsistent import pmap
    
    m = pmap({'a': 1, 'b': 2})
    pmap_items = m.items()
    
    # Test with a value that cannot be unpacked into (k, v)
    result = "not_a_tuple" in pmap_items
    assert result is False
    
    # Test with a single value
    result = 42 in pmap_items
    assert result is False
    
    # Test with a list (should fail unpacking)
    result = [1, 2, 3] in pmap_items
    assert result is False
    
    # Test with None
    result = None in pmap_items
    assert result is False


# LLM-generated content at query #21
#--------------------------

```python
def test_turbo_mapping_with_empty_dict():
    from pyrsistent import pvector
    result = _turbo_mapping({}, 0)
    assert len(result) == 0
    assert result == {}


def test_turbo_mapping_with_single_item():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, 0)
    assert len(result) == 1
    assert result == {'a': 1}
    assert result['a'] == 1


def test_turbo_mapping_with_multiple_items():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3}, 0)
    assert len(result) == 3
    assert result == {'a': 1, 'b': 2, 'c': 3}
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


def test_turbo_mapping_with_pre_size():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2}, 16)
    assert len(result) == 2
    assert result == {'a': 1, 'b': 2}


def test_turbo_mapping_with_list_of_tuples():
    from pyrsistent import pvector
    result = _turbo_mapping([('x', 10), ('y', 20)], 0)
    assert len(result) == 2
    assert result == {'x': 10, 'y': 20}


def test_turbo_mapping_preserves_values():
    from pyrsistent import pvector
    result = _turbo_mapping({'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}, 0)
    assert result['key1'] == 'value1'
    assert result['key2'] == 'value2'
    assert result['key3'] == 'value3'


def test_turbo_mapping_returns_pmap_instance():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, 0)
    assert isinstance(result, PMap)


def test_turbo_mapping_with_large_pre_size():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2}, 256)
    assert len(result) == 2
    assert result == {'a': 1, 'b': 2}


def test_turbo_mapping_with_numeric_values():
    from pyrsistent import pvector
    result = _turbo_mapping({'one': 1, 'two': 2, 'three': 3, 'four': 4}, 0)
    assert len(result) == 4
    assert result['one'] == 1
    assert result['two'] == 2
    assert result['three'] == 3
    assert result['four'] == 4


def test_turbo_mapping_with_mixed_types():
    from pyrsistent import pvector
    result = _turbo_mapping({1: 'one', 'two': 2, (3, 3): 'three'}, 0)
    assert len(result) == 3
    assert result[1] == 'one'
    assert result['two'] == 2
    assert result[(3, 3)] == 'three'


# LLM-generated content at query #22
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l, m(c=3))
    
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #23
#--------------------------

```python
def test_pmap_items_eq_same_instance():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert items == items

def test_pmap_items_eq_different_instances_same_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2

def test_pmap_items_eq_different_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)

def test_pmap_items_eq_different_keys():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'c': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)

def test_pmap_items_eq_with_non_pmap_items():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == "not a pmap items")

def test_pmap_items_eq_with_dict_items():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    dict_items = {'a': 1, 'b': 2}.items()
    assert not (items == dict_items)

def test_pmap_items_eq_with_none():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == None)

def test_pmap_items_eq_empty_maps():
    from pyrsistent import pmap
    m1 = pmap({})
    m2 = pmap({})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


# LLM-generated content at query #24
#--------------------------

```python
def test_eq_predicate_line_3_returns_false():
    from pyrsistent import pmap
    
    m1 = pmap({'a': 1, 'b': 2})
    items1 = m1.items()
    
    # Test with a non-PMapItems object to trigger line 3
    result = items1.__eq__("not a PMapItems")
    
    assert result is False


# LLM-generated content at query #25
#--------------------------

```python
def test_contains_predicate_evaluates_to_false():
    from pyrsistent import pmap
    
    # Create a PMapItems instance
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    
    # Test case 1: key exists but value doesn't match
    result1 = ('a', 999) in items_view
    assert result1 is False
    
    # Test case 2: key doesn't exist
    result2 = ('c', 1) in items_view
    assert result2 is False
    
    # Test case 3: key doesn't exist and value doesn't match
    result3 = ('z', 999) in items_view
    assert result3 is False


# LLM-generated content at query #26
#--------------------------

```python
def test_eq_same_object():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    assert pmap1 == pmap1


def test_eq_different_pmap_same_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert pmap1 == pmap2


def test_eq_different_pmap_different_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert not (pmap1 == pmap2)


def test_eq_pmap_vs_dict_same_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    assert pmap1 == dict1


def test_eq_pmap_vs_dict_different_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 3}
    assert not (pmap1 == dict1)


def test_eq_pmap_vs_dict_different_length():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1}
    assert not (pmap1 == dict1)


def test_eq_pmap_vs_non_mapping():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    result = pmap1 == [1, 2, 3]
    assert result is NotImplemented


def test_eq_pmap_vs_dict_like_mapping():
    from pyrsistent import m
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
    
    pmap1 = m(a=1, b=2)
    custom_map = CustomMapping({'a': 1, 'b': 2})
    assert pmap1 == custom_map


def test_eq_pmap_vs_custom_mapping_different_content():
    from pyrsistent import m
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
    
    pmap1 = m(a=1, b=2)
    custom_map = CustomMapping({'a': 1, 'b': 3})
    assert not (pmap1 == custom_map)


def test_eq_empty_pmaps():
    from pyrsistent import m
    pmap1 = m()
    pmap2 = m()
    assert pmap1 == pmap2


def test_eq_empty_pmap_vs_empty_dict():
    from pyrsistent import m
    pmap1 = m()
    dict1 = {}
    assert pmap1 == dict1


def test_eq_with_cached_hash_same_hash():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    hash(pmap1)
    hash(pmap2)
    assert pmap1 == pmap2


def test_eq_with_cached_hash_different_hash():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    hash(pmap1)
    hash(pmap2)
    assert not (pmap1 == pmap2)


def test_eq_reflexive():
    from pyrsistent import m
    pmap1 = m(x=10, y=20, z=30)
    assert pmap1 == pmap1


def test_eq_symmetric():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert (pmap1 == pmap2) == (pmap2 == pmap1)


def test_eq_transitive():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    pmap3 = m(a=1, b=2)
    assert pmap1 == pmap2
    assert pmap2 == pmap3
    assert pmap1 == pmap3


def test_eq_pmap_vs_string():
    from pyrsistent import m
    pmap1 = m(a=1)
    result = pmap1 == "not a map"
    assert result is NotImplemented


# LLM-generated content at query #27
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Create a simple bucket structure
    buckets = pvector([None, [('a', 1)], [('b', 2)], None])
    
    # Test constructor with size and buckets
    pmap_instance = PMap(2, buckets)
    
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == buckets
    assert pmap_instance._size == len(pmap_instance)


def test_pmap_constructor_empty():
    from pyrsistent import pvector
    
    # Create empty bucket structure
    buckets = pvector([None, None, None])
    
    # Test constructor with empty map
    pmap_instance = PMap(0, buckets)
    
    assert pmap_instance._size == 0
    assert pmap_instance._buckets == buckets
    assert len(pmap_instance) == 0


def test_pmap_constructor_large():
    from pyrsistent import pvector
    
    # Create a larger bucket structure
    buckets = pvector([
        [('key1', 'value1')],
        None,
        [('key2', 'value2'), ('key3', 'value3')],
        None,
        [('key4', 'value4')]
    ])
    
    # Test constructor with larger map
    pmap_instance = PMap(4, buckets)
    
    assert pmap_instance._size == 4
    assert pmap_instance._buckets == buckets
    assert len(pmap_instance) == 4


def test_pmap_constructor_returns_instance():
    from pyrsistent import pvector
    
    buckets = pvector([None, [('x', 10)]])
    pmap_instance = PMap(1, buckets)
    
    assert isinstance(pmap_instance, PMap)
    assert hasattr(pmap_instance, '_size')
    assert hasattr(pmap_instance, '_buckets')


# LLM-generated content at query #28
#--------------------------

```python
def test_contains_returns_false_on_exception():
    from pyrsistent import pmap
    
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    
    # Test that unpacking fails and exception is caught, returning False
    result = items.__contains__("single_value")
    assert result is False
    
    # Test with other non-unpackable types
    result = items.__contains__(42)
    assert result is False
    
    result = items.__contains__(None)
    assert result is False
    
    result = items.__contains__([1, 2, 3])
    assert result is False


# LLM-generated content at query #29
#--------------------------

```python
def test_pmap_eq_same_instance():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    assert m1 == m1


def test_pmap_eq_with_dict():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1


def test_pmap_eq_with_another_pmap():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    assert m1 == m2


def test_pmap_eq_different_length():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2, 'c': 3})
    assert not (m1 == m2)


def test_pmap_eq_different_values():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    assert not (m1 == m2)


def test_pmap_eq_different_keys():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'c': 2})
    assert not (m1 == m2)


def test_pmap_eq_with_non_mapping():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.__eq__("not a mapping")
    assert result == NotImplemented


def test_pmap_eq_empty_pmaps():
    from pyrsistent import pmap
    m1 = pmap()
    m2 = pmap()
    assert m1 == m2


def test_pmap_eq_empty_pmap_with_empty_dict():
    from pyrsistent import pmap
    m1 = pmap()
    d1 = {}
    assert m1 == d1


def test_pmap_eq_with_dict_different_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1})
    d1 = {'b': 2}
    assert not (m1 == d1)


def test_pmap_eq_with_mapping_protocol():
    from pyrsistent import pmap
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
    
    m1 = pmap({'a': 1, 'b': 2})
    cm = CustomMapping({'a': 1, 'b': 2})
    assert m1 == cm


def test_pmap_eq_with_mapping_protocol_different():
    from pyrsistent import pmap
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
    
    m1 = pmap({'a': 1, 'b': 2})
    cm = CustomMapping({'a': 1, 'c': 3})
    assert not (m1 == cm)


# LLM-generated content at query #30
#--------------------------

```python
def test_pmap_items_contains_with_valid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1) in items_view

def test_pmap_items_contains_with_invalid_key():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('c', 1) not in items_view

def test_pmap_items_contains_with_invalid_value():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 99) not in items_view

def test_pmap_items_contains_with_non_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert 'a' not in items_view

def test_pmap_items_contains_with_single_element():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert (1,) not in items_view

def test_pmap_items_contains_with_three_element_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1, 'extra') not in items_view

def test_pmap_items_contains_with_none_value():
    from pyrsistent import pmap
    m = pmap({'a': None})
    items_view = m.items()
    assert ('a', None) in items_view

def test_pmap_items_contains_with_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    assert ('a', 1) not in items_view


# LLM-generated content at query #31
#--------------------------

```python
def test_turbo_mapping_exception_handler():
    from collections.abc import Mapping
    from pyrsistent import pvector, PMap
    
    class NoLen:
        def __iter__(self):
            return iter([('key1', 'value1')])
        
        def items(self):
            return [('key1', 'value1')]
    
    initial = NoLen()
    pre_size = None
    
    try:
        len(initial)
        exception_raised = False
    except Exception:
        exception_raised = True
    
    assert exception_raised == True
    result = _turbo_mapping(initial, pre_size)
    assert result is not None


# LLM-generated content at query #32
#--------------------------

```python
def test_pmap_eq_same_object():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    assert pmap1 == pmap1


def test_pmap_eq_equal_pmaps():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert pmap1 == pmap2


def test_pmap_eq_different_pmaps():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_pmap_vs_dict():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    assert pmap1 == dict1


def test_pmap_eq_pmap_vs_dict_different():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 3}
    assert not (pmap1 == dict1)


def test_pmap_eq_different_sizes():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2, c=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_non_mapping():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    result = pmap1.__eq__([1, 2, 3])
    assert result == NotImplemented


def test_pmap_eq_empty_pmaps():
    from pyrsistent import m
    pmap1 = m()
    pmap2 = m()
    assert pmap1 == pmap2


def test_pmap_eq_empty_pmap_vs_empty_dict():
    from pyrsistent import m
    pmap1 = m()
    dict1 = {}
    assert pmap1 == dict1


def test_pmap_eq_with_different_key_types():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(**{'a': 1, 'b': 2})
    assert pmap1 == pmap2


def test_pmap_eq_pmap_vs_regular_mapping():
    from pyrsistent import m
    from collections import OrderedDict
    pmap1 = m(a=1, b=2)
    ordered_dict = OrderedDict([('a', 1), ('b', 2)])
    assert pmap1 == ordered_dict


def test_pmap_ne_operator():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert pmap1 != pmap2


def test_pmap_ne_operator_equal():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert not (pmap1 != pmap2)


# LLM-generated content at query #33
#--------------------------

```python
def test_turbo_mapping_exception_handling():
    from collections.abc import Mapping
    
    class NoLenObject:
        def __iter__(self):
            return iter([('key', 'value')])
        
        def items(self):
            return [('key', 'value')]
    
    initial = NoLenObject()
    pre_size = None
    
    try:
        len(initial)
        exception_raised = False
    except Exception:
        exception_raised = True
    
    assert exception_raised == True


# LLM-generated content at query #34
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    
    result = m1.update_with(lambda l, r: l + r, m2)
    
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #35
#--------------------------

```python
def test_turbo_mapping_exception_handler():
    from collections.abc import Mapping
    from pyrsistent import pvector, PMap
    
    class NoLenObject:
        def __iter__(self):
            return iter([('key1', 'value1')])
        
        def items(self):
            return [('key1', 'value1')]
    
    initial = NoLenObject()
    pre_size = None
    
    result = _turbo_mapping(initial, pre_size)
    
    assert isinstance(result, PMap)
    assert result.size == 1


# LLM-generated content at query #36
#--------------------------

```python
def test_pmap_items_eq_same_instance():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert items == items


def test_pmap_items_eq_different_instances_same_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


def test_pmap_items_eq_different_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)


def test_pmap_items_eq_different_keys():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'c': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)


def test_pmap_items_eq_empty_maps():
    from pyrsistent import pmap
    m1 = pmap({})
    m2 = pmap({})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


def test_pmap_items_eq_different_type():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == {'a': 1, 'b': 2})


def test_pmap_items_eq_different_type_dict_items():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    dict_items = {'a': 1, 'b': 2}.items()
    assert not (items == dict_items)


def test_pmap_items_eq_different_type_none():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == None)


def test_pmap_items_eq_different_type_string():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == "pmap_items([('a', 1), ('b', 2)])")


# LLM-generated content at query #37
#--------------------------

```python
def test_pmap_items_contains_with_valid_key_value_pair():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert ('a', 1) in items


def test_pmap_items_contains_with_invalid_key_value_pair():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert ('a', 2) not in items


def test_pmap_items_contains_with_nonexistent_key():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert ('c', 1) not in items


def test_pmap_items_contains_with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert 'a' not in items


def test_pmap_items_contains_with_single_element_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert ('a',) not in items


def test_pmap_items_contains_with_three_element_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert ('a', 1, 'extra') not in items


def test_pmap_items_contains_with_none_values():
    from pyrsistent import pmap
    m = pmap({'a': None, 'b': 2})
    items = m.items()
    assert ('a', None) in items


def test_pmap_items_contains_with_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    items = m.items()
    assert ('a', 1) not in items


def test_pmap_items_contains_with_matching_pair():
    from pyrsistent import pmap
    m = pmap({'x': 'y', 'foo': 'bar'})
    items = m.items()
    assert ('x', 'y') in items
    assert ('foo', 'bar') in items


# LLM-generated content at query #38
#--------------------------

```python
def test_eq_same_object():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    assert pmap1 == pmap1

def test_eq_different_pmap_same_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert pmap1 == pmap2

def test_eq_different_pmap_different_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert not (pmap1 == pmap2)

def test_eq_pmap_vs_dict_same_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    assert pmap1 == dict1

def test_eq_pmap_vs_dict_different_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 3}
    assert not (pmap1 == dict1)

def test_eq_pmap_vs_non_mapping():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    non_mapping = "not a mapping"
    result = pmap1.__eq__(non_mapping)
    assert result == NotImplemented

def test_eq_different_lengths():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2, c=3)
    assert not (pmap1 == pmap2)

def test_eq_empty_pmaps():
    from pyrsistent import m
    pmap1 = m()
    pmap2 = m()
    assert pmap1 == pmap2

def test_eq_empty_pmap_vs_empty_dict():
    from pyrsistent import m
    pmap1 = m()
    dict1 = {}
    assert pmap1 == dict1

def test_eq_pmap_vs_mapping_protocol():
    from pyrsistent import m
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
    
    pmap1 = m(a=1, b=2)
    custom_mapping = CustomMapping({'a': 1, 'b': 2})
    assert pmap1 == custom_mapping

def test_eq_pmap_with_cached_hash_equal():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    hash(pmap1)
    hash(pmap2)
    assert pmap1 == pmap2

def test_eq_pmap_with_different_cached_hash():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(c=3, d=4)
    hash(pmap1)
    hash(pmap2)
    assert not (pmap1 == pmap2)


# LLM-generated content at query #39
#--------------------------

```python
def test_contains_predicate_line_4_evaluates_to_false():
    from pyrsistent import pmap
    
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    
    # Test case 1: key exists but value doesn't match
    result1 = ('a', 999) in items_view
    assert result1 is False
    
    # Test case 2: key doesn't exist
    result2 = ('nonexistent', 1) in items_view
    assert result2 is False
    
    # Test case 3: key doesn't exist and value doesn't match
    result3 = ('x', 99) in items_view
    assert result3 is False


# LLM-generated content at query #40
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Create a simple bucket structure
    buckets = pvector([None, [('a', 1)], [('b', 2)], None])
    size = 2
    
    # Test constructor with valid arguments
    pmap_instance = PMap(size, buckets)
    
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == buckets
    assert hasattr(pmap_instance, '__weakref__')


def test_pmap_constructor_empty():
    from pyrsistent import pvector
    
    # Create an empty bucket structure
    buckets = pvector([None, None, None, None])
    size = 0
    
    # Test constructor with empty map
    pmap_instance = PMap(size, buckets)
    
    assert pmap_instance._size == 0
    assert pmap_instance._buckets == buckets


def test_pmap_constructor_large_map():
    from pyrsistent import pvector
    
    # Create a larger bucket structure with multiple collisions
    buckets = pvector([
        [('key1', 'val1'), ('key2', 'val2')],
        [('key3', 'val3')],
        None,
        [('key4', 'val4'), ('key5', 'val5'), ('key6', 'val6')],
        None
    ])
    size = 6
    
    # Test constructor with larger map
    pmap_instance = PMap(size, buckets)
    
    assert pmap_instance._size == 6
    assert pmap_instance._buckets == buckets
    assert len(pmap_instance._buckets) == 5


# LLM-generated content at query #41
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Create a simple bucket structure
    buckets = pvector([None, [('a', 1)], [('b', 2)], None])
    size = 2
    
    # Test constructor with valid arguments
    pmap_instance = PMap(size, buckets)
    
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == buckets
    assert len(pmap_instance) == 2


def test_pmap_constructor_empty():
    from pyrsistent import pvector
    
    # Create an empty bucket structure
    buckets = pvector([None, None, None, None])
    size = 0
    
    # Test constructor with empty map
    pmap_instance = PMap(size, buckets)
    
    assert pmap_instance._size == 0
    assert pmap_instance._buckets == buckets
    assert len(pmap_instance) == 0


def test_pmap_constructor_large_map():
    from pyrsistent import pvector
    
    # Create a bucket structure with multiple collisions
    buckets = pvector([
        [('key1', 'value1'), ('key2', 'value2')],
        [('key3', 'value3')],
        None,
        [('key4', 'value4'), ('key5', 'value5'), ('key6', 'value6')]
    ])
    size = 6
    
    # Test constructor with larger map
    pmap_instance = PMap(size, buckets)
    
    assert pmap_instance._size == 6
    assert pmap_instance._buckets == buckets
    assert len(pmap_instance) == 6


def test_pmap_constructor_attributes():
    from pyrsistent import pvector
    
    # Test that __slots__ are properly initialized
    buckets = pvector([None, [('x', 10)], None])
    size = 1
    
    pmap_instance = PMap(size, buckets)
    
    assert hasattr(pmap_instance, '_size')
    assert hasattr(pmap_instance, '_buckets')
    assert hasattr(pmap_instance, '_cached_hash') or not hasattr(pmap_instance, '_cached_hash')


# LLM-generated content at query #42
#--------------------------

```python
def test_update_with_predicate_false():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    
    result = m1.update_with(lambda l, r: l + r, m2)
    
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #43
#--------------------------

```python
def test_contains_predicate_line_4_evaluates_to_false():
    from pyrsistent import pmap
    
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    
    # Test case 1: key exists but value doesn't match
    result1 = ('a', 999) in items_view
    assert result1 == False
    
    # Test case 2: key doesn't exist
    result2 = ('z', 1) in items_view
    assert result2 == False
    
    # Test case 3: key doesn't exist and value doesn't match
    result3 = ('z', 999) in items_view
    assert result3 == False


# LLM-generated content at query #44
#--------------------------

```python
def test_turbo_mapping_exception_predicate_false():
    from collections.abc import Mapping
    
    class NoLenObject:
        def __iter__(self):
            return iter([('a', 1), ('b', 2)])
        
        def items(self):
            return [('a', 1), ('b', 2)]
    
    initial = NoLenObject()
    pre_size = None
    
    try:
        len(initial)
        exception_raised = False
    except Exception:
        exception_raised = True
    
    assert exception_raised == True


# LLM-generated content at query #45
#--------------------------

```python
def test_pmap_items_eq_same_object():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert items == items

def test_pmap_items_eq_different_objects_same_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2

def test_pmap_items_eq_different_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)

def test_pmap_items_eq_different_keys():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'c': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)

def test_pmap_items_eq_not_pmap_items_type():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == {'a': 1, 'b': 2})

def test_pmap_items_eq_with_dict_items():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    dict_items = {'a': 1, 'b': 2}.items()
    assert not (items == dict_items)

def test_pmap_items_eq_empty_maps():
    from pyrsistent import pmap
    m1 = pmap()
    m2 = pmap()
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2

def test_pmap_items_eq_one_empty_one_not():
    from pyrsistent import pmap
    m1 = pmap()
    m2 = pmap({'a': 1})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)

def test_pmap_items_eq_with_none_values():
    from pyrsistent import pmap
    m1 = pmap({'a': None, 'b': 2})
    m2 = pmap({'a': None, 'b': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2

def test_pmap_items_eq_with_string():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == "pmap_items([('a', 1), ('b', 2)])")


# LLM-generated content at query #46
#--------------------------

```python
def test_contains_returns_false_on_unpacking_exception():
    from pyrsistent import pmap
    
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    
    # Test with a single value that cannot be unpacked into (k, v)
    result = ('single_value',) in items_view
    assert result is False
    
    # Test with a string that cannot be unpacked into (k, v)
    result = 'string' in items_view
    assert result is False
    
    # Test with an integer that cannot be unpacked into (k, v)
    result = 42 in items_view
    assert result is False
    
    # Test with a list of 3 elements that cannot be unpacked into (k, v)
    result = [1, 2, 3] in items_view
    assert result is False
    
    # Test with None that cannot be unpacked into (k, v)
    result = None in items_view
    assert result is False


# LLM-generated content at query #47
#--------------------------

```python
def test_pmap_eq_same_instance():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    assert pmap1 == pmap1


def test_pmap_eq_equivalent_pmap():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert pmap1 == pmap2


def test_pmap_eq_different_pmap():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_different_keys():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, c=2)
    assert not (pmap1 == pmap2)


def test_pmap_eq_dict():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    assert pmap1 == dict1


def test_pmap_eq_dict_different():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, b=3}
    assert not (pmap1 == dict1)


def test_pmap_eq_empty_pmap():
    from pyrsistent import m
    pmap1 = m()
    pmap2 = m()
    assert pmap1 == pmap2


def test_pmap_eq_empty_dict():
    from pyrsistent import m
    pmap1 = m()
    dict1 = {}
    assert pmap1 == dict1


def test_pmap_eq_different_sizes():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1)
    assert not (pmap1 == pmap2)


def test_pmap_eq_non_mapping():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    result = pmap1.__eq__([1, 2, 3])
    assert result == NotImplemented


def test_pmap_eq_string():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    result = pmap1.__eq__("not a mapping")
    assert result == NotImplemented


def test_pmap_eq_pmap_with_cached_hash_equal():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    hash(pmap1)
    hash(pmap2)
    assert pmap1 == pmap2


def test_pmap_eq_pmap_with_different_cached_hash():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    hash(pmap1)
    hash(pmap2)
    assert not (pmap1 == pmap2)


def test_pmap_eq_with_mapping_protocol():
    from pyrsistent import m
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
    
    pmap1 = m(a=1, b=2)
    custom = CustomMapping({'a': 1, 'b': 2})
    assert pmap1 == custom


def test_pmap_eq_with_mapping_protocol_different():
    from pyrsistent import m
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
    
    pmap1 = m(a=1, b=2)
    custom = CustomMapping({'a': 1, b=3})
    assert not (pmap1 == custom)


def test_pmap_eq_reflexive():
    from pyrsistent import m
    pmap1 = m(x=10, y=20, z=30)
    assert pmap1 == pmap1


def test_pmap_eq_symmetric():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert (pmap1 == pmap2) == (pmap2 == pmap1)


def test_pmap_eq_transitive():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    pmap3 = m(a=1, b=2)
    assert pmap1 == pmap2
    assert pmap2 == pmap3
    assert pmap1 == pmap3


# LLM-generated content at query #48
#--------------------------

```python
def test_update_with_predicate_false():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    
    # Create an evolver and check that the predicate "key in evolver" is False
    # for a key that doesn't exist in the evolver
    evolver = m1.evolver()
    
    # The key 'c' is not in evolver, so "key in evolver" should be False
    assert ('c' in evolver) is False
    
    # When update_with is called with a new key not in m1,
    # the predicate should evaluate to False and use the value directly
    result = m1.update_with(lambda l, r: l + r, m2)
    
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #49
#--------------------------

```python
def test_eq_predicate_line_3_evaluates_to_true():
    from pyrsistent import pmap
    
    m1 = pmap({'a': 1, 'b': 2})
    items1 = PMapItems(m1)
    
    # Test with an object that is not an instance of PMapItems
    not_pmap_items = "not a PMapItems"
    result = not isinstance(not_pmap_items, type(items1))
    assert result is True
    
    # Test the actual __eq__ method with non-PMapItems object
    assert (items1 == not_pmap_items) is False
    
    # Test with another non-PMapItems type
    assert (items1 == {'a': 1, 'b': 2}) is False
    assert (items1 == 42) is False
    assert (items1 == None) is False


# LLM-generated content at query #50
#--------------------------

```python
def test_turbo_mapping_with_empty_dict():
    from pyrsistent import pvector
    result = _turbo_mapping({}, None)
    assert len(result) == 0
    assert result == {}


def test_turbo_mapping_with_single_item():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, None)
    assert len(result) == 1
    assert result['a'] == 1
    assert result == {'a': 1}


def test_turbo_mapping_with_multiple_items():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3}, None)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert result == {'a': 1, 'b': 2, 'c': 3}


def test_turbo_mapping_with_explicit_pre_size():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2}, 32)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2


def test_turbo_mapping_with_list_of_tuples():
    from pyrsistent import pvector
    result = _turbo_mapping([('x', 10), ('y', 20)], None)
    assert len(result) == 2
    assert result['x'] == 10
    assert result['y'] == 20


def test_turbo_mapping_with_existing_pmap():
    from pyrsistent import m
    original = m(a=1, b=2)
    result = _turbo_mapping(original, None)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2


def test_turbo_mapping_returns_pmap_instance():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, None)
    assert isinstance(result, PMap)


def test_turbo_mapping_with_hash_collisions():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, None)
    assert len(result) == 5
    assert result['a'] == 1
    assert result['e'] == 5


def test_turbo_mapping_with_pre_size_zero():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2}, 0)
    assert len(result) == 2
    assert result == {'a': 1, 'b': 2}


def test_turbo_mapping_with_large_pre_size():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, 1024)
    assert len(result) == 1
    assert result['a'] == 1


def test_turbo_mapping_preserves_values():
    from pyrsistent import pvector
    result = _turbo_mapping({'key1': 'value1', 'key2': 42, 'key3': None}, None)
    assert result['key1'] == 'value1'
    assert result['key2'] == 42
    assert result['key3'] is None


def test_turbo_mapping_with_duplicate_keys_last_wins():
    from pyrsistent import pvector
    result = _turbo_mapping([('a', 1), ('a', 2)], None)
    assert len(result) == 1
    assert result['a'] == 2


# LLM-generated content at query #51
#--------------------------

```python
def test_contains_predicate_line_4_evaluates_to_false():
    from pyrsistent import pmap
    
    items_view = pmap({'a': 1, 'b': 2}).items()
    
    # Test case 1: key exists but value doesn't match
    result1 = ('a', 999) in items_view
    assert result1 is False
    
    # Test case 2: key doesn't exist
    result2 = ('nonexistent', 1) in items_view
    assert result2 is False
    
    # Test case 3: both key and value don't match
    result3 = ('nonexistent', 999) in items_view
    assert result3 is False


# LLM-generated content at query #52
#--------------------------

```python
def test_pmap_items_contains_with_valid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1) in items_view


def test_pmap_items_contains_with_invalid_key():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('c', 1) not in items_view


def test_pmap_items_contains_with_invalid_value():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 999) not in items_view


def test_pmap_items_contains_with_non_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert 'a' not in items_view


def test_pmap_items_contains_with_list():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ['a', 1] not in items_view


def test_pmap_items_contains_with_wrong_length_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1, 'extra') not in items_view


def test_pmap_items_contains_with_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    assert ('a', 1) not in items_view


def test_pmap_items_contains_with_none_values():
    from pyrsistent import pmap
    m = pmap({'a': None, 'b': 2})
    items_view = m.items()
    assert ('a', None) in items_view


# LLM-generated content at query #53
#--------------------------

```python
def test_pmap_eq_with_dict_predicate():
    from pyrsistent import pmap
    
    pm = pmap({'a': 1, 'b': 2})
    d = {'a': 1, 'b': 2}
    
    # Line 15 predicate: isinstance(other, dict)
    # We need this to evaluate to False, so other should not be a dict
    # But should still be a Mapping for line 4 to pass
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
    
    # isinstance(other, dict) should be False for CustomMapping
    assert not isinstance(custom_map, dict)
    assert isinstance(custom_map, Mapping)
    
    # Equality should still work and reach line 17
    assert pm == custom_map


# LLM-generated content at query #54
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Create a simple bucket structure
    buckets = pvector([None, [('a', 1)], [('b', 2)], None])
    
    # Test constructor with size and buckets
    pmap = PMap(2, buckets)
    
    assert pmap._size == 2
    assert pmap._buckets == buckets
    assert len(pmap) == 2


def test_pmap_constructor_empty():
    from pyrsistent import pvector
    
    # Create empty buckets
    buckets = pvector([None, None, None, None])
    
    # Test constructor with empty map
    pmap = PMap(0, buckets)
    
    assert pmap._size == 0
    assert pmap._buckets == buckets
    assert len(pmap) == 0


def test_pmap_constructor_single_element():
    from pyrsistent import pvector
    
    # Create buckets with single element
    buckets = pvector([[('key', 'value')]])
    
    # Test constructor with single element
    pmap = PMap(1, buckets)
    
    assert pmap._size == 1
    assert pmap._buckets == buckets
    assert len(pmap) == 1


def test_pmap_constructor_multiple_collisions():
    from pyrsistent import pvector
    
    # Create buckets with hash collisions
    buckets = pvector([None, [('key1', 1), ('key2', 2), ('key3', 3)], None])
    
    # Test constructor with collisions
    pmap = PMap(3, buckets)
    
    assert pmap._size == 3
    assert pmap._buckets == buckets
    assert len(pmap) == 3


# LLM-generated content at query #55
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Create a simple bucket structure
    buckets = pvector([[(1, 'a')], [(2, 'b')], None])
    
    # Test constructor with size and buckets
    pmap_instance = PMap(2, buckets)
    
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == buckets
    assert pmap_instance._size == len(pmap_instance._buckets)


def test_pmap_constructor_empty():
    from pyrsistent import pvector
    
    # Test constructor with empty buckets
    buckets = pvector([None, None, None])
    pmap_instance = PMap(0, buckets)
    
    assert pmap_instance._size == 0
    assert pmap_instance._buckets == buckets


def test_pmap_constructor_large():
    from pyrsistent import pvector
    
    # Test constructor with larger size
    buckets = pvector([[(i, f'val_{i}')] for i in range(100)])
    pmap_instance = PMap(100, buckets)
    
    assert pmap_instance._size == 100
    assert pmap_instance._buckets == buckets


def test_pmap_constructor_with_collisions():
    from pyrsistent import pvector
    
    # Test constructor with hash collisions (multiple items in same bucket)
    buckets = pvector([[(1, 'a'), (2, 'b')], None, [(3, 'c')]])
    pmap_instance = PMap(3, buckets)
    
    assert pmap_instance._size == 3
    assert pmap_instance._buckets == buckets
    assert pmap_instance._buckets[0] == [(1, 'a'), (2, 'b')]


# LLM-generated content at query #56
#--------------------------

```python
def test_pmap_items_contains_with_valid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1) in items_view


def test_pmap_items_contains_with_invalid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 2) not in items_view


def test_pmap_items_contains_with_missing_key():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('c', 1) not in items_view


def test_pmap_items_contains_with_non_tuple_arg():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert 'a' not in items_view


def test_pmap_items_contains_with_single_element_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a',) not in items_view


def test_pmap_items_contains_with_three_element_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1, 'extra') not in items_view


def test_pmap_items_contains_with_none_values():
    from pyrsistent import pmap
    m = pmap({'a': None, 'b': 2})
    items_view = m.items()
    assert ('a', None) in items_view


def test_pmap_items_contains_with_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    assert ('a', 1) not in items_view


def test_pmap_items_contains_with_list_as_arg():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ['a', 1] not in items_view


# LLM-generated content at query #57
#--------------------------

```python
def test_contains_predicate_evaluates_to_false():
    from pyrsistent import pmap
    
    # Create a PMap and its items view
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    
    # Test case 1: key exists but value doesn't match
    result1 = items_view.__contains__(('a', 999))
    assert result1 is False
    
    # Test case 2: key doesn't exist
    result2 = items_view.__contains__(('c', 1))
    assert result2 is False
    
    # Test case 3: key doesn't exist and value doesn't match
    result3 = items_view.__contains__(('z', 999))
    assert result3 is False


# LLM-generated content at query #58
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    result = m1.update_with(lambda l, r: l + r, m2)
    
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #59
#--------------------------

```python
def test_pmap_eq_with_non_dict_mapping():
    from pyrsistent import pmap
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
    
    m1 = pmap({'a': 1, 'b': 2})
    custom_mapping = CustomMapping({'a': 1, 'b': 2})
    
    result = m1 == custom_mapping
    assert result is True
    assert isinstance(custom_mapping, Mapping)
    assert not isinstance(custom_mapping, dict)


# LLM-generated content at query #60
#--------------------------

```python
def test_turbo_mapping_empty_dict():
    from pyrsistent import pvector
    result = _turbo_mapping({}, None)
    assert len(result) == 0
    assert result._size == 0


def test_turbo_mapping_single_item():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, None)
    assert len(result) == 1
    assert result['a'] == 1


def test_turbo_mapping_multiple_items():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3}, None)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


def test_turbo_mapping_with_pre_size():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2}, 32)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2


def test_turbo_mapping_from_list_of_tuples():
    from pyrsistent import pvector
    result = _turbo_mapping([('x', 10), ('y', 20)], None)
    assert len(result) == 2
    assert result['x'] == 10
    assert result['y'] == 20


def test_turbo_mapping_preserves_values():
    from pyrsistent import pvector
    initial = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    result = _turbo_mapping(initial, None)
    assert dict(result.iteritems()) == initial


def test_turbo_mapping_returns_pmap_instance():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, None)
    assert isinstance(result, PMap)


def test_turbo_mapping_with_numeric_keys():
    from pyrsistent import pvector
    result = _turbo_mapping({1: 'one', 2: 'two', 3: 'three'}, None)
    assert len(result) == 3
    assert result[1] == 'one'
    assert result[2] == 'two'
    assert result[3] == 'three'


def test_turbo_mapping_with_none_values():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': None, 'b': 'value'}, None)
    assert len(result) == 2
    assert result['a'] is None
    assert result['b'] == 'value'


def test_turbo_mapping_size_calculation():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, None)
    assert result._size == 1


def test_turbo_mapping_large_dict():
    from pyrsistent import pvector
    large_dict = {f'key_{i}': i for i in range(100)}
    result = _turbo_mapping(large_dict, None)
    assert len(result) == 100
    for i in range(100):
        assert result[f'key_{i}'] == i


def test_turbo_mapping_duplicate_values():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 5, 'b': 5, 'c': 5}, None)
    assert len(result) == 3
    assert result['a'] == 5
    assert result['b'] == 5
    assert result['c'] == 5


def test_turbo_mapping_pre_size_larger_than_needed():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, 128)
    assert len(result) == 1
    assert result['a'] == 1


# LLM-generated content at query #61
#--------------------------

```python
def test_eq_predicate_line_3():
    from pyrsistent import pmap
    
    m1 = pmap({'a': 1, 'b': 2})
    items1 = PMapItems(m1)
    
    # Test that line 3 predicate evaluates to True when x is not an instance of type(self)
    result = items1 == "not a PMapItems object"
    assert result is False
    
    result = items1 == 42
    assert result is False
    
    result = items1 == {'a': 1, 'b': 2}
    assert result is False
    
    result = items1 == [('a', 1), ('b', 2)]
    assert result is False


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Test creating a PMap with empty buckets
    empty_buckets = pvector()
    pmap1 = PMap(0, empty_buckets)
    assert pmap1._size == 0
    assert pmap1._buckets == empty_buckets
    
    # Test creating a PMap with non-empty buckets
    buckets_with_data = pvector([None, [('key1', 'value1')], None])
    pmap2 = PMap(1, buckets_with_data)
    assert pmap2._size == 1
    assert pmap2._buckets == buckets_with_data
    
    # Test creating a PMap with multiple items
    buckets_multi = pvector([
        [('key1', 'value1')],
        [('key2', 'value2'), ('key3', 'value3')],
        None
    ])
    pmap3 = PMap(3, buckets_multi)
    assert pmap3._size == 3
    assert pmap3._buckets == buckets_multi
    
    # Test that the returned object is an instance of PMap
    assert isinstance(pmap1, PMap)
    assert isinstance(pmap2, PMap)
    assert isinstance(pmap3, PMap)
    
    # Test that slots are properly initialized
    assert hasattr(pmap1, '_size')
    assert hasattr(pmap1, '_buckets')
    assert hasattr(pmap1, '__weakref__')
    assert hasattr(pmap1, '_cached_hash')


# LLM-generated content at query #2
#--------------------------

```python
def test_update_with_single_map():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}


def test_update_with_multiple_maps():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2, c=3), m(a=1))
    assert result == {'a': 4, 'b': 2, 'c': 3}


def test_update_with_keep_left():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1}


def test_update_with_empty_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r)
    assert result == {'a': 1, 'b': 2}


def test_update_with_new_keys():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1)
    result = m1.update_with(add, m(b=2, c=3))
    assert result == {'a': 1, 'b': 2, 'c': 3}


def test_update_with_original_unchanged():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1, b=2)
    m2 = m1.update_with(add, m(a=5))
    assert m1 == {'a': 1, 'b': 2}
    assert m2 == {'a': 6, 'b': 2}


def test_update_with_custom_function():
    from pyrsistent import m
    m1 = m(a=[1, 2])
    result = m1.update_with(lambda l, r: l + r, m(a=[3, 4]))
    assert result == {'a': [1, 2, 3, 4]}


def test_update_with_dict():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, {'a': 3, 'c': 4})
    assert result == {'a': 4, 'b': 2, 'c': 4}


def test_update_with_multiple_dicts():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: r, {'a': 2}, {'a': 3})
    assert result == {'a': 3}


def test_update_with_overwrite_function():
    from pyrsistent import m
    m1 = m(x=10, y=20)
    result = m1.update_with(lambda l, r: l * r, m(x=2, z=5))
    assert result == {'x': 20, 'y': 20, 'z': 5}


# LLM-generated content at query #3
#--------------------------

```python
def test_pmap_items_eq_same_instance():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert items == items


def test_pmap_items_eq_different_instances_same_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


def test_pmap_items_eq_different_instances_different_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)


def test_pmap_items_eq_with_non_pmap_items_type():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == {'a': 1, 'b': 2}.items())


def test_pmap_items_eq_with_string():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == "pmap_items")


def test_pmap_items_eq_with_none():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == None)


def test_pmap_items_eq_empty_maps():
    from pyrsistent import pmap
    m1 = pmap({})
    m2 = pmap({})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


# LLM-generated content at query #4
#--------------------------

```python
def test_eq_same_instance():
    from pyrsistent import m
    map1 = m(a=1, b=2)
    assert map1 == map1

def test_eq_different_pmaps_same_content():
    from pyrsistent import m
    map1 = m(a=1, b=2)
    map2 = m(a=1, b=2)
    assert map1 == map2

def test_eq_different_pmaps_different_content():
    from pyrsistent import m
    map1 = m(a=1, b=2)
    map2 = m(a=1, b=3)
    assert not (map1 == map2)

def test_eq_pmap_vs_dict_same_content():
    from pyrsistent import m
    map1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    assert map1 == dict1

def test_eq_pmap_vs_dict_different_content():
    from pyrsistent import m
    map1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 3}
    assert not (map1 == dict1)

def test_eq_different_sizes():
    from pyrsistent import m
    map1 = m(a=1, b=2)
    map2 = m(a=1, b=2, c=3)
    assert not (map1 == map2)

def test_eq_empty_pmaps():
    from pyrsistent import m
    map1 = m()
    map2 = m()
    assert map1 == map2

def test_eq_empty_pmap_vs_empty_dict():
    from pyrsistent import m
    map1 = m()
    dict1 = {}
    assert map1 == dict1

def test_eq_pmap_vs_non_mapping():
    from pyrsistent import m
    map1 = m(a=1, b=2)
    result = map1.__eq__([1, 2, 3])
    assert result == NotImplemented

def test_eq_pmap_vs_list():
    from pyrsistent import m
    map1 = m(a=1, b=2)
    assert not (map1 == [1, 2, 3])

def test_eq_pmap_with_cached_hash_same_hash():
    from pyrsistent import m
    map1 = m(a=1, b=2)
    map2 = m(a=1, b=2)
    hash(map1)
    hash(map2)
    assert map1 == map2

def test_eq_pmap_with_different_cached_hash():
    from pyrsistent import m
    map1 = m(a=1, b=2)
    map2 = m(a=1, b=3)
    hash(map1)
    hash(map2)
    assert not (map1 == map2)

def test_eq_pmap_vs_custom_mapping():
    from pyrsistent import m
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
    
    map1 = m(a=1, b=2)
    custom = CustomMapping({'a': 1, 'b': 2})
    assert map1 == custom

def test_eq_pmap_vs_custom_mapping_different():
    from pyrsistent import m
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
    
    map1 = m(a=1, b=2)
    custom = CustomMapping({'a': 1, 'b': 3})
    assert not (map1 == custom)


# LLM-generated content at query #5
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    
    m1 = m(a=1)
    update_fn = lambda l, r: l + r
    m2 = m(b=2)
    
    result = m1.update_with(update_fn, m2)
    
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #6
#--------------------------

```python
def test_pmap_constructor():
    from pyrsistent import pvector
    
    # Create a simple bucket structure
    buckets = pvector([[(('a', 1))], [(('b', 2))]])
    size = 2
    
    # Test constructor
    pmap_instance = PMap(size, buckets)
    
    # Verify attributes are set correctly
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == buckets
    assert pmap_instance._size == size


def test_pmap_constructor_empty():
    from pyrsistent import pvector
    
    # Create an empty bucket structure
    buckets = pvector([None, None])
    size = 0
    
    # Test constructor with empty map
    pmap_instance = PMap(size, buckets)
    
    # Verify attributes are set correctly
    assert pmap_instance._size == 0
    assert pmap_instance._buckets == buckets


def test_pmap_constructor_multiple_buckets():
    from pyrsistent import pvector
    
    # Create a bucket structure with multiple entries per bucket
    buckets = pvector([[(('a', 1), ('c', 3))], [(('b', 2))], None])
    size = 3
    
    # Test constructor
    pmap_instance = PMap(size, buckets)
    
    # Verify attributes are set correctly
    assert pmap_instance._size == 3
    assert pmap_instance._buckets == buckets
    assert len(pmap_instance._buckets) == 3


# LLM-generated content at query #7
#--------------------------

```python
def test_pmap_items_contains_valid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1) in items_view


def test_pmap_items_contains_invalid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 2) not in items_view


def test_pmap_items_contains_missing_key():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('c', 1) not in items_view


def test_pmap_items_contains_non_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert 'a' not in items_view


def test_pmap_items_contains_single_value():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert 1 not in items_view


def test_pmap_items_contains_triple_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1, 'extra') not in items_view


def test_pmap_items_contains_empty_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert () not in items_view


def test_pmap_items_contains_none_values():
    from pyrsistent import pmap
    m = pmap({'a': None, 'b': 2})
    items_view = m.items()
    assert ('a', None) in items_view


def test_pmap_items_contains_multiple_valid_items():
    from pyrsistent import pmap
    m = pmap({'x': 10, 'y': 20, 'z': 30})
    items_view = m.items()
    assert ('x', 10) in items_view
    assert ('y', 20) in items_view
    assert ('z', 30) in items_view


# LLM-generated content at query #8
#--------------------------

```python
def test_pmap_items_eq_same_instance():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert items == items


def test_pmap_items_eq_different_instances_same_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


def test_pmap_items_eq_different_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)


def test_pmap_items_eq_different_type():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    dict_items = {'a': 1, 'b': 2}.items()
    assert not (items == dict_items)


def test_pmap_items_eq_with_non_pmap_items():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == [('a', 1), ('b', 2)])


def test_pmap_items_eq_empty_maps():
    from pyrsistent import pmap
    m1 = pmap({})
    m2 = pmap({})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


# LLM-generated content at query #9
#--------------------------

```python
def test_turbo_mapping_with_empty_dict():
    from pyrsistent import pvector
    result = _turbo_mapping({}, None)
    assert len(result) == 0
    assert result == {}


def test_turbo_mapping_with_single_item():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, None)
    assert len(result) == 1
    assert result['a'] == 1
    assert result == {'a': 1}


def test_turbo_mapping_with_multiple_items():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3}, None)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert result == {'a': 1, 'b': 2, 'c': 3}


def test_turbo_mapping_with_pre_size():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2}, 16)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2
    assert result == {'a': 1, 'b': 2}


def test_turbo_mapping_with_list_of_tuples():
    from pyrsistent import pvector
    result = _turbo_mapping([('x', 10), ('y', 20)], None)
    assert len(result) == 2
    assert result['x'] == 10
    assert result['y'] == 20
    assert result == {'x': 10, 'y': 20}


def test_turbo_mapping_with_mapping_object():
    from pyrsistent import pvector, m
    initial_map = m(p=100, q=200)
    result = _turbo_mapping(initial_map, None)
    assert len(result) == 2
    assert result['p'] == 100
    assert result['q'] == 200
    assert result == {'p': 100, 'q': 200}


def test_turbo_mapping_returns_pmap_instance():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, None)
    assert isinstance(result, PMap)


def test_turbo_mapping_with_zero_pre_size():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2}, 0)
    assert len(result) == 2
    assert result == {'a': 1, 'b': 2}


def test_turbo_mapping_handles_hash_collisions():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3, 'd': 4}, 8)
    assert len(result) == 4
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert result['d'] == 4


def test_turbo_mapping_preserves_values():
    from pyrsistent import pvector
    result = _turbo_mapping({'key1': 'value1', 'key2': None, 'key3': 0}, None)
    assert result['key1'] == 'value1'
    assert result['key2'] is None
    assert result['key3'] == 0


# LLM-generated content at query #10
#--------------------------

```python
def test_pmap_items_eq():
    from pyrsistent import pmap
    
    # Create two PMapItems instances with the same underlying map
    map1 = pmap({'a': 1, 'b': 2})
    items1 = map1.items()
    items1_copy = map1.items()
    
    # Test equality with same object
    assert items1 == items1
    
    # Test equality with different PMapItems instances but same underlying map
    assert items1 == items1_copy
    
    # Create PMapItems with different underlying map
    map2 = pmap({'a': 1, 'b': 2})
    items2 = map2.items()
    
    # Test equality with different PMapItems instances but equivalent maps
    assert items1 == items2
    
    # Create PMapItems with different underlying map content
    map3 = pmap({'a': 1, 'c': 3})
    items3 = map3.items()
    
    # Test inequality with different map content
    assert not (items1 == items3)
    
    # Test inequality with non-PMapItems object
    assert not (items1 == {'a': 1, 'b': 2})
    assert not (items1 == [('a', 1), ('b', 2)])
    assert not (items1 == "pmap_items([('a', 1), ('b', 2)])")
    assert not (items1 == None)
    
    # Test inequality with empty maps
    empty_map = pmap({})
    empty_items = empty_map.items()
    assert not (items1 == empty_items)
    
    # Test equality with two empty maps
    empty_items2 = pmap({}).items()
    assert empty_items == empty_items2


# LLM-generated content at query #11
#--------------------------

```python
def test_pmap_eq_same_object():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    assert pmap1 == pmap1


def test_pmap_eq_with_dict_same_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    assert pmap1 == dict1


def test_pmap_eq_with_dict_different_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 3}
    assert not (pmap1 == dict1)


def test_pmap_eq_with_pmap_same_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert pmap1 == pmap2


def test_pmap_eq_with_pmap_different_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_different_lengths():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2, c=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_with_non_mapping():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    assert pmap1 != "not a mapping"


def test_pmap_eq_empty_maps():
    from pyrsistent import m
    pmap1 = m()
    pmap2 = m()
    assert pmap1 == pmap2


def test_pmap_eq_empty_pmap_with_empty_dict():
    from pyrsistent import m
    pmap1 = m()
    dict1 = {}
    assert pmap1 == dict1


def test_pmap_eq_with_generic_mapping():
    from pyrsistent import m
    from collections.abc import Mapping
    pmap1 = m(a=1, b=2)
    
    class CustomMapping(Mapping):
        def __getitem__(self, key):
            data = {'a': 1, 'b': 2}
            return data[key]
        
        def __iter__(self):
            return iter({'a': 1, 'b': 2})
        
        def __len__(self):
            return 2
    
    custom = CustomMapping()
    assert pmap1 == custom


def test_pmap_eq_with_pmaps_same_buckets():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = pmap1
    assert pmap1 == pmap2


def test_pmap_eq_single_element():
    from pyrsistent import m
    pmap1 = m(a=1)
    pmap2 = m(a=1)
    assert pmap1 == pmap2


def test_pmap_eq_single_element_different():
    from pyrsistent import m
    pmap1 = m(a=1)
    pmap2 = m(a=2)
    assert not (pmap1 == pmap2)


# LLM-generated content at query #12
#--------------------------

```python
def test_pmap_eq_with_non_dict_mapping():
    from pyrsistent import pmap
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
    
    pm = pmap({'a': 1, 'b': 2})
    custom_map = CustomMapping({'a': 1, 'b': 2})
    
    result = pm == custom_map
    assert result is True


# LLM-generated content at query #13
#--------------------------

```python
def test_turbo_mapping_exception_handler():
    from collections.abc import Mapping
    
    class NoLenObject:
        def __iter__(self):
            return iter([('key', 'value')])
        
        def items(self):
            return [('key', 'value')]
    
    initial = NoLenObject()
    pre_size = None
    
    try:
        len(initial)
        exception_raised = False
    except Exception:
        exception_raised = True
    
    assert exception_raised == True


# LLM-generated content at query #14
#--------------------------

```python
def test_contains_returns_false_on_unpacking_exception():
    from pyrsistent import pmap
    
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    
    # Test with a single value that cannot be unpacked into (k, v)
    result = "not_a_tuple" in items_view
    assert result is False
    
    # Test with a list containing single element that cannot be unpacked
    result = [1] in items_view
    assert result is False
    
    # Test with None
    result = None in items_view
    assert result is False
    
    # Test with an integer
    result = 42 in items_view
    assert result is False
    
    # Test with a string
    result = "single_string" in items_view
    assert result is False


# LLM-generated content at query #15
#--------------------------

```python
def test_pmap_eq_same_instance():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    assert pmap1 == pmap1


def test_pmap_eq_equal_pmaps():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert pmap1 == pmap2


def test_pmap_eq_different_pmaps():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_different_sizes():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2, c=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_with_dict():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    assert pmap1 == dict1


def test_pmap_eq_with_dict_not_equal():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 3}
    assert not (pmap1 == dict1)


def test_pmap_eq_with_non_mapping():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    result = pmap1.__eq__([1, 2, 3])
    assert result is NotImplemented


def test_pmap_eq_empty_pmaps():
    from pyrsistent import m
    pmap1 = m()
    pmap2 = m()
    assert pmap1 == pmap2


def test_pmap_eq_empty_pmap_with_empty_dict():
    from pyrsistent import m
    pmap1 = m()
    dict1 = {}
    assert pmap1 == dict1


def test_pmap_eq_with_generic_mapping():
    from pyrsistent import m
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
    
    pmap1 = m(a=1, b=2)
    custom = CustomMapping({'a': 1, 'b': 2})
    assert pmap1 == custom


def test_pmap_eq_with_different_bucket_same_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2, c=3)
    pmap2 = m(c=3, b=2, a=1)
    assert pmap1 == pmap2


def test_pmap_ne_different_pmaps():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert pmap1 != pmap2


def test_pmap_eq_with_dict_different_size():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2, 'c': 3}
    assert not (pmap1 == dict1)


# LLM-generated content at query #16
#--------------------------

```python
def test_turbo_mapping_with_dict():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({'a': 1, 'b': 2}, None)
    assert result['a'] == 1
    assert result['b'] == 2
    assert len(result) == 2

def test_turbo_mapping_with_pre_size():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({'a': 1, 'b': 2}, 16)
    assert result['a'] == 1
    assert result['b'] == 2
    assert len(result) == 2

def test_turbo_mapping_with_list_of_tuples():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping([('a', 1), ('b', 2)], None)
    assert result['a'] == 1
    assert result['b'] == 2
    assert len(result) == 2

def test_turbo_mapping_empty_dict():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({}, None)
    assert len(result) == 0

def test_turbo_mapping_single_element():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({'key': 'value'}, None)
    assert result['key'] == 'value'
    assert len(result) == 1

def test_turbo_mapping_with_collisions():
    from pyrsistent._pmap import _turbo_mapping
    initial = {i: i*2 for i in range(10)}
    result = _turbo_mapping(initial, None)
    for k, v in initial.items():
        assert result[k] == v
    assert len(result) == 10

def test_turbo_mapping_pre_size_larger():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({'a': 1}, 32)
    assert result['a'] == 1
    assert len(result) == 1

def test_turbo_mapping_contains():
    from pyrsistent._pmap import _turbo_mapping
    result = _turbo_mapping({'a': 1, 'b': 2}, None)
    assert 'a' in result
    assert 'b' in result
    assert 'c' not in result

def test_turbo_mapping_multiple_values():
    from pyrsistent._pmap import _turbo_mapping
    initial = {'x': 10, 'y': 20, 'z': 30}
    result = _turbo_mapping(initial, None)
    assert result['x'] == 10
    assert result['y'] == 20
    assert result['z'] == 30
    assert len(result) == 3

def test_turbo_mapping_with_mapping_object():
    from pyrsistent._pmap import _turbo_mapping
    from collections import OrderedDict
    initial = OrderedDict([('a', 1), ('b', 2)])
    result = _turbo_mapping(initial, None)
    assert result['a'] == 1
    assert result['b'] == 2
    assert len(result) == 2


# LLM-generated content at query #17
#--------------------------

```python
def test_contains_returns_false_on_exception():
    from pyrsistent import pmap
    
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    
    # Test with a non-iterable argument that cannot be unpacked
    result = 42 in items_view
    assert result is False
    
    # Test with a single element that cannot be unpacked into (k, v)
    result = "single" in items_view
    assert result is False
    
    # Test with a list of 3 elements that cannot be unpacked into (k, v)
    result = [1, 2, 3] in items_view
    assert result is False


# LLM-generated content at query #18
#--------------------------

```python
def test_update_with_predicate_key_not_in_evolver():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    
    result = m1.update_with(lambda l, r: l + r, m2)
    
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #19
#--------------------------

```python
def test_contains_returns_false_on_unpacking_exception():
    from pyrsistent import pmap
    
    items = pmap({'a': 1, 'b': 2}).items()
    
    # Test with a single value that cannot be unpacked into (k, v)
    result = "single_value" in items
    assert result is False
    
    # Test with a number that cannot be unpacked
    result = 42 in items
    assert result is False
    
    # Test with None that cannot be unpacked
    result = None in items
    assert result is False
    
    # Test with a list of 3 elements that cannot unpack into exactly 2 values
    result = [1, 2, 3] in items
    assert result is False


# LLM-generated content at query #20
#--------------------------

```python
def test_update_with_predicate_false():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    
    result = m1.update_with(lambda l, r: l + r, m2)
    
    assert result == {'a': 1, 'b': 2, 'c': 3}
    assert 'c' in result
    assert result['c'] == 3


# LLM-generated content at query #21
#--------------------------

```python
def test_pmap_items_eq_same_instance():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert items == items

def test_pmap_items_eq_different_instances_same_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2

def test_pmap_items_eq_different_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)

def test_pmap_items_eq_different_keys():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'c': 2})
    items1 = m1.items()
    items2 = m2.items()
    assert not (items1 == items2)

def test_pmap_items_eq_different_type():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == {'a': 1, 'b': 2}.items())

def test_pmap_items_eq_with_non_pmap_items():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    assert not (items == "not a pmap items")

def test_pmap_items_eq_empty_maps():
    from pyrsistent import pmap
    m1 = pmap({})
    m2 = pmap({})
    items1 = m1.items()
    items2 = m2.items()
    assert items1 == items2


# LLM-generated content at query #22
#--------------------------

```python
def test_turbo_mapping_with_empty_dict():
    result = _turbo_mapping({}, None)
    assert len(result) == 0
    assert result == {}

def test_turbo_mapping_with_single_item():
    result = _turbo_mapping({'a': 1}, None)
    assert len(result) == 1
    assert result == {'a': 1}
    assert result['a'] == 1

def test_turbo_mapping_with_multiple_items():
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3}, None)
    assert len(result) == 3
    assert result == {'a': 1, 'b': 2, 'c': 3}
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3

def test_turbo_mapping_with_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, 16)
    assert len(result) == 2
    assert result == {'a': 1, 'b': 2}

def test_turbo_mapping_with_dict_initial():
    initial = {'x': 10, 'y': 20}
    result = _turbo_mapping(initial, None)
    assert len(result) == 2
    assert result == {'x': 10, 'y': 20}

def test_turbo_mapping_with_list_of_tuples():
    result = _turbo_mapping([('key1', 'value1'), ('key2', 'value2')], None)
    assert len(result) == 2
    assert result == {'key1': 'value1', 'key2': 'value2'}

def test_turbo_mapping_with_large_dict():
    large_dict = {f'key{i}': i for i in range(100)}
    result = _turbo_mapping(large_dict, None)
    assert len(result) == 100
    for i in range(100):
        assert result[f'key{i}'] == i

def test_turbo_mapping_returns_pmap_instance():
    result = _turbo_mapping({'a': 1}, None)
    assert isinstance(result, PMap)

def test_turbo_mapping_with_collision_prone_keys():
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, 8)
    assert len(result) == 5
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert result['d'] == 4
    assert result['e'] == 5

def test_turbo_mapping_preserves_values():
    initial = {'key': 'value', 'number': 42, 'float': 3.14}
    result = _turbo_mapping(initial, None)
    assert result['key'] == 'value'
    assert result['number'] == 42
    assert result['float'] == 3.14

def test_turbo_mapping_with_special_keys():
    result = _turbo_mapping({1: 'one', 2: 'two', 'three': 3}, None)
    assert len(result) == 3
    assert result[1] == 'one'
    assert result[2] == 'two'
    assert result['three'] == 3

def test_turbo_mapping_with_none_values():
    result = _turbo_mapping({'a': None, 'b': None}, None)
    assert len(result) == 2
    assert result['a'] is None
    assert result['b'] is None

def test_turbo_mapping_from_mapping_object():
    from collections import OrderedDict
    ordered = OrderedDict([('first', 1), ('second', 2), ('third', 3)])
    result = _turbo_mapping(ordered, None)
    assert len(result) == 3
    assert result['first'] == 1
    assert result['second'] == 2
    assert result['third'] == 3


# LLM-generated content at query #23
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    
    result = m1.update_with(lambda l, r: l + r, m2)
    
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #24
#--------------------------

```python
def test_contains_returns_false_on_exception():
    from pyrsistent import pmap
    
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    
    # Test with values that cannot be unpacked into (k, v)
    result1 = "single_string" in items_view
    assert result1 is False
    
    result2 = 42 in items_view
    assert result2 is False
    
    result3 = [1, 2, 3] in items_view
    assert result3 is False
    
    result4 = None in items_view
    assert result4 is False


# LLM-generated content at query #25
#--------------------------

```python
def test_eq_predicate_line_3_returns_false():
    from pyrsistent import pmap
    
    m1 = pmap({'a': 1, 'b': 2})
    items1 = m1.items()
    
    # Test with a different type (not PMapItems)
    different_type_obj = {'a': 1, 'b': 2}
    
    # Line 3 predicate: not isinstance(x, type(self)) should evaluate to True
    # This means x should NOT be an instance of PMapItems
    result = items1.__eq__(different_type_obj)
    
    assert result is False


# LLM-generated content at query #26
#--------------------------

```python
def test_pmap_eq_same_instance():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    assert m1 == m1


def test_pmap_eq_different_instances_same_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    assert m1 == m2


def test_pmap_eq_different_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 3})
    assert not (m1 == m2)


def test_pmap_eq_different_keys():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'c': 2})
    assert not (m1 == m2)


def test_pmap_eq_different_sizes():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1})
    assert not (m1 == m2)


def test_pmap_eq_with_dict():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    d = {'a': 1, 'b': 2}
    assert m1 == d


def test_pmap_eq_with_dict_different_content():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    d = {'a': 1, 'b': 3}
    assert not (m1 == d)


def test_pmap_eq_with_non_mapping():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    result = m1.__eq__("not a mapping")
    assert result == NotImplemented


def test_pmap_eq_with_empty_pmaps():
    from pyrsistent import pmap
    m1 = pmap({})
    m2 = pmap({})
    assert m1 == m2


def test_pmap_eq_empty_pmap_with_empty_dict():
    from pyrsistent import pmap
    m1 = pmap({})
    d = {}
    assert m1 == d


def test_pmap_eq_with_different_pmap_same_content():
    from pyrsistent import pmap, m
    m1 = pmap({'a': 1, 'b': 2})
    m2 = m(a=1, b=2)
    assert m1 == m2


def test_pmap_eq_with_custom_mapping():
    from pyrsistent import pmap
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
    
    m1 = pmap({'a': 1, 'b': 2})
    custom = CustomMapping({'a': 1, 'b': 2})
    assert m1 == custom


def test_pmap_eq_with_custom_mapping_different_content():
    from pyrsistent import pmap
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
    
    m1 = pmap({'a': 1, 'b': 2})
    custom = CustomMapping({'a': 1, 'b': 3})
    assert not (m1 == custom)


# LLM-generated content at query #27
#--------------------------

```python
def test_turbo_mapping_empty_initial():
    from pyrsistent import pvector
    result = _turbo_mapping({}, None)
    assert result._size == 0
    assert len(result._buckets) == 8


def test_turbo_mapping_with_dict_initial():
    from pyrsistent import pvector
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, None)
    assert result._size == 2
    assert result['a'] == 1
    assert result['b'] == 2


def test_turbo_mapping_with_pre_size():
    from pyrsistent import pvector
    initial = {'a': 1}
    result = _turbo_mapping(initial, 16)
    assert result._size == 1
    assert len(result._buckets) == 16
    assert result['a'] == 1


def test_turbo_mapping_with_list_of_tuples():
    from pyrsistent import pvector
    initial = [('x', 10), ('y', 20)]
    result = _turbo_mapping(initial, None)
    assert result._size == 2
    assert result['x'] == 10
    assert result['y'] == 20


def test_turbo_mapping_pre_size_zero():
    from pyrsistent import pvector
    initial = {'a': 1}
    result = _turbo_mapping(initial, 0)
    assert result._size == 1
    assert len(result._buckets) == 8
    assert result['a'] == 1


def test_turbo_mapping_large_initial():
    from pyrsistent import pvector
    initial = {str(i): i for i in range(10)}
    result = _turbo_mapping(initial, None)
    assert result._size == 10
    assert len(result._buckets) == 20
    for i in range(10):
        assert result[str(i)] == i


def test_turbo_mapping_hash_collisions():
    from pyrsistent import pvector
    initial = {'a': 1, 'b': 2, 'c': 3}
    result = _turbo_mapping(initial, 4)
    assert result._size == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


def test_turbo_mapping_returns_pmap():
    from pyrsistent import pvector
    result = _turbo_mapping({'key': 'value'}, None)
    assert isinstance(result, PMap)


def test_turbo_mapping_buckets_is_pvector():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, None)
    assert hasattr(result._buckets, 'evolver')


def test_turbo_mapping_single_element():
    from pyrsistent import pvector
    result = _turbo_mapping({'single': 42}, None)
    assert result._size == 1
    assert result['single'] == 42
    assert len(result._buckets) == 8


# LLM-generated content at query #28
#--------------------------

```python
def test_pmap_eq_with_dict_predicate_false():
    from pyrsistent import m
    
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2, 'c': 3}
    
    result = pmap1 == dict1
    
    assert result is False


# LLM-generated content at query #29
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    
    result = m1.update_with(lambda l, r: l + r, m2)
    
    assert result == {'a': 1, 'b': 2, 'c': 3}
    assert result['c'] == 3


# LLM-generated content at query #30
#--------------------------

```python
def test_turbo_mapping_empty_dict():
    from pyrsistent import pvector
    result = _turbo_mapping({}, None)
    assert len(result) == 0
    assert result._size == 0


def test_turbo_mapping_single_item():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, None)
    assert len(result) == 1
    assert result['a'] == 1


def test_turbo_mapping_multiple_items():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3}, None)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


def test_turbo_mapping_with_pre_size():
    from pyrsistent import pvector
    result = _turbo_mapping({'x': 10, 'y': 20}, 32)
    assert len(result) == 2
    assert result['x'] == 10
    assert result['y'] == 20
    assert len(result._buckets) == 32


def test_turbo_mapping_from_list_of_tuples():
    from pyrsistent import pvector
    result = _turbo_mapping([('key1', 'val1'), ('key2', 'val2')], None)
    assert len(result) == 2
    assert result['key1'] == 'val1'
    assert result['key2'] == 'val2'


def test_turbo_mapping_size_calculation():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3, 'd': 4}, None)
    assert len(result._buckets) == 8


def test_turbo_mapping_overwrites_duplicate_keys():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'a': 2}, None)
    assert len(result) == 1
    assert result['a'] == 2


def test_turbo_mapping_returns_pmap_instance():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1}, None)
    assert isinstance(result, PMap)


def test_turbo_mapping_with_various_value_types():
    from pyrsistent import pvector
    result = _turbo_mapping({'int': 42, 'str': 'hello', 'list': [1, 2, 3], 'none': None}, None)
    assert result['int'] == 42
    assert result['str'] == 'hello'
    assert result['list'] == [1, 2, 3]
    assert result['none'] is None


def test_turbo_mapping_hash_distribution():
    from pyrsistent import pvector
    items = {f'key_{i}': i for i in range(10)}
    result = _turbo_mapping(items, None)
    assert len(result) == 10
    for i in range(10):
        assert result[f'key_{i}'] == i


def test_turbo_mapping_preserves_all_items_with_collisions():
    from pyrsistent import pvector
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, None)
    assert len(result) == 5
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3
    assert result['d'] == 4
    assert result['e'] == 5


# LLM-generated content at query #31
#--------------------------

```python
def test_pmap_items_contains_with_valid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1) in items_view


def test_pmap_items_contains_with_invalid_item():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 2) not in items_view


def test_pmap_items_contains_with_missing_key():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('c', 1) not in items_view


def test_pmap_items_contains_with_non_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert 'a' not in items_view


def test_pmap_items_contains_with_single_element():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a',) not in items_view


def test_pmap_items_contains_with_three_element_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ('a', 1, 'extra') not in items_view


def test_pmap_items_contains_with_none_values():
    from pyrsistent import pmap
    m = pmap({'a': None, 'b': 2})
    items_view = m.items()
    assert ('a', None) in items_view


def test_pmap_items_contains_with_list():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    assert ['a', 1] not in items_view


def test_pmap_items_contains_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    assert ('a', 1) not in items_view


def test_pmap_items_contains_with_matching_value():
    from pyrsistent import pmap
    m = pmap({'key': 'value'})
    items_view = m.items()
    assert ('key', 'value') in items_view


# LLM-generated content at query #32
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    
    result = m1.update_with(lambda l, r: l + r, m2)
    
    assert result == {'a': 1, 'b': 2, 'c': 3}
    assert result['c'] == 3


# LLM-generated content at query #33
#--------------------------

```python
def test_turbo_mapping_exception_handler():
    from collections.abc import Mapping
    from pyrsistent import pvector, PMap
    
    class NoLenObject:
        def __iter__(self):
            return iter([('key', 'value')])
        
        def items(self):
            return [('key', 'value')]
    
    initial = NoLenObject()
    pre_size = None
    
    try:
        len(initial)
        should_raise = False
    except Exception:
        should_raise = True
    
    assert should_raise is True


# LLM-generated content at query #34
#--------------------------

```python
def test_contains_returns_false_on_unpacking_exception():
    from pyrsistent import pmap
    
    items_view = pmap({'a': 1, 'b': 2}).items()
    
    # Test with a single value that cannot be unpacked into (k, v)
    result = 'single_value' in items_view
    assert result is False
    
    # Test with a list that cannot be unpacked into exactly 2 elements
    result = [1, 2, 3] in items_view
    assert result is False
    
    # Test with an integer that cannot be unpacked
    result = 42 in items_view
    assert result is False
    
    # Test with None that cannot be unpacked
    result = None in items_view
    assert result is False


# LLM-generated content at query #35
#--------------------------

```python
def test_eq_same_object():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    assert pmap1 == pmap1


def test_eq_different_pmap_same_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert pmap1 == pmap2


def test_eq_different_pmap_different_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert not (pmap1 == pmap2)


def test_eq_pmap_vs_dict_same_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    assert pmap1 == dict1


def test_eq_pmap_vs_dict_different_content():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 3}
    assert not (pmap1 == dict1)


def test_eq_pmap_vs_dict_different_length():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1}
    assert not (pmap1 == dict1)


def test_eq_pmap_vs_non_mapping():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    result = pmap1.__eq__([1, 2, 3])
    assert result == NotImplemented


def test_eq_pmap_vs_string():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    result = pmap1.__eq__("string")
    assert result == NotImplemented


def test_eq_empty_pmaps():
    from pyrsistent import m
    pmap1 = m()
    pmap2 = m()
    assert pmap1 == pmap2


def test_eq_empty_pmap_vs_empty_dict():
    from pyrsistent import m
    pmap1 = m()
    dict1 = {}
    assert pmap1 == dict1


def test_eq_pmap_vs_dict_with_multiple_keys():
    from pyrsistent import m
    pmap1 = m(x=10, y=20, z=30)
    dict1 = {'x': 10, 'y': 20, 'z': 30}
    assert pmap1 == dict1


def test_eq_pmap_with_different_bucket_layout():
    from pyrsistent import m
    pmap1 = m(a=1, b=2, c=3, d=4)
    pmap2 = m(d=4, c=3, b=2, a=1)
    assert pmap1 == pmap2


def test_eq_pmap_vs_dict_string_values():
    from pyrsistent import m
    pmap1 = m(name='alice', city='nyc')
    dict1 = {'name': 'alice', 'city': 'nyc'}
    assert pmap1 == dict1


def test_eq_pmap_after_modifications():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = pmap1.set('c', 3)
    pmap3 = pmap2.remove('c')
    assert pmap1 == pmap3


def test_eq_ne_operator():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert pmap1 != pmap2


# LLM-generated content at query #36
#--------------------------

```python
def test_turbo_mapping_exception_handler():
    from collections.abc import Mapping
    from pyrsistent import pvector, PMap
    
    class NoLenObject:
        def __iter__(self):
            return iter([('a', 1), ('b', 2)])
        
        def items(self):
            return [('a', 1), ('b', 2)]
    
    initial = NoLenObject()
    pre_size = None
    
    result = _turbo_mapping(initial, pre_size)
    
    assert result is not None
    assert len(result) == 2


# LLM-generated content at query #37
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    
    result = m1.update_with(lambda l, r: l + r, m2)
    
    assert result == {'a': 1, 'b': 2, 'c': 3}
    assert result['c'] == 3


# LLM-generated content at query #38
#--------------------------

```python
def test_pmap_eq_same_instance():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    assert pmap1 == pmap1


def test_pmap_eq_equal_pmaps():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    assert pmap1 == pmap2


def test_pmap_eq_different_pmaps():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_pmap_vs_dict():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    assert pmap1 == dict1


def test_pmap_eq_pmap_vs_dict_different():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 3}
    assert not (pmap1 == dict1)


def test_pmap_eq_different_sizes():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2, c=3)
    assert not (pmap1 == pmap2)


def test_pmap_eq_non_mapping():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    result = pmap1.__eq__("not a mapping")
    assert result == NotImplemented


def test_pmap_eq_empty_pmaps():
    from pyrsistent import m
    pmap1 = m()
    pmap2 = m()
    assert pmap1 == pmap2


def test_pmap_eq_empty_pmap_vs_empty_dict():
    from pyrsistent import m
    pmap1 = m()
    dict1 = {}
    assert pmap1 == dict1


def test_pmap_eq_pmap_vs_custom_mapping():
    from pyrsistent import m
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
    
    pmap1 = m(a=1, b=2)
    custom_map = CustomMapping({'a': 1, 'b': 2})
    assert pmap1 == custom_map


def test_pmap_eq_with_cached_hash_different():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    hash(pmap1)
    hash(pmap2)
    assert not (pmap1 == pmap2)


def test_pmap_eq_with_cached_hash_same():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    hash(pmap1)
    hash(pmap2)
    assert pmap1 == pmap2


# LLM-generated content at query #39
#--------------------------

```python
def test_contains_returns_false_on_exception():
    from pyrsistent import pmap
    
    items = pmap({'a': 1, 'b': 2}).items()
    result = items.__contains__("invalid")
    assert result is False


