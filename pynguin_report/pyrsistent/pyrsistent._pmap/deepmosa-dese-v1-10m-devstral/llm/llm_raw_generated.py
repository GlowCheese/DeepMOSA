####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_pmap_equality_with_itself():
    m = pmap(a=1, b=2)
    assert m == m

def test_pmap_equality_with_other_pmap():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    assert m1 == m2

def test_pmap_equality_with_dict():
    m = pmap(a=1, b=2)
    d = {'a': 1, 'b': 2}
    assert m == d

def test_pmap_equality_with_other_mapping():
    m = pmap(a=1, b=2)
    from collections import OrderedDict
    od = OrderedDict([('a', 1), ('b', 2)])
    assert m == od

def test_pmap_inequality_different_sizes():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2, c=3)
    assert not (m1 == m2)

def test_pmap_inequality_different_values():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=3)
    assert not (m1 == m2)

def test_pmap_inequality_different_keys():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, c=2)
    assert not (m1 == m2)

def test_pmap_equality_with_cached_hash():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 42
    m2._cached_hash = 42
    assert m1 == m2

def test_pmap_inequality_with_different_cached_hash():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 42
    m2._cached_hash = 43
    assert not (m1 == m2)

def test_pmap_equality_with_same_buckets():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m2._buckets = m1._buckets
    assert m1 == m2

def test_pmap_inequality_with_non_mapping():
    m = pmap(a=1, b=2)
    assert not (m == "not a mapping")

def test_pmap_not_implemented_for_non_mapping():
    m = pmap(a=1, b=2)
    assert m.__eq__("not a mapping") == NotImplemented


# LLM-generated content at query #2
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert len(pmap_instance._buckets) == 3
    assert pmap_instance._buckets[1] == [('a', 1)]
    assert pmap_instance._buckets[2] == [('b', 2)]


# LLM-generated content at query #3
#--------------------------

```python
def test_getattr_existing_key():
    m = pmap(a=1, b=2)
    assert m.a == 1
    assert m.b == 2

def test_getattr_non_existing_key():
    m = pmap(a=1, b=2)
    try:
        _ = m.c
        assert False, "Expected AttributeError"
    except AttributeError as e:
        assert str(e) == "PMap has no attribute 'c'"


# LLM-generated content at query #4
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


# LLM-generated content at query #5
#--------------------------

```python
def test_eq_same_instance():
    pmap_items = PMapItems({1: 2})
    assert pmap_items == pmap_items

def test_eq_different_instance_same_map():
    pmap_items1 = PMapItems({1: 2})
    pmap_items2 = PMapItems({1: 2})
    assert pmap_items1 == pmap_items2

def test_eq_different_instance_different_map():
    pmap_items1 = PMapItems({1: 2})
    pmap_items2 = PMapItems({3: 4})
    assert not (pmap_items1 == pmap_items2)

def test_eq_non_pmapitems_instance():
    pmap_items = PMapItems({1: 2})
    assert not (pmap_items == {1: 2})


# LLM-generated content at query #6
#--------------------------

```python
def test_eq_with_non_dict_mapping():
    pmap_instance = pmap({'a': 1, 'b': 2})
    non_dict_mapping = NonDictMapping({'a': 1, 'b': 2})
    assert pmap_instance == non_dict_mapping


# LLM-generated content at query #7
#--------------------------

```python
def test_eq_same_instance():
    m = pmap(a=1, b=2)
    assert m == m

def test_eq_different_pmaps_same_content():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    assert m1 == m2

def test_eq_different_pmaps_different_content():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=3)
    assert not m1 == m2

def test_eq_with_dict_same_content():
    m = pmap(a=1, b=2)
    d = {'a': 1, 'b': 2}
    assert m == d

def test_eq_with_dict_different_content():
    m = pmap(a=1, b=2)
    d = {'a': 1, 'b': 3}
    assert not m == d

def test_eq_with_other_mapping_same_content():
    m = pmap(a=1, b=2)
    from collections import OrderedDict
    od = OrderedDict([('a', 1), ('b', 2)])
    assert m == od

def test_eq_with_other_mapping_different_content():
    m = pmap(a=1, b=2)
    from collections import OrderedDict
    od = OrderedDict([('a', 1), ('b', 3)])
    assert not m == od

def test_eq_with_non_mapping():
    m = pmap(a=1, b=2)
    assert not m == "not a mapping"

def test_eq_with_different_sized_pmaps():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2, c=3)
    assert not m1 == m2

def test_eq_with_different_sized_dict():
    m = pmap(a=1, b=2)
    d = {'a': 1, 'b': 2, 'c': 3}
    assert not m == d

def test_eq_with_cached_hash_same():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 123
    m2._cached_hash = 123
    assert m1 == m2

def test_eq_with_cached_hash_different():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 123
    m2._cached_hash = 456
    assert not m1 == m2

def test_eq_with_same_buckets():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m2._buckets = m1._buckets
    assert m1 == m2


# LLM-generated content at query #8
#--------------------------

```python
def test_eq_predicate_line_15():
    pmap_instance = PMap(0, pvector())
    dict_instance = {}
    assert dict(pmap_instance.iteritems()) == dict_instance


# LLM-generated content at query #9
#--------------------------

```python
def test_pmap_equality_same_instance():
    m1 = m(a=1, b=2)
    assert m1 == m1

def test_pmap_equality_different_instances_same_content():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    assert m1 == m2

def test_pmap_equality_different_content():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    assert not m1 == m2

def test_pmap_equality_with_dict():
    m1 = m(a=1, b=2)
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

def test_pmap_equality_with_different_length():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2, c=3)
    assert not m1 == m2

def test_pmap_equality_with_non_mapping():
    m1 = m(a=1, b=2)
    assert not m1 == "not a mapping"

def test_pmap_equality_with_cached_hash():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = 42
    m2._cached_hash = 42
    assert m1 == m2

def test_pmap_equality_with_different_cached_hash():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = 42
    m2._cached_hash = 43
    assert not m1 == m2

def test_pmap_equality_with_same_buckets():
    m1 = m(a=1, b=2)
    m2 = PMap(m1._size, m1._buckets)
    assert m1 == m2


# LLM-generated content at query #10
#--------------------------

```python
def test_pmapitems_contains_existing_item():
    pmap = PMap({'a': 1, 'b': 2})
    view = PMapItems(pmap)
    assert ('a', 1) in view
    assert ('b', 2) in view

def test_pmapitems_contains_non_existing_item():
    pmap = PMap({'a': 1, 'b': 2})
    view = PMapItems(pmap)
    assert ('c', 3) not in view
    assert ('a', 2) not in view

def test_pmapitems_contains_invalid_arg():
    pmap = PMap({'a': 1, 'b': 2})
    view = PMapItems(pmap)
    assert 1 not in view
    assert 'a' not in view
    assert (1, 2, 3) not in view


# LLM-generated content at query #11
#--------------------------

```python
def test_contains_returns_false_for_non_tuple_arg():
    pmap_items = PMapItems({})
    assert not (object() in pmap_items)


# LLM-generated content at query #12
#--------------------------

```python
def test_buckets_not_equal():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    assert m1._buckets != m2._buckets


# LLM-generated content at query #13
#--------------------------

```python
def test__turbo_mapping_empty():
    result = _turbo_mapping({}, None)
    assert len(result) == 0

def test__turbo_mapping_with_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, 8)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_without_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, None)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_with_list():
    result = _turbo_mapping([('a', 1), ('b', 2)], None)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_with_large_input():
    large_dict = {i: str(i) for i in range(100)}
    result = _turbo_mapping(large_dict, None)
    assert len(result) == 100
    for i in range(100):
        assert result[i] == str(i)


# LLM-generated content at query #14
#--------------------------

```python
def test__turbo_mapping_empty_dict():
    result = _turbo_mapping({}, None)
    assert result == PMap(0, pvector().extend(8 * [None]))

def test__turbo_mapping_with_pre_size():
    result = _turbo_mapping({'a': 1}, 4)
    assert len(result) == 1
    assert result['a'] == 1

def test__turbo_mapping_without_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, None)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_with_list_of_tuples():
    result = _turbo_mapping([('a', 1), ('b', 2)], None)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_with_large_input():
    large_dict = {f'key_{i}': i for i in range(100)}
    result = _turbo_mapping(large_dict, None)
    assert len(result) == 100
    for i in range(100):
        assert result[f'key_{i}'] == i


# LLM-generated content at query #15
#--------------------------

```python
def test_pmap_equality_same_instance():
    m1 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    assert m1 == m1

def test_pmap_equality_different_pmaps_same_content():
    m1 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    m2 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    assert m1 == m2

def test_pmap_equality_different_pmaps_different_content():
    m1 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    m2 = PMap(2, [(None, [('a', 1), ('b', 3)])])
    assert m1 != m2

def test_pmap_equality_with_dict_same_content():
    m1 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    d1 = {'a': 1, 'b': 2}
    assert m1 == d1

def test_pmap_equality_with_dict_different_content():
    m1 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    d1 = {'a': 1, 'b': 3}
    assert m1 != d1

def test_pmap_equality_with_mapping_same_content():
    from collections import OrderedDict
    m1 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    od1 = OrderedDict([('a', 1), ('b', 2)])
    assert m1 == od1

def test_pmap_equality_with_mapping_different_content():
    from collections import OrderedDict
    m1 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    od1 = OrderedDict([('a', 1), ('b', 3)])
    assert m1 != od1

def test_pmap_equality_different_sizes():
    m1 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    m2 = PMap(3, [(None, [('a', 1), ('b', 2), ('c', 3)])])
    assert m1 != m2

def test_pmap_equality_with_non_mapping():
    m1 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    assert m1 != "not a mapping"
    assert m1 != 42
    assert m1 != [1, 2, 3]


# LLM-generated content at query #16
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


# LLM-generated content at query #17
#--------------------------

```python
def test_cached_hash_comparison():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = 1
    m2._cached_hash = 2
    assert m1 != m2


# LLM-generated content at query #18
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


# LLM-generated content at query #19
#--------------------------

```python
def test_buckets_not_equal():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    assert m1._buckets != m2._buckets


# LLM-generated content at query #20
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_false():
    initial = [1, 2, 3]
    pre_size = 0
    result = _turbo_mapping(initial, pre_size)
    assert result is not None


# LLM-generated content at query #21
#--------------------------

```python
def test_update_with_basic_merge():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2, c=3))
    assert m2 == {'a': 3, 'b': 2, 'c': 3}

def test_update_with_keep_leftmost():
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m2 == {'a': 1}

def test_update_with_keep_rightmost():
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert m2 == {'a': 3}

def test_update_with_multiple_maps():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l * r, m(a=2, c=3), {'a': 3, 'd': 4})
    assert m2 == {'a': 6, 'b': 2, 'c': 3, 'd': 4}

def test_update_with_no_overlap():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(c=3), {'d': 4})
    assert m2 == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

def test_update_with_empty_map():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m())
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_empty_original():
    m1 = m()
    m2 = m1.update_with(lambda l, r: l + r, m(a=1), {'b': 2})
    assert m2 == {'a': 1, 'b': 2}


# LLM-generated content at query #22
#--------------------------

```python
def test_contains_returns_true_for_existing_key_value_pair():
    pmap_items = PMapItems({"a": 1, "b": 2})
    assert ("a", 1) in pmap_items


# LLM-generated content at query #23
#--------------------------

```python
def test_equality_with_non_dict_mapping():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    assert m1 == m2
    assert m1 == {'a': 1, 'b': 2}
    assert not (m1 == [('a', 1), ('b', 2)])


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_false():
    assert not (2 * len([]) or 8)


# LLM-generated content at query #25
#--------------------------

```python
def test_contains_returns_false_for_invalid_arg():
    pmap_items = PMapItems({})
    assert not (object() in pmap_items)


# LLM-generated content at query #26
#--------------------------

```python
def test_contains_with_valid_key_value_pair():
    pmap_items = PMapItems(PMap({1: 2}))
    assert (1, 2) in pmap_items


# LLM-generated content at query #27
#--------------------------

```python
def test_contains_with_valid_item():
    pmap_items = PMapItems(PMap({1: 2, 3: 4}))
    assert (1, 2) in pmap_items


# LLM-generated content at query #28
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(b=2))
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #29
#--------------------------

```python
def test_contains_returns_false_for_non_tuple_arg():
    pmap_items = PMapItems({})
    assert not (object() in pmap_items)
    assert not ("key" in pmap_items)
    assert not (1 in pmap_items)
    assert not (None in pmap_items)


# LLM-generated content at query #30
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(b=2))
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #31
#--------------------------

```python
def test_cached_hash_comparison_returns_false_when_different():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 10
    m2._cached_hash = 20
    assert m1 != m2


# LLM-generated content at query #32
#--------------------------

```python
def test_pmapitems_contains_with_valid_item():
    pmap = PMap({'a': 1, 'b': 2})
    items_view = PMapItems(pmap)
    assert ('a', 1) in items_view


# LLM-generated content at query #33
#--------------------------

```python
def test_update_with_predicate_false():
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(b=2))
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #34
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


# LLM-generated content at query #35
#--------------------------

```python
def test__turbo_mapping_empty_dict():
    result = _turbo_mapping({}, None)
    assert len(result) == 0

def test__turbo_mapping_with_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, 16)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_no_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, None)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_large_dict():
    initial = {f'key_{i}': i for i in range(100)}
    result = _turbo_mapping(initial, None)
    assert len(result) == 100
    for i in range(100):
        assert result[f'key_{i}'] == i

def test__turbo_mapping_list_of_tuples():
    initial = [('a', 1), ('b', 2), ('c', 3)]
    result = _turbo_mapping(initial, None)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3

def test__turbo_mapping_non_mapping_iterable():
    initial = [('a', 1), ('b', 2), ('c', 3)]
    result = _turbo_mapping(initial, None)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3

def test__turbo_mapping_single_item():
    result = _turbo_mapping({'a': 1}, None)
    assert len(result) == 1
    assert result['a'] == 1

def test__turbo_mapping_collision_handling():
    # Create keys that will collide in a small bucket size
    initial = {i: str(i) for i in range(10)}
    result = _turbo_mapping(initial, 5)
    assert len(result) == 10
    for i in range(10):
        assert result[i] == str(i)


# LLM-generated content at query #36
#--------------------------

```python
def test_pmap_eq_same_instance():
    m1 = pmap(a=1, b=2)
    assert m1 == m1

def test_pmap_eq_different_pmaps_same_content():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    assert m1 == m2

def test_pmap_eq_with_dict():
    m = pmap(a=1, b=2)
    d = {'a': 1, 'b': 2}
    assert m == d

def test_pmap_eq_with_other_mapping():
    m = pmap(a=1, b=2)
    from collections import OrderedDict
    od = OrderedDict([('a', 1), ('b', 2)])
    assert m == od

def test_pmap_eq_different_sizes():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2, c=3)
    assert not (m1 == m2)

def test_pmap_eq_different_content():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=3)
    assert not (m1 == m2)

def test_pmap_eq_with_non_mapping():
    m = pmap(a=1, b=2)
    assert not (m == "not a mapping")

def test_pmap_eq_with_cached_hash():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 42
    m2._cached_hash = 42
    assert m1 == m2

def test_pmap_eq_with_different_cached_hash():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 42
    m2._cached_hash = 43
    assert not (m1 == m2)

def test_pmap_eq_with_same_buckets():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._buckets = m2._buckets
    assert m1 == m2


# LLM-generated content at query #37
#--------------------------

```python
def test_cached_hash_inequality():
    pmap1 = pmap(a=1, b=2)
    pmap2 = pmap(a=1, b=2)
    pmap1._cached_hash = 1
    pmap2._cached_hash = 2
    assert pmap1 != pmap2


# LLM-generated content at query #38
#--------------------------

```python
def test_update_with_predicate_false():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l, m(c=3))
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #39
#--------------------------

```python
def test_pmapitems_contains_existing_item():
    m = PMap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

def test_pmapitems_contains_non_existing_item():
    m = PMap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('c', 3) not in view
    assert ('a', 2) not in view

def test_pmapitems_contains_invalid_arg():
    m = PMap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert 1 not in view
    assert 'a' not in view
    assert (1, 2, 3) not in view


# LLM-generated content at query #40
#--------------------------

```python
def test_contains_returns_false_for_non_tuple_arg():
    pmap_items = PMapItems({})
    assert not (object() in pmap_items)


# LLM-generated content at query #41
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


# LLM-generated content at query #42
#--------------------------

```python
def test_eq_with_different_cached_hash():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 1
    m2._cached_hash = 2
    assert m1 != m2


# LLM-generated content at query #43
#--------------------------

```python
def test_pmapitems_contains_existing_item():
    m = PMap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

def test_pmapitems_contains_non_existing_item():
    m = PMap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 2) not in view
    assert ('c', 3) not in view

def test_pmapitems_contains_invalid_arg():
    m = PMap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert 1 not in view
    assert 'a' not in view
    assert (1, 2, 3) not in view


# LLM-generated content at query #44
#--------------------------

```python
def test__turbo_mapping_with_empty_dict():
    result = _turbo_mapping({}, None)
    assert len(result) == 0
    assert dict(result) == {}

def test__turbo_mapping_with_non_empty_dict():
    initial = {'a': 1, 'b': 2, 'c': 3}
    result = _turbo_mapping(initial, None)
    assert len(result) == 3
    assert dict(result) == initial

def test__turbo_mapping_with_pre_size():
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, 16)
    assert len(result) == 2
    assert dict(result) == initial

def test__turbo_mapping_with_list_of_tuples():
    initial = [('a', 1), ('b', 2), ('c', 3)]
    result = _turbo_mapping(initial, None)
    assert len(result) == 3
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3}

def test__turbo_mapping_with_colliding_hashes():
    class HashCollision:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            return 1

        def __eq__(self, other):
            return isinstance(other, HashCollision) and self.value == other.value

    initial = {HashCollision('a'): 1, HashCollision('b'): 2}
    result = _turbo_mapping(initial, None)
    assert len(result) == 2
    assert result[HashCollision('a')] == 1
    assert result[HashCollision('b')] == 2


# LLM-generated content at query #45
#--------------------------

```python
def test_update_with_single_map():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=3, c=4))
    assert m2 == {'a': 4, 'b': 2, 'c': 4}

def test_update_with_multiple_maps():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=3), m(a=5, c=4))
    assert m2 == {'a': 9, 'b': 2, 'c': 4}

def test_update_with_no_overlap():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(c=3, d=4))
    assert m2 == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

def test_update_with_empty_map():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m())
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_keep_leftmost():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m(a=3, c=4))
    assert m2 == {'a': 1, 'b': 2, 'c': 4}

def test_update_with_keep_rightmost():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r, m(a=3, c=4))
    assert m2 == {'a': 3, 'b': 2, 'c': 4}

def test_update_with_complex_merge():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l * r, m(a=3, c=4))
    assert m2 == {'a': 3, 'b': 2, 'c': 4}

def test_update_with_dict_arguments():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, {'a': 3, 'c': 4})
    assert m2 == {'a': 4, 'b': 2, 'c': 4}

def test_update_with_mixed_arguments():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=3), {'c': 4})
    assert m2 == {'a': 4, 'b': 2, 'c': 4}


# LLM-generated content at query #46
#--------------------------

```python
def test_pmap_equality_same_instance():
    m = pmap(a=1, b=2)
    assert m == m

def test_pmap_equality_different_instances_same_content():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    assert m1 == m2

def test_pmap_equality_different_content():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=3)
    assert m1 != m2

def test_pmap_equality_with_dict():
    m = pmap(a=1, b=2)
    d = {'a': 1, 'b': 2}
    assert m == d

def test_pmap_equality_with_different_dict():
    m = pmap(a=1, b=2)
    d = {'a': 1, 'b': 3}
    assert m != d

def test_pmap_equality_with_non_mapping():
    m = pmap(a=1, b=2)
    assert m != [1, 2, 3]

def test_pmap_equality_different_sizes():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2, c=3)
    assert m1 != m2

def test_pmap_equality_with_cached_hash():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 123
    m2._cached_hash = 123
    assert m1 == m2

def test_pmap_equality_with_different_cached_hash():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 123
    m2._cached_hash = 456
    assert m1 != m2

def test_pmap_equality_same_buckets():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m2._buckets = m1._buckets
    assert m1 == m2


# LLM-generated content at query #47
#--------------------------

```python
def test_contains_returns_true_for_existing_key_value_pair():
    pmap_items = PMapItems({"a": 1, "b": 2})
    assert ("a", 1) in pmap_items


# LLM-generated content at query #48
#--------------------------

```python
def test__turbo_mapping_with_empty_dict():
    result = _turbo_mapping({}, None)
    assert len(result) == 0

def test__turbo_mapping_with_non_empty_dict():
    result = _turbo_mapping({'a': 1, 'b': 2}, None)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_with_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, 16)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_with_list_of_tuples():
    result = _turbo_mapping([('a', 1), ('b', 2)], None)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_with_large_input():
    large_dict = {f'key_{i}': i for i in range(100)}
    result = _turbo_mapping(large_dict, None)
    assert len(result) == 100
    for i in range(100):
        assert result[f'key_{i}'] == i

def test__turbo_mapping_with_collision():
    class CollisionKey:
        def __init__(self, value):
            self.value = value
        def __hash__(self):
            return 1
        def __eq__(self, other):
            return isinstance(other, CollisionKey) and self.value == other.value

    keys = [CollisionKey(i) for i in range(10)]
    result = _turbo_mapping({k: k.value for k in keys}, None)
    assert len(result) == 10
    for k in keys:
        assert result[k] == k.value


# LLM-generated content at query #49
#--------------------------

```python
def test_contains_returns_true_for_existing_key_value_pair():
    pmap_items = PMapItems(PMap({1: 2}))
    assert (1, 2) in pmap_items


# LLM-generated content at query #50
#--------------------------

```python
def test_pmapitems_contains_existing_item():
    m = PMap(a=1, b=2)
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

def test_pmapitems_contains_nonexistent_item():
    m = PMap(a=1, b=2)
    view = PMapItems(m)
    assert ('c', 3) not in view
    assert ('a', 2) not in view

def test_pmapitems_contains_invalid_arg():
    m = PMap(a=1, b=2)
    view = PMapItems(m)
    assert 1 not in view
    assert 'a' not in view
    assert (1, 2, 3) not in view


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__turbo_mapping_empty():
    result = _turbo_mapping({}, None)
    assert result == PMap(0, pvector().extend(8 * [None]))

def test__turbo_mapping_with_pre_size():
    result = _turbo_mapping({'a': 1}, 16)
    assert len(result._buckets) == 16
    assert result._size == 1
    assert result['a'] == 1

def test__turbo_mapping_without_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, None)
    assert len(result._buckets) == 8
    assert result._size == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_non_mapping_input():
    result = _turbo_mapping([('a', 1), ('b', 2)], None)
    assert result._size == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_collision_handling():
    # Force a collision by using keys with the same hash modulo bucket size
    class CollidingKey:
        def __init__(self, val):
            self.val = val
        def __hash__(self):
            return 1
        def __eq__(self, other):
            return isinstance(other, CollidingKey) and self.val == other.val

    result = _turbo_mapping({CollidingKey(1): 'a', CollidingKey(2): 'b'}, 4)
    assert result._size == 2
    assert result[CollidingKey(1)] == 'a'
    assert result[CollidingKey(2)] == 'b'


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test__turbo_mapping_with_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, 4)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_without_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, None)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_with_empty_initial():
    result = _turbo_mapping({}, None)
    assert len(result) == 0

def test__turbo_mapping_with_non_mapping_initial():
    result = _turbo_mapping([('a', 1), ('b', 2)], None)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_with_large_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, 100)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #2
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


# LLM-generated content at query #3
#--------------------------

```python
def test_pmap_equality_with_identical_pmap():
    m1 = PMap(2, [(None, None), [('a', 1), ('b', 2)]])
    m2 = PMap(2, [(None, None), [('a', 1), ('b', 2)]])
    assert m1 == m2

def test_pmap_equality_with_different_pmap():
    m1 = PMap(2, [(None, None), [('a', 1), ('b', 2)]])
    m2 = PMap(2, [(None, None), [('a', 1), ('c', 2)]])
    assert not (m1 == m2)

def test_pmap_equality_with_dict():
    m = PMap(2, [(None, None), [('a', 1), ('b', 2)]])
    d = {'a': 1, 'b': 2}
    assert m == d

def test_pmap_equality_with_different_dict():
    m = PMap(2, [(None, None), [('a', 1), ('b', 2)]])
    d = {'a': 1, 'c': 2}
    assert not (m == d)

def test_pmap_equality_with_same_reference():
    m = PMap(2, [(None, None), [('a', 1), ('b', 2)]])
    assert m == m

def test_pmap_equality_with_different_size():
    m1 = PMap(2, [(None, None), [('a', 1), ('b', 2)]])
    m2 = PMap(3, [(None, None), [('a', 1), ('b', 2), ('c', 3)]])
    assert not (m1 == m2)

def test_pmap_equality_with_non_mapping():
    m = PMap(2, [(None, None), [('a', 1), ('b', 2)]])
    assert m != "not a mapping"


# LLM-generated content at query #4
#--------------------------

```python
def test_pmapitems_contains_existing_item():
    pmap = PMap({'a': 1, 'b': 2})
    view = PMapItems(pmap)
    assert ('a', 1) in view

def test_pmapitems_contains_non_existing_item():
    pmap = PMap({'a': 1, 'b': 2})
    view = PMapItems(pmap)
    assert ('c', 3) not in view

def test_pmapitems_contains_wrong_value():
    pmap = PMap({'a': 1, 'b': 2})
    view = PMapItems(pmap)
    assert ('a', 2) not in view

def test_pmapitems_contains_invalid_arg():
    pmap = PMap({'a': 1, 'b': 2})
    view = PMapItems(pmap)
    assert 'a' not in view


# LLM-generated content at query #5
#--------------------------

```python
def test_pmap_equality_with_different_cached_hash():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 1
    m2._cached_hash = 2
    assert m1 != m2


# LLM-generated content at query #6
#--------------------------

```python
def test_cached_hash_comparison():
    m1 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    m2 = PMap(2, [(None, [('a', 1), ('b', 2)])])
    m1._cached_hash = 42
    m2._cached_hash = 43
    assert m1 != m2


# LLM-generated content at query #7
#--------------------------

```python
def test_eq_predicate_line_15_false():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    assert not (m1._buckets == m2._buckets)


# LLM-generated content at query #8
#--------------------------

```python
def test_eq_with_non_dict_mapping():
    m = pmap(a=1, b=2)
    class CustomMapping:
        def __len__(self):
            return 2
        def __iter__(self):
            return iter(['a', 'b'])
        def __getitem__(self, key):
            if key == 'a':
                return 1
            elif key == 'b':
                return 2
            raise KeyError(key)
        def items(self):
            return [('a', 1), ('b', 2)]
    custom = CustomMapping()
    assert m == custom


# LLM-generated content at query #9
#--------------------------

```python
def test_update_with_basic_merge():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l + r, m(a=3, c=4))
    assert result == {'a': 4, 'b': 2, 'c': 4}

def test_update_with_left_priority():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l, m(a=3, c=4))
    assert result == {'a': 1, 'b': 2, 'c': 4}

def test_update_with_right_priority():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(a=3, c=4))
    assert result == {'a': 3, 'b': 2, 'c': 4}

def test_update_with_multiple_maps():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l + r, m(a=3), m(a=5, d=6))
    assert result == {'a': 9, 'b': 2, 'd': 6}

def test_update_with_no_overlap():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l + r, m(c=3, d=4))
    assert result == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

def test_update_with_empty_map():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l + r, m())
    assert result == {'a': 1, 'b': 2}

def test_update_with_empty_original():
    m1 = m()
    result = m1.update_with(lambda l, r: l + r, m(a=1, b=2))
    assert result == {'a': 1, 'b': 2}

def test_update_with_string_concatenation():
    m1 = m(a='hello', b='world')
    result = m1.update_with(lambda l, r: l + r, m(a=' there', c='!'))
    assert result == {'a': 'hello there', 'b': 'world', 'c': '!'}

def test_update_with_list_concatenation():
    m1 = m(a=[1, 2], b=[3])
    result = m1.update_with(lambda l, r: l + r, m(a=[4, 5], c=[6]))
    assert result == {'a': [1, 2, 4, 5], 'b': [3], 'c': [6]}

def test_update_with_operator_add():
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=3, c=4))
    assert result == {'a': 4, 'b': 2, 'c': 4}

def test_update_with_operator_mul():
    from operator import mul
    m1 = m(a=2, b=3)
    result = m1.update_with(mul, m(a=3, c=4))
    assert result == {'a': 6, 'b': 3, 'c': 4}


# LLM-generated content at query #10
#--------------------------

```python
def test_equality_with_non_dict_mapping():
    from collections import OrderedDict
    m = pmap(a=1, b=2)
    od = OrderedDict([('a', 1), ('b', 2)])
    assert m == od


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_false():
    assert not (2 * len([]) or 8)


# LLM-generated content at query #12
#--------------------------

```python
def test_update_with_basic_merge():
    m1 = pmap(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, pmap(a=3))
    assert m2 == pmap(a=4, b=2)

def test_update_with_left_priority():
    m1 = pmap(a=1)
    m2 = m1.update_with(lambda l, r: l, pmap(a=2), {'a': 3})
    assert m2 == pmap(a=1)

def test_update_with_right_priority():
    m1 = pmap(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r, pmap(a=3, c=4), {'a': 5, 'd': 6})
    assert m2 == pmap(a=5, b=2, c=4, d=6)

def test_update_with_new_keys():
    m1 = pmap(a=1)
    m2 = m1.update_with(lambda l, r: l * r, pmap(b=2), {'c': 3})
    assert m2 == pmap(a=1, b=2, c=3)

def test_update_with_empty_map():
    m1 = pmap(a=1)
    m2 = m1.update_with(lambda l, r: l + r)
    assert m2 == pmap(a=1)

def test_update_with_multiple_maps():
    m1 = pmap(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, pmap(a=3, c=4), {'a': 5, 'd': 6}, pmap(b=7))
    assert m2 == pmap(a=9, b=9, c=4, d=6)


# LLM-generated content at query #13
#--------------------------

```python
def test_eq_with_different_cached_hash():
    m1 = pmap(a=1, b=2)
    m2 = pmap(a=1, b=2)
    m1._cached_hash = 1
    m2._cached_hash = 2
    assert not (m1 == m2)


# LLM-generated content at query #14
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(b=2))
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #15
#--------------------------

```python
def test_pmapitems_contains_existing_item():
    m = PMap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('a', 1) in view
    assert ('b', 2) in view

def test_pmapitems_contains_non_existing_item():
    m = PMap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert ('c', 3) not in view
    assert ('a', 2) not in view

def test_pmapitems_contains_invalid_arg():
    m = PMap({'a': 1, 'b': 2})
    view = PMapItems(m)
    assert 'a' not in view
    assert 1 not in view
    assert (('a', 1), ('b', 2)) not in view


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_false():
    assert not (2 * len([]) or 8)


# LLM-generated content at query #17
#--------------------------

```python
def test_update_with_predicate_false():
    m1 = m(a=1)
    assert m1.update_with(lambda l, r: l, m(a=2), {'a': 3}) == {'a': 1}


# LLM-generated content at query #18
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


# LLM-generated content at query #19
#--------------------------

```python
def test_contains_non_tuple_arg():
    pmap_items = PMapItems({})
    assert not (object() in pmap_items)


# LLM-generated content at query #20
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


# LLM-generated content at query #21
#--------------------------

```python
def test_contains_returns_false_on_exception():
    pmap_items = PMapItems({})
    assert not (object() in pmap_items)


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (2 * len([]) or 8)


# LLM-generated content at query #23
#--------------------------

```python
def test_eq_predicate_line_15():
    pmap_instance = PMap(2, [[('a', 1)], [('b', 2)]])
    dict_instance = {'a': 1, 'b': 2}
    assert isinstance(dict_instance, dict)
    assert not isinstance(dict_instance, PMap)
    assert pmap_instance == dict_instance


# LLM-generated content at query #24
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


# LLM-generated content at query #25
#--------------------------

```python
def test_eq_with_dict():
    m = pmap(a=1, b=2)
    d = {'a': 1, 'b': 2}
    assert m == d


# LLM-generated content at query #26
#--------------------------

```python
def test_contains_returns_false_for_non_tuple_arg():
    pmap_items = PMapItems({})
    assert False == (1 in pmap_items)


# LLM-generated content at query #27
#--------------------------

```python
def test_pmap_constructor_creates_instance_with_given_size_and_buckets():
    size = 2
    buckets = [None, [('a', 1)], [('b', 2)]]
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets


# LLM-generated content at query #28
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_false():
    assert not (2 * len([]) or 8)


# LLM-generated content at query #29
#--------------------------

```python
def test_update_with_predicate_false():
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(b=2))
    assert m1 == {'a': 1}
    assert m2 == {'a': 1, 'b': 2}


# LLM-generated content at query #30
#--------------------------

```python
def test_pmapitems_contains_non_tuple_arg():
    pmap_items = PMapItems({})
    assert not (object() in pmap_items)


# LLM-generated content at query #31
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(b=2))
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #32
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_false():
    assert not (2 * len([1, 2, 3]) or 8)


# LLM-generated content at query #33
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


# LLM-generated content at query #34
#--------------------------

```python
def test_eq_with_non_dict_mapping():
    m = m(a=1, b=2)
    class NonDictMapping:
        def __len__(self):
            return 2
        def __iter__(self):
            return iter(['a', 'b'])
        def __getitem__(self, key):
            if key == 'a':
                return 1
            elif key == 'b':
                return 2
            raise KeyError(key)
    assert m == NonDictMapping()


# LLM-generated content at query #35
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (2 * len([]) or 8)


# LLM-generated content at query #36
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


# LLM-generated content at query #37
#--------------------------

```python
def test_update_with_key_not_in_evolver():
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(b=2))
    assert m2 == {'a': 1, 'b': 2}


# LLM-generated content at query #38
#--------------------------

```python
def test_contains_returns_false_for_non_tuple_arg():
    pmap_items = PMapItems({})
    assert False == (1 in pmap_items)


# LLM-generated content at query #39
#--------------------------

```python
def test_eq_predicate_line_15():
    pmap_instance = PMap(1, pvector([[(1, 2)]]))
    dict_instance = {1: 2}
    assert not isinstance(dict_instance, PMap)
    assert isinstance(dict_instance, dict)


# LLM-generated content at query #40
#--------------------------

```python
def test_contains_returns_false_for_non_tuple_arg():
    pmap_items = PMapItems({})
    assert (5 in pmap_items) is False


# LLM-generated content at query #41
#--------------------------

```python
def test_pmap_not_equal_to_non_dict_mapping():
    class CustomMapping:
        def __init__(self, items):
            self._items = items

        def __getitem__(self, key):
            return self._items[key]

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            return len(self._items)

    m = pmap(a=1, b=2)
    custom_mapping = CustomMapping({'a': 1, 'b': 2})
    assert m != custom_mapping


# LLM-generated content at query #42
#--------------------------

```python
def test_predicate_at_line_7_evaluates_to_false():
    initial = [1, 2, 3]
    pre_size = 0
    size = 2 * len(initial) or 8
    assert size == 6


# LLM-generated content at query #43
#--------------------------

```python
def test_update_with_basic_merge():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=3, b=4))
    assert m2 == {'a': 4, 'b': 6}

def test_update_with_multiple_maps():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=3), m(b=4), m(a=5))
    assert m2 == {'a': 9, 'b': 6}

def test_update_with_new_key():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(c=3))
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_left_priority():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m(a=3, b=4))
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_right_priority():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r, m(a=3, b=4))
    assert m2 == {'a': 3, 'b': 4}

def test_update_with_empty_map():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m())
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_no_overlap():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(c=3, d=4))
    assert m2 == {'a': 1, 'b': 2, 'c': 3, 'd': 4}


# LLM-generated content at query #44
#--------------------------

```python
def test_pmap_constructor():
    pmap_instance = PMap(2, [None, [('a', 1)], [('b', 2)]])
    assert pmap_instance._size == 2
    assert pmap_instance._buckets == [None, [('a', 1)], [('b', 2)]]


