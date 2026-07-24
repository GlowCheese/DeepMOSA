####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_eq_same_instance():
    m1 = m(a=1, b=2)
    result = m1 == m1
    assert result is True

def test_eq_equal_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test_eq_different_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    result = m1 == m2
    assert result is False

def test_eq_pmap_vs_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test_eq_pmap_vs_dict_different():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = m1 == d
    assert result is False

def test_eq_pmap_vs_other_mapping():
    from collections.abc import Mapping
    class TestMapping(Mapping):
        def __init__(self, d):
            self._d = d
        def __getitem__(self, key):
            return self._d[key]
        def __iter__(self):
            return iter(self._d)
        def __len__(self):
            return len(self._d)
    m1 = m(a=1, b=2)
    tm = TestMapping({'a': 1, 'b': 2})
    result = m1 == tm
    assert result is True

def test_eq_pmap_vs_non_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [('a', 1), ('b', 2)]
    assert result is False

def test_eq_different_lengths():
    m1 = m(a=1, b=2)
    m2 = m(a=1)
    result = m1 == m2
    assert result is False

def test_eq_with_cached_hash_equal():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    hash(m1)
    hash(m2)
    result = m1 == m2
    assert result is True

def test_eq_with_cached_hash_different():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    hash(m1)
    hash(m2)
    result = m1 == m2
    assert result is False


# LLM-generated content at query #2
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_returns_pmap_instance():
    pmap = PMap(0, ())
    assert isinstance(pmap, PMap)

def test_constructor_sets_size_to_zero_for_empty_pmap():
    pmap = PMap(0, ())
    assert pmap._size == 0

def test_constructor_sets_buckets_to_empty_tuple_for_empty_pmap():
    pmap = PMap(0, ())
    assert pmap._buckets == ()

def test_constructor_sets_size_correctly_for_non_empty_pmap():
    buckets = (('a', 1), ('b', 2))
    pmap = PMap(2, buckets)
    assert pmap._size == 2

def test_constructor_sets_buckets_correctly_for_non_empty_pmap():
    buckets = (('a', 1), ('b', 2))
    pmap = PMap(2, buckets)
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_mixed_type_keys_and_values():
    buckets = ((1, 'one'), ('two', 2), (3.0, [1, 2, 3]))
    pmap = PMap(3, buckets)
    assert pmap._size == 3
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_none_key_and_value():
    buckets = ((None, 'null'), ('key', None))
    pmap = PMap(2, buckets)
    assert pmap._size == 2
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_duplicate_keys_in_buckets():
    buckets = (('key', 'value1'), ('key', 'value2'))
    pmap = PMap(2, buckets)
    assert pmap._size == 2
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_empty_buckets_but_non_zero_size():
    pmap = PMap(5, ())
    assert pmap._size == 5
    assert pmap._buckets == ()

def test_constructor_creates_pmap_with_single_bucket():
    buckets = (('single', 'item'),)
    pmap = PMap(1, buckets)
    assert pmap._size == 1
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_large_number_of_buckets():
    buckets = tuple((i, i*2) for i in range(1000))
    pmap = PMap(1000, buckets)
    assert pmap._size == 1000
    assert pmap._buckets == buckets

def test_constructor_sets_cached_hash_to_none_by_default():
    pmap = PMap(0, ())
    assert not hasattr(pmap, '_cached_hash')

def test_constructor_does_not_initialize_weakref():
    pmap = PMap(0, ())
    assert hasattr(pmap, '__weakref__')

def test_constructor_creates_pmap_with_custom_object_keys():
    class CustomKey:
        def __init__(self, id):
            self.id = id
    key1 = CustomKey(1)
    key2 = CustomKey(2)
    buckets = ((key1, 'val1'), (key2, 'val2'))
    pmap = PMap(2, buckets)
    assert pmap._size == 2
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_pmap_as_value():
    inner_pmap = PMap(1, (('inner', 'value'),))
    buckets = (('outer', inner_pmap),)
    pmap = PMap(1, buckets)
    assert pmap._size == 1
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_empty_string_key_and_value():
    buckets = (('', ''),)
    pmap = PMap(1, buckets)
    assert pmap._size == 1
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_boolean_key_and_value():
    buckets = ((True, False), (False, True))
    pmap = PMap(2, buckets)
    assert pmap._size == 2
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_tuple_key():
    buckets = (((1, 2), 'tuple_key'),)
    pmap = PMap(1, buckets)
    assert pmap._size == 1
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_dict_as_value():
    buckets = (('key', {'a': 1, 'b': 2}),)
    pmap = PMap(1, buckets)
    assert pmap._size == 1
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_list_as_value():
    buckets = (('key', [1, 2, 3]),)
    pmap = PMap(1, buckets)
    assert pmap._size == 1
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_set_as_value():
    buckets = (('key', {1, 2, 3}),)
    pmap = PMap(1, buckets)
    assert pmap._size == 1
    assert pmap._buckets == buckets

def test_constructor_handles_negative_size():
    pmap = PMap(-1, ())
    assert pmap._size == -1
    assert pmap._buckets == ()

def test_constructor_creates_pmap_with_function_as_value():
    def my_func():
        return 42
    buckets = (('func', my_func),)
    pmap = PMap(1, buckets)
    assert pmap._size == 1
    assert pmap._buckets == buckets


# LLM-generated content at query #3
#--------------------------

def test_eq_with_dict_and_different_buckets():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    result = pmap1 == dict1
    assert result is True
    result = pmap2 == dict1
    assert result is True


# LLM-generated content at query #4
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    assert m._size == 2
    assert len(m._buckets) > 0

def test_constructor_returns_pmap_instance():
    from pyrsistent import PMap
    from pyrsistent import pvector
    buckets = pvector([None, None, None, None])
    pm = PMap(0, buckets)
    assert isinstance(pm, PMap)

def test_constructor_sets_size_and_buckets():
    from pyrsistent import PMap
    from pyrsistent import pvector
    buckets = pvector([None, None])
    pm = PMap(5, buckets)
    assert pm._size == 5
    assert pm._buckets is buckets

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    assert m._size == 0
    assert len(m._buckets) > 0

def test_constructor_handles_cached_hash_attribute():
    from pyrsistent import PMap
    from pyrsistent import pvector
    buckets = pvector([None])
    pm = PMap(0, buckets)
    assert not hasattr(pm, '_cached_hash')

def test_constructor_preserves_weakref_slot():
    import weakref
    from pyrsistent import pmap
    m = pmap({'x': 10})
    wr = weakref.ref(m)
    assert wr() is m


# LLM-generated content at query #5
#--------------------------

def test_eq_returns_true_for_same_instance():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    values_view = m.values()
    result = values_view == values_view
    assert result is True

def test_eq_returns_false_for_different_instance_with_same_values():
    from pyrsistent import pmap
    m1 = pmap({1: 'a', 2: 'b'})
    m2 = pmap({1: 'a', 2: 'b'})
    values_view1 = m1.values()
    values_view2 = m2.values()
    result = values_view1 == values_view2
    assert result is False

def test_eq_returns_false_for_list_with_same_values():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    values_view = m.values()
    result = values_view == ['a', 'b']
    assert result is False

def test_eq_returns_false_for_tuple_with_same_values():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    values_view = m.values()
    result = values_view == ('a', 'b')
    assert result is False

def test_eq_returns_false_for_empty_values_view():
    from pyrsistent import pmap
    m = pmap({})
    values_view = m.values()
    result = values_view == []
    assert result is False

def test_eq_returns_false_for_none():
    from pyrsistent import pmap
    m = pmap({1: 'a'})
    values_view = m.values()
    result = values_view == None
    assert result is False

def test_eq_returns_false_for_different_values_view():
    from pyrsistent import pmap
    m1 = pmap({1: 'a', 2: 'b'})
    m2 = pmap({3: 'c', 4: 'd'})
    values_view1 = m1.values()
    values_view2 = m2.values()
    result = values_view1 == values_view2
    assert result is False


# LLM-generated content at query #6
#--------------------------

def test___contains___with_valid_key_value_pair():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 1) in items
    assert result is True

def test___contains___with_valid_key_but_wrong_value():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 2) in items
    assert result is False

def test___contains___with_key_not_in_map():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('c', 1) in items
    assert result is False

def test___contains___with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = 'a' in items
    assert result is False

def test___contains___with_tuple_of_wrong_length():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 1, 'extra') in items
    assert result is False

def test___contains___with_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    items = m.items()
    result = ('a', 1) in items
    assert result is False

def test___contains___with_nested_structure_in_value():
    from pyrsistent import pmap
    m = pmap({'a': [1, 2]})
    items = m.items()
    result = ('a', [1, 2]) in items
    assert result is True


# LLM-generated content at query #7
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap_instance = pmap(buckets)
    assert len(pmap_instance) == size
    assert pmap_instance['key1'] == 'value1'
    assert pmap_instance['key2'] == 'value2'

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap()
    assert len(pmap_instance) == 0
    assert dict(pmap_instance) == {}

def test_constructor_creates_pmap_from_dict():
    from pyrsistent import pmap
    original_dict = {'a': 1, 'b': 2}
    pmap_instance = pmap(original_dict)
    assert len(pmap_instance) == 2
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2

def test_constructor_creates_pmap_from_keyword_arguments():
    from pyrsistent import m
    pmap_instance = m(x=10, y=20)
    assert len(pmap_instance) == 2
    assert pmap_instance['x'] == 10
    assert pmap_instance['y'] == 20

def test_constructor_creates_pmap_from_iterable_of_pairs():
    from pyrsistent import pmap
    pairs = [('k1', 'v1'), ('k2', 'v2')]
    pmap_instance = pmap(pairs)
    assert len(pmap_instance) == 2
    assert pmap_instance['k1'] == 'v1'
    assert pmap_instance['k2'] == 'v2'

def test_constructor_handles_nested_pmap_creation():
    from pyrsistent import pmap
    inner = pmap({'inner_key': 'inner_value'})
    outer = pmap({'outer_key': inner})
    assert len(outer) == 1
    assert outer['outer_key']['inner_key'] == 'inner_value'

def test_constructor_preserves_immutability_after_creation():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    pmap_instance2 = pmap_instance.set('b', 2)
    assert len(pmap_instance) == 1
    assert len(pmap_instance2) == 2
    assert 'b' not in pmap_instance
    assert 'b' in pmap_instance2

def test_constructor_creates_pmap_with_duplicate_keys_last_wins():
    from pyrsistent import pmap
    pairs = [('key', 'first'), ('key', 'last')]
    pmap_instance = pmap(pairs)
    assert len(pmap_instance) == 1
    assert pmap_instance['key'] == 'last'

def test_constructor_with_empty_buckets_results_in_empty_pmap():
    from pyrsistent import PMap
    size = 0
    from pyrsistent import pvector
    buckets = pvector()
    pmap_instance = PMap(size, buckets)
    assert len(pmap_instance) == 0
    assert dict(pmap_instance) == {}

def test_constructor_with_non_empty_buckets_results_in_correct_pmap():
    from pyrsistent import PMap, pvector
    size = 3
    bucket_list = [None, None, [('a', 1), ('b', 2)], None, [('c', 3)]]
    buckets = pvector(bucket_list)
    pmap_instance = PMap(size, buckets)
    assert len(pmap_instance) == 3
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2
    assert pmap_instance['c'] == 3


# LLM-generated content at query #8
#--------------------------

def test_update_with_merges_values_using_update_fn():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2))
    assert m2 == {'a': 3, 'b': 2}

def test_update_with_keeps_leftmost_value_when_update_fn_returns_left():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m2 == {'a': 1}

def test_update_with_keeps_rightmost_value_when_update_fn_returns_right():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert m2 == {'a': 3}

def test_update_with_inserts_new_key_when_key_not_present():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l + r, m(b=2))
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_handles_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2, b=1), {'a': 3, 'c': 4})
    assert m2 == {'a': 6, 'b': 1, 'c': 4}

def test_update_with_returns_same_instance_when_no_changes():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m())
    assert m2 is m1

def test_update_with_works_with_empty_pmap():
    from pyrsistent import m
    m1 = m()
    m2 = m1.update_with(lambda l, r: l + r, m(a=1, b=2))
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_uses_initial_value_for_first_merge():
    from pyrsistent import m
    m1 = m(a=10)
    m2 = m1.update_with(lambda l, r: l * r, m(a=2), {'a': 3})
    assert m2 == {'a': 60}

def test_update_with_handles_non_integer_values():
    from pyrsistent import m
    m1 = m(a='hello')
    m2 = m1.update_with(lambda l, r: l + ' ' + r, m(a='world'))
    assert m2 == {'a': 'hello world'}

def test_update_with_merges_from_dict_and_pmap():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l + r, {'a': 2}, m(a=3))
    assert m2 == {'a': 6}


# LLM-generated content at query #9
#--------------------------

def test_eq_same_instance():
    m = PMap()
    items = PMapItems(m)
    result = items == items
    assert result is True

def test_eq_different_type():
    m = PMap()
    items = PMapItems(m)
    other = object()
    result = items == other
    assert result is False

def test_eq_same_map():
    m = PMap()
    items1 = PMapItems(m)
    items2 = PMapItems(m)
    result = items1 == items2
    assert result is True

def test_eq_different_map():
    m1 = PMap()
    m2 = PMap()
    items1 = PMapItems(m1)
    items2 = PMapItems(m2)
    result = items1 == items2
    assert result is False


# LLM-generated content at query #10
#--------------------------

def test_eq_returns_true_for_same_instance():
    m = {}
    view = PMapValues(m)
    result = view == view
    assert result is True

def test_eq_returns_false_for_different_instance():
    m = {}
    view1 = PMapValues(m)
    view2 = PMapValues(m)
    result = view1 == view2
    assert result is False

def test_eq_returns_false_for_non_pmapvalues_object():
    m = {}
    view = PMapValues(m)
    result = view == "not a view"
    assert result is False

def test_eq_returns_false_for_none():
    m = {}
    view = PMapValues(m)
    result = view == None
    assert result is False


# LLM-generated content at query #11
#--------------------------

def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    pmap1 = m(a=1)
    result = pmap1.update_with(lambda l, r: l + r, m(b=2))
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #12
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap_instance = pmap(buckets)
    assert len(pmap_instance) == size
    assert pmap_instance['key1'] == 'value1'
    assert pmap_instance['key2'] == 'value2'

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap()
    assert len(pmap_instance) == 0
    assert dict(pmap_instance) == {}

def test_constructor_with_keyword_arguments():
    from pyrsistent import pmap
    pmap_instance = pmap(a=1, b=2)
    assert len(pmap_instance) == 2
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2

def test_constructor_with_dict_argument():
    from pyrsistent import pmap
    pmap_instance = pmap({'x': 10, 'y': 20})
    assert len(pmap_instance) == 2
    assert pmap_instance['x'] == 10
    assert pmap_instance['y'] == 20

def test_constructor_with_iterable_of_pairs():
    from pyrsistent import pmap
    pairs = [('k1', 'v1'), ('k2', 'v2')]
    pmap_instance = pmap(pairs)
    assert len(pmap_instance) == 2
    assert pmap_instance['k1'] == 'v1'
    assert pmap_instance['k2'] == 'v2'

def test_constructor_preserves_identity_for_same_input():
    from pyrsistent import pmap
    data = {'a': 1}
    pmap1 = pmap(data)
    pmap2 = pmap(data)
    assert pmap1 == pmap2
    assert pmap1 is not pmap2

def test_constructor_handles_none_keys():
    from pyrsistent import pmap
    pmap_instance = pmap({None: 'null'})
    assert pmap_instance[None] == 'null'
    assert len(pmap_instance) == 1

def test_constructor_handles_integer_keys():
    from pyrsistent import pmap
    pmap_instance = pmap({1: 'one', 2: 'two'})
    assert pmap_instance[1] == 'one'
    assert pmap_instance[2] == 'two'
    assert len(pmap_instance) == 2

def test_constructor_handles_float_keys():
    from pyrsistent import pmap
    pmap_instance = pmap({1.5: 'one point five', 2.7: 'two point seven'})
    assert pmap_instance[1.5] == 'one point five'
    assert pmap_instance[2.7] == 'two point seven'
    assert len(pmap_instance) == 2

def test_constructor_handles_tuple_keys():
    from pyrsistent import pmap
    pmap_instance = pmap({(1, 2): 'tuple key'})
    assert pmap_instance[(1, 2)] == 'tuple key'
    assert len(pmap_instance) == 1

def test_constructor_with_empty_inputs():
    from pyrsistent import pmap
    pmap1 = pmap({})
    pmap2 = pmap([])
    pmap3 = pmap()
    assert len(pmap1) == 0
    assert len(pmap2) == 0
    assert len(pmap3) == 0
    assert pmap1 == pmap2 == pmap3

def test_constructor_raises_on_duplicate_keys():
    from pyrsistent import pmap
    pmap_instance = pmap([('a', 1), ('a', 2)])
    assert pmap_instance['a'] == 2
    assert len(pmap_instance) == 1

def test_constructor_with_mixed_input_types():
    from pyrsistent import pmap
    pmap_instance = pmap([('a', 1)], b=2, **{'c': 3})
    assert len(pmap_instance) == 3
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2
    assert pmap_instance['c'] == 3

def test_constructor_creates_hashable_instance():
    from pyrsistent import pmap
    pmap_instance = pmap({'key': 'value'})
    hash_value = hash(pmap_instance)
    assert isinstance(hash_value, int)

def test_constructor_with_complex_nested_structure():
    from pyrsistent import pmap, v
    pmap_instance = pmap({'vec': v(1, 2, 3), 'nested': {'inner': 'data'}})
    assert len(pmap_instance) == 2
    assert pmap_instance['vec'] == v(1, 2, 3)
    assert pmap_instance['nested'] == {'inner': 'data'}

def test_constructor_pmap_is_immutable():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    try:
        pmap_instance['a'] = 2
    except TypeError:
        pass
    assert pmap_instance['a'] == 1


# LLM-generated content at query #13
#--------------------------

def test_eq_same_instance():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = items_view == items_view
    assert result is True

def test_eq_different_pmap_items_same_map():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view1 = m.items()
    items_view2 = m.items()
    result = items_view1 == items_view2
    assert result is True

def test_eq_different_pmap_items_different_map():
    from pyrsistent import pmap
    m1 = pmap({1: 'a', 2: 'b'})
    m2 = pmap({1: 'a', 2: 'b'})
    items_view1 = m1.items()
    items_view2 = m2.items()
    result = items_view1 == items_view2
    assert result is True

def test_eq_different_pmap_items_different_content():
    from pyrsistent import pmap
    m1 = pmap({1: 'a', 2: 'b'})
    m2 = pmap({1: 'a', 3: 'c'})
    items_view1 = m1.items()
    items_view2 = m2.items()
    result = items_view1 == items_view2
    assert result is False

def test_eq_with_non_pmap_items_instance():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = items_view == [(1, 'a'), (2, 'b')]
    assert result is False

def test_eq_with_none():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = items_view == None
    assert result is False


# LLM-generated content at query #14
#--------------------------

def test_pmap_constructor_creates_instance_with_size_and_buckets():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    assert m._size == 2
    assert len(m._buckets) > 0

def test_pmap_constructor_sets_correct_size():
    from pyrsistent import pmap
    m = pmap({'x': 10, 'y': 20, 'z': 30})
    assert m._size == 3

def test_pmap_constructor_creates_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    assert m._size == 0
    assert len(m._buckets) > 0

def test_pmap_constructor_handles_same_hash_collisions():
    from pyrsistent import pmap
    class SameHash:
        def __init__(self, val):
            self.val = val
        def __hash__(self):
            return 1
        def __eq__(self, other):
            return isinstance(other, SameHash) and self.val == other.val
    key1 = SameHash(1)
    key2 = SameHash(2)
    m = pmap({key1: 'a', key2: 'b'})
    assert m[key1] == 'a'
    assert m[key2] == 'b'
    assert m._size == 2

def test_pmap_constructor_preserves_key_value_pairs():
    from pyrsistent import pmap
    m = pmap({'key1': 'value1', 'key2': 'value2'})
    assert m['key1'] == 'value1'
    assert m['key2'] == 'value2'

def test_pmap_constructor_allows_none_as_key():
    from pyrsistent import pmap
    m = pmap({None: 'null'})
    assert m[None] == 'null'
    assert m._size == 1

def test_pmap_constructor_allows_none_as_value():
    from pyrsistent import pmap
    m = pmap({'key': None})
    assert m['key'] is None
    assert m._size == 1

def test_pmap_constructor_creates_distinct_instances():
    from pyrsistent import pmap
    m1 = pmap({'a': 1})
    m2 = pmap({'a': 1})
    assert m1 is not m2
    assert m1 == m2

def test_pmap_constructor_with_large_number_of_items():
    from pyrsistent import pmap
    items = {str(i): i for i in range(100)}
    m = pmap(items)
    assert m._size == 100
    for i in range(100):
        assert m[str(i)] == i

def test_pmap_constructor_handles_duplicate_keys_last_wins():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'a': 2})
    assert m['a'] == 2
    assert m._size == 1


# LLM-generated content at query #15
#--------------------------

def test_eq_same_instance():
    m1 = m(a=1, b=2)
    result = m1 == m1
    assert result is True

def test_eq_equal_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test_eq_different_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    result = m1 == m2
    assert result is False

def test_eq_pmap_vs_dict_equal():
    pm = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = pm == d
    assert result is True

def test_eq_pmap_vs_dict_different():
    pm = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = pm == d
    assert result is False

def test_eq_pmap_vs_dict_different_length():
    pm = m(a=1, b=2)
    d = {'a': 1}
    result = pm == d
    assert result is False

def test_eq_pmap_vs_other_mapping_equal():
    from collections import OrderedDict
    pm = m(a=1, b=2)
    od = OrderedDict([('a', 1), ('b', 2)])
    result = pm == od
    assert result is True

def test_eq_pmap_vs_other_mapping_different():
    from collections import OrderedDict
    pm = m(a=1, b=2)
    od = OrderedDict([('a', 1), ('b', 3)])
    result = pm == od
    assert result is False

def test_eq_not_implemented_for_non_mapping():
    pm = m(a=1, b=2)
    result = pm == [('a', 1), ('b', 2)]
    assert result is NotImplemented

def test_eq_cached_hash_mismatch():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    m1._cached_hash = hash(frozenset(m1.iteritems()))
    m2._cached_hash = hash(frozenset(m2.iteritems()))
    result = m1 == m2
    assert result is False

def test_eq_buckets_equal():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1._buckets == m2._buckets
    assert result is True

def test_eq_buckets_different():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    result = m1._buckets == m2._buckets
    assert result is False


# LLM-generated content at query #16
#--------------------------

def test_contains_with_invalid_arg_returns_false():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    items_view = m.items()
    result = ('a', 2) in items_view
    assert result == False


# LLM-generated content at query #17
#--------------------------

def test__turbo_mapping_with_empty_initial_and_no_pre_size():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0
    assert dict(result) == {}


def test__turbo_mapping_with_empty_initial_and_pre_size():
    result = _turbo_mapping({}, 10)
    assert len(result) == 0
    assert dict(result) == {}


def test__turbo_mapping_with_dict_initial_and_no_pre_size():
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert dict(result) == {'a': 1, 'b': 2}


def test__turbo_mapping_with_dict_initial_and_pre_size():
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, 20)
    assert len(result) == 2
    assert dict(result) == {'a': 1, 'b': 2}


def test__turbo_mapping_with_iterable_initial_and_no_pre_size():
    initial = [('a', 1), ('b', 2)]
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert dict(result) == {'a': 1, 'b': 2}


def test__turbo_mapping_with_iterable_initial_and_pre_size():
    initial = [('a', 1), ('b', 2)]
    result = _turbo_mapping(initial, 15)
    assert len(result) == 2
    assert dict(result) == {'a': 1, 'b': 2}


def test__turbo_mapping_handles_collisions():
    class FixedHash:
        def __init__(self, value, hash_value):
            self.value = value
            self.hash_value = hash_value
        def __hash__(self):
            return self.hash_value
        def __eq__(self, other):
            return isinstance(other, FixedHash) and self.value == other.value
    k1 = FixedHash('key1', 5)
    k2 = FixedHash('key2', 5)
    initial = {k1: 'val1', k2: 'val2'}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert result[k1] == 'val1'
    assert result[k2] == 'val2'


def test__turbo_mapping_with_large_initial_and_no_pre_size():
    initial = {i: i*2 for i in range(100)}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 100
    assert all(result[i] == i*2 for i in range(100))


def test__turbo_mapping_with_large_initial_and_small_pre_size():
    initial = {i: i*2 for i in range(100)}
    result = _turbo_mapping(initial, 50)
    assert len(result) == 100
    assert all(result[i] == i*2 for i in range(100))


# LLM-generated content at query #18
#--------------------------

def test___eq___same_instance():
    pmap1 = m(a=1, b=2)
    result = pmap1 == pmap1
    assert result is True

def test___eq___equal_pmaps():
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    result = pmap1 == pmap2
    assert result is True

def test___eq___different_pmaps():
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    result = pmap1 == pmap2
    assert result is False

def test___eq___different_lengths():
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1)
    result = pmap1 == pmap2
    assert result is False

def test___eq___with_dict_equal():
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    result = pmap1 == dict1
    assert result is True

def test___eq___with_dict_not_equal():
    pmap1 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 3}
    result = pmap1 == dict1
    assert result is False

def test___eq___with_other_mapping_equal():
    from collections import OrderedDict
    pmap1 = m(a=1, b=2)
    od = OrderedDict([('a', 1), ('b', 2)])
    result = pmap1 == od
    assert result is True

def test___eq___with_other_mapping_not_equal():
    from collections import OrderedDict
    pmap1 = m(a=1, b=2)
    od = OrderedDict([('a', 1), ('b', 3)])
    result = pmap1 == od
    assert result is False

def test___eq___with_non_mapping():
    pmap1 = m(a=1, b=2)
    result = pmap1 == [1, 2]
    assert result is NotImplemented

def test___eq___cached_hash_mismatch():
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=3)
    pmap1._cached_hash = hash(frozenset(pmap1.iteritems()))
    pmap2._cached_hash = hash(frozenset(pmap2.iteritems()))
    result = pmap1 == pmap2
    assert result is False

def test___eq___same_buckets():
    pmap1 = m(a=1, b=2)
    pmap2 = pmap1
    result = pmap1 == pmap2
    assert result is True


# LLM-generated content at query #19
#--------------------------

def test_eq_same_instance():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = items_view == items_view
    assert result is True

def test_eq_different_pmapitems_same_map():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view1 = m.items()
    items_view2 = m.items()
    result = items_view1 == items_view2
    assert result is True

def test_eq_different_pmapitems_different_map_same_content():
    from pyrsistent import pmap
    m1 = pmap({1: 'a', 2: 'b'})
    m2 = pmap({1: 'a', 2: 'b'})
    items_view1 = m1.items()
    items_view2 = m2.items()
    result = items_view1 == items_view2
    assert result is True

def test_eq_different_pmapitems_different_map_different_content():
    from pyrsistent import pmap
    m1 = pmap({1: 'a', 2: 'b'})
    m2 = pmap({1: 'a', 3: 'c'})
    items_view1 = m1.items()
    items_view2 = m2.items()
    result = items_view1 == items_view2
    assert result is False

def test_eq_with_non_pmapitems_instance():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = items_view == [(1, 'a'), (2, 'b')]
    assert result is False

def test_eq_with_none():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = items_view == None
    assert result is False

def test_eq_empty_maps():
    from pyrsistent import pmap
    m1 = pmap({})
    m2 = pmap({})
    items_view1 = m1.items()
    items_view2 = m2.items()
    result = items_view1 == items_view2
    assert result is True


# LLM-generated content at query #20
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap, m
    size = 2
    buckets = (('a', 1), ('b', 2))
    pmap_instance = pmap({'a': 1, 'b': 2})
    assert pmap_instance._size == size
    assert dict(pmap_instance._buckets) == dict(buckets)

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert pmap_instance._size == 0
    assert len(pmap_instance._buckets) == 0

def test_constructor_creates_pmap_with_single_element():
    from pyrsistent import pmap
    pmap_instance = pmap({'key': 'value'})
    assert pmap_instance._size == 1
    assert pmap_instance['key'] == 'value'

def test_constructor_creates_pmap_with_multiple_elements():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    assert pmap_instance._size == 3
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2
    assert pmap_instance['c'] == 3

def test_constructor_creates_pmap_with_none_value():
    from pyrsistent import pmap
    pmap_instance = pmap({'key': None})
    assert pmap_instance._size == 1
    assert pmap_instance['key'] is None

def test_constructor_creates_pmap_with_zero_size():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert pmap_instance._size == 0
    assert len(pmap_instance) == 0

def test_constructor_creates_pmap_that_is_hashable():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2})
    hash_value = hash(pmap_instance)
    assert isinstance(hash_value, int)

def test_constructor_creates_pmap_that_supports_equality():
    from pyrsistent import pmap
    pmap1 = pmap({'a': 1, 'b': 2})
    pmap2 = pmap({'a': 1, 'b': 2})
    assert pmap1 == pmap2

def test_constructor_creates_pmap_with_correct_buckets_structure():
    from pyrsistent import pmap
    pmap_instance = pmap({'x': 10, 'y': 20})
    assert pmap_instance._buckets is not None

def test_constructor_creates_pmap_that_is_immutable():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    try:
        pmap_instance['a'] = 2
        assert False
    except TypeError:
        assert True

def test_constructor_creates_pmap_with_mixed_key_types():
    from pyrsistent import pmap
    pmap_instance = pmap({1: 'one', 'two': 2, (3, 4): 'tuple'})
    assert pmap_instance._size == 3
    assert pmap_instance[1] == 'one'
    assert pmap_instance['two'] == 2
    assert pmap_instance[(3, 4)] == 'tuple'

def test_constructor_creates_pmap_from_empty_dict():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert pmap_instance._size == 0
    assert len(pmap_instance._buckets) == 0

def test_constructor_creates_pmap_that_is_iterable():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2})
    keys = list(pmap_instance)
    assert set(keys) == {'a', 'b'}

def test_constructor_creates_pmap_with_duplicate_keys_last_wins():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'a': 2})
    assert pmap_instance._size == 1
    assert pmap_instance['a'] == 2


# LLM-generated content at query #21
#--------------------------

def test___contains___with_valid_key_value_pair_present():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1) in items_view
    assert result == True

def test___contains___with_valid_key_value_pair_absent():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 2) in items_view
    assert result == False

def test___contains___with_key_not_in_map():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('c', 1) in items_view
    assert result == False

def test___contains___with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = 'a' in items_view
    assert result == False

def test___contains___with_tuple_of_wrong_length():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1, 'extra') in items_view
    assert result == False

def test___contains___with_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    result = ('a', 1) in items_view
    assert result == False


# LLM-generated content at query #22
#--------------------------

def test_turbo_mapping_with_empty_initial():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0
    assert dict(result) == {}

def test_turbo_mapping_with_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, 16)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_turbo_mapping_without_pre_size():
    result = _turbo_mapping({'x': 10, 'y': 20}, 0)
    assert len(result) == 2
    assert result['x'] == 10
    assert result['y'] == 20

def test_turbo_mapping_with_collision():
    class FixedHash:
        def __init__(self, value, hash_value):
            self.value = value
            self.hash_value = hash_value
        def __hash__(self):
            return self.hash_value
        def __eq__(self, other):
            return isinstance(other, FixedHash) and self.value == other.value
    obj1 = FixedHash('a', 5)
    obj2 = FixedHash('b', 5)
    result = _turbo_mapping({obj1: 100, obj2: 200}, 0)
    assert len(result) == 2
    assert result[obj1] == 100
    assert result[obj2] == 200

def test_turbo_mapping_with_non_mapping_initial():
    result = _turbo_mapping([('key1', 'val1'), ('key2', 'val2')], 0)
    assert len(result) == 2
    assert result['key1'] == 'val1'
    assert result['key2'] == 'val2'

def test_turbo_mapping_preserves_hashability():
    result = _turbo_mapping({'a': 1, 'b': 2}, 0)
    hash1 = hash(result)
    result2 = result.set('c', 3)
    hash2 = hash(result2)
    assert hash1 != hash2

def test_turbo_mapping_large_pre_size():
    result = _turbo_mapping({'a': 1}, 100)
    assert len(result) == 1
    assert result['a'] == 1

def test_turbo_mapping_with_zero_pre_size_and_empty():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0

def test_turbo_mapping_initial_dict_unchanged():
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, 0)
    assert initial == {'a': 1, 'b': 2}
    assert dict(result) == {'a': 1, 'b': 2}

def test_turbo_mapping_with_non_integer_len_fallback():
    class BadLenMapping:
        def __init__(self, data):
            self.data = dict(data)
        def items(self):
            return self.data.items()
        def __len__(self):
            raise Exception("no length")
    bad = BadLenMapping([('x', 10)])
    result = _turbo_mapping(bad, 0)
    assert len(result) == 1
    assert result['x'] == 10


# LLM-generated content at query #23
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap, m
    size = 2
    buckets = (('a', 1), ('b', 2))
    pmap_instance = pmap(buckets)
    assert len(pmap_instance) == size
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert len(pmap_instance) == 0
    assert list(pmap_instance) == []

def test_constructor_creates_pmap_from_dict():
    from pyrsistent import pmap
    original_dict = {'x': 10, 'y': 20}
    pmap_instance = pmap(original_dict)
    assert len(pmap_instance) == 2
    assert pmap_instance['x'] == 10
    assert pmap_instance['y'] == 20

def test_constructor_creates_pmap_from_keyword_arguments():
    from pyrsistent import m
    pmap_instance = m(alpha=100, beta=200)
    assert len(pmap_instance) == 2
    assert pmap_instance['alpha'] == 100
    assert pmap_instance['beta'] == 200

def test_constructor_creates_pmap_from_iterable_of_pairs():
    from pyrsistent import pmap
    pairs = [('key1', 'value1'), ('key2', 'value2')]
    pmap_instance = pmap(pairs)
    assert len(pmap_instance) == 2
    assert pmap_instance['key1'] == 'value1'
    assert pmap_instance['key2'] == 'value2'

def test_constructor_pmap_is_hashable():
    from pyrsistent import pmap
    pmap1 = pmap({'a': 1, 'b': 2})
    pmap2 = pmap({'a': 1, 'b': 2})
    assert hash(pmap1) == hash(pmap2)
    assert pmap1 == pmap2

def test_constructor_pmap_supports_dot_notation():
    from pyrsistent import m
    pmap_instance = m(foo='bar')
    assert pmap_instance.foo == 'bar'

def test_constructor_pmap_raises_key_error_on_missing_key():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    try:
        _ = pmap_instance['missing']
        assert False
    except KeyError:
        assert True

def test_constructor_pmap_raises_attribute_error_on_missing_attribute():
    from pyrsistent import m
    pmap_instance = m()
    try:
        _ = pmap_instance.missing
        assert False
    except AttributeError:
        assert True

def test_constructor_pmap_is_immutable():
    from pyrsistent import pmap
    pmap_instance = pmap({'k': 'v'})
    try:
        pmap_instance['k'] = 'new'
        assert False
    except TypeError:
        assert True

def test_constructor_pmap_preserves_insertion_order():
    from pyrsistent import pmap
    pmap_instance = pmap([('z', 1), ('a', 2), ('m', 3)])
    keys = list(pmap_instance.keys())
    assert keys == ['z', 'a', 'm']

def test_constructor_pmap_with_duplicate_keys_keeps_last():
    from pyrsistent import pmap
    pmap_instance = pmap([('k', 1), ('k', 2)])
    assert pmap_instance['k'] == 2
    assert len(pmap_instance) == 1

def test_constructor_pmap_from_empty_iterable():
    from pyrsistent import pmap
    pmap_instance = pmap([])
    assert len(pmap_instance) == 0
    assert dict(pmap_instance) == {}

def test_constructor_pmap_equality_with_dict():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2})
    assert pmap_instance == {'a': 1, 'b': 2}
    assert not (pmap_instance != {'a': 1, 'b': 2})

def test_constructor_pmap_equality_with_other_pmap():
    from pyrsistent import pmap
    pmap1 = pmap({'x': 100})
    pmap2 = pmap({'x': 100})
    assert pmap1 == pmap2
    assert not (pmap1 != pmap2)

def test_constructor_pmap_inequality_with_different_size():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    other = pmap({'a': 1, 'b': 2})
    assert pmap_instance != other

def test_constructor_pmap_inequality_with_different_values():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    other = pmap({'a': 2})
    assert pmap_instance != other

def test_constructor_pmap_implements_mapping_protocol():
    from collections.abc import Mapping
    from pyrsistent import pmap
    pmap_instance = pmap({'k': 'v'})
    assert isinstance(pmap_instance, Mapping)

def test_constructor_pmap_is_not_orderable():
    from pyrsistent import pmap
    pmap1 = pmap({'a': 1})
    pmap2 = pmap({'b': 2})
    try:
        _ = pmap1 < pmap2
        assert False
    except TypeError:
        assert True

def test_constructor_pmap_representation():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    assert repr(pmap_instance) == "pmap({'a': 1})"
    assert str(pmap_instance) == "pmap({'a': 1})"

def test_constructor_pmap_iteritems_yields_key_value_pairs():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2})
    items = list(pmap_instance.iteritems())
    assert set(items) == {('a', 1), ('b', 2)}

def test_constructor_pmap_iterkeys_yields_keys():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2})
    keys = list(pmap_instance.iterkeys())
    assert set(keys) == {'a', 'b'}

def test_constructor_pmap_itervalues_yields_values():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2})
    values = list(pmap_instance.itervalues())
    assert set(values) == {1, 2}

def test_constructor_pmap_contains_key():
    from pyrsistent import pmap
    pmap_instance = pmap({'present': True})
    assert 'present' in pmap_instance
    assert 'absent' not in pmap_instance

def test_constructor_pmap_get_method_with_default():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    assert pmap_instance.get('a') == 1
    assert pmap_instance.get('b') is None
    assert pmap_instance.get('b', 'default') == 'default'

def test_constructor_pmap_reversed_raises_type_error():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    try:
        _ = reversed(pmap_instance)
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #24
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap_instance = pmap(buckets)
    assert len(pmap_instance) == size
    assert pmap_instance['key1'] == 'value1'
    assert pmap_instance['key2'] == 'value2'

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap()
    assert len(pmap_instance) == 0
    assert dict(pmap_instance) == {}

def test_constructor_handles_nested_pmaps():
    from pyrsistent import pmap
    inner = pmap({'a': 1})
    outer = pmap({'inner': inner})
    assert outer['inner']['a'] == 1
    assert isinstance(outer['inner'], type(inner))

def test_constructor_preserves_hash_collisions_handling():
    from pyrsistent import pmap
    class SameHash:
        def __init__(self, value):
            self.value = value
        def __hash__(self):
            return 1
        def __eq__(self, other):
            return isinstance(other, SameHash) and self.value == other.value
    key1 = SameHash(1)
    key2 = SameHash(2)
    pmap_instance = pmap({key1: 'val1', key2: 'val2'})
    assert pmap_instance[key1] == 'val1'
    assert pmap_instance[key2] == 'val2'

def test_constructor_with_different_mapping_types():
    from pyrsistent import pmap
    dict_input = {'a': 1, 'b': 2}
    pmap_from_dict = pmap(dict_input)
    assert pmap_from_dict['a'] == 1
    assert pmap_from_dict['b'] == 2
    pmap_from_pmap = pmap(pmap_from_dict)
    assert pmap_from_pmap['a'] == 1
    assert pmap_from_pmap['b'] == 2

def test_constructor_creates_pmap_with_correct_size_attribute():
    from pyrsistent import pmap
    pmap_instance = pmap({'x': 10, 'y': 20})
    assert pmap_instance._size == 2

def test_constructor_supports_keyword_arguments():
    from pyrsistent import m
    pmap_instance = m(alpha=100, beta=200)
    assert pmap_instance['alpha'] == 100
    assert pmap_instance['beta'] == 200
    assert len(pmap_instance) == 2

def test_constructor_with_empty_buckets():
    from pyrsistent import PMap
    empty_buckets = ()
    pmap_instance = PMap(0, empty_buckets)
    assert len(pmap_instance) == 0
    assert list(pmap_instance.items()) == []

def test_constructor_does_not_allow_direct_instantiation_without_factory():
    from pyrsistent import PMap
    pmap_instance = PMap(0, ())
    assert isinstance(pmap_instance, PMap)
    assert len(pmap_instance) == 0

def test_constructor_pmap_is_hashable_when_empty():
    from pyrsistent import pmap
    pmap_instance = pmap()
    hash_value = hash(pmap_instance)
    assert isinstance(hash_value, int)

def test_constructor_pmap_is_hashable_with_elements():
    from pyrsistent import pmap
    pmap_instance = pmap({'k1': 'v1', 'k2': 'v2'})
    hash_value = hash(pmap_instance)
    assert isinstance(hash_value, int)

def test_constructor_pmap_with_same_items_has_same_hash():
    from pyrsistent import pmap
    pmap1 = pmap({'a': 1, 'b': 2})
    pmap2 = pmap({'b': 2, 'a': 1})
    assert hash(pmap1) == hash(pmap2)

def test_constructor_pmap_with_different_items_has_different_hash():
    from pyrsistent import pmap
    pmap1 = pmap({'a': 1})
    pmap2 = pmap({'a': 2})
    assert hash(pmap1) != hash(pmap2)


# LLM-generated content at query #25
#--------------------------

def test_update_with_does_not_call_update_fn_when_key_not_in_evolver():
    from pyrsistent import m
    call_count = 0
    def dummy_update_fn(l, r):
        nonlocal call_count
        call_count += 1
        return l + r
    m1 = m(a=1)
    m2 = m(b=2)
    result = m1.update_with(dummy_update_fn, m2)
    assert call_count == 0
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #26
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets

def test_constructor_returns_pmap_instance():
    pmap_instance = PMap(0, ())
    assert isinstance(pmap_instance, PMap)

def test_constructor_sets_size_to_zero_for_empty_pmap():
    pmap_instance = PMap(0, ())
    assert pmap_instance._size == 0

def test_constructor_sets_buckets_to_empty_tuple_for_empty_pmap():
    pmap_instance = PMap(0, ())
    assert pmap_instance._buckets == ()

def test_constructor_sets_size_correctly_for_non_empty_pmap():
    buckets = (('a', 1), ('b', 2))
    pmap_instance = PMap(2, buckets)
    assert pmap_instance._size == 2

def test_constructor_assigns_buckets_directly():
    custom_buckets = (('x', 10), ('y', 20))
    pmap_instance = PMap(2, custom_buckets)
    assert pmap_instance._buckets is custom_buckets

def test_constructor_creates_pmap_with_single_key_value_pair():
    size = 1
    buckets = (('single_key', 'single_value'),)
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets

def test_constructor_handles_large_size_value():
    large_size = 1000
    buckets = tuple(('key' + str(i), i) for i in range(large_size))
    pmap_instance = PMap(large_size, buckets)
    assert pmap_instance._size == large_size
    assert len(pmap_instance._buckets) == large_size

def test_constructor_sets_cached_hash_to_none_by_default():
    pmap_instance = PMap(0, ())
    assert not hasattr(pmap_instance, '_cached_hash')

def test_constructor_does_not_modify_input_buckets():
    original_buckets = (('k1', 'v1'), ('k2', 'v2'))
    buckets_copy = original_buckets
    pmap_instance = PMap(2, buckets_copy)
    assert pmap_instance._buckets == original_buckets
    assert pmap_instance._buckets is buckets_copy


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_eq_same_instance():
    m1 = m(a=1, b=2)
    result = m1 == m1
    assert result is True

def test_eq_equal_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test_eq_different_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    result = m1 == m2
    assert result is False

def test_eq_pmap_vs_dict_equal():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test_eq_pmap_vs_dict_not_equal():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = m1 == d
    assert result is False

def test_eq_pmap_vs_dict_different_length():
    m1 = m(a=1, b=2)
    d = {'a': 1}
    result = m1 == d
    assert result is False

def test_eq_pmap_vs_other_mapping_equal():
    from collections import OrderedDict
    m1 = m(a=1, b=2)
    od = OrderedDict([('a', 1), ('b', 2)])
    result = m1 == od
    assert result is True

def test_eq_pmap_vs_non_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [('a', 1), ('b', 2)]
    assert result is False

def test_eq_pmap_vs_pmap_with_same_buckets():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = hash(frozenset(m1.iteritems()))
    m2._cached_hash = hash(frozenset(m2.iteritems()))
    result = m1 == m2
    assert result is True

def test_eq_pmap_vs_pmap_with_different_cached_hash():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = 123
    m2._cached_hash = 456
    result = m1 == m2
    assert result is False

def test_eq_pmap_vs_pmap_with_different_buckets_same_items():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = None
    m2._cached_hash = None
    result = m1 == m2
    assert result is True


# LLM-generated content at query #2
#--------------------------

def test___contains___with_valid_key_value_pair_present():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1) in items_view
    assert result == True

def test___contains___with_valid_key_value_pair_absent():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 2) in items_view
    assert result == False

def test___contains___with_key_not_in_map():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('c', 1) in items_view
    assert result == False

def test___contains___with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = 'a' in items_view
    assert result == False

def test___contains___with_tuple_of_wrong_length():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1, 'extra') in items_view
    assert result == False

def test___contains___with_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    result = ('a', 1) in items_view
    assert result == False


# LLM-generated content at query #3
#--------------------------

def test_turbo_mapping_with_empty_initial_and_no_pre_size():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0
    assert dict(result) == {}

def test_turbo_mapping_with_empty_initial_and_pre_size():
    result = _turbo_mapping({}, 10)
    assert len(result) == 0
    assert dict(result) == {}

def test_turbo_mapping_with_dict_initial_and_no_pre_size():
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert dict(result) == {'a': 1, 'b': 2}

def test_turbo_mapping_with_dict_initial_and_pre_size():
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, 20)
    assert len(result) == 2
    assert dict(result) == {'a': 1, 'b': 2}

def test_turbo_mapping_with_non_mapping_initial():
    initial = [('a', 1), ('b', 2)]
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert dict(result) == {'a': 1, 'b': 2}

def test_turbo_mapping_with_collision_keys():
    class FixedHash:
        def __init__(self, value, hash_value):
            self.value = value
            self.hash_value = hash_value
        def __hash__(self):
            return self.hash_value
        def __eq__(self, other):
            return isinstance(other, FixedHash) and self.value == other.value
    key1 = FixedHash('key1', 5)
    key2 = FixedHash('key2', 5)
    initial = {key1: 'val1', key2: 'val2'}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert result[key1] == 'val1'
    assert result[key2] == 'val2'

def test_turbo_mapping_initial_len_exception_falls_back():
    class BadLenMapping:
        def __len__(self):
            raise Exception("no length")
        def items(self):
            return [('a', 1)].__iter__()
    initial = BadLenMapping()
    result = _turbo_mapping(initial, 0)
    assert len(result) == 1
    assert dict(result) == {'a': 1}

def test_turbo_mapping_preserves_hash_and_equality():
    initial = {'x': 10, 'y': 20}
    pmap1 = _turbo_mapping(initial, 0)
    pmap2 = _turbo_mapping(initial, 0)
    assert pmap1 == pmap2
    assert hash(pmap1) == hash(pmap2)

def test_turbo_mapping_with_large_pre_size():
    initial = {'a': 1}
    result = _turbo_mapping(initial, 100)
    assert len(result) == 1
    assert dict(result) == {'a': 1}

def test_turbo_mapping_with_zero_pre_size_and_empty_initial():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0


# LLM-generated content at query #4
#--------------------------

def test_eq_with_cached_hash_mismatch_returns_false():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m1._cached_hash = 123
    m2._cached_hash = 456
    result = m1 == m2
    assert result == False


# LLM-generated content at query #5
#--------------------------

def test_update_with_merge_function():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1, b=2)
    m2 = m1.update_with(add, m(a=2))
    assert m2 == {'a': 3, 'b': 2}

def test_update_with_keep_leftmost():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m2 == {'a': 1}

def test_update_with_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2, c=3), {'a': 10, 'd': 4})
    assert m2 == {'a': 13, 'b': 2, 'c': 3, 'd': 4}

def test_update_with_empty_map():
    from pyrsistent import m
    m1 = m()
    m2 = m1.update_with(lambda l, r: r, {'a': 1, 'b': 2})
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_no_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r)
    assert m2 == m1

def test_update_with_new_key():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, {'b': 2})
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_overwrites_existing():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r * 2, {'a': 5})
    assert m2 == {'a': 10, 'b': 2}

def test_update_with_identity_function():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r, m(a=3, c=4))
    assert m2 == {'a': 3, 'b': 2, 'c': 4}


# LLM-generated content at query #6
#--------------------------

def test_contains_with_existing_key_value_pair():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 1) in items
    assert result == True

def test_contains_with_existing_key_but_different_value():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 2) in items
    assert result == False

def test_contains_with_non_existing_key():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('c', 1) in items
    assert result == False

def test_contains_with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = 'a' in items
    assert result == False

def test_contains_with_wrong_length_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 1, 3) in items
    assert result == False

def test_contains_with_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    items = m.items()
    result = ('a', 1) in items
    assert result == False


# LLM-generated content at query #7
#--------------------------

def test_contains_predicate_true():
    class MockMap:
        def __init__(self, data):
            self.data = data
        def __contains__(self, key):
            return key in self.data
        def __getitem__(self, key):
            return self.data[key]
    class TestPMapItems:
        def __init__(self, m):
            self._map = m
        def __contains__(self, arg):
            try: (k,v) = arg
            except Exception: return False
            return k in self._map and self._map[k] == v
    mock_map = MockMap({'a': 1, 'b': 2})
    items_view = TestPMapItems(mock_map)
    result = ('a', 1) in items_view
    assert result == True


# LLM-generated content at query #8
#--------------------------

def test_eq_same_instance():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = items_view == items_view
    assert result is True

def test_eq_different_pmap_items_same_map():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view1 = m.items()
    items_view2 = m.items()
    result = items_view1 == items_view2
    assert result is True

def test_eq_different_pmap_items_different_map():
    from pyrsistent import pmap
    m1 = pmap({1: 'a', 2: 'b'})
    m2 = pmap({1: 'a', 2: 'b'})
    items_view1 = m1.items()
    items_view2 = m2.items()
    result = items_view1 == items_view2
    assert result is True

def test_eq_different_pmap_items_different_content():
    from pyrsistent import pmap
    m1 = pmap({1: 'a', 2: 'b'})
    m2 = pmap({1: 'a', 3: 'c'})
    items_view1 = m1.items()
    items_view2 = m2.items()
    result = items_view1 == items_view2
    assert result is False

def test_eq_with_non_pmap_items_instance():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = items_view == [(1, 'a'), (2, 'b')]
    assert result is False

def test_eq_with_none():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = items_view == None
    assert result is False

def test_eq_empty_pmap_items():
    from pyrsistent import pmap
    m1 = pmap({})
    m2 = pmap({})
    items_view1 = m1.items()
    items_view2 = m2.items()
    result = items_view1 == items_view2
    assert result is True


# LLM-generated content at query #9
#--------------------------

def test___getattr___returns_value_for_existing_key():
    pm = m(a=1, b=2)
    result = pm.a
    assert result == 1


def test___getattr___raises_attribute_error_for_missing_key():
    pm = m(a=1, b=2)
    try:
        pm.c
        assert False
    except AttributeError:
        assert True


def test___getattr___accesses_nested_pmap_via_dot_notation():
    pm = m(a=m(b=5))
    result = pm.a.b
    assert result == 5


def test___getattr___works_with_keys_that_are_valid_identifiers():
    pm = m(valid_identifier=42)
    result = pm.valid_identifier
    assert result == 42


def test___getattr___raises_attribute_error_for_key_with_dot():
    pm = m(**{'key.with.dot': 100})
    try:
        pm.key.with.dot
        assert False
    except AttributeError:
        assert True


# LLM-generated content at query #10
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap_instance = pmap(buckets)
    assert len(pmap_instance) == size

def test_constructor_sets_size_attribute():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2})
    assert pmap_instance._size == 2

def test_constructor_sets_buckets_attribute():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    assert pmap_instance._buckets is not None

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert len(pmap_instance) == 0
    assert pmap_instance._size == 0

def test_constructor_handles_none_values():
    from pyrsistent import pmap
    pmap_instance = pmap({'key': None})
    assert pmap_instance['key'] is None

def test_constructor_preserves_given_key_value_pairs():
    from pyrsistent import pmap
    pmap_instance = pmap({'x': 10, 'y': 20})
    assert pmap_instance['x'] == 10
    assert pmap_instance['y'] == 20

def test_constructor_with_duplicate_keys_keeps_last_value():
    from pyrsistent import pmap
    pmap_instance = pmap([('a', 1), ('a', 2)])
    assert pmap_instance['a'] == 2

def test_constructor_creates_hashable_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    hash_value = hash(pmap_instance)
    assert isinstance(hash_value, int)

def test_constructor_supports_various_key_types():
    from pyrsistent import pmap
    pmap_instance = pmap({1: 'int', 'str': 'string', (1, 2): 'tuple'})
    assert pmap_instance[1] == 'int'
    assert pmap_instance['str'] == 'string'
    assert pmap_instance[(1, 2)] == 'tuple'

def test_constructor_creates_pmap_that_is_equal_to_equivalent_dict():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2})
    assert pmap_instance == {'a': 1, 'b': 2}


# LLM-generated content at query #11
#--------------------------

def test_turbo_mapping_predicate_at_line_7_false():
    initial = [1, 2, 3]
    pre_size = None
    result = _turbo_mapping(initial, pre_size)
    assert isinstance(result, PMap)


# LLM-generated content at query #12
#--------------------------

def test_update_with_merges_values_using_update_fn():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l + r, m(a=2))
    expected = m(a=3, b=2)
    assert result == expected

def test_update_with_keeps_leftmost_value_when_update_fn_returns_left():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    expected = m(a=1)
    assert result == expected

def test_update_with_inserts_new_key_from_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l + r, m(b=2), m(c=3))
    expected = m(a=1, b=2, c=3)
    assert result == expected

def test_update_with_handles_empty_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l + r)
    assert result == m1

def test_update_with_merges_colliding_keys_from_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1, b=1)
    result = m1.update_with(lambda l, r: l * r, m(a=2, b=2), m(a=3, b=3))
    expected = m(a=6, b=6)
    assert result == expected

def test_update_with_returns_same_instance_when_no_changes():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l, m())
    assert result is m1

def test_update_with_works_with_dict_and_pmap():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l + r, {'a': 2}, m(b=3))
    expected = m(a=3, b=3)
    assert result == expected

def test_update_with_uses_default_value_for_new_keys():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l + r if l is not None else r, m(b=2))
    expected = m(a=1, b=2)
    assert result == expected


# LLM-generated content at query #13
#--------------------------

def test_turbo_mapping_with_empty_initial_and_no_pre_size():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0
    assert dict(result) == {}

def test_turbo_mapping_with_empty_initial_and_pre_size():
    result = _turbo_mapping({}, 10)
    assert len(result) == 0
    assert dict(result) == {}

def test_turbo_mapping_with_dict_initial_and_no_pre_size():
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert dict(result) == initial

def test_turbo_mapping_with_dict_initial_and_pre_size():
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, 20)
    assert len(result) == 2
    assert dict(result) == initial

def test_turbo_mapping_with_non_mapping_initial():
    initial = [('a', 1), ('b', 2)]
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert dict(result) == {'a': 1, 'b': 2}

def test_turbo_mapping_with_collision_keys():
    class FixedHash:
        def __init__(self, value, hash_value):
            self.value = value
            self.hash_value = hash_value
        def __hash__(self):
            return self.hash_value
        def __eq__(self, other):
            return isinstance(other, FixedHash) and self.value == other.value
    key1 = FixedHash('key1', 5)
    key2 = FixedHash('key2', 5)
    initial = {key1: 'val1', key2: 'val2'}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert result[key1] == 'val1'
    assert result[key2] == 'val2'

def test_turbo_mapping_with_initial_length_hint_failure():
    class BadLenMapping:
        def __len__(self):
            raise Exception("Cannot get length")
        def items(self):
            return [('a', 1), ('b', 2)].__iter__()
    initial = BadLenMapping()
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert dict(result) == {'a': 1, 'b': 2}

def test_turbo_mapping_preserves_hash_and_equality():
    initial = {'x': 10, 'y': 20}
    pmap1 = _turbo_mapping(initial, 0)
    pmap2 = _turbo_mapping(initial, 0)
    assert pmap1 == pmap2
    assert hash(pmap1) == hash(pmap2)

def test_turbo_mapping_with_large_pre_size():
    initial = {'a': 1}
    result = _turbo_mapping(initial, 100)
    assert len(result) == 1
    assert dict(result) == initial

def test_turbo_mapping_with_small_pre_size():
    initial = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    result = _turbo_mapping(initial, 2)
    assert len(result) == 5
    assert dict(result) == initial


# LLM-generated content at query #14
#--------------------------

def test_eq_with_dict_and_different_buckets_but_same_items():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = {'a': 1, 'b': 2}
    result = m1 == m2
    assert result is True


# LLM-generated content at query #15
#--------------------------

def test_contains_with_non_tuple_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items = m.items()
    result = (1, 'a', 'extra') in items
    assert result == False

def test_contains_with_single_value_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items = m.items()
    result = 1 in items
    assert result == False

def test_contains_with_string_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items = m.items()
    result = 'key' in items
    assert result == False

def test_contains_with_none_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items = m.items()
    result = None in items
    assert result == False

def test_contains_with_list_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items = m.items()
    result = [1, 'a'] in items
    assert result == False


# LLM-generated content at query #16
#--------------------------

def test_eq_same_instance():
    from pyrsistent import pmap
    from pyrsistent._pmap import PMapItems
    m = pmap({1: 'a', 2: 'b'})
    items_view = PMapItems(m)
    result = items_view == items_view
    assert result is True

def test_eq_different_type():
    from pyrsistent import pmap
    from pyrsistent._pmap import PMapItems
    m = pmap({1: 'a', 2: 'b'})
    items_view = PMapItems(m)
    result = items_view == 'not a PMapItems'
    assert result is False

def test_eq_different_instance_same_map():
    from pyrsistent import pmap
    from pyrsistent._pmap import PMapItems
    m = pmap({1: 'a', 2: 'b'})
    items_view1 = PMapItems(m)
    items_view2 = PMapItems(m)
    result = items_view1 == items_view2
    assert result is True

def test_eq_different_instance_different_map():
    from pyrsistent import pmap
    from pyrsistent._pmap import PMapItems
    m1 = pmap({1: 'a', 2: 'b'})
    m2 = pmap({3: 'c', 4: 'd'})
    items_view1 = PMapItems(m1)
    items_view2 = PMapItems(m2)
    result = items_view1 == items_view2
    assert result is False


# LLM-generated content at query #17
#--------------------------

def test_eq_same_instance():
    m1 = m(a=1, b=2)
    result = m1 == m1
    assert result is True

def test_eq_equal_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test_eq_different_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    result = m1 == m2
    assert result is False

def test_eq_different_sizes():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2, c=3)
    result = m1 == m2
    assert result is False

def test_eq_equal_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test_eq_different_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = m1 == d
    assert result is False

def test_eq_not_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [1, 2, 3]
    assert result is NotImplemented

def test_eq_same_buckets():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1._buckets == m2._buckets
    assert result is True

def test_eq_different_buckets_same_content():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test_eq_cached_hash_mismatch():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    m1._cached_hash = hash(frozenset(m1.iteritems()))
    m2._cached_hash = hash(frozenset(m2.iteritems()))
    result = m1 == m2
    assert result is False

def test_eq_cached_hash_match():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = hash(frozenset(m1.iteritems()))
    m2._cached_hash = hash(frozenset(m2.iteritems()))
    result = m1 == m2
    assert result is True

def test_eq_with_other_mapping():
    from collections.abc import Mapping
    class TestMapping(Mapping):
        def __init__(self, d):
            self._d = d
        def __getitem__(self, key):
            return self._d[key]
        def __iter__(self):
            return iter(self._d)
        def __len__(self):
            return len(self._d)
    m1 = m(a=1, b=2)
    tm = TestMapping({'a': 1, 'b': 2})
    result = m1 == tm
    assert result is True

def test_eq_with_other_mapping_different():
    from collections.abc import Mapping
    class TestMapping(Mapping):
        def __init__(self, d):
            self._d = d
        def __getitem__(self, key):
            return self._d[key]
        def __iter__(self):
            return iter(self._d)
        def __len__(self):
            return len(self._d)
    m1 = m(a=1, b=2)
    tm = TestMapping({'a': 1, 'b': 3})
    result = m1 == tm
    assert result is False


# LLM-generated content at query #18
#--------------------------

def test_contains_with_valid_key_value_pair():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 1) in items
    assert result == True

def test_contains_with_valid_key_but_wrong_value():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 2) in items
    assert result == False

def test_contains_with_key_not_in_map():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('c', 1) in items
    assert result == False

def test_contains_with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = 'a' in items
    assert result == False

def test_contains_with_tuple_of_wrong_length():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 1, 'extra') in items
    assert result == False

def test_contains_with_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    items = m.items()
    result = ('a', 1) in items
    assert result == False


# LLM-generated content at query #19
#--------------------------

def test___contains___with_valid_key_value_pair_present():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1) in items_view
    assert result == True

def test___contains___with_valid_key_value_pair_absent():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 2) in items_view
    assert result == False

def test___contains___with_key_not_in_map():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('c', 1) in items_view
    assert result == False

def test___contains___with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = 'a' in items_view
    assert result == False

def test___contains___with_tuple_of_wrong_length():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1, 'extra') in items_view
    assert result == False

def test___contains___with_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    result = ('a', 1) in items_view
    assert result == False


# LLM-generated content at query #20
#--------------------------

def test___contains___with_existing_key_value_pair():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items = m.items()
    result = ("a", 1) in items
    assert result is True

def test___contains___with_existing_key_but_different_value():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items = m.items()
    result = ("a", 2) in items
    assert result is False

def test___contains___with_non_existing_key():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items = m.items()
    result = ("c", 1) in items
    assert result is False

def test___contains___with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items = m.items()
    result = "a" in items
    assert result is False

def test___contains___with_wrong_length_tuple():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items = m.items()
    result = ("a", 1, 3) in items
    assert result is False

def test___contains___with_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    items = m.items()
    result = ("a", 1) in items
    assert result is False


# LLM-generated content at query #21
#--------------------------

def test_eq_with_different_cached_hash():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m1._cached_hash = 123
    m2._cached_hash = 456
    result = m1 == m2
    assert result is False


# LLM-generated content at query #22
#--------------------------

def test_eq_same_instance():
    m1 = m(a=1, b=2)
    result = m1 == m1
    assert result is True

def test_eq_equal_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test_eq_different_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    result = m1 == m2
    assert result is False

def test_eq_pmap_vs_dict_equal():
    pm = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = pm == d
    assert result is True

def test_eq_pmap_vs_dict_not_equal():
    pm = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = pm == d
    assert result is False

def test_eq_pmap_vs_dict_different_length():
    pm = m(a=1, b=2, c=3)
    d = {'a': 1, 'b': 2}
    result = pm == d
    assert result is False

def test_eq_pmap_vs_other_mapping_equal():
    from collections import OrderedDict
    pm = m(a=1, b=2)
    od = OrderedDict([('a', 1), ('b', 2)])
    result = pm == od
    assert result is True

def test_eq_pmap_vs_non_mapping():
    pm = m(a=1, b=2)
    result = pm == [1, 2]
    assert result is NotImplemented

def test_eq_pmap_vs_dict_with_same_hash():
    pm1 = m(a=1, b=2)
    pm1._cached_hash = 123
    pm2 = m(a=1, b=2)
    pm2._cached_hash = 123
    result = pm1 == pm2
    assert result is True

def test_eq_pmap_vs_dict_with_different_hash():
    pm1 = m(a=1, b=2)
    pm1._cached_hash = 123
    pm2 = m(a=1, b=2)
    pm2._cached_hash = 456
    result = pm1 == pm2
    assert result is False

def test_eq_pmap_with_identical_buckets():
    from pyrsistent import pvector
    bucket = [('a', 1), ('b', 2)]
    buckets = pvector([bucket, None])
    pm1 = PMap(2, buckets)
    pm2 = PMap(2, buckets)
    result = pm1 == pm2
    assert result is True

def test_eq_pmap_with_different_buckets_same_content():
    pm1 = m(a=1, b=2)
    pm2 = m(a=1, b=2)
    result = pm1 == pm2
    assert result is True

def test_eq_pmap_vs_dict_with_same_items_different_order():
    pm = m(a=1, b=2, c=3)
    d = {'c': 3, 'a': 1, 'b': 2}
    result = pm == d
    assert result is True


# LLM-generated content at query #23
#--------------------------

def test_turbo_mapping_with_empty_initial():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0
    assert dict(result) == {}

def test_turbo_mapping_with_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2}, 16)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_turbo_mapping_without_pre_size():
    result = _turbo_mapping({'x': 10, 'y': 20}, 0)
    assert len(result) == 2
    assert result['x'] == 10
    assert result['y'] == 20

def test_turbo_mapping_with_non_mapping_initial():
    result = _turbo_mapping([('key1', 100), ('key2', 200)], 0)
    assert len(result) == 2
    assert result['key1'] == 100
    assert result['key2'] == 200

def test_turbo_mapping_with_collision_handling():
    class FixedHash:
        def __init__(self, value, hash_value):
            self.value = value
            self.hash_value = hash_value
        def __hash__(self):
            return self.hash_value
        def __eq__(self, other):
            return isinstance(other, FixedHash) and self.value == other.value
    obj1 = FixedHash('a', 5)
    obj2 = FixedHash('b', 5)
    result = _turbo_mapping({obj1: 1, obj2: 2}, 4)
    assert len(result) == 2
    assert result[obj1] == 1
    assert result[obj2] == 2

def test_turbo_mapping_preserves_hashability():
    result = _turbo_mapping({'a': 1, 'b': 2}, 0)
    hash1 = hash(result)
    result2 = result.set('c', 3)
    hash2 = hash(result2)
    assert hash1 != hash2

def test_turbo_mapping_contains_key():
    result = _turbo_mapping({'a': 1, 'b': 2}, 0)
    assert 'a' in result
    assert 'c' not in result

def test_turbo_mapping_iteration():
    result = _turbo_mapping({'a': 1, 'b': 2}, 0)
    keys = list(result)
    assert set(keys) == {'a', 'b'}

def test_turbo_mapping_with_zero_pre_size_and_empty():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0

def test_turbo_mapping_with_large_pre_size():
    result = _turbo_mapping({'a': 1}, 100)
    assert len(result) == 1
    assert result['a'] == 1


# LLM-generated content at query #24
#--------------------------

def test_eq_same_instance():
    m1 = m(a=1, b=2)
    result = m1 == m1
    assert result is True

def test_eq_equal_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test_eq_different_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    result = m1 == m2
    assert result is False

def test_eq_different_sizes():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2, c=3)
    result = m1 == m2
    assert result is False

def test_eq_equal_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test_eq_different_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = m1 == d
    assert result is False

def test_eq_not_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [1, 2]
    assert result is NotImplemented

def test_eq_same_buckets():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = 123
    m2._cached_hash = 123
    result = m1 == m2
    assert result is True

def test_eq_different_cached_hash():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = 123
    m2._cached_hash = 456
    result = m1 == m2
    assert result is False

def test_eq_with_other_mapping():
    class CustomMapping(Mapping):
        def __init__(self, d):
            self._d = d
        def __getitem__(self, key):
            return self._d[key]
        def __iter__(self):
            return iter(self._d)
        def __len__(self):
            return len(self._d)
    m1 = m(a=1, b=2)
    cm = CustomMapping({'a': 1, 'b': 2})
    result = m1 == cm
    assert result is True


# LLM-generated content at query #25
#--------------------------

def test_update_with_merges_values_using_update_fn():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l + r, m(a=2))
    expected = m(a=3, b=2)
    assert result == expected

def test_update_with_keeps_leftmost_value_when_update_fn_returns_left():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    expected = m(a=1)
    assert result == expected

def test_update_with_inserts_new_key_from_maps():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: r, m(b=2), {'c': 3})
    expected = m(a=1, b=2, c=3)
    assert result == expected

def test_update_with_handles_multiple_maps_and_merge_fn():
    from pyrsistent import m
    m1 = m(a=1, b=1)
    result = m1.update_with(lambda l, r: l * r, m(a=2, b=2), {'a': 3, 'c': 5})
    expected = m(a=6, b=2, c=5)
    assert result == expected

def test_update_with_returns_same_instance_if_no_changes():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m())
    assert result is m1

def test_update_with_on_empty_pmap():
    from pyrsistent import m
    m1 = m()
    result = m1.update_with(lambda l, r: l + r, {'x': 10}, m(y=20))
    expected = m(x=10, y=20)
    assert result == expected

def test_update_with_uses_update_fn_only_for_existing_keys():
    from pyrsistent import m
    call_count = 0
    def counting_update_fn(l, r):
        nonlocal call_count
        call_count += 1
        return l + r
    m1 = m(a=1, b=2)
    result = m1.update_with(counting_update_fn, m(a=10, c=30))
    assert call_count == 1
    expected = m(a=11, b=2, c=30)
    assert result == expected


# LLM-generated content at query #26
#--------------------------

def test_update_with_merge_function():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1, b=2)
    m2 = m1.update_with(add, m(a=2))
    assert m2 == {'a': 3, 'b': 2}

def test_update_with_keep_leftmost():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(a=2), {'a':3})
    assert m2 == {'a': 1}

def test_update_with_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2, c=3), {'a': 10, 'd': 4})
    assert m2 == {'a': 13, 'b': 2, 'c': 3, 'd': 4}

def test_update_with_empty_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r)
    assert m2 == m1

def test_update_with_new_key():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(b=2))
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_overwrites_existing():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r * 2, m(a=3))
    assert m2 == {'a': 6, 'b': 2}

def test_update_with_identity_function():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r, m(a=5, c=7))
    assert m2 == {'a': 5, 'b': 2, 'c': 7}

def test_update_with_constant_function():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: 99, m(a=5, c=7))
    assert m2 == {'a': 99, 'b': 2, 'c': 99}


# LLM-generated content at query #27
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap, m
    size = 2
    buckets = (('a', 1), ('b', 2))
    pmap_instance = pmap({'a': 1, 'b': 2})
    assert pmap_instance._size == size
    assert dict(pmap_instance._buckets) == dict(buckets)

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert pmap_instance._size == 0
    assert len(pmap_instance._buckets) > 0

def test_constructor_handles_single_key_value_pair():
    from pyrsistent import pmap
    pmap_instance = pmap({'key': 'value'})
    assert pmap_instance._size == 1
    assert pmap_instance['key'] == 'value'

def test_constructor_creates_pmap_with_multiple_entries():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    assert pmap_instance._size == 3
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2
    assert pmap_instance['c'] == 3

def test_constructor_preserves_hash_collisions_handling():
    from pyrsistent import pmap
    class SameHash:
        def __hash__(self):
            return 1
    key1 = SameHash()
    key2 = SameHash()
    pmap_instance = pmap({key1: 'first', key2: 'second'})
    assert pmap_instance._size == 2
    assert pmap_instance[key1] == 'first'
    assert pmap_instance[key2] == 'second'

def test_constructor_creates_pmap_from_empty_dict():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert pmap_instance._size == 0
    assert list(pmap_instance) == []

def test_constructor_creates_pmap_with_none_value():
    from pyrsistent import pmap
    pmap_instance = pmap({'key': None})
    assert pmap_instance._size == 1
    assert pmap_instance['key'] is None

def test_constructor_creates_pmap_with_zero_size_and_empty_buckets():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert pmap_instance._size == 0
    assert all(bucket is None or len(bucket) == 0 for bucket in pmap_instance._buckets)

def test_constructor_creates_pmap_that_is_hashable():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2})
    hash_value = hash(pmap_instance)
    assert isinstance(hash_value, int)

def test_constructor_creates_pmap_with_identical_buckets_as_input():
    from pyrsistent import pmap, m
    pmap1 = pmap({'x': 10, 'y': 20})
    pmap2 = PMap(pmap1._size, pmap1._buckets)
    assert pmap2._size == pmap1._size
    assert pmap2._buckets == pmap1._buckets

def test_constructor_creates_pmap_with_correct_internal_structure():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2})
    assert hasattr(pmap_instance, '_size')
    assert hasattr(pmap_instance, '_buckets')
    assert pmap_instance._size == 2
    assert isinstance(pmap_instance._buckets, type(pmap_instance._buckets))

def test_constructor_creates_pmap_that_is_immutable():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    try:
        pmap_instance['a'] = 2
        assert False
    except TypeError:
        assert True

def test_constructor_creates_pmap_with_mixed_key_types():
    from pyrsistent import pmap
    pmap_instance = pmap({1: 'int', 'str': 'string', (1, 2): 'tuple'})
    assert pmap_instance._size == 3
    assert pmap_instance[1] == 'int'
    assert pmap_instance['str'] == 'string'
    assert pmap_instance[(1, 2)] == 'tuple'

def test_constructor_creates_pmap_with_duplicate_keys_last_wins():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'a': 2})
    assert pmap_instance._size == 1
    assert pmap_instance['a'] == 2


# LLM-generated content at query #28
#--------------------------

def test_contains_with_valid_key_value_pair():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("a", 1) in items_view
    assert result == True

def test_contains_with_valid_key_value_pair_second_item():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("b", 2) in items_view
    assert result == True

def test_contains_with_valid_key_but_different_value():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("a", 2) in items_view
    assert result == False

def test_contains_with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = "a" in items_view
    assert result == False

def test_contains_with_tuple_of_wrong_length():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("a", 1, "extra") in items_view
    assert result == False

def test_contains_with_none_argument():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = None in items_view
    assert result == False

def test_contains_with_non_existent_key():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("c", 1) in items_view
    assert result == False

def test_contains_with_existing_key_and_matching_value():
    from pyrsistent import pmap
    m = pmap({"x": 10, "y": 20})
    items_view = m.items()
    result = ("x", 10) in items_view
    assert result == True

def test_contains_with_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    result = ("any", 1) in items_view
    assert result == False


# LLM-generated content at query #29
#--------------------------

def test_eq_with_different_cached_hash():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m1._cached_hash = 123
    m2._cached_hash = 456
    result = m1 == m2
    assert result == False


# LLM-generated content at query #30
#--------------------------

def test_constructor_creates_pmap_with_correct_size_and_buckets():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    assert m._size == 2
    assert len(m._buckets) > 0

def test_constructor_returns_pmap_instance():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    assert isinstance(m, pmap)

def test_constructor_with_empty_dict():
    from pyrsistent import pmap
    m = pmap({})
    assert m._size == 0
    assert len(m._buckets) > 0

def test_constructor_preserves_key_value_pairs():
    from pyrsistent import pmap
    m = pmap({'x': 10, 'y': 20})
    assert m['x'] == 10
    assert m['y'] == 20

def test_constructor_handles_multiple_entries():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    assert m._size == 3
    assert m['a'] == 1
    assert m['b'] == 2
    assert m['c'] == 3

def test_constructor_with_none_value():
    from pyrsistent import pmap
    m = pmap({'key': None})
    assert m['key'] is None

def test_constructor_with_false_value():
    from pyrsistent import pmap
    m = pmap({'key': False})
    assert m['key'] is False

def test_constructor_with_zero_value():
    from pyrsistent import pmap
    m = pmap({'key': 0})
    assert m['key'] == 0

def test_constructor_with_empty_string_key():
    from pyrsistent import pmap
    m = pmap({'': 'empty'})
    assert m[''] == 'empty'

def test_constructor_with_integer_keys():
    from pyrsistent import pmap
    m = pmap({1: 'one', 2: 'two'})
    assert m[1] == 'one'
    assert m[2] == 'two'

def test_constructor_with_tuple_keys():
    from pyrsistent import pmap
    m = pmap({(1, 2): 'tuple'})
    assert m[(1, 2)] == 'tuple'

def test_constructor_creates_distinct_instances():
    from pyrsistent import pmap
    m1 = pmap({'a': 1})
    m2 = pmap({'a': 1})
    assert m1 is not m2
    assert m1 == m2

def test_constructor_handles_duplicate_keys():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'a': 2})
    assert m['a'] == 2
    assert m._size == 1

def test_constructor_with_single_key_value():
    from pyrsistent import pmap
    m = pmap({'single': 'value'})
    assert m._size == 1
    assert m['single'] == 'value'

def test_constructor_with_large_dict():
    from pyrsistent import pmap
    large_dict = {str(i): i for i in range(100)}
    m = pmap(large_dict)
    assert m._size == 100
    for i in range(100):
        assert m[str(i)] == i


# LLM-generated content at query #31
#--------------------------

def test_eq_same_instance():
    m1 = m(a=1, b=2)
    result = m1 == m1
    assert result is True

def test_eq_equal_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test_eq_different_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    result = m1 == m2
    assert result is False

def test_eq_pmap_vs_dict_equal():
    pm = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = pm == d
    assert result is True

def test_eq_pmap_vs_dict_not_equal():
    pm = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = pm == d
    assert result is False

def test_eq_different_lengths():
    m1 = m(a=1, b=2, c=3)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is False

def test_eq_with_non_mapping():
    pm = m(a=1, b=2)
    result = pm == [1, 2]
    assert result is NotImplemented

def test_eq_cached_hash_mismatch():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    hash(m1)
    hash(m2)
    result = m1 == m2
    assert result is False

def test_eq_same_buckets():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    result = m1 == m2
    assert result is True

def test_eq_other_mapping_type():
    from collections.abc import Mapping
    class CustomMapping(Mapping):
        def __init__(self, d):
            self._d = d
        def __getitem__(self, key):
            return self._d[key]
        def __iter__(self):
            return iter(self._d)
        def __len__(self):
            return len(self._d)
    pm = m(a=1, b=2)
    cm = CustomMapping({'a': 1, 'b': 2})
    result = pm == cm
    assert result is True

def test_eq_other_mapping_type_not_equal():
    from collections.abc import Mapping
    class CustomMapping(Mapping):
        def __init__(self, d):
            self._d = d
        def __getitem__(self, key):
            return self._d[key]
        def __iter__(self):
            return iter(self._d)
        def __len__(self):
            return len(self._d)
    pm = m(a=1, b=2)
    cm = CustomMapping({'a': 1, 'b': 3})
    result = pm == cm
    assert result is False


# LLM-generated content at query #32
#--------------------------

def test_contains_with_valid_key_value_pair():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("a", 1) in items_view
    assert result == True

def test_contains_with_key_not_in_map():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("c", 1) in items_view
    assert result == False

def test_contains_with_wrong_value_for_key():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("a", 3) in items_view
    assert result == False

def test_contains_with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = "a" in items_view
    assert result == False

def test_contains_with_tuple_of_wrong_length():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("a", 1, "extra") in items_view
    assert result == False

def test_contains_with_empty_tuple():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = () in items_view
    assert result == False

def test_contains_with_non_iterable_argument():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = 42 in items_view
    assert result == False

def test_contains_with_none_argument():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = None in items_view
    assert result == False

def test_contains_with_tuple_key_not_hashable():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ([1, 2], 3) in items_view
    assert result == False

def test_contains_with_exact_match_for_multiple_items():
    from pyrsistent import pmap
    m = pmap({"x": 10, "y": 20, "z": 30})
    items_view = m.items()
    result = ("y", 20) in items_view
    assert result == True


# LLM-generated content at query #33
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('a', 1), ('b', 2))
    pmap_instance = pmap(buckets)
    assert len(pmap_instance) == size
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap()
    assert len(pmap_instance) == 0
    assert dict(pmap_instance) == {}

def test_constructor_handles_nested_pmaps():
    from pyrsistent import pmap
    inner = pmap({'x': 10})
    outer = pmap({'inner': inner})
    assert outer['inner']['x'] == 10

def test_constructor_preserves_hash_collisions():
    from pyrsistent import pmap
    class BadHash:
        def __hash__(self):
            return 1
    key1 = BadHash()
    key2 = BadHash()
    pmap_instance = pmap({key1: 'value1', key2: 'value2'})
    assert pmap_instance[key1] == 'value1'
    assert pmap_instance[key2] == 'value2'

def test_constructor_with_keyword_arguments():
    from pyrsistent import m
    pmap_instance = m(a=1, b=2)
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2
    assert len(pmap_instance) == 2

def test_constructor_from_dict():
    from pyrsistent import pmap
    d = {'key1': 'val1', 'key2': 'val2'}
    pmap_instance = pmap(d)
    assert pmap_instance['key1'] == 'val1'
    assert pmap_instance['key2'] == 'val2'

def test_constructor_from_iterable_of_pairs():
    from pyrsistent import pmap
    pairs = [('k1', 'v1'), ('k2', 'v2')]
    pmap_instance = pmap(pairs)
    assert pmap_instance['k1'] == 'v1'
    assert pmap_instance['k2'] == 'v2'

def test_constructor_does_not_share_internal_state():
    from pyrsistent import pmap
    d = {'a': 1}
    pmap1 = pmap(d)
    d['b'] = 2
    pmap2 = pmap(d)
    assert len(pmap1) == 1
    assert len(pmap2) == 2

def test_constructor_with_duplicate_keys_keeps_last():
    from pyrsistent import pmap
    pairs = [('a', 1), ('a', 2)]
    pmap_instance = pmap(pairs)
    assert pmap_instance['a'] == 2

def test_constructor_with_none_key():
    from pyrsistent import pmap
    pmap_instance = pmap({None: 'value'})
    assert pmap_instance[None] == 'value'

def test_constructor_with_false_key():
    from pyrsistent import pmap
    pmap_instance = pmap({False: 'false_value'})
    assert pmap_instance[False] == 'false_value'

def test_constructor_with_true_key():
    from pyrsistent import pmap
    pmap_instance = pmap({True: 'true_value'})
    assert pmap_instance[True] == 'true_value'

def test_constructor_with_int_key():
    from pyrsistent import pmap
    pmap_instance = pmap({42: 'answer'})
    assert pmap_instance[42] == 'answer'

def test_constructor_with_float_key():
    from pyrsistent import pmap
    pmap_instance = pmap({3.14: 'pi'})
    assert pmap_instance[3.14] == 'pi'

def test_constructor_with_tuple_key():
    from pyrsistent import pmap
    pmap_instance = pmap({(1, 2): 'tuple_value'})
    assert pmap_instance[(1, 2)] == 'tuple_value'

def test_constructor_with_empty_dict():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert len(pmap_instance) == 0
    assert list(pmap_instance) == []

def test_constructor_with_large_dict():
    from pyrsistent import pmap
    large_dict = {i: i*2 for i in range(1000)}
    pmap_instance = pmap(large_dict)
    assert len(pmap_instance) == 1000
    assert pmap_instance[500] == 1000

def test_constructor_preserves_order_of_insertion_iteration():
    from pyrsistent import pmap
    d = {'z': 1, 'a': 2, 'm': 3}
    pmap_instance = pmap(d)
    keys = list(pmap_instance.keys())
    assert set(keys) == {'z', 'a', 'm'}

def test_constructor_with_custom_hashable_objects():
    from pyrsistent import pmap
    class Custom:
        def __init__(self, val):
            self.val = val
        def __hash__(self):
            return hash(self.val)
        def __eq__(self, other):
            return isinstance(other, Custom) and self.val == other.val
    c1 = Custom('hello')
    c2 = Custom('world')
    pmap_instance = pmap({c1: 1, c2: 2})
    assert pmap_instance[c1] == 1
    assert pmap_instance[c2] == 2


# LLM-generated content at query #34
#--------------------------

def test_eq_with_different_cached_hash():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m1._cached_hash = 123
    m2._cached_hash = 456
    result = m1 == m2
    assert result == False


