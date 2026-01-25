####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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

def test_constructor_handles_nested_pmap():
    from pyrsistent import pmap
    nested = {'a': {'b': 1}}
    pmap_instance = pmap(nested)
    assert pmap_instance['a']['b'] == 1

def test_constructor_with_m_factory_function():
    from pyrsistent import m
    pmap_instance = m(a=1, b=2)
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2
    assert len(pmap_instance) == 2

def test_constructor_pmap_from_dict():
    from pyrsistent import pmap
    d = {'x': 10, 'y': 20}
    pmap_instance = pmap(d)
    assert pmap_instance['x'] == 10
    assert pmap_instance['y'] == 20

def test_constructor_pmap_from_pmap():
    from pyrsistent import pmap
    original = pmap({'key': 'value'})
    new = pmap(original)
    assert new['key'] == 'value'
    assert len(new) == 1

def test_constructor_with_iterable_of_pairs():
    from pyrsistent import pmap
    pairs = [('k1', 'v1'), ('k2', 'v2')]
    pmap_instance = pmap(pairs)
    assert pmap_instance['k1'] == 'v1'
    assert pmap_instance['k2'] == 'v2'

def test_constructor_hash_is_cached_after_access():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    hash1 = hash(pmap_instance)
    hash2 = hash(pmap_instance)
    assert hash1 == hash2
    assert hasattr(pmap_instance, '_cached_hash')

def test_constructor_creates_pmap_with_correct_slots():
    pmap_instance = PMap.__new__(PMap)
    pmap_instance._size = 0
    pmap_instance._buckets = ()
    assert pmap_instance._size == 0
    assert pmap_instance._buckets == ()
    assert not hasattr(pmap_instance, '_cached_hash')

def test_constructor_via_new_with_size_and_buckets():
    size = 3
    buckets = (('x', 1), ('y', 2), ('z', 3))
    pmap_instance = PMap.__new__(PMap, size, buckets)
    pmap_instance._size = size
    pmap_instance._buckets = buckets
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets


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

def test___contains___with_wrong_length_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1, 3) in items_view
    assert result == False

def test___contains___with_empty_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = () in items_view
    assert result == False

def test___contains___with_single_element_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a',) in items_view
    assert result == False

def test___contains___with_none_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = None in items_view
    assert result == False


# LLM-generated content at query #3
#--------------------------

def test_constructor_creates_pmap_with_correct_size_and_buckets():
    size = 2
    buckets = (('a', 1), ('b', 2))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_zero_size_and_empty_buckets():
    size = 0
    buckets = ()
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_large_size_and_buckets():
    size = 1000
    buckets = tuple((str(i), i) for i in range(size))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_none_buckets():
    size = 0
    buckets = None
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_single_bucket():
    size = 1
    buckets = (('key', 'value'),)
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets


# LLM-generated content at query #4
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

def test_turbo_mapping_with_initial_having_collisions():
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

def test_turbo_mapping_with_pre_size_smaller_than_initial():
    initial = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    result = _turbo_mapping(initial, 2)
    assert len(result) == 4
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

def test_turbo_mapping_with_initial_as_mapping_subclass():
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
    initial = CustomMapping({'x': 10, 'y': 20})
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert dict(result) == {'x': 10, 'y': 20}

def test_turbo_mapping_preserves_hash_and_equality():
    initial = {'foo': 'bar', 'baz': 'qux'}
    result = _turbo_mapping(initial, 0)
    other = _turbo_mapping(initial, 0)
    assert result == other
    assert hash(result) == hash(other)

def test_turbo_mapping_with_large_initial():
    initial = {i: i*2 for i in range(100)}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 100
    for i in range(100):
        assert result[i] == i*2


# LLM-generated content at query #5
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

def test_update_with_inserts_new_key_from_single_map():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(b=2))
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_inserts_new_keys_from_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(b=2), {'c': 3})
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_overwrites_with_rightmost_when_update_fn_returns_right():
    from pyrsistent import m
    m1 = m(a=1, b=1)
    m2 = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert m2 == {'a': 3, 'b': 1}

def test_update_with_on_empty_map():
    from pyrsistent import m
    m1 = m()
    m2 = m1.update_with(lambda l, r: l + r, m(a=1, b=2))
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_no_maps_returns_same_map():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r)
    assert m2 is m1

def test_update_with_using_operator_add():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1, b=2)
    m2 = m1.update_with(add, m(a=2, c=3))
    assert m2 == {'a': 3, 'b': 2, 'c': 3}

def test_update_with_handles_none_values():
    from pyrsistent import m
    m1 = m(a=None)
    m2 = m1.update_with(lambda l, r: r if r is not None else l, m(a=1))
    assert m2 == {'a': 1}

def test_update_with_returns_new_map_when_changes_made():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(a=2))
    assert m1 == {'a': 1}
    assert m2 == {'a': 2}
    assert m1 is not m2

def test_update_with_returns_same_map_when_no_changes():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(a=1))
    assert m2 is m1


# LLM-generated content at query #6
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('a', 1), ('b', 2))
    m = pmap({'a': 1, 'b': 2})
    assert m._size == size
    assert dict(m._buckets) == dict(buckets)

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    assert m._size == 0
    assert len(m._buckets) == 0

def test_constructor_creates_pmap_with_single_element():
    from pyrsistent import pmap
    m = pmap({'key': 'value'})
    assert m._size == 1
    assert m['key'] == 'value'

def test_constructor_creates_pmap_with_multiple_elements():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    assert m._size == 3
    assert m['a'] == 1
    assert m['b'] == 2
    assert m['c'] == 3

def test_constructor_handles_none_values():
    from pyrsistent import pmap
    m = pmap({'key': None})
    assert m._size == 1
    assert m['key'] is None

def test_constructor_creates_pmap_with_colliding_keys():
    from pyrsistent import pmap
    class SameHash:
        def __init__(self, value):
            self.value = value
        def __hash__(self):
            return 1
        def __eq__(self, other):
            return isinstance(other, SameHash) and self.value == other.value
    key1 = SameHash('a')
    key2 = SameHash('b')
    m = pmap({key1: 1, key2: 2})
    assert m._size == 2
    assert m[key1] == 1
    assert m[key2] == 2

def test_constructor_creates_pmap_with_immutable_keys():
    from pyrsistent import pmap
    m = pmap({(1, 2): 'tuple_key', frozenset([3, 4]): 'frozenset_key'})
    assert m._size == 2
    assert m[(1, 2)] == 'tuple_key'
    assert m[frozenset([3, 4])] == 'frozenset_key'

def test_constructor_creates_pmap_from_another_pmap():
    from pyrsistent import pmap
    original = pmap({'x': 10, 'y': 20})
    new = pmap(original)
    assert new._size == 2
    assert new['x'] == 10
    assert new['y'] == 20

def test_constructor_creates_pmap_with_zero_size_and_empty_buckets():
    from pyrsistent import pmap
    m = pmap({})
    assert m._size == 0
    assert len(m._buckets) == 0

def test_constructor_creates_pmap_with_negative_hash_keys():
    from pyrsistent import pmap
    m = pmap({-1: 'minus_one', -2: 'minus_two'})
    assert m._size == 2
    assert m[-1] == 'minus_one'
    assert m[-2] == 'minus_two'


# LLM-generated content at query #7
#--------------------------

def test_update_with_does_not_call_update_fn_when_key_not_in_evolver():
    from pyrsistent import m
    def update_fn(left, right):
        raise AssertionError("update_fn should not be called")
    m1 = m(a=1)
    result = m1.update_with(update_fn, m(b=2))
    expected = m(a=1, b=2)
    assert result == expected


# LLM-generated content at query #8
#--------------------------

def test_eq_same_instance():
    m1 = m(a=1, b=2)
    result = m1 == m1
    assert result is True

def test_eq_equal_pmap_same_buckets():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test_eq_equal_pmap_different_buckets():
    m1 = m(a=1, b=2)
    evolver = m1.evolver()
    evolver.set('c', 3)
    evolver.remove('c')
    m2 = evolver.persistent()
    result = m1 == m2
    assert result is True

def test_eq_equal_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test_eq_not_equal_different_length():
    m1 = m(a=1, b=2)
    d = {'a': 1}
    result = m1 == d
    assert result is False

def test_eq_not_equal_different_keys():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'c': 2}
    result = m1 == d
    assert result is False

def test_eq_not_equal_different_values():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = m1 == d
    assert result is False

def test_eq_not_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [('a', 1), ('b', 2)]
    assert result is False

def test_eq_cached_hash_mismatch():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    m1_hash = hash(m1)
    m2_hash = hash(m2)
    result = m1 == m2
    assert result is False

def test_eq_same_cached_hash_equal():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1_hash = hash(m1)
    m2_hash = hash(m2)
    result = m1 == m2
    assert result is True

def test_eq_other_mapping_protocol():
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
    m1 = m(a=1, b=2)
    custom = CustomMapping({'a': 1, 'b': 2})
    result = m1 == custom
    assert result is True

def test_eq_other_mapping_protocol_not_equal():
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
    m1 = m(a=1, b=2)
    custom = CustomMapping({'a': 1, 'b': 3})
    result = m1 == custom
    assert result is False


# LLM-generated content at query #9
#--------------------------

def test_turbo_mapping_predicate_false():
    initial = []
    pre_size = None
    result = _turbo_mapping(initial, pre_size)
    assert result is not None


# LLM-generated content at query #10
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


# LLM-generated content at query #11
#--------------------------

def test_contains_predicate_true():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("a", 1) in items_view
    assert result == True


# LLM-generated content at query #12
#--------------------------

def test_eq_with_dict_and_different_iteritems():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    other = {'a': 1, 'b': 2}
    result = m1 == other
    assert result == True

def test_eq_with_dict_and_different_length():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    other = {'a': 1}
    result = m1 == other
    assert result == False

def test_eq_with_dict_and_different_keys():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    other = {'a': 1, 'c': 2}
    result = m1 == other
    assert result == False

def test_eq_with_dict_and_different_values():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    other = {'a': 1, 'b': 3}
    result = m1 == other
    assert result == False

def test_eq_with_dict_and_same_items():
    from pyrsistent import m
    m1 = m(a=1, b=2, c=3)
    other = {'a': 1, 'b': 2, 'c': 3}
    result = m1 == other
    assert result == True


# LLM-generated content at query #13
#--------------------------

def test_contains_with_non_tuple_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'a', 'extra') in items_view
    assert result == False


# LLM-generated content at query #14
#--------------------------

def test_contains_returns_false_on_invalid_arg():
    from pyrsistent import pmap
    m = pmap({"a": 1})
    items = m.items()
    result = (1, 2, 3) in items
    assert result == False


# LLM-generated content at query #15
#--------------------------

def test_constructor_creates_pmap_with_correct_size_and_buckets():
    size = 2
    buckets = (('a', 1), ('b', 2))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_zero_size_and_empty_buckets():
    size = 0
    buckets = ()
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_large_size_and_buckets():
    size = 100
    buckets = tuple((str(i), i) for i in range(size))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_none_buckets():
    size = 0
    buckets = None
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_single_element_buckets():
    size = 1
    buckets = (('key', 'value'),)
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets


# LLM-generated content at query #16
#--------------------------

def test_turbo_mapping_predicate_at_line_7_false():
    initial = [1, 2, 3]
    pre_size = None
    result = _turbo_mapping(initial, pre_size)
    assert isinstance(result, PMap)


# LLM-generated content at query #17
#--------------------------

def test_eq_with_dict_and_different_cached_hash():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m1._cached_hash = 123
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test_eq_with_dict_and_same_cached_hash():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m1._cached_hash = 456
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test_eq_with_dict_and_no_cached_hash():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test_eq_with_dict_and_different_buckets():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test_eq_with_dict_and_different_values():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    d = {'a': 1, 'b': 3}
    result = m1 == d
    assert result is False


# LLM-generated content at query #18
#--------------------------

def test___contains___with_existing_key_value_pair():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1) in items_view
    assert result == True

def test___contains___with_existing_key_but_different_value():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 2) in items_view
    assert result == False

def test___contains___with_non_existing_key():
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

def test___contains___with_wrong_length_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1, 3) in items_view
    assert result == False

def test___contains___with_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    result = ('a', 1) in items_view
    assert result == False


# LLM-generated content at query #19
#--------------------------

def test_constructor_creates_pmap_with_correct_size_and_buckets():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    assert m._size == 2
    assert len(m._buckets) > 0

def test_constructor_returns_same_instance_for_empty_pmap():
    from pyrsistent import pmap
    m1 = pmap({})
    m2 = pmap({})
    assert m1 is m2

def test_constructor_preserves_key_value_pairs():
    from pyrsistent import pmap
    m = pmap({'x': 10, 'y': 20})
    assert m['x'] == 10
    assert m['y'] == 20

def test_constructor_handles_none_value():
    from pyrsistent import pmap
    m = pmap({'key': None})
    assert m['key'] is None

def test_constructor_with_initial_data_from_dict():
    from pyrsistent import pmap
    d = {'a': 1, 'b': 2, 'c': 3}
    m = pmap(d)
    assert len(m) == 3
    assert m['a'] == 1
    assert m['b'] == 2
    assert m['c'] == 3

def test_constructor_with_initial_data_from_pmap():
    from pyrsistent import pmap
    m1 = pmap({'a': 1})
    m2 = pmap(m1)
    assert m2['a'] == 1
    assert m1 is m2

def test_constructor_creates_hashable_pmap():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    hash_value = hash(m)
    assert isinstance(hash_value, int)

def test_constructor_with_empty_dict():
    from pyrsistent import pmap
    m = pmap({})
    assert len(m) == 0
    assert dict(m) == {}

def test_constructor_with_single_key():
    from pyrsistent import pmap
    m = pmap({'single': 42})
    assert m['single'] == 42
    assert len(m) == 1

def test_constructor_with_multiple_identical_keys_last_wins():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'a': 2})
    assert m['a'] == 2
    assert len(m) == 1


# LLM-generated content at query #20
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

def test_eq_different_lengths():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2, c=3)
    result = m1 == m2
    assert result is False

def test_eq_not_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [1, 2, 3]
    assert result is NotImplemented

def test_eq_cached_hash_mismatch():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = hash(frozenset(m1.iteritems()))
    m2._cached_hash = 12345
    result = m1 == m2
    assert result is False

def test_eq_same_buckets():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1._buckets == m2._buckets
    assert result is True
    result = m1 == m2
    assert result is True


# LLM-generated content at query #21
#--------------------------

def test_turbo_mapping_predicate_at_line_7_false():
    initial = [1, 2, 3]
    pre_size = None
    result = _turbo_mapping(initial, pre_size)
    assert result is not None


# LLM-generated content at query #22
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

def test_constructor_sets_size_and_buckets_correctly():
    size = 3
    buckets = (('a', 1), ('b', 2), ('c', 3))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_with_empty_pmap():
    pmap = PMap(0, ())
    assert pmap._size == 0
    assert pmap._buckets == ()

def test_constructor_sets_correct_size_for_non_empty_pmap():
    buckets = (('x', 10), ('y', 20))
    pmap = PMap(len(buckets), buckets)
    assert pmap._size == len(buckets)

def test_constructor_preserves_bucket_structure():
    bucket_structure = (('k1', 'v1'), None, ('k2', 'v2'))
    pmap = PMap(2, bucket_structure)
    assert pmap._buckets == bucket_structure


# LLM-generated content at query #23
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('a', 1), ('b', 2))
    pmap_instance = pmap({'a': 1, 'b': 2})
    assert pmap_instance._size == size
    assert dict(pmap_instance._buckets) == dict(buckets)

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap()
    assert pmap_instance._size == 0
    assert len(pmap_instance._buckets) > 0

def test_constructor_creates_pmap_with_single_key_value_pair():
    from pyrsistent import pmap
    pmap_instance = pmap({'key': 'value'})
    assert pmap_instance._size == 1
    assert pmap_instance['key'] == 'value'

def test_constructor_creates_pmap_with_multiple_key_value_pairs():
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

def test_constructor_creates_pmap_with_zero_size_and_empty_buckets():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert pmap_instance._size == 0
    assert len(pmap_instance._buckets) > 0

def test_constructor_creates_pmap_that_is_hashable():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    hash_value = hash(pmap_instance)
    assert isinstance(hash_value, int)

def test_constructor_creates_pmap_that_is_equal_to_itself():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    assert pmap_instance == pmap_instance

def test_constructor_creates_pmap_that_is_not_equal_to_different_pmap():
    from pyrsistent import pmap
    pmap_instance1 = pmap({'a': 1})
    pmap_instance2 = pmap({'b': 2})
    assert pmap_instance1 != pmap_instance2

def test_constructor_creates_pmap_with_correct_length():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2, 'c': 3})
    assert len(pmap_instance) == 3

def test_constructor_creates_pmap_that_is_iterable():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 2})
    keys = list(pmap_instance)
    assert set(keys) == {'a', 'b'}

def test_constructor_creates_pmap_that_supports_in_operator():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    assert 'a' in pmap_instance
    assert 'b' not in pmap_instance

def test_constructor_creates_pmap_that_supports_getitem():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    assert pmap_instance['a'] == 1

def test_constructor_creates_pmap_that_raises_keyerror_for_missing_key():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    try:
        pmap_instance['b']
        assert False
    except KeyError:
        assert True

def test_constructor_creates_pmap_that_supports_dot_notation():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    assert pmap_instance.a == 1

def test_constructor_creates_pmap_that_raises_attributeerror_for_missing_attribute():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    try:
        pmap_instance.b
        assert False
    except AttributeError:
        assert True

def test_constructor_creates_pmap_with_string_keys_and_integer_values():
    from pyrsistent import pmap
    pmap_instance = pmap({'one': 1, 'two': 2})
    assert pmap_instance._size == 2
    assert pmap_instance['one'] == 1
    assert pmap_instance['two'] == 2

def test_constructor_creates_pmap_with_integer_keys_and_string_values():
    from pyrsistent import pmap
    pmap_instance = pmap({1: 'one', 2: 'two'})
    assert pmap_instance._size == 2
    assert pmap_instance[1] == 'one'
    assert pmap_instance[2] == 'two'

def test_constructor_creates_pmap_with_tuple_keys():
    from pyrsistent import pmap
    pmap_instance = pmap({(1, 2): 'tuple_key'})
    assert pmap_instance._size == 1
    assert pmap_instance[(1, 2)] == 'tuple_key'

def test_constructor_creates_pmap_with_mixed_key_types():
    from pyrsistent import pmap
    pmap_instance = pmap({'str': 1, 123: 'int', (1, 2): 'tuple'})
    assert pmap_instance._size == 3
    assert pmap_instance['str'] == 1
    assert pmap_instance[123] == 'int'
    assert pmap_instance[(1, 2)] == 'tuple'


# LLM-generated content at query #24
#--------------------------

def test_contains_with_invalid_arg():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    items_view = m.items()
    result = ('a', 2) in items_view
    assert result == False


# LLM-generated content at query #25
#--------------------------

def test_contains_with_valid_key_value_pair():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1) in items_view
    assert result == True

def test_contains_with_key_in_map_but_wrong_value():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 2) in items_view
    assert result == False

def test_contains_with_key_not_in_map():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('c', 1) in items_view
    assert result == False

def test_contains_with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = 'a' in items_view
    assert result == False

def test_contains_with_tuple_of_wrong_length():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1, 'extra') in items_view
    assert result == False

def test_contains_with_empty_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = () in items_view
    assert result == False

def test_contains_with_none_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = None in items_view
    assert result == False


# LLM-generated content at query #26
#--------------------------

def test_eq_with_dict_and_different_iteritems():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    other = {'a': 1, 'b': 2}
    result = m1 == other
    assert result is True


# LLM-generated content at query #27
#--------------------------

def test_update_with_single_map_merge():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2))
    assert m2 == {'a': 3, 'b': 2}

def test_update_with_multiple_maps_merge():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2), {'a': 3, 'c': 4})
    assert m2 == {'a': 6, 'b': 2, 'c': 4}

def test_update_with_keep_leftmost():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_new_key():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(c=3))
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_empty_map():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, {})
    assert m2 == m1

def test_update_with_no_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r)
    assert m2 == m1

def test_update_with_identity_merge():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r, m(a=2, c=3))
    assert m2 == {'a': 2, 'b': 2, 'c': 3}

def test_update_with_complex_merge():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: max(l, r), m(a=5, b=1), {'a': 3, 'c': 4})
    assert m2 == {'a': 5, 'b': 2, 'c': 4}

def test_update_with_preserves_original():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2))
    assert m1 == {'a': 1, 'b': 2}
    assert m2 == {'a': 3, 'b': 2}

def test_update_with_merge_on_nonexistent_key():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(c=3))
    assert m2 == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #28
#--------------------------

def test_contains_with_valid_key_value_pair():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1) in items_view
    assert result == True

def test_contains_with_key_in_map_but_different_value():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 2) in items_view
    assert result == False

def test_contains_with_key_not_in_map():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('c', 1) in items_view
    assert result == False

def test_contains_with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = 'a' in items_view
    assert result == False

def test_contains_with_tuple_of_wrong_length():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1, 3) in items_view
    assert result == False

def test_contains_with_empty_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = () in items_view
    assert result == False

def test_contains_with_none_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = None in items_view
    assert result == False


# LLM-generated content at query #29
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap_instance = pmap(buckets)
    assert len(pmap_instance) == size
    assert pmap_instance['key1'] == 'value1'
    assert pmap_instance['key2'] == 'value2'

def test_constructor_with_empty_buckets_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert len(pmap_instance) == 0
    assert dict(pmap_instance) == {}

def test_constructor_with_none_buckets_raises_error():
    from pyrsistent import PMap
    try:
        PMap(0, None)
        assert False
    except Exception:
        assert True

def test_constructor_preserves_hash_collision_handling():
    from pyrsistent import pmap
    class SameHash:
        def __hash__(self):
            return 1
    key1 = SameHash()
    key2 = SameHash()
    pmap_instance = pmap({key1: 'val1', key2: 'val2'})
    assert len(pmap_instance) == 2
    assert pmap_instance[key1] == 'val1'
    assert pmap_instance[key2] == 'val2'

def test_constructor_creates_immutable_instance():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    try:
        pmap_instance['a'] = 2
        assert False
    except TypeError:
        assert True

def test_constructor_with_duplicate_keys_keeps_last_value():
    from pyrsistent import pmap
    pmap_instance = pmap([('a', 1), ('a', 2)])
    assert pmap_instance['a'] == 2
    assert len(pmap_instance) == 1

def test_constructor_supports_various_key_types():
    from pyrsistent import pmap
    pmap_instance = pmap({1: 'int', 'str': 'string', (1, 2): 'tuple'})
    assert pmap_instance[1] == 'int'
    assert pmap_instance['str'] == 'string'
    assert pmap_instance[(1, 2)] == 'tuple'

def test_constructor_supports_various_value_types():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'b': 'string', 'c': [1, 2], 'd': {'nested': 'dict'}})
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 'string'
    assert pmap_instance['c'] == [1, 2]
    assert pmap_instance['d'] == {'nested': 'dict'}

def test_constructor_creates_pmap_that_is_hashable():
    from pyrsistent import pmap
    pmap1 = pmap({'a': 1})
    pmap2 = pmap({'a': 1})
    assert hash(pmap1) == hash(pmap2)
    assert pmap1 == pmap2

def test_constructor_with_large_number_of_elements():
    from pyrsistent import pmap
    items = {str(i): i for i in range(1000)}
    pmap_instance = pmap(items)
    assert len(pmap_instance) == 1000
    assert pmap_instance['500'] == 500


# LLM-generated content at query #30
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('a', 1), ('b', 2))
    m = pmap(buckets)
    assert len(m) == size
    assert m['a'] == 1
    assert m['b'] == 2

def test_constructor_handles_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    assert len(m) == 0
    assert list(m) == []

def test_constructor_creates_pmap_from_dict():
    from pyrsistent import pmap
    d = {'x': 10, 'y': 20}
    m = pmap(d)
    assert len(m) == 2
    assert m['x'] == 10
    assert m['y'] == 20

def test_constructor_creates_pmap_from_keyword_arguments():
    from pyrsistent import m
    pm = m(alpha=100, beta=200)
    assert pm['alpha'] == 100
    assert pm['beta'] == 200
    assert len(pm) == 2

def test_constructor_pmap_is_hashable():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    assert hash(m1) == hash(m2)

def test_constructor_pmap_supports_equality():
    from pyrsistent import pmap
    m1 = pmap({'k1': 'v1', 'k2': 'v2'})
    m2 = pmap({'k1': 'v1', 'k2': 'v2'})
    assert m1 == m2
    assert not (m1 != m2)

def test_constructor_pmap_is_immutable():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    try:
        m['a'] = 2
    except TypeError:
        pass
    else:
        assert False, "PMap should be immutable"

def test_constructor_pmap_supports_dot_notation():
    from pyrsistent import m
    pm = m(foo=42)
    assert pm.foo == 42

def test_constructor_pmap_raises_keyerror_on_missing_key():
    from pyrsistent import pmap
    m = pmap({})
    try:
        _ = m['missing']
    except KeyError:
        pass
    else:
        assert False, "Should raise KeyError"

def test_constructor_pmap_raises_attributeerror_on_missing_attribute():
    from pyrsistent import m
    pm = m()
    try:
        _ = pm.missing
    except AttributeError:
        pass
    else:
        assert False, "Should raise AttributeError"


# LLM-generated content at query #31
#--------------------------

def test_update_with_single_map():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2))
    assert m2 == {'a': 3, 'b': 2}

def test_update_with_multiple_maps():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2), {'b': 3, 'c': 4})
    assert m2 == {'a': 3, 'b': 5, 'c': 4}

def test_update_with_keep_leftmost():
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m2 == {'a': 1}

def test_update_with_keep_rightmost():
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert m2 == {'a': 3}

def test_update_with_empty_map():
    m1 = m()
    m2 = m1.update_with(lambda l, r: l + r, m(a=1, b=2))
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_no_maps():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r)
    assert m2 == m1

def test_update_with_new_key():
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l + r, m(b=2))
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_identity_function():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m(a=5, c=3))
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_constant_function():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: 42, m(a=5, c=3))
    assert m2 == {'a': 42, 'b': 2, 'c': 42}

def test_update_with_original_unchanged():
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2))
    assert m1 == {'a': 1, 'b': 2}


# LLM-generated content at query #32
#--------------------------

def test_turbo_mapping_predicate_false():
    result = _turbo_mapping([], 0)
    assert result is not None


# LLM-generated content at query #33
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap, m
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap_instance = pmap(buckets)
    assert len(pmap_instance) == size
    assert pmap_instance['key1'] == 'value1'
    assert pmap_instance['key2'] == 'value2'

def test_constructor_with_empty_buckets_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert len(pmap_instance) == 0
    assert dict(pmap_instance) == {}

def test_constructor_preserves_hash_collision_handling():
    from pyrsistent import pmap
    class SameHash:
        def __hash__(self):
            return 1
    key1 = SameHash()
    key2 = SameHash()
    pmap_instance = pmap({key1: 'val1', key2: 'val2'})
    assert pmap_instance[key1] == 'val1'
    assert pmap_instance[key2] == 'val2'

def test_constructor_creates_pmap_that_is_hashable():
    from pyrsistent import pmap
    pmap1 = pmap({'a': 1, 'b': 2})
    pmap2 = pmap({'a': 1, 'b': 2})
    assert hash(pmap1) == hash(pmap2)
    assert pmap1 == pmap2

def test_constructor_supports_nested_mappings():
    from pyrsistent import pmap
    nested = {'inner': {'key': 'value'}}
    pmap_instance = pmap(nested)
    assert pmap_instance['inner'] == {'key': 'value'}

def test_constructor_with_pmap_as_input_returns_equal_pmap():
    from pyrsistent import pmap
    original = pmap({'x': 10, 'y': 20})
    new = pmap(original)
    assert new == original
    assert new is not original

def test_constructor_handles_large_number_of_efficiently():
    from pyrsistent import pmap
    large_dict = {i: i*2 for i in range(1000)}
    pmap_instance = pmap(large_dict)
    assert len(pmap_instance) == 1000
    assert all(pmap_instance[i] == i*2 for i in range(1000))

def test_constructor_with_keyword_arguments():
    from pyrsistent import m
    pmap_instance = m(a=1, b=2, c=3)
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2
    assert pmap_instance['c'] == 3
    assert len(pmap_instance) == 3

def test_constructor_creates_pmap_that_is_immutable():
    from pyrsistent import pmap
    pmap_instance = pmap({'k': 'v'})
    try:
        pmap_instance['k'] = 'new'
        assert False, "Should not allow assignment"
    except TypeError:
        pass

def test_constructor_with_duplicate_keys_keeps_last_value():
    from pyrsistent import pmap
    pmap_instance = pmap([('a', 1), ('a', 2), ('b', 3)])
    assert pmap_instance['a'] == 2
    assert pmap_instance['b'] == 3
    assert len(pmap_instance) == 2


# LLM-generated content at query #34
#--------------------------

def test_contains_returns_false_on_invalid_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'a', 'extra') in items_view
    assert result == False


# LLM-generated content at query #35
#--------------------------

def test_eq_with_dict_and_different_buckets_but_same_items():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = {'a': 1, 'b': 2}
    result = m1 == m2
    assert result is True


# LLM-generated content at query #36
#--------------------------

def test___contains___with_valid_key_value_pair_present():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("a", 1) in items_view
    assert result is True

def test___contains___with_valid_key_value_pair_absent():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("a", 2) in items_view
    assert result is False

def test___contains___with_key_not_in_map():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("c", 1) in items_view
    assert result is False

def test___contains___with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = "not_a_tuple" in items_view
    assert result is False

def test___contains___with_tuple_wrong_length():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items_view = m.items()
    result = ("a", 1, "extra") in items_view
    assert result is False

def test___contains___with_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    result = ("a", 1) in items_view
    assert result is False


# LLM-generated content at query #37
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    assert m._size == 2
    assert len(m._buckets) > 0

def test_constructor_sets_correct_attributes():
    from pyrsistent import PMap
    buckets = (None, None, None)
    pm = PMap.__new__(PMap)
    pm._size = 5
    pm._buckets = buckets
    assert pm._size == 5
    assert pm._buckets is buckets

def test_constructor_returns_pmap_instance():
    from pyrsistent import PMap
    pm = PMap.__new__(PMap)
    pm._size = 0
    pm._buckets = ()
    assert isinstance(pm, PMap)

def test_constructor_allows_weakref_support():
    import weakref
    from pyrsistent import pmap
    m = pmap({'x': 10})
    ref = weakref.ref(m)
    assert ref() is m

def test_constructor_initializes_without_cached_hash():
    from pyrsistent import PMap
    pm = PMap.__new__(PMap)
    pm._size = 3
    pm._buckets = (None, None)
    assert not hasattr(pm, '_cached_hash')

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    assert m._size == 0
    assert len(m._buckets) >= 0

def test_constructor_handles_non_empty_buckets():
    from pyrsistent import PMap, pvector
    bucket_list = [('key1', 'value1')]
    buckets = pvector([None, bucket_list, None])
    pm = PMap.__new__(PMap)
    pm._size = 1
    pm._buckets = buckets
    assert pm._size == 1
    assert pm._buckets[1] == bucket_list

def test_constructor_produces_hashable_instance():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    hash(m)
    assert True

def test_constructor_supports_generic_types():
    from pyrsistent import PMap
    pm = PMap.__new__(PMap)
    pm._size = 0
    pm._buckets = ()
    assert pm.__class__.__name__ == 'PMap'

def test_constructor_maintains_slots():
    from pyrsistent import PMap
    pm = PMap.__new__(PMap)
    pm._size = 0
    pm._buckets = ()
    assert not hasattr(pm, '__dict__')
    assert hasattr(pm, '_size')
    assert hasattr(pm, '_buckets')


# LLM-generated content at query #38
#--------------------------

def test_eq_with_dict_and_different_buckets_but_same_items():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    assert m1 == d
    assert not (hasattr(m1, '_cached_hash') and hasattr(d, '_cached_hash') and m1._cached_hash != d._cached_hash)


# LLM-generated content at query #39
#--------------------------

def test___contains___with_valid_key_value_pair_present():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 1) in items
    assert result == True

def test___contains___with_valid_key_value_pair_absent():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 2) in items
    assert result == False

def test___contains___with_key_not_in_map():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('c', 1) in items
    assert result == False

def test___contains___with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = 'a' in items
    assert result == False

def test___contains___with_tuple_of_wrong_length():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = ('a', 1, 2) in items
    assert result == False

def test___contains___with_empty_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = m.items()
    result = () in items
    assert result == False

def test___contains___with_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    items = m.items()
    result = ('a', 1) in items
    assert result == False


# LLM-generated content at query #40
#--------------------------

def test_update_with_merge_function():
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    expected = {'a': 3, 'b': 2}
    assert result == expected

def test_update_with_keep_leftmost():
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    expected = {'a': 1}
    assert result == expected

def test_update_with_multiple_maps():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l + r, m(a=2, c=3), {'a': 10, 'd': 4})
    expected = {'a': 13, 'b': 2, 'c': 3, 'd': 4}
    assert result == expected

def test_update_with_empty_maps():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r)
    assert result is m1

def test_update_with_new_key():
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: r, m(b=2))
    expected = {'a': 1, 'b': 2}
    assert result == expected

def test_update_with_overwrites_existing():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r * 2, m(a=3))
    expected = {'a': 6, 'b': 2}
    assert result == expected

def test_update_with_identity_function():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(a=5, c=7))
    expected = {'a': 5, 'b': 2, 'c': 7}
    assert result == expected


# LLM-generated content at query #41
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
    result = _turbo_mapping({obj1: 1, obj2: 2}, 0)
    assert len(result) == 2
    assert result[obj1] == 1
    assert result[obj2] == 2

def test_turbo_mapping_preserves_hash_collision_buckets():
    class SameHash:
        def __init__(self, val):
            self.val = val
        def __hash__(self):
            return 1
        def __eq__(self, other):
            return isinstance(other, SameHash) and self.val == other.val
    a = SameHash('a')
    b = SameHash('b')
    result = _turbo_mapping({a: 1, b: 2}, 0)
    assert len(result) == 2
    assert result[a] == 1
    assert result[b] == 2

def test_turbo_mapping_with_large_pre_size():
    result = _turbo_mapping({'a': 1}, 100)
    assert len(result) == 1
    assert result['a'] == 1

def test_turbo_mapping_initial_length_hint_fallback():
    class BadLenMapping:
        def __init__(self):
            self.data = {'a': 1, 'b': 2}
        def __getitem__(self, key):
            return self.data[key]
        def __iter__(self):
            return iter(self.data)
        def __len__(self):
            raise Exception("no length")
        def items(self):
            return self.data.items()
    result = _turbo_mapping(BadLenMapping(), 0)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #42
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


# LLM-generated content at query #43
#--------------------------

def test_eq_same_instance():
    pm = m(a=1, b=2)
    result = pm == pm
    assert result is True


def test_eq_equal_pmaps():
    pm1 = m(a=1, b=2)
    pm2 = m(a=1, b=2)
    result = pm1 == pm2
    assert result is True


def test_eq_different_pmaps():
    pm1 = m(a=1, b=2)
    pm2 = m(a=1, b=3)
    result = pm1 == pm2
    assert result is False


def test_eq_different_sizes():
    pm1 = m(a=1, b=2)
    pm2 = m(a=1)
    result = pm1 == pm2
    assert result is False


def test_eq_with_dict():
    pm = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = pm == d
    assert result is True


def test_eq_with_dict_different():
    pm = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = pm == d
    assert result is False


def test_eq_with_mapping_protocol():
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


def test_eq_not_implemented_for_non_mapping():
    pm = m(a=1, b=2)
    result = pm == [1, 2]
    assert result is NotImplemented


def test_eq_cached_hash_mismatch():
    pm1 = m(a=1, b=2)
    pm2 = m(a=1, b=3)
    hash(pm1)
    hash(pm2)
    result = pm1 == pm2
    assert result is False


def test_eq_same_buckets():
    from pyrsistent import pmap
    d = {'a': 1, 'b': 2}
    pm1 = pmap(d)
    pm2 = pmap(d)
    result = pm1 == pm2
    assert result is True


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_creates_pmap_with_correct_size_and_buckets():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    assert m._size == 2
    assert len(m._buckets) > 0

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    assert m._size == 0
    assert len(m._buckets) > 0

def test_constructor_creates_pmap_from_dict():
    from pyrsistent import pmap
    d = {'x': 10, 'y': 20}
    m = pmap(d)
    assert m['x'] == 10
    assert m['y'] == 20
    assert len(m) == 2

def test_constructor_creates_pmap_from_keyword_args():
    from pyrsistent import m
    pm = m(a=1, b=2)
    assert pm['a'] == 1
    assert pm['b'] == 2
    assert len(pm) == 2

def test_constructor_creates_pmap_from_iterable_of_pairs():
    from pyrsistent import pmap
    items = [('k1', 'v1'), ('k2', 'v2')]
    m = pmap(items)
    assert m['k1'] == 'v1'
    assert m['k2'] == 'v2'
    assert len(m) == 2

def test_constructor_creates_pmap_that_is_hashable():
    from pyrsistent import pmap
    m1 = pmap({'a': 1})
    m2 = pmap({'a': 1})
    assert hash(m1) == hash(m2)

def test_constructor_creates_pmap_that_supports_dot_notation():
    from pyrsistent import m
    pm = m(key='value')
    assert pm.key == 'value'

def test_constructor_creates_pmap_that_is_immutable():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    try:
        m['a'] = 2
        assert False
    except TypeError:
        pass

def test_constructor_creates_pmap_with_correct_iteritems():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items = list(m.iteritems())
    assert ('a', 1) in items
    assert ('b', 2) in items
    assert len(items) == 2

def test_constructor_creates_pmap_that_compares_equal_to_dict():
    from pyrsistent import pmap
    d = {'a': 1, 'b': 2}
    m = pmap(d)
    assert m == d

def test_constructor_creates_pmap_with_cached_hash_after_hashing():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    h = hash(m)
    assert hasattr(m, '_cached_hash')
    assert m._cached_hash == h

def test_constructor_creates_pmap_that_is_not_orderable():
    from pyrsistent import pmap
    m1 = pmap({'a': 1})
    m2 = pmap({'b': 2})
    try:
        m1 < m2
        assert False
    except TypeError:
        pass

def test_constructor_creates_pmap_with_correct_string_representation():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    assert repr(m) == "pmap({'a': 1})"
    assert str(m) == "pmap({'a': 1})"

def test_constructor_creates_pmap_that_is_reversible_error():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    try:
        reversed(m)
        assert False
    except TypeError:
        pass

def test_constructor_creates_pmap_with_get_method():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    assert m.get('a') == 1
    assert m.get('b') is None
    assert m.get('b', 'default') == 'default'

def test_constructor_creates_pmap_with_keys_and_values_and_items():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    assert set(m.keys()) == {'a', 'b'}
    assert set(m.values()) == {1, 2}
    assert set(m.items()) == {('a', 1), ('b', 2)}


# LLM-generated content at query #2
#--------------------------

def test___contains___with_valid_key_value_pair_present():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items = m.items()
    result = ("a", 1) in items
    assert result == True

def test___contains___with_valid_key_value_pair_absent():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items = m.items()
    result = ("a", 2) in items
    assert result == False

def test___contains___with_key_not_in_map():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items = m.items()
    result = ("c", 1) in items
    assert result == False

def test___contains___with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items = m.items()
    result = "not_a_tuple" in items
    assert result == False

def test___contains___with_wrong_length_tuple():
    from pyrsistent import pmap
    m = pmap({"a": 1, "b": 2})
    items = m.items()
    result = ("a", 1, "extra") in items
    assert result == False

def test___contains___with_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    items = m.items()
    result = ("a", 1) in items
    assert result == False


# LLM-generated content at query #3
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
    result = 'not_a_tuple' in items_view
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


# LLM-generated content at query #4
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

def test_update_with_inserts_new_key_from_single_map():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, {'b': 2})
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_inserts_new_keys_from_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, {'b': 2}, {'c': 3})
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_overwrites_with_rightmost_when_update_fn_returns_right():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, {'a': 2}, {'a': 3})
    assert m2 == {'a': 3}

def test_update_with_on_empty_map():
    from pyrsistent import m
    m1 = m()
    m2 = m1.update_with(lambda l, r: l + r, {'a': 1, 'b': 2})
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_returns_same_instance_if_no_changes():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, {})
    assert m2 is m1

def test_update_with_using_custom_update_fn():
    from pyrsistent import m
    def concat(l, r):
        return str(l) + str(r)
    m1 = m(a='x', b='y')
    m2 = m1.update_with(concat, {'a': 'z'})
    assert m2 == {'a': 'xz', 'b': 'y'}


# LLM-generated content at query #5
#--------------------------

def test_update_with_merges_values_using_update_fn():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1, b=2)
    m2 = m1.update_with(add, m(a=2))
    assert m2 == {'a': 3, 'b': 2}

def test_update_with_keeps_leftmost_value_when_update_fn_returns_left():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m2 == {'a': 1}

def test_update_with_inserts_new_keys_from_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(b=2), {'c': 3})
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_overwrites_existing_keys_using_update_fn():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=10, b=20), {'a': 100})
    assert m2 == {'a': 111, 'b': 22}

def test_update_with_empty_maps_returns_original():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r)
    assert m2 is m1

def test_update_with_single_map_merges_values():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r * 2, m(a=3, c=4))
    assert m2 == {'a': 6, 'b': 2, 'c': 4}

def test_update_with_non_existing_key_uses_new_value():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l + r, m(b=2))
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_returns_new_pmap_when_changes_made():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(a=2))
    assert m2 is not m1
    assert m2 == {'a': 2}

def test_update_with_returns_same_pmap_when_no_changes():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(a=2))
    assert m2 is m1
    assert m2 == {'a': 1}

def test_update_with_complex_update_fn():
    from pyrsistent import m
    m1 = m(a=5, b=10)
    m2 = m1.update_with(lambda l, r: max(l, r), m(a=3, b=15, c=20))
    assert m2 == {'a': 5, 'b': 15, 'c': 20}


# LLM-generated content at query #6
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
    m2 = m(a=1)
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

def test_eq_other_mapping():
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
    m1 = m(a=1, b=2)
    cm = CustomMapping({'a': 1, 'b': 2})
    result = m1 == cm
    assert result is True

def test_eq_other_mapping_different():
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
    m1 = m(a=1, b=2)
    cm = CustomMapping({'a': 1, 'b': 3})
    result = m1 == cm
    assert result is False


# LLM-generated content at query #7
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

def test___contains___with_wrong_length_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a', 1, 3) in items_view
    assert result == False

def test___contains___with_empty_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = () in items_view
    assert result == False

def test___contains___with_single_element_tuple():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    items_view = m.items()
    result = ('a',) in items_view
    assert result == False


# LLM-generated content at query #8
#--------------------------

def test_constructor_creates_pmap_with_correct_size_and_buckets():
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
    assert list(pmap_instance) == []

def test_constructor_with_dict_creates_correct_pmap():
    from pyrsistent import pmap
    d = {'a': 1, 'b': 2}
    pmap_instance = pmap(d)
    assert len(pmap_instance) == 2
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2

def test_constructor_with_keyword_arguments():
    from pyrsistent import pmap
    pmap_instance = pmap(x=10, y=20)
    assert len(pmap_instance) == 2
    assert pmap_instance['x'] == 10
    assert pmap_instance['y'] == 20

def test_constructor_with_mixed_arguments():
    from pyrsistent import pmap
    d = {'a': 1}
    pmap_instance = pmap(d, b=2)
    assert len(pmap_instance) == 2
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2

def test_constructor_handles_nested_pmaps():
    from pyrsistent import pmap
    inner = pmap({'x': 100})
    outer = pmap({'inner': inner})
    assert len(outer) == 1
    assert outer['inner']['x'] == 100

def test_constructor_preserves_hashability():
    from pyrsistent import pmap
    pmap1 = pmap({'a': 1})
    pmap2 = pmap({'a': 1})
    assert hash(pmap1) == hash(pmap2)

def test_constructor_with_duplicate_keys_keeps_last():
    from pyrsistent import pmap
    pmap_instance = pmap([('a', 1), ('a', 2)])
    assert len(pmap_instance) == 1
    assert pmap_instance['a'] == 2

def test_constructor_with_iterable_of_pairs():
    from pyrsistent import pmap
    pairs = [('k1', 'v1'), ('k2', 'v2')]
    pmap_instance = pmap(pairs)
    assert len(pmap_instance) == 2
    assert pmap_instance['k1'] == 'v1'
    assert pmap_instance['k2'] == 'v2'

def test_constructor_creates_distinct_instances():
    from pyrsistent import pmap
    pmap1 = pmap({'a': 1})
    pmap2 = pmap({'a': 1})
    assert pmap1 is not pmap2
    assert pmap1 == pmap2


# LLM-generated content at query #9
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

def test_turbo_mapping_initial_length_exception_falls_back():
    class BadLenMapping:
        def __len__(self):
            raise Exception("Cannot get length")
        def items(self):
            return [('a', 1)].__iter__()
    initial = BadLenMapping()
    result = _turbo_mapping(initial, 0)
    assert len(result) == 1
    assert result['a'] == 1

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
    assert result['a'] == 1

def test_turbo_mapping_with_zero_pre_size_and_non_empty_initial():
    initial = {'a': 1, 'b': 2, 'c': 3}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 3
    assert dict(result) == initial


# LLM-generated content at query #10
#--------------------------

def test_eq_with_dict_and_different_buckets():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = {'a': 1, 'b': 2}
    result = m1 == m2
    assert result == True
    m3 = pmap({'a': 1, 'b': 2})
    m4 = {'a': 1, 'b': 3}
    result = m3 == m4
    assert result == False


# LLM-generated content at query #11
#--------------------------

def test_update_with_merge_function():
    from operator import add
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    expected = {'a': 3, 'b': 2}
    assert result == expected

def test_update_with_keep_leftmost():
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a':3})
    expected = {'a': 1}
    assert result == expected

def test_update_with_multiple_maps():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l + r, m(a=2, c=3), {'a': 10, 'd': 4})
    expected = {'a': 13, 'b': 2, 'c': 3, 'd': 4}
    assert result == expected

def test_update_with_empty_maps():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r)
    assert result is m1

def test_update_with_new_key():
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: r, m(b=2))
    expected = {'a': 1, 'b': 2}
    assert result == expected

def test_update_with_overwrites_existing():
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r * 2, m(a=3))
    expected = {'a': 6, 'b': 2}
    assert result == expected


# LLM-generated content at query #12
#--------------------------

def test_eq_with_dict_and_different_iteritems():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    other = {'a': 1, 'b': 2}
    result = m1 == other
    assert result is True
    m2 = pmap({'a': 1, 'b': 2, 'c': 3})
    other2 = {'a': 1, 'b': 2}
    result2 = m2 == other2
    assert result2 is False
    m3 = pmap({'a': 1, 'b': 2})
    other3 = {'a': 1, 'b': 3}
    result3 = m3 == other3
    assert result3 is False


# LLM-generated content at query #13
#--------------------------

def test_turbo_mapping_predicate_false():
    initial = {}
    pre_size = 0
    result = _turbo_mapping(initial, pre_size)
    assert result is not None


# LLM-generated content at query #14
#--------------------------

def test_update_with_single_map():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2))
    assert m2 == {'a': 3, 'b': 2}

def test_update_with_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2), {'a': 3, 'c': 4})
    assert m2 == {'a': 6, 'b': 2, 'c': 4}

def test_update_with_keep_leftmost():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_keep_rightmost():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert m2 == {'a': 3, 'b': 2}

def test_update_with_new_key():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(c=3))
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_empty_map():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, {})
    assert m2 == m1

def test_update_with_no_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r)
    assert m2 == m1

def test_update_with_identity():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m(a=2, c=3))
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_constant():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: 42, m(a=2, c=3))
    assert m2 == {'a': 42, 'b': 2, 'c': 42}

def test_update_with_original_unchanged():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2))
    assert m1 == {'a': 1, 'b': 2}


# LLM-generated content at query #15
#--------------------------

def test_predicate_at_line_7_false():
    result = _turbo_mapping([], None)
    assert result is not None


# LLM-generated content at query #16
#--------------------------

def test_turbo_mapping_predicate_false():
    result = _turbo_mapping([], 0)
    assert result is not None


# LLM-generated content at query #17
#--------------------------

def test_contains_with_non_iterable_arg():
    from pyrsistent import pmap
    m = pmap_items(pmap({'a': 1}))
    result = (1,) in m
    assert result == False

def test_contains_with_wrong_length_iterable():
    from pyrsistent import pmap
    m = pmap_items(pmap({'a': 1}))
    result = (1, 2, 3) in m
    assert result == False

def test_contains_with_non_tuple_arg():
    from pyrsistent import pmap
    m = pmap_items(pmap({'a': 1}))
    result = 42 in m
    assert result == False

def test_contains_with_string_arg():
    from pyrsistent import pmap
    m = pmap_items(pmap({'a': 1}))
    result = 'ab' in m
    assert result == False


# LLM-generated content at query #18
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

def test_eq_other_mapping():
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
    m1 = m(a=1, b=2)
    cm = CustomMapping({'a': 1, 'b': 2})
    result = m1 == cm
    assert result is True


# LLM-generated content at query #19
#--------------------------

def test_turbo_mapping_with_empty_initial():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0
    assert dict(result) == {}

def test_turbo_mapping_with_dict_initial():
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_turbo_mapping_with_pre_size():
    initial = {'x': 10, 'y': 20}
    result = _turbo_mapping(initial, 16)
    assert len(result) == 2
    assert result['x'] == 10
    assert result['y'] == 20

def test_turbo_mapping_with_non_mapping_initial():
    initial = [('p', 100), ('q', 200)]
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert result['p'] == 100
    assert result['q'] == 200

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
    result = _turbo_mapping(initial, 4)
    assert len(result) == 2
    assert result[key1] == 'val1'
    assert result[key2] == 'val2'

def test_turbo_mapping_handles_exception_in_len():
    class BadLenMapping:
        def __len__(self):
            raise ValueError("no length")
        def items(self):
            return [('a', 1)].__iter__()
    initial = BadLenMapping()
    result = _turbo_mapping(initial, 0)
    assert len(result) == 1
    assert result['a'] == 1


# LLM-generated content at query #20
#--------------------------

def test_eq_with_dict_and_different_buckets():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = {'a': 1, 'b': 2}
    result = m1 == m2
    assert result is True


# LLM-generated content at query #21
#--------------------------

def test_update_with_merges_values_using_update_fn():
    from pyrsistent import m
    from operator import add
    m1 = m(a=1, b=2)
    m2 = m1.update_with(add, m(a=2))
    assert m2 == {'a': 3, 'b': 2}

def test_update_with_keeps_leftmost_value_when_update_fn_returns_left():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m2 == {'a': 1}

def test_update_with_inserts_new_keys_from_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(b=2), {'c': 3})
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_applies_update_fn_for_overlapping_keys():
    from pyrsistent import m
    m1 = m(a=1, b=10)
    m2 = m1.update_with(lambda x, y: x * y, m(a=2, b=3), {'b': 4})
    assert m2 == {'a': 2, 'b': 120}

def test_update_with_returns_same_instance_if_no_changes():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m())
    assert m2 is m1

def test_update_with_handles_empty_maps():
    from pyrsistent import m
    m1 = m()
    m2 = m1.update_with(lambda l, r: r, {}, m(a=1))
    assert m2 == {'a': 1}

def test_update_with_uses_update_fn_for_each_key_collision():
    from pyrsistent import m
    def concat(left, right):
        return left + ',' + right
    m1 = m(x='a', y='b')
    m2 = m1.update_with(concat, m(x='c', y='d'), {'x': 'e'})
    assert m2 == {'x': 'a,c,e', 'y': 'b,d'}

def test_update_with_preserves_non_updated_keys():
    from pyrsistent import m
    m1 = m(a=1, b=2, c=3)
    m2 = m1.update_with(lambda l, r: r, m(b=20))
    assert m2 == {'a': 1, 'b': 20, 'c': 3}

def test_update_with_works_with_different_map_types():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l + r, {'a': 10}, dict(b=100))
    assert m2 == {'a': 11, 'b': 100}

def test_update_with_handles_none_values():
    from pyrsistent import m
    m1 = m(a=None)
    m2 = m1.update_with(lambda l, r: r if r is not None else l, m(a=5))
    assert m2 == {'a': 5}


# LLM-generated content at query #22
#--------------------------

def test_contains_with_non_tuple_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a'})
    items = m.items()
    result = (1, 'a', 'extra') in items
    assert result == False


# LLM-generated content at query #23
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
    k1 = FixedHash('key1', 5)
    k2 = FixedHash('key2', 5)
    initial = {k1: 'val1', k2: 'val2'}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert result[k1] == 'val1'
    assert result[k2] == 'val2'

def test_turbo_mapping_with_initial_len_exception():
    class BadLen:
        def __iter__(self):
            yield ('a', 1)
        def __len__(self):
            raise Exception("no length")
    initial = BadLen()
    result = _turbo_mapping(initial, 0)
    assert len(result) == 1
    assert result['a'] == 1

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
    assert result['a'] == 1

def test_turbo_mapping_with_zero_pre_size_and_empty_initial():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0


# LLM-generated content at query #24
#--------------------------

def test_eq_with_dict_other():
    from pyrsistent import m
    pmap_instance = m(a=1, b=2)
    dict_other = {'a': 1, 'b': 2}
    result = pmap_instance == dict_other
    assert result is True


# LLM-generated content at query #25
#--------------------------

def test_contains_with_invalid_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items = m.items()
    result = (1, 'a', 'extra') in items
    assert result == False


# LLM-generated content at query #26
#--------------------------

def test_update_with_does_not_call_update_fn_when_key_not_in_evolver():
    from pyrsistent import m
    call_count = 0
    def update_fn(l, r):
        nonlocal call_count
        call_count += 1
        return l + r
    m1 = m(a=1, b=2)
    m2 = m(c=3)
    result = m1.update_with(update_fn, m2)
    assert call_count == 0
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #27
#--------------------------

def test_eq_with_dict_and_different_buckets_but_same_items():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == dict(m2)
    assert result is True


# LLM-generated content at query #28
#--------------------------

def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l + r, m(b=2))
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #29
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
    key1 = FixedHash('key1', 5)
    key2 = FixedHash('key2', 5)
    result = _turbo_mapping({key1: 100, key2: 200}, 0)
    assert len(result) == 2
    assert result[key1] == 100
    assert result[key2] == 200

def test_turbo_mapping_with_non_mapping_initial():
    result = _turbo_mapping([('a', 1), ('b', 2)], 0)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test_turbo_mapping_preserves_hash():
    mapping = {'foo': 'bar'}
    result = _turbo_mapping(mapping, 0)
    assert hash(result) == hash(frozenset(mapping.items()))

def test_turbo_mapping_large_initial():
    large_dict = {i: i*2 for i in range(100)}
    result = _turbo_mapping(large_dict, 0)
    assert len(result) == 100
    for i in range(100):
        assert result[i] == i*2

def test_turbo_mapping_with_zero_pre_size():
    result = _turbo_mapping({'a': 1}, 0)
    assert len(result) == 1
    assert result['a'] == 1

def test_turbo_mapping_with_small_pre_size():
    result = _turbo_mapping({'a': 1, 'b': 2, 'c': 3}, 4)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3


# LLM-generated content at query #30
#--------------------------

def test_contains_with_non_tuple_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'a', 'extra') in items_view
    assert result == False

def test_contains_with_single_value_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = 1 in items_view
    assert result == False

def test_contains_with_string_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = 'key' in items_view
    assert result == False

def test_contains_with_none_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = None in items_view
    assert result == False

def test_contains_with_list_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = [1, 'a'] in items_view
    assert result == False


# LLM-generated content at query #31
#--------------------------

def test_turbo_mapping_predicate_false():
    initial = {}
    pre_size = 0
    result = _turbo_mapping(initial, pre_size)
    assert result is not None


# LLM-generated content at query #32
#--------------------------

def test_contains_with_invalid_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items = m.items()
    result = (1, 'a', 'extra') in items
    assert result == False


# LLM-generated content at query #33
#--------------------------

def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    pm = m(a=1)
    result = pm.update_with(lambda l, r: l + r, m(b=2))
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #34
#--------------------------

def test_eq_with_dict_and_different_buckets_but_same_items():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = {'a': 1, 'b': 2}
    result = m1 == m2
    assert result is True


