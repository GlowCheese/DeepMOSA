####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_update_with_merges_values_using_update_fn():
    from operator import add

    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(add, m(a=2))
    expected = {'a': 3, 'b': 2}
    assert result == expected

def test_update_with_keeps_leftmost_value_when_update_fn_returns_left():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    expected = {'a': 1}
    assert result == expected

def test_update_with_inserts_new_keys_from_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: r, m(b=2), {'c': 3})
    expected = {'a': 1, 'b': 2, 'c': 3}
    assert result == expected

def test_update_with_handles_empty_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, {}, m(c=3))
    expected = {'a': 1, 'b': 2, 'c': 3}
    assert result == expected

def test_update_with_returns_same_instance_if_no_changes():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l, {})
    assert result is m1

def test_update_with_uses_update_fn_for_collisions():
    from pyrsistent import m
    def concat(left, right):
        return left + ',' + right
    m1 = m(key='hello')
    result = m1.update_with(concat, m(key='world'), {'key': 'test'})
    expected = {'key': 'hello,world,test'}
    assert result == expected

def test_update_with_works_with_non_pmap_mappings():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l + r, {'a': 2, 'b': 3})
    expected = {'a': 3, 'b': 3}
    assert result == expected

def test_update_with_preserves_original_when_update_fn_returns_existing_value():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l, m(a=1))
    assert result is m1


# LLM-generated content at query #2
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

def test_constructor_handles_colliding_keys():
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

def test_constructor_preserves_insertion_order_in_buckets():
    from pyrsistent import pmap
    m = pmap({'x': 10, 'y': 20, 'z': 30})
    items = list(m.iteritems())
    assert ('x', 10) in items
    assert ('y', 20) in items
    assert ('z', 30) in items

def test_constructor_with_none_values():
    from pyrsistent import pmap
    m = pmap({'a': None, 'b': None})
    assert m._size == 2
    assert m['a'] is None
    assert m['b'] is None

def test_constructor_with_false_values():
    from pyrsistent import pmap
    m = pmap({'a': False, 'b': 0})
    assert m._size == 2
    assert m['a'] is False
    assert m['b'] == 0

def test_constructor_with_empty_buckets():
    from pyrsistent import pmap
    m = pmap({})
    assert m._size == 0
    assert len(m._buckets) == 0

def test_constructor_creates_independent_instances():
    from pyrsistent import pmap
    m1 = pmap({'a': 1})
    m2 = pmap({'a': 1})
    assert m1 is not m2
    assert m1 == m2


# LLM-generated content at query #3
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
    from collections import OrderedDict
    m1 = m(a=1, b=2)
    od = OrderedDict([('a', 1), ('b', 2)])
    result = m1 == od
    assert result is True

def test_eq_different_lengths():
    m1 = m(a=1, b=2)
    m2 = m(a=1)
    result = m1 == m2
    assert result is False

def test_eq_not_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [('a', 1), ('b', 2)]
    assert result is False

def test_eq_same_buckets():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1._buckets == m2._buckets
    assert result is True

def test_eq_cached_hash_mismatch():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = hash(frozenset(m1.iteritems()))
    m2._cached_hash = hash(frozenset(m2.iteritems())) + 1
    result = m1 == m2
    assert result is False


# LLM-generated content at query #4
#--------------------------

def test_pmap_constructor_creates_instance_with_size_and_buckets():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    assert m._size == 2
    assert len(m._buckets) > 0

def test_pmap_constructor_returns_pmap_instance():
    from pyrsistent import PMap, pvector
    buckets = pvector([None, None, None, None])
    pm = PMap(0, buckets)
    assert isinstance(pm, PMap)

def test_pmap_constructor_sets_size_and_buckets():
    from pyrsistent import PMap, pvector
    buckets = pvector([None, None])
    pm = PMap(5, buckets)
    assert pm._size == 5
    assert pm._buckets is buckets

def test_pmap_constructor_creates_empty_map():
    from pyrsistent import pmap
    m = pmap({})
    assert m._size == 0
    assert len(m._buckets) > 0

def test_pmap_constructor_handles_non_empty_buckets():
    from pyrsistent import PMap, pvector
    bucket = [('key1', 'value1'), ('key2', 'value2')]
    buckets = pvector([bucket, None, None])
    pm = PMap(2, buckets)
    assert pm._size == 2
    assert pm._buckets[0] == bucket


# LLM-generated content at query #5
#--------------------------

def test_eq_with_different_cached_hash_returns_false():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m1._cached_hash = 123
    m2._cached_hash = 456
    result = m1 == m2
    assert result == False


# LLM-generated content at query #6
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
    assert list(pmap_instance) == []

def test_constructor_handles_nested_pmaps():
    from pyrsistent import pmap
    inner = pmap({'a': 1})
    outer = pmap({'inner': inner})
    assert outer['inner']['a'] == 1
    assert isinstance(outer['inner'], type(inner))

def test_constructor_preserves_hash_collisions_handling():
    from pyrsistent import pmap
    class SameHash:
        def __hash__(self):
            return 1
    key1 = SameHash()
    key2 = SameHash()
    pmap_instance = pmap({key1: 'first', key2: 'second'})
    assert pmap_instance[key1] == 'first'
    assert pmap_instance[key2] == 'second'

def test_constructor_with_dict_argument():
    from pyrsistent import pmap
    d = {'x': 10, 'y': 20}
    pmap_instance = pmap(d)
    assert pmap_instance['x'] == 10
    assert pmap_instance['y'] == 20
    assert len(pmap_instance) == 2

def test_constructor_with_keyword_arguments():
    from pyrsistent import pmap
    pmap_instance = pmap(a=100, b=200)
    assert pmap_instance['a'] == 100
    assert pmap_instance['b'] == 200
    assert len(pmap_instance) == 2

def test_constructor_with_mixed_dict_and_kwargs():
    from pyrsistent import pmap
    pmap_instance = pmap({'c': 300}, d=400)
    assert pmap_instance['c'] == 300
    assert pmap_instance['d'] == 400
    assert len(pmap_instance) == 2

def test_constructor_creates_immutable_copy():
    from pyrsistent import pmap
    mutable_dict = {'change': 'original'}
    pmap_instance = pmap(mutable_dict)
    mutable_dict['change'] = 'modified'
    assert pmap_instance['change'] == 'original'

def test_constructor_with_empty_buckets():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert len(pmap_instance) == 0
    assert pmap_instance == {}

def test_constructor_supports_various_key_types():
    from pyrsistent import pmap
    pmap_instance = pmap({1: 'int', 'str': 'string', (1,2): 'tuple'})
    assert pmap_instance[1] == 'int'
    assert pmap_instance['str'] == 'string'
    assert pmap_instance[(1,2)] == 'tuple'


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

def test_constructor_creates_pmap_with_keyword_arguments():
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

def test_constructor_pmap_is_hashable():
    from pyrsistent import pmap
    pmap1 = pmap({'a': 1})
    pmap2 = pmap({'a': 1})
    assert hash(pmap1) == hash(pmap2)
    assert pmap1 == pmap2

def test_constructor_pmap_supports_dot_notation():
    from pyrsistent import m
    pmap_instance = m(alpha=100)
    assert pmap_instance.alpha == 100

def test_constructor_pmap_raises_key_error_for_missing_key():
    from pyrsistent import pmap
    pmap_instance = pmap()
    try:
        _ = pmap_instance['missing']
        assert False
    except KeyError:
        assert True

def test_constructor_pmap_raises_attribute_error_for_missing_attribute():
    from pyrsistent import pmap
    pmap_instance = pmap()
    try:
        _ = pmap_instance.missing
        assert False
    except AttributeError:
        assert True

def test_constructor_pmap_is_immutable():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1})
    try:
        pmap_instance['a'] = 2
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #8
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
    result = ('a', 1, 'extra') in items_view
    assert result == False

def test___contains___with_empty_mapping():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    result = ('a', 1) in items_view
    assert result == False


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

def test_turbo_mapping_with_initial_length_exception():
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
    assert dict(result) == {'a': 1}

def test_turbo_mapping_with_zero_pre_size_and_empty_initial():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0
    assert dict(result) == {}


# LLM-generated content at query #10
#--------------------------

def test_turbo_mapping_predicate_false():
    initial = {1: 'a', 2: 'b'}
    pre_size = 4
    result = _turbo_mapping(initial, pre_size)
    assert result is not None


# LLM-generated content at query #11
#--------------------------

def test_eq_with_dict_and_different_buckets():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = {'a': 1, 'b': 2}
    result = m1 == m2
    assert result is True
    m3 = pmap({'a': 1, 'b': 2, 'c': 3})
    m4 = {'a': 1, 'b': 2, 'c': 3}
    result2 = m3 == m4
    assert result2 is True
    m5 = pmap({'x': 10, 'y': 20})
    m6 = {'x': 10, 'y': 30}
    result3 = m5 == m6
    assert result3 is False


# LLM-generated content at query #12
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent._pmap import PMap
    size = 2
    buckets = (('a', 1), ('b', 2))
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets

def test_constructor_returns_pmap_instance():
    from pyrsistent._pmap import PMap
    size = 0
    buckets = ()
    pmap_instance = PMap(size, buckets)
    assert isinstance(pmap_instance, PMap)

def test_constructor_with_zero_size_and_empty_buckets():
    from pyrsistent._pmap import PMap
    size = 0
    buckets = ()
    pmap_instance = PMap(size, buckets)
    assert len(pmap_instance) == 0

def test_constructor_with_non_zero_size_and_buckets():
    from pyrsistent._pmap import PMap
    size = 3
    buckets = (('x', 10), ('y', 20), ('z', 30))
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets

def test_constructor_sets_correct_attributes():
    from pyrsistent._pmap import PMap
    size = 5
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets
    assert not hasattr(pmap_instance, '_cached_hash')


# LLM-generated content at query #13
#--------------------------

def test_contains_with_invalid_arg_returns_false():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'a', 'extra') in items_view
    assert result == False


# LLM-generated content at query #14
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

def test_eq_other_mapping():
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

def test_eq_not_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [1, 2, 3]
    assert result is NotImplemented

def test_eq_cached_hash_mismatch():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    hash(m1)
    hash(m2)
    result = m1 == m2
    assert result is False

def test_eq_same_buckets():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1._buckets == m2._buckets
    assert result is True
    result = m1 == m2
    assert result is True


# LLM-generated content at query #15
#--------------------------

def test_eq_with_dict_equal_but_different_buckets():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    result = m1 == m2
    assert result is True


# LLM-generated content at query #16
#--------------------------

def test___contains___with_valid_key_value_pair_present():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'a') in items_view
    assert result is True

def test___contains___with_valid_key_value_pair_absent_due_to_wrong_value():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'b') in items_view
    assert result is False

def test___contains___with_valid_key_value_pair_absent_due_to_missing_key():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (3, 'a') in items_view
    assert result is False

def test___contains___with_argument_not_a_two_element_tuple():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'a', 'extra') in items_view
    assert result is False

def test___contains___with_argument_not_iterable():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = 42 in items_view
    assert result is False

def test___contains___with_argument_as_single_element_tuple():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1,) in items_view
    assert result is False

def test___contains___with_empty_mapping():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    result = (1, 'a') in items_view
    assert result is False


# LLM-generated content at query #17
#--------------------------

def test__turbo_mapping_with_dict():
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_with_pre_size():
    initial = {'x': 10, 'y': 20}
    result = _turbo_mapping(initial, 16)
    assert len(result) == 2
    assert result['x'] == 10
    assert result['y'] == 20

def test__turbo_mapping_with_empty_dict():
    initial = {}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 0

def test__turbo_mapping_with_non_mapping_iterable():
    initial = [('key1', 100), ('key2', 200)]
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert result['key1'] == 100
    assert result['key2'] == 200

def test__turbo_mapping_with_collision_keys():
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

def test__turbo_mapping_preserves_hash_based_lookup():
    initial = {'foo': 42, 'bar': 84}
    result = _turbo_mapping(initial, 0)
    assert 'foo' in result
    assert 'bar' in result
    assert 'baz' not in result

def test__turbo_mapping_with_zero_pre_size_and_empty():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0

def test__turbo_mapping_with_large_pre_size():
    initial = {'a': 1}
    result = _turbo_mapping(initial, 128)
    assert len(result) == 1
    assert result['a'] == 1

def test__turbo_mapping_identity_of_empty():
    result1 = _turbo_mapping({}, 0)
    result2 = _turbo_mapping({}, 0)
    assert result1 == result2

def test__turbo_mapping_with_non_string_keys():
    initial = {1: 'one', 2.5: 'two point five', (3, 4): 'tuple'}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 3
    assert result[1] == 'one'
    assert result[2.5] == 'two point five'
    assert result[(3, 4)] == 'tuple'


# LLM-generated content at query #18
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

def test_constructor_creates_pmap_with_single_element():
    from pyrsistent import pmap
    m = pmap({'key': 'value'})
    assert m._size == 1
    assert m['key'] == 'value'

def test_constructor_creates_pmap_with_multiple_elements():
    from pyrsistent import pmap
    m = pmap({1: 'one', 2: 'two', 3: 'three'})
    assert m._size == 3
    assert m[1] == 'one'
    assert m[2] == 'two'
    assert m[3] == 'three'

def test_constructor_handles_none_key():
    from pyrsistent import pmap
    m = pmap({None: 'null'})
    assert m._size == 1
    assert m[None] == 'null'

def test_constructor_handles_duplicate_keys_last_wins():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'a': 2})
    assert m._size == 1
    assert m['a'] == 2

def test_constructor_creates_pmap_from_empty_dict():
    from pyrsistent import pmap
    m = pmap({})
    assert m._size == 0
    assert dict(m) == {}

def test_constructor_creates_pmap_from_dict_with_various_types():
    from pyrsistent import pmap
    m = pmap({'int': 42, 'float': 3.14, 'str': 'hello', 'tuple': (1, 2)})
    assert m._size == 4
    assert m['int'] == 42
    assert m['float'] == 3.14
    assert m['str'] == 'hello'
    assert m['tuple'] == (1, 2)

def test_constructor_creates_pmap_with_colliding_keys():
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
    m = pmap({key1: 'first', key2: 'second'})
    assert m._size == 2
    assert m[key1] == 'first'
    assert m[key2] == 'second'

def test_constructor_creates_pmap_that_is_hashable():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    assert hash(m1) == hash(m2)
    assert m1 == m2

def test_constructor_creates_pmap_with_identical_buckets_for_same_input():
    from pyrsistent import pmap
    m1 = pmap({'x': 10, 'y': 20})
    m2 = pmap({'x': 10, 'y': 20})
    assert m1._buckets == m2._buckets

def test_constructor_creates_pmap_with_correct_bucket_structure():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    index = hash('a') % len(m._buckets)
    bucket = m._buckets[index]
    assert bucket is not None
    assert len(bucket) == 1
    assert bucket[0] == ('a', 1)


# LLM-generated content at query #19
#--------------------------

def test___eq___same_instance():
    m1 = m(a=1, b=2)
    result = m1 == m1
    assert result is True

def test___eq___equal_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test___eq___different_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    result = m1 == m2
    assert result is False

def test___eq___different_lengths():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2, c=3)
    result = m1 == m2
    assert result is False

def test___eq___equal_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test___eq___different_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = m1 == d
    assert result is False

def test___eq___not_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [1, 2, 3]
    assert result is NotImplemented

def test___eq___same_buckets():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1._buckets == m2._buckets
    assert result is True

def test___eq___different_cached_hash():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    m1._cached_hash = hash(frozenset(m1.iteritems()))
    m2._cached_hash = hash(frozenset(m2.iteritems()))
    result = m1 == m2
    assert result is False

def test___eq___same_cached_hash():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = hash(frozenset(m1.iteritems()))
    m2._cached_hash = hash(frozenset(m2.iteritems()))
    result = m1 == m2
    assert result is True


# LLM-generated content at query #20
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

def test_turbo_mapping_handles_exception_in_len():
    class BadLen:
        def __len__(self):
            raise Exception("no length")
    initial = BadLen()
    result = _turbo_mapping(initial, 0)
    assert len(result) == 0
    assert dict(result) == {}


# LLM-generated content at query #21
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

def test_update_with_inserts_new_key_from_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, {'b': 2}, {'c': 3})
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_overwrites_with_rightmost_when_update_fn_returns_right():
    from pyrsistent import m
    m1 = m(a=1, b=1)
    m2 = m1.update_with(lambda l, r: r, {'a': 2}, {'a': 3})
    assert m2 == {'a': 3, 'b': 1}

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

def test_update_with_handles_complex_merge_logic():
    from pyrsistent import m
    def choose_larger(l, r):
        return l if l > r else r
    m1 = m(a=5, b=10)
    m2 = m1.update_with(choose_larger, {'a': 3, 'b': 15}, {'a': 7})
    assert m2 == {'a': 7, 'b': 15}

def test_update_with_preserves_original_when_other_maps_empty():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r)
    assert m2 is m1


# LLM-generated content at query #22
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


# LLM-generated content at query #23
#--------------------------

def test_constructor_creates_pmap_with_size_and_buckets():
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

def test_constructor_with_dict():
    from pyrsistent import pmap
    data = {'a': 1, 'b': 2}
    pmap_instance = pmap(data)
    assert len(pmap_instance) == 2
    assert pmap_instance['a'] == 1
    assert pmap_instance['b'] == 2

def test_constructor_with_keyword_arguments():
    from pyrsistent import m
    pmap_instance = m(x=10, y=20)
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

def test_constructor_preserves_identity_for_empty():
    from pyrsistent import pmap
    empty1 = pmap()
    empty2 = pmap()
    assert empty1 is empty2

def test_constructor_handles_none_values():
    from pyrsistent import pmap
    data = {'key': None}
    pmap_instance = pmap(data)
    assert len(pmap_instance) == 1
    assert pmap_instance['key'] is None

def test_constructor_with_duplicate_keys_last_wins():
    from pyrsistent import pmap
    pairs = [('a', 1), ('a', 2)]
    pmap_instance = pmap(pairs)
    assert len(pmap_instance) == 1
    assert pmap_instance['a'] == 2


# LLM-generated content at query #24
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
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert m2 == {'a': 3}

def test_update_with_on_empty_map():
    from pyrsistent import m
    m1 = m()
    m2 = m1.update_with(lambda l, r: l + r, m(a=1, b=2))
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_returns_same_instance_if_no_changes():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m())
    assert m2 is m1

def test_update_with_handles_complex_update_fn():
    from pyrsistent import m
    m1 = m(a=5, b=10)
    m2 = m1.update_with(lambda l, r: l * r, m(a=2, b=3))
    assert m2 == {'a': 10, 'b': 30}

def test_update_with_preserves_original_map():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=10))
    assert m1 == {'a': 1, 'b': 2}
    assert m2 == {'a': 11, 'b': 2}

def test_update_with_using_dict_and_pmap():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l + r, {'a': 2}, m(a=3))
    assert m2 == {'a': 6}


# LLM-generated content at query #25
#--------------------------

def test_constructor_creates_pmap_with_correct_size_and_buckets():
    size = 5
    buckets = pvector([None, None, None, None, None])
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets

def test_constructor_creates_pmap_with_non_empty_buckets():
    size = 2
    buckets = pvector([[('key1', 'value1')], [('key2', 'value2')]])
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets

def test_constructor_creates_pmap_with_mixed_buckets():
    size = 3
    buckets = pvector([None, [('key1', 'value1')], [('key2', 'value2'), ('key3', 'value3')]])
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets

def test_constructor_creates_pmap_with_zero_size():
    size = 0
    buckets = pvector([])
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets

def test_constructor_creates_pmap_with_single_bucket():
    size = 1
    buckets = pvector([[('key', 'value')]])
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets


# LLM-generated content at query #26
#--------------------------

def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(c=3)
    result = pmap1.update_with(lambda l, r: l + r, pmap2)
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #27
#--------------------------

def test_pmap_constructor_creates_instance_with_size_and_buckets():
    from pyrsistent import PVector
    size = 2
    buckets = PVector([None, [('a', 1), ('b', 2)], None])
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets

def test_pmap_constructor_returns_pmap_instance():
    from pyrsistent import PVector
    size = 0
    buckets = PVector([None, None, None])
    pmap_instance = PMap(size, buckets)
    assert isinstance(pmap_instance, PMap)

def test_pmap_constructor_sets_correct_size():
    from pyrsistent import PVector
    size = 5
    buckets = PVector([None] * 10)
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size

def test_pmap_constructor_sets_correct_buckets():
    from pyrsistent import PVector
    buckets = PVector([[('key1', 'value1')], None, [('key2', 'value2')]])
    pmap_instance = PMap(2, buckets)
    assert pmap_instance._buckets == buckets

def test_pmap_constructor_creates_empty_map():
    from pyrsistent import PVector
    size = 0
    buckets = PVector([])
    pmap_instance = PMap(size, buckets)
    assert len(pmap_instance) == 0

def test_pmap_constructor_handles_non_empty_buckets():
    from pyrsistent import PVector
    buckets = PVector([[('x', 10)], None, [('y', 20)]])
    pmap_instance = PMap(2, buckets)
    assert pmap_instance['x'] == 10
    assert pmap_instance['y'] == 20

def test_pmap_constructor_sets_weakref_slot():
    from pyrsistent import PVector
    size = 0
    buckets = PVector([None])
    pmap_instance = PMap(size, buckets)
    assert hasattr(pmap_instance, '__weakref__')

def test_pmap_constructor_initializes_without_cached_hash():
    from pyrsistent import PVector
    size = 0
    buckets = PVector([None])
    pmap_instance = PMap(size, buckets)
    assert not hasattr(pmap_instance, '_cached_hash')

def test_pmap_constructor_produces_hashable_instance():
    from pyrsistent import PVector
    size = 1
    buckets = PVector([[('k', 'v')]])
    pmap_instance = PMap(size, buckets)
    hash_value = hash(pmap_instance)
    assert isinstance(hash_value, int)

def test_pmap_constructor_allows_dot_notation_access():
    from pyrsistent import PVector
    buckets = PVector([[('attr', 42)]])
    pmap_instance = PMap(1, buckets)
    assert pmap_instance.attr == 42


# LLM-generated content at query #28
#--------------------------

def test_contains_with_valid_key_value_pair():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'a') in items_view
    assert result == True

def test_contains_with_valid_key_but_wrong_value():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'b') in items_view
    assert result == False

def test_contains_with_missing_key():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (3, 'a') in items_view
    assert result == False

def test_contains_with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = 'not_a_tuple' in items_view
    assert result == False

def test_contains_with_tuple_of_wrong_length():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'a', 'extra') in items_view
    assert result == False

def test_contains_with_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    result = (1, 'a') in items_view
    assert result == False


# LLM-generated content at query #29
#--------------------------

def test_turbo_mapping_predicate_at_line_7_false():
    initial = [1, 2, 3]
    pre_size = None
    result = _turbo_mapping(initial, pre_size)
    assert isinstance(result, PMap)


# LLM-generated content at query #30
#--------------------------

def test___eq___same_instance():
    m1 = m(a=1, b=2)
    result = m1 == m1
    assert result is True

def test___eq___equal_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test___eq___different_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    result = m1 == m2
    assert result is False

def test___eq___different_lengths():
    m1 = m(a=1, b=2)
    m2 = m(a=1)
    result = m1 == m2
    assert result is False

def test___eq___with_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test___eq___with_dict_different():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = m1 == d
    assert result is False

def test___eq___with_other_mapping():
    from collections import OrderedDict
    m1 = m(a=1, b=2)
    od = OrderedDict([('a', 1), ('b', 2)])
    result = m1 == od
    assert result is True

def test___eq___non_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [('a', 1), ('b', 2)]
    assert result is False

def test___eq___cached_hash_mismatch():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    m1._cached_hash = hash(frozenset(m1.iteritems()))
    m2._cached_hash = hash(frozenset(m2.iteritems()))
    result = m1 == m2
    assert result is False

def test___eq___cached_hash_match():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = hash(frozenset(m1.iteritems()))
    m2._cached_hash = hash(frozenset(m2.iteritems()))
    result = m1 == m2
    assert result is True

def test___eq___same_buckets():
    from pyrsistent import pvector
    bucket = [('a', 1), ('b', 2)]
    buckets = pvector([bucket])
    m1 = PMap(2, buckets)
    m2 = PMap(2, buckets)
    result = m1 == m2
    assert result is True


# LLM-generated content at query #31
#--------------------------

def test_contains_with_non_tuple_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a'})
    items = m.items()
    result = (1, 'a', 'extra') in items
    assert result == False


# LLM-generated content at query #32
#--------------------------

def test_pmap_constructor_creates_instance_with_size_and_buckets():
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == size
    assert pmap_instance._buckets == buckets

def test_pmap_constructor_returns_pmap_instance():
    size = 0
    buckets = ()
    pmap_instance = PMap(size, buckets)
    assert isinstance(pmap_instance, PMap)

def test_pmap_constructor_sets_size_zero_for_empty_map():
    size = 0
    buckets = ()
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == 0

def test_pmap_constructor_sets_size_for_non_empty_map():
    size = 3
    buckets = (('a', 1), ('b', 2), ('c', 3))
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._size == 3

def test_pmap_constructor_assigns_buckets_correctly():
    size = 1
    buckets = (('test_key', 'test_value'),)
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._buckets == buckets

def test_pmap_constructor_creates_instance_with_correct_slots():
    size = 0
    buckets = ()
    pmap_instance = PMap(size, buckets)
    assert hasattr(pmap_instance, '_size')
    assert hasattr(pmap_instance, '_buckets')
    assert hasattr(pmap_instance, '_cached_hash')
    assert not hasattr(pmap_instance, '__dict__')

def test_pmap_constructor_initializes_cached_hash_as_not_set():
    size = 0
    buckets = ()
    pmap_instance = PMap(size, buckets)
    assert not hasattr(pmap_instance, '_cached_hash') or pmap_instance._cached_hash is None

def test_pmap_constructor_handles_empty_buckets():
    size = 0
    buckets = ()
    pmap_instance = PMap(size, buckets)
    assert pmap_instance._buckets == ()

def test_pmap_constructor_handles_non_empty_buckets():
    size = 2
    buckets = (('x', 10), ('y', 20))
    pmap_instance = PMap(size, buckets)
    assert len(pmap_instance._buckets) == 2

def test_pmap_constructor_maintains_identity():
    size = 5
    buckets = tuple(range(5))
    pmap_instance1 = PMap(size, buckets)
    pmap_instance2 = PMap(size, buckets)
    assert pmap_instance1 is not pmap_instance2
    assert pmap_instance1._size == pmap_instance2._size
    assert pmap_instance1._buckets == pmap_instance2._buckets


# LLM-generated content at query #33
#--------------------------

def test_update_with_merges_values_using_function():
    from operator import add

    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(add, m(a=2))
    assert m2 == {'a': 3, 'b': 2}

def test_update_with_keeps_leftmost_value_when_function_returns_left():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert m2 == {'a': 1}

def test_update_with_inserts_new_keys_from_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r, m(b=2), {'c': 3})
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_handles_empty_maps():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r)
    assert m2 == m1

def test_update_with_merges_with_overriding_values():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: r + l, m(a=10, c=30), {'a': 100})
    assert m2 == {'a': 110, 'b': 2, 'c': 30}

def test_update_with_on_empty_pmap():
    from pyrsistent import m
    m1 = m()
    m2 = m1.update_with(lambda l, r: r, {'x': 10}, m(y=20))
    assert m2 == {'x': 10, 'y': 20}

def test_update_with_returns_same_instance_if_no_changes():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m())
    assert m2 is m1

def test_update_with_uses_function_for_existing_keys_only():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: 999, m(a=5, c=10))
    assert m2 == {'a': 999, 'b': 2, 'c': 10}


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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

def test_constructor_handles_duplicate_keys_last_wins():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'a': 2})
    assert m._size == 1
    assert m['a'] == 2

def test_constructor_preserves_hash_collisions_handling():
    from pyrsistent import pmap
    class SameHash:
        def __hash__(self):
            return 1
    key1 = SameHash()
    key2 = SameHash()
    m = pmap({key1: 'first', key2: 'second'})
    assert m._size == 2
    assert m[key1] == 'first'
    assert m[key2] == 'second'

def test_constructor_creates_pmap_from_empty_dict():
    from pyrsistent import pmap
    m = pmap({})
    assert m._size == 0
    assert len(m) == 0
    assert list(m) == []

def test_constructor_creates_pmap_from_non_empty_dict():
    from pyrsistent import pmap
    m = pmap({'x': 10, 'y': 20})
    assert m._size == 2
    assert m['x'] == 10
    assert m['y'] == 20

def test_constructor_creates_pmap_with_mixed_key_types():
    from pyrsistent import pmap
    m = pmap({1: 'int', 'str': 'string', (1, 2): 'tuple'})
    assert m._size == 3
    assert m[1] == 'int'
    assert m['str'] == 'string'
    assert m[(1, 2)] == 'tuple'


# LLM-generated content at query #2
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

def test_eq_different_lengths():
    m1 = m(a=1, b=2)
    m2 = m(a=1)
    result = m1 == m2
    assert result is False

def test_eq_with_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test_eq_with_dict_different():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = m1 == d
    assert result is False

def test_eq_with_mapping_protocol():
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

def test_eq_not_implemented_for_non_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [1, 2]
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
    result_eq = m1 == m2
    assert result_eq is True

def test_eq_with_empty():
    m1 = m()
    m2 = m()
    result = m1 == m2
    assert result is True

def test_eq_empty_with_dict():
    m1 = m()
    d = {}
    result = m1 == d
    assert result is True


# LLM-generated content at query #3
#--------------------------

def test_eq_with_different_cached_hash_returns_false():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    m1._cached_hash = 123
    m2._cached_hash = 456
    result = m1 == m2
    assert result is False


# LLM-generated content at query #4
#--------------------------

def test_update_with_merges_values_using_update_fn():
    from operator import add

    from pyrsistent import m
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
    from operator import sub

    from pyrsistent import m
    m1 = m(a=10, b=5)
    m2 = m1.update_with(sub, m(a=2, b=1))
    assert m2 == {'a': 8, 'b': 4}

def test_update_with_returns_same_instance_if_no_changes():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m())
    assert m2 is m1

def test_update_with_handles_empty_maps():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: r)
    assert m2 == {'a': 1}
    assert m2 is m1

def test_update_with_uses_update_fn_for_each_key_collision():
    from pyrsistent import m
    def concat(l, r):
        return l + ',' + r
    m1 = m(x='hello', y='world')
    m2 = m1.update_with(concat, m(x='there', y='universe'))
    assert m2 == {'x': 'hello,there', 'y': 'world,universe'}

def test_update_with_preserves_non_colliding_keys():
    from pyrsistent import m
    m1 = m(a=1, b=2, c=3)
    m2 = m1.update_with(lambda l, r: r, m(b=20, d=4))
    assert m2 == {'a': 1, 'b': 20, 'c': 3, 'd': 4}


# LLM-generated content at query #5
#--------------------------

def test__turbo_mapping_with_empty_initial_and_no_pre_size():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0
    assert dict(result) == {}

def test__turbo_mapping_with_dict_initial_and_no_pre_size():
    initial = {'a': 1, 'b': 2}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert result['a'] == 1
    assert result['b'] == 2

def test__turbo_mapping_with_mapping_initial_and_no_pre_size():
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
    initial = TestMapping({'x': 10, 'y': 20})
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert result['x'] == 10
    assert result['y'] == 20

def test__turbo_mapping_with_iterable_initial_and_no_pre_size():
    initial = [('p', 100), ('q', 200)]
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert result['p'] == 100
    assert result['q'] == 200

def test__turbo_mapping_with_pre_size_specified():
    initial = {'a': 1, 'b': 2, 'c': 3}
    result = _turbo_mapping(initial, 32)
    assert len(result) == 3
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['c'] == 3

def test__turbo_mapping_with_collision_handling():
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

def test__turbo_mapping_returns_pmap_instance():
    from pyrsistent import PMap
    result = _turbo_mapping({'test': 42}, 0)
    assert isinstance(result, PMap)

def test__turbo_mapping_with_zero_pre_size_and_non_empty_initial():
    initial = {'z': 99}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 1
    assert result['z'] == 99

def test__turbo_mapping_with_large_initial_and_small_pre_size():
    initial = {i: i*2 for i in range(10)}
    result = _turbo_mapping(initial, 4)
    assert len(result) == 10
    for i in range(10):
        assert result[i] == i*2


# LLM-generated content at query #6
#--------------------------

def test_eq_with_dict_and_different_buckets():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = {'a': 1, 'b': 2}
    result = m1 == m2
    assert result is True


# LLM-generated content at query #7
#--------------------------

def test_turbo_mapping_predicate_at_line_7_false():
    initial = [1, 2, 3]
    pre_size = None
    result = _turbo_mapping(initial, pre_size)
    assert isinstance(result, PMap)


# LLM-generated content at query #8
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

def test_turbo_mapping_preserves_hash():
    input_dict = {'foo': 'bar'}
    result = _turbo_mapping(input_dict, 0)
    assert hash(result) == hash(frozenset(input_dict.items()))

def test_turbo_mapping_large_pre_size():
    result = _turbo_mapping({'a': 1}, 100)
    assert len(result) == 1
    assert result['a'] == 1

def test_turbo_mapping_with_zero_pre_size_and_empty():
    result = _turbo_mapping({}, 0)
    assert len(result) == 0
    assert dict(result) == {}

def test_turbo_mapping_with_existing_dict():
    original = {'z': 9, 'w': 8}
    result = _turbo_mapping(original, 0)
    assert dict(result) == original

def test_turbo_mapping_handles_non_integer_len():
    class BadLenMapping:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            raise TypeError("no length")
        def items(self):
            return self.data.items()
    bad_mapping = BadLenMapping({'k': 'v'})
    result = _turbo_mapping(bad_mapping, 0)
    assert len(result) == 1
    assert result['k'] == 'v'


# LLM-generated content at query #9
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

def test_eq_pmap_and_dict_equal():
    pm = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = pm == d
    assert result is True

def test_eq_pmap_and_dict_not_equal():
    pm = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = pm == d
    assert result is False

def test_eq_pmap_and_dict_different_length():
    pm = m(a=1, b=2, c=3)
    d = {'a': 1, 'b': 2}
    result = pm == d
    assert result is False

def test_eq_with_non_mapping():
    pm = m(a=1, b=2)
    result = pm == [1, 2]
    assert result is NotImplemented

def test_eq_pmap_and_other_mapping_equal():
    from collections import OrderedDict
    pm = m(a=1, b=2)
    od = OrderedDict([('a', 1), ('b', 2)])
    result = pm == od
    assert result is True

def test_eq_pmap_and_other_mapping_not_equal():
    from collections import OrderedDict
    pm = m(a=1, b=2)
    od = OrderedDict([('a', 1), ('b', 3)])
    result = pm == od
    assert result is False

def test_eq_pmaps_with_different_buckets_same_content():
    pm1 = m(a=1, b=2)
    evolver = pm1.evolver()
    evolver.set('c', 3)
    evolver.remove('c')
    pm2 = evolver.persistent()
    result = pm1 == pm2
    assert result is True

def test_eq_pmaps_with_cached_hash_mismatch():
    pm1 = m(a=1, b=2)
    pm2 = m(a=1, b=3)
    hash(pm1)
    hash(pm2)
    result = pm1 == pm2
    assert result is False

def test_eq_pmaps_with_cached_hash_match_but_different_content():
    pm1 = m(a=1, b=2)
    pm2 = m(a=1, b=2)
    hash(pm1)
    hash(pm2)
    result = pm1 == pm2
    assert result is True


# LLM-generated content at query #10
#--------------------------

def test_eq_with_dict_equal_but_different_buckets():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True


# LLM-generated content at query #11
#--------------------------

def test___contains___with_existing_key_value_pair():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'a') in items_view
    assert result is True

def test___contains___with_existing_key_but_different_value():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'b') in items_view
    assert result is False

def test___contains___with_non_existing_key():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (3, 'a') in items_view
    assert result is False

def test___contains___with_non_tuple_argument():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = 'not_a_tuple' in items_view
    assert result is False

def test___contains___with_wrong_length_tuple():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'a', 'extra') in items_view
    assert result is False

def test___contains___with_empty_pmap():
    from pyrsistent import pmap
    m = pmap({})
    items_view = m.items()
    result = (1, 'a') in items_view
    assert result is False


# LLM-generated content at query #12
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

def test__turbo_mapping_with_non_mapping_initial():
    initial = [('a', 1), ('b', 2)]
    result = _turbo_mapping(initial, 0)
    assert len(result) == 2
    assert dict(result) == {'a': 1, 'b': 2}

def test__turbo_mapping_with_collision_keys():
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

def test__turbo_mapping_with_large_initial():
    initial = {i: i*2 for i in range(100)}
    result = _turbo_mapping(initial, 0)
    assert len(result) == 100
    for i in range(100):
        assert result[i] == i*2

def test__turbo_mapping_preserves_hash_based_lookup():
    initial = {'x': 10, 'y': 20}
    result = _turbo_mapping(initial, 0)
    assert result['x'] == 10
    assert result['y'] == 20

def test__turbo_mapping_handles_initial_with_duplicate_keys():
    initial = [('a', 1), ('a', 2)]
    result = _turbo_mapping(initial, 0)
    assert len(result) == 1
    assert result['a'] == 2

def test__turbo_mapping_with_pre_size_smaller_than_initial():
    initial = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    result = _turbo_mapping(initial, 2)
    assert len(result) == 4
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3, 'd': 4}


# LLM-generated content at query #13
#--------------------------

def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    pm = m(a=1)
    result = pm.update_with(lambda l, r: l + r, m(b=2))
    assert result == {'a': 1, 'b': 2}

def test_update_with_key_in_evolver():
    from pyrsistent import m
    pm = m(a=1)
    result = pm.update_with(lambda l, r: l + r, m(a=2))
    assert result == {'a': 3}

def test_update_with_multiple_maps_key_not_in_evolver():
    from pyrsistent import m
    pm = m(a=1)
    result = pm.update_with(lambda l, r: l + r, m(b=2), m(c=3))
    assert result == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_multiple_maps_key_in_evolver():
    from pyrsistent import m
    pm = m(a=1, b=2)
    result = pm.update_with(lambda l, r: l + r, m(a=10, c=30), m(b=20))
    assert result == {'a': 11, 'b': 22, 'c': 30}

def test_update_with_empty_maps():
    from pyrsistent import m
    pm = m(a=1, b=2)
    result = pm.update_with(lambda l, r: l + r)
    assert result == {'a': 1, 'b': 2}

def test_update_with_initial_empty_pmap():
    from pyrsistent import m
    pm = m()
    result = pm.update_with(lambda l, r: l + r, m(a=1, b=2))
    assert result == {'a': 1, 'b': 2}

def test_update_with_using_dict():
    from pyrsistent import m
    pm = m(a=1)
    result = pm.update_with(lambda l, r: l + r, {'b': 2})
    assert result == {'a': 1, 'b': 2}

def test_update_with_using_dict_key_in_evolver():
    from pyrsistent import m
    pm = m(a=1)
    result = pm.update_with(lambda l, r: l + r, {'a': 2})
    assert result == {'a': 3}

def test_update_with_update_fn_returns_leftmost():
    from pyrsistent import m
    pm = m(a=1)
    result = pm.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1}

def test_update_with_update_fn_returns_rightmost():
    from pyrsistent import m
    pm = m(a=1)
    result = pm.update_with(lambda l, r: r, m(a=2), {'a': 3})
    assert result == {'a': 3}


# LLM-generated content at query #14
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('a', 1), ('b', 2))
    m = pmap({'a': 1, 'b': 2})
    assert m._size == size
    assert len(m._buckets) > 0

def test_constructor_returns_pmap_instance():
    from pyrsistent import PMap
    size = 0
    buckets = ()
    m = PMap(size, buckets)
    assert isinstance(m, PMap)

def test_constructor_sets_size_and_buckets():
    from pyrsistent import PMap
    size = 5
    buckets = (1, 2, 3, 4, 5)
    m = PMap(size, buckets)
    assert m._size == size
    assert m._buckets == buckets

def test_constructor_with_zero_size_and_empty_buckets():
    from pyrsistent import PMap
    size = 0
    buckets = ()
    m = PMap(size, buckets)
    assert m._size == 0
    assert len(m._buckets) == 0

def test_constructor_pmap_is_hashable():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    hash(m)

def test_constructor_pmap_supports_dot_notation():
    from pyrsistent import pmap
    m = pmap({'key': 'value'})
    assert m.key == 'value'

def test_constructor_pmap_implements_mapping_protocol():
    from collections.abc import Mapping

    from pyrsistent import pmap
    m = pmap({'a': 1})
    assert isinstance(m, Mapping)

def test_constructor_pmap_has_correct_length():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2, 'c': 3})
    assert len(m) == 3

def test_constructor_pmap_is_iterable():
    from pyrsistent import pmap
    m = pmap({'a': 1, 'b': 2})
    keys = list(m)
    assert set(keys) == {'a', 'b'}

def test_constructor_pmap_has_no_public_constructor():
    from pyrsistent import PMap
    try:
        PMap()
    except TypeError:
        pass
    else:
        assert False, "PMap should not have a public constructor"

def test_constructor_pmap_created_via_factory_function():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    assert m._size == 1
    assert len(m._buckets) > 0

def test_constructor_pmap_with_complex_keys():
    from pyrsistent import pmap
    key1 = ('tuple', 'key')
    key2 = frozenset([1, 2, 3])
    m = pmap({key1: 'value1', key2: 'value2'})
    assert m[key1] == 'value1'
    assert m[key2] == 'value2'

def test_constructor_pmap_buckets_are_immutable():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    try:
        m._buckets.append('something')
    except AttributeError:
        pass
    else:
        assert False, "Buckets should be immutable"

def test_constructor_pmap_weakref_support():
    import weakref

    from pyrsistent import pmap
    m = pmap({'a': 1})
    ref = weakref.ref(m)
    assert ref() is m

def test_constructor_pmap_slots_defined():
    from pyrsistent import PMap
    assert PMap.__slots__ == ('_size', '_buckets', '__weakref__', '_cached_hash')

def test_constructor_pmap_no_extra_attributes():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    try:
        m.new_attribute = 'value'
    except AttributeError:
        pass
    else:
        assert False, "PMap should not allow new attributes"


# LLM-generated content at query #15
#--------------------------

def test_eq_with_dict_equal_but_different_buckets():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    result = m1 == m2
    assert result is True


# LLM-generated content at query #16
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    size = 2
    buckets = (("key1", "value1"), ("key2", "value2"))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_returns_pmap_instance():
    pmap = PMap(0, ())
    assert isinstance(pmap, PMap)

def test_constructor_sets_correct_size_for_empty_pmap():
    pmap = PMap(0, ())
    assert len(pmap) == 0

def test_constructor_sets_correct_size_for_non_empty_pmap():
    pmap = PMap(3, (("a", 1), ("b", 2), ("c", 3)))
    assert len(pmap) == 3

def test_constructor_pmap_has_no_cached_hash_initially():
    pmap = PMap(0, ())
    assert not hasattr(pmap, "_cached_hash")

def test_constructor_pmap_with_buckets_contains_keys():
    pmap = PMap(2, (("x", 10), ("y", 20)))
    assert "x" in pmap
    assert "y" in pmap

def test_constructor_pmap_with_buckets_returns_correct_values():
    pmap = PMap(2, (("x", 10), ("y", 20)))
    assert pmap["x"] == 10
    assert pmap["y"] == 20

def test_constructor_pmap_with_duplicate_keys_in_buckets_handles_collisions():
    bucket = [("k1", "v1"), ("k2", "v2")]
    pmap = PMap(2, (bucket,))
    assert pmap._size == 2
    assert pmap._buckets == (bucket,)

def test_constructor_pmap_with_none_buckets_allowed():
    pmap = PMap(0, (None, None, None))
    assert pmap._size == 0
    assert pmap._buckets == (None, None, None)

def test_constructor_pmap_is_hashable_after_creation():
    pmap = PMap(0, ())
    hash(pmap)

def test_constructor_pmap_equality_with_itself():
    pmap = PMap(1, (("a", 1),))
    assert pmap == pmap

def test_constructor_pmap_equality_with_same_buckets():
    buckets = (("k", "v"),)
    pmap1 = PMap(1, buckets)
    pmap2 = PMap(1, buckets)
    assert pmap1 == pmap2

def test_constructor_pmap_iteration_over_keys():
    pmap = PMap(3, (("a", 1), ("b", 2), ("c", 3)))
    keys = list(pmap)
    assert set(keys) == {"a", "b", "c"}

def test_constructor_pmap_iteritems_yields_all_items():
    pmap = PMap(2, (("k1", "v1"), ("k2", "v2")))
    items = list(pmap.iteritems())
    assert items == [("k1", "v1"), ("k2", "v2")]

def test_constructor_pmap_getattr_accesses_items():
    pmap = PMap(1, (("attr", 42),))
    assert pmap.attr == 42

def test_constructor_pmap_getattr_raises_attribute_error_for_missing_key():
    pmap = PMap(0, ())
    try:
        pmap.missing
        assert False
    except AttributeError:
        assert True

def test_constructor_pmap_repr_for_empty_map():
    pmap = PMap(0, ())
    assert repr(pmap) == "pmap({})"

def test_constructor_pmap_repr_for_non_empty_map():
    pmap = PMap(1, (("key", "value"),))
    assert repr(pmap) == "pmap({'key': 'value'})"

def test_constructor_pmap_str_equals_repr():
    pmap = PMap(1, (("x", 1),))
    assert str(pmap) == repr(pmap)

def test_constructor_pmap_set_creates_new_pmap():
    pmap = PMap(1, (("a", 1),))
    new_pmap = pmap.set("b", 2)
    assert pmap._size == 1
    assert new_pmap._size == 2

def test_constructor_pmap_remove_creates_new_pmap():
    pmap = PMap(2, (("a", 1), ("b", 2)))
    new_pmap = pmap.remove("a")
    assert pmap._size == 2
    assert new_pmap._size == 1

def test_constructor_pmap_discard_returns_same_pmap_if_key_missing():
    pmap = PMap(1, (("a", 1),))
    same_pmap = pmap.discard("b")
    assert pmap is same_pmap

def test_constructor_pmap_update_creates_new_pmap():
    pmap = PMap(1, (("a", 1),))
    new_pmap = pmap.update({"b": 2})
    assert pmap._size == 1
    assert new_pmap._size == 2

def test_constructor_pmap_evolver_returns_evolver_instance():
    pmap = PMap(0, ())
    evolver = pmap.evolver()
    assert isinstance(evolver, PMap._Evolver)

def test_constructor_pmap_evolver_has_same_size():
    pmap = PMap(3, (("a", 1), ("b", 2), ("c", 3)))
    evolver = pmap.evolver()
    assert len(evolver) == 3


# LLM-generated content at query #17
#--------------------------

def test_update_with_key_not_in_evolver():
    from operator import add

    from pyrsistent import m
    pm = m(a=1, b=2)
    result = pm.update_with(add, m(c=3))
    assert result == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_key_in_evolver():
    from operator import add

    from pyrsistent import m
    pm = m(a=1, b=2)
    result = pm.update_with(add, m(a=2))
    assert result == {'a': 3, 'b': 2}

def test_update_with_multiple_maps_key_not_in_evolver():
    from pyrsistent import m
    pm = m(a=1)
    result = pm.update_with(lambda l, r: l + r, m(b=2), m(c=3))
    assert result == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_empty_maps():
    from pyrsistent import m
    pm = m(a=1, b=2)
    result = pm.update_with(lambda l, r: l + r)
    assert result == pm

def test_update_with_key_not_in_evolver_using_dict():
    from pyrsistent import m
    pm = m(a=1)
    result = pm.update_with(lambda l, r: r, {'b': 2})
    assert result == {'a': 1, 'b': 2}

def test_update_with_key_in_evolver_from_second_map():
    from pyrsistent import m
    pm = m(a=1, b=2)
    result = pm.update_with(lambda l, r: l * r, m(b=3), m(b=4))
    assert result == {'a': 1, 'b': 12}


# LLM-generated content at query #18
#--------------------------

def test_update_with_merges_values_using_function():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l + r, m(a=2))
    assert result == {'a': 3, 'b': 2}

def test_update_with_keeps_leftmost_value_when_function_returns_left():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: l, m(a=2), {'a': 3})
    assert result == {'a': 1}

def test_update_with_inserts_new_keys_from_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1)
    result = m1.update_with(lambda l, r: r, m(b=2), {'c': 3})
    assert result == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_overwrites_with_rightmost_when_function_returns_right():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: r, m(a=2, b=3), {'a': 4})
    assert result == {'a': 4, 'b': 3}

def test_update_with_on_empty_map():
    from pyrsistent import m
    m1 = m()
    result = m1.update_with(lambda l, r: l + r, {'a': 1, 'b': 2})
    assert result == {'a': 1, 'b': 2}

def test_update_with_returns_same_instance_if_no_changes():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    result = m1.update_with(lambda l, r: l, m())
    assert result is m1

def test_update_with_handles_complex_merge_logic():
    from pyrsistent import m
    m1 = m(x=10, y=20)
    def choose_larger(l, r):
        return l if l > r else r
    result = m1.update_with(choose_larger, m(x=5, y=30), {'x': 15, 'y': 10})
    assert result == {'x': 15, 'y': 30}

def test_update_with_preserves_original_map_unchanged():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l + r, m(a=3))
    assert m1 == {'a': 1, 'b': 2}
    assert m2 == {'a': 4, 'b': 2}


# LLM-generated content at query #19
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


# LLM-generated content at query #20
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap_instance = pmap(dict(buckets))
    assert len(pmap_instance) == size
    assert pmap_instance['key1'] == 'value1'
    assert pmap_instance['key2'] == 'value2'

def test_constructor_creates_empty_pmap():
    from pyrsistent import pmap
    pmap_instance = pmap({})
    assert len(pmap_instance) == 0
    assert list(pmap_instance) == []

def test_constructor_handles_none_values():
    from pyrsistent import pmap
    pmap_instance = pmap({'key': None})
    assert pmap_instance['key'] is None

def test_constructor_preserves_hash_collisions():
    from pyrsistent import pmap
    class SameHash:
        def __hash__(self):
            return 1
    key1 = SameHash()
    key2 = SameHash()
    pmap_instance = pmap({key1: 'val1', key2: 'val2'})
    assert pmap_instance[key1] == 'val1'
    assert pmap_instance[key2] == 'val2'

def test_constructor_with_large_dict():
    from pyrsistent import pmap
    large_dict = {str(i): i for i in range(100)}
    pmap_instance = pmap(large_dict)
    assert len(pmap_instance) == 100
    assert pmap_instance['50'] == 50

def test_constructor_creates_independent_instances():
    from pyrsistent import pmap
    dict1 = {'a': 1}
    pmap1 = pmap(dict1)
    pmap2 = pmap(dict1)
    assert pmap1 == pmap2
    assert pmap1 is not pmap2

def test_constructor_with_immutable_keys():
    from pyrsistent import pmap
    tuple_key = (1, 2)
    pmap_instance = pmap({tuple_key: 'value'})
    assert pmap_instance[tuple_key] == 'value'

def test_constructor_handles_duplicate_keys():
    from pyrsistent import pmap
    pmap_instance = pmap({'a': 1, 'a': 2})
    assert pmap_instance['a'] == 2

def test_constructor_from_pmap_returns_same_instance():
    from pyrsistent import pmap
    original = pmap({'x': 10})
    new = pmap(original)
    assert new is original

def test_constructor_with_mapping_protocol():
    from pyrsistent import pmap
    class CustomMapping:
        def __init__(self, data):
            self.data = data
        def __getitem__(self, key):
            return self.data[key]
        def __iter__(self):
            return iter(self.data)
        def __len__(self):
            return len(self.data)
    custom = CustomMapping({'alpha': 'beta'})
    pmap_instance = pmap(custom)
    assert pmap_instance['alpha'] == 'beta'


# LLM-generated content at query #21
#--------------------------

def test_contains_returns_false_on_exception():
    class MockMap:
        pass
    view = PMapItems(MockMap())
    result = (1,) in view
    assert result == False


# LLM-generated content at query #22
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

def test_contains_with_same_object():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    items = m.items()
    result = items in items
    assert result == False


# LLM-generated content at query #23
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

def test_update_with_inserts_new_key_from_maps():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l + r, m(b=2), {'c': 3})
    assert m2 == {'a': 1, 'b': 2, 'c': 3}

def test_update_with_handles_multiple_maps():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l + r, m(a=2, b=1), m(b=3, c=4))
    assert m2 == {'a': 3, 'b': 4, 'c': 4}

def test_update_with_on_empty_map():
    from pyrsistent import m
    m1 = m()
    m2 = m1.update_with(lambda l, r: l + r, m(a=1, b=2))
    assert m2 == {'a': 1, 'b': 2}

def test_update_with_returns_same_instance_if_no_changes():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: l, m())
    assert m2 is m1

def test_update_with_uses_update_fn_for_existing_keys_only():
    from pyrsistent import m
    m1 = m(a=1, b=2)
    m2 = m1.update_with(lambda l, r: 99, m(a=10))
    assert m2 == {'a': 99, 'b': 2}

def test_update_with_handles_non_pmap_mappings():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l + r, {'a': 2}, m(b=3))
    assert m2 == {'a': 3, 'b': 3}

def test_update_with_preserves_original_when_update_fn_does_not_change_value():
    from pyrsistent import m
    m1 = m(a=1)
    m2 = m1.update_with(lambda l, r: l, m(a=1))
    assert m2 is m1


# LLM-generated content at query #24
#--------------------------

def test___eq___same_instance():
    m1 = m(a=1, b=2)
    result = m1 == m1
    assert result is True

def test___eq___equal_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    result = m1 == m2
    assert result is True

def test___eq___different_pmaps():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    result = m1 == m2
    assert result is False

def test___eq___different_lengths():
    m1 = m(a=1, b=2)
    m2 = m(a=1)
    result = m1 == m2
    assert result is False

def test___eq___equal_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 2}
    result = m1 == d
    assert result is True

def test___eq___different_dict():
    m1 = m(a=1, b=2)
    d = {'a': 1, 'b': 3}
    result = m1 == d
    assert result is False

def test___eq___not_mapping():
    m1 = m(a=1, b=2)
    result = m1 == [1, 2]
    assert result is NotImplemented

def test___eq___cached_hash_mismatch():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=3)
    m1._cached_hash = hash(frozenset(m1.iteritems()))
    m2._cached_hash = hash(frozenset(m2.iteritems()))
    result = m1 == m2
    assert result is False

def test___eq___cached_hash_match_but_different_buckets():
    m1 = m(a=1, b=2)
    m2 = m(a=1, b=2)
    m1._cached_hash = 12345
    m2._cached_hash = 12345
    result = m1 == m2
    assert result is True

def test___eq___same_buckets():
    m1 = m(a=1, b=2)
    m2 = PMap(m1._size, m1._buckets)
    result = m1 == m2
    assert result is True


# LLM-generated content at query #25
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


# LLM-generated content at query #26
#--------------------------

def test_turbo_mapping_predicate_at_line_7_false():
    initial = [1, 2, 3]
    pre_size = None
    result = _turbo_mapping(initial, pre_size)
    assert isinstance(result, PMap)


# LLM-generated content at query #27
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    size = 2
    buckets = (('a', 1), ('b', 2))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_empty_pmap():
    size = 0
    buckets = ()
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_single_element():
    size = 1
    buckets = (('key', 'value'),)
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_multiple_elements():
    size = 3
    buckets = (('a', 1), ('b', 2), ('c', 3))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_none_values():
    size = 2
    buckets = (('a', None), ('b', None))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_complex_keys():
    size = 2
    buckets = ((('nested', 'key'), 'value'), (123, 456))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_duplicate_keys_in_buckets():
    size = 2
    buckets = (('a', 1), ('a', 2))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_empty_buckets():
    size = 0
    buckets = ()
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_large_size():
    size = 1000
    buckets = tuple(('key' + str(i), i) for i in range(size))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_mixed_types():
    size = 4
    buckets = (('string', 'value'), (123, 456), (3.14, 'pi'), (True, False))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets


# LLM-generated content at query #28
#--------------------------

def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    pmap = m(a=1)
    result = pmap.update_with(lambda l, r: l + r, m(b=2))
    assert result == {'a': 1, 'b': 2}


# LLM-generated content at query #29
#--------------------------

def test_contains_with_invalid_arg():
    from pyrsistent import pmap
    m = pmap({1: 'a', 2: 'b'})
    items_view = m.items()
    result = (1, 'a', 'extra') in items_view
    assert result == False


# LLM-generated content at query #30
#--------------------------

def test_predicate_at_line_7_evaluates_to_false():
    initial = []
    pre_size = None
    result = _turbo_mapping(initial, pre_size)
    assert result is not None


# LLM-generated content at query #31
#--------------------------

def test_constructor_creates_pmap_with_correct_size_and_buckets():
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_zero_size_and_empty_buckets():
    size = 0
    buckets = ()
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_none_buckets():
    size = 0
    buckets = None
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_list_buckets():
    size = 1
    buckets = [('key', 'value')]
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_pvector_buckets():
    from pyrsistent import pvector
    size = 1
    buckets = pvector([('key', 'value')])
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_large_size_and_buckets():
    size = 1000
    buckets = tuple(('key' + str(i), 'value' + str(i)) for i in range(size))
    pmap = PMap(size, buckets)
    assert pmap._size == size
    assert pmap._buckets == buckets

def test_constructor_creates_pmap_with_same_buckets_reference():
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap = PMap(size, buckets)
    assert pmap._buckets is buckets

def test_constructor_creates_pmap_without_cached_hash():
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap = PMap(size, buckets)
    assert not hasattr(pmap, '_cached_hash')

def test_constructor_creates_pmap_without_weakref():
    size = 2
    buckets = (('key1', 'value1'), ('key2', 'value2'))
    pmap = PMap(size, buckets)
    assert pmap.__weakref__ is None

def test_constructor_creates_pmap_with_slots():
    size = 0
    buckets = ()
    pmap = PMap(size, buckets)
    assert not hasattr(pmap, '__dict__')
    assert hasattr(pmap, '_size')
    assert hasattr(pmap, '_buckets')
    assert hasattr(pmap, '__weakref__')


# LLM-generated content at query #32
#--------------------------

def test_eq_with_dict_and_different_buckets():
    from pyrsistent import pmap
    m1 = pmap({'a': 1, 'b': 2})
    m2 = pmap({'a': 1, 'b': 2})
    result = m1 == dict(m2)
    assert result is True


# LLM-generated content at query #33
#--------------------------

def test_update_with_key_not_in_evolver():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(c=3)
    result = pmap1.update_with(lambda l, r: l + r, pmap2)
    assert result == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #34
#--------------------------

def test_constructor_creates_pmap_with_given_size_and_buckets():
    from pyrsistent import pmap
    size = 2
    buckets = (('a', 1), ('b', 2))
    m = pmap({'a': 1, 'b': 2})
    assert m._size == size
    assert len(m._buckets) > 0

def test_constructor_returns_pmap_instance():
    from pyrsistent import PMap
    size = 0
    buckets = ()
    m = PMap(size, buckets)
    assert isinstance(m, PMap)

def test_constructor_sets_size_attribute():
    from pyrsistent import PMap
    size = 5
    buckets = ()
    m = PMap(size, buckets)
    assert m._size == size

def test_constructor_sets_buckets_attribute():
    from pyrsistent import PMap
    size = 0
    buckets = (('key', 'value'),)
    m = PMap(size, buckets)
    assert m._buckets == buckets

def test_constructor_with_empty_buckets():
    from pyrsistent import PMap
    size = 0
    buckets = ()
    m = PMap(size, buckets)
    assert m._size == 0
    assert len(m._buckets) == 0

def test_constructor_with_non_empty_buckets():
    from pyrsistent import PMap
    size = 1
    buckets = (('test', 42),)
    m = PMap(size, buckets)
    assert m._size == size
    assert m._buckets == buckets

def test_constructor_size_matches_buckets_length():
    from pyrsistent import pmap
    m = pmap({'x': 10, 'y': 20})
    assert m._size == len(m)

def test_constructor_creates_hashable_instance():
    from pyrsistent import PMap
    size = 0
    buckets = ()
    m = PMap(size, buckets)
    assert hash(m) is not None

def test_constructor_preserves_weakref_support():
    import weakref

    from pyrsistent import PMap
    size = 0
    buckets = ()
    m = PMap(size, buckets)
    ref = weakref.ref(m)
    assert ref() is m

def test_constructor_with_custom_buckets_structure():
    from pyrsistent import PMap
    size = 2
    buckets = (('k1', 'v1'), ('k2', 'v2'))
    m = PMap(size, buckets)
    assert m._size == size
    assert m._buckets == buckets


# LLM-generated content at query #35
#--------------------------

def test_contains_with_invalid_arg():
    from pyrsistent import pmap
    m = pmap({'a': 1})
    items_view = m.items()
    result = ('a', 2) in items_view
    assert result == False


# LLM-generated content at query #36
#--------------------------

def test_eq_with_dict_and_different_buckets_but_same_items():
    from pyrsistent import m
    pmap1 = m(a=1, b=2)
    pmap2 = m(a=1, b=2)
    dict1 = {'a': 1, 'b': 2}
    result = pmap1 == dict1
    assert result is True
    result = pmap2 == dict1
    assert result is True


