####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    expected = pmap({})
    assert result == expected


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_nested_dict():
    result = freeze({'x': {'y': 5}})
    expected = pmap({'x': pmap({'y': 5})})
    assert result == expected


def test_freeze_empty_list():
    result = freeze([])
    expected = pvector([])
    assert result == expected


def test_freeze_list_with_elements():
    result = freeze([1, 2, 3])
    expected = pvector([1, 2, 3])
    assert result == expected


def test_freeze_list_with_dict():
    result = freeze([{'a': 1}])
    expected = pvector([pmap({'a': 1})])
    assert result == expected


def test_freeze_empty_tuple():
    result = freeze(())
    expected = ()
    assert result == expected


def test_freeze_tuple_with_elements():
    result = freeze((1, [2, 3]))
    expected = (1, pvector([2, 3]))
    assert result == expected


def test_freeze_empty_set():
    result = freeze(set())
    expected = pset()
    assert result == expected


def test_freeze_set_with_elements():
    result = freeze({1, 2, 3})
    expected = pset([1, 2, 3])
    assert result == expected


def test_freeze_defaultdict():
    dd = collections.defaultdict(list, {'a': [1, 2]})
    result = freeze(dd)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    expected = 42
    assert result == expected


def test_freeze_string():
    result = freeze('hello')
    expected = 'hello'
    assert result == expected


def test_freeze_strict_false_with_pmap():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=False)
    expected = pm
    assert result == expected


def test_freeze_strict_false_with_pvector():
    pv = pvector([1, 2, 3])
    result = freeze(pv, strict=False)
    expected = pv
    assert result == expected


def test_freeze_strict_true_with_pmap():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_strict_true_with_pvector():
    pv = pvector([1, [2, 3]])
    result = freeze(pv, strict=True)
    expected = pvector([1, pvector([2, 3])])
    assert result == expected


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_with_defaultdict_and_strict_true():
    from collections import defaultdict
    d = defaultdict(list, {'a': [1, 2]})
    result = freeze(d, strict=True)
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_freeze_with_defaultdict_and_strict_false():
    from collections import defaultdict
    d = defaultdict(list, {'a': [1, 2]})
    result = freeze(d, strict=False)
    assert isinstance(result, PMap)
    assert result['a'] == [1, 2]

def test_freeze_with_pmap_and_strict_true():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=True)
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_freeze_with_pmap_and_strict_false():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=False)
    assert result is m


# LLM-generated content at query #3
#--------------------------

def test_freeze_defaultdict_with_strict_true():
    from collections import defaultdict
    from pyrsistent import pmap, PVector, PMap, pvector, pset
    dd = defaultdict(list, {'a': [1, 2]})
    result = freeze(dd, strict=True)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


# LLM-generated content at query #4
#--------------------------

def test_mutant_with_list_argument():
    mutable_list = [1, 2, 3]
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst
    result = modify_list(mutable_list)
    assert result == pvector([1, 2, 3])
    assert mutable_list == [1, 2, 3]

def test_mutant_with_dict_argument():
    mutable_dict = {'a': 1, 'b': 2}
    @mutant
    def modify_dict(d):
        d['c'] = 3
        return d
    result = modify_dict(mutable_dict)
    assert result == pmap({'a': 1, 'b': 2})
    assert mutable_dict == {'a': 1, 'b': 2}

def test_mutant_with_set_argument():
    mutable_set = {1, 2, 3}
    @mutant
    def modify_set(s):
        s.add(4)
        return s
    result = modify_set(mutable_set)
    assert result == pset([1, 2, 3])
    assert mutable_set == {1, 2, 3}

def test_mutant_with_nested_structures():
    nested = {'list': [1, 2], 'dict': {'inner': [3, 4]}}
    @mutant
    def modify_nested(n):
        n['list'].append(5)
        n['dict']['inner'].append(6)
        return n
    result = modify_nested(nested)
    expected = pmap({'list': pvector([1, 2]), 'dict': pmap({'inner': pvector([3, 4])})})
    assert result == expected
    assert nested == {'list': [1, 2], 'dict': {'inner': [3, 4]}}

def test_mutant_with_keyword_arguments():
    @mutant
    def combine(a, b):
        return {'a': a, 'b': b}
    result = combine([1, 2], b={'x': 10})
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'x': 10})})
    assert result == expected

def test_mutant_with_positional_and_keyword_arguments():
    @mutant
    def func(x, y, z):
        x.append(y)
        z['key'] = 'new'
        return (x, y, z)
    result = func([1], 2, {'a': 'b'})
    expected = (pvector([1]), 2, pmap({'a': 'b'}))
    assert result == expected

def test_mutant_returns_frozen_result():
    @mutant
    def return_mutable():
        return [1, {'a': 2}]
    result = return_mutable()
    expected = pvector([1, pmap({'a': 2})])
    assert result == expected

def test_mutant_with_no_arguments():
    mutable_global = []
    @mutant
    def no_args():
        mutable_global.append(1)
        return mutable_global
    result = no_args()
    assert result == pvector([])
    assert mutable_global == []

def test_mutant_preserves_function_metadata():
    @mutant
    def original(a, b):
        """Original docstring."""
        return a + b
    assert original.__name__ == 'original'
    assert original.__doc__ == "Original docstring."

def test_mutant_with_tuple_argument():
    mutable_list = [1, 2]
    @mutant
    def modify_tuple(t):
        t[1].append(3)
        return t
    result = modify_tuple((5, mutable_list))
    expected = (5, pvector([1, 2]))
    assert result == expected
    assert mutable_list == [1, 2]


# LLM-generated content at query #5
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    assert result == pmap({})


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_list():
    result = freeze([1, {'x': 2}, [3, 4]])
    expected = pvector([1, pmap({'x': 2}), pvector([3, 4])])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, [2, 3], {'a': 4}))
    expected = (1, pvector([2, 3]), pmap({'a': 4}))
    assert result == expected


def test_freeze_set():
    result = freeze({1, 2, 3})
    assert result == pset({1, 2, 3})


def test_freeze_defaultdict():
    d = collections.defaultdict(list, {'a': [1, 2]})
    result = freeze(d)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_nested_mixed():
    data = {'list': [1, 2], 'tuple': (3, 4), 'set': {5, 6}}
    result = freeze(data)
    expected = pmap({
        'list': pvector([1, 2]),
        'tuple': (3, 4),
        'set': pset({5, 6})
    })
    assert result == expected


def test_freeze_strict_false():
    data = {'a': [1, 2]}
    result = freeze(data, strict=False)
    expected = pmap({'a': [1, 2]})
    assert result == expected


def test_freeze_already_frozen_strict():
    data = pmap({'a': pvector([1, 2])})
    result = freeze(data, strict=True)
    assert result == data


def test_freeze_already_frozen_not_strict():
    data = pmap({'a': pvector([1, 2])})
    result = freeze(data, strict=False)
    assert result == data


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    result = freeze("hello")
    assert result == "hello"


def test_freeze_none():
    result = freeze(None)
    assert result is None


# LLM-generated content at query #6
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    assert isinstance(result, PMap)
    assert len(result) == 0


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert isinstance(result['b'], PVector)
    assert result['b'][0] == 2
    assert result['b'][1] == 3


def test_freeze_list():
    result = freeze([1, {'x': 2}, [3, 4]])
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert isinstance(result[1], PMap)
    assert result[1]['x'] == 2
    assert isinstance(result[2], PVector)
    assert result[2][0] == 3
    assert result[2][1] == 4


def test_freeze_set():
    result = freeze({1, 2, 3})
    assert isinstance(result, PSet)
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_freeze_tuple():
    result = freeze((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)
    assert result[0] == 1
    assert isinstance(result[1], PVector)
    assert result[1][0] == 2
    assert result[1][1] == 3
    assert isinstance(result[2], PMap)
    assert result[2]['a'] == 4


def test_freeze_defaultdict():
    d = collections.defaultdict(list)
    d['a'].append(1)
    result = freeze(d)
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert result['a'][0] == 1


def test_freeze_strict_false_with_pmap():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=False)
    assert result is pm


def test_freeze_strict_false_with_pvector():
    pv = pvector([1, 2, 3])
    result = freeze(pv, strict=False)
    assert result is pv


def test_freeze_strict_true_with_pmap():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert result['a'][0] == 1
    assert result['a'][1] == 2


def test_freeze_strict_true_with_pvector():
    pv = pvector([1, [2, 3]])
    result = freeze(pv, strict=True)
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert isinstance(result[1], PVector)
    assert result[1][0] == 2
    assert result[1][1] == 3


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    result = freeze("hello")
    assert result == "hello"


def test_freeze_nested_dict():
    result = freeze({'a': {'b': {'c': [1, 2]}}})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PMap)
    assert isinstance(result['a']['b'], PMap)
    assert isinstance(result['a']['b']['c'], PVector)
    assert result['a']['b']['c'][0] == 1
    assert result['a']['b']['c'][1] == 2


def test_freeze_empty_list():
    result = freeze([])
    assert isinstance(result, PVector)
    assert len(result) == 0


def test_freeze_empty_set():
    result = freeze(set())
    assert isinstance(result, PSet)
    assert len(result) == 0


def test_freeze_empty_tuple():
    result = freeze(())
    assert isinstance(result, tuple)
    assert len(result) == 0


# LLM-generated content at query #7
#--------------------------

def test_mutant_decorator_freezes_inputs_and_output():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]

def test_mutant_decorator_with_dict():
    @mutant
    def update_dict(d, key, val):
        d[key] = val
        return d
    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_decorator_with_set():
    @mutant
    def add_to_set(s, element):
        s.add(element)
        return s
    original_set = {1, 2}
    result = add_to_set(original_set, 3)
    assert original_set == {1, 2}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}

def test_mutant_decorator_with_tuple():
    @mutant
    def modify_tuple(t):
        return t + (4,)
    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple)
    assert original_tuple == (1, 2, 3)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)

def test_mutant_decorator_with_keyword_arguments():
    @mutant
    def combine_dicts(d1, d2):
        d1.update(d2)
        return d1
    dict1 = {'x': 10}
    dict2 = {'y': 20}
    result = combine_dicts(d1=dict1, d2=dict2)
    assert dict1 == {'x': 10}
    assert dict2 == {'y': 20}
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 10, 'y': 20}

def test_mutant_decorator_preserves_function_metadata():
    @mutant
    def sample_func(a, b):
        """Sample docstring."""
        return a + b
    assert sample_func.__name__ == 'sample_func'
    assert sample_func.__doc__ == 'Sample docstring.'

def test_mutant_decorator_with_nested_structures():
    @mutant
    def process_data(data):
        data['list'].append(99)
        return data
    original = {'list': [1, 2], 'set': {3, 4}}
    result = process_data(original)
    assert original == {'list': [1, 2], 'set': {3, 4}}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert list(result['list']) == [1, 2, 99]
    assert isinstance(result['set'], PSet)
    assert set(result['set']) == {3, 4}

def test_mutant_decorator_with_no_mutation():
    @mutant
    def pure_function(x):
        return x * 2
    original = 5
    result = pure_function(original)
    assert original == 5
    assert result == 10


# LLM-generated content at query #8
#--------------------------

```python
def test_freeze_with_defaultdict_and_strict_true():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    dd = defaultdict(list, {'a': [1, 2], 'b': [3, 4]})
    result = freeze(dd, strict=True)
    expected = pmap({'a': [1, 2], 'b': [3, 4]})
    assert result == expected
    assert isinstance(result, type(pmap()))


# LLM-generated content at query #9
#--------------------------

```python
def test_freeze_with_non_dict_non_pmap_and_strict_true():
    result = freeze([1, 2, 3], strict=True)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant_decorator_does_not_mutate_inputs():
    original = [1, 2, 3]
    frozen_copy = freeze(original)
    result = mutant(lambda x: x.append(4))(original)
    assert original == [1, 2, 3]
    assert result == [1, 2, 3, 4]
    assert result is not original


# LLM-generated content at query #11
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return_value():
    @mutant
    def add_one_to_list(lst):
        lst.append(1)
        return lst
    original_list = [5, 6]
    result = add_one_to_list(original_list)
    assert result == pvector([5, 6, 1])
    assert original_list == [5, 6]
    assert isinstance(result, PVector)

def test_mutant_decorator_freezes_keyword_arguments():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    original_dict = {'a': 1}
    result = update_dict(original_dict, key='b', value=2)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}
    assert isinstance(result, PMap)

def test_mutant_decorator_handles_mixed_arguments():
    @mutant
    def combine(set_arg, list_arg, dict_arg):
        return (set_arg, list_arg, dict_arg)
    result = combine({1, 2}, [3, 4], {'x': 5})
    assert result == (pset([1, 2]), pvector([3, 4]), pmap({'x': 5}))
    assert isinstance(result[0], PSet)
    assert isinstance(result[1], PVector)
    assert isinstance(result[2], PMap)

def test_mutant_decorator_preserves_function_metadata():
    @mutant
    def sample_func(x, y=2):
        """Sample docstring."""
        return x + y
    assert sample_func.__name__ == 'sample_func'
    assert sample_func.__doc__ == 'Sample docstring.'

def test_mutant_decorator_with_nested_structures():
    @mutant
    def process(data):
        data['list'][0] = 99
        return data
    original = {'list': [1, 2], 'set': {3, 4}}
    result = process(original)
    expected = pmap({'list': pvector([99, 2]), 'set': pset([3, 4])})
    assert result == expected
    assert original == {'list': [1, 2], 'set': {3, 4}}

def test_mutant_decorator_with_no_arguments():
    @mutant
    def constant():
        return [1, 2, 3]
    result = constant()
    assert result == pvector([1, 2, 3])
    assert isinstance(result, PVector)

def test_mutant_decorator_freezes_returned_tuple():
    @mutant
    def return_tuple():
        return ([1, 2], {'a': 3})
    result = return_tuple()
    assert result == (pvector([1, 2]), pmap({'a': 3}))
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)

def test_mutant_decorator_with_already_frozen_arguments():
    @mutant
    def identity(x):
        return x
    frozen_arg = pvector([1, 2])
    result = identity(frozen_arg)
    assert result == pvector([1, 2])
    assert result is frozen_arg


# LLM-generated content at query #12
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    expected = pmap({})
    assert result == expected


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_list():
    result = freeze([1, {'x': 2}, 3])
    expected = pvector([1, pmap({'x': 2}), 3])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, [2, 3]))
    expected = (1, pvector([2, 3]))
    assert result == expected


def test_freeze_set():
    result = freeze({1, 2, 3})
    expected = pset([1, 2, 3])
    assert result == expected


def test_freeze_nested_structures():
    result = freeze({'a': [1, {2, 3}], 'b': (4, [5])})
    expected = pmap({'a': pvector([1, pset([2, 3])]), 'b': (4, pvector([5]))})
    assert result == expected


def test_freeze_with_strict_false():
    result = freeze([pmap({'x': 1})], strict=False)
    expected = pvector([pmap({'x': 1})])
    assert result == expected


def test_freeze_with_strict_true():
    result = freeze([pmap({'x': [1]})], strict=True)
    expected = pvector([pmap({'x': pvector([1])})])
    assert result == expected


def test_freeze_defaultdict():
    d = collections.defaultdict(list)
    d['a'].append(1)
    result = freeze(d)
    expected = pmap({'a': pvector([1])})
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    expected = 42
    assert result == expected


def test_freeze_string():
    result = freeze("hello")
    expected = "hello"
    assert result == expected


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def modify_data(data, extra):
        data['key'] = 'new_value'
        extra.add(42)
        return {'result': data, 'extra': extra}

    initial_map = m(a=1, b=2)
    initial_set = s(1, 2, 3)
    result = modify_data(initial_map, initial_set)

    assert initial_map == m(a=1, b=2)
    assert initial_set == s(1, 2, 3)
    assert result['result'] == m(a=1, b=2, key='new_value')
    assert result['extra'] == s(1, 2, 3, 42)
    assert isinstance(result, type(freeze({})))
    assert isinstance(result['result'], type(freeze({})))
    assert isinstance(result['extra'], type(freeze(set())))


# LLM-generated content at query #14
#--------------------------

```python
def test_freeze_pmap_strict_true():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


# LLM-generated content at query #15
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def modify_map(pmap_arg):
        pmap_arg['new_key'] = 100
        return pmap_arg

    @mutant
    def modify_set(pset_arg):
        pset_arg.add(999)
        return pset_arg

    original_map = m(a=1, b=2)
    original_set = s(1, 2, 3)

    result_map = modify_map(original_map)
    result_set = modify_set(original_set)

    assert original_map == m(a=1, b=2)
    assert original_set == s(1, 2, 3)
    assert result_map == m(a=1, b=2, new_key=100)
    assert result_set == s(1, 2, 3, 999)


# LLM-generated content at query #16
#--------------------------

```python
def test_freeze_with_defaultdict_and_strict_true():
    from collections import defaultdict
    d = defaultdict(list, {'a': [1, 2]})
    result = freeze(d, strict=True)
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_freeze_with_defaultdict_and_strict_false():
    from collections import defaultdict
    d = defaultdict(list, {'a': [1, 2]})
    result = freeze(d, strict=False)
    assert isinstance(result, PMap)
    assert result['a'] == [1, 2]

def test_freeze_with_pmap_and_strict_true():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=True)
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_freeze_with_pmap_and_strict_false():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=False)
    assert result is m


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def modify_map(pmap_arg):
        pmap_arg['new_key'] = 'new_value'
        return pmap_arg

    @mutant
    def modify_set(pset_arg):
        pset_arg.add('new_element')
        return pset_arg

    original_map = m(a=1, b=2)
    original_set = s(1, 2, 3)

    result_map = modify_map(original_map)
    result_set = modify_set(original_set)

    assert original_map == m(a=1, b=2)
    assert original_set == s(1, 2, 3)
    assert result_map == m(a=1, b=2, new_key='new_value')
    assert result_set == s(1, 2, 3, 'new_element')
    assert isinstance(result_map, type(freeze({})))
    assert isinstance(result_set, type(freeze(set())))


# LLM-generated content at query #18
#--------------------------

def test_mutant_decorator_freezes_inputs_and_output():
    def add_to_list(lst, value):
        lst.append(value)
        return lst
    decorated = mutant(add_to_list)
    original_list = [1, 2]
    result = decorated(original_list, 3)
    assert original_list == [1, 2]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]

def test_mutant_decorator_with_dict():
    def update_dict(d, key, val):
        d[key] = val
        return d
    decorated = mutant(update_dict)
    original_dict = {'a': 1}
    result = decorated(original_dict, 'b', 2)
    assert original_dict == {'a': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_decorator_with_set():
    def add_to_set(s, element):
        s.add(element)
        return s
    decorated = mutant(add_to_set)
    original_set = {1, 2}
    result = decorated(original_set, 3)
    assert original_set == {1, 2}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}

def test_mutant_decorator_with_keyword_arguments():
    def func(x, y=0):
        return [x, y]
    decorated = mutant(func)
    result = decorated(5, y=10)
    assert isinstance(result, PVector)
    assert list(result) == [5, 10]

def test_mutant_decorator_preserves_function_metadata():
    def example():
        """Example docstring."""
        pass
    decorated = mutant(example)
    assert decorated.__name__ == 'example'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_nested_structures():
    def modify(data):
        data['list'][0] = 99
        return data
    decorated = mutant(modify)
    original = {'list': [1, 2, 3]}
    result = decorated(original)
    assert original == {'list': [1, 2, 3]}
    assert isinstance(result, PMap)
    inner_list = result['list']
    assert isinstance(inner_list, PVector)
    assert list(inner_list) == [99, 2, 3]

def test_mutant_decorator_with_tuple():
    def extend_tuple(t, item):
        return t + (item,)
    decorated = mutant(extend_tuple)
    original = (1, 2)
    result = decorated(original, 3)
    assert original == (1, 2)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)
    assert isinstance(result[0], int)

def test_mutant_decorator_with_strict_false_implicitly():
    def identity(x):
        return x
    decorated = mutant(identity)
    pvec = pvector([1, 2])
    result = decorated(pvec)
    assert result is pvec

def test_mutant_decorator_with_empty_inputs():
    def empty():
        return {}
    decorated = mutant(empty)
    result = decorated()
    assert isinstance(result, PMap)
    assert dict(result) == {}

def test_mutant_decorator_with_mixed_arguments():
    def mixed(a, b, c=0):
        return {'a': a, 'b': b, 'c': c}
    decorated = mutant(mixed)
    result = decorated([1], {'x': 2}, c=3)
    assert isinstance(result, PMap)
    a_val = result['a']
    b_val = result['b']
    c_val = result['c']
    assert isinstance(a_val, PVector) and list(a_val) == [1]
    assert isinstance(b_val, PMap) and dict(b_val) == {'x': 2}
    assert c_val == 3


# LLM-generated content at query #19
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    assert result == pmap({})


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_list():
    result = freeze([1, {'x': 2}, 3])
    expected = pvector([1, pmap({'x': 2}), 3])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, [2, 3], {'a': 4}))
    expected = (1, pvector([2, 3]), pmap({'a': 4}))
    assert result == expected


def test_freeze_set():
    result = freeze({1, 2, 3})
    expected = pset({1, 2, 3})
    assert result == expected


def test_freeze_nested_structure():
    result = freeze({'a': [1, 2], 'b': {'c': {3, 4}}})
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'c': pset({3, 4})})})
    assert result == expected


def test_freeze_already_frozen_pmap_strict():
    frozen_map = pmap({'x': [1, 2]})
    result = freeze(frozen_map, strict=True)
    expected = pmap({'x': pvector([1, 2])})
    assert result == expected


def test_freeze_already_frozen_pvector_strict():
    frozen_vec = pvector([{'a': 1}])
    result = freeze(frozen_vec, strict=True)
    expected = pvector([pmap({'a': 1})])
    assert result == expected


def test_freeze_already_frozen_pmap_non_strict():
    frozen_map = pmap({'x': [1, 2]})
    result = freeze(frozen_map, strict=False)
    assert result is frozen_map


def test_freeze_already_frozen_pvector_non_strict():
    frozen_vec = pvector([{'a': 1}])
    result = freeze(frozen_vec, strict=False)
    assert result is frozen_vec


def test_freeze_defaultdict():
    dd = collections.defaultdict(list, {'a': [1, 2]})
    result = freeze(dd)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    result = freeze("hello")
    assert result == "hello"


def test_freeze_none():
    result = freeze(None)
    assert result is None


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def modify_data(data_dict, data_set):
        data_dict['new_key'] = 'new_value'
        data_set.add('new_element')
        return {'modified': True, 'dict': data_dict, 'set': data_set}

    original_dict = m(a=1, b=2)
    original_set = s(1, 2, 3)
    result = modify_data(original_dict, original_set)
    assert original_dict == m(a=1, b=2)
    assert original_set == s(1, 2, 3)
    assert isinstance(result, type(freeze({})))
    assert result['dict'] == m(a=1, b=2, new_key='new_value')
    assert result['set'] == s(1, 2, 3, 'new_element')


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def modify_map(pmap_arg):
        pmap_arg['new_key'] = 'new_value'
        return pmap_arg

    @mutant
    def modify_set(pset_arg):
        pset_arg.add('new_element')
        return pset_arg

    original_map = m(a=1, b=2)
    original_set = s(1, 2, 3)

    result_map = modify_map(original_map)
    result_set = modify_set(original_set)

    assert original_map == m(a=1, b=2)
    assert original_set == s(1, 2, 3)
    assert result_map == m(a=1, b=2, new_key='new_value')
    assert result_set == s(1, 2, 3, 'new_element')
    assert isinstance(result_map, type(freeze({})))
    assert isinstance(result_set, type(freeze(set())))


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, pset, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def modify_set(s):
        s.add(4)
        return s

    @mutant
    def modify_map(m):
        m['new_key'] = 'new_value'
        return m

    original_set = pset([1, 2, 3])
    original_map = pmap({'a': 1, 'b': 2})

    result_set = modify_set(original_set)
    result_map = modify_map(original_map)

    assert original_set == pset([1, 2, 3])
    assert original_map == pmap({'a': 1, 'b': 2})
    assert result_set == pset([1, 2, 3, 4])
    assert result_map == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert isinstance(result_set, type(freeze(pset())))
    assert isinstance(result_map, type(freeze(pmap())))


# LLM-generated content at query #23
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    def mutable_func(lst, dct):
        lst.append(4)
        dct['new'] = 5
        return [lst, dct]
    decorated = mutant(mutable_func)
    original_list = [1, 2, 3]
    original_dict = {'a': 1}
    result = decorated(original_list, original_dict)
    assert original_list == [1, 2, 3]
    assert original_dict == {'a': 1}
    assert isinstance(result, PVector)
    assert len(result) == 2
    assert isinstance(result[0], PVector)
    assert result[0] == pvector([1, 2, 3, 4])
    assert isinstance(result[1], PMap)
    assert result[1] == pmap({'a': 1, 'new': 5})

def test_mutant_decorator_with_kwargs():
    def mutable_func(x, y=0):
        x.append(y)
        return x
    decorated = mutant(mutable_func)
    original = [1]
    result = decorated(original, y=2)
    assert original == [1]
    assert isinstance(result, PVector)
    assert result == pvector([1, 2])

def test_mutant_decorator_preserves_function_metadata():
    def example(a, b):
        """Example docstring."""
        return a + b
    decorated = mutant(example)
    assert decorated.__name__ == 'example'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_freezes_nested_structures():
    def func(data):
        data['list'][0] = 99
        return data
    decorated = mutant(func)
    original = {'list': [1, 2, 3], 'tuple': (4, 5)}
    result = decorated(original)
    assert original == {'list': [1, 2, 3], 'tuple': (4, 5)}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert result['list'] == pvector([99, 2, 3])
    assert isinstance(result['tuple'], tuple)
    assert result['tuple'] == (4, 5)

def test_mutant_decorator_with_empty_arguments():
    def func():
        return {'empty': []}
    decorated = mutant(func)
    result = decorated()
    assert isinstance(result, PMap)
    assert isinstance(result['empty'], PVector)
    assert result['empty'] == pvector([])

def test_mutant_decorator_freezes_set():
    def func(s):
        s.add(4)
        return s
    decorated = mutant(func)
    original = {1, 2, 3}
    result = decorated(original)
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})

def test_mutant_decorator_with_strict_false_implicitly():
    def func(d):
        d['inner'] = [1, 2]
        return d
    decorated = mutant(func)
    original = {}
    result = decorated(original)
    assert original == {}
    assert isinstance(result, PMap)
    assert isinstance(result['inner'], PVector)
    assert result['inner'] == pvector([1, 2])


# LLM-generated content at query #24
#--------------------------

```python
def test_freeze_with_pmap_and_strict_true():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert isinstance(result, type(pm))
    assert result['a'][0] == 1
    assert result['a'][1] == 2


# LLM-generated content at query #25
#--------------------------

```python
def test_freeze_pmap_strict_true():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert isinstance(result, type(pm))
    assert result['a'] != [1, 2]
    assert isinstance(result['a'], type(pvector()))
    assert list(result['a']) == [1, 2]


# LLM-generated content at query #26
#--------------------------

def test_mutant_with_positional_args():
    result = mutant(lambda x, y: [x, y])(1, 2)
    assert result == pvector([1, 2])


def test_mutant_with_keyword_args():
    result = mutant(lambda a, b: {a: b})(1, 2)
    assert result == pmap({1: 2})


def test_mutant_with_mutable_input():
    result = mutant(lambda lst: lst.append(4))([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])


def test_mutant_with_nested_mutable_input():
    result = mutant(lambda d: d.update({"c": 3}))({"a": 1, "b": 2})
    assert result == pmap({"a": 1, "b": 2, "c": 3})


def test_mutant_with_set_input():
    result = mutant(lambda s: s.union({4}))({1, 2, 3})
    assert result == pset({1, 2, 3, 4})


def test_mutant_with_tuple_input():
    result = mutant(lambda t: t + (4,))((1, 2, 3))
    assert result == (1, 2, 3, 4)


def test_mutant_with_mixed_args():
    result = mutant(lambda x, y, z: {"x": x, "y": y, "z": z})(1, 2, 3)
    assert result == pmap({"x": 1, "y": 2, "z": 3})


def test_mutant_with_no_args():
    result = mutant(lambda: [1, 2, 3])()
    assert result == pvector([1, 2, 3])


def test_mutant_with_strict_freeze():
    result = mutant(lambda v: v.append(4))(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])


def test_mutant_with_dict_keys_not_frozen():
    key = [1, 2]
    try:
        hash(key)
        assert False
    except TypeError:
        pass
    result = mutant(lambda d: d)({key: "value"})
    assert isinstance(result, PMap)
    assert list(result.keys())[0] == key


# LLM-generated content at query #27
#--------------------------

def test_mutant_decorator_freezes_inputs_and_output():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]

def test_mutant_decorator_with_dict():
    @mutant
    def update_dict(d, key, val):
        d[key] = val
        return d
    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_decorator_with_set():
    @mutant
    def add_to_set(s, element):
        s.add(element)
        return s
    original_set = {1, 2}
    result = add_to_set(original_set, 3)
    assert original_set == {1, 2}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}

def test_mutant_decorator_with_tuple():
    @mutant
    def modify_tuple(t, new_element):
        return t + (new_element,)
    original_tuple = (1, 2)
    result = modify_tuple(original_tuple, 3)
    assert original_tuple == (1, 2)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)

def test_mutant_decorator_with_keyword_arguments():
    @mutant
    def combine_dicts(d1, d2):
        d1.update(d2)
        return d1
    dict1 = {'x': 10}
    dict2 = {'y': 20}
    result = combine_dicts(d1=dict1, d2=dict2)
    assert dict1 == {'x': 10}
    assert dict2 == {'y': 20}
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 10, 'y': 20}

def test_mutant_decorator_with_mixed_arguments():
    @mutant
    def mixed_operation(lst, d, key, value):
        lst.append(value)
        d[key] = lst
        return d
    lst_arg = [1, 2]
    dict_arg = {'a': [3, 4]}
    result = mixed_operation(lst_arg, dict_arg, 'b', 5)
    assert lst_arg == [1, 2]
    assert dict_arg == {'a': [3, 4]}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': pvector([3, 4]), 'b': pvector([1, 2, 5])}

def test_mutant_decorator_preserves_function_metadata():
    @mutant
    def sample_func(x):
        """Sample docstring."""
        return x
    assert sample_func.__name__ == 'sample_func'
    assert sample_func.__doc__ == 'Sample docstring.'

def test_mutant_decorator_with_nested_structures():
    @mutant
    def process_data(data):
        data['list'][0] = 100
        data['set'].add(200)
        return data
    original = {'list': [1, 2], 'set': {3, 4}}
    result = process_data(original)
    assert original == {'list': [1, 2], 'set': {3, 4}}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)
    assert list(result['list']) == [100, 2]
    assert set(result['set']) == {3, 4, 200}

def test_mutant_decorator_with_no_mutation():
    @mutant
    def pure_function(a, b):
        return a + b
    result = pure_function(10, 20)
    assert result == 30

def test_mutant_decorator_with_strict_false_implicitly():
    @mutant
    def modify_pvector(pv):
        return pv.append(99)
    pv_input = pvector([1, 2])
    result = modify_pvector(pv_input)
    assert pv_input == pvector([1, 2])
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 99]


# LLM-generated content at query #28
#--------------------------

```python
def test_freeze_with_non_dict_non_pmap_and_strict_true():
    result = freeze([1, 2, 3], strict=True)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]


# LLM-generated content at query #29
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    expected = pmap({})
    assert result == expected


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_empty_list():
    result = freeze([])
    expected = pvector([])
    assert result == expected


def test_freeze_list_with_elements():
    result = freeze([1, {'x': 2}, [3, 4]])
    expected = pvector([1, pmap({'x': 2}), pvector([3, 4])])
    assert result == expected


def test_freeze_empty_set():
    result = freeze(set())
    expected = pset()
    assert result == expected


def test_freeze_set_with_elements():
    result = freeze({1, 2, 3})
    expected = pset([1, 2, 3])
    assert result == expected


def test_freeze_empty_tuple():
    result = freeze(())
    expected = ()
    assert result == expected


def test_freeze_tuple_with_elements():
    result = freeze((1, [2, 3], {'a': 4}))
    expected = (1, pvector([2, 3]), pmap({'a': 4}))
    assert result == expected


def test_freeze_defaultdict():
    d = collections.defaultdict(list, {'x': [1, 2]})
    result = freeze(d)
    expected = pmap({'x': pvector([1, 2])})
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    expected = 42
    assert result == expected


def test_freeze_string():
    result = freeze('hello')
    expected = 'hello'
    assert result == expected


def test_freeze_nested_structure():
    data = {'a': [1, 2, {'b': {3, 4}}], 'c': (5, [6])}
    result = freeze(data)
    expected = pmap({'a': pvector([1, 2, pmap({'b': pset([3, 4])})]), 'c': (5, pvector([6]))})
    assert result == expected


def test_freeze_strict_false_with_pmap():
    pm = pmap({'x': [1, 2]})
    result = freeze(pm, strict=False)
    expected = pmap({'x': [1, 2]})
    assert result == expected


def test_freeze_strict_false_with_pvector():
    pv = pvector([1, {'a': 2}])
    result = freeze(pv, strict=False)
    expected = pvector([1, {'a': 2}])
    assert result == expected


def test_freeze_strict_true_with_pmap():
    pm = pmap({'x': [1, 2]})
    result = freeze(pm, strict=True)
    expected = pmap({'x': pvector([1, 2])})
    assert result == expected


def test_freeze_strict_true_with_pvector():
    pv = pvector([1, {'a': 2}])
    result = freeze(pv, strict=True)
    expected = pvector([1, pmap({'a': 2})])
    assert result == expected


# LLM-generated content at query #30
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    expected = pmap({})
    assert result == expected


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_list():
    result = freeze([1, {'x': 2}, [3, 4]])
    expected = pvector([1, pmap({'x': 2}), pvector([3, 4])])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, [2, 3], {'a': 4}))
    expected = (1, pvector([2, 3]), pmap({'a': 4}))
    assert result == expected


def test_freeze_set():
    result = freeze({1, 2, 3})
    expected = pset([1, 2, 3])
    assert result == expected


def test_freeze_defaultdict():
    dd = collections.defaultdict(list, {'a': [1, 2]})
    result = freeze(dd)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_strict_false():
    result = freeze([pmap({'x': 1})], strict=False)
    expected = pvector([pmap({'x': 1})])
    assert result == expected


def test_freeze_strict_true():
    result = freeze([pmap({'x': [1]})], strict=True)
    expected = pvector([pmap({'x': pvector([1])})])
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    expected = 42
    assert result == expected


def test_freeze_nested_dict():
    result = freeze({'a': {'b': {'c': 1}}})
    expected = pmap({'a': pmap({'b': pmap({'c': 1})})})
    assert result == expected


# LLM-generated content at query #31
#--------------------------

def test_freeze_with_pmap_and_strict_false():
    from pyrsistent import pmap, freeze
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=False)
    assert result is m


# LLM-generated content at query #32
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    expected = pmap({})
    assert result == expected


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_nested_dict():
    result = freeze({'x': {'y': 5}})
    expected = pmap({'x': pmap({'y': 5})})
    assert result == expected


def test_freeze_empty_list():
    result = freeze([])
    expected = pvector([])
    assert result == expected


def test_freeze_list_with_elements():
    result = freeze([1, 2, 3])
    expected = pvector([1, 2, 3])
    assert result == expected


def test_freeze_list_with_dict():
    result = freeze([{'a': 1}])
    expected = pvector([pmap({'a': 1})])
    assert result == expected


def test_freeze_empty_set():
    result = freeze(set())
    expected = pset()
    assert result == expected


def test_freeze_set_with_elements():
    result = freeze({1, 2, 3})
    expected = pset([1, 2, 3])
    assert result == expected


def test_freeze_empty_tuple():
    result = freeze(())
    expected = ()
    assert result == expected


def test_freeze_tuple_with_elements():
    result = freeze((1, [2], {'a': 3}))
    expected = (1, pvector([2]), pmap({'a': 3}))
    assert result == expected


def test_freeze_defaultdict():
    d = collections.defaultdict(list)
    d['x'].append(1)
    result = freeze(d)
    expected = pmap({'x': pvector([1])})
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    expected = 42
    assert result == expected


def test_freeze_string():
    result = freeze("hello")
    expected = "hello"
    assert result == expected


def test_freeze_strict_false_with_pmap():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=False)
    expected = m
    assert result == expected


def test_freeze_strict_false_with_pvector():
    v = pvector([1, 2, 3])
    result = freeze(v, strict=False)
    expected = v
    assert result == expected


def test_freeze_strict_true_with_pmap():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=True)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_strict_true_with_pvector():
    v = pvector([1, [2, 3]])
    result = freeze(v, strict=True)
    expected = pvector([1, pvector([2, 3])])
    assert result == expected


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def modify_map(pmap_arg):
        pmap_arg['new_key'] = 'new_value'
        return pmap_arg

    @mutant
    def modify_set(pset_arg):
        pset_arg.add('new_element')
        return pset_arg

    original_map = m(a=1, b=2)
    original_set = s(1, 2, 3)

    result_map = modify_map(original_map)
    result_set = modify_set(original_set)

    assert original_map == m(a=1, b=2)
    assert original_set == s(1, 2, 3)
    assert result_map == m(a=1, b=2, new_key='new_value')
    assert result_set == s(1, 2, 3, 'new_element')
    assert isinstance(result_map, type(freeze({})))
    assert isinstance(result_set, type(freeze(set())))


# LLM-generated content at query #34
#--------------------------

def test_freeze_defaultdict_with_strict_true():
    from collections import defaultdict
    from pyrsistent import freeze, PMap
    dd = defaultdict(list, {'a': [1, 2]})
    result = freeze(dd, strict=True)
    assert isinstance(result, PMap)
    assert result['a'] == [1, 2]


# LLM-generated content at query #35
#--------------------------

```python
def test_freeze_dict_with_strict_true():
    from pyrsistent import freeze, pmap
    d = {'a': [1, 2], 'b': {'c': 3}}
    result = freeze(d, strict=True)
    assert isinstance(result, type(pmap()))
    assert result['a'][0] == 1
    assert result['a'][1] == 2
    assert isinstance(result['b'], type(pmap()))
    assert result['b']['c'] == 3

def test_freeze_pmap_with_strict_true():
    from pyrsistent import freeze, pmap
    p = pmap({'x': [4, 5], 'y': {'z': 6}})
    result = freeze(p, strict=True)
    assert isinstance(result, type(pmap()))
    assert result['x'][0] == 4
    assert result['x'][1] == 5
    assert isinstance(result['y'], type(pmap()))
    assert result['y']['z'] == 6

def test_freeze_dict_with_strict_false():
    from pyrsistent import freeze, pmap
    d = {'a': [1, 2], 'b': {'c': 3}}
    result = freeze(d, strict=False)
    assert isinstance(result, type(pmap()))
    assert result['a'][0] == 1
    assert result['a'][1] == 2
    assert isinstance(result['b'], dict)
    assert result['b']['c'] == 3

def test_freeze_pmap_with_strict_false():
    from pyrsistent import freeze, pmap
    p = pmap({'x': [4, 5], 'y': {'z': 6}})
    result = freeze(p, strict=False)
    assert isinstance(result, type(pmap()))
    assert result['x'][0] == 4
    assert result['x'][1] == 5
    assert isinstance(result['y'], type(pmap()))
    assert result['y']['z'] == 6


# LLM-generated content at query #36
#--------------------------

```python
def test_freeze_with_strict_and_pmap():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': [1, 2]})
    result = freeze(pmap_instance, strict=True)
    assert isinstance(result, type(pmap_instance))
    assert result['a'] == freeze([1, 2], strict=True)


# LLM-generated content at query #37
#--------------------------

def test_mutant_with_positional_arguments():
    def add_one(x):
        x[0] = x[0] + 1
        return x
    decorated = mutant(add_one)
    original = [1, 2, 3]
    result = decorated(original)
    assert original == [1, 2, 3]
    assert result == pvector([2, 2, 3])

def test_mutant_with_keyword_arguments():
    def update_dict(d, key, value):
        d[key] = value
        return d
    decorated = mutant(update_dict)
    original = {'a': 1, 'b': 2}
    result = decorated(original, 'c', 3)
    assert original == {'a': 1, 'b': 2}
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_with_mixed_arguments():
    def modify_list_and_dict(lst, d, factor):
        lst.append(factor)
        d['factor'] = factor
        return lst, d
    decorated = mutant(modify_list_and_dict)
    lst = [1, 2]
    d = {'x': 10}
    result_lst, result_d = decorated(lst, d, 5)
    assert lst == [1, 2]
    assert d == {'x': 10}
    assert result_lst == pvector([1, 2, 5])
    assert result_d == pmap({'x': 10, 'factor': 5})

def test_mutant_return_value_frozen():
    def return_mutable():
        return [1, {'a': 2}]
    decorated = mutant(return_mutable)
    result = decorated()
    assert result == pvector([1, pmap({'a': 2})])

def test_mutant_with_nested_mutables():
    def swap_first_and_key(d, lst):
        d['first'], lst[0] = lst[0], d['first']
        return d, lst
    decorated = mutant(swap_first_and_key)
    d = {'first': 'a', 'second': 'b'}
    lst = [1, 2, 3]
    result_d, result_lst = decorated(d, lst)
    assert d == {'first': 'a', 'second': 'b'}
    assert lst == [1, 2, 3]
    assert result_d == pmap({'first': 1, 'second': 'b'})
    assert result_lst == pvector(['a', 2, 3])

def test_mutant_no_arguments():
    def constant():
        return {'answer': 42}
    decorated = mutant(constant)
    result = decorated()
    assert result == pmap({'answer': 42})

def test_mutant_preserves_function_metadata():
    def example():
        """Example docstring."""
        pass
    decorated = mutant(example)
    assert decorated.__name__ == 'example'
    assert decorated.__doc__ == 'Example docstring.'


# LLM-generated content at query #38
#--------------------------

```python
def test_freeze_with_strict_and_pmap():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': 1, 'b': 2})
    result = freeze(pmap_instance, strict=True)
    assert isinstance(result, type(pmap_instance))
    assert result == pmap_instance


# LLM-generated content at query #39
#--------------------------

```python
def test_freeze_pmap_strict_true():
    from pyrsistent import freeze, pmap
    result = freeze(pmap({'a': [1, 2]}))
    assert isinstance(result, type(pmap()))
    assert result['a'][0] == 1
    assert result['a'][1] == 2

def test_freeze_pvector_strict_true():
    from pyrsistent import freeze, pvector
    result = freeze(pvector([{'b': 3}]))
    assert isinstance(result, type(pvector()))
    assert isinstance(result[0], type(pmap()))
    assert result[0]['b'] == 3


# LLM-generated content at query #40
#--------------------------

```python
def test_freeze_with_pmap_and_strict_true():
    from pyrsistent import pmap, freeze
    pmap_obj = pmap({'a': 1})
    result = freeze(pmap_obj, strict=True)
    assert result == pmap({'a': 1})

def test_freeze_with_pmap_and_strict_false():
    from pyrsistent import pmap, freeze
    pmap_obj = pmap({'a': 1})
    result = freeze(pmap_obj, strict=False)
    assert result == pmap_obj

def test_freeze_with_pvector_and_strict_true():
    from pyrsistent import pvector, freeze
    pvec_obj = pvector([1, 2])
    result = freeze(pvec_obj, strict=True)
    assert result == pvector([1, 2])

def test_freeze_with_pvector_and_strict_false():
    from pyrsistent import pvector, freeze
    pvec_obj = pvector([1, 2])
    result = freeze(pvec_obj, strict=False)
    assert result == pvec_obj

def test_freeze_with_dict_and_strict_true():
    from pyrsistent import pmap, freeze
    dict_obj = {'a': 1}
    result = freeze(dict_obj, strict=True)
    assert result == pmap({'a': 1})

def test_freeze_with_defaultdict_and_strict_true():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    def_dict = defaultdict(int, {'a': 1})
    result = freeze(def_dict, strict=True)
    assert result == pmap({'a': 1})

def test_freeze_with_list_and_strict_true():
    from pyrsistent import pvector, freeze
    list_obj = [1, 2]
    result = freeze(list_obj, strict=True)
    assert result == pvector([1, 2])

def test_freeze_with_tuple_and_strict_true():
    from pyrsistent import freeze
    tuple_obj = (1, 2)
    result = freeze(tuple_obj, strict=True)
    assert result == (1, 2)

def test_freeze_with_set_and_strict_true():
    from pyrsistent import pset, freeze
    set_obj = {1, 2}
    result = freeze(set_obj, strict=True)
    assert result == pset([1, 2])

def test_freeze_with_int_and_strict_true():
    from pyrsistent import freeze
    int_obj = 42
    result = freeze(int_obj, strict=True)
    assert result == 42


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    expected = pmap({})
    assert result == expected


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_list():
    result = freeze([1, {'x': 2}, 3])
    expected = pvector([1, pmap({'x': 2}), 3])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, [2, 3]))
    expected = (1, pvector([2, 3]))
    assert result == expected


def test_freeze_set():
    result = freeze({1, 2, 3})
    expected = pset({1, 2, 3})
    assert result == expected


def test_freeze_nested_dict():
    result = freeze({'a': {'b': [1, 2]}})
    expected = pmap({'a': pmap({'b': pvector([1, 2])})})
    assert result == expected


def test_freeze_with_strict_false():
    result = freeze([1, pmap({'a': 2})], strict=False)
    expected = pvector([1, pmap({'a': 2})])
    assert result == expected


def test_freeze_defaultdict():
    dd = collections.defaultdict(list, {'x': [1, 2]})
    result = freeze(dd)
    expected = pmap({'x': pvector([1, 2])})
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    expected = 42
    assert result == expected


def test_freeze_string():
    result = freeze("hello")
    expected = "hello"
    assert result == expected


def test_freeze_pvector_strict():
    pv = pvector([1, 2])
    result = freeze(pv, strict=True)
    expected = pvector([1, 2])
    assert result == expected


def test_freeze_pmap_strict():
    pm = pmap({'a': 1})
    result = freeze(pm, strict=True)
    expected = pmap({'a': 1})
    assert result == expected


# LLM-generated content at query #2
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    expected = pmap({})
    assert result == expected


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_list():
    result = freeze([1, {'x': 2}, 3])
    expected = pvector([1, pmap({'x': 2}), 3])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, [2, 3]))
    expected = (1, pvector([2, 3]))
    assert result == expected


def test_freeze_set():
    result = freeze({1, 2, 3})
    expected = pset({1, 2, 3})
    assert result == expected


def test_freeze_nested_dict():
    result = freeze({'a': {'b': [1, 2]}})
    expected = pmap({'a': pmap({'b': pvector([1, 2])})})
    assert result == expected


def test_freeze_with_strict_false():
    result = freeze([1, pmap({'a': 2})], strict=False)
    expected = pvector([1, pmap({'a': 2})])
    assert result == expected


def test_freeze_with_strict_true():
    result = freeze([1, pmap({'a': 2})], strict=True)
    expected = pvector([1, pmap({'a': freeze(2, strict=True)})])
    assert result == expected


def test_freeze_defaultdict():
    d = collections.defaultdict(list, {'x': [1, 2]})
    result = freeze(d)
    expected = pmap({'x': pvector([1, 2])})
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    expected = 42
    assert result == expected


def test_freeze_empty_list():
    result = freeze([])
    expected = pvector([])
    assert result == expected


def test_freeze_empty_set():
    result = freeze(set())
    expected = pset()
    assert result == expected


def test_freeze_empty_tuple():
    result = freeze(())
    expected = ()
    assert result == expected


# LLM-generated content at query #3
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    assert result == pmap({})


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_list():
    result = freeze([1, {'x': 2}, 3])
    expected = pvector([1, pmap({'x': 2}), 3])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, [2, 3]))
    expected = (1, pvector([2, 3]))
    assert result == expected


def test_freeze_set():
    result = freeze({1, 2, 3})
    expected = pset([1, 2, 3])
    assert result == expected


def test_freeze_nested_structure():
    result = freeze({'a': [1, 2], 'b': {'c': {3, 4}}})
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'c': pset([3, 4])})})
    assert result == expected


def test_freeze_with_strict_false():
    result = freeze([1, pmap({'a': 2})], strict=False)
    expected = pvector([1, pmap({'a': 2})])
    assert result == expected


def test_freeze_with_strict_true():
    result = freeze([1, pmap({'a': [2]})], strict=True)
    expected = pvector([1, pmap({'a': pvector([2])})])
    assert result == expected


def test_freeze_defaultdict():
    dd = collections.defaultdict(list, {'x': [1, 2]})
    result = freeze(dd)
    expected = pmap({'x': pvector([1, 2])})
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    result = freeze("hello")
    assert result == "hello"


def test_freeze_pvector_strict():
    pv = pvector([1, {'a': 2}])
    result = freeze(pv, strict=True)
    expected = pvector([1, pmap({'a': 2})])
    assert result == expected


def test_freeze_pmap_strict():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


# LLM-generated content at query #4
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    def add_to_list(lst, value):
        lst.append(value)
        return lst
    decorated = mutant(add_to_list)
    original_list = [1, 2]
    result = decorated(original_list, 3)
    assert original_list == [1, 2]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]

def test_mutant_decorator_with_dict_argument():
    def update_dict(d, key, value):
        d[key] = value
        return d
    decorated = mutant(update_dict)
    original_dict = {'a': 1}
    result = decorated(original_dict, 'b', 2)
    assert original_dict == {'a': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_decorator_with_set_argument():
    def add_to_set(s, element):
        s.add(element)
        return s
    decorated = mutant(add_to_set)
    original_set = {1, 2}
    result = decorated(original_set, 3)
    assert original_set == {1, 2}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}

def test_mutant_decorator_with_keyword_arguments():
    def merge_dicts(d1, d2):
        d1.update(d2)
        return d1
    decorated = mutant(merge_dicts)
    dict1 = {'x': 10}
    dict2 = {'y': 20}
    result = decorated(d1=dict1, d2=dict2)
    assert dict1 == {'x': 10}
    assert dict2 == {'y': 20}
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 10, 'y': 20}

def test_mutant_decorator_preserves_function_metadata():
    def sample_func():
        """Sample docstring."""
        pass
    decorated = mutant(sample_func)
    assert decorated.__name__ == 'sample_func'
    assert decorated.__doc__ == 'Sample docstring.'

def test_mutant_decorator_with_nested_mutable_structures():
    def modify_nested(obj):
        obj['list'][0] = 99
        return obj
    decorated = mutant(modify_nested)
    original = {'list': [1, 2], 'tuple': (3, [4])}
    result = decorated(original)
    assert original == {'list': [1, 2], 'tuple': (3, [4])}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert list(result['list']) == [99, 2]
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['tuple'][1], PVector)
    assert list(result['tuple'][1]) == [4]

def test_mutant_decorator_with_strict_false_implicitly():
    def identity(obj):
        return obj
    decorated = mutant(identity)
    pvec = pvector([1, 2])
    pmap_obj = pmap({'a': 3})
    result_vec = decorated(pvec)
    result_map = decorated(pmap_obj)
    assert result_vec is pvec
    assert result_map is pmap_obj

def test_mutant_decorator_return_frozen():
    def return_mutable():
        return [1, {'a': 2}]
    decorated = mutant(return_mutable)
    result = decorated()
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)


# LLM-generated content at query #5
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    dd = defaultdict(list, {'a': [1, 2], 'b': [3, 4]})
    result = freeze(dd, strict=True)
    expected = pmap({'a': [1, 2], 'b': [3, 4]})
    assert result == expected
    assert isinstance(result, type(pmap()))


# LLM-generated content at query #6
#--------------------------

def test_thaw_pvector():
    from pyrsistent import v
    result = thaw(v(1, 2, 3))
    expected = [1, 2, 3]
    assert result == expected

def test_thaw_pvector_nested():
    from pyrsistent import v, m
    result = thaw(v(1, m(a=2)))
    expected = [1, {'a': 2}]
    assert result == expected

def test_thaw_pmap():
    from pyrsistent import m
    result = thaw(m(a=1, b=2))
    expected = {'a': 1, 'b': 2}
    assert result == expected

def test_thaw_pmap_nested():
    from pyrsistent import m, v
    result = thaw(m(a=v(1, 2)))
    expected = {'a': [1, 2]}
    assert result == expected

def test_thaw_pset():
    from pyrsistent import s
    result = thaw(s(1, 2, 3))
    expected = {1, 2, 3}
    assert result == expected

def test_thaw_tuple():
    from pyrsistent import v
    result = thaw((1, v(2, 3)))
    expected = (1, [2, 3])
    assert result == expected

def test_thaw_strict_false_list():
    from pyrsistent import v
    result = thaw(v(1, 2), strict=False)
    expected = [1, 2]
    assert result == expected

def test_thaw_strict_false_dict():
    from pyrsistent import m
    result = thaw(m(a=1), strict=False)
    expected = {'a': 1}
    assert result == expected

def test_thaw_strict_true_list():
    result = thaw([1, 2], strict=True)
    expected = [1, 2]
    assert result == expected

def test_thaw_strict_true_dict():
    result = thaw({'a': 1}, strict=True)
    expected = {'a': 1}
    assert result == expected

def test_thaw_non_container():
    result = thaw(42)
    expected = 42
    assert result == expected

def test_thaw_mixed_nested():
    from pyrsistent import v, m, s
    result = thaw(v(m(a=s(1, 2)), (3, 4)))
    expected = [{'a': {1, 2}}, (3, 4)]
    assert result == expected

def test_thaw_empty_containers():
    from pyrsistent import v, m, s
    result = thaw(v(), strict=True)
    expected = []
    assert result == expected
    result = thaw(m(), strict=True)
    expected = {}
    assert result == expected
    result = thaw(s(), strict=True)
    expected = set()
    assert result == expected
    result = thaw((), strict=True)
    expected = ()
    assert result == expected


# LLM-generated content at query #7
#--------------------------

def test_thaw_pset_converts_to_set():
    from pyrsistent import s
    result = thaw(s(1, 2, 3))
    expected = {1, 2, 3}
    assert result == expected


# LLM-generated content at query #8
#--------------------------

def test_mutant_with_positional_args():
    result = mutant(lambda x: x + [1])([0])
    assert result == pvector([0, 1])

def test_mutant_with_keyword_args():
    result = mutant(lambda x=[]: x + [1])(x=[0])
    assert result == pvector([0, 1])

def test_mutant_with_multiple_args():
    result = mutant(lambda x, y: x + y)([1], [2])
    assert result == pvector([1, 2])

def test_mutant_with_args_and_kwargs():
    result = mutant(lambda x, y: x + y)([1], y=[2])
    assert result == pvector([1, 2])

def test_mutant_return_frozen_dict():
    result = mutant(lambda: {'a': [1]})()
    assert result == pmap({'a': pvector([1])})

def test_mutant_return_frozen_set():
    result = mutant(lambda: {1, 2})()
    assert result == pset([1, 2])

def test_mutant_return_frozen_tuple():
    result = mutant(lambda: ([1],))()
    assert result == (pvector([1]),)

def test_mutant_input_frozen():
    mutable_list = [1, 2]
    result = mutant(lambda x: x.append(3) or x)(mutable_list)
    assert result == pvector([1, 2, 3])
    assert mutable_list == [1, 2]

def test_mutant_input_dict_frozen():
    mutable_dict = {'a': [1]}
    result = mutant(lambda d: d['a'].append(2) or d)(mutable_dict)
    assert result == pmap({'a': pvector([1, 2])})
    assert mutable_dict == {'a': [1]}

def test_mutant_input_set_frozen():
    mutable_set = {1, 2}
    result = mutant(lambda s: s.add(3) or s)(mutable_set)
    assert result == pset([1, 2, 3])
    assert mutable_set == {1, 2}

def test_mutant_nested_input_frozen():
    nested = {'list': [1, 2]}
    result = mutant(lambda x: x['list'].append(3) or x)(nested)
    assert result == pmap({'list': pvector([1, 2, 3])})
    assert nested == {'list': [1, 2]}

def test_mutant_with_defaultdict():
    from collections import defaultdict
    dd = defaultdict(list, {'a': [1]})
    result = mutant(lambda d: d['a'].append(2) or d)(dd)
    assert result == pmap({'a': pvector([1, 2])})

def test_mutant_preserves_function_name():
    def my_func(x):
        return x
    decorated = mutant(my_func)
    assert decorated.__name__ == 'my_func'


# LLM-generated content at query #9
#--------------------------

```python
def test_freeze_pmap_strict_true():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert isinstance(result, type(pm))
    assert result['a'] == pmap({'a': [1, 2]})['a']


# LLM-generated content at query #10
#--------------------------

def test_freeze_empty_list():
    result = freeze([])
    assert result == pvector([])


def test_freeze_list_with_elements():
    result = freeze([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_freeze_list_with_nested_dict():
    result = freeze([1, {'a': 2}])
    assert result == pvector([1, pmap({'a': 2})])


def test_freeze_empty_dict():
    result = freeze({})
    assert result == pmap({})


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_dict_with_nested_list():
    result = freeze({'a': [1, 2]})
    assert result == pmap({'a': pvector([1, 2])})


def test_freeze_empty_set():
    result = freeze(set())
    assert result == pset([])


def test_freeze_set_with_elements():
    result = freeze({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_freeze_empty_tuple():
    result = freeze(())
    assert result == ()


def test_freeze_tuple_with_elements():
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)


def test_freeze_tuple_with_nested_list():
    result = freeze((1, [2, 3]))
    assert result == (1, pvector([2, 3]))


def test_freeze_defaultdict():
    dd = collections.defaultdict(list)
    dd['a'] = [1, 2]
    result = freeze(dd)
    assert result == pmap({'a': pvector([1, 2])})


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    result = freeze("hello")
    assert result == "hello"


def test_freeze_nested_structure():
    data = {'a': [1, {'b': set([2, 3])}], 'c': (4, [5])}
    result = freeze(data)
    expected = pmap({'a': pvector([1, pmap({'b': pset([2, 3])})]), 'c': (4, pvector([5]))})
    assert result == expected


def test_freeze_strict_false_with_pmap():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=False)
    assert result is pm


def test_freeze_strict_false_with_pvector():
    pv = pvector([1, 2])
    result = freeze(pv, strict=False)
    assert result is pv


def test_freeze_strict_true_with_pmap():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert result == pmap({'a': pvector([1, 2])})


def test_freeze_strict_true_with_pvector():
    pv = pvector([1, [2, 3]])
    result = freeze(pv, strict=True)
    assert result == pvector([1, pvector([2, 3])])


# LLM-generated content at query #11
#--------------------------

```python
def test_freeze_with_strict_true_and_pmap():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': 1, 'b': 2})
    result = freeze(pmap_instance, strict=True)
    assert isinstance(result, type(pmap_instance))
    assert result == pmap_instance

def test_freeze_with_strict_true_and_pvector():
    from pyrsistent import pvector, freeze
    pvector_instance = pvector([1, 2, 3])
    result = freeze(pvector_instance, strict=True)
    assert isinstance(result, type(pvector_instance))
    assert result == pvector_instance

def test_freeze_with_strict_false_and_pmap():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': 1, 'b': 2})
    result = freeze(pmap_instance, strict=False)
    assert result is pmap_instance

def test_freeze_with_strict_false_and_pvector():
    from pyrsistent import pvector, freeze
    pvector_instance = pvector([1, 2, 3])
    result = freeze(pvector_instance, strict=False)
    assert result is pvector_instance

def test_freeze_with_strict_true_and_dict():
    from pyrsistent import freeze, pmap
    dict_instance = {'a': 1, 'b': 2}
    result = freeze(dict_instance, strict=True)
    assert isinstance(result, pmap)
    assert result == dict_instance

def test_freeze_with_strict_true_and_list():
    from pyrsistent import freeze, pvector
    list_instance = [1, 2, 3]
    result = freeze(list_instance, strict=True)
    assert isinstance(result, pvector)
    assert result == list_instance

def test_freeze_with_strict_true_and_defaultdict():
    from collections import defaultdict
    from pyrsistent import freeze, pmap
    defaultdict_instance = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(defaultdict_instance, strict=True)
    assert isinstance(result, pmap)
    assert result == {'a': 1, 'b': 2}

def test_freeze_with_strict_true_and_set():
    from pyrsistent import freeze, pset
    set_instance = {1, 2, 3}
    result = freeze(set_instance, strict=True)
    assert isinstance(result, pset)
    assert result == set_instance

def test_freeze_with_strict_true_and_tuple():
    from pyrsistent import freeze
    tuple_instance = (1, 2, 3)
    result = freeze(tuple_instance, strict=True)
    assert isinstance(result, tuple)
    assert result == tuple_instance

def test_freeze_with_strict_true_and_nested_pmap():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': pmap({'b': 1})})
    result = freeze(pmap_instance, strict=True)
    assert isinstance(result, type(pmap_instance))
    assert result == pmap_instance

def test_freeze_with_strict_true_and_nested_pvector():
    from pyrsistent import pvector, freeze
    pvector_instance = pvector([pvector([1, 2])])
    result = freeze(pvector_instance, strict=True)
    assert isinstance(result, type(pvector_instance))
    assert result == pvector_instance


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_pmap_strict_true():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert isinstance(result, type(pm))
    assert result['a'] != pm['a']
    assert isinstance(result['a'], type(pm['a']))

def test_freeze_pmap_strict_false():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=False)
    assert result is pm

def test_freeze_pvector_strict_true():
    from pyrsistent import pvector, freeze
    pv = pvector([{'b': 3}])
    result = freeze(pv, strict=True)
    assert isinstance(result, type(pv))
    assert result[0] != pv[0]
    assert isinstance(result[0], type(pv[0]))

def test_freeze_pvector_strict_false():
    from pyrsistent import pvector, freeze
    pv = pvector([{'b': 3}])
    result = freeze(pv, strict=False)
    assert result is pv

def test_freeze_dict_strict_true():
    from pyrsistent import pmap, freeze
    d = {'x': [4, 5]}
    result = freeze(d, strict=True)
    assert isinstance(result, type(pmap()))
    assert isinstance(result['x'], type(pmap()['x']))

def test_freeze_defaultdict_strict_true():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    dd = defaultdict(list, {'y': [6, 7]})
    result = freeze(dd, strict=True)
    assert isinstance(result, type(pmap()))
    assert isinstance(result['y'], type(pmap()['y']))

def test_freeze_list_strict_true():
    from pyrsistent import pvector, freeze
    lst = [{'z': 8}]
    result = freeze(lst, strict=True)
    assert isinstance(result, type(pvector()))
    assert isinstance(result[0], type(pmap()))

def test_freeze_tuple_strict_true():
    from pyrsistent import pmap, freeze
    tup = ([9, 10],)
    result = freeze(tup, strict=True)
    assert isinstance(result, tuple)
    assert isinstance(result[0], type(pmap()['x']))

def test_freeze_set_strict_true():
    from pyrsistent import pset, freeze
    s = {11, 12}
    result = freeze(s, strict=True)
    assert isinstance(result, type(pset()))

def test_freeze_non_container_strict_true():
    from pyrsistent import freeze
    obj = 42
    result = freeze(obj, strict=True)
    assert result is obj


# LLM-generated content at query #13
#--------------------------

def test_freeze_empty_list():
    result = freeze([])
    assert result == pvector([])


def test_freeze_list_with_elements():
    result = freeze([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_freeze_list_with_nested_dict():
    result = freeze([1, {'a': 3}])
    expected = pvector([1, pmap({'a': 3})])
    assert result == expected


def test_freeze_empty_dict():
    result = freeze({})
    assert result == pmap({})


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_dict_with_nested_list():
    result = freeze({'a': [1, 2]})
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_empty_set():
    result = freeze(set())
    assert result == pset([])


def test_freeze_set_with_elements():
    result = freeze(set([1, 2]))
    assert result == pset([1, 2])


def test_freeze_empty_tuple():
    result = freeze(())
    assert result == ()


def test_freeze_tuple_with_elements():
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)


def test_freeze_tuple_with_nested_list():
    result = freeze((1, []))
    expected = (1, pvector([]))
    assert result == expected


def test_freeze_defaultdict():
    dd = collections.defaultdict(list)
    dd['a'] = [1, 2]
    result = freeze(dd)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    result = freeze("hello")
    assert result == "hello"


def test_freeze_strict_false_with_pmap():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=False)
    assert result == pm


def test_freeze_strict_false_with_pvector():
    pv = pvector([1, 2])
    result = freeze(pv, strict=False)
    assert result == pv


def test_freeze_strict_true_with_pmap():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_strict_true_with_pvector():
    pv = pvector([1, [3, 4]])
    result = freeze(pv, strict=True)
    expected = pvector([1, pvector([3, 4])])
    assert result == expected


def test_freeze_nested_structure():
    data = {'a': [1, {'b': set([2, 3])}], 'c': (4, [5])}
    result = freeze(data)
    expected = pmap({'a': pvector([1, pmap({'b': pset([2, 3])})]), 'c': (4, pvector([5]))})
    assert result == expected


# LLM-generated content at query #14
#--------------------------

```python
def test_freeze_pmap_strict_true():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert isinstance(result, type(pm))
    assert result['a'] != pm['a']
    assert isinstance(result['a'], type(pm['a']))

def test_freeze_pvector_strict_true():
    from pyrsistent import pvector, freeze
    pv = pvector([{'b': 3}])
    result = freeze(pv, strict=True)
    assert isinstance(result, type(pv))
    assert result[0] != pv[0]
    assert isinstance(result[0], type(pv[0]))

def test_freeze_pmap_strict_false():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=False)
    assert result is pm

def test_freeze_pvector_strict_false():
    from pyrsistent import pvector, freeze
    pv = pvector([{'b': 3}])
    result = freeze(pv, strict=False)
    assert result is pv

def test_freeze_dict_strict_true():
    from pyrsistent import pmap, freeze
    d = {'x': [1, 2]}
    result = freeze(d, strict=True)
    assert isinstance(result, type(pmap()))
    assert isinstance(result['x'], type(pmap()['x']))

def test_freeze_list_strict_true():
    from pyrsistent import pvector, freeze
    lst = [{'y': 4}]
    result = freeze(lst, strict=True)
    assert isinstance(result, type(pvector()))
    assert isinstance(result[0], type(pmap()))

def test_freeze_defaultdict_strict_true():
    from pyrsistent import pmap, freeze
    from collections import defaultdict
    dd = defaultdict(list, {'z': [5, 6]})
    result = freeze(dd, strict=True)
    assert isinstance(result, type(pmap()))
    assert isinstance(result['z'], type(pvector()))

def test_freeze_tuple_strict_true():
    from pyrsistent import pvector, freeze
    t = ([7, 8],)
    result = freeze(t, strict=True)
    assert isinstance(result, tuple)
    assert isinstance(result[0], type(pvector()))

def test_freeze_set_strict_true():
    from pyrsistent import pset, freeze
    s = {9, 10}
    result = freeze(s, strict=True)
    assert isinstance(result, type(pset()))

def test_freeze_non_container():
    from pyrsistent import freeze
    obj = 42
    result = freeze(obj, strict=True)
    assert result is obj


# LLM-generated content at query #15
#--------------------------

```python
def test_freeze_with_pmap_and_strict_true():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': [1, 2]})
    result = freeze(pmap_instance, strict=True)
    assert isinstance(result, type(pmap_instance))
    assert result['a'][0] == 1
    assert result['a'][1] == 2


# LLM-generated content at query #16
#--------------------------

```python
def test_freeze_with_strict_and_pmap():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': 1, 'b': 2})
    result = freeze(pmap_instance, strict=True)
    assert isinstance(result, type(pmap_instance))
    assert result == pmap_instance


# LLM-generated content at query #17
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    expected = pmap({})
    assert result == expected


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_nested_dict():
    result = freeze({'x': {'y': 5}})
    expected = pmap({'x': pmap({'y': 5})})
    assert result == expected


def test_freeze_empty_list():
    result = freeze([])
    expected = pvector([])
    assert result == expected


def test_freeze_list_with_elements():
    result = freeze([1, 2, 3])
    expected = pvector([1, 2, 3])
    assert result == expected


def test_freeze_nested_list():
    result = freeze([[1, 2], [3, 4]])
    expected = pvector([pvector([1, 2]), pvector([3, 4])])
    assert result == expected


def test_freeze_empty_tuple():
    result = freeze(())
    expected = ()
    assert result == expected


def test_freeze_tuple_with_elements():
    result = freeze((1, 2, 3))
    expected = (1, 2, 3)
    assert result == expected


def test_freeze_nested_tuple():
    result = freeze(([1, 2], (3, 4)))
    expected = (pvector([1, 2]), (3, 4))
    assert result == expected


def test_freeze_empty_set():
    result = freeze(set())
    expected = pset()
    assert result == expected


def test_freeze_set_with_elements():
    result = freeze({1, 2, 3})
    expected = pset([1, 2, 3])
    assert result == expected


def test_freeze_defaultdict():
    dd = collections.defaultdict(list, {'a': [1, 2]})
    result = freeze(dd)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    expected = 42
    assert result == expected


def test_freeze_string():
    result = freeze("hello")
    expected = "hello"
    assert result == expected


def test_freeze_strict_false_with_dict():
    nested_dict = {'a': [1, 2]}
    frozen = freeze(nested_dict, strict=False)
    result = freeze(frozen, strict=False)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_strict_false_with_list():
    nested_list = [[1, 2], [3, 4]]
    frozen = freeze(nested_list, strict=False)
    result = freeze(frozen, strict=False)
    expected = pvector([pvector([1, 2]), pvector([3, 4])])
    assert result == expected


def test_freeze_mixed_structure():
    data = {'list': [1, 2], 'tuple': (3, 4), 'set': {5, 6}, 'nested': {'x': [7, 8]}}
    result = freeze(data)
    expected = pmap({
        'list': pvector([1, 2]),
        'tuple': (3, 4),
        'set': pset([5, 6]),
        'nested': pmap({'x': pvector([7, 8])})
    })
    assert result == expected


# LLM-generated content at query #18
#--------------------------

def test_freeze_empty_dict():
    result = freeze({})
    assert result == pmap({})


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


def test_freeze_list():
    result = freeze([1, {'x': 2}])
    expected = pvector([1, pmap({'x': 2})])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, [2, 3]))
    expected = (1, pvector([2, 3]))
    assert result == expected


def test_freeze_set():
    result = freeze({1, 2, 3})
    expected = pset({1, 2, 3})
    assert result == expected


def test_freeze_nested_structure():
    result = freeze({'a': [1, 2], 'b': {'c': {3, 4}}})
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'c': pset({3, 4})})})
    assert result == expected


def test_freeze_defaultdict():
    d = collections.defaultdict(list, {'x': [1, 2]})
    result = freeze(d)
    expected = pmap({'x': pvector([1, 2])})
    assert result == expected


def test_freeze_strict_false():
    result = freeze([1, pmap({'a': 2})], strict=False)
    expected = pvector([1, pmap({'a': 2})])
    assert result == expected


def test_freeze_strict_true():
    result = freeze([1, pmap({'a': [2]})], strict=True)
    expected = pvector([1, pmap({'a': pvector([2])})])
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    result = freeze("hello")
    assert result == "hello"


# LLM-generated content at query #19
#--------------------------

```python
def test_freeze_with_defaultdict_and_strict_true():
    from collections import defaultdict
    d = defaultdict(list, {'a': [1, 2]})
    result = freeze(d, strict=True)
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_freeze_with_defaultdict_and_strict_false():
    from collections import defaultdict
    d = defaultdict(list, {'a': [1, 2]})
    result = freeze(d, strict=False)
    assert isinstance(result, PMap)
    assert result['a'] == [1, 2]

def test_freeze_with_pmap_and_strict_true():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=True)
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_freeze_with_pmap_and_strict_false():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=False)
    assert result is m


