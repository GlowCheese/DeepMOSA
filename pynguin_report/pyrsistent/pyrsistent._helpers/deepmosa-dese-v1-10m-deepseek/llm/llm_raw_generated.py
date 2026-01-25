####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return_value():
    def mutable_function(lst, dct):
        lst.append(4)
        dct['new'] = 'value'
        return [lst, dct]
    decorated = mutant(mutable_function)
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
    assert result[1] == pmap({'a': 1, 'new': 'value'})

def test_mutant_decorator_with_kwargs():
    def mutable_function(x, y=0):
        x.append(y)
        return x
    decorated = mutant(mutable_function)
    original = [1, 2]
    result = decorated(original, y=3)
    assert original == [1, 2]
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_decorator_preserves_function_metadata():
    def example_func(a, b):
        """Example docstring."""
        return a + b
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_non_container_arguments():
    def simple_func(a, b):
        return a + b
    decorated = mutant(simple_func)
    result = decorated(5, 3)
    assert result == 8

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

def test_mutant_decorator_with_set():
    def func(s):
        s.add(4)
        return s
    decorated = mutant(func)
    original = {1, 2, 3}
    result = decorated(original)
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})

def test_mutant_decorator_with_defaultdict():
    from collections import defaultdict
    def func(dd):
        dd['extra'] = 100
        return dd
    decorated = mutant(func)
    original = defaultdict(int, {'a': 1})
    result = decorated(original)
    assert original == defaultdict(int, {'a': 1})
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'extra': 100})

def test_mutant_decorator_strict_false_behavior():
    def func(pvec):
        return pvec.append(10)
    decorated = mutant(func)
    original = pvector([1, 2])
    result = decorated(original)
    assert original == pvector([1, 2])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 10])

def test_mutant_decorator_empty_arguments():
    def func():
        return {'empty': []}
    decorated = mutant(func)
    result = decorated()
    assert isinstance(result, PMap)
    assert isinstance(result['empty'], PVector)
    assert result['empty'] == pvector([])


# LLM-generated content at query #2
#--------------------------

def test_freeze_empty_list():
    result = freeze([])
    assert result == pvector()


def test_freeze_list_with_int():
    result = freeze([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_freeze_list_with_dict():
    result = freeze([{'a': 1}])
    assert result == pvector([pmap({'a': 1})])


def test_freeze_nested_list():
    result = freeze([[1, 2], [3, 4]])
    assert result == pvector([pvector([1, 2]), pvector([3, 4])])


def test_freeze_empty_dict():
    result = freeze({})
    assert result == pmap({})


def test_freeze_dict_with_int_values():
    result = freeze({'x': 5, 'y': 6})
    assert result == pmap({'x': 5, 'y': 6})


def test_freeze_dict_with_list_values():
    result = freeze({'key': [1, 2, 3]})
    assert result == pmap({'key': pvector([1, 2, 3])})


def test_freeze_nested_dict():
    result = freeze({'outer': {'inner': 10}})
    assert result == pmap({'outer': pmap({'inner': 10})})


def test_freeze_empty_set():
    result = freeze(set())
    assert result == pset()


def test_freeze_set_with_elements():
    result = freeze({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_freeze_empty_tuple():
    result = freeze(())
    assert result == ()


def test_freeze_tuple_with_int():
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)


def test_freeze_tuple_with_list():
    result = freeze(([1, 2], [3, 4]))
    assert result == (pvector([1, 2]), pvector([3, 4]))


def test_freeze_defaultdict():
    dd = collections.defaultdict(list, {'a': [1, 2]})
    result = freeze(dd)
    assert result == pmap({'a': pvector([1, 2])})


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
    pv = pvector([1, 2, 3])
    result = freeze(pv, strict=False)
    assert result == pv


def test_freeze_strict_true_with_pmap():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert result == pmap({'a': pvector([1, 2])})


def test_freeze_strict_true_with_pvector():
    pv = pvector([1, [3, 4]])
    result = freeze(pv, strict=True)
    assert result == pvector([1, pvector([3, 4])])


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_with_strict_true_and_pmap():
    from pyrsistent import freeze, pmap, pvector
    import collections
    pmap_instance = pmap({'a': [1, 2], 'b': {'c': 3}})
    result = freeze(pmap_instance, strict=True)
    assert isinstance(result, type(pmap_instance))
    assert isinstance(result['a'], pvector)
    assert isinstance(result['b'], type(pmap_instance))


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_pmap_strict_true():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)


# LLM-generated content at query #5
#--------------------------

```python
def test_freeze_with_strict_true_and_pmap():
    from pyrsistent import freeze, pmap, pvector
    pmap_instance = pmap({'a': [1, 2]})
    result = freeze(pmap_instance, strict=True)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


# LLM-generated content at query #6
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def modify_map(pmap_arg):
        pmap_arg["new_key"] = 100
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
    assert isinstance(result_map, type(freeze({})))
    assert isinstance(result_set, type(freeze(set())))


# LLM-generated content at query #7
#--------------------------

```python
def test_freeze_with_defaultdict_and_strict_true():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    dd = defaultdict(list, {'a': [1, 2], 'b': [3, 4]})
    result = freeze(dd, strict=True)
    assert isinstance(result, type(pmap()))
    assert result['a'] == [1, 2]
    assert result['b'] == [3, 4]

def test_freeze_with_defaultdict_and_strict_false():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    dd = defaultdict(list, {'a': [1, 2], 'b': [3, 4]})
    result = freeze(dd, strict=False)
    assert isinstance(result, type(pmap()))
    assert result['a'] == [1, 2]
    assert result['b'] == [3, 4]

def test_freeze_with_pmap_and_strict_true():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2], 'b': [3, 4]})
    result = freeze(pm, strict=True)
    assert isinstance(result, type(pmap()))
    assert isinstance(result['a'], type(pmap().evolver().persistent()))
    assert result['a'] == [1, 2]
    assert isinstance(result['b'], type(pmap().evolver().persistent()))
    assert result['b'] == [3, 4]

def test_freeze_with_pmap_and_strict_false():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2], 'b': [3, 4]})
    result = freeze(pm, strict=False)
    assert result is pm


# LLM-generated content at query #8
#--------------------------

def test_mutant_decorator_freezes_args_and_return():
    @mutant
    def add_one_to_list(lst):
        lst.append(1)
        return lst
    original_list = [5, 6]
    result = add_one_to_list(original_list)
    assert result == pvector([5, 6, 1])
    assert original_list == [5, 6]
    assert isinstance(result, PVector)

def test_mutant_decorator_freezes_kwargs():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    original_dict = {'a': 1}
    result = update_dict(original_dict, key='b', value=2)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}
    assert isinstance(result, PMap)

def test_mutant_decorator_preserves_function_metadata():
    @mutant
    def sample_func(x):
        """Sample docstring."""
        return x
    assert sample_func.__name__ == 'sample_func'
    assert sample_func.__doc__ == 'Sample docstring.'

def test_mutant_decorator_with_recursive_structures():
    @mutant
    def modify_nested(obj):
        if isinstance(obj, dict):
            obj['inner'] = [7, 8]
        return obj
    original = {'a': [1, 2]}
    result = modify_nested(original)
    expected = pmap({'a': pvector([1, 2]), 'inner': pvector([7, 8])})
    assert result == expected
    assert original == {'a': [1, 2]}

def test_mutant_decorator_with_frozen_inputs():
    @mutant
    def identity(x):
        return x
    frozen_input = pvector([1, 2, 3])
    result = identity(frozen_input)
    assert result == pvector([1, 2, 3])
    assert result is frozen_input

def test_mutant_decorator_with_set():
    @mutant
    def add_to_set(s, elem):
        s.add(elem)
        return s
    original_set = {1, 2}
    result = add_to_set(original_set, 3)
    assert result == pset({1, 2, 3})
    assert original_set == {1, 2}
    assert isinstance(result, PSet)

def test_mutant_decorator_with_tuple():
    @mutant
    def wrap_tuple(t):
        return (list(t),)
    original = (1, 2)
    result = wrap_tuple(original)
    assert result == (pvector([1, 2]),)
    assert original == (1, 2)

def test_mutant_decorator_no_side_effects_on_multiple_calls():
    counter = 0
    @mutant
    def increment_counter(x):
        nonlocal counter
        counter += 1
        return x + counter
    assert increment_counter(10) == 11
    assert increment_counter(10) == 12
    assert counter == 2


# LLM-generated content at query #9
#--------------------------

```python
def test_freeze_with_defaultdict_and_strict_true():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    dd = defaultdict(list, {'a': [1, 2], 'b': [3, 4]})
    result = freeze(dd, strict=True)
    assert isinstance(result, type(pmap()))
    assert result['a'] == pmap({'a': [1, 2], 'b': [3, 4]})['a']
    assert result['b'] == pmap({'a': [1, 2], 'b': [3, 4]})['b']


# LLM-generated content at query #10
#--------------------------

```python
def test_freeze_with_strict_true_and_pmap():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': 1, 'b': 2})
    result = freeze(pmap_instance, strict=True)
    assert isinstance(result, type(pmap_instance))
    assert result == pmap_instance


# LLM-generated content at query #11
#--------------------------

```python
def test_freeze_pmap_strict_true():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert isinstance(result, type(pm))
    assert result['a'] == pmap({'a': [1, 2]})['a']


# LLM-generated content at query #12
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    def add_one(x):
        x[0] += 1
        return x
    decorated = mutant(add_one)
    original = [1, 2, 3]
    result = decorated(original)
    assert original == [1, 2, 3]
    assert isinstance(result, PVector)
    assert result == pvector([2, 2, 3])

def test_mutant_decorator_with_dict_argument():
    def update_dict(d):
        d['key'] = 'new'
        return d
    decorated = mutant(update_dict)
    original = {'key': 'old'}
    result = decorated(original)
    assert original == {'key': 'old'}
    assert isinstance(result, PMap)
    assert result == pmap({'key': 'new'})

def test_mutant_decorator_with_set_argument():
    def add_to_set(s):
        s.add(4)
        return s
    decorated = mutant(add_to_set)
    original = {1, 2, 3}
    result = decorated(original)
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})

def test_mutant_decorator_with_multiple_arguments():
    def concat_and_modify(a, b):
        a.append(99)
        b['x'] = 100
        return (a, b)
    decorated = mutant(concat_and_modify)
    list_arg = [1, 2]
    dict_arg = {'x': 10}
    result = decorated(list_arg, dict_arg)
    assert list_arg == [1, 2]
    assert dict_arg == {'x': 10}
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert result[0] == pvector([1, 2, 99])
    assert result[1] == pmap({'x': 100})

def test_mutant_decorator_with_keyword_arguments():
    def merge_dicts(d1, d2):
        d1.update(d2)
        return d1
    decorated = mutant(merge_dicts)
    original = {'a': 1}
    extra = {'b': 2}
    result = decorated(original, d2=extra)
    assert original == {'a': 1}
    assert extra == {'b': 2}
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_decorator_preserves_function_metadata():
    def example_func(x):
        """Example docstring."""
        return x
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_nested_structures():
    def modify_nested(obj):
        obj['list'][0] = 'changed'
        obj['tuple'][1]['inner'] = 'updated'
        return obj
    decorated = mutant(modify_nested)
    original = {'list': ['original'], 'tuple': (1, {'inner': 'old'})}
    result = decorated(original)
    assert original == {'list': ['original'], 'tuple': (1, {'inner': 'old'})}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['tuple'], tuple)
    assert result['list'] == pvector(['changed'])
    assert result['tuple'][1] == pmap({'inner': 'updated'})

def test_mutant_decorator_with_no_mutation():
    def pure_function(x):
        return x * 2
    decorated = mutant(pure_function)
    result = decorated(5)
    assert result == 10

def test_mutant_decorator_freezes_returned_mutable():
    def return_mutable():
        return [1, 2, 3]
    decorated = mutant(return_mutable)
    result = decorated()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #13
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    def mutable_function(lst, dct):
        lst.append(4)
        dct['new'] = 'value'
        return [lst, dct]
    decorated = mutant(mutable_function)
    original_list = [1, 2, 3]
    original_dict = {'key': 'old'}
    result = decorated(original_list, original_dict)
    assert original_list == [1, 2, 3]
    assert original_dict == {'key': 'old'}
    assert isinstance(result, PVector)
    assert len(result) == 2
    assert isinstance(result[0], PVector)
    assert result[0] == pvector([1, 2, 3, 4])
    assert isinstance(result[1], PMap)
    assert result[1] == pmap({'key': 'old', 'new': 'value'})

def test_mutant_decorator_with_keyword_arguments():
    def mutable_function(a, b=[]):
        b.append(a)
        return b
    decorated = mutant(mutable_function)
    result = decorated(1, b=[2, 3])
    assert isinstance(result, PVector)
    assert result == pvector([2, 3, 1])

def test_mutant_decorator_freezes_nested_structures():
    def mutable_function(data):
        data['list'][0] = 'mutated'
        data['tuple'][1].append('mutated')
        return data
    decorated = mutant(mutable_function)
    original = {'list': ['original'], 'tuple': (1, ['original'])}
    result = decorated(original)
    assert original == {'list': ['original'], 'tuple': (1, ['original'])}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert result['list'] == pvector(['mutated'])
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['tuple'][1], PVector)
    assert result['tuple'][1] == pvector(['original', 'mutated'])

def test_mutant_decorator_preserves_function_metadata():
    def sample_function(x):
        """Sample docstring."""
        return x
    decorated = mutant(sample_function)
    assert decorated.__name__ == 'sample_function'
    assert decorated.__doc__ == 'Sample docstring.'

def test_mutant_decorator_with_no_arguments():
    def mutable_function():
        return {'a': [1, 2, 3]}
    decorated = mutant(mutable_function)
    result = decorated()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert result['a'] == pvector([1, 2, 3])

def test_mutant_decorator_with_strict_false():
    def mutable_function(pmap_obj, pvector_obj):
        return [pmap_obj, pvector_obj]
    decorated = mutant(mutable_function)
    pmap_arg = pmap({'x': 1})
    pvector_arg = pvector([1, 2, 3])
    result = decorated(pmap_arg, pvector_arg)
    assert result[0] is pmap_arg
    assert result[1] is pvector_arg


# LLM-generated content at query #14
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
    result = freeze([1, {'x': 2}])
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert isinstance(result[1], PMap)
    assert result[1]['x'] == 2


def test_freeze_tuple():
    result = freeze((1, [2, 3]))
    assert isinstance(result, tuple)
    assert result[0] == 1
    assert isinstance(result[1], PVector)
    assert result[1][0] == 2
    assert result[1][1] == 3


def test_freeze_set():
    result = freeze({1, 2, 3})
    assert isinstance(result, PSet)
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_freeze_defaultdict():
    d = collections.defaultdict(list)
    d['a'].append(1)
    result = freeze(d)
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert result['a'][0] == 1


def test_freeze_strict_false():
    result = freeze([1, {'x': 2}], strict=False)
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert isinstance(result[1], dict)
    assert result[1]['x'] == 2


def test_freeze_pmap_strict():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=True)
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert result['a'][0] == 1
    assert result['a'][1] == 2


def test_freeze_pvector_strict():
    v = pvector([1, {'x': 2}])
    result = freeze(v, strict=True)
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert isinstance(result[1], PMap)
    assert result[1]['x'] == 2


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_nested_structure():
    data = {'a': [1, {2, 3}], 'b': (4, [5])}
    result = freeze(data)
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert result['a'][0] == 1
    assert isinstance(result['a'][1], PSet)
    assert 2 in result['a'][1]
    assert 3 in result['a'][1]
    assert isinstance(result['b'], tuple)
    assert result['b'][0] == 4
    assert isinstance(result['b'][1], PVector)
    assert result['b'][1][0] == 5


# LLM-generated content at query #15
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    def mutable_func(lst, dct):
        lst.append(4)
        dct['new'] = 'value'
        return {'modified': True, 'list': lst, 'dict': dct}
    decorated = mutant(mutable_func)
    original_list = [1, 2, 3]
    original_dict = {'key': 'original'}
    result = decorated(original_list, original_dict)
    assert original_list == [1, 2, 3]
    assert original_dict == {'key': 'original'}
    assert isinstance(result, PMap)
    assert result['modified'] is True
    assert isinstance(result['list'], PVector)
    assert result['list'].to_list() == [1, 2, 3, 4]
    assert isinstance(result['dict'], PMap)
    assert result['dict']['new'] == 'value'

def test_mutant_decorator_with_kwargs():
    def func(a, b=2):
        a.append(b)
        return a
    decorated = mutant(func)
    arg = [1]
    result = decorated(arg, b=3)
    assert arg == [1]
    assert isinstance(result, PVector)
    assert result.to_list() == [1, 3]

def test_mutant_decorator_preserves_function_metadata():
    def example():
        """Example docstring."""
        pass
    decorated = mutant(example)
    assert decorated.__name__ == 'example'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_no_arguments():
    def func():
        return {'a': [1, 2]}
    decorated = mutant(func)
    result = decorated()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert result['a'].to_list() == [1, 2]

def test_mutant_decorator_freezes_nested_structures():
    def func(data):
        data['list'][0] = 'changed'
        data['tuple'][1].append(99)
        return data
    decorated = mutant(func)
    input_data = {'list': ['original'], 'tuple': (1, [2, 3])}
    result = decorated(input_data)
    assert input_data['list'][0] == 'original'
    assert input_data['tuple'][1] == [2, 3]
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert result['list'][0] == 'changed'
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['tuple'][1], PVector)
    assert result['tuple'][1].to_list() == [2, 3, 99]


# LLM-generated content at query #16
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    def mutable_func(lst, dct):
        lst.append(1)
        dct['key'] = 'value'
        return [lst, dct]
    decorated = mutant(mutable_func)
    original_list = []
    original_dict = {}
    result = decorated(original_list, original_dict)
    assert original_list == []
    assert original_dict == {}
    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)

def test_mutant_decorator_with_keyword_arguments():
    def func(a, b):
        return {'a': a, 'b': b}
    decorated = mutant(func)
    result = decorated(a=[1, 2], b={'x': 3})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)

def test_mutant_decorator_preserves_function_metadata():
    def example():
        pass
    example.__doc__ = 'test doc'
    example.__name__ = 'example'
    decorated = mutant(example)
    assert decorated.__doc__ == 'test doc'
    assert decorated.__name__ == 'example'

def test_mutant_decorator_with_no_arguments():
    def func():
        return {'a': [1, 2]}
    decorated = mutant(func)
    result = decorated()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)

def test_mutant_decorator_with_strict_freeze():
    def func(pmap_arg):
        return pmap_arg
    decorated = mutant(func)
    input_pmap = pmap({'inner': [1, 2]})
    result = decorated(input_pmap)
    assert isinstance(result, PMap)
    assert isinstance(result['inner'], PVector)


# LLM-generated content at query #17
#--------------------------

def test_mutant_decorator_predicate_false():
    from pyrsistent import freeze, mutant
    from pyrsistent import pmap, pset
    original_map = pmap({'a': 1})
    original_set = pset([1, 2])
    original_list = [1, 2, 3]
    original_dict = {'x': 10}
    @mutant
    def modify_structures(m, s, lst, d):
        m['a'] = 2
        s.add(3)
        lst.append(4)
        d['x'] = 20
        return m, s, lst, d
    result_map, result_set, result_list, result_dict = modify_structures(original_map, original_set, original_list, original_dict)
    assert original_map['a'] == 1
    assert 3 not in original_set
    assert len(original_list) == 3
    assert original_dict['x'] == 10
    assert result_map['a'] == 1
    assert 3 not in result_set
    assert len(result_list) == 3
    assert result_dict['x'] == 10


# LLM-generated content at query #18
#--------------------------

def test_mutant_with_positional_args():
    def add_one(x):
        x[0] = x[0] + 1
        return x
    decorated = mutant(add_one)
    original = [1]
    result = decorated(original)
    assert original == [1]
    assert isinstance(result, PVector)
    assert result[0] == 2

def test_mutant_with_keyword_args():
    def update_dict(d, key, value):
        d[key] = value
        return d
    decorated = mutant(update_dict)
    original = {'a': 1}
    result = decorated(original, key='a', value=2)
    assert original == {'a': 1}
    assert isinstance(result, PMap)
    assert result['a'] == 2

def test_mutant_with_mixed_args():
    def modify_list_and_dict(lst, d, factor):
        lst[0] = lst[0] * factor
        d['multiplied'] = lst[0]
        return lst, d
    decorated = mutant(modify_list_and_dict)
    lst = [5]
    d = {}
    result_lst, result_dict = decorated(lst, d, factor=3)
    assert lst == [5]
    assert d == {}
    assert isinstance(result_lst, PVector)
    assert result_lst[0] == 15
    assert isinstance(result_dict, PMap)
    assert result_dict['multiplied'] == 15

def test_mutant_returns_frozen_result():
    def return_mutable():
        return {'inner': [1, 2, 3]}
    decorated = mutant(return_mutable)
    result = decorated()
    assert isinstance(result, PMap)
    assert isinstance(result['inner'], PVector)

def test_mutant_with_set():
    def add_to_set(s, element):
        s.add(element)
        return s
    decorated = mutant(add_to_set)
    original = {1, 2}
    result = decorated(original, 3)
    assert original == {1, 2}
    assert isinstance(result, PSet)
    assert 3 in result

def test_mutant_with_tuple():
    def modify_tuple(t):
        return (t[0] + 1,)
    decorated = mutant(modify_tuple)
    original = (5,)
    result = decorated(original)
    assert original == (5,)
    assert result == (6,)

def test_mutant_strict_false_implicitly():
    def modify_pvector(pv):
        pv[0] = pv[0] + 1
        return pv
    decorated = mutant(modify_pvector)
    original = pvector([10])
    result = decorated(original)
    assert original[0] == 10
    assert isinstance(result, PVector)
    assert result[0] == 11

def test_mutant_preserves_function_metadata():
    def example_func(x):
        """Example docstring."""
        return x
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == """Example docstring."""


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def modify_map(pmap_arg):
        pmap_arg["new_key"] = "new_value"
        return pmap_arg

    @mutant
    def modify_set(pset_arg):
        pset_arg.add("new_element")
        return pset_arg

    original_map = m(a=1, b=2)
    original_set = s(1, 2, 3)

    result_map = modify_map(original_map)
    result_set = modify_set(original_set)

    assert original_map == m(a=1, b=2)
    assert original_set == s(1, 2, 3)
    assert result_map == m(a=1, b=2, new_key="new_value")
    assert result_set == s(1, 2, 3, "new_element")
    assert isinstance(result_map, type(freeze({})))
    assert isinstance(result_set, type(freeze(set())))


# LLM-generated content at query #20
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return_value():
    def mutable_function(lst, dct):
        lst.append(4)
        dct['new'] = 'value'
        return [lst, dct]
    decorated = mutant(mutable_function)
    original_list = [1, 2, 3]
    original_dict = {'key': 'old'}
    result = decorated(original_list, original_dict)
    assert original_list == [1, 2, 3]
    assert original_dict == {'key': 'old'}
    assert isinstance(result, PVector)
    assert len(result) == 2
    assert isinstance(result[0], PVector)
    assert result[0] == pvector([1, 2, 3, 4])
    assert isinstance(result[1], PMap)
    assert result[1] == pmap({'key': 'old', 'new': 'value'})

def test_mutant_decorator_with_keyword_arguments():
    def mutable_function(a, b=[]):
        b.append(a)
        return b
    decorated = mutant(mutable_function)
    result = decorated(1, b=[0])
    assert isinstance(result, PVector)
    assert result == pvector([0, 1])

def test_mutant_decorator_preserves_function_metadata():
    def sample_function():
        """Sample docstring."""
        pass
    decorated = mutant(sample_function)
    assert decorated.__name__ == 'sample_function'
    assert decorated.__doc__ == 'Sample docstring.'

def test_mutant_decorator_with_no_arguments():
    def constant_function():
        return {'answer': 42}
    decorated = mutant(constant_function)
    result = decorated()
    assert isinstance(result, PMap)
    assert result == pmap({'answer': 42})

def test_mutant_decorator_freezes_nested_structures():
    def function_with_nested_args(data):
        data['list'][0] = 'mutated'
        return data
    decorated = mutant(function_with_nested_args)
    input_data = {'list': ['original'], 'set': {1, 2}}
    result = decorated(input_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert result['list'] == pvector(['mutated'])
    assert isinstance(result['set'], PSet)
    assert result['set'] == pset({1, 2})


# LLM-generated content at query #21
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return_value():
    def mutable_function(lst, dct):
        lst.append(4)
        dct['new'] = 'value'
        return [lst, dct]
    decorated = mutant(mutable_function)
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
    assert result[1] == pmap({'a': 1, 'new': 'value'})

def test_mutant_decorator_with_keyword_arguments():
    def mutable_function(x, y=[]):
        y.append(x)
        return y
    decorated = mutant(mutable_function)
    result = decorated(1, y=[0])
    assert isinstance(result, PVector)
    assert result == pvector([0, 1])

def test_mutant_decorator_preserves_function_metadata():
    def example_func(a, b):
        """Example docstring."""
        return a + b
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == "Example docstring."

def test_mutant_decorator_with_non_container_arguments():
    def simple_add(a, b):
        return a + b
    decorated = mutant(simple_add)
    result = decorated(5, 3)
    assert result == 8

def test_mutant_decorator_freezes_nested_structures_in_arguments():
    def func(data):
        return data
    decorated = mutant(func)
    input_data = {'list': [1, 2, {'inner': 'dict'}]}
    result = decorated(input_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['list'][2], PMap)

def test_mutant_decorator_with_strict_false_implicitly():
    original_freeze = freeze
    def mock_freeze(o, strict=True):
        if isinstance(o, list) and o == [1, 2, 3]:
            return 'mock_frozen_list'
        return original_freeze(o, strict)
    import pyrsistent._helpers
    pyrsistent._helpers.freeze = mock_freeze
    try:
        def func(lst):
            return lst
        decorated = mutant(func)
        result = decorated([1, 2, 3])
        assert result == 'mock_frozen_list'
    finally:
        pyrsistent._helpers.freeze = original_freeze


# LLM-generated content at query #22
#--------------------------

def test_mutant_with_positional_args():
    def add_one(x):
        x[0] = x[0] + 1
        return x
    decorated = mutant(add_one)
    original = [1, 2, 3]
    result = decorated(original)
    assert result == pvector([2, 2, 3])
    assert original == [1, 2, 3]

def test_mutant_with_keyword_args():
    def update_dict(d, key, value):
        d[key] = value
        return d
    decorated = mutant(update_dict)
    original = {'a': 1}
    result = decorated(original, key='a', value=2)
    assert result == pmap({'a': 2})
    assert original == {'a': 1}

def test_mutant_with_mixed_args():
    def modify_list_and_dict(lst, d, factor):
        lst.append(factor)
        d['factor'] = factor
        return lst, d
    decorated = mutant(modify_list_and_dict)
    lst = [1, 2]
    d = {'initial': 0}
    result = decorated(lst, d, 3)
    expected = (pvector([1, 2, 3]), pmap({'initial': 0, 'factor': 3}))
    assert result == expected
    assert lst == [1, 2]
    assert d == {'initial': 0}

def test_mutant_return_value_frozen():
    def return_mutable():
        return [1, 2, 3]
    decorated = mutant(return_mutable)
    result = decorated()
    assert result == pvector([1, 2, 3])

def test_mutant_with_nested_mutables():
    def deep_update(x):
        x['a'][0] = 99
        x['b'].add(4)
        return x
    decorated = mutant(deep_update)
    original = {'a': [1, 2], 'b': {1, 2, 3}}
    result = decorated(original)
    assert result == pmap({'a': pvector([99, 2]), 'b': pset({1, 2, 3, 4})})
    assert original == {'a': [1, 2], 'b': {1, 2, 3}}

def test_mutant_no_side_effects_on_kwargs():
    def change_kwarg(**kwargs):
        kwargs['x'] = 100
        return kwargs
    decorated = mutant(change_kwarg)
    original = {'x': 1}
    result = decorated(**original)
    assert result == pmap({'x': 100})
    assert original == {'x': 1}

def test_mutant_with_empty_args():
    def constant():
        return {'empty': []}
    decorated = mutant(constant)
    result = decorated()
    assert result == pmap({'empty': pvector([])})

def test_mutant_preserves_function_metadata():
    def example():
        """Example docstring."""
        pass
    decorated = mutant(example)
    assert decorated.__name__ == 'example'
    assert decorated.__doc__ == 'Example docstring.'


# LLM-generated content at query #23
#--------------------------

def test_mutant_with_list_argument():
    def append_to_list(lst):
        lst.append(1)
        return lst
    decorated = mutant(append_to_list)
    original = []
    result = decorated(original)
    assert original == []
    assert isinstance(result, PVector)
    assert list(result) == [1]

def test_mutant_with_dict_argument():
    def add_to_dict(d):
        d['new'] = 42
        return d
    decorated = mutant(add_to_dict)
    original = {}
    result = decorated(original)
    assert original == {}
    assert isinstance(result, PMap)
    assert dict(result) == {'new': 42}

def test_mutant_with_set_argument():
    def add_to_set(s):
        s.add(5)
        return s
    decorated = mutant(add_to_set)
    original = set()
    result = decorated(original)
    assert original == set()
    assert isinstance(result, PSet)
    assert set(result) == {5}

def test_mutant_with_tuple_argument():
    def modify_tuple(t):
        return t + (1,)
    decorated = mutant(modify_tuple)
    original = (0,)
    result = decorated(original)
    assert original == (0,)
    assert isinstance(result, tuple)
    assert result == (0, 1)

def test_mutant_with_keyword_arguments():
    def combine(**kwargs):
        kwargs['combined'] = True
        return kwargs
    decorated = mutant(combine)
    result = decorated(a=1, b=2)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2, 'combined': True}

def test_mutant_with_mixed_arguments():
    def func(lst, d, s):
        lst.append(1)
        d['key'] = 'value'
        s.add(99)
        return lst, d, s
    decorated = mutant(func)
    lst_arg = []
    dict_arg = {}
    set_arg = set()
    result = decorated(lst_arg, dict_arg, set_arg)
    assert lst_arg == []
    assert dict_arg == {}
    assert set_arg == set()
    assert isinstance(result, tuple)
    res_lst, res_dict, res_set = result
    assert isinstance(res_lst, PVector)
    assert list(res_lst) == [1]
    assert isinstance(res_dict, PMap)
    assert dict(res_dict) == {'key': 'value'}
    assert isinstance(res_set, PSet)
    assert set(res_set) == {99}

def test_mutant_return_value_frozen():
    def return_mutable():
        return [1, 2, 3]
    decorated = mutant(return_mutable)
    result = decorated()
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]

def test_mutant_nested_structures():
    def process(data):
        data['list'][0] = 'modified'
        return data
    decorated = mutant(process)
    original = {'list': ['original']}
    result = decorated(original)
    assert original == {'list': ['original']}
    assert isinstance(result, PMap)
    inner = result['list']
    assert isinstance(inner, PVector)
    assert list(inner) == ['modified']


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_decorator_does_not_freeze_arguments_when_no_mutation():
    from pyrsistent import freeze, pset, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def no_mutation_func(s, m):
        return (s, m)

    original_set = pset([1, 2, 3])
    original_map = pmap({'a': 1})
    result_set, result_map = no_mutation_func(original_set, original_map)
    assert result_set is original_set
    assert result_map is original_map


# LLM-generated content at query #25
#--------------------------

def test_mutant_with_positional_args():
    result = mutant(lambda x, y: x + y)(1, 2)
    assert result == 3

def test_mutant_with_keyword_args():
    result = mutant(lambda a, b: a * b)(a=3, b=4)
    assert result == 12

def test_mutant_with_mutable_list_input():
    result = mutant(lambda lst: lst.append(4))([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_mutable_dict_input():
    result = mutant(lambda d: d.update({'c': 3}))({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_with_mutable_set_input():
    result = mutant(lambda s: s.add(4))({1, 2, 3})
    assert result == pset({1, 2, 3, 4})

def test_mutant_with_nested_mutable_input():
    result = mutant(lambda x: x[0].append(3))([[1, 2]])
    assert result == pvector([pvector([1, 2, 3])])

def test_mutant_with_mixed_args():
    result = mutant(lambda x, y, z: x + y + z)(1, y=2, z=3)
    assert result == 6

def test_mutant_returns_frozen_output():
    result = mutant(lambda: [1, 2, 3])()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_no_args():
    result = mutant(lambda: 42)()
    assert result == 42

def test_mutant_preserves_function_name():
    def my_func():
        return 1
    decorated = mutant(my_func)
    assert decorated.__name__ == 'my_func'


# LLM-generated content at query #26
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

    initial_dict = m(a=1, b=2)
    initial_set = s(1, 2, 3)
    result = modify_data(initial_dict, initial_set)

    assert initial_dict == m(a=1, b=2)
    assert initial_set == s(1, 2, 3)
    assert isinstance(result, type(freeze({})))
    assert result['modified'] is True
    assert isinstance(result['dict'], type(freeze({})))
    assert isinstance(result['set'], type(freeze(set())))


# LLM-generated content at query #27
#--------------------------

def test_mutant_decorator_does_not_mutate_inputs():
    from pyrsistent import freeze, pset, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def modify_set(s):
        s.add(4)
        return s

    @mutant
    def modify_map(m):
        m['new'] = 99
        return m

    original_set = pset([1, 2, 3])
    original_map = pmap({'a': 1})
    result_set = modify_set(original_set)
    result_map = modify_map(original_map)
    assert original_set == pset([1, 2, 3])
    assert original_map == pmap({'a': 1})
    assert result_set == pset([1, 2, 3, 4])
    assert result_map == pmap({'a': 1, 'new': 99})


# LLM-generated content at query #28
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    def mutable_func(lst, dct):
        lst.append(4)
        dct['new'] = 5
        return [lst, dct]
    decorated = mutant(mutable_func)
    input_list = [1, 2, 3]
    input_dict = {'a': 1}
    result = decorated(input_list, input_dict)
    assert input_list == [1, 2, 3]
    assert input_dict == {'a': 1}
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
    arg = [1]
    result = decorated(arg, y=2)
    assert arg == [1]
    assert isinstance(result, PVector)
    assert result == pvector([1, 2])

def test_mutant_decorator_preserves_function_metadata():
    def original(a, b):
        """Original docstring."""
        return a + b
    decorated = mutant(original)
    assert decorated.__name__ == 'original'
    assert decorated.__doc__ == 'Original docstring.'

def test_mutant_decorator_with_non_mutable_return():
    def func():
        return 42
    decorated = mutant(func)
    result = decorated()
    assert result == 42

def test_mutant_decorator_freezes_nested_structures():
    def func(data):
        data['list'].append(99)
        return data
    decorated = mutant(func)
    input_data = {'list': [1, 2], 'tuple': (3, [4])}
    result = decorated(input_data)
    assert input_data == {'list': [1, 2], 'tuple': (3, [4])}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert result['list'] == pvector([1, 2, 99])
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['tuple'][1], PVector)
    assert result['tuple'][1] == pvector([4])


# LLM-generated content at query #29
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    def mutable_func(lst, dct):
        lst.append(4)
        dct['new'] = 5
        return lst, dct
    decorated = mutant(mutable_func)
    input_list = [1, 2, 3]
    input_dict = {'a': 1}
    result = decorated(input_list, input_dict)
    assert input_list == [1, 2, 3]
    assert input_dict == {'a': 1}
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert result[0] == pvector([1, 2, 3, 4])
    assert result[1] == pmap({'a': 1, 'new': 5})

def test_mutant_decorator_with_keyword_arguments():
    def mutable_func(x, y=0):
        x.append(y)
        return x
    decorated = mutant(mutable_func)
    arg = [1]
    result = decorated(arg, y=2)
    assert arg == [1]
    assert isinstance(result, PVector)
    assert result == pvector([1, 2])

def test_mutant_decorator_preserves_function_metadata():
    def original(a, b):
        """Original docstring."""
        return a + b
    decorated = mutant(original)
    assert decorated.__name__ == 'original'
    assert decorated.__doc__ == 'Original docstring.'

def test_mutant_decorator_with_frozen_inputs():
    def func(pvec, pmap):
        return pvec.append(1), pmap.set('b', 2)
    decorated = mutant(func)
    pv = pvector([0])
    pm = pmap({'a': 1})
    result = decorated(pv, pm)
    assert pv == pvector([0])
    assert pm == pmap({'a': 1})
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert result[0] == pvector([0, 1])
    assert result[1] == pmap({'a': 1, 'b': 2})

def test_mutant_decorator_with_non_container_arguments():
    def func(num, string):
        return num + 1, string + '!'
    decorated = mutant(func)
    result = decorated(5, 'hello')
    assert result == (6, 'hello!')

def test_mutant_decorator_with_strict_false():
    def func(d):
        d['x'] = [1, 2]
        return d
    decorated = mutant(func)
    result = decorated({})
    assert isinstance(result, PMap)
    assert result['x'] == [1, 2]


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
    result = freeze([1, pmap({'x': 2})], strict=False)
    expected = pvector([1, pmap({'x': 2})])
    assert result == expected


def test_freeze_with_strict_true():
    result = freeze([1, pmap({'x': [2]})], strict=True)
    expected = pvector([1, pmap({'x': pvector([2])})])
    assert result == expected


def test_freeze_defaultdict():
    d = collections.defaultdict(list, {'a': [1, 2]})
    result = freeze(d)
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


# LLM-generated content at query #31
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return_value():
    def mutable_function(lst, dct):
        lst.append(4)
        dct['new'] = 'value'
        return [lst, dct]
    decorated = mutant(mutable_function)
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
    assert result[1] == pmap({'a': 1, 'new': 'value'})

def test_mutant_decorator_with_keyword_arguments():
    def mutable_function(x, y=[]):
        y.append(x)
        return y
    decorated = mutant(mutable_function)
    result = decorated(1, y=[0])
    assert isinstance(result, PVector)
    assert result == pvector([0, 1])

def test_mutant_decorator_preserves_function_metadata():
    def example_func(a, b):
        """Example docstring."""
        return a + b
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_non_container_arguments():
    def simple_add(a, b):
        return a + b
    decorated = mutant(simple_add)
    result = decorated(5, 3)
    assert result == 8

def test_mutant_decorator_with_strict_freeze():
    def modify_pvector(v):
        return v.append(10)
    decorated = mutant(modify_pvector)
    original = pvector([1, 2])
    result = decorated(original)
    assert original == pvector([1, 2])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 10])

def test_mutant_decorator_with_nested_structures():
    def process(data):
        data['list'][0] = 'changed'
        return data
    decorated = mutant(process)
    original = {'list': ['original'], 'tuple': (1, [2, 3])}
    result = decorated(original)
    assert original == {'list': ['original'], 'tuple': (1, [2, 3])}
    assert isinstance(result, PMap)
    assert result['list'] == pvector(['changed'])
    assert isinstance(result['tuple'], tuple)
    assert result['tuple'][0] == 1
    assert isinstance(result['tuple'][1], PVector)
    assert result['tuple'][1] == pvector([2, 3])

def test_mutant_decorator_with_set_argument():
    def add_to_set(s, element):
        return s.union({element})
    decorated = mutant(add_to_set)
    original = {1, 2, 3}
    result = decorated(original, 4)
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})

def test_mutant_decorator_freezes_kwargs_dict_values():
    def update_kwargs(**kwargs):
        kwargs['list'] = [100]
        return kwargs
    decorated = mutant(update_kwargs)
    result = decorated(list=[1, 2])
    assert isinstance(result, PMap)
    assert result == pmap({'list': pvector([100])})


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_decorator_does_not_freeze_arguments_when_no_mutation_occurs():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def no_mutation_func(ps, pm):
        return ps, pm

    original_ps = s(1, 2, 3)
    original_pm = m(a=1, b=2)
    result_ps, result_pm = no_mutation_func(original_ps, original_pm)
    assert result_ps is original_ps
    assert result_pm is original_pm


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant_decorator_does_not_freeze_arguments_when_no_mutation_occurs():
    @mutant
    def func(x):
        return x

    original = [1, 2, 3]
    result = func(original)
    assert result == [1, 2, 3]
    assert original == [1, 2, 3]
    assert result is not original


# LLM-generated content at query #34
#--------------------------

```python
def test_freeze_with_defaultdict_and_strict_true():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    dd = defaultdict(list, {'a': [1, 2]})
    result = freeze(dd, strict=True)
    assert isinstance(result, type(pmap()))
    assert result['a'] == [1, 2]

def test_freeze_with_pmap_and_strict_true():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert isinstance(result, type(pmap()))
    assert result['a'] == [1, 2]

def test_freeze_with_defaultdict_and_strict_false():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    dd = defaultdict(list, {'a': [1, 2]})
    result = freeze(dd, strict=False)
    assert isinstance(result, type(pmap()))
    assert result['a'] == [1, 2]

def test_freeze_with_pmap_and_strict_false():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=False)
    assert result is pm


# LLM-generated content at query #35
#--------------------------

def test_mutant_with_list_argument():
    mutable_list = [1, 2, 3]
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst
    result = modify_list(mutable_list)
    assert mutable_list == [1, 2, 3]
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_dict_argument():
    mutable_dict = {'a': 1, 'b': 2}
    @mutant
    def modify_dict(d):
        d['c'] = 3
        return d
    result = modify_dict(mutable_dict)
    assert mutable_dict == {'a': 1, 'b': 2}
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_with_set_argument():
    mutable_set = {1, 2, 3}
    @mutant
    def modify_set(s):
        s.add(4)
        return s
    result = modify_set(mutable_set)
    assert mutable_set == {1, 2, 3}
    assert result == pset([1, 2, 3, 4])

def test_mutant_with_multiple_arguments():
    @mutant
    def combine(a, b):
        a.append(b)
        return a
    list_arg = [1, 2]
    result = combine(list_arg, 3)
    assert list_arg == [1, 2]
    assert result == pvector([1, 2, 3])

def test_mutant_with_keyword_arguments():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    dict_arg = {'x': 10}
    result = update_dict(dict_arg, key='y', value=20)
    assert dict_arg == {'x': 10}
    assert result == pmap({'x': 10, 'y': 20})

def test_mutant_returns_frozen_result():
    @mutant
    def return_mutable():
        return [1, 2, 3]
    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_nested_mutable_structures():
    mutable_nested = {'a': [1, 2], 'b': {'c': 3}}
    @mutant
    def modify_nested(d):
        d['a'].append(3)
        d['b']['d'] = 4
        return d
    result = modify_nested(mutable_nested)
    assert mutable_nested == {'a': [1, 2], 'b': {'c': 3}}
    assert result == pmap({'a': pvector([1, 2, 3]), 'b': pmap({'c': 3, 'd': 4})})

def test_mutant_preserves_function_metadata():
    def original_func(x):
        """Original docstring"""
        return x
    decorated_func = mutant(original_func)
    assert decorated_func.__name__ == 'original_func'
    assert decorated_func.__doc__ == 'Original docstring'

def test_mutant_with_empty_arguments():
    @mutant
    def no_op():
        return []
    result = no_op()
    assert result == pvector([])

def test_mutant_with_tuple_argument():
    mutable_list = [1, 2]
    @mutant
    def process_tuple(t):
        lst = list(t)
        lst.append(3)
        return tuple(lst)
    result = process_tuple((mutable_list,))
    assert mutable_list == [1, 2]
    assert result == (pvector([1, 2]), 3)


# LLM-generated content at query #36
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


# LLM-generated content at query #37
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    def mutable_function(lst, dct):
        lst.append(4)
        dct['new'] = 'value'
        return [lst, dct]
    decorated = mutant(mutable_function)
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
    assert result[1] == pmap({'a': 1, 'new': 'value'})

def test_mutant_decorator_with_keyword_arguments():
    def mutable_function(x, y=[]):
        y.append(x)
        return y
    decorated = mutant(mutable_function)
    result = decorated(1, y=[0])
    assert isinstance(result, PVector)
    assert result == pvector([0, 1])

def test_mutant_decorator_preserves_function_metadata():
    def example_func(a, b):
        """Example docstring."""
        return a + b
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_no_mutation():
    def pure_function(a, b):
        return a + b
    decorated = mutant(pure_function)
    result = decorated(1, 2)
    assert result == 3

def test_mutant_decorator_freezes_nested_structures():
    def func(data):
        data['list'][0] = 'changed'
        return data
    decorated = mutant(func)
    input_data = {'list': ['original'], 'tuple': (1, [2, 3])}
    result = decorated(input_data)
    assert input_data == {'list': ['original'], 'tuple': (1, [2, 3])}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert result['list'] == pvector(['changed'])
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['tuple'][1], PVector)
    assert result['tuple'][1] == pvector([2, 3])


# LLM-generated content at query #38
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
    result = freeze({'a': [1, 2], 'b': (3, {4, 5})})
    expected = pmap({'a': pvector([1, 2]), 'b': (3, pset({4, 5}))})
    assert result == expected


def test_freeze_defaultdict():
    d = collections.defaultdict(list, {'a': [1, 2]})
    result = freeze(d)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_strict_false():
    result = freeze([1, {'a': 2}], strict=False)
    expected = pvector([1, {'a': 2}])
    assert result == expected


def test_freeze_pmap_strict():
    m = pmap({'x': [1, 2]})
    result = freeze(m, strict=True)
    expected = pmap({'x': pvector([1, 2])})
    assert result == expected


def test_freeze_pvector_strict():
    v = pvector([1, {'a': 2}])
    result = freeze(v, strict=True)
    expected = pvector([1, pmap({'a': 2})])
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    result = freeze("hello")
    assert result == "hello"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
    result = thaw((1, 2, 3))
    expected = (1, 2, 3)
    assert result == expected

def test_thaw_tuple_nested():
    from pyrsistent import v
    result = thaw((1, v(2, 3)))
    expected = (1, [2, 3])
    assert result == expected

def test_thaw_strict_false_list():
    from pyrsistent import v
    result = thaw(v(1, 2, 3), strict=False)
    expected = [1, 2, 3]
    assert result == expected

def test_thaw_strict_false_dict():
    from pyrsistent import m, v
    result = thaw(m(a=v(1, 2)), strict=False)
    expected = {'a': v(1, 2)}
    assert result == expected

def test_thaw_strict_false_tuple():
    from pyrsistent import v
    result = thaw((1, v(2, 3)), strict=False)
    expected = (1, v(2, 3))
    assert result == expected

def test_thaw_non_container():
    result = thaw(42)
    expected = 42
    assert result == expected

def test_thaw_string():
    result = thaw("hello")
    expected = "hello"
    assert result == expected

def test_thaw_list_strict_true():
    result = thaw([1, 2, 3], strict=True)
    expected = [1, 2, 3]
    assert result == expected

def test_thaw_dict_strict_true():
    result = thaw({'a': 1, 'b': 2}, strict=True)
    expected = {'a': 1, 'b': 2}
    assert result == expected

def test_thaw_nested_list_strict_true():
    from pyrsistent import v
    result = thaw([v(1, 2)], strict=True)
    expected = [[1, 2]]
    assert result == expected

def test_thaw_nested_dict_strict_true():
    from pyrsistent import m
    result = thaw({'a': m(b=1)}, strict=True)
    expected = {'a': {'b': 1}}
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


def test_freeze_dict_nested():
    result = freeze({'x': {'y': 5}})
    expected = pmap({'x': pmap({'y': 5})})
    assert result == expected


def test_freeze_list_empty():
    result = freeze([])
    expected = pvector([])
    assert result == expected


def test_freeze_list_with_elements():
    result = freeze([1, 2, 3])
    expected = pvector([1, 2, 3])
    assert result == expected


def test_freeze_list_nested():
    result = freeze([{'a': 1}, [2, 3]])
    expected = pvector([pmap({'a': 1}), pvector([2, 3])])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, [2, 3]))
    expected = (1, pvector([2, 3]))
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
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_strict_true_with_pvector():
    pv = pvector([1, [2, 3]])
    result = freeze(pv, strict=True)
    expected = pvector([1, pvector([2, 3])])
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


# LLM-generated content at query #3
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
    result = freeze([1, {'a': 2}, [3, 4]])
    expected = pvector([1, pmap({'a': 2}), pvector([3, 4])])
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


# LLM-generated content at query #4
#--------------------------

def test_mutant_decorator_freezes_args_and_return():
    def add_to_list(lst, value):
        lst.append(value)
        return lst
    decorated = mutant(add_to_list)
    original = [1, 2]
    result = decorated(original, 3)
    assert original == [1, 2]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]

def test_mutant_decorator_freezes_kwargs():
    def update_dict(d, key, val):
        d[key] = val
        return d
    decorated = mutant(update_dict)
    original = {'a': 1}
    result = decorated(original, key='b', val=2)
    assert original == {'a': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_decorator_with_mixed_args():
    def modify_collections(lst, d, s, t):
        lst.append(4)
        d['new'] = 'entry'
        s.add(5)
        return (lst, d, s, t)
    decorated = mutant(modify_collections)
    lst_arg = [1, 2, 3]
    dict_arg = {'old': 'value'}
    set_arg = {1, 2, 3}
    tuple_arg = (10, 20)
    result = decorated(lst_arg, dict_arg, set_arg, tuple_arg)
    assert lst_arg == [1, 2, 3]
    assert dict_arg == {'old': 'value'}
    assert set_arg == {1, 2, 3}
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert list(result[0]) == [1, 2, 3, 4]
    assert isinstance(result[1], PMap)
    assert dict(result[1]) == {'old': 'value', 'new': 'entry'}
    assert isinstance(result[2], PSet)
    assert set(result[2]) == {1, 2, 3, 5}
    assert isinstance(result[3], tuple)
    assert result[3] == (10, 20)

def test_mutant_decorator_preserves_function_metadata():
    def example_func(x):
        """Example docstring."""
        return x
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_non_container_args():
    def add_numbers(a, b):
        return a + b
    decorated = mutant(add_numbers)
    result = decorated(5, 10)
    assert result == 15

def test_mutant_decorator_nested_freezing():
    def process(data):
        data['list'][0] = 'modified'
        return data
    decorated = mutant(process)
    original = {'list': ['original'], 'tuple': (1, [2, 3])}
    result = decorated(original)
    assert original == {'list': ['original'], 'tuple': (1, [2, 3])}
    assert isinstance(result, PMap)
    inner_list = result['list']
    assert isinstance(inner_list, PVector)
    assert list(inner_list) == ['modified']
    inner_tuple = result['tuple']
    assert isinstance(inner_tuple, tuple)
    assert isinstance(inner_tuple[1], PVector)
    assert list(inner_tuple[1]) == [2, 3]


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
    result = freeze({'a': [1, {2, 3}], 'b': (4, [5])})
    expected = pmap({'a': pvector([1, pset({2, 3})]), 'b': (4, pvector([5]))})
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
    d = collections.defaultdict(list, {'x': [1, 2]})
    result = freeze(d)
    expected = pmap({'x': pvector([1, 2])})
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    result = freeze("hello")
    assert result == "hello"


# LLM-generated content at query #6
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


# LLM-generated content at query #7
#--------------------------

def test_freeze_strict_pmap():
    from pyrsistent import pmap, freeze
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=True)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


# LLM-generated content at query #8
#--------------------------

def test_mutant_with_positional_arguments():
    result = mutant(lambda x, y: x + y)(1, 2)
    assert result == 3

def test_mutant_with_keyword_arguments():
    result = mutant(lambda a, b: a * b)(a=3, b=4)
    assert result == 12

def test_mutant_with_mixed_arguments():
    result = mutant(lambda x, y, z: x + y + z)(1, y=2, z=3)
    assert result == 6

def test_mutant_with_list_argument():
    result = mutant(lambda lst: lst.append(4))([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_dict_argument():
    result = mutant(lambda d: d.update({'c': 3}))({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_with_set_argument():
    result = mutant(lambda s: s.add(4))({1, 2, 3})
    assert result == pset([1, 2, 3, 4])

def test_mutant_with_tuple_argument():
    result = mutant(lambda t: t + (4,))((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_nested_structures():
    result = mutant(lambda obj: obj['a'].append(4))({'a': [1, 2, 3]})
    assert result == pmap({'a': pvector([1, 2, 3, 4])})

def test_mutant_returns_frozen_result():
    def mutable_operation(x):
        x['key'] = [1, 2, 3]
        return x
    result = mutant(mutable_operation)({})
    assert isinstance(result, PMap)
    assert isinstance(result['key'], PVector)

def test_mutant_preserves_function_metadata():
    def original_func(x):
        """Original docstring"""
        return x
    decorated = mutant(original_func)
    assert decorated.__name__ == 'original_func'
    assert decorated.__doc__ == 'Original docstring'


# LLM-generated content at query #9
#--------------------------

def test_mutant_with_list_argument():
    def append_to_list(lst):
        lst.append(1)
        return lst
    decorated = mutant(append_to_list)
    original = []
    result = decorated(original)
    assert original == []
    assert isinstance(result, PVector)
    assert list(result) == [1]

def test_mutant_with_dict_argument():
    def add_to_dict(d):
        d['new'] = 42
        return d
    decorated = mutant(add_to_dict)
    original = {}
    result = decorated(original)
    assert original == {}
    assert isinstance(result, PMap)
    assert dict(result) == {'new': 42}

def test_mutant_with_set_argument():
    def add_to_set(s):
        s.add(5)
        return s
    decorated = mutant(add_to_set)
    original = set()
    result = decorated(original)
    assert original == set()
    assert isinstance(result, PSet)
    assert set(result) == {5}

def test_mutant_with_tuple_argument():
    def modify_tuple(t):
        return t + (1,)
    decorated = mutant(modify_tuple)
    original = (0,)
    result = decorated(original)
    assert original == (0,)
    assert isinstance(result, tuple)
    assert result == (0, 1)

def test_mutant_with_keyword_arguments():
    def combine_kwargs(**kwargs):
        return kwargs
    decorated = mutant(combine_kwargs)
    result = decorated(a=1, b=2)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_with_positional_and_keyword_arguments():
    def func(pos, **kwargs):
        return (pos, kwargs)
    decorated = mutant(func)
    result = decorated([1], extra=3)
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert list(result[0]) == [1]
    assert isinstance(result[1], PMap)
    assert dict(result[1]) == {'extra': 3}

def test_mutant_returns_frozen_result():
    def return_mutable():
        return {'a': [1, 2]}
    decorated = mutant(return_mutable)
    result = decorated()
    assert isinstance(result, PMap)
    inner = result['a']
    assert isinstance(inner, PVector)
    assert list(inner) == [1, 2]

def test_mutant_preserves_function_metadata():
    def example():
        """Example docstring."""
        pass
    decorated = mutant(example)
    assert decorated.__name__ == 'example'
    assert decorated.__doc__ == 'Example docstring.'


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant_decorator_does_not_freeze_arguments_when_function_raises_exception():
    def failing_function(x):
        raise ValueError("Test exception")

    decorated = mutant(failing_function)
    original_set = pset([1, 2, 3])
    original_map = pmap({'a': 1, 'b': 2})

    try:
        decorated(original_set, original_map)
    except ValueError:
        pass

    assert original_set == pset([1, 2, 3])
    assert original_map == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #11
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

def test_freeze_with_regular_dict_and_strict_true():
    d = {'a': [1, 2]}
    result = freeze(d, strict=True)
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_freeze_with_regular_dict_and_strict_false():
    d = {'a': [1, 2]}
    result = freeze(d, strict=False)
    assert isinstance(result, PMap)
    assert result['a'] == [1, 2]


# LLM-generated content at query #12
#--------------------------

def test_mutant_with_positional_args():
    result = mutant(lambda x, y: x + y)(1, 2)
    assert result == 3

def test_mutant_with_keyword_args():
    result = mutant(lambda a, b: a * b)(a=3, b=4)
    assert result == 12

def test_mutant_with_mixed_args():
    result = mutant(lambda x, y, z: x + y - z)(5, y=3, z=2)
    assert result == 6

def test_mutant_freezes_list_arg():
    mutable_list = [1, 2, 3]
    result = mutant(lambda lst: lst.append(4) or lst)(mutable_list)
    assert result == pvector([1, 2, 3, 4])

def test_mutant_freezes_dict_arg():
    mutable_dict = {'a': 1}
    result = mutant(lambda d: d.update({'b': 2}) or d)(mutable_dict)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_freezes_set_arg():
    mutable_set = {1, 2}
    result = mutant(lambda s: s.add(3) or s)(mutable_set)
    assert result == pset([1, 2, 3])

def test_mutant_freezes_tuple_arg():
    mutable_inner = [1, 2]
    result = mutant(lambda t: t[1].append(3) or t)((10, mutable_inner))
    assert result == (10, pvector([1, 2, 3]))

def test_mutant_freezes_nested_args():
    nested = {'list': [1, 2], 'set': {3, 4}}
    result = mutant(lambda obj: obj['list'].append(5) or obj['set'].add(6) or obj)(nested)
    assert result == pmap({'list': pvector([1, 2, 5]), 'set': pset([3, 4, 6])})

def test_mutant_freezes_return_value():
    result = mutant(lambda: [1, 2, 3])()
    assert result == pvector([1, 2, 3])

def test_mutant_freezes_kwargs_values():
    result = mutant(lambda **kw: kw.update({'c': 3}) or kw)(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_preserves_function_name():
    def my_func(x):
        return x
    decorated = mutant(my_func)
    assert decorated.__name__ == 'my_func'

def test_mutant_with_no_args():
    result = mutant(lambda: 42)()
    assert result == 42

def test_mutant_with_empty_args():
    result = mutant(lambda *args, **kwargs: (args, kwargs))()
    assert result == ((), pmap({}))

def test_mutant_with_strict_false_implicitly():
    mutable = [1, [2, 3]]
    result = mutant(lambda lst: lst[1].append(4) or lst)(mutable)
    assert result == pvector([1, pvector([2, 3, 4])])


# LLM-generated content at query #13
#--------------------------

def test_mutant_decorator_predicate_false():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant
    @mutant
    def modify_map(pmap_arg):
        pmap_arg['new_key'] = 100
        return pmap_arg
    original = m(a=1)
    result = modify_map(original)
    assert result == m(a=1, new_key=100)
    assert original == m(a=1)
    @mutant
    def modify_set(pset_arg):
        pset_arg.add(999)
        return pset_arg
    original_set = s(1, 2)
    result_set = modify_set(original_set)
    assert result_set == s(1, 2, 999)
    assert original_set == s(1, 2)


# LLM-generated content at query #14
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
    def update_dict(d, key, value):
        d[key] = value
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
        """Sample function."""
        return a + b
    assert sample_func.__name__ == 'sample_func'
    assert sample_func.__doc__ == 'Sample function.'

def test_mutant_decorator_with_nested_structures():
    @mutant
    def process_data(data):
        data['list'].append(99)
        data['tuple'] = data['tuple'] + (5,)
        return data
    original_data = {'list': [1, 2], 'tuple': (3, 4)}
    result = process_data(original_data)
    assert original_data == {'list': [1, 2], 'tuple': (3, 4)}
    assert isinstance(result, PMap)
    result_dict = dict(result)
    assert isinstance(result_dict['list'], PVector)
    assert list(result_dict['list']) == [1, 2, 99]
    assert isinstance(result_dict['tuple'], tuple)
    assert result_dict['tuple'] == (3, 4, 5)

def test_mutant_decorator_with_no_mutation():
    @mutant
    def pure_function(x):
        return x * 2
    original = 5
    result = pure_function(original)
    assert original == 5
    assert result == 10

def test_mutant_decorator_with_defaultdict():
    from collections import defaultdict
    @mutant
    def default_dict_operation(dd):
        dd['new'].append(100)
        return dd
    original_dd = defaultdict(list, {'existing': [1]})
    result = default_dict_operation(original_dd)
    assert original_dd == defaultdict(list, {'existing': [1]})
    assert isinstance(result, PMap)
    assert dict(result) == {'existing': [1], 'new': [100]}


# LLM-generated content at query #15
#--------------------------

def test_mutant_with_positional_args():
    def add_one(x):
        x[0] = x[0] + 1
        return x
    decorated = mutant(add_one)
    original = [1]
    result = decorated(original)
    assert result == pvector([2])
    assert original == [1]

def test_mutant_with_keyword_args():
    def update_dict(d, key, value):
        d[key] = value
        return d
    decorated = mutant(update_dict)
    original = {'a': 1}
    result = decorated(original, key='a', value=2)
    assert result == pmap({'a': 2})
    assert original == {'a': 1}

def test_mutant_with_mixed_args():
    def modify_list_and_dict(lst, d, factor):
        lst.append(factor)
        d['factor'] = factor
        return lst, d
    decorated = mutant(modify_list_and_dict)
    lst = [1, 2]
    d = {'initial': 0}
    result_lst, result_d = decorated(lst, d, 3)
    assert result_lst == pvector([1, 2, 3])
    assert result_d == pmap({'initial': 0, 'factor': 3})
    assert lst == [1, 2]
    assert d == {'initial': 0}

def test_mutant_return_frozen():
    def return_mutable():
        return {'a': [1, 2]}
    decorated = mutant(return_mutable)
    result = decorated()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert result == pmap({'a': pvector([1, 2])})

def test_mutant_with_nested_mutables():
    def swap_items(data):
        data['list'][0], data['list'][1] = data['list'][1], data['list'][0]
        return data
    decorated = mutant(swap_items)
    original = {'list': [1, 2], 'set': {3, 4}}
    result = decorated(original)
    assert result == pmap({'list': pvector([2, 1]), 'set': pset({3, 4})})
    assert original == {'list': [1, 2], 'set': {3, 4}}

def test_mutant_preserves_function_metadata():
    def example_func(x):
        """Example docstring."""
        return x
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'


# LLM-generated content at query #16
#--------------------------

def test_freeze_pmap_strict():
    from pyrsistent import pmap, freeze
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=True)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


# LLM-generated content at query #17
#--------------------------

def test_mutant_decorator_predicate_false():
    from pyrsistent import freeze
    from functools import wraps
    def mutant(fn):
        @wraps(fn)
        def inner_f(*args, **kwargs):
            return freeze(fn(*[freeze(e) for e in args], **dict(freeze(item) for item in kwargs.items())))
        return inner_f
    @mutant
    def add_one(x):
        return x + 1
    result = add_one(1)
    assert result == 2


# LLM-generated content at query #18
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
    expected = pvector([1, pmap({'a': 2})])
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
    result = freeze("hello")
    expected = "hello"
    assert result == expected


def test_freeze_pvector_strict():
    v = pvector([1, 2])
    result = freeze(v, strict=True)
    expected = pvector([1, 2])
    assert result == expected


def test_freeze_pmap_strict():
    m = pmap({'a': 1})
    result = freeze(m, strict=True)
    expected = pmap({'a': 1})
    assert result == expected


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_decorator_does_not_freeze_output_when_fn_returns_non_freezable():
    def fn():
        return object()
    decorated = mutant(fn)
    result = decorated()
    assert not isinstance(result, (PMap, PVector, PSet))


# LLM-generated content at query #20
#--------------------------

def test_mutant_with_positional_args():
    result = mutant(lambda x: [x])([1, 2, 3])
    assert isinstance(result, PVector)
    assert result == pvector([pvector([1, 2, 3])])

def test_mutant_with_keyword_args():
    result = mutant(lambda x=[]: x.append(1) or x)(x=[2])
    assert isinstance(result, PVector)
    assert result == pvector([2, 1])

def test_mutant_with_multiple_args():
    result = mutant(lambda x, y: {'a': x, 'b': y})([1], {'key': 'value'})
    assert isinstance(result, PMap)
    assert result == pmap({'a': pvector([1]), 'b': pmap({'key': 'value'})})

def test_mutant_with_no_args():
    result = mutant(lambda: {'a': [1]})()
    assert isinstance(result, PMap)
    assert result == pmap({'a': pvector([1])})

def test_mutant_with_mixed_args():
    result = mutant(lambda a, b=[]: a + b)([1], b=[2])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2])

def test_mutant_preserves_original_input():
    original = {'x': [1, 2]}
    mutant(lambda d: d['x'].append(3) or d)(original)
    assert original == {'x': [1, 2]}

def test_mutant_with_set():
    result = mutant(lambda s: s.union({4}))({1, 2, 3})
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})

def test_mutant_with_tuple():
    result = mutant(lambda t: t + (4,))((1, 2, 3))
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)

def test_mutant_with_nested_structures():
    result = mutant(lambda d: d['a'].append(3) or d)({'a': [1, 2]})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert result == pmap({'a': pvector([1, 2, 3])})

def test_mutant_with_strict_false():
    original = pvector([1, 2])
    result = mutant(lambda x: x.append(3) or x, strict=False)(original)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #21
#--------------------------

def test_mutant_with_positional_args():
    result = mutant(lambda x, y: x + y)(1, 2)
    assert result == 3

def test_mutant_with_keyword_args():
    result = mutant(lambda a, b: a * b)(a=3, b=4)
    assert result == 12

def test_mutant_with_mixed_args():
    result = mutant(lambda x, y, z: x + y + z)(1, y=2, z=3)
    assert result == 6

def test_mutant_with_list_arg():
    result = mutant(lambda lst: lst.append(4))([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_dict_arg():
    result = mutant(lambda d: d.update({'c': 3}))({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_with_set_arg():
    result = mutant(lambda s: s.add(4))({1, 2, 3})
    assert result == pset([1, 2, 3, 4])

def test_mutant_with_nested_structures():
    result = mutant(lambda x: x[0].append(3))([[1, 2], {'a': 1}])
    assert result == pvector([pvector([1, 2, 3]), pmap({'a': 1})])

def test_mutant_returns_frozen_result():
    result = mutant(lambda: [1, 2, 3])()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_preserves_function_name():
    def my_func(x):
        return x
    decorated = mutant(my_func)
    assert decorated.__name__ == 'my_func'

def test_mutant_with_empty_args():
    result = mutant(lambda: 42)()
    assert result == 42

def test_mutant_with_no_return():
    result = mutant(lambda x: None)(5)
    assert result is None

def test_mutant_with_tuple_arg():
    result = mutant(lambda t: t + (4,))((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_freezes_kwargs_dict_values():
    result = mutant(lambda **kwargs: kwargs['x'].append(1))(x=[0])
    assert result == pmap({'x': pvector([0, 1])})


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pset, pmap, freeze
    from pyrsistent._helpers import mutant

    @mutant
    def modify_set(s):
        s.add(4)
        return s

    @mutant
    def modify_map(m):
        m['new_key'] = 'new_value'
        return m

    initial_set = pset([1, 2, 3])
    initial_map = pmap({'a': 1, 'b': 2})

    result_set = modify_set(initial_set)
    result_map = modify_map(initial_map)

    assert initial_set == pset([1, 2, 3])
    assert result_set == pset([1, 2, 3, 4])
    assert initial_map == pmap({'a': 1, 'b': 2})
    assert result_map == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert isinstance(result_set, type(freeze(pset())))
    assert isinstance(result_map, type(freeze(pmap())))


# LLM-generated content at query #23
#--------------------------

def test_mutant_decorator_does_not_mutate_inputs():
    from pyrsistent import pset, pmap, freeze
    from pyrsistent._helpers import mutant

    @mutant
    def modify_set(s):
        s.add(4)
        return s

    @mutant
    def modify_map(m):
        m['new'] = 100
        return m

    original_set = pset([1, 2, 3])
    original_map = pmap({'a': 1, 'b': 2})
    result_set = modify_set(original_set)
    result_map = modify_map(original_map)
    assert original_set == pset([1, 2, 3])
    assert original_map == pmap({'a': 1, 'b': 2})
    assert result_set == pset([1, 2, 3, 4])
    assert result_map == pmap({'a': 1, 'b': 2, 'new': 100})


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_decorator_does_not_freeze_arguments_when_no_mutation():
    from pyrsistent import freeze, m, s

    @mutant
    def func(pmap_arg, pset_arg):
        return pmap_arg, pset_arg

    input_pmap = m(a=1)
    input_pset = s(1, 2)
    result_pmap, result_pset = func(input_pmap, input_pset)
    assert result_pmap is input_pmap
    assert result_pset is input_pset


# LLM-generated content at query #26
#--------------------------

def test_mutant_decorator_does_not_mutate_inputs():
    from pyrsistent import freeze, pset, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def modify_set(s):
        s.add(4)
        return s

    @mutant
    def modify_map(m):
        m['new'] = 99
        return m

    original_set = pset([1, 2, 3])
    original_map = pmap({'a': 1})
    result_set = modify_set(original_set)
    result_map = modify_map(original_map)
    assert original_set == pset([1, 2, 3])
    assert original_map == pmap({'a': 1})
    assert result_set == pset([1, 2, 3, 4])
    assert result_map == pmap({'a': 1, 'new': 99})


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_decorator_does_not_mutate_inputs():
    original = [1, 2, 3]
    frozen_copy = freeze(original)
    assert original == [1, 2, 3]
    assert original is not frozen_copy
    assert not isinstance(original, type(frozen_copy))


# LLM-generated content at query #28
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
    original = {'a': 1}
    result = decorated(original, key='a', value=2)
    assert original == {'a': 1}
    assert result == pmap({'a': 2})

def test_mutant_with_mixed_arguments():
    def modify_list_and_dict(lst, d, factor):
        lst.append(factor)
        d['factor'] = factor
        return lst, d
    decorated = mutant(modify_list_and_dict)
    lst = [1, 2]
    d = {'initial': 0}
    result_lst, result_d = decorated(lst, d, factor=3)
    assert lst == [1, 2]
    assert d == {'initial': 0}
    assert result_lst == pvector([1, 2, 3])
    assert result_d == pmap({'initial': 0, 'factor': 3})

def test_mutant_returns_frozen_result():
    def return_mutable():
        return [1, {'a': 2}]
    decorated = mutant(return_mutable)
    result = decorated()
    assert result == pvector([1, pmap({'a': 2})])

def test_mutant_preserves_function_metadata():
    def example_func(x):
        """Example docstring."""
        return x
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_with_no_arguments():
    def constant():
        return {'key': 'value'}
    decorated = mutant(constant)
    result = decorated()
    assert result == pmap({'key': 'value'})

def test_mutant_with_nested_mutables():
    def swap_keys(d):
        d['a'], d['b'] = d['b'], d['a']
        return d
    decorated = mutant(swap_keys)
    original = {'a': 1, 'b': 2}
    result = decorated(original)
    assert original == {'a': 1, 'b': 2}
    assert result == pmap({'a': 2, 'b': 1})

def test_mutant_with_set_argument():
    def add_to_set(s, element):
        s.add(element)
        return s
    decorated = mutant(add_to_set)
    original = {1, 2}
    result = decorated(original, 3)
    assert original == {1, 2}
    assert result == pset({1, 2, 3})

def test_mutant_with_tuple_argument():
    def modify_tuple(t):
        return list(t)
    decorated = mutant(modify_tuple)
    original = (1, [2, 3])
    result = decorated(original)
    assert original == (1, [2, 3])
    assert result == pvector([1, pvector([2, 3])])

def test_mutant_freezes_kwargs_values():
    def update_from_kwargs(d, **kwargs):
        d.update(kwargs)
        return d
    decorated = mutant(update_from_kwargs)
    original = {'x': 1}
    result = decorated(original, y=[2, 3])
    assert original == {'x': 1}
    assert result == pmap({'x': 1, 'y': pvector([2, 3])})


# LLM-generated content at query #29
#--------------------------

```python
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
    assert isinstance(result_map, type(freeze({})))
    assert isinstance(result_set, type(freeze(set())))


# LLM-generated content at query #30
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    def mutable_func(lst, dct):
        lst.append(4)
        dct['new'] = 'value'
        return [lst, dct]
    decorated = mutant(mutable_func)
    original_list = [1, 2, 3]
    original_dict = {'a': 1}
    result = decorated(original_list, original_dict)
    assert original_list == [1, 2, 3]
    assert original_dict == {'a': 1}
    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)
    assert result[0] == pvector([1, 2, 3, 4])
    assert isinstance(result[1], PMap)
    assert result[1] == pmap({'a': 1, 'new': 'value'})

def test_mutant_decorator_with_keyword_arguments():
    def mutable_func(x, y=[]):
        y.append(x)
        return y
    decorated = mutant(mutable_func)
    result = decorated(1, y=[0])
    assert result == pvector([0, 1])

def test_mutant_decorator_preserves_function_metadata():
    def example(a, b):
        """Example docstring."""
        return a + b
    decorated = mutant(example)
    assert decorated.__name__ == 'example'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_no_arguments():
    def constant():
        return {'key': [1, 2, 3]}
    decorated = mutant(constant)
    result = decorated()
    assert isinstance(result, PMap)
    assert isinstance(result['key'], PVector)
    assert result['key'] == pvector([1, 2, 3])

def test_mutant_decorator_freezes_nested_structures():
    def func(data):
        data['list'][0] = 'changed'
        return data
    decorated = mutant(func)
    input_data = {'list': ['original'], 'set': {1, 2}}
    result = decorated(input_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert result['list'][0] == 'changed'
    assert isinstance(result['set'], PSet)
    assert result['set'] == pset({1, 2})


# LLM-generated content at query #31
#--------------------------

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


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def modify_inputs_and_return_new(x, y):
        x['key'] = 'new_value'
        y.add(42)
        return {'modified': True, 'x': x, 'y': y}

    original_dict = {'a': 1}
    original_set = {1, 2, 3}
    result = modify_inputs_and_return_new(original_dict, original_set)

    assert original_dict == {'a': 1}
    assert original_set == {1, 2, 3}
    assert isinstance(result, type(freeze({})))
    assert result['modified'] is True
    assert isinstance(result['x'], type(freeze({})))
    assert isinstance(result['y'], type(freeze(set())))
    assert result['x']['key'] == 'new_value'
    assert 42 in result['y']


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant_decorator_does_not_freeze_arguments_when_no_mutation():
    def add(a, b):
        return a + b

    decorated_add = mutant(add)
    result = decorated_add(1, 2)
    assert result == 3


# LLM-generated content at query #34
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, pset, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def modify_data(s, m):
        s.add(4)
        m['c'] = 15
        return {'modified': True, 's': s, 'm': m}

    original_set = pset([1, 2, 3])
    original_map = pmap({'a': 13, 'b': 14})
    result = modify_data(original_set, original_map)

    assert original_set == pset([1, 2, 3])
    assert original_map == pmap({'a': 13, 'b': 14})
    assert isinstance(result, type(freeze({})))
    assert result['modified'] is True
    assert isinstance(result['s'], type(freeze(set())))
    assert isinstance(result['m'], type(freeze({})))
    assert result['s'] == pset([1, 2, 3, 4])
    assert result['m'] == pmap({'a': 13, 'b': 14, 'c': 15})


# LLM-generated content at query #35
#--------------------------

def test_mutant_decorator_freezes_args_and_return():
    def add_one(x):
        x[0] += 1
        return x
    decorated = mutant(add_one)
    original = [1, 2, 3]
    result = decorated(original)
    assert original == [1, 2, 3]
    assert isinstance(result, PVector)
    assert result == pvector([2, 2, 3])

def test_mutant_decorator_with_kwargs():
    def update_dict(d, key, value):
        d[key] = value
        return d
    decorated = mutant(update_dict)
    original = {'a': 1}
    result = decorated(original, key='b', value=2)
    assert original == {'a': 1}
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_decorator_preserves_function_metadata():
    def example_func(x):
        """Example docstring."""
        return x
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_multiple_args():
    def concat_and_modify(a, b):
        a.append(99)
        b.add(100)
        return (a, b)
    decorated = mutant(concat_and_modify)
    list_arg = [1, 2]
    set_arg = {3, 4}
    result = decorated(list_arg, set_arg)
    assert list_arg == [1, 2]
    assert set_arg == {3, 4}
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PSet)
    assert result[0] == pvector([1, 2, 99])
    assert result[1] == pset({3, 4, 100})

def test_mutant_decorator_nested_structures():
    def modify_nested(obj):
        obj['list'][0] = 'changed'
        obj['set'].add(999)
        return obj
    decorated = mutant(modify_nested)
    original = {'list': ['original'], 'set': {1}}
    result = decorated(original)
    assert original == {'list': ['original'], 'set': {1}}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)
    assert result['list'] == pvector(['changed'])
    assert result['set'] == pset({1, 999})

def test_mutant_decorator_with_strict_false_implicitly():
    def func(x):
        return x
    decorated = mutant(func)
    pmap_arg = pmap({'a': [1, 2]})
    result = decorated(pmap_arg)
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)

def test_mutant_decorator_no_side_effects_on_frozen_inputs():
    def attempt_mutation(x):
        if isinstance(x, dict):
            x['key'] = 'mutated'
        return x
    decorated = mutant(attempt_mutation)
    frozen_input = freeze({'key': 'original'})
    result = decorated(frozen_input)
    assert frozen_input == pmap({'key': 'original'})
    assert result == pmap({'key': 'original'})


# LLM-generated content at query #36
#--------------------------

def test_freeze_defaultdict_strict_true():
    from collections import defaultdict
    dd = defaultdict(list, {'a': [1, 2]})
    result = freeze(dd, strict=True)
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_freeze_defaultdict_strict_false():
    from collections import defaultdict
    dd = defaultdict(list, {'a': [1, 2]})
    result = freeze(dd, strict=False)
    assert isinstance(result, PMap)
    assert result['a'] == [1, 2]

def test_freeze_pmap_strict_true():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_freeze_pmap_strict_false():
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=False)
    assert result is pm


# LLM-generated content at query #37
#--------------------------

def test_mutant_decorator_freezes_inputs_and_output():
    def mutable_func(lst, dct):
        lst.append(4)
        dct['new'] = 'value'
        return {'modified': lst, 'updated': dct}
    decorated_func = mutant(mutable_func)
    original_list = [1, 2, 3]
    original_dict = {'key': 'old'}
    result = decorated_func(original_list, original_dict)
    assert original_list == [1, 2, 3]
    assert original_dict == {'key': 'old'}
    assert isinstance(result, PMap)
    assert result['modified'] == pvector([1, 2, 3, 4])
    assert result['updated'] == pmap({'key': 'old', 'new': 'value'})

def test_mutant_decorator_with_kwargs():
    def mutable_func(a, b=[]):
        b.append(a)
        return b
    decorated_func = mutant(mutable_func)
    result = decorated_func(1, b=[0])
    assert result == pvector([0, 1])

def test_mutant_decorator_returns_frozen_result():
    def func():
        return [1, {'a': 2}]
    decorated_func = mutant(func)
    result = decorated_func()
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)

def test_mutant_decorator_handles_no_arguments():
    def func():
        return {'set': {1, 2}}
    decorated_func = mutant(func)
    result = decorated_func()
    assert isinstance(result, PMap)
    assert isinstance(result['set'], PSet)

def test_mutant_decorator_freezes_nested_structures():
    def func(data):
        data['list'][0] = 'changed'
        return data
    decorated_func = mutant(func)
    original = {'list': ['original'], 'tuple': (1, [2])}
    result = decorated_func(original)
    assert original == {'list': ['original'], 'tuple': (1, [2])}
    assert isinstance(result, PMap)
    assert result['list'] == pvector(['changed'])
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['tuple'][1], PVector)


# LLM-generated content at query #38
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant

    @mutant
    def modify_map(m):
        m['new_key'] = 'new_value'
        return m

    @mutant
    def modify_set(s):
        s.add(999)
        return s

    original_map = pmap({'a': 1, 'b': 2})
    original_set = pset([1, 2, 3])

    result_map = modify_map(original_map)
    result_set = modify_set(original_set)

    assert original_map == pmap({'a': 1, 'b': 2})
    assert original_set == pset([1, 2, 3])
    assert result_map == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert result_set == pset([1, 2, 3, 999])
    assert isinstance(result_map, type(freeze({})))
    assert isinstance(result_set, type(freeze(set())))


