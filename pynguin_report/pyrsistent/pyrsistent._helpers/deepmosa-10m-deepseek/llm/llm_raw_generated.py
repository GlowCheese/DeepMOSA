####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_mutant_with_positional_arguments():
    def add_one(x):
        x[0] = x[0] + 1
        return x
    decorated = mutant(add_one)
    original = [1]
    result = decorated(original)
    assert original == [1]
    assert result == pvector([2])

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
        return {'a': [1, 2]}
    decorated = mutant(return_mutable)
    result = decorated()
    assert result == pmap({'a': pvector([1, 2])})

def test_mutant_preserves_function_metadata():
    def example_func(x):
        """Example docstring."""
        return x
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_with_no_arguments():
    def constant():
        return [1, 2, 3]
    decorated = mutant(constant)
    result = decorated()
    assert result == pvector([1, 2, 3])

def test_mutant_with_nested_mutables():
    def swap_keys(d):
        d['inner'] = {'old': d['inner']['new']}
        return d
    decorated = mutant(swap_keys)
    original = {'inner': {'new': 1, 'old': 2}}
    result = decorated(original)
    assert original == {'inner': {'new': 1, 'old': 2}}
    assert result == pmap({'inner': pmap({'old': 1})})

def test_mutant_freezes_arguments_recursively():
    def access_inner(lst):
        return lst[0]
    decorated = mutant(access_inner)
    original = [[1, 2], [3, 4]]
    result = decorated(original)
    assert original == [[1, 2], [3, 4]]
    assert result == pvector([1, 2])

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


# LLM-generated content at query #2
#--------------------------

def test_mutant_decorator_freezes_args_and_return():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]

def test_mutant_decorator_freezes_kwargs():
    @mutant
    def update_dict(d, key, val):
        d[key] = val
        return d
    original_dict = {'a': 1}
    result = update_dict(original_dict, key='b', val=2)
    assert original_dict == {'a': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_decorator_freezes_nested_structures():
    @mutant
    def modify_nested(data):
        data['list'][0] = 99
        data['tuple'] = (5, 6)
        return data
    original = {'list': [1, 2], 'tuple': (3, 4)}
    result = modify_nested(original)
    assert original == {'list': [1, 2], 'tuple': (3, 4)}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert list(result['list']) == [1, 2]
    assert result['tuple'] == (3, 4)

def test_mutant_decorator_with_set():
    @mutant
    def add_to_set(s, element):
        s.add(element)
        return s
    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)
    assert original_set == {1, 2, 3}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}

def test_mutant_decorator_preserves_function_metadata():
    @mutant
    def example_func(x):
        """Example docstring."""
        return x
    assert example_func.__name__ == 'example_func'
    assert example_func.__doc__ == "Example docstring."

def test_mutant_decorator_with_strict_false_implicitly():
    @mutant
    def identity(obj):
        return obj
    pvec = pvector([1, 2])
    pmap_obj = pmap({'a': 3})
    result_vec = identity(pvec)
    result_map = identity(pmap_obj)
    assert result_vec is pvec
    assert result_map is pmap_obj

def test_mutant_decorator_freezes_defaultdict():
    @mutant
    def use_defaultdict(dd):
        dd['new'] = 100
        return dd
    original_dd = collections.defaultdict(int, {'a': 1})
    result = use_defaultdict(original_dd)
    assert original_dd == collections.defaultdict(int, {'a': 1})
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'new': 100}

def test_mutant_decorator_with_multiple_args():
    @mutant
    def combine(a, b, c):
        return [a, b, c]
    arg1 = [1, 2]
    arg2 = {'x': 10}
    arg3 = {3, 4}
    result = combine(arg1, arg2, arg3)
    assert arg1 == [1, 2]
    assert arg2 == {'x': 10}
    assert arg3 == {3, 4}
    assert isinstance(result, PVector)
    assert list(result[0]) == [1, 2]
    assert isinstance(result[1], PMap)
    assert dict(result[1]) == {'x': 10}
    assert isinstance(result[2], PSet)
    assert set(result[2]) == {3, 4}

def test_mutant_decorator_returns_non_container_unchanged():
    @mutant
    def return_non_container(x):
        return x
    assert return_non_container(42) == 42
    assert return_non_container("hello") == "hello"
    assert return_non_container(None) is None

def test_mutant_decorator_freezes_empty_structures():
    @mutant
    def return_empty(obj):
        return obj
    empty_list = []
    empty_dict = {}
    empty_set = set()
    result_list = return_empty(empty_list)
    result_dict = return_empty(empty_dict)
    result_set = return_empty(empty_set)
    assert isinstance(result_list, PVector)
    assert len(result_list) == 0
    assert isinstance(result_dict, PMap)
    assert len(result_dict) == 0
    assert isinstance(result_set, PSet)
    assert len(result_set) == 0


# LLM-generated content at query #3
#--------------------------

def test_freeze_empty_list():
    result = freeze([])
    assert result == pvector()


def test_freeze_list_with_ints():
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
    result = freeze({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_dict_with_list_values():
    result = freeze({'a': [1, 2]})
    assert result == pmap({'a': pvector([1, 2])})


def test_freeze_nested_dict():
    result = freeze({'a': {'b': 1}})
    assert result == pmap({'a': pmap({'b': 1})})


def test_freeze_empty_set():
    result = freeze(set())
    assert result == pset()


def test_freeze_set_with_ints():
    result = freeze({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_freeze_empty_tuple():
    result = freeze(())
    assert result == ()


def test_freeze_tuple_with_ints():
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)


def test_freeze_tuple_with_list():
    result = freeze((1, [2, 3]))
    assert result == (1, pvector([2, 3]))


def test_freeze_nested_tuple():
    result = freeze((1, (2, [3])))
    assert result == (1, (2, pvector([3])))


def test_freeze_defaultdict():
    d = collections.defaultdict(list)
    d['a'].append(1)
    result = freeze(d)
    assert result == pmap({'a': pvector([1])})


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    result = freeze('hello')
    assert result == 'hello'


def test_freeze_strict_false_with_pmap():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=False)
    assert result == m


def test_freeze_strict_false_with_pvector():
    v = pvector([1, 2, 3])
    result = freeze(v, strict=False)
    assert result == v


def test_freeze_strict_true_with_pmap():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=True)
    assert result == pmap({'a': pvector([1, 2])})


def test_freeze_strict_true_with_pvector():
    v = pvector([1, [2, 3]])
    result = freeze(v, strict=True)
    assert result == pvector([1, pvector([2, 3])])


# LLM-generated content at query #4
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    def mutable_func(lst, dct):
        lst.append(4)
        dct['new'] = 'value'
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
    assert result[1] == pmap({'a': 1, 'new': 'value'})

def test_mutant_decorator_with_keyword_arguments():
    def mutable_func(a, b=[]):
        b.append(a)
        return b
    decorated = mutant(mutable_func)
    result = decorated(1, b=[0])
    assert isinstance(result, PVector)
    assert result == pvector([0, 1])

def test_mutant_decorator_preserves_function_metadata():
    def original(a, b):
        """Original docstring."""
        return a + b
    decorated = mutant(original)
    assert decorated.__name__ == 'original'
    assert decorated.__doc__ == 'Original docstring.'

def test_mutant_decorator_with_no_mutation():
    def pure_func(x, y):
        return x * y
    decorated = mutant(pure_func)
    result = decorated(3, 4)
    assert result == 12

def test_mutant_decorator_freezes_nested_structures():
    def func(data):
        data['list'][0] = 99
        return data
    decorated = mutant(func)
    input_data = {'list': [1, 2, 3], 'tuple': (4, 5)}
    result = decorated(input_data)
    assert input_data == {'list': [1, 2, 3], 'tuple': (4, 5)}
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
    input_set = {1, 2, 3}
    result = decorated(input_set)
    assert input_set == {1, 2, 3}
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})

def test_mutant_decorator_strict_false_behavior():
    def func(pmap_obj):
        return pmap_obj.set('a', 100)
    decorated = mutant(func)
    input_pmap = pmap({'a': 1})
    result = decorated(input_pmap)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 100})


# LLM-generated content at query #5
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


# LLM-generated content at query #6
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
    result = freeze((1, {'y': 2}, [3]))
    expected = (1, pmap({'y': 2}), pvector([3]))
    assert result == expected


def test_freeze_set():
    result = freeze({1, 2, 3})
    expected = pset({1, 2, 3})
    assert result == expected


def test_freeze_nested_dict():
    result = freeze({'a': {'b': [1, 2]}})
    expected = pmap({'a': pmap({'b': pvector([1, 2])})})
    assert result == expected


def test_freeze_defaultdict():
    dd = collections.defaultdict(list, {'x': [1, 2]})
    result = freeze(dd)
    expected = pmap({'x': pvector([1, 2])})
    assert result == expected


def test_freeze_strict_false():
    result = freeze([pmap({'a': 1})], strict=False)
    expected = pvector([pmap({'a': 1})])
    assert result == expected


def test_freeze_strict_true():
    result = freeze([pmap({'a': 1})], strict=True)
    expected = pvector([pmap({'a': 1})])
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    expected = 42
    assert result == expected


def test_freeze_string():
    result = freeze("hello")
    expected = "hello"
    assert result == expected


def test_freeze_none():
    result = freeze(None)
    expected = None
    assert result == expected


# LLM-generated content at query #7
#--------------------------

def test_mutant_with_list_argument():
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    assert original == [1, 2, 3]
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_dict_argument():
    def update_dict(d, key, value):
        d[key] = value
        return d
    decorated = mutant(update_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    assert original == {'a': 1}
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_set_argument():
    def add_to_set(s, element):
        s.add(element)
        return s
    decorated = mutant(add_to_set)
    original = {1, 2}
    result = decorated(original, 3)
    assert original == {1, 2}
    assert result == pset([1, 2, 3])

def test_mutant_with_tuple_argument():
    def modify_tuple(t):
        return list(t)
    decorated = mutant(modify_tuple)
    original = (1, 2, 3)
    result = decorated(original)
    assert original == (1, 2, 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_keyword_arguments():
    def combine_dicts(d1, d2):
        d1.update(d2)
        return d1
    decorated = mutant(combine_dicts)
    result = decorated({'x': 10}, d2={'y': 20})
    assert result == pmap({'x': 10, 'y': 20})

def test_mutant_returns_frozen_result():
    def return_mutable():
        return [1, 2, 3]
    decorated = mutant(return_mutable)
    result = decorated()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_nested_structures():
    def update_nested(data):
        data['list'][0] = 99
        return data
    decorated = mutant(update_nested)
    original = {'list': [1, 2, 3], 'set': {4, 5}}
    result = decorated(original)
    assert original == {'list': [1, 2, 3], 'set': {4, 5}}
    assert result == pmap({'list': pvector([99, 2, 3]), 'set': pset([4, 5])})

def test_mutant_with_no_arguments():
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


# LLM-generated content at query #8
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
    d = collections.defaultdict(list, {'a': [1, 2]})
    result = freeze(d)
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


# LLM-generated content at query #9
#--------------------------

def test_freeze_pmap_with_strict_true():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': [1, 2], 'b': {'c': 3}})
    result = freeze(pmap_instance, strict=True)
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})
    assert result == expected

def test_freeze_defaultdict_with_strict_true():
    import collections
    from pyrsistent import pmap, freeze
    defaultdict_instance = collections.defaultdict(list, {'a': [1, 2], 'b': {'c': 3}})
    result = freeze(defaultdict_instance, strict=True)
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})
    assert result == expected


# LLM-generated content at query #10
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]

def test_mutant_decorator_with_dict_argument():
    @mutant
    def update_dict(d, key, val):
        d[key] = val
        return d
    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_decorator_with_set_argument():
    @mutant
    def add_to_set(s, element):
        s.add(element)
        return s
    original_set = {1, 2}
    result = add_to_set(original_set, 3)
    assert original_set == {1, 2}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}

def test_mutant_decorator_with_tuple_argument():
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
    def combine_kwargs(**kwargs):
        return kwargs
    result = combine_kwargs(x=10, y=20)
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 10, 'y': 20}

def test_mutant_decorator_preserves_function_metadata():
    @mutant
    def example_func(a, b):
        """Example docstring."""
        return a + b
    assert example_func.__name__ == 'example_func'
    assert example_func.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_nested_structures():
    @mutant
    def process(data):
        data['list'].append(99)
        return data
    original = {'list': [1, 2], 'set': {3, 4}}
    result = process(original)
    assert original == {'list': [1, 2], 'set': {3, 4}}
    assert isinstance(result, PMap)
    inner_list = result['list']
    assert isinstance(inner_list, PVector)
    assert list(inner_list) == [1, 2, 99]
    inner_set = result['set']
    assert isinstance(inner_set, PSet)
    assert set(inner_set) == {3, 4}

def test_mutant_decorator_with_no_arguments():
    @mutant
    def constant():
        return [1, 2, 3]
    result = constant()
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]

def test_mutant_decorator_with_mixed_arguments():
    @mutant
    def mixed(a, b=[]):
        b.append(a)
        return b
    result1 = mixed(1)
    assert isinstance(result1, PVector)
    assert list(result1) == [1]
    result2 = mixed(2, [0])
    assert isinstance(result2, PVector)
    assert list(result2) == [0, 2]

def test_mutant_decorator_freezes_returned_set():
    @mutant
    def make_set():
        return {5, 6, 7}
    result = make_set()
    assert isinstance(result, PSet)
    assert set(result) == {5, 6, 7}


# LLM-generated content at query #11
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
    result = mutant(lambda data: data['list'].append(4))({'list': [1, 2, 3], 'dict': {'x': 10}})
    expected = pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'x': 10})})
    assert result == expected

def test_mutant_with_tuple_arg():
    result = mutant(lambda t: t + (4,))((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_no_args():
    result = mutant(lambda: 42)()
    assert result == 42

def test_mutant_preserves_input_immutability():
    original_list = [1, 2, 3]
    original_dict = {'a': 1}
    result = mutant(lambda lst, dct: (lst.append(4), dct.update({'b': 2})))(original_list, original_dict)
    assert original_list == [1, 2, 3]
    assert original_dict == {'a': 1}
    assert result == (pvector([1, 2, 3, 4]), pmap({'a': 1, 'b': 2}))

def test_mutant_with_defaultdict():
    from collections import defaultdict
    dd = defaultdict(list, {'a': [1, 2]})
    result = mutant(lambda d: d['a'].append(3))(dd)
    assert result == pmap({'a': pvector([1, 2, 3])})

def test_mutant_with_pvector_arg():
    pv = pvector([1, 2, 3])
    result = mutant(lambda v: v.append(4))(pv)
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_pmap_arg():
    pm = pmap({'x': 1, 'y': 2})
    result = mutant(lambda m: m.set('z', 3))(pm)
    assert result == pmap({'x': 1, 'y': 2, 'z': 3})

def test_mutant_with_pset_arg():
    ps = pset([1, 2, 3])
    result = mutant(lambda s: s.add(4))(ps)
    assert result == pset([1, 2, 3, 4])

def test_mutant_with_function_returning_none():
    result = mutant(lambda x: None)(5)
    assert result is None

def test_mutant_with_function_modifying_multiple_args():
    result = mutant(lambda a, b: (a.append(1), b.update({'k': 'v'})))([], {})
    assert result == (pvector([1]), pmap({'k': 'v'}))

def test_mutant_with_lambda_using_star_args():
    result = mutant(lambda *args: sum(args))(1, 2, 3, 4)
    assert result == 10

def test_mutant_with_lambda_using_star_kwargs():
    result = mutant(lambda **kwargs: sum(kwargs.values()))(a=1, b=2, c=3)
    assert result == 6


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_pmap_strict_true():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': 1, 'b': 2})
    result = freeze(pmap_instance, strict=True)
    assert isinstance(result, type(pmap_instance))
    assert result == pmap_instance

def test_freeze_pmap_strict_false():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': 1, 'b': 2})
    result = freeze(pmap_instance, strict=False)
    assert result is pmap_instance

def test_freeze_dict_strict_true():
    from pyrsistent import pmap, freeze
    dict_instance = {'a': 1, 'b': 2}
    result = freeze(dict_instance, strict=True)
    assert isinstance(result, type(pmap({})))
    assert result == dict_instance

def test_freeze_dict_strict_false():
    from pyrsistent import pmap, freeze
    dict_instance = {'a': 1, 'b': 2}
    result = freeze(dict_instance, strict=False)
    assert isinstance(result, type(pmap({})))
    assert result == dict_instance

def test_freeze_pvector_strict_true():
    from pyrsistent import pvector, freeze
    pvector_instance = pvector([1, 2, 3])
    result = freeze(pvector_instance, strict=True)
    assert isinstance(result, type(pvector_instance))
    assert result == pvector_instance

def test_freeze_pvector_strict_false():
    from pyrsistent import pvector, freeze
    pvector_instance = pvector([1, 2, 3])
    result = freeze(pvector_instance, strict=False)
    assert result is pvector_instance

def test_freeze_list_strict_true():
    from pyrsistent import pvector, freeze
    list_instance = [1, 2, 3]
    result = freeze(list_instance, strict=True)
    assert isinstance(result, type(pvector([])))
    assert result == list_instance

def test_freeze_list_strict_false():
    from pyrsistent import pvector, freeze
    list_instance = [1, 2, 3]
    result = freeze(list_instance, strict=False)
    assert isinstance(result, type(pvector([])))
    assert result == list_instance


# LLM-generated content at query #13
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


# LLM-generated content at query #14
#--------------------------

```python
def test_freeze_with_strict_true_and_pmap():
    from pyrsistent import pmap, freeze
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=True)
    assert isinstance(result, type(m))
    assert result['a'] == [1, 2]


# LLM-generated content at query #15
#--------------------------

def test_freeze_empty_list():
    result = freeze([])
    assert result == pvector([])


def test_freeze_list_with_elements():
    result = freeze([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_freeze_nested_list():
    result = freeze([[1, 2], [3, 4]])
    assert result == pvector([pvector([1, 2]), pvector([3, 4])])


def test_freeze_empty_dict():
    result = freeze({})
    assert result == pmap({})


def test_freeze_dict_with_values():
    result = freeze({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_nested_dict():
    result = freeze({'a': {'b': 1}})
    assert result == pmap({'a': pmap({'b': 1})})


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


def test_freeze_nested_tuple():
    result = freeze(([1, 2], (3, 4)))
    assert result == (pvector([1, 2]), (3, 4))


def test_freeze_defaultdict():
    d = collections.defaultdict(list)
    d['a'].append(1)
    result = freeze(d)
    assert result == pmap({'a': pvector([1])})


def test_freeze_strict_false_with_pvector():
    pv = pvector([1, 2])
    result = freeze(pv, strict=False)
    assert result is pv


def test_freeze_strict_false_with_pmap():
    pm = pmap({'a': 1})
    result = freeze(pm, strict=False)
    assert result is pm


def test_freeze_strict_true_with_pvector():
    pv = pvector([1, 2])
    result = freeze(pv, strict=True)
    assert result == pvector([1, 2])


def test_freeze_strict_true_with_pmap():
    pm = pmap({'a': 1})
    result = freeze(pm, strict=True)
    assert result == pmap({'a': 1})


def test_freeze_non_container():
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    result = freeze("hello")
    assert result == "hello"


def test_freeze_nested_mixed():
    data = {'a': [1, 2], 'b': (3, {4, 5})}
    result = freeze(data)
    expected = pmap({'a': pvector([1, 2]), 'b': (3, pset([4, 5]))})
    assert result == expected


# LLM-generated content at query #16
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

def test_freeze_with_defaultdict_and_strict_false():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    dd = defaultdict(list, {'a': [1, 2], 'b': [3, 4]})
    result = freeze(dd, strict=False)
    expected = pmap({'a': [1, 2], 'b': [3, 4]})
    assert result == expected
    assert isinstance(result, type(pmap()))

def test_freeze_with_pmap_and_strict_true():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2], 'b': [3, 4]})
    result = freeze(pm, strict=True)
    expected = pmap({'a': [1, 2], 'b': [3, 4]})
    assert result == expected
    assert isinstance(result, type(pmap()))

def test_freeze_with_pmap_and_strict_false():
    from pyrsistent import pmap, freeze
    pm = pmap({'a': [1, 2], 'b': [3, 4]})
    result = freeze(pm, strict=False)
    assert result is pm


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


def test_freeze_nested_structure():
    result = freeze({'a': [1, 2], 'b': {'c': {3, 4}}})
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'c': pset({3, 4})})})
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


# LLM-generated content at query #18
#--------------------------

```python
def test_freeze_with_pmap_and_strict_true():
    from pyrsistent import freeze, pmap, PVector, PMap
    import collections
    pmap_instance = pmap({'a': [1, 2], 'b': {'c': 3}})
    result = freeze(pmap_instance, strict=True)
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)

def test_freeze_with_pmap_and_strict_false():
    from pyrsistent import freeze, pmap
    pmap_instance = pmap({'a': [1, 2], 'b': {'c': 3}})
    result = freeze(pmap_instance, strict=False)
    assert result == pmap_instance

def test_freeze_with_defaultdict_and_strict_true():
    from pyrsistent import freeze, pmap, PVector, PMap
    import collections
    dd = collections.defaultdict(list, {'a': [1, 2], 'b': {'c': 3}})
    result = freeze(dd, strict=True)
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)

def test_freeze_with_defaultdict_and_strict_false():
    from pyrsistent import freeze, pmap
    import collections
    dd = collections.defaultdict(list, {'a': [1, 2], 'b': {'c': 3}})
    result = freeze(dd, strict=False)
    assert isinstance(result, PMap)
    assert result == pmap({'a': [1, 2], 'b': {'c': 3}})


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_thaw_pvector():
    from pyrsistent import v
    result = thaw(v(1, 2, 3))
    assert result == [1, 2, 3]
    assert isinstance(result, list)

def test_thaw_pvector_recursive():
    from pyrsistent import v, m
    result = thaw(v(1, m(a=2)))
    assert result == [1, {'a': 2}]
    assert isinstance(result, list)
    assert isinstance(result[1], dict)

def test_thaw_pmap():
    from pyrsistent import m
    result = thaw(m(a=1, b=2))
    assert result == {'a': 1, 'b': 2}
    assert isinstance(result, dict)

def test_thaw_pmap_recursive():
    from pyrsistent import m, v
    result = thaw(m(a=v(1, 2)))
    assert result == {'a': [1, 2]}
    assert isinstance(result, dict)
    assert isinstance(result['a'], list)

def test_thaw_pset():
    from pyrsistent import s
    result = thaw(s(1, 2, 3))
    assert result == {1, 2, 3}
    assert isinstance(result, set)

def test_thaw_tuple():
    result = thaw((1, 2, 3))
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)

def test_thaw_tuple_recursive():
    from pyrsistent import v
    result = thaw((1, v(2, 3)))
    assert result == (1, [2, 3])
    assert isinstance(result, tuple)
    assert isinstance(result[1], list)

def test_thaw_nested_mixed():
    from pyrsistent import v, m, s
    result = thaw(v(m(a=s(1, 2)), (3, v(4))))
    assert result == [{'a': {1, 2}}, (3, [4])]
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert isinstance(result[0]['a'], set)
    assert isinstance(result[1], tuple)
    assert isinstance(result[1][1], list)

def test_thaw_non_container():
    result = thaw(42)
    assert result == 42
    result = thaw("hello")
    assert result == "hello"

def test_thaw_strict_false_pvector():
    from pyrsistent import v
    result = thaw(v(1, 2, 3), strict=False)
    assert result == [1, 2, 3]

def test_thaw_strict_false_list():
    result = thaw([1, 2, 3], strict=False)
    assert result == [1, 2, 3]
    assert isinstance(result, list)

def test_thaw_strict_false_pmap():
    from pyrsistent import m
    result = thaw(m(a=1), strict=False)
    assert result == {'a': 1}

def test_thaw_strict_false_dict():
    result = thaw({'a': 1}, strict=False)
    assert result == {'a': 1}
    assert isinstance(result, dict)

def test_thaw_strict_false_tuple():
    result = thaw((1, 2), strict=False)
    assert result == (1, 2)

def test_thaw_strict_false_pset():
    from pyrsistent import s
    result = thaw(s(1, 2), strict=False)
    assert result == {1, 2}

def test_thaw_strict_true_list_recursive():
    from pyrsistent import v
    result = thaw([v(1, 2)], strict=True)
    assert result == [[1, 2]]

def test_thaw_strict_true_dict_recursive():
    from pyrsistent import v
    result = thaw({'a': v(1, 2)}, strict=True)
    assert result == {'a': [1, 2]}

def test_thaw_strict_false_list_no_recursion():
    from pyrsistent import v
    result = thaw([v(1, 2)], strict=False)
    assert result == [v(1, 2)]

def test_thaw_strict_false_dict_no_recursion():
    from pyrsistent import v
    result = thaw({'a': v(1, 2)}, strict=False)
    assert result == {'a': v(1, 2)}


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


def test_freeze_nested_dict():
    result = freeze({'x': {'y': 5}})
    expected = pmap({'x': pmap({'y': 5})})
    assert result == expected


def test_freeze_list():
    result = freeze([1, 2, 3])
    expected = pvector([1, 2, 3])
    assert result == expected


def test_freeze_nested_list():
    result = freeze([[1, 2], [3, 4]])
    expected = pvector([pvector([1, 2]), pvector([3, 4])])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, 2, 3))
    expected = (1, 2, 3)
    assert result == expected


def test_freeze_nested_tuple():
    result = freeze(([1, 2], (3, 4)))
    expected = (pvector([1, 2]), (3, 4))
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
    expected = 42
    assert result == expected


def test_freeze_string():
    result = freeze("hello")
    expected = "hello"
    assert result == expected


def test_freeze_mixed_container():
    data = {'list': [1, 2], 'tuple': (3, 4), 'set': {5, 6}}
    result = freeze(data)
    expected = pmap({'list': pvector([1, 2]), 'tuple': (3, 4), 'set': pset([5, 6])})
    assert result == expected


# LLM-generated content at query #3
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

def test_mutant_decorator_with_dict_input():
    @mutant
    def update_dict(d, key, val):
        d[key] = val
        return d
    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_decorator_with_set_input():
    @mutant
    def add_to_set(s, element):
        s.add(element)
        return s
    original_set = {1, 2}
    result = add_to_set(original_set, 3)
    assert original_set == {1, 2}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}

def test_mutant_decorator_with_tuple_input():
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
    def combine_kwargs(**kwargs):
        return kwargs
    result = combine_kwargs(x=1, y=2)
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 1, 'y': 2}

def test_mutant_decorator_with_positional_and_keyword_arguments():
    @mutant
    def func(a, b, c=0):
        return [a, b, c]
    result = func(1, 2, c=3)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]

def test_mutant_decorator_preserves_function_metadata():
    @mutant
    def example():
        """Example docstring."""
        pass
    assert example.__name__ == 'example'
    assert example.__doc__ == 'Example docstring.'


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_with_strict_and_pmap():
    from pyrsistent import pmap, freeze
    pmap_instance = pmap({'a': [1, 2]})
    result = freeze(pmap_instance, strict=True)
    assert isinstance(result, type(pmap_instance))
    assert result['a'] == pmap_instance['a']


# LLM-generated content at query #5
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
    result = freeze([1, 2, 3])
    expected = pvector([1, 2, 3])
    assert result == expected


def test_freeze_nested_list():
    result = freeze([[1, 2], [3, 4]])
    expected = pvector([pvector([1, 2]), pvector([3, 4])])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, 2, 3))
    expected = (1, 2, 3)
    assert result == expected


def test_freeze_nested_tuple():
    result = freeze((1, [2, 3]))
    expected = (1, pvector([2, 3]))
    assert result == expected


def test_freeze_set():
    result = freeze({1, 2, 3})
    expected = pset({1, 2, 3})
    assert result == expected


def test_freeze_defaultdict():
    d = collections.defaultdict(list)
    d['a'] = [1, 2]
    result = freeze(d)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_strict_false_with_pmap():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=False)
    expected = pmap({'a': [1, 2]})
    assert result == expected


def test_freeze_strict_false_with_pvector():
    v = pvector([1, 2, 3])
    result = freeze(v, strict=False)
    expected = pvector([1, 2, 3])
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    expected = 42
    assert result == expected


def test_freeze_dict_with_nested_dict():
    result = freeze({'a': {'b': 1}})
    expected = pmap({'a': pmap({'b': 1})})
    assert result == expected


def test_freeze_list_with_dict():
    result = freeze([{'a': 1}, {'b': 2}])
    expected = pvector([pmap({'a': 1}), pmap({'b': 2})])
    assert result == expected


# LLM-generated content at query #6
#--------------------------

def test_mutant_decorator_freezes_inputs_and_output():
    def add_one(x):
        x[0] = x[0] + 1
        return x
    decorated = mutant(add_one)
    input_list = [1]
    result = decorated(input_list)
    assert input_list == [1]
    assert isinstance(result, PVector)
    assert result[0] == 2

def test_mutant_decorator_with_multiple_args():
    def concat_and_modify(a, b):
        a.append(99)
        b.append(100)
        return a + b
    decorated = mutant(concat_and_modify)
    list1 = [1, 2]
    list2 = [3, 4]
    result = decorated(list1, list2)
    assert list1 == [1, 2]
    assert list2 == [3, 4]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 99, 3, 4, 100]

def test_mutant_decorator_with_kwargs():
    def update_dict(d, key, value):
        d[key] = value
        return d
    decorated = mutant(update_dict)
    input_dict = {'a': 1}
    result = decorated(input_dict, key='b', value=2)
    assert input_dict == {'a': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_decorator_preserves_function_metadata():
    def original(x):
        """Original docstring."""
        return x
    decorated = mutant(original)
    assert decorated.__name__ == 'original'
    assert decorated.__doc__ == 'Original docstring.'

def test_mutant_decorator_with_nested_structures():
    def modify_nested(data):
        data['list'][0] = 'modified'
        data['set'].add(99)
        return data
    decorated = mutant(modify_nested)
    input_data = {'list': ['original'], 'set': {1, 2}}
    result = decorated(input_data)
    assert input_data == {'list': ['original'], 'set': {1, 2}}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert result['list'][0] == 'modified'
    assert isinstance(result['set'], PSet)
    assert 99 in result['set']

def test_mutant_decorator_with_frozen_inputs():
    def identity(x):
        return x
    decorated = mutant(identity)
    frozen_input = pvector([1, 2, 3])
    result = decorated(frozen_input)
    assert result is frozen_input

def test_mutant_decorator_returns_frozen_output_for_non_container():
    def return_number(x):
        return x + 1
    decorated = mutant(return_number)
    result = decorated(5)
    assert result == 6

def test_mutant_decorator_with_strict_false_implicitly():
    def modify_dict(d):
        d['inner'] = [1, 2]
        return d
    decorated = mutant(modify_dict)
    input_dict = {}
    result = decorated(input_dict)
    assert isinstance(result, PMap)
    assert isinstance(result['inner'], PVector)


# LLM-generated content at query #7
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
    result = freeze([1, 2, 3])
    expected = pvector([1, 2, 3])
    assert result == expected


def test_freeze_nested_list():
    result = freeze([[1, 2], [3, 4]])
    expected = pvector([pvector([1, 2]), pvector([3, 4])])
    assert result == expected


def test_freeze_tuple():
    result = freeze((1, 2, 3))
    expected = (1, 2, 3)
    assert result == expected


def test_freeze_nested_tuple():
    result = freeze((1, [2, 3]))
    expected = (1, pvector([2, 3]))
    assert result == expected


def test_freeze_set():
    result = freeze({1, 2, 3})
    expected = pset({1, 2, 3})
    assert result == expected


def test_freeze_defaultdict():
    d = collections.defaultdict(list)
    d['a'].append(1)
    result = freeze(d)
    expected = pmap({'a': pvector([1])})
    assert result == expected


def test_freeze_pmap_strict():
    m = pmap({'a': [1, 2]})
    result = freeze(m, strict=True)
    expected = pmap({'a': pvector([1, 2])})
    assert result == expected


def test_freeze_pvector_strict():
    v = pvector([[1, 2], [3, 4]])
    result = freeze(v, strict=True)
    expected = pvector([pvector([1, 2]), pvector([3, 4])])
    assert result == expected


def test_freeze_non_container():
    result = freeze(42)
    expected = 42
    assert result == expected


def test_freeze_string():
    result = freeze("hello")
    expected = "hello"
    assert result == expected


def test_freeze_nested_dict():
    result = freeze({'a': {'b': [1, 2]}})
    expected = pmap({'a': pmap({'b': pvector([1, 2])})})
    assert result == expected


def test_freeze_mixed_structure():
    result = freeze({'a': [1, {'b': 2}], 'c': (3, [4])})
    expected = pmap({'a': pvector([1, pmap({'b': 2})]), 'c': (3, pvector([4]))})
    assert result == expected


# LLM-generated content at query #8
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
    assert len(result) == 2
    assert isinstance(result[0], PVector)
    assert result[0] == pvector([1, 2, 3, 4])
    assert isinstance(result[1], PMap)
    assert result[1] == pmap({'a': 1, 'new': 'value'})

def test_mutant_decorator_with_keyword_arguments():
    def mutable_func(a, b=[]):
        b.append(a)
        return b
    decorated = mutant(mutable_func)
    result = decorated(1, b=[0])
    assert isinstance(result, PVector)
    assert result == pvector([0, 1])

def test_mutant_decorator_preserves_function_metadata():
    def sample_func(x):
        """Sample docstring."""
        return x
    decorated = mutant(sample_func)
    assert decorated.__name__ == 'sample_func'
    assert decorated.__doc__ == 'Sample docstring.'

def test_mutant_decorator_with_non_container_arguments():
    def func(x, y):
        return x + y
    decorated = mutant(func)
    result = decorated(2, 3)
    assert result == 5

def test_mutant_decorator_with_strict_freeze():
    def func(pmap_arg):
        return pmap_arg.set('new', 100)
    decorated = mutant(func)
    input_pmap = pmap({'old': 50})
    result = decorated(input_pmap)
    assert isinstance(result, PMap)
    assert result == pmap({'old': 50, 'new': 100})
    assert input_pmap == pmap({'old': 50})


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def modify_data(data_dict, data_set):
        data_dict['new_key'] = 'new_value'
        data_set.add('new_item')
        return {'modified': True, 'dict': data_dict, 'set': data_set}

    original_dict = m(a=1, b=2)
    original_set = s(1, 2, 3)
    result = modify_data(original_dict, original_set)
    assert original_dict == m(a=1, b=2)
    assert original_set == s(1, 2, 3)
    assert isinstance(result, type(freeze({})))
    assert result['modified'] is True
    assert isinstance(result['dict'], type(freeze({})))
    assert isinstance(result['set'], type(freeze(set())))


# LLM-generated content at query #10
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
    def combine(**kwargs):
        return kwargs
    result = combine(a=1, b=2)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_decorator_with_mixed_arguments():
    @mutant
    def func(a, b, c=10):
        return [a, b, c]
    result = func(1, 2, c=3)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]

def test_mutant_decorator_preserves_function_metadata():
    @mutant
    def example():
        """Example docstring."""
        pass
    assert example.__name__ == 'example'
    assert example.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_nested_structures():
    @mutant
    def process(data):
        data['list'].append(4)
        data['set'].add(5)
        return data
    original = {'list': [1, 2, 3], 'set': {1, 2, 3}}
    result = process(original)
    assert original == {'list': [1, 2, 3], 'set': {1, 2, 3}}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)
    assert list(result['list']) == [1, 2, 3, 4]
    assert set(result['set']) == {1, 2, 3, 5}

def test_mutant_decorator_with_strict_false_behavior():
    @mutant
    def func(x):
        return x
    pvec = pvector([1, 2, 3])
    result = func(pvec)
    assert result is pvec

def test_mutant_decorator_handles_non_container_arguments():
    @mutant
    def add(a, b):
        return a + b
    result = add(1, 2)
    assert result == 3


# LLM-generated content at query #11
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

def test_mutant_decorator_with_tuple():
    def modify_tuple(t, new_element):
        return t + (new_element,)
    decorated = mutant(modify_tuple)
    original_tuple = (1, 2)
    result = decorated(original_tuple, 3)
    assert original_tuple == (1, 2)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)

def test_mutant_decorator_with_keyword_arguments():
    def combine_dicts(d1, d2):
        d1.update(d2)
        return d1
    decorated = mutant(combine_dicts)
    dict1 = {'x': 10}
    dict2 = {'y': 20}
    result = decorated(d1=dict1, d2=dict2)
    assert dict1 == {'x': 10}
    assert dict2 == {'y': 20}
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 10, 'y': 20}

def test_mutant_decorator_preserves_function_metadata():
    def example_func(a, b):
        """Example docstring."""
        return a + b
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_nested_structures():
    def process(data):
        data['list'].append(99)
        return data
    decorated = mutant(process)
    original = {'list': [1, 2], 'set': {3, 4}}
    result = decorated(original)
    assert original == {'list': [1, 2], 'set': {3, 4}}
    assert isinstance(result, PMap)
    inner_list = result['list']
    assert isinstance(inner_list, PVector)
    assert list(inner_list) == [1, 2, 99]
    inner_set = result['set']
    assert isinstance(inner_set, PSet)
    assert set(inner_set) == {3, 4}

def test_mutant_decorator_with_no_arguments():
    def constant():
        return {'answer': 42}
    decorated = mutant(constant)
    result = decorated()
    assert isinstance(result, PMap)
    assert dict(result) == {'answer': 42}

def test_mutant_decorator_with_strict_false_implicitly():
    def modify_vector(v):
        v.append(100)
        return v
    decorated = mutant(modify_vector)
    original = pvector([1, 2])
    result = decorated(original)
    assert list(original) == [1, 2]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 100]


# LLM-generated content at query #12
#--------------------------

def test_mutant_decorator_freezes_args_and_return():
    @mutant
    def add_one_to_list(lst):
        lst.append(1)
        return lst
    original = [1, 2, 3]
    result = add_one_to_list(original)
    assert result == pvector([1, 2, 3, 1])
    assert original == [1, 2, 3]
    assert isinstance(result, PVector)

def test_mutant_decorator_freezes_kwargs():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    original = {'a': 1}
    result = update_dict(original, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})
    assert original == {'a': 1}
    assert isinstance(result, PMap)

def test_mutant_decorator_with_set():
    @mutant
    def add_to_set(s, element):
        s.add(element)
        return s
    original = {1, 2, 3}
    result = add_to_set(original, 4)
    assert result == pset([1, 2, 3, 4])
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)

def test_mutant_decorator_with_tuple():
    @mutant
    def modify_tuple(t):
        return t + (4,)
    original = (1, 2, 3)
    result = modify_tuple(original)
    assert result == (1, 2, 3, 4)
    assert original == (1, 2, 3)
    assert isinstance(result, tuple)

def test_mutant_decorator_with_mixed_args():
    @mutant
    def mixed_operation(lst, d, s):
        lst.append(4)
        d['new'] = 'value'
        s.add(5)
        return lst, d, s
    lst_orig = [1, 2, 3]
    d_orig = {'a': 1}
    s_orig = {1, 2, 3}
    result = mixed_operation(lst_orig, d_orig, s_orig)
    expected = (pvector([1, 2, 3, 4]), pmap({'a': 1, 'new': 'value'}), pset([1, 2, 3, 5]))
    assert result == expected
    assert lst_orig == [1, 2, 3]
    assert d_orig == {'a': 1}
    assert s_orig == {1, 2, 3}
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)

def test_mutant_decorator_preserves_function_metadata():
    @mutant
    def example_func(x):
        """Example docstring."""
        return x
    assert example_func.__name__ == 'example_func'
    assert example_func.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_no_mutation():
    @mutant
    def pure_function(x):
        return x * 2
    result = pure_function(5)
    assert result == 10

def test_mutant_decorator_with_nested_structures():
    @mutant
    def process_data(data):
        data['list'][0] = 100
        return data
    original = {'list': [1, 2, 3], 'tuple': (4, 5)}
    result = process_data(original)
    expected = pmap({'list': pvector([100, 2, 3]), 'tuple': (4, 5)})
    assert result == expected
    assert original == {'list': [1, 2, 3], 'tuple': (4, 5)}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)

def test_mutant_decorator_freezes_defaultdict():
    @mutant
    def handle_defaultdict(dd):
        dd['new_key'] = 10
        return dd
    original = collections.defaultdict(int, {'a': 1})
    result = handle_defaultdict(original)
    assert result == pmap({'a': 1, 'new_key': 10})
    assert original == collections.defaultdict(int, {'a': 1})
    assert isinstance(result, PMap)


# LLM-generated content at query #13
#--------------------------

def test_mutant_decorator_freezes_args_and_return():
    def add_one(x):
        x[0] = x[0] + 1
        return x
    decorated = mutant(add_one)
    original = [1, 2, 3]
    result = decorated(original)
    assert original == [1, 2, 3]
    assert isinstance(result, PVector)
    assert result[0] == 2
    assert result[1] == 2
    assert result[2] == 3

def test_mutant_decorator_with_kwargs():
    def update_dict(d, key, value):
        d[key] = value
        return d
    decorated = mutant(update_dict)
    original = {'a': 1}
    result = decorated(original, key='b', value=2)
    assert original == {'a': 1}
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2

def test_mutant_decorator_freezes_nested_structures():
    def modify_nested(lst):
        lst[1]['x'] = 100
        return lst
    decorated = mutant(modify_nested)
    original = [0, {'x': 1, 'y': 2}, 3]
    result = decorated(original)
    assert original == [0, {'x': 1, 'y': 2}, 3]
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['x'] == 100
    assert result[1]['y'] == 2

def test_mutant_decorator_preserves_function_metadata():
    def example_func(a, b=2):
        """Example docstring."""
        return a + b
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_set():
    def add_to_set(s, element):
        s.add(element)
        return s
    decorated = mutant(add_to_set)
    original = {1, 2, 3}
    result = decorated(original, 4)
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)
    assert 4 in result
    assert 1 in result
    assert 2 in result
    assert 3 in result

def test_mutant_decorator_with_tuple():
    def modify_tuple(t):
        return t + (4,)
    decorated = mutant(modify_tuple)
    original = (1, 2, 3)
    result = decorated(original)
    assert original == (1, 2, 3)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)
    assert isinstance(result[0], int)

def test_mutant_decorator_with_strict_false_implicitly():
    def identity(x):
        return x
    decorated = mutant(identity)
    pvec = pvector([1, 2, 3])
    result = decorated(pvec)
    assert result is pvec

def test_mutant_decorator_with_empty_args():
    def return_constant():
        return {'key': 'value'}
    decorated = mutant(return_constant)
    result = decorated()
    assert isinstance(result, PMap)
    assert result['key'] == 'value'

def test_mutant_decorator_mutation_isolated():
    mutable_list = [1]
    def append_and_return(lst):
        lst.append(2)
        return lst
    decorated = mutant(append_and_return)
    result = decorated(mutable_list)
    assert mutable_list == [1]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2]


# LLM-generated content at query #14
#--------------------------

def test_mutant_decorator_does_not_freeze_return_value_when_it_is_already_frozen():
    from pyrsistent import freeze, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def return_frozen_pmap():
        return pmap({'a': 1})

    @mutant
    def return_frozen_pset():
        return pset([1, 2, 3])

    result_pmap = return_frozen_pmap()
    result_pset = return_frozen_pset()
    assert result_pmap == pmap({'a': 1})
    assert result_pset == pset([1, 2, 3])


# LLM-generated content at query #15
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    d = defaultdict(list, {'a': [1, 2], 'b': [3, 4]})
    result = freeze(d, strict=True)
    expected = pmap({'a': [1, 2], 'b': [3, 4]})
    assert result == expected
    assert isinstance(result, type(pmap()))

def test_freeze_pmap_with_strict_true():
    from pyrsistent import pmap, freeze
    p = pmap({'x': [1, 2], 'y': [3, 4]})
    result = freeze(p, strict=True)
    expected = pmap({'x': [1, 2], 'y': [3, 4]})
    assert result == expected
    assert isinstance(result, type(pmap()))

def test_freeze_defaultdict_with_strict_false():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    d = defaultdict(list, {'a': [1, 2], 'b': [3, 4]})
    result = freeze(d, strict=False)
    expected = pmap({'a': [1, 2], 'b': [3, 4]})
    assert result == expected
    assert isinstance(result, type(pmap()))

def test_freeze_pmap_with_strict_false():
    from pyrsistent import pmap, freeze
    p = pmap({'x': [1, 2], 'y': [3, 4]})
    result = freeze(p, strict=False)
    expected = pmap({'x': [1, 2], 'y': [3, 4]})
    assert result == expected
    assert isinstance(result, type(pmap()))


# LLM-generated content at query #16
#--------------------------

def test_mutant_with_positional_arguments():
    def add_to_list(lst, value):
        lst.append(value)
        return lst
    decorated = mutant(add_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    assert result == pvector([1, 2, 3, 4])
    assert original == [1, 2, 3]

def test_mutant_with_keyword_arguments():
    def update_dict(d, key, val):
        d[key] = val
        return d
    decorated = mutant(update_dict)
    original = {'a': 1}
    result = decorated(original, key='b', val=2)
    assert result == pmap({'a': 1, 'b': 2})
    assert original == {'a': 1}

def test_mutant_with_mixed_arguments():
    def modify_collections(lst, d, s, add):
        lst.append(add)
        d['new'] = add
        s.add(add)
        return lst, d, s
    decorated = mutant(modify_collections)
    list_arg = [1]
    dict_arg = {'x': 10}
    set_arg = {5}
    result = decorated(list_arg, dict_arg, set_arg, 99)
    expected = (pvector([1, 99]), pmap({'x': 10, 'new': 99}), pset({5, 99}))
    assert result == expected
    assert list_arg == [1]
    assert dict_arg == {'x': 10}
    assert set_arg == {5}

def test_mutant_returns_frozen_result():
    def return_mutable():
        return {'inner': [1, 2, 3]}
    decorated = mutant(return_mutable)
    result = decorated()
    assert result == pmap({'inner': pvector([1, 2, 3])})

def test_mutant_with_nested_mutables_in_args():
    def process(data):
        data['list'][0] = 'changed'
        data['set'].add(100)
        return data
    decorated = mutant(process)
    arg = {'list': ['original'], 'set': {1}}
    result = decorated(arg)
    assert result == pmap({'list': pvector(['changed']), 'set': pset({1, 100})})
    assert arg == {'list': ['original'], 'set': {1}}

def test_mutant_preserves_function_metadata():
    def example(a, b):
        """Example docstring."""
        return a + b
    decorated = mutant(example)
    assert decorated.__name__ == 'example'
    assert decorated.__doc__ == 'Example docstring.'


# LLM-generated content at query #17
#--------------------------

def test_mutant_decorator_freezes_args_and_return():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]

def test_mutant_decorator_freezes_kwargs():
    @mutant
    def update_dict(d, key, val):
        d[key] = val
        return d
    original_dict = {'a': 1}
    result = update_dict(original_dict, key='b', val=2)
    assert original_dict == {'a': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}

def test_mutant_decorator_freezes_nested_structures():
    @mutant
    def modify_nested(data):
        data['list'][0] = 99
        data['tuple'][1].append(100)
        return data
    original = {'list': [1, 2], 'tuple': (3, [4, 5])}
    result = modify_nested(original)
    assert original == {'list': [1, 2], 'tuple': (3, [4, 5])}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert list(result['list']) == [99, 2]
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['tuple'][1], PVector)
    assert list(result['tuple'][1]) == [4, 5, 100]

def test_mutant_decorator_with_set():
    @mutant
    def add_to_set(s, element):
        s.add(element)
        return s
    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)
    assert original_set == {1, 2, 3}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}

def test_mutant_decorator_preserves_non_container_return():
    @mutant
    def identity(x):
        return x
    assert identity(5) == 5
    assert identity("hello") == "hello"
    assert identity(None) is None

def test_mutant_decorator_with_strict_false_implicitly():
    @mutant
    def return_pmap(pm):
        return pm
    original_pmap = pmap({'a': [1, 2]})
    result = return_pmap(original_pmap)
    assert result is original_pmap
    assert isinstance(result['a'], list)
    assert result['a'] == [1, 2]

def test_mutant_decorator_freezes_defaultdict():
    @mutant
    def use_defaultdict(dd):
        dd['new'].append(1)
        return dd
    original_dd = collections.defaultdict(list, existing=[0])
    result = use_defaultdict(original_dd)
    assert original_dd['new'] == []
    assert isinstance(result, PMap)
    assert dict(result) == {'existing': [0], 'new': [1]}

def test_mutant_decorator_with_multiple_args():
    @mutant
    def combine(a, b, c):
        a.extend(b)
        a.append(c)
        return a
    arg1 = [1, 2]
    arg2 = [3, 4]
    arg3 = 5
    result = combine(arg1, arg2, arg3)
    assert arg1 == [1, 2]
    assert arg2 == [3, 4]
    assert arg3 == 5
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4, 5]


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant_decorator_does_not_mutate_inputs():
    original_list = [1, 2, 3]
    original_dict = {'a': 4, 'b': 5}
    original_set = {6, 7, 8}

    @mutant
    def mutating_function(lst, dct, st):
        lst.append(9)
        dct['c'] = 10
        st.add(11)
        return (lst, dct, st)

    result = mutating_function(original_list, original_dict, original_set)
    assert original_list == [1, 2, 3]
    assert original_dict == {'a': 4, 'b': 5}
    assert original_set == {6, 7, 8}
    assert isinstance(result[0], pvector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)


# LLM-generated content at query #19
#--------------------------

def test_mutant_decorator_freezes_arguments_and_return():
    from pyrsistent import freeze, m, s
    from pyrsistent._helpers import mutant

    @mutant
    def modify_map(pmap_arg):
        pmap_arg['new_key'] = 'new_value'
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
    assert result_map == m(a=1, b=2, new_key='new_value')
    assert isinstance(result_map, type(freeze({})))
    assert original_set == s(1, 2, 3)
    assert result_set == s(1, 2, 3, 999)
    assert isinstance(result_set, type(freeze(set())))


# LLM-generated content at query #20
#--------------------------

def test_mutant_decorator_does_not_mutate_inputs():
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


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import freeze, pset, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def modify_pset(s):
        s.add(4)
        return s

    @mutant
    def modify_pmap(m):
        m.set('c', 15)
        return m

    original_pset = pset([1, 2, 3])
    original_pmap = pmap({'a': 13, 'b': 14})

    result_pset = modify_pset(original_pset)
    result_pmap = modify_pmap(original_pmap)

    assert original_pset == pset([1, 2, 3])
    assert original_pmap == pmap({'a': 13, 'b': 14})
    assert result_pset == pset([1, 2, 3, 4])
    assert result_pmap == pmap({'a': 13, 'b': 14, 'c': 15})
    assert isinstance(result_pset, type(original_pset))
    assert isinstance(result_pmap, type(original_pmap))


# LLM-generated content at query #22
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
    def process(data):
        data['list'].append(99)
        return data
    original = {'list': [1, 2], 'set': {3, 4}}
    result = process(original)
    assert original == {'list': [1, 2], 'set': {3, 4}}
    assert isinstance(result, PMap)
    inner_list = result['list']
    assert isinstance(inner_list, PVector)
    assert list(inner_list) == [1, 2, 99]
    inner_set = result['set']
    assert isinstance(inner_set, PSet)
    assert set(inner_set) == {3, 4}

def test_mutant_decorator_with_no_mutation():
    @mutant
    def pure_function(x):
        return x * 2
    original = 5
    result = pure_function(original)
    assert original == 5
    assert result == 10

def test_mutant_decorator_with_strict_false_implicitly():
    @mutant
    def return_container(x):
        return [x, {'a': x}]
    result = return_container(1)
    assert isinstance(result, PVector)
    first_elem = result[0]
    assert first_elem == 1
    second_elem = result[1]
    assert isinstance(second_elem, PMap)
    assert dict(second_elem) == {'a': 1}


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_decorator_does_not_mutate_inputs():
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


# LLM-generated content at query #24
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
    original_data = {'list': [1, 2], 'set': {3, 4}}
    result = process_data(original_data)
    assert original_data == {'list': [1, 2], 'set': {3, 4}}
    assert isinstance(result, PMap)
    result_dict = dict(result)
    assert isinstance(result_dict['list'], PVector)
    assert list(result_dict['list']) == [1, 2, 99]
    assert isinstance(result_dict['set'], PSet)
    assert set(result_dict['set']) == {3, 4}

def test_mutant_decorator_with_no_mutation():
    @mutant
    def pure_function(x):
        return x * 2
    original = 5
    result = pure_function(original)
    assert original == 5
    assert result == 10


# LLM-generated content at query #25
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
        pset_arg.add(999)
        return pset_arg

    original_map = m(a=1, b=2)
    original_set = s(1, 2, 3)

    result_map = modify_map(original_map)
    result_set = modify_set(original_set)

    assert original_map == m(a=1, b=2)
    assert original_set == s(1, 2, 3)
    assert result_map == m(a=1, b=2, new_key="new_value")
    assert result_set == s(1, 2, 3, 999)
    assert isinstance(result_map, type(freeze({})))
    assert isinstance(result_set, type(freeze(set())))


# LLM-generated content at query #26
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
    assert result_set == pset([1, 2, 3, 4])
    assert isinstance(result_set, type(freeze(pset())))
    assert original_map == pmap({'a': 1, 'b': 2})
    assert result_map == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert isinstance(result_map, type(freeze(pmap())))


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_decorator_does_not_mutate_inputs():
    original_list = [1, 2, 3]
    original_dict = {'a': 1, 'b': 2}
    
    @mutant
    def modify_inputs(lst, dct):
        lst.append(4)
        dct['c'] = 3
        return lst, dct
    
    result = modify_inputs(original_list, original_dict)
    assert original_list == [1, 2, 3]
    assert original_dict == {'a': 1, 'b': 2}
    assert isinstance(result[0], pvector)
    assert isinstance(result[1], PMap)


# LLM-generated content at query #28
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
    def combine(data, prefix):
        return {prefix + k: v for k, v in data.items()}
    decorated = mutant(combine)
    original_data = {'x': 10}
    result = decorated(data=original_data, prefix='key_')
    assert original_data == {'x': 10}
    assert isinstance(result, PMap)
    assert dict(result) == {'key_x': 10}

def test_mutant_decorator_preserves_function_metadata():
    def example_func():
        """Example docstring."""
        pass
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring.'

def test_mutant_decorator_with_nested_structures():
    def modify_nested(obj):
        if isinstance(obj, dict):
            obj['inner'] = [4, 5]
        return obj
    decorated = mutant(modify_nested)
    original = {'a': [1, 2]}
    result = decorated(original)
    assert original == {'a': [1, 2]}
    assert isinstance(result, PMap)
    inner_val = result['a']
    assert isinstance(inner_val, PVector)
    assert list(inner_val) == [1, 2]

def test_mutant_decorator_with_empty_arguments():
    def return_constant():
        return {'status': 'ok'}
    decorated = mutant(return_constant)
    result = decorated()
    assert isinstance(result, PMap)
    assert dict(result) == {'status': 'ok'}

def test_mutant_decorator_freezes_arguments_recursively():
    def access_element(container):
        return container[0]
    decorated = mutant(access_element)
    original = [[1, 2], {'a': 3}]
    result = decorated(original)
    assert original == [[1, 2], {'a': 3}]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2]


# LLM-generated content at query #29
#--------------------------

def test_freeze_defaultdict_with_strict_true():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    dd = defaultdict(list, {'a': [1, 2]})
    result = freeze(dd, strict=True)
    expected = pmap({'a': [1, 2]})
    assert result == expected


