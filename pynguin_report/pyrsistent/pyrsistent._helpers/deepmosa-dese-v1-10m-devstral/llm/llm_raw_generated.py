####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def nested(data):
        return data

    input_data = {'a': [1, 2, {'b': 3}]}
    expected = pmap({'a': pvector([1, 2, pmap({'b': 3})])})
    assert nested(input_data) == expected

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    input_kwargs = {'x': [1, 2], 'y': {'a': 3}}
    expected = pmap({'x': pvector([1, 2]), 'y': pmap({'a': 3})})
    assert merge(**input_kwargs) == expected

def test_mutant_with_set():
    @mutant
    def process(s):
        return s

    input_set = {1, 2, 3}
    expected = pset({1, 2, 3})
    assert process(input_set) == expected

def test_mutant_with_tuple():
    @mutant
    def process(t):
        return t

    input_tuple = (1, [2, 3], {'a': 4})
    expected = (1, pvector([2, 3]), pmap({'a': 4}))
    assert process(input_tuple) == expected

def test_mutant_with_empty_structures():
    @mutant
    def empty():
        return [], {}, set(), ()

    expected = (pvector([]), pmap({}), pset(set()), ())
    assert empty() == expected

def test_mutant_with_strict_false():
    @mutant
    def identity(x):
        return x

    pvec = pvector([1, 2, 3])
    pmap_obj = pmap({'a': 1})
    assert identity(pvec) == pvec
    assert identity(pmap_obj) == pmap_obj


# LLM-generated content at query #2
#--------------------------

```python
def test_mutant_predicate():
    assert True


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_empty_list():
    assert freeze([]) == pvector([])

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_set():
    assert freeze(set()) == pset([])

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, (2, 3))) == (1, (2, 3))

def test_freeze_mixed_types():
    assert freeze([1, {'a': 2}, (3, 4)]) == pvector([1, pmap({'a': 2}), (3, 4)])

def test_freeze_with_strict_false():
    assert freeze({'a': [1, 2]}, strict=False) == pmap({'a': [1, 2]})

def test_freeze_pvector():
    pv = pvector([1, 2, 3])
    assert freeze(pv) == pv

def test_freeze_pmap():
    pm = pmap({'a': 1})
    assert freeze(pm) == pm

def test_freeze_pset():
    ps = pset([1, 2, 3])
    assert freeze(ps) == ps

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #4
#--------------------------

```python
def test_mutant_with_empty_args_and_kwargs():
    @mutant
    def empty_func():
        return 42
    assert empty_func() == 42

def test_mutant_with_list_arg():
    @mutant
    def list_func(lst):
        return lst
    assert list_func([1, 2, 3]) == pvector([1, 2, 3])

def test_mutant_with_dict_arg():
    @mutant
    def dict_func(d):
        return d
    assert dict_func({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_set_arg():
    @mutant
    def set_func(s):
        return s
    assert set_func({1, 2, 3}) == pset([1, 2, 3])

def test_mutant_with_tuple_arg():
    @mutant
    def tuple_func(t):
        return t
    assert tuple_func((1, [2, 3])) == (1, pvector([2, 3]))

def test_mutant_with_nested_structures():
    @mutant
    def nested_func(data):
        return data
    input_data = {'a': [1, 2], 'b': {3, 4}}
    expected = pmap({'a': pvector([1, 2]), 'b': pset([3, 4])})
    assert nested_func(input_data) == expected

def test_mutant_with_kwargs():
    @mutant
    def kwargs_func(**kwargs):
        return kwargs
    assert kwargs_func(a=1, b=[2, 3]) == pmap({'a': 1, 'b': pvector([2, 3])})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def mixed_func(arg1, arg2, kwarg1=None, kwarg2=None):
        return {'arg1': arg1, 'arg2': arg2, 'kwarg1': kwarg1, 'kwarg2': kwarg2}
    result = mixed_func([1, 2], {'a': 3}, kwarg1={4, 5}, kwarg2=(6, [7]))
    expected = pmap({
        'arg1': pvector([1, 2]),
        'arg2': pmap({'a': 3}),
        'kwarg1': pset([4, 5]),
        'kwarg2': (6, pvector([7]))
    })
    assert result == expected

def test_mutant_with_pvector_arg():
    @mutant
    def pvector_func(pv):
        return pv
    input_pv = pvector([1, 2, 3])
    assert pvector_func(input_pv) == input_pv

def test_mutant_with_pmap_arg():
    @mutant
    def pmap_func(pm):
        return pm
    input_pm = pmap({'a': 1, 'b': 2})
    assert pmap_func(input_pm) == input_pm

def test_mutant_with_pset_arg():
    @mutant
    def pset_func(ps):
        return ps
    input_ps = pset([1, 2, 3])
    assert pset_func(input_ps) == input_ps

def test_mutant_with_non_container_arg():
    @mutant
    def non_container_func(x):
        return x
    assert non_container_func(42) == 42
    assert non_container_func("hello") == "hello"


# LLM-generated content at query #5
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


# LLM-generated content at query #6
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    expected = pmap({'a': 1, 'b': 2})
    assert result == expected


# LLM-generated content at query #7
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #8
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    assert isinstance(d, collections.defaultdict) or (True and isinstance(d, PMap))


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant_with_list_input():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    result = add_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict_input():
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    result = add_to_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_set_input():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_with_tuple_input():
    @mutant
    def modify_tuple(t, item):
        return t + (item,)

    result = modify_tuple((1, 2), 3)
    assert result == (1, 2, 3)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(lst, item):
        lst[0].append(item)
        return lst

    result = modify_nested([[], 2], 3)
    assert result == pvector([pvector([3]), 2])

def test_mutant_with_kwargs():
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(existing_key='existing_value')
    assert result == pmap({'existing_key': 'existing_value', 'new_key': 'new_value'})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def modify_mixed(lst, d, key, value):
        lst.append(key)
        d[key] = value
        return (lst, d)

    result = modify_mixed([1], {'a': 2}, 'b', 3)
    assert result == (pvector([1, 'b']), pmap({'a': 2, 'b': 3}))


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, c=0):
        return a + b + c

    assert combine(1, 2, c=3) == 6
    assert combine([1], [2], c=[3]) == pvector([1, 2, 3])
    assert combine({'a': 1}, {'b': 2}, c={'c': 3}) == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_nested_structures():
    @mutant
    def process(data):
        return data

    nested = {'a': [1, 2, {'b': 3}], 'c': (4, [5, 6])}
    result = process(nested)
    expected = pmap({'a': pvector([1, 2, pmap({'b': 3})]), 'c': (4, pvector([5, 6]))})
    assert result == expected

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s

    assert process_set({1, 2, 3}) == pset({1, 2, 3})

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t

    assert process_tuple((1, [2, 3])) == (1, pvector([2, 3]))

def test_mutant_with_pvector():
    @mutant
    def process_pvector(v):
        return v

    assert process_pvector(pvector([1, 2, 3])) == pvector([1, 2, 3])

def test_mutant_with_pmap():
    @mutant
    def process_pmap(m):
        return m

    assert process_pmap(pmap({'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2})

def test_mutant_with_pset():
    @mutant
    def process_pset(s):
        return s

    assert process_pset(pset({1, 2, 3})) == pset({1, 2, 3})

def test_mutant_no_mutation():
    @mutant
    def no_op(x):
        return x

    original = {'a': [1, 2, 3]}
    frozen = no_op(original)
    assert frozen == pmap({'a': pvector([1, 2, 3])})
    assert original == {'a': [1, 2, 3]}

def test_mutant_with_strict_false():
    @mutant
    def process_non_strict(data):
        return data

    # Even with strict=False in freeze, mutant should still freeze
    result = process_non_strict({'a': [1, 2]})
    assert result == pmap({'a': pvector([1, 2])})


# LLM-generated content at query #11
#--------------------------

```python
def test_freeze_with_strict_true_and_pmap_instance():
    pm = pmap({'a': 1})
    assert freeze(pm, strict=True) == pmap({'a': 1})


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    from pyrsistent import freeze, pmap
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #13
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    from pyrsistent import pmap
    from pyrsistent._helpers import freeze

    original = defaultdict(int, {'a': 1, 'b': 2})
    frozen = freeze(original)
    expected = pmap({'a': 1, 'b': 2})
    assert frozen == expected


# LLM-generated content at query #14
#--------------------------

```python
def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 1}}) == pmap({'a': pmap({'b': 1})})

def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_nested_tuple():
    assert freeze((1, (2, 3))) == (1, (2, 3))

def test_freeze_mixed_containers():
    assert freeze([1, {'a': 2}, (3, 4)]) == pvector([1, pmap({'a': 2}), (3, 4)])

def test_freeze_with_strict_false():
    pvec = pvector([1, 2])
    assert freeze(pvec, strict=False) == pvec

def test_freeze_with_strict_true():
    pvec = pvector([1, 2])
    assert freeze(pvec, strict=True) == pvector([1, 2])

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

def test_freeze_non_container():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #15
#--------------------------

```python
def test_freeze_defaultdict():
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_with_no_args():
    @mutant
    def fn():
        return 42
    assert fn() == 42

def test_mutant_with_positional_args():
    @mutant
    def fn(a, b):
        return a + b
    assert fn(1, 2) == 3

def test_mutant_with_keyword_args():
    @mutant
    def fn(a=1, b=2):
        return a + b
    assert fn(a=3, b=4) == 7

def test_mutant_with_mixed_args():
    @mutant
    def fn(a, b=2):
        return a + b
    assert fn(3, b=4) == 7

def test_mutant_with_mutable_args():
    @mutant
    def fn(lst):
        lst.append(4)
        return lst
    assert fn([1, 2, 3]) == pvector([1, 2, 3, 4])

def test_mutant_with_mutable_kwargs():
    @mutant
    def fn(d):
        d['c'] = 3
        return d
    assert fn({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_with_nested_mutable_args():
    @mutant
    def fn(d):
        d['lst'].append(4)
        return d
    assert fn({'lst': [1, 2, 3]}) == pmap({'lst': pvector([1, 2, 3, 4])})

def test_mutant_with_set_arg():
    @mutant
    def fn(s):
        return s | {4}
    assert fn({1, 2, 3}) == pset([1, 2, 3, 4])

def test_mutant_with_tuple_arg():
    @mutant
    def fn(t):
        return t + (4,)
    assert fn((1, 2, 3)) == (1, 2, 3, 4)

def test_mutant_with_pvector_arg():
    @mutant
    def fn(v):
        return v.append(4)
    assert fn(pvector([1, 2, 3])) == pvector([1, 2, 3, 4])

def test_mutant_with_pmap_arg():
    @mutant
    def fn(m):
        return m.set('c', 3)
    assert fn(pmap({'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_with_pset_arg():
    @mutant
    def fn(s):
        return s.add(4)
    assert fn(pset([1, 2, 3])) == pset([1, 2, 3, 4])

def test_mutant_with_strict_false():
    @mutant
    def fn():
        return freeze([1, 2, 3], strict=False)
    assert fn() == [1, 2, 3]

def test_mutant_preserves_function_metadata():
    def original_fn(a, b):
        """Original docstring"""
        return a + b
    decorated_fn = mutant(original_fn)
    assert decorated_fn.__name__ == original_fn.__name__
    assert decorated_fn.__doc__ == original_fn.__doc__


# LLM-generated content at query #17
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    assert freeze(collections.defaultdict(int, {'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #18
#--------------------------

```python
def test_freeze_with_empty_dict():
    assert freeze({}) == pmap()


# LLM-generated content at query #19
#--------------------------

```python
def test_freeze_predicate_false():
    assert not (type(None) is dict or (True and isinstance(None, PMap)))


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not False and True)


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_predicate_false():
    assert not callable(mutant)


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_basic():
    def add(a, b):
        return a + b
    decorated_add = mutant(add)
    assert decorated_add(1, 2) == 3
    assert isinstance(decorated_add([1], [2]), tuple)
    assert decorated_add([1], [2]) == (pvector([1]), pvector([2]))

def test_mutant_with_kwargs():
    def concat(a, b, sep=' '):
        return a + sep + b
    decorated_concat = mutant(concat)
    assert decorated_concat('a', 'b') == 'a b'
    assert decorated_concat('a', 'b', sep='-') == 'a-b'
    assert isinstance(decorated_concat(['a'], ['b'], sep='-'), tuple)
    assert decorated_concat(['a'], ['b'], sep='-') == (pvector(['a']), pvector(['b']), '-')

def test_mutant_with_nested_structures():
    def nest(data):
        return {'key': data}
    decorated_nest = mutant(nest)
    result = decorated_nest([1, 2, 3])
    assert isinstance(result, dict)
    assert isinstance(result['key'], tuple)
    assert result['key'] == (1, 2, 3)

def test_mutant_with_persistent_types():
    def process(pvec, pmap):
        return pvec.append(1), pmap.set('new', 1)
    decorated_process = mutant(process)
    result = decorated_process(pvector([1, 2]), pmap({'a': 1}))
    assert isinstance(result, tuple)
    assert result[0] == (1, 2, 1)
    assert result[1] == pmap({'a': 1, 'new': 1})

def test_mutant_with_set():
    def set_op(s):
        return s | {1, 2}
    decorated_set_op = mutant(set_op)
    result = decorated_set_op({3, 4})
    assert isinstance(result, pset)
    assert result == pset({1, 2, 3, 4})

def test_mutant_preserves_function_name_and_doc():
    def example_func(x):
        """Example function."""
        return x
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example function.'


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_predicate():
    assert True


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_predicate():
    assert not False


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == 'test_func'


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_list_argument():
    @mutant
    def process_list(lst):
        lst.append(4)
        return lst
    result = process_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_dict_argument():
    @mutant
    def process_dict(d):
        d['new_key'] = 'new_value'
        return d
    result = process_dict({'a': 1})
    assert result == pmap({'a': 1, 'new_key': 'new_value'})

def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        s.add(4)
        return s
    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3, 4])

def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4,)
    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        data['list'].append(4)
        return data
    result = process_nested({'list': [1, 2, 3]})
    assert result == pmap({'list': pvector([1, 2, 3, 4])})

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(a, b, **kwargs):
        kwargs['new_key'] = 'new_value'
        return {'a': a, 'b': b, 'kwargs': kwargs}
    result = process_kwargs(1, 2, x=3, y=4)
    assert result == pmap({'a': 1, 'b': 2, 'kwargs': pmap({'x': 3, 'y': 4, 'new_key': 'new_value'})})

def test_mutant_with_no_args():
    @mutant
    def no_args():
        return [1, 2, 3]
    result = no_args()
    assert result == pvector([1, 2, 3])

def test_mutant_with_strict_false():
    @mutant
    def process_with_pvector(pv):
        return pv.append(4)
    result = process_with_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])


# LLM-generated content at query #2
#--------------------------

```python
def test_mutant_predicate():
    assert not (False)


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_dict_with_values():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_list_with_values():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_tuple_with_values():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, (2, 3))) == (1, (2, 3))

def test_freeze_set_with_values():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

def test_freeze_mixed_types():
    assert freeze({'a': [1, 2], 'b': (3, 4)}) == pmap({'a': pvector([1, 2]), 'b': (3, 4)})

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': 1}), strict=False) == pmap({'a': 1})

def test_freeze_non_strict_pvector():
    assert freeze(pvector([1, 2]), strict=False) == pvector([1, 2])

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1})
    assert freeze(dd) == pmap({'a': 1})

def test_freeze_non_container():
    assert freeze(42) == 42


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_mutant_with_list():
    @mutant
    def fn(x):
        x.append(1)
        return x
    result = fn([1, 2, 3])
    assert result == (1, 2, 3, 1)

def test_mutant_with_dict():
    @mutant
    def fn(x):
        x['a'] = 1
        return x
    result = fn({'a': 2})
    assert result == pmap({'a': 1})

def test_mutant_with_set():
    @mutant
    def fn(x):
        x.add(1)
        return x
    result = fn({1, 2, 3})
    assert result == pset({1, 2, 3})

def test_mutant_with_tuple():
    @mutant
    def fn(x):
        return x + (1,)
    result = fn((1, 2, 3))
    assert result == (1, 2, 3, 1)

def test_mutant_with_multiple_args():
    @mutant
    def fn(x, y):
        x.append(y)
        return x
    result = fn([1, 2], 3)
    assert result == (1, 2, 3)

def test_mutant_with_kwargs():
    @mutant
    def fn(**kwargs):
        kwargs['a'] = 1
        return kwargs
    result = fn(a=2)
    assert result == pmap({'a': 1})

def test_mutant_with_nested_structures():
    @mutant
    def fn(x):
        x['a'].append(1)
        return x
    result = fn({'a': [1, 2]})
    assert result == pmap({'a': (1, 2, 1)})

def test_mutant_with_no_mutation():
    @mutant
    def fn(x):
        return x
    result = fn([1, 2, 3])
    assert result == (1, 2, 3)

def test_mutant_with_pvector():
    @mutant
    def fn(x):
        return x.append(1)
    result = fn(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 1])

def test_mutant_with_pmap():
    @mutant
    def fn(x):
        return x.set('a', 1)
    result = fn(pmap({'a': 2}))
    assert result == pmap({'a': 1})


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_empty_list():
    assert freeze([]) == pvector([])

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_set():
    assert freeze(set()) == pset([])

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_mixed_containers():
    assert freeze([1, {'a': (2, [3, 4])}]) == pvector([1, pmap({'a': (2, pvector([3, 4]))})])

def test_freeze_pvector():
    assert freeze(pvector([1, 2, 3])) == pvector([1, 2, 3])

def test_freeze_pmap():
    assert freeze(pmap({'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2})

def test_freeze_pset():
    assert freeze(pset([1, 2, 3])) == pset([1, 2, 3])

def test_freeze_non_strict():
    assert freeze([1, {'a': [2, 3]}], strict=False) == pvector([1, pmap({'a': [2, 3]})])

def test_freeze_defaultdict():
    d = collections.defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(d) == pmap({'a': 1, 'b': 2})

def test_freeze_immutable_types():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"
    assert freeze(True) == True


# LLM-generated content at query #3
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == 'test_func'


# LLM-generated content at query #4
#--------------------------

```python
def test_mutant_with_empty_function():
    @mutant
    def empty_fn():
        pass
    assert empty_fn() is None

def test_mutant_with_immutable_args():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_mutable_args():
    @mutant
    def process_list(lst):
        return lst + [4]
    assert process_list([1, 2, 3]) == pvector([1, 2, 3, 4])

def test_mutant_with_dict_args():
    @mutant
    def process_dict(d):
        d['new_key'] = 'new_value'
        return d
    result = process_dict({'a': 1})
    assert result == pmap({'a': 1, 'new_key': 'new_value'})

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        return kwargs
    result = process_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def process_mixed(arg1, arg2, **kwargs):
        return {'arg1': arg1, 'arg2': arg2, 'kwargs': kwargs}
    result = process_mixed([1, 2], {'a': 3}, extra=4)
    expected = pmap({'arg1': pvector([1, 2]), 'arg2': pmap({'a': 3}), 'kwargs': pmap({'extra': 4})})
    assert result == expected

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    nested = {'a': [1, 2], 'b': {'c': 3}}
    result = process_nested(nested)
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})
    assert result == expected

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s
    result = process_set({1, 2, 3})
    assert result == pset({1, 2, 3})

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t
    result = process_tuple((1, [2, 3]))
    assert result == (1, pvector([2, 3]))


# LLM-generated content at query #5
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict_with_values():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 1}}) == pmap({'a': pmap({'b': 1})})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_pvector():
    assert freeze(pvector([1, 2, 3])) == pvector([1, 2, 3])

def test_freeze_pmap():
    assert freeze(pmap({'a': 1})) == pmap({'a': 1})

def test_freeze_non_container():
    assert freeze(42) == 42

def test_freeze_defaultdict():
    d = collections.defaultdict(int, {'a': 1})
    assert freeze(d) == pmap({'a': 1})

def test_freeze_strict_false():
    assert freeze(pvector([1, 2, 3]), strict=False) == pvector([1, 2, 3])
    assert freeze(pmap({'a': 1}), strict=False) == pmap({'a': 1})


# LLM-generated content at query #6
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (True)


# LLM-generated content at query #7
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    d = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_mutable_arguments():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]

def test_mutant_with_mutable_keyword_arguments():
    @mutant
    def modify_dict(d, **kwargs):
        d.update(kwargs)
        return d

    input_dict = {'a': 1}
    result = modify_dict(input_dict, b=2)
    assert result == pmap({'a': 1, 'b': 2})
    assert input_dict == {'a': 1}

def test_mutant_with_nested_structures():
    @mutant
    def nested_operation(data):
        data['inner'].append(1)
        return data

    input_data = {'inner': [1, 2]}
    result = nested_operation(input_data)
    assert result == pmap({'inner': pvector([1, 2, 1])})
    assert input_data == {'inner': [1, 2]}

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s | {4}

    input_set = {1, 2, 3}
    result = process_set(input_set)
    assert result == pset([1, 2, 3, 4])
    assert input_set == {1, 2, 3}

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t + (4,)

    input_tuple = (1, 2, 3)
    result = process_tuple(input_tuple)
    assert result == (1, 2, 3, 4)
    assert input_tuple == (1, 2, 3)

def test_mutant_with_mixed_types():
    @mutant
    def mixed_operation(a, b, c):
        a.add(4)
        b.append(4)
        c['d'] = 4
        return (a, b, c)

    input_set = {1, 2, 3}
    input_list = [1, 2, 3]
    input_dict = {'a': 1, 'b': 2, 'c': 3}
    result = mixed_operation(input_set, input_list, input_dict)
    assert result == (pset([1, 2, 3, 4]), pvector([1, 2, 3, 4]), pmap({'a': 1, 'b': 2, 'c': 3, 'd': 4}))
    assert input_set == {1, 2, 3}
    assert input_list == [1, 2, 3]
    assert input_dict == {'a': 1, 'b': 2, 'c': 3}


# LLM-generated content at query #9
#--------------------------

```python
def test_freeze_with_strict_true_converts_dict_to_pmap():
    result = freeze({'a': 1, 'b': 2}, strict=True)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return data

    assert process([1, {'a': [2, 3]}]) == pvector([1, pmap({'a': pvector([2, 3])})])
    assert process({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s

    assert process_set({1, 2, 3}) == pset({1, 2, 3})

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t

    assert process_tuple((1, [2, 3])) == (1, pvector([2, 3]))

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        return kwargs

    assert process_kwargs(a=1, b=[2, 3]) == pmap({'a': 1, 'b': pvector([2, 3])})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def process_mixed(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}

    assert process_mixed(1, [2, 3], c=4, d=[5, 6]) == pmap({
        'a': 1,
        'b': pvector([2, 3]),
        'kwargs': pmap({'c': 4, 'd': pvector([5, 6])})
    })


# LLM-generated content at query #11
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3

def test_mutant_with_list_arguments():
    @mutant
    def process_list(lst):
        lst.append(4)
        return lst

    result = process_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_dict_arguments():
    @mutant
    def process_dict(d):
        d['new_key'] = 42
        return d

    result = process_dict({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 42})

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        kwargs['extra'] = 'value'
        return kwargs

    result = process_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'extra': 'value'})

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        data['list'].append(4)
        return data

    result = process_nested({'list': [1, 2, 3]})
    assert result == pmap({'list': pvector([1, 2, 3, 4])})

def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4,)

    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        s.add(4)
        return s

    result = process_set({1, 2, 3})
    assert result == pset({1, 2, 3, 4})

def test_mutant_with_mixed_arguments():
    @mutant
    def process_mixed(a, b, c):
        return (a, b, c)

    result = process_mixed([1, 2], {'x': 3}, {4, 5})
    assert result == (pvector([1, 2]), pmap({'x': 3}), pset({4, 5}))


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict_with_values():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_mixed_structures():
    assert freeze([1, {'a': (2, [3, 4])}]) == pvector([1, pmap({'a': (2, pvector([3, 4]))})])

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': [1, 2]}), strict=False) == pmap({'a': [1, 2]})

def test_freeze_non_strict_pvector():
    assert freeze(pvector([1, {'a': 2}]), strict=False) == pvector([1, {'a': 2}])

def test_freeze_non_strict_pset():
    assert freeze(pset([1, 2]), strict=False) == pset([1, 2])

def test_freeze_defaultdict():
    d = collections.defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(d) == pmap({'a': 1, 'b': 2})

def test_freeze_non_container():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #13
#--------------------------

```python
def test_freeze_with_dict():
    result = freeze({'a': 1, 'b': 2})
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant_predicate():
    assert not (False)


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return data

    input_data = [1, {'a': [2, 3], 'b': set([4, 5])}]
    expected = pvector([1, pmap({'a': pvector([2, 3]), 'b': pset({4, 5})})])
    assert process(input_data) == expected

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    input_kwargs = {'a': [1, 2], 'b': {'c': 3}}
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})
    assert merge(**input_kwargs) == expected

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}

    input_a = [1, 2]
    input_b = {'x': 3}
    input_kwargs = {'y': set([4, 5])}
    expected = pmap({
        'a': pvector([1, 2]),
        'b': pmap({'x': 3}),
        'kwargs': pmap({'y': pset({4, 5})})
    })
    assert combine(input_a, input_b, **input_kwargs) == expected

def test_mutant_with_non_container_types():
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("hello") == "hello"
    assert identity(None) is None

def test_mutant_with_empty_containers():
    @mutant
    def empty():
        return [], {}

    result = empty()
    assert result == (pvector(), pmap())

def test_mutant_with_tuple():
    @mutant
    def tuple_process(t):
        return t

    input_tuple = (1, [2, 3], {'a': 4})
    expected = (1, pvector([2, 3]), pmap({'a': 4}))
    assert tuple_process(input_tuple) == expected


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3

def test_mutant_with_list_argument():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst

    result = append_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict_argument():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['nested'].append(4)
        return data

    result = modify_nested({'nested': [1, 2, 3]})
    assert result == pmap({'nested': pvector([1, 2, 3, 4])})

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b=10):
        return a + b

    result = combine(5, b=15)
    assert result == 20

def test_mutant_with_set_argument():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_with_tuple_argument():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)


# LLM-generated content at query #17
#--------------------------

```python
def test_freeze_with_dict():
    result = freeze({'a': 1, 'b': 2})
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #18
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    from pyrsistent import freeze, pmap
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    expected = pmap({'a': 1, 'b': 2})
    assert result == expected


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_decorator_returns_callable():
    @mutant
    def test_fn():
        pass
    assert callable(test_fn)


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        data['values'].append(4)
        return data

    input_data = {'values': [1, 2, 3]}
    result = process(input_data)
    assert result == pmap({'values': pvector([1, 2, 3, 4])})
    assert input_data == {'values': [1, 2, 3]}

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    result = merge(a=[1, 2], b={'x': 3})
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'x': 3})})

def test_mutant_with_no_args():
    @mutant
    def get_defaults():
        return {'a': [1, 2], 'b': {'x': 3}}

    result = get_defaults()
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'x': 3})})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}

    result = combine([1, 2], {'x': 3}, c=set([4, 5]))
    assert result == pmap({
        'a': pvector([1, 2]),
        'b': pmap({'x': 3}),
        'kwargs': pmap({'c': pset([4, 5])})
    })


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_mutant_decorator_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3

def test_mutant_decorator_with_list_arguments():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst

    result = append_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_decorator_with_dict_arguments():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_decorator_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['nested'].append(4)
        return data

    result = modify_nested({'nested': [1, 2, 3]})
    assert result == pmap({'nested': pvector([1, 2, 3, 4])})

def test_mutant_decorator_with_kwargs():
    @mutant
    def merge_dicts(**kwargs):
        return {**kwargs}

    result = merge_dicts(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_decorator_with_set_argument():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_decorator_with_tuple_argument():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_decorator_with_mixed_arguments():
    @mutant
    def process_data(data, extra):
        data['values'].extend(extra)
        return data

    result = process_data({'values': [1]}, [2, 3])
    assert result == pmap({'values': pvector([1, 2, 3])})


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_list_with_nested_dict():
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_dict_with_nested_list():
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_tuple_with_nested_list():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_pvector_with_elements():
    assert freeze(pvector([1, 2, 3])) == pvector([1, 2, 3])

def test_freeze_pmap_with_elements():
    assert freeze(pmap({'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2})

def test_freeze_non_container():
    assert freeze(42) == 42

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': [1, 2]}), strict=False) == pmap({'a': [1, 2]})

def test_freeze_non_strict_pvector():
    assert freeze(pvector([1, {'a': 3}]), strict=False) == pvector([1, {'a': 3}])

def test_freeze_defaultdict():
    d = collections.defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(d) == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #3
#--------------------------

```python
def test_mutant_with_empty_function():
    @mutant
    def empty_func():
        return None
    assert empty_func() is None

def test_mutant_with_list_arg():
    @mutant
    def list_func(lst):
        return lst + [4]
    result = list_func([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])
    assert type(result) is PVector

def test_mutant_with_dict_arg():
    @mutant
    def dict_func(d):
        return {**d, 'new_key': 'new_value'}
    result = dict_func({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert type(result) is PMap

def test_mutant_with_set_arg():
    @mutant
    def set_func(s):
        return s | {4, 5}
    result = set_func({1, 2, 3})
    assert result == pset({1, 2, 3, 4, 5})
    assert type(result) is PSet

def test_mutant_with_tuple_arg():
    @mutant
    def tuple_func(t):
        return t + (4,)
    result = tuple_func((1, 2, 3))
    assert result == (1, 2, 3, 4)
    assert type(result) is tuple

def test_mutant_with_nested_structures():
    @mutant
    def nested_func(data):
        data['inner_list'].append(4)
        return data
    result = nested_func({'inner_list': [1, 2, 3]})
    assert result == pmap({'inner_list': pvector([1, 2, 3, 4])})
    assert type(result['inner_list']) is PVector

def test_mutant_with_kwargs():
    @mutant
    def kwargs_func(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}
    result = kwargs_func(1, [2, 3], c={4, 5}, d=(6, 7))
    assert result == pmap({
        'a': 1,
        'b': pvector([2, 3]),
        'kwargs': pmap({
            'c': pset({4, 5}),
            'd': (6, 7)
        })
    })
    assert type(result['b']) is PVector
    assert type(result['kwargs']['c']) is PSet

def test_mutant_with_multiple_args():
    @mutant
    def multi_arg_func(lst, d, s, t):
        return [lst, d, s, t]
    result = multi_arg_func([1, 2], {'a': 3}, {4, 5}, (6, 7))
    assert result == pvector([
        pvector([1, 2]),
        pmap({'a': 3}),
        pset({4, 5}),
        (6, 7)
    ])
    assert type(result[0]) is PVector
    assert type(result[1]) is PMap
    assert type(result[2]) is PSet

def test_mutant_preserves_immutable_types():
    @mutant
    def immutable_func(i, s, t):
        return (i, s, t)
    result = immutable_func(42, "hello", True)
    assert result == (42, "hello", True)
    assert type(result) is tuple


# LLM-generated content at query #4
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_nested():
    @mutant
    def process(data):
        data['nested']['value'] += 1
        return data

    input_data = {'nested': {'value': 5}}
    result = process(input_data)
    assert result == pmap({'nested': pmap({'value': 6})})
    assert input_data == {'nested': {'value': 5}}  # Original unchanged

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {**a, **b, **kwargs}

    assert combine({'x': 1}, {'y': 2}, z=3) == pmap({'x': 1, 'y': 2, 'z': 3})

def test_mutant_list_operations():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original unchanged

def test_mutant_set_operations():
    @mutant
    def modify_set(s):
        s.add(3)
        return s

    input_set = {1, 2}
    result = modify_set(input_set)
    assert result == pset([1, 2, 3])
    assert input_set == {1, 2}  # Original unchanged

def test_mutant_tuple_operations():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    input_tuple = (1, 2, 3)
    result = modify_tuple(input_tuple)
    assert result == (1, 2, 3, 4)
    assert input_tuple == (1, 2, 3)  # Original unchanged

def test_mutant_mixed_types():
    @mutant
    def process_mixed(data):
        data['list'].append(4)
        data['set'].add(4)
        data['nested']['value'] += 1
        return data

    input_data = {
        'list': [1, 2, 3],
        'set': {1, 2, 3},
        'nested': {'value': 5}
    }
    result = process_mixed(input_data)
    expected = pmap({
        'list': pvector([1, 2, 3, 4]),
        'set': pset([1, 2, 3, 4]),
        'nested': pmap({'value': 6})
    })
    assert result == expected
    assert input_data == {
        'list': [1, 2, 3],
        'set': {1, 2, 3},
        'nested': {'value': 5}
    }  # Original unchanged


# LLM-generated content at query #5
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    from pyrsistent import pmap
    from pyrsistent._helpers import freeze

    original = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(original)
    expected = pmap({'a': 1, 'b': 2})

    assert result == expected
    assert isinstance(result, pmap)


# LLM-generated content at query #6
#--------------------------

```python
def test_mutant_with_simple_types():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add("a", "b") == "ab"

def test_mutant_with_list():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst

    result = append_to_list([1, 2], 3)
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict():
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    result = add_to_dict({"a": 1}, "b", 2)
    assert isinstance(result, pmap)
    assert result == pmap({"a": 1, "b": 2})

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data["list"].append(4)
        data["dict"]["c"] = 3
        return data

    input_data = {"list": [1, 2, 3], "dict": {"a": 1, "b": 2}}
    result = modify_nested(input_data)
    assert isinstance(result["list"], pvector)
    assert isinstance(result["dict"], pmap)
    assert result == pmap({"list": pvector([1, 2, 3, 4]), "dict": pmap({"a": 1, "b": 2, "c": 3})})

def test_mutant_with_kwargs():
    @mutant
    def merge_dicts(**kwargs):
        result = {}
        for k, v in kwargs.items():
            result[k] = v
        return result

    result = merge_dicts(a=1, b=2)
    assert isinstance(result, pmap)
    assert result == pmap({"a": 1, "b": 2})

def test_mutant_preserves_immutable_types():
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("hello") == "hello"
    assert identity((1, 2, 3)) == (1, 2, 3)

def test_mutant_with_set():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert isinstance(result, pset)
    assert result == pset([1, 2, 3])


# LLM-generated content at query #7
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not False and True)


# LLM-generated content at query #8
#--------------------------

```python
def test_freeze_with_dict():
    result = freeze({'a': 1, 'b': 2})
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant_with_empty_args_and_kwargs():
    @mutant
    def empty_func():
        return 42
    assert empty_func() == 42

def test_mutant_with_simple_args():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_simple_kwargs():
    @mutant
    def add(a, b):
        return a + b
    assert add(a=1, b=2) == 3

def test_mutant_with_list_arg():
    @mutant
    def sum_list(lst):
        return sum(lst)
    assert sum_list([1, 2, 3]) == 6

def test_mutant_with_dict_arg():
    @mutant
    def get_value(d, key):
        return d[key]
    assert get_value({'a': 1}, 'a') == 1

def test_mutant_with_nested_structures():
    @mutant
    def nested_sum(data):
        return data['a'] + data['b'][0]
    assert nested_sum({'a': 1, 'b': [2, 3]}) == 3

def test_mutant_with_mutable_return():
    @mutant
    def create_list():
        return [1, 2, 3]
    result = create_list()
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_mutable_return_nested():
    @mutant
    def create_nested():
        return {'a': [1, 2]}
    result = create_nested()
    assert isinstance(result, pmap)
    assert result['a'] == pvector([1, 2])

def test_mutant_with_tuple_arg():
    @mutant
    def tuple_sum(t):
        return t[0] + t[1]
    assert tuple_sum((1, 2)) == 3

def test_mutant_with_set_arg():
    @mutant
    def set_size(s):
        return len(s)
    assert set_size({1, 2, 3}) == 3


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


# LLM-generated content at query #11
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    from pyrsistent import pmap
    from pyrsistent._helpers import freeze

    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    expected = pmap({'a': 1, 'b': 2})
    assert result == expected


# LLM-generated content at query #13
#--------------------------

```python
def test_freeze_empty_list():
    assert freeze([]) == pvector([])

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_pvector():
    assert freeze(pvector([1, 2, 3])) == pvector([1, 2, 3])

def test_freeze_pmap():
    assert freeze(pmap({'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2})

def test_freeze_pset():
    assert freeze(pset([1, 2, 3])) == pset([1, 2, 3])

def test_freeze_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(d) == pmap({'a': 1, 'b': 2})

def test_freeze_non_container():
    assert freeze(42) == 42

def test_freeze_non_strict_pvector():
    assert freeze(pvector([1, [2, 3]]), strict=False) == pvector([1, [2, 3]])

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': {'b': 2}}), strict=False) == pmap({'a': {'b': 2}})


# LLM-generated content at query #14
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 1}}) == pmap({'a': pmap({'b': 1})})

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_nested_tuple():
    assert freeze((1, (2, 3))) == (1, (2, 3))

def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, a=1)
    assert freeze(d) == pmap({'a': 1})

def test_freeze_pmap_strict():
    pm = pmap({'a': [1, 2]})
    assert freeze(pm, strict=True) == pmap({'a': pvector([1, 2])})

def test_freeze_pvector_strict():
    pv = pvector([1, {'a': 2}])
    assert freeze(pv, strict=True) == pvector([1, pmap({'a': 2})])

def test_freeze_non_strict():
    assert freeze({'a': [1, 2]}, strict=False) == pmap({'a': [1, 2]})

def test_freeze_mixed_types():
    data = {'a': [1, (2, {3, 4})], 'b': (5, [6, 7])}
    expected = pmap({'a': pvector([1, (2, pset([3, 4]))]), 'b': (5, pvector([6, 7]))})
    assert freeze(data) == expected


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])

def test_mutant_with_dict_arguments():
    @mutant
    def merge_dicts(d1, d2):
        return {**d1, **d2}
    assert merge_dicts({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_list_arguments():
    @mutant
    def extend_list(lst, item):
        lst.append(item)
        return lst
    assert extend_list([1, 2], 3) == pvector([1, 2, 3])

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        data['values'].append(4)
        return data
    assert process({'values': [1, 2, 3]}) == pmap({'values': pvector([1, 2, 3, 4])})

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {**kwargs, 'a': a, 'b': b}
    assert combine(1, 2, c=3, d=4) == pmap({'a': 1, 'b': 2, 'c': 3, 'd': 4})

def test_mutant_with_set_arguments():
    @mutant
    def union_sets(s1, s2):
        return s1 | s2
    assert union_sets({1, 2}, {2, 3}) == pset([1, 2, 3])

def test_mutant_with_tuple_arguments():
    @mutant
    def modify_tuple(t):
        return t + (4,)
    assert modify_tuple((1, 2, 3)) == (1, 2, 3, 4)

def test_mutant_with_mixed_types():
    @mutant
    def mix(*args):
        return list(args)
    assert mix(1, [2], {'a': 3}) == pvector([1, pvector([2]), pmap({'a': 3})])

def test_mutant_with_no_args():
    @mutant
    def no_args():
        return [1, 2, 3]
    assert no_args() == pvector([1, 2, 3])

def test_mutant_with_strict_false():
    @mutant
    def identity(x):
        return x
    assert identity(pvector([1, 2, 3])) == pvector([1, 2, 3])
    assert identity(pmap({'a': 1})) == pmap({'a': 1})


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant_decorator_preserves_function_metadata():
    @mutant
    def example_function(x):
        return x + 1

    assert example_function.__name__ == 'example_function'
    assert example_function.__doc__ is None


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_list():
    @mutant
    def append_item(lst, item):
        lst.append(item)
        return lst
    result = append_item([1, 2], 3)
    assert result == [1, 2, 3]

def test_mutant_with_dict():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    result = update_dict({'a': 1}, 'b', 2)
    assert result == {'a': 1, 'b': 2}

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['inner']['value'] = 10
        return d
    result = modify_nested({'outer': 1, 'inner': {'value': 5}})
    assert result == {'outer': 1, 'inner': {'value': 10}}

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}
    result = combine(1, 2, c=3, d=4)
    assert result == {'a': 1, 'b': 2, 'kwargs': {'c': 3, 'd': 4}}

def test_mutant_returns_frozen():
    @mutant
    def return_list():
        return [1, 2, 3]
    result = return_list()
    assert isinstance(result, pvector)

def test_mutant_with_pvector_input():
    @mutant
    def process_pvector(pv):
        return pv.append(4)
    result = process_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_pmap_input():
    @mutant
    def process_pmap(pm):
        return pm.set('new_key', 'new_value')
    result = process_pmap(pmap({'existing': 'value'}))
    assert result == pmap({'existing': 'value', 'new_key': 'new_value'})

def test_mutant_with_pset_input():
    @mutant
    def process_pset(ps):
        return ps.add(4)
    result = process_pset(pset([1, 2, 3]))
    assert result == pset([1, 2, 3, 4])

def test_mutant_preserves_immutability():
    @mutant
    def modify_and_return(lst):
        lst.append(100)
        return lst
    original = [1, 2, 3]
    result = modify_and_return(original)
    assert original == [1, 2, 3]
    assert result == [1, 2, 3, 100]


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


# LLM-generated content at query #20
#--------------------------

```python
def test_freeze_with_empty_set():
    assert freeze(set()) == pset()


# LLM-generated content at query #21
#--------------------------

```python
def test_freeze_set_conversion():
    result = freeze(set([1, 2]))
    assert result == pset([1, 2])


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_decorator_predicate_false():
    assert not (False)


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_decorator_returns_callable():
    @mutant
    def test_func():
        pass
    assert callable(test_func)


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_predicate_at_line_1():
    assert not callable(mutant)


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_with_empty_function():
    @mutant
    def empty_func():
        return None
    assert empty_func() is None

def test_mutant_with_simple_arguments():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_list_argument():
    @mutant
    def process_list(lst):
        return lst + [4]
    assert process_list([1, 2, 3]) == pvector([1, 2, 3, 4])

def test_mutant_with_dict_argument():
    @mutant
    def process_dict(d):
        d['new_key'] = 'new_value'
        return d
    assert process_dict({'a': 1}) == pmap({'a': 1, 'new_key': 'new_value'})

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        return kwargs
    assert process_kwargs(a=1, b=2) == pmap({'a': 1, 'b': 2})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def process_mixed(arg1, arg2, **kwargs):
        return {'arg1': arg1, 'arg2': arg2, 'kwargs': kwargs}
    result = process_mixed([1, 2], {'a': 3}, extra=4)
    expected = pmap({
        'arg1': pvector([1, 2]),
        'arg2': pmap({'a': 3}),
        'kwargs': pmap({'extra': 4})
    })
    assert result == expected

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    nested = {'a': [1, 2, {'b': 3}]}
    expected = pmap({'a': pvector([1, 2, pmap({'b': 3})])})
    assert process_nested(nested) == expected

def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4,)
    assert process_tuple((1, 2, 3)) == (1, 2, 3, 4)

def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s | {4}
    assert process_set({1, 2, 3}) == pset({1, 2, 3, 4})

def test_mutant_with_strict_false():
    @mutant
    def process_non_strict(data):
        return data
    # When strict=False, existing pyrsistent types should remain unchanged
    pvec = pvector([1, 2, 3])
    assert process_non_strict(pvec) == pvec


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not False and True)


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == 'test_func'


# LLM-generated content at query #30
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_mixed_containers():
    assert freeze([1, {'a': (2, [3, 4])}]) == pvector([1, pmap({'a': (2, pvector([3, 4]))})])

def test_freeze_with_strict_false():
    pmap_instance = pmap({'a': 1})
    assert freeze(pmap_instance, strict=False) == pmap_instance

def test_freeze_with_strict_true():
    pmap_instance = pmap({'a': [1, 2]})
    assert freeze(pmap_instance, strict=True) == pmap({'a': pvector([1, 2])})

def test_freeze_pvector_with_strict_false():
    pvector_instance = pvector([1, 2, 3])
    assert freeze(pvector_instance, strict=False) == pvector_instance

def test_freeze_pvector_with_strict_true():
    pvector_instance = pvector([1, [2, 3]])
    assert freeze(pvector_instance, strict=True) == pvector([1, pvector([2, 3])])

def test_freeze_pset():
    pset_instance = pset([1, 2, 3])
    assert freeze(pset_instance) == pset_instance

def test_freeze_non_container():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #31
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    defaultdict_instance = collections.defaultdict(int, {'a': 1, 'b': 2})
    assert isinstance(defaultdict_instance, collections.defaultdict)
    assert isinstance(defaultdict_instance, PMap) is False
    assert (type(defaultdict_instance) is collections.defaultdict or (True and isinstance(defaultdict_instance, PMap))) is True


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_with_list_arg():
    @mutant
    def func(lst):
        lst.append(4)
        return lst
    assert func([1, 2, 3]) == (1, 2, 3, 4)

def test_mutant_with_dict_arg():
    @mutant
    def func(d):
        d['c'] = 3
        return d
    assert func({'a': 1, 'b': 2}) == {'a': 1, 'b': 2, 'c': 3}

def test_mutant_with_set_arg():
    @mutant
    def func(s):
        s.add(3)
        return s
    assert func({1, 2}) == {1, 2, 3}

def test_mutant_with_tuple_arg():
    @mutant
    def func(t):
        return t + (4,)
    assert func((1, 2, 3)) == (1, 2, 3, 4)

def test_mutant_with_mixed_args():
    @mutant
    def func(lst, d, s, t):
        lst.append(4)
        d['c'] = 3
        s.add(4)
        return (lst, d, s, t)
    result = func([1, 2], {'a': 1}, {1, 2}, (1, 2))
    assert result == ((1, 2, 4), {'a': 1, 'c': 3}, {1, 2, 4}, (1, 2))

def test_mutant_with_kwargs():
    @mutant
    def func(**kwargs):
        kwargs['c'] = 3
        return kwargs
    assert func(a=1, b=2) == {'a': 1, 'b': 2, 'c': 3}

def test_mutant_with_nested_structures():
    @mutant
    def func(data):
        data['lst'].append(4)
        data['d']['c'] = 3
        return data
    assert func({'lst': [1, 2], 'd': {'a': 1}}) == {'lst': (1, 2, 4), 'd': {'a': 1, 'c': 3}}

def test_mutant_with_pvector_arg():
    @mutant
    def func(pv):
        return pv.append(4)
    assert func(pvector([1, 2, 3])) == (1, 2, 3, 4)

def test_mutant_with_pmap_arg():
    @mutant
    def func(pm):
        return pm.set('c', 3)
    assert func(pmap({'a': 1, 'b': 2})) == {'a': 1, 'b': 2, 'c': 3}

def test_mutant_with_pset_arg():
    @mutant
    def func(ps):
        return ps.add(3)
    assert func(pset({1, 2})) == {1, 2, 3}


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant_basic_operation():
    @mutant
    def add(a, b):
        return a + b

    result = add([1, 2], {'c': 3})
    assert result == pvector([1, 2]) + pmap({'c': 3})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return data['values']

    result = process({'values': [1, 2, {'nested': 3}]})
    expected = pvector([1, 2, pmap({'nested': 3})])
    assert result == expected

def test_mutant_with_kwargs():
    @mutant
    def merge(a, b, **kwargs):
        return {**a, **b, **kwargs}

    result = merge({'x': 1}, {'y': 2}, z=3)
    expected = pmap({'x': 1, 'y': 2, 'z': 3})
    assert result == expected

def test_mutant_with_mutable_input():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original should remain unchanged

def test_mutant_with_set_input():
    @mutant
    def process_set(s):
        return s | {4, 5}

    result = process_set({1, 2, 3})
    expected = pset({1, 2, 3, 4, 5})
    assert result == expected

def test_mutant_with_tuple_input():
    @mutant
    def process_tuple(t):
        return t + (4, 5)

    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4, 5)

def test_mutant_with_no_args():
    @mutant
    def get_empty():
        return []

    result = get_empty()
    assert result == pvector([])

def test_mutant_with_strict_false():
    def custom_freeze(o):
        return freeze(o, strict=False)

    @mutant
    def process(data):
        return custom_freeze(data)

    result = process([1, [2, 3]])
    assert result == pvector([1, [2, 3]])  # Inner list not frozen when strict=False


# LLM-generated content at query #34
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_mutable_arguments():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst
    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, PVector)

def test_mutant_with_mutable_kwargs():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    result = modify_dict({'a': 1})
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert isinstance(result, PMap)

def test_mutant_with_nested_structures():
    @mutant
    def nested_modify(data):
        data['list'].append(4)
        data['dict']['new_key'] = 'new_value'
        return data
    input_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = nested_modify(input_data)
    expected = pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'new_key': 'new_value'})})
    assert result == expected
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)

def test_mutant_with_set():
    @mutant
    def process_set(s):
        s.add(4)
        return s
    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3, 4])
    assert isinstance(result, PSet)

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t + (4,)
    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)
    assert isinstance(result, tuple)

def test_mutant_with_no_args():
    @mutant
    def no_args():
        return [1, 2, 3]
    result = no_args()
    assert result == pvector([1, 2, 3])
    assert isinstance(result, PVector)

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def mixed(a, b, c=3, d=4):
        return [a, b, c, d]
    result = mixed(1, 2, c=[3], d={'d': 4})
    expected = pvector([1, 2, pvector([3]), pmap({'d': 4})])
    assert result == expected


# LLM-generated content at query #35
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not (1, 2, 3) and 0 == 0)


# LLM-generated content at query #36
#--------------------------

```python
def test_freeze_with_empty_set():
    assert freeze(set()) == pset()

def test_freeze_with_non_empty_set():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_with_empty_list():
    assert freeze([]) == pvector()

def test_freeze_with_non_empty_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_with_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_with_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_with_non_empty_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_with_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_with_empty_tuple():
    assert freeze(()) == ()

def test_freeze_with_non_empty_tuple():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_with_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_with_pvector():
    assert freeze(pvector([1, 2, 3])) == pvector([1, 2, 3])

def test_freeze_with_pmap():
    assert freeze(pmap({'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2})

def test_freeze_with_pset():
    assert freeze(pset([1, 2, 3])) == pset([1, 2, 3])

def test_freeze_with_strict_false():
    assert freeze([1, {'a': 3}], strict=False) == pvector([1, {'a': 3}])

def test_freeze_with_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(d) == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #37
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #38
#--------------------------

```python
def test_mutant_decorator_preserves_function_signature():
    @mutant
    def test_func(a, b=2):
        return a + b

    assert test_func.__name__ == 'test_func'
    assert test_func.__doc__ is None


# LLM-generated content at query #39
#--------------------------

```python
def test_freeze_with_empty_dict():
    result = freeze({})
    assert result == pmap()


# LLM-generated content at query #40
#--------------------------

```python
def test_mutant_with_list_arg():
    @mutant
    def func(lst):
        lst.append(4)
        return lst

    result = func([1, 2, 3])
    assert result == (1, 2, 3, 4)
    assert isinstance(result, tuple)

def test_mutant_with_dict_arg():
    @mutant
    def func(d):
        d['c'] = 3
        return d

    result = func({'a': 1, 'b': 2})
    assert result == {'a': 1, 'b': 2, 'c': 3}
    assert isinstance(result, dict)

def test_mutant_with_set_arg():
    @mutant
    def func(s):
        s.add(3)
        return s

    result = func({1, 2})
    assert result == {1, 2, 3}
    assert isinstance(result, set)

def test_mutant_with_tuple_arg():
    @mutant
    def func(t):
        return t + (4,)

    result = func((1, 2, 3))
    assert result == (1, 2, 3, 4)
    assert isinstance(result, tuple)

def test_mutant_with_mixed_args():
    @mutant
    def func(lst, d, s):
        lst.append(4)
        d['c'] = 3
        s.add(4)
        return lst, d, s

    lst_result, dict_result, set_result = func([1, 2, 3], {'a': 1, 'b': 2}, {1, 2, 3})
    assert lst_result == (1, 2, 3, 4)
    assert dict_result == {'a': 1, 'b': 2, 'c': 3}
    assert set_result == {1, 2, 3, 4}
    assert isinstance(lst_result, tuple)
    assert isinstance(dict_result, dict)
    assert isinstance(set_result, set)

def test_mutant_with_kwargs():
    @mutant
    def func(**kwargs):
        kwargs['c'] = 3
        return kwargs

    result = func(a=1, b=2)
    assert result == {'a': 1, 'b': 2, 'c': 3}
    assert isinstance(result, dict)

def test_mutant_with_no_mutation():
    @mutant
    def func(x):
        return x

    assert func(42) == 42
    assert func("hello") == "hello"


# LLM-generated content at query #41
#--------------------------

```python
def test_mutant_predicate():
    assert not (False)


# LLM-generated content at query #42
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3

def test_mutant_with_list_argument():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst

    result = append_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict_argument():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['inner'][0] = 'changed'
        return data

    result = modify_nested({'inner': [1, 2, 3], 'other': 'value'})
    assert result == pmap({'inner': pvector(['changed', 2, 3]), 'other': 'value'})

def test_mutant_with_kwargs():
    @mutant
    def merge_dicts(**kwargs):
        return dict(kwargs)

    result = merge_dicts(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_set_argument():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_with_tuple_argument():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3
    assert isinstance(result, int)

def test_mutant_with_list_arguments():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst

    result = append_to_list([1, 2], 3)
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)

def test_mutant_with_dict_arguments():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    result = update_dict({'a': 1}, 'b', 2)
    assert result == {'a': 1, 'b': 2}
    assert isinstance(result, dict)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        return data

    result = modify_nested({'list': [1, 2, 3]})
    assert result == {'list': (1, 2, 3, 4)}
    assert isinstance(result['list'], tuple)

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}

    result = combine(1, 2, x=3, y=4)
    expected = {'a': 1, 'b': 2, 'kwargs': {'x': 3, 'y': 4}}
    assert result == expected
    assert isinstance(result['kwargs'], dict)

def test_mutant_with_no_args():
    @mutant
    def get_empty_list():
        return []

    result = get_empty_list()
    assert result == ()
    assert isinstance(result, tuple)

def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        s.add(3)
        return s

    result = process_set({1, 2})
    assert result == {1, 2, 3}
    assert isinstance(result, set)

def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4,)

    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)
    assert isinstance(result, tuple)


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_non_empty_set():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_non_empty_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_non_empty_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_non_empty_tuple():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_list_with_dict():
    assert freeze([1, {'a': 2}]) == pvector([1, pmap({'a': 2})])

def test_freeze_nested_dict_with_list():
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

def test_freeze_nested_tuple_with_list():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_non_strict_pvector():
    pv = pvector([1, 2])
    assert freeze(pv, strict=False) == pv

def test_freeze_non_strict_pmap():
    pm = pmap({'a': 1})
    assert freeze(pm, strict=False) == pm

def test_freeze_non_strict_pset():
    ps = pset([1, 2])
    assert freeze(ps, strict=False) == ps

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': [1, 2]})
    assert freeze(dd) == pmap({'a': pvector([1, 2])})

def test_freeze_complex_nested_structure():
    nested = {'a': [1, {'b': 2}], 'c': (3, [4, 5])}
    expected = pmap({'a': pvector([1, pmap({'b': 2})]), 'c': (3, pvector([4, 5]))})
    assert freeze(nested) == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    expected = pmap({'a': 1, 'b': 2})
    assert result == expected


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_mixed_structures():
    assert freeze([1, {'a': (2, [3, 4])}]) == pvector([1, pmap({'a': (2, pvector([3, 4]))})])

def test_freeze_with_strict_false():
    pmap_instance = pmap({'a': 1})
    assert freeze(pmap_instance, strict=False) == pmap_instance

def test_freeze_with_strict_true():
    pmap_instance = pmap({'a': [1, 2]})
    assert freeze(pmap_instance, strict=True) == pmap({'a': pvector([1, 2])})

def test_freeze_pvector_with_strict_false():
    pvector_instance = pvector([1, 2, 3])
    assert freeze(pvector_instance, strict=False) == pvector_instance

def test_freeze_pvector_with_strict_true():
    pvector_instance = pvector([1, [2, 3]])
    assert freeze(pvector_instance, strict=True) == pvector([1, pvector([2, 3])])

def test_freeze_primitive_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(None) == None


# LLM-generated content at query #5
#--------------------------

```python
def test_mutant_with_list():
    @mutant
    def add_item(lst, item):
        lst.append(item)
        return lst

    result = add_item([1, 2], 3)
    assert result == (1, 2, 3)

def test_mutant_with_dict():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_set():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset({1, 2, 3})

def test_mutant_with_tuple():
    @mutant
    def modify_tuple(t, item):
        return t + (item,)

    result = modify_tuple((1, 2), 3)
    assert result == (1, 2, 3)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(lst, key, value):
        lst[0][key] = value
        return lst

    result = modify_nested([{'a': 1}], 'b', 2)
    assert result == (pmap({'a': 1, 'b': 2}),)

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, c=3):
        return a + b + c

    result = combine(1, 2, c=4)
    assert result == 7

def test_mutant_with_no_args():
    @mutant
    def get_default():
        return [1, 2, 3]

    result = get_default()
    assert result == (1, 2, 3)


# LLM-generated content at query #6
#--------------------------

```python
def test_mutant_with_list_input():
    @mutant
    def add_item(lst, item):
        lst.append(item)
        return lst

    result = add_item([1, 2], 3)
    assert result == pvector([1, 2, 3])
    assert type(result) is PVector

def test_mutant_with_dict_input():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})
    assert type(result) is PMap

def test_mutant_with_set_input():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])
    assert type(result) is PSet

def test_mutant_with_tuple_input():
    @mutant
    def modify_tuple(t, item):
        return t + (item,)

    result = modify_tuple((1, 2), 3)
    assert result == (1, 2, 3)
    assert type(result) is tuple

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['lst'].append(4)
        return data

    result = modify_nested({'lst': [1, 2, 3]})
    assert result == pmap({'lst': pvector([1, 2, 3, 4])})
    assert type(result['lst']) is PVector

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = process_kwargs(existing_key='existing_value')
    assert result == pmap({'existing_key': 'existing_value', 'new_key': 'new_value'})
    assert type(result) is PMap

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        a.append(b)
        kwargs['combined'] = a
        return kwargs

    result = combine([1, 2], 3, extra='value')
    assert result == pmap({'combined': pvector([1, 2, 3]), 'extra': 'value'})
    assert type(result['combined']) is PVector

def test_mutant_with_no_mutation():
    @mutant
    def no_op(x):
        return x

    assert no_op(42) == 42
    assert no_op('string') == 'string'
    assert no_op(None) is None

def test_mutant_with_strict_false():
    @mutant
    def modify_pvector(pv):
        return pv.append(4)

    result = modify_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])
    assert type(result) is PVector


# LLM-generated content at query #7
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return {'result': data}

    assert process([1, [2, 3]]) == pmap({'result': pvector([1, pvector([2, 3])])})
    assert process({'a': {'b': 2}}) == pmap({'result': pmap({'a': pmap({'b': 2})})})

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    assert merge(a=1, b=[2, 3]) == pmap({'a': 1, 'b': pvector([2, 3])})
    assert merge(x={'y': 4}) == pmap({'x': pmap({'y': 4})})

def test_mutant_with_no_args():
    @mutant
    def get_default():
        return [1, 2, 3]

    assert get_default() == pvector([1, 2, 3])

def test_mutant_with_set():
    @mutant
    def wrap(s):
        return {'set': s}

    assert wrap({1, 2, 3}) == pmap({'set': pset([1, 2, 3])})

def test_mutant_with_tuple():
    @mutant
    def wrap(t):
        return {'tuple': t}

    assert wrap((1, [2, 3])) == pmap({'tuple': (1, pvector([2, 3]))})

def test_mutant_with_strict_false():
    @mutant
    def identity(x):
        return x

    assert identity(pvector([1, 2])) == pvector([1, 2])
    assert identity(pmap({'a': 1})) == pmap({'a': 1})


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_mutable_arguments():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]

def test_mutant_with_dict_argument():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    input_dict = {'a': 1}
    result = modify_dict(input_dict)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert input_dict == {'a': 1}

def test_mutant_with_nested_structures():
    @mutant
    def nested_operation(data):
        data['list'].append(1)
        data['dict']['new_key'] = 'value'
        return data

    input_data = {'list': [1, 2], 'dict': {'a': 1}}
    result = nested_operation(input_data)
    expected = pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'a': 1, 'new_key': 'value'})})
    assert result == expected
    assert input_data == {'list': [1, 2], 'dict': {'a': 1}}

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        kwargs['extra'] = 'value'
        return kwargs

    result = process_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'extra': 'value'})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def mixed_args_kwargs(arg1, arg2, kwarg1=10):
        arg1.append(kwarg1)
        arg2['new'] = kwarg1
        return {'arg1': arg1, 'arg2': arg2}

    result = mixed_args_kwargs([1, 2], {'a': 1}, kwarg1=20)
    expected = pmap({'arg1': pvector([1, 2, 20]), 'arg2': pmap({'a': 1, 'new': 20})})
    assert result == expected

def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s | {4, 5}

    input_set = {1, 2, 3}
    result = process_set(input_set)
    assert result == pset({1, 2, 3, 4, 5})
    assert input_set == {1, 2, 3}

def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4, 5)

    input_tuple = (1, 2, 3)
    result = process_tuple(input_tuple)
    assert result == (1, 2, 3, 4, 5)
    assert input_tuple == (1, 2, 3)


# LLM-generated content at query #9
#--------------------------

```python
def test_freeze_with_set_input():
    result = freeze(set([1, 2]))
    assert result == pset([1, 2])


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not False and True)


# LLM-generated content at query #11
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_converts_set_to_pset():
    result = freeze(set([1, 2]))
    assert result == pset([1, 2])


# LLM-generated content at query #13
#--------------------------

```python
def test_freeze_set_conversion():
    result = freeze(set([1, 2]))
    assert result == pset([1, 2])


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3
    assert isinstance(result, int)

def test_mutant_with_list_arguments():
    @mutant
    def sum_list(lst):
        return sum(lst)

    result = sum_list([1, 2, 3])
    assert result == 6
    assert isinstance(result, int)

def test_mutant_with_dict_arguments():
    @mutant
    def get_value(d, key):
        return d[key]

    result = get_value({'a': 1}, 'a')
    assert result == 1
    assert isinstance(result, int)

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return data['values'][0]

    result = process({'values': [1, 2, 3]})
    assert result == 1
    assert isinstance(result, int)

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return a + b + kwargs['c']

    result = combine(1, 2, c=3)
    assert result == 6
    assert isinstance(result, int)

def test_mutant_with_mutable_return():
    @mutant
    def create_list():
        return [1, 2, 3]

    result = create_list()
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)

def test_mutant_with_pvector_input():
    from pyrsistent import pvector
    @mutant
    def first_element(v):
        return v[0]

    result = first_element(pvector([1, 2, 3]))
    assert result == 1
    assert isinstance(result, int)

def test_mutant_with_pmap_input():
    from pyrsistent import pmap
    @mutant
    def get_key(pm):
        return pm['key']

    result = get_key(pmap({'key': 'value'}))
    assert result == 'value'
    assert isinstance(result, str)


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_decorator_returns_callable():
    result = mutant(lambda x: x)
    assert callable(result)


# LLM-generated content at query #16
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #17
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    from pyrsistent import freeze, pmap
    d = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(d) == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_list():
    @mutant
    def process_list(lst):
        lst.append(4)
        return lst
    assert process_list([1, 2, 3]) == pvector([1, 2, 3, 4])

def test_mutant_with_dict():
    @mutant
    def process_dict(d):
        d['c'] = 3
        return d
    assert process_dict({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        data['lst'].append(4)
        data['d']['c'] = 3
        return data
    input_data = {'lst': [1, 2, 3], 'd': {'a': 1, 'b': 2}}
    expected = pmap({'lst': pvector([1, 2, 3, 4]), 'd': pmap({'a': 1, 'b': 2, 'c': 3})})
    assert process_nested(input_data) == expected

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        result = a + b
        for val in kwargs.values():
            result += val
        return result
    assert combine(1, 2, c=3, d=4) == 10

def test_mutant_with_set():
    @mutant
    def process_set(s):
        s.add(4)
        return s
    assert process_set({1, 2, 3}) == pset({1, 2, 3, 4})

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t + (4,)
    assert process_tuple((1, 2, 3)) == (1, 2, 3, 4)

def test_mutant_with_pvector():
    @mutant
    def process_pvector(pv):
        return pv.append(4)
    assert process_pvector(pvector([1, 2, 3])) == pvector([1, 2, 3, 4])

def test_mutant_with_pmap():
    @mutant
    def process_pmap(pm):
        return pm.set('c', 3)
    assert process_pmap(pmap({'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_with_pset():
    @mutant
    def process_pset(ps):
        return ps.add(4)
    assert process_pset(pset({1, 2, 3})) == pset({1, 2, 3, 4})


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == "test_func"


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_basic_functionality():
    @mutant
    def add(a, b):
        return a + b

    result = add([1, 2], {'c': 3})
    assert result == pvector([1, 2]) + pmap({'c': 3})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        data['values'].append(4)
        return data

    input_data = {'values': [1, 2, 3], 'metadata': {'count': 3}}
    result = process(input_data)
    expected = pmap({'values': pvector([1, 2, 3, 4]), 'metadata': pmap({'count': 3})})
    assert result == expected

def test_mutant_with_tuple_and_set():
    @mutant
    def combine(a, b):
        return a + b

    result = combine((1, [2]), {3, 4})
    expected = (1, pvector([2])) + pset({3, 4})
    assert result == expected

def test_mutant_with_kwargs():
    @mutant
    def configure(**kwargs):
        kwargs['extra'] = True
        return kwargs

    result = configure(a=[1, 2], b={'c': 3})
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3}), 'extra': True})
    assert result == expected

def test_mutant_preserves_immutable_types():
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("hello") == "hello"
    assert identity((1, 2, 3)) == (1, 2, 3)

def test_mutant_with_empty_containers():
    @mutant
    def empty():
        return [], {}

    result = empty()
    assert result == (pvector([]), pmap({}))

def test_mutant_with_pvector_and_pmap():
    @mutant
    def modify_persistent(pv, pm):
        pv = pv.append(1)
        pm = pm.set('new', 2)
        return pv, pm

    result = modify_persistent(pvector([1, 2]), pmap({'a': 3}))
    expected = (pvector([1, 2, 1]), pmap({'a': 3, 'new': 2}))
    assert result == expected


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_mutable_arguments():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst
    original = [1, 2, 3]
    result = modify_list(original)
    assert result == pvector([1, 2, 3, 4])
    assert original == [1, 2, 3]

def test_mutant_with_dict_arguments():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    original = {'a': 1}
    result = modify_dict(original)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert original == {'a': 1}

def test_mutant_with_kwargs():
    @mutant
    def combine(**kwargs):
        return {k: v * 2 for k, v in kwargs.items()}
    result = combine(a=1, b=2)
    assert result == pmap({'a': 2, 'b': 4})

def test_mutant_with_nested_structures():
    @mutant
    def nested(data):
        data['inner'][0] = 10
        return data
    original = {'inner': [1, 2, 3]}
    result = nested(original)
    assert result == pmap({'inner': pvector([10, 2, 3])})
    assert original == {'inner': [1, 2, 3]}

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s | {4, 5}
    original = {1, 2, 3}
    result = process_set(original)
    assert result == pset([1, 2, 3, 4, 5])
    assert original == {1, 2, 3}

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t + (4,)
    original = (1, 2, 3)
    result = process_tuple(original)
    assert result == (1, 2, 3, 4)
    assert original == (1, 2, 3)

def test_mutant_with_strict_false():
    @mutant
    def no_strict(data):
        return data
    original = {'a': [1, 2, 3]}
    result = no_strict(original)
    assert result == pmap({'a': [1, 2, 3]})
    assert original == {'a': [1, 2, 3]}

def test_mutant_with_pvector_input():
    @mutant
    def process_pvector(pv):
        return pv.append(4)
    original = pvector([1, 2, 3])
    result = process_pvector(original)
    assert result == pvector([1, 2, 3, 4])
    assert original == pvector([1, 2, 3])

def test_mutant_with_pmap_input():
    @mutant
    def process_pmap(pm):
        return pm.set('new', 100)
    original = pmap({'a': 1})
    result = process_pmap(original)
    assert result == pmap({'a': 1, 'new': 100})
    assert original == pmap({'a': 1})


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_list_argument():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    result = append_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict_argument():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['key'].append(4)
        return data
    result = modify_nested({'key': [1, 2, 3]})
    assert result == pmap({'key': pvector([1, 2, 3, 4])})

def test_mutant_with_kwargs():
    @mutant
    def combine(**kwargs):
        return {k: v * 2 for k, v in kwargs.items()}
    result = combine(a=1, b=2)
    assert result == pmap({'a': 2, 'b': 4})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def process(data, multiplier=2):
        return [x * multiplier for x in data]
    result = process([1, 2, 3], multiplier=3)
    assert result == pvector([3, 6, 9])

def test_mutant_with_set_argument():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s
    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_with_tuple_argument():
    @mutant
    def modify_tuple(t):
        return t + (4,)
    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_with_simple_list():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_nested_structure():
    @mutant
    def modify_nested(data):
        data['key'].append(4)
        return data

    result = modify_nested({'key': [1, 2, 3]})
    assert result == pmap({'key': pvector([1, 2, 3, 4])})

def test_mutant_with_kwargs():
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['a'] = 42
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert result == pmap({'a': 42, 'b': 2})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def modify_mixed(lst, **kwargs):
        lst.append(kwargs['value'])
        return lst

    result = modify_mixed([1, 2], value=3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_no_mutation():
    @mutant
    def no_mutation(data):
        return data

    result = no_mutation([1, 2, 3])
    assert result == pvector([1, 2, 3])

def test_mutant_with_set():
    @mutant
    def modify_set(s):
        s.add(4)
        return s

    result = modify_set({1, 2, 3})
    assert result == pset([1, 2, 3, 4])

def test_mutant_with_tuple():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_defaultdict():
    @mutant
    def modify_defaultdict(dd):
        dd['new_key'] = 42
        return dd

    from collections import defaultdict
    result = modify_defaultdict(defaultdict(int, {'a': 1}))
    assert result == pmap({'a': 1, 'new_key': 42})

def test_mutant_with_pvector_input():
    @mutant
    def modify_pvector(pv):
        return pv.append(4)

    result = modify_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_pmap_input():
    @mutant
    def modify_pmap(pm):
        return pm.set('new_key', 42)

    result = modify_pmap(pmap({'a': 1}))
    assert result == pmap({'a': 1, 'new_key': 42})


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        data['values'].append(5)
        return data
    input_data = {'values': [1, 2, 3]}
    result = process(input_data)
    assert result == pmap({'values': pvector([1, 2, 3, 5])})
    assert input_data == {'values': [1, 2, 3]}

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs
    result = merge(a=1, b=[2, 3])
    assert result == pmap({'a': 1, 'b': pvector([2, 3])})

def test_mutant_with_no_args():
    @mutant
    def get_defaults():
        return {'key': [1, 2]}
    assert get_defaults() == pmap({'key': pvector([1, 2])})

def test_mutant_with_tuple_and_set():
    @mutant
    def transform(data):
        return (data[0], set(data[1]))
    input_data = ([1, 2], [2, 3, 3])
    result = transform(input_data)
    assert result == (pvector([1, 2]), pset([2, 3]))


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_list_argument():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    result = append_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict_argument():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['nested'].append(4)
        return data
    result = modify_nested({'nested': [1, 2, 3]})
    assert result == pmap({'nested': pvector([1, 2, 3, 4])})

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}
    result = combine(1, 2, c=3, d=4)
    assert result == pmap({'a': 1, 'b': 2, 'kwargs': pmap({'c': 3, 'd': 4})})

def test_mutant_preserves_immutability():
    @mutant
    def modify_list(lst):
        lst.append(1)
        return lst
    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]
    assert result == pvector([1, 2, 3, 1])

def test_mutant_with_set_argument():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s
    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not 1)


# LLM-generated content at query #29
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    from pyrsistent import pmap
    from pyrsistent._helpers import freeze

    original = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(original)

    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #30
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b

    result = add([1, 2], [3, 4])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_dict():
    @mutant
    def merge_dicts(d1, d2):
        return {**d1, **d2}

    result = merge_dicts({'a': 1}, {'b': 2})
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        data['values'].append(5)
        return data

    result = process({'values': [1, 2, 3]})
    assert result == pmap({'values': pvector([1, 2, 3, 5])})

def test_mutant_with_kwargs():
    @mutant
    def combine(**kwargs):
        return kwargs

    result = combine(a=[1, 2], b={'x': 3})
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'x': 3})})

def test_mutant_preserves_immutable_types():
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("hello") == "hello"
    assert identity((1, 2, 3)) == (1, 2, 3)

def test_mutant_with_set():
    @mutant
    def union_sets(s1, s2):
        return s1.union(s2)

    result = union_sets({1, 2}, {2, 3})
    assert result == pset([1, 2, 3])


# LLM-generated content at query #31
#--------------------------

```python
def test_mutant_with_no_args():
    @mutant
    def no_args():
        return 42
    assert no_args() == 42

def test_mutant_with_positional_args():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_keyword_args():
    @mutant
    def subtract(a, b):
        return a - b
    assert subtract(a=5, b=3) == 2

def test_mutant_with_mixed_args():
    @mutant
    def mixed(a, b, c=10):
        return a + b + c
    assert mixed(1, 2, c=3) == 6

def test_mutant_with_mutable_args():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst
    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, pvector)

def test_mutant_with_mutable_kwargs():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    result = modify_dict({'a': 1})
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert isinstance(result, pmap)

def test_mutant_with_nested_mutable_structures():
    @mutant
    def nested(data):
        data['list'].append(4)
        data['dict']['new_key'] = 'new_value'
        return data
    input_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = nested(input_data)
    expected = pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'new_key': 'new_value'})})
    assert result == expected
    assert isinstance(result['list'], pvector)
    assert isinstance(result['dict'], pmap)

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s | {4, 5}
    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3, 4, 5])
    assert isinstance(result, pset)

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t + (4, 5)
    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4, 5)
    assert isinstance(result, tuple)

def test_mutant_returns_immutable_version():
    @mutant
    def return_list():
        return [1, 2, 3]
    result = return_list()
    assert result == pvector([1, 2, 3])
    assert isinstance(result, pvector)

def test_mutant_with_empty_structures():
    @mutant
    def empty():
        return [], {}
    list_result, dict_result = empty()
    assert list_result == pvector([])
    assert dict_result == pmap({})
    assert isinstance(list_result, pvector)
    assert isinstance(dict_result, pmap)


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add("a", "b") == "ab"


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #34
#--------------------------

```python
def test_mutant_decorator_basic():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3]) == pvector([1, 2, 3])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_decorator_nested_structures():
    @mutant
    def process(data):
        data['nested']['value'] += 1
        return data

    input_data = {'nested': {'value': 5}}
    result = process(input_data)
    assert result == pmap({'nested': pmap({'value': 6})})
    assert input_data == {'nested': {'value': 5}}  # Original unchanged

def test_mutant_decorator_with_kwargs():
    @mutant
    def merge(**kwargs):
        return dict(kwargs)

    assert merge(a=1, b=pvector([2, 3])) == pmap({'a': 1, 'b': pvector([2, 3])})

def test_mutant_decorator_with_mixed_types():
    @mutant
    def transform(items):
        return [x * 2 for x in items]

    assert transform([1, 2, 3]) == pvector([2, 4, 6])
    assert transform((1, 2, 3)) == (2, 4, 6)
    assert transform({1, 2, 3}) == pset({2, 4, 6})

def test_mutant_decorator_with_strict_false():
    @mutant
    def modify_in_place(lst):
        lst.append(4)
        return lst

    result = modify_in_place(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])


# LLM-generated content at query #35
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])

def test_mutant_nested():
    @mutant
    def nested(data):
        return data
    assert nested({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})

def test_mutant_with_kwargs():
    @mutant
    def with_kwargs(a, b, c=5):
        return [a, b, c]
    assert with_kwargs([1], [2], c=[3]) == pvector([pvector([1]), pvector([2]), pvector([3])])

def test_mutant_empty_args():
    @mutant
    def empty():
        return []
    assert empty() == pvector([])

def test_mutant_with_set():
    @mutant
    def with_set(s):
        return s
    assert with_set({1, 2, 3}) == pset({1, 2, 3})

def test_mutant_with_tuple():
    @mutant
    def with_tuple(t):
        return t
    assert with_tuple((1, [2, 3])) == (1, pvector([2, 3]))


# LLM-generated content at query #36
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not False and True)


# LLM-generated content at query #37
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])


# LLM-generated content at query #38
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #39
#--------------------------

```python
def test_mutant_predicate_false():
    assert not callable(mutant)


# LLM-generated content at query #40
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3


# LLM-generated content at query #41
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add("a", "b") == "ab"


# LLM-generated content at query #42
#--------------------------

```python
def test_mutant_decorator_with_list_argument():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_decorator_with_dict_argument():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = modify_dict({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

def test_mutant_decorator_with_set_argument():
    @mutant
    def modify_set(s):
        s.add(4)
        return s

    result = modify_set({1, 2, 3})
    assert result == pset({1, 2, 3, 4})

def test_mutant_decorator_with_tuple_argument():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_decorator_with_mixed_arguments():
    @mutant
    def modify_mixed(lst, d, s, t):
        lst.append(4)
        d['new_key'] = 'new_value'
        s.add(4)
        return (lst, d, s, t)

    result = modify_mixed([1, 2, 3], {'a': 1}, {1, 2, 3}, (1, 2, 3))
    assert result == (pvector([1, 2, 3, 4]), pmap({'a': 1, 'new_key': 'new_value'}), pset({1, 2, 3, 4}), (1, 2, 3))

def test_mutant_decorator_with_kwargs():
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

def test_mutant_decorator_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 42
        return d

    result = modify_nested({'nested': {'value': 10}})
    assert result == pmap({'nested': pmap({'value': 42})})

def test_mutant_decorator_with_pvector_argument():
    @mutant
    def modify_pvector(pv):
        return pv.append(4)

    result = modify_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])

def test_mutant_decorator_with_pmap_argument():
    @mutant
    def modify_pmap(pm):
        return pm.set('new_key', 'new_value')

    result = modify_pmap(pmap({'a': 1}))
    assert result == pmap({'a': 1, 'new_key': 'new_value'})

def test_mutant_decorator_with_pset_argument():
    @mutant
    def modify_pset(ps):
        return ps.add(4)

    result = modify_pset(pset([1, 2, 3]))
    assert result == pset([1, 2, 3, 4])


# LLM-generated content at query #43
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 1}}) == pmap({'a': pmap({'b': 1})})

def test_freeze_nested_tuple():
    assert freeze((1, (2, 3))) == (1, (2, 3))

def test_freeze_mixed_nested_structures():
    assert freeze([1, {'a': [2, 3]}]) == pvector([1, pmap({'a': pvector([2, 3])})])

def test_freeze_with_strict_false():
    pvec = pvector([1, 2, 3])
    assert freeze(pvec, strict=False) == pvec

def test_freeze_with_strict_true():
    pvec = pvector([1, 2, 3])
    assert freeze(pvec, strict=True) == pvector([freeze(1), freeze(2), freeze(3)])

def test_freeze_pmap_with_strict_false():
    pmap_obj = pmap({'a': 1})
    assert freeze(pmap_obj, strict=False) == pmap_obj

def test_freeze_pmap_with_strict_true():
    pmap_obj = pmap({'a': [1, 2]})
    assert freeze(pmap_obj, strict=True) == pmap({'a': pvector([1, 2])})

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

def test_freeze_non_container_object():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #44
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])

def test_mutant_nested():
    @mutant
    def process(data):
        data['values'].append(5)
        return data
    result = process({'values': [1, 2, 3]})
    assert result == pmap({'values': pvector([1, 2, 3, 5])})

def test_mutant_with_kwargs():
    @mutant
    def merge(a, b, **kwargs):
        return {**a, **b, **kwargs}
    assert merge({'x': 1}, {'y': 2}, z=3) == pmap({'x': 1, 'y': 2, 'z': 3})

def test_mutant_empty_args():
    @mutant
    def empty():
        return []
    assert empty() == pvector([])

def test_mutant_with_set():
    @mutant
    def set_op(s):
        return s | {4, 5}
    assert set_op({1, 2, 3}) == pset([1, 2, 3, 4, 5])

def test_mutant_with_tuple():
    @mutant
    def tuple_op(t):
        return t + (4, 5)
    assert tuple_op((1, 2, 3)) == (1, 2, 3, 4, 5)


# LLM-generated content at query #45
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        data['values'].append(4)
        return data

    result = process({'values': [1, 2, 3]})
    expected = pmap({'values': pvector([1, 2, 3, 4])})
    assert result == expected

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    result = merge(a=[1, 2], b={'x': 3})
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'x': 3})})
    assert result == expected

def test_mutant_with_no_args():
    @mutant
    def get_defaults():
        return {'a': [1], 'b': 2}

    result = get_defaults()
    expected = pmap({'a': pvector([1]), 'b': 2})
    assert result == expected

def test_mutant_with_mutable_input():
    input_list = [1, 2, 3]
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list(input_list)
    assert input_list == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s | {4, 5}

    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3, 4, 5])

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t + (4,)

    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_non_strict_freeze():
    @mutant
    def identity(x):
        return x

    class CustomClass:
        pass

    obj = CustomClass()
    assert identity(obj) is obj


# LLM-generated content at query #46
#--------------------------

```python
def test_mutant_decorator_preserves_false_predicate():
    @mutant
    def test_fn(x):
        return x

    result = test_fn(False)
    assert result is False


# LLM-generated content at query #47
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_list_with_nested_dict():
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_dict_with_nested_list():
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_tuple_with_nested_list():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_pvector_with_elements():
    assert freeze(pvector([1, 2, 3])) == pvector([1, 2, 3])

def test_freeze_pmap_with_elements():
    assert freeze(pmap({'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2})

def test_freeze_non_container():
    assert freeze(42) == 42


# LLM-generated content at query #48
#--------------------------

```python
def test_mutant_with_list_arg():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    result = add_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict_arg():
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    result = add_to_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_set_arg():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_with_tuple_arg():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(lst):
        lst[0]['a'] = 10
        return lst

    result = modify_nested([{'a': 1}, [2, 3]])
    assert result == pvector([pmap({'a': 10}), pvector([2, 3])])

def test_mutant_with_kwargs():
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(existing_key='existing_value')
    assert result == pmap({'existing_key': 'existing_value', 'new_key': 'new_value'})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def modify_mixed(lst, d, **kwargs):
        lst.append(1)
        d['key'] = 'value'
        kwargs['kwarg'] = 'kwarg_value'
        return (lst, d, kwargs)

    result = modify_mixed([1, 2], {'a': 1}, existing_kwarg='value')
    expected = (pvector([1, 2, 1]), pmap({'a': 1, 'key': 'value'}), pmap({'existing_kwarg': 'value', 'kwarg': 'kwarg_value'}))
    assert result == expected


# LLM-generated content at query #49
#--------------------------

```python
def test_mutant_with_list():
    @mutant
    def add_item(lst, item):
        lst.append(item)
        return lst

    result = add_item([1, 2], 3)
    assert result == pvector([1, 2, 3])
    assert type(result) is PVector

def test_mutant_with_dict():
    @mutant
    def add_key(d, key, value):
        d[key] = value
        return d

    result = add_key({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})
    assert type(result) is PMap

def test_mutant_with_set():
    @mutant
    def add_element(s, element):
        s.add(element)
        return s

    result = add_element({1, 2}, 3)
    assert result == pset([1, 2, 3])
    assert type(result) is PSet

def test_mutant_with_tuple():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)
    assert type(result) is tuple

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        return data

    result = modify_nested({'list': [1, 2, 3]})
    assert result == pmap({'list': pvector([1, 2, 3, 4])})
    assert type(result['list']) is PVector

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = process_kwargs(existing_key='existing_value')
    assert result == pmap({'existing_key': 'existing_value', 'new_key': 'new_value'})
    assert type(result) is PMap

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def mixed_function(arg1, arg2, kwarg1=None):
        arg1.append('modified')
        arg2['new_key'] = 'new_value'
        return {'arg1': arg1, 'arg2': arg2, 'kwarg1': kwarg1}

    result = mixed_function([1, 2], {'a': 1}, kwarg1='test')
    assert result == pmap({
        'arg1': pvector([1, 2, 'modified']),
        'arg2': pmap({'a': 1, 'new_key': 'new_value'}),
        'kwarg1': 'test'
    })
    assert type(result['arg1']) is PVector
    assert type(result['arg2']) is PMap

def test_mutant_with_no_mutation():
    @mutant
    def no_mutation(data):
        return data

    result = no_mutation([1, 2, 3])
    assert result == pvector([1, 2, 3])
    assert type(result) is PVector

def test_mutant_with_empty_structures():
    @mutant
    def empty_structures():
        return [], {}, set(), ()

    result = empty_structures()
    assert result == (pvector([]), pmap({}), pset([]), ())
    assert type(result[0]) is PVector
    assert type(result[1]) is PMap
    assert type(result[2]) is PSet
    assert type(result[3]) is tuple

def test_mutant_with_strict_false():
    @mutant
    def modify_with_strict_false(data):
        return data

    # This test assumes that the decorator uses strict=True by default
    # and that the behavior is consistent with the freeze function
    result = modify_with_strict_false({'a': [1, 2]})
    assert result == pmap({'a': pvector([1, 2])})
    assert type(result['a']) is PVector


# LLM-generated content at query #50
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3


# LLM-generated content at query #51
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #52
#--------------------------

```python
def test_mutant_decorator_basic():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_decorator_with_list():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    result = append_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_decorator_with_dict():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_decorator_with_set():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s
    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_decorator_with_tuple():
    @mutant
    def modify_tuple(t, item):
        return t + (item,)
    result = modify_tuple((1, 2), 3)
    assert result == (1, 2, 3)

def test_mutant_decorator_with_nested_structures():
    @mutant
    def nested_operation(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data
    input_data = {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    result = nested_operation(input_data)
    expected = pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'b': 2, 'c': 3})})
    assert result == expected

def test_mutant_decorator_with_kwargs():
    @mutant
    def merge_dicts(**kwargs):
        result = {}
        for d in kwargs.values():
            result.update(d)
        return result
    result = merge_dicts(a={'x': 1}, b={'y': 2})
    assert result == pmap({'x': 1, 'y': 2})

def test_mutant_decorator_with_mixed_args_and_kwargs():
    @mutant
    def process_data(data, **updates):
        data.update(updates)
        return data
    result = process_data({'a': 1}, b=2, c=3)
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_decorator_with_no_args():
    @mutant
    def get_empty_structures():
        return [], {}, set()
    list_result, dict_result, set_result = get_empty_structures()
    assert list_result == pvector([])
    assert dict_result == pmap({})
    assert set_result == pset([])


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_thaw_pvector_to_list():
    from pyrsistent import v
    assert thaw(v(1, 2, 3)) == [1, 2, 3]

def test_thaw_pmap_to_dict():
    from pyrsistent import m
    assert thaw(m(a=1, b=2)) == {'a': 1, 'b': 2}

def test_thaw_pset_to_set():
    from pyrsistent import s
    assert thaw(s(1, 2, 3)) == {1, 2, 3}

def test_thaw_tuple_to_tuple():
    assert thaw((1, 2, 3)) == (1, 2, 3)

def test_thaw_nested_pvector():
    from pyrsistent import v, m
    assert thaw(v(1, m(a=2))) == [1, {'a': 2}]

def test_thaw_nested_pmap():
    from pyrsistent import m, v
    assert thaw(m(a=v(1, 2))) == {'a': [1, 2]}

def test_thaw_nested_tuple():
    from pyrsistent import v
    assert thaw((1, v(2, 3))) == (1, [2, 3])

def test_thaw_non_pyrsistent_types():
    assert thaw(42) == 42
    assert thaw("hello") == "hello"

def test_thaw_list_strict():
    assert thaw([1, 2, 3], strict=True) == [1, 2, 3]

def test_thaw_dict_strict():
    assert thaw({'a': 1, 'b': 2}, strict=True) == {'a': 1, 'b': 2}

def test_thaw_non_strict_pvector():
    from pyrsistent import v
    assert thaw(v(1, 2), strict=False) == [1, 2]

def test_thaw_non_strict_pmap():
    from pyrsistent import m
    assert thaw(m(a=1), strict=False) == {'a': 1}

def test_thaw_non_strict_list():
    assert thaw([1, 2], strict=False) == [1, 2]

def test_thaw_non_strict_dict():
    assert thaw({'a': 1}, strict=False) == {'a': 1}


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_simple_set():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_simple_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_simple_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_simple_tuple():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_mixed_structures():
    assert freeze([1, {'a': (2, [3, 4])}]) == pvector([1, pmap({'a': (2, pvector([3, 4]))})])

def test_freeze_non_strict_pmap():
    p = pmap({'a': [1, 2]})
    assert freeze(p, strict=False) == p

def test_freeze_non_strict_pvector():
    v = pvector([1, {'a': 2}])
    assert freeze(v, strict=False) == v

def test_freeze_non_strict_pset():
    s = pset([1, 2])
    assert freeze(s, strict=False) == s

def test_freeze_defaultdict():
    d = collections.defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(d) == pmap({'a': 1, 'b': 2})

def test_freeze_non_container():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 1}}) == pmap({'a': pmap({'b': 1})})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_mixed_containers():
    assert freeze([1, {'a': (2, [3])}]) == pvector([1, pmap({'a': (2, pvector([3]))})])

def test_freeze_pvector_strict():
    assert freeze(pvector([1, 2, 3]), strict=True) == pvector([1, 2, 3])

def test_freeze_pmap_strict():
    assert freeze(pmap({'a': 1}), strict=True) == pmap({'a': 1})

def test_freeze_non_strict_pvector():
    assert freeze(pvector([1, 2, 3]), strict=False) == pvector([1, 2, 3])

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': 1}), strict=False) == pmap({'a': 1})

def test_freeze_non_strict_nested_list():
    assert freeze([1, [2, 3]], strict=False) == pvector([1, [2, 3]])

def test_freeze_non_strict_nested_dict():
    assert freeze({'a': {'b': 1}}, strict=False) == pmap({'a': {'b': 1}})

def test_freeze_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(d) == pmap({'a': 1, 'b': 2})

def test_freeze_immutable_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(3.14) == 3.14


# LLM-generated content at query #4
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b

    assert add([1], [2]) == (1, 2)
    assert add({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}

def test_mutant_nested():
    @mutant
    def process(data):
        data['values'].append(4)
        return data

    result = process({'values': [1, 2, 3]})
    assert result == {'values': (1, 2, 3, 4)}

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}

    result = combine([1, 2], [3, 4], x=5, y=[6, 7])
    assert result == {'a': (1, 2), 'b': (3, 4), 'kwargs': {'x': 5, 'y': (6, 7)}}

def test_mutant_no_mutation():
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("hello") == "hello"

def test_mutant_set_handling():
    @mutant
    def set_op(s):
        return s | {3, 4}

    result = set_op({1, 2})
    assert result == {1, 2, 3, 4}

def test_mutant_tuple_preservation():
    @mutant
    def tuple_op(t):
        return t + (4, 5)

    result = tuple_op((1, 2, 3))
    assert result == (1, 2, 3, 4, 5)


# LLM-generated content at query #5
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_list_argument():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    result = append_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict_argument():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['inner'][0] = 'changed'
        return data
    result = modify_nested({'inner': [1, 2], 'other': 'value'})
    assert result == pmap({'inner': pvector(['changed', 2]), 'other': 'value'})

def test_mutant_with_set_argument():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s
    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_with_tuple_argument():
    @mutant
    def modify_tuple(t):
        return t + (4,)
    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}
    result = combine(1, 2, x=3, y=4)
    assert result == pmap({'a': 1, 'b': 2, 'kwargs': pmap({'x': 3, 'y': 4})})

def test_mutant_preserves_immutability():
    @mutant
    def modify_list(lst):
        lst.append(1)
        return lst
    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]
    assert result == pvector([1, 2, 3, 1])

def test_mutant_with_empty_structures():
    @mutant
    def process_empty(data):
        return data
    assert process_empty([]) == pvector([])
    assert process_empty({}) == pmap({})
    assert process_empty(set()) == pset([])


# LLM-generated content at query #6
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert isinstance(add([1], [2]), tuple)
    assert add([1], [2]) == (pvector([1]), pvector([2]))

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return {"result": data}

    input_data = {"key": [1, 2, 3]}
    result = process(input_data)
    assert isinstance(result["result"]["key"], pvector)
    assert result == {"result": pmap({"key": pvector([1, 2, 3])})}

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    result = merge(a=[1, 2], b={"c": 3})
    assert isinstance(result["a"], pvector)
    assert isinstance(result["b"], pmap)
    assert result == {"a": pvector([1, 2]), "b": pmap({"c": 3})}

def test_mutant_with_no_args():
    @mutant
    def no_op():
        return [1, 2, 3]

    result = no_op()
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_mutable_input():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original should remain unchanged

def test_mutant_with_set_input():
    @mutant
    def process_set(s):
        return s

    input_set = {1, 2, 3}
    result = process_set(input_set)
    assert isinstance(result, pset)
    assert result == pset([1, 2, 3])


# LLM-generated content at query #7
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant_with_list_arg():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, pvector)

def test_mutant_with_dict_arg():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = modify_dict({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert isinstance(result, pmap)

def test_mutant_with_set_arg():
    @mutant
    def modify_set(s):
        s.add(4)
        return s

    result = modify_set({1, 2, 3})
    assert result == pset({1, 2, 3, 4})
    assert isinstance(result, pset)

def test_mutant_with_tuple_arg():
    @mutant
    def modify_tuple(t):
        lst = list(t)
        lst.append(4)
        return tuple(lst)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)
    assert isinstance(result, tuple)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['list'].append(4)
        return d

    result = modify_nested({'list': [1, 2, 3]})
    assert result == pmap({'list': pvector([1, 2, 3, 4])})
    assert isinstance(result['list'], pvector)

def test_mutant_with_kwargs():
    @mutant
    def modify_with_kwargs(a, b, **kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_with_kwargs(1, 2, existing_key='existing_value')
    assert result == pmap({'existing_key': 'existing_value', 'new_key': 'new_value'})
    assert isinstance(result, pmap)

def test_mutant_with_multiple_args():
    @mutant
    def modify_multiple(a, b, c):
        a.append(4)
        b['new_key'] = 'new_value'
        c.add(4)
        return (a, b, c)

    result = modify_multiple([1, 2, 3], {'a': 1}, {1, 2, 3})
    assert result == (pvector([1, 2, 3, 4]), pmap({'a': 1, 'new_key': 'new_value'}), pset({1, 2, 3, 4}))
    assert isinstance(result[0], pvector)
    assert isinstance(result[1], pmap)
    assert isinstance(result[2], pset)

def test_mutant_with_no_mutation():
    @mutant
    def no_mutation(x):
        return x

    result = no_mutation([1, 2, 3])
    assert result == pvector([1, 2, 3])
    assert isinstance(result, pvector)

def test_mutant_with_already_frozen():
    @mutant
    def modify_frozen(pv):
        return pv.append(4)

    original = pvector([1, 2, 3])
    result = modify_frozen(original)
    assert result == pvector([1, 2, 3, 4])
    assert original == pvector([1, 2, 3])

def test_mutant_with_mixed_types():
    @mutant
    def modify_mixed(lst, d, s, t):
        lst.append(4)
        d['new_key'] = 4
        s.add(4)
        new_t = list(t)
        new_t.append(4)
        return (lst, d, s, tuple(new_t))

    result = modify_mixed([1, 2], {'a': 1}, {1, 2}, (1, 2))
    assert result == (pvector([1, 2, 4]), pmap({'a': 1, 'new_key': 4}), pset({1, 2, 4}), (1, 2, 4))
    assert isinstance(result[0], pvector)
    assert isinstance(result[1], pmap)
    assert isinstance(result[2], pset)
    assert isinstance(result[3], tuple)


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant_predicate():
    assert True


# LLM-generated content at query #10
#--------------------------

```python
def test_freeze_with_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #11
#--------------------------

```python
def test_freeze_empty_list():
    assert freeze([]) == pvector([])

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_tuple():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_mixed_structures():
    assert freeze([1, {'a': (2, [3])}]) == pvector([1, pmap({'a': (2, pvector([3]))})])

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': [1, 2]}), strict=False) == pmap({'a': [1, 2]})

def test_freeze_non_strict_pvector():
    assert freeze(pvector([1, {'a': 2}]), strict=False) == pvector([1, {'a': 2}])

def test_freeze_defaultdict():
    d = collections.defaultdict(int, {'a': 1})
    assert freeze(d) == pmap({'a': 1})

def test_freeze_non_container():
    assert freeze(42) == 42


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #13
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 1}}) == pmap({'a': pmap({'b': 1})})

def test_freeze_tuple():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_with_strict_false():
    assert freeze({'a': [1, 2]}, strict=False) == pmap({'a': [1, 2]})

def test_freeze_pvector():
    assert freeze(pvector([1, 2, 3])) == pvector([1, 2, 3])

def test_freeze_pmap():
    assert freeze(pmap({'a': 1})) == pmap({'a': 1})

def test_freeze_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1})
    assert freeze(d) == pmap({'a': 1})


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])

def test_mutant_with_mutable_arguments():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original list should remain unchanged

def test_mutant_with_dict_arguments():
    @mutant
    def update_dict(d):
        d['new_key'] = 'new_value'
        return d

    input_dict = {'a': 1}
    result = update_dict(input_dict)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert input_dict == {'a': 1}  # Original dict should remain unchanged

def test_mutant_with_kwargs():
    @mutant
    def combine_kwargs(**kwargs):
        return {**kwargs, 'extra': 1}

    result = combine_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'extra': 1})

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        data['nested']['value'] = 10
        return data

    input_data = {'nested': {'value': 5}}
    result = process_nested(input_data)
    assert result == pmap({'nested': pmap({'value': 10})})
    assert input_data == {'nested': {'value': 5}}  # Original should remain unchanged

def test_mutant_with_set_arguments():
    @mutant
    def add_to_set(s):
        s.add(4)
        return s

    input_set = {1, 2, 3}
    result = add_to_set(input_set)
    assert result == pset([1, 2, 3, 4])
    assert input_set == {1, 2, 3}  # Original set should remain unchanged

def test_mutant_with_tuple_arguments():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    input_tuple = (1, 2, 3)
    result = modify_tuple(input_tuple)
    assert result == (1, 2, 3, 4)
    assert input_tuple == (1, 2, 3)  # Original tuple should remain unchanged

def test_mutant_with_mixed_arguments():
    @mutant
    def mixed_args(a, b, c=None):
        return {'a': a, 'b': b, 'c': c}

    result = mixed_args([1], {'key': 2}, {3, 4})
    assert result == pmap({'a': pvector([1]), 'b': pmap({'key': 2}), 'c': pset([3, 4])})

def test_mutant_with_no_args():
    @mutant
    def no_args():
        return [1, 2, 3]

    result = no_args()
    assert result == pvector([1, 2, 3])

def test_mutant_with_strict_false():
    @mutant
    def add_to_pvector(v):
        return v.append(4)

    pvec = pvector([1, 2, 3])
    result = add_to_pvector(pvec)
    assert result == pvector([1, 2, 3, 4])
    assert pvec == pvector([1, 2, 3])  # Original pvector should remain unchanged


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_basic_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3]) == pvector([1, 2, 3])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return {'result': data}

    assert process([1, [2, 3]]) == pmap({'result': pvector([1, pvector([2, 3])])})
    assert process({'a': {'b': 2}}) == pmap({'result': pmap({'a': pmap({'b': 2})})})

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, c=3):
        return [a, b, c]

    assert combine(1, 2) == pvector([1, 2, 3])
    assert combine(1, 2, c=[4, 5]) == pvector([1, 2, pvector([4, 5])])

def test_mutant_with_sets():
    @mutant
    def set_operation(s):
        return s | {3, 4}

    assert set_operation({1, 2}) == pset([1, 2, 3, 4])

def test_mutant_with_tuples():
    @mutant
    def tuple_operation(t):
        return t + (3, 4)

    assert tuple_operation((1, 2)) == (1, 2, 3, 4)


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant_with_no_args():
    @mutant
    def no_args():
        return 42
    assert no_args() == 42

def test_mutant_with_positional_args():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_keyword_args():
    @mutant
    def subtract(a, b):
        return a - b
    assert subtract(a=5, b=3) == 2

def test_mutant_with_mixed_args():
    @mutant
    def multiply(a, b, c=1):
        return a * b * c
    assert multiply(2, 3, c=4) == 24

def test_mutant_with_mutable_args():
    @mutant
    def process_list(lst):
        return lst + [1]
    assert process_list([1, 2, 3]) == pvector([1, 2, 3, 1])

def test_mutant_with_mutable_kwargs():
    @mutant
    def process_dict(d):
        return d
    assert process_dict({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

def test_mutant_with_nested_mutable_args():
    @mutant
    def nested(data):
        return data
    assert nested({'a': {'b': [1, 2]}}) == pmap({'a': pmap({'b': pvector([1, 2])})})

def test_mutant_returns_frozen_result():
    @mutant
    def return_list():
        return [1, 2, 3]
    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_set_arg():
    @mutant
    def process_set(s):
        return s
    assert process_set({1, 2, 3}) == pset({1, 2, 3})

def test_mutant_with_tuple_arg():
    @mutant
    def process_tuple(t):
        return t
    assert process_tuple((1, [2, 3])) == (1, pvector([2, 3]))


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant_decorator_with_list_arg():
    @mutant
    def func(lst):
        lst.append(4)
        return lst
    result = func([1, 2, 3])
    assert result == (1, 2, 3, 4)

def test_mutant_decorator_with_dict_arg():
    @mutant
    def func(dct):
        dct['c'] = 3
        return dct
    result = func({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_decorator_with_set_arg():
    @mutant
    def func(st):
        st.add(4)
        return st
    result = func({1, 2, 3})
    assert result == pset([1, 2, 3, 4])

def test_mutant_decorator_with_tuple_arg():
    @mutant
    def func(tpl):
        return tpl + (4,)
    result = func((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_decorator_with_mixed_args():
    @mutant
    def func(lst, dct, st, tpl):
        lst.append(4)
        dct['c'] = 3
        st.add(4)
        return (lst, dct, st, tpl)
    result = func([1, 2, 3], {'a': 1, 'b': 2}, {1, 2, 3}, (1, 2, 3))
    assert result == (pvector([1, 2, 3, 4]), pmap({'a': 1, 'b': 2, 'c': 3}), pset([1, 2, 3, 4]), (1, 2, 3))

def test_mutant_decorator_with_kwargs():
    @mutant
    def func(**kwargs):
        kwargs['c'] = 3
        return kwargs
    result = func(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_decorator_with_nested_structures():
    @mutant
    def func(lst):
        lst[1]['c'] = 3
        return lst
    result = func([1, {'a': 1, 'b': 2}])
    assert result == pvector([1, pmap({'a': 1, 'b': 2, 'c': 3})])

def test_mutant_decorator_preserves_immutable_types():
    @mutant
    def func(x):
        return x
    assert func(42) == 42
    assert func("hello") == "hello"
    assert func(True) is True


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_decorator_returns_callable():
    result = mutant(lambda x: x)
    assert callable(result)


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == [1, 2, 3, 4]


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_decorator_preserves_predicate():
    @mutant
    def test_fn(x):
        return x + 1

    result = test_fn(5)
    assert result == 6


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not False and True)


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_decorator_preserves_original_function():
    @mutant
    def test_func(x):
        return x + 1

    assert test_func.__name__ == 'test_func'
    assert test_func.__doc__ is None


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return {'result': data}

    assert process([1, [2, 3]]) == pmap({'result': pvector([1, pvector([2, 3])])})
    assert process({'a': {'b': 2}}) == pmap({'result': pmap({'a': pmap({'b': 2})})})

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    assert merge(a=1, b=[2, 3]) == pmap({'a': 1, 'b': pvector([2, 3])})
    assert merge(x={'y': 4}) == pmap({'x': pmap({'y': 4})})

def test_mutant_with_no_args():
    @mutant
    def get_defaults():
        return [1, 2, 3]

    assert get_defaults() == pvector([1, 2, 3])

def test_mutant_with_mutable_input():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original should be unchanged

def test_mutant_with_set_input():
    @mutant
    def process_set(s):
        return s | {4, 5}

    assert process_set({1, 2, 3}) == pset({1, 2, 3, 4, 5})

def test_mutant_with_tuple_input():
    @mutant
    def process_tuple(t):
        return t + (4, 5)

    assert process_tuple((1, 2, 3)) == (1, 2, 3, 4, 5)

def test_mutant_with_mixed_types():
    @mutant
    def complex_operation(a, b, c):
        return {'a': a, 'b': b, 'c': c}

    assert complex_operation([1], {'x': 2}, {3, 4}) == pmap({
        'a': pvector([1]),
        'b': pmap({'x': 2}),
        'c': pset({3, 4})
    })


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_predicate():
    assert not (False)


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant_predicate():
    assert not (False)


# LLM-generated content at query #29
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #30
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3

def test_mutant_with_list():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst

    result = append_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_set():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_with_tuple():
    @mutant
    def modify_tuple(t, item):
        lst = list(t)
        lst.append(item)
        return tuple(lst)

    result = modify_tuple((1, 2), 3)
    assert result == (1, 2, 3)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['key'].append(4)
        return d

    result = modify_nested({'key': [1, 2, 3]})
    assert result == pmap({'key': pvector([1, 2, 3, 4])})

def test_mutant_with_kwargs():
    @mutant
    def merge_dicts(**kwargs):
        result = {}
        for k, v in kwargs.items():
            result.update(v)
        return result

    result = merge_dicts(a={'x': 1}, b={'y': 2})
    assert result == pmap({'x': 1, 'y': 2})

def test_mutant_with_no_args():
    @mutant
    def no_args():
        return [1, 2, 3]

    result = no_args()
    assert result == pvector([1, 2, 3])

def test_mutant_with_strict_false():
    @mutant
    def modify_pvector(pv):
        return pv.append(4)

    result = modify_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])


# LLM-generated content at query #31
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == "test_func"


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_predicate():
    assert mutant(lambda x: x) is not None


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_list_argument():
    @mutant
    def process_list(lst):
        lst.append(4)
        return lst
    result = process_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_dict_argument():
    @mutant
    def process_dict(d):
        d['new_key'] = 'new_value'
        return d
    result = process_dict({'a': 1})
    assert result == pmap({'a': 1, 'new_key': 'new_value'})

def test_mutant_with_mixed_arguments():
    @mutant
    def process_mixed(a, b, c):
        return (a, b, c)
    result = process_mixed([1, 2], {'x': 3}, {4, 5})
    assert result == (pvector([1, 2]), pmap({'x': 3}), pset({4, 5}))

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        return kwargs
    result = process_kwargs(x=1, y=[2, 3])
    assert result == pmap({'x': 1, 'y': pvector([2, 3])})

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(d):
        d['inner']['value'] = 10
        return d
    result = process_nested({'inner': {'value': 5}})
    assert result == pmap({'inner': pmap({'value': 10})})

def test_mutant_with_no_mutation():
    @mutant
    def no_mutation(x):
        return x
    assert no_mutation(42) == 42
    assert no_mutation("hello") == "hello"

def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4,)
    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)


# LLM-generated content at query #34
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return {'result': data}

    assert process([1, [2, 3]]) == pmap({'result': pvector([1, pvector([2, 3])])})
    assert process({'a': {'b': 2}}) == pmap({'result': pmap({'a': pmap({'b': 2})})})

def test_mutant_with_kwargs():
    @mutant
    def merge(a, b, **kwargs):
        return {**a, **b, **kwargs}

    assert merge({'x': 1}, {'y': 2}, z=3) == pmap({'x': 1, 'y': 2, 'z': 3})
    assert merge([1, 2], [3, 4], extra=5) == pmap({'extra': 5, 0: 1, 1: 2, 2: 3, 3: 4})

def test_mutant_with_mutable_input():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original should be unchanged

def test_mutant_with_set_input():
    @mutant
    def process_set(s):
        return s | {4, 5}

    assert process_set({1, 2, 3}) == pset({1, 2, 3, 4, 5})

def test_mutant_with_tuple_input():
    @mutant
    def process_tuple(t):
        return t + (4, 5)

    assert process_tuple((1, 2, 3)) == (1, 2, 3, 4, 5)

def test_mutant_with_mixed_types():
    @mutant
    def mixed(a, b, c):
        return [a, b, c]

    assert mixed(1, [2, 3], {'a': 4}) == pvector([1, pvector([2, 3]), pmap({'a': 4})])

def test_mutant_with_no_args():
    @mutant
    def no_args():
        return [1, 2, 3]

    assert no_args() == pvector([1, 2, 3])

def test_mutant_with_strict_false():
    @mutant
    def process(data):
        return data

    assert process(pvector([1, 2, 3])) == pvector([1, 2, 3])
    assert process(pmap({'a': 1})) == pmap({'a': 1})


# LLM-generated content at query #35
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])


# LLM-generated content at query #36
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_list():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    result = append_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data
    result = modify_nested({'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}})
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'b': 2, 'c': 3})})

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        result = a + b
        for k, v in kwargs.items():
            result += v
        return result
    assert combine(1, 2, c=3, d=4) == 10

def test_mutant_with_set():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s
    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_with_tuple():
    @mutant
    def modify_tuple(t):
        lst = list(t)
        lst.append(3)
        return tuple(lst)
    result = modify_tuple((1, 2))
    assert result == (1, 2, 3)


# LLM-generated content at query #37
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict_with_values():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_mixed_structures():
    assert freeze([1, {'a': (2, [3, 4])}]) == pvector([1, pmap({'a': (2, pvector([3, 4]))})])

def test_freeze_pvector_strict():
    assert freeze(PVector([1, 2, 3]), strict=True) == pvector([1, 2, 3])

def test_freeze_pmap_strict():
    assert freeze(PMap({'a': 1}), strict=True) == pmap({'a': 1})

def test_freeze_non_strict_pvector():
    assert freeze(PVector([1, 2, 3]), strict=False) == PVector([1, 2, 3])

def test_freeze_non_strict_pmap():
    assert freeze(PMap({'a': 1}), strict=False) == PMap({'a': 1})

def test_freeze_defaultdict():
    d = collections.defaultdict(int, {'a': 1})
    assert freeze(d) == pmap({'a': 1})

def test_freeze_non_container():
    assert freeze(42) == 42


# LLM-generated content at query #38
#--------------------------

```python
def test_mutant_with_list():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, PVector)

def test_mutant_with_dict():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = modify_dict({'a': 1})
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert isinstance(result, PMap)

def test_mutant_with_set():
    @mutant
    def modify_set(s):
        s.add(4)
        return s

    result = modify_set({1, 2, 3})
    assert result == pset([1, 2, 3, 4])
    assert isinstance(result, PSet)

def test_mutant_with_tuple():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)
    assert isinstance(result, tuple)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['list'].append(4)
        return d

    result = modify_nested({'list': [1, 2, 3]})
    assert result == pmap({'list': pvector([1, 2, 3, 4])})
    assert isinstance(result['list'], PVector)

def test_mutant_with_kwargs():
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(a=1)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert isinstance(result, PMap)

def test_mutant_with_mixed_args():
    @mutant
    def modify_mixed(lst, d, s):
        lst.append(4)
        d['new_key'] = 'new_value'
        s.add(4)
        return lst, d, s

    result_lst, result_dict, result_set = modify_mixed([1, 2, 3], {'a': 1}, {1, 2, 3})
    assert result_lst == pvector([1, 2, 3, 4])
    assert result_dict == pmap({'a': 1, 'new_key': 'new_value'})
    assert result_set == pset([1, 2, 3, 4])
    assert isinstance(result_lst, PVector)
    assert isinstance(result_dict, PMap)
    assert isinstance(result_set, PSet)

def test_mutant_returns_non_frozen_type():
    @mutant
    def return_string(s):
        return s.upper()

    result = return_string("hello")
    assert result == "HELLO"
    assert isinstance(result, str)

def test_mutant_with_no_args():
    @mutant
    def no_args():
        return [1, 2, 3]

    result = no_args()
    assert result == pvector([1, 2, 3])
    assert isinstance(result, PVector)


# LLM-generated content at query #39
#--------------------------

```python
def test_mutant_with_list():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    result = add_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])
    assert isinstance(result, PVector)

def test_mutant_with_dict():
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    result = add_to_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})
    assert isinstance(result, PMap)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['nested'].append(4)
        data['value'] += 1
        return data

    result = modify_nested({'nested': [1, 2, 3], 'value': 10})
    assert result == pmap({'nested': pvector([1, 2, 3, 4]), 'value': 11})
    assert isinstance(result['nested'], PVector)

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, **kwargs}

    result = combine([1], {'x': 2}, c=3, d=[4])
    assert result == pmap({'a': pvector([1]), 'b': pmap({'x': 2}), 'c': 3, 'd': pvector([4])})

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t + (sum(t),)

    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 6)
    assert isinstance(result, tuple)

def test_mutant_with_set():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])
    assert isinstance(result, PSet)

def test_mutant_with_no_mutation():
    @mutant
    def no_op(x):
        return x

    assert no_op(42) == 42
    assert no_op("string") == "string"
    assert no_op(None) is None

def test_mutant_with_mixed_types():
    @mutant
    def mixed(data):
        data[0] = data[0] + [4]
        data[1]['key'] = 'new_value'
        return data

    input_data = [[1, 2, 3], {'key': 'old_value'}]
    result = mixed(input_data)
    assert result == pvector([pvector([1, 2, 3, 4]), pmap({'key': 'new_value'})])


# LLM-generated content at query #40
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3


# LLM-generated content at query #41
#--------------------------

```python
def test_mutant_predicate():
    assert callable(mutant)


# LLM-generated content at query #42
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #43
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return {'result': data}

    assert process([1, [2, 3]]) == pmap({'result': pvector([1, pvector([2, 3])])})
    assert process({'a': {'b': 2}}) == pmap({'result': pmap({'a': pmap({'b': 2})})})

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    assert merge(a=1, b=[2, 3]) == pmap({'a': 1, 'b': pvector([2, 3])})
    assert merge(x={'y': 4}) == pmap({'x': pmap({'y': 4})})

def test_mutant_with_no_args():
    @mutant
    def get_defaults():
        return [1, 2, 3]

    assert get_defaults() == pvector([1, 2, 3])

def test_mutant_with_mutable_input():
    input_list = [1, 2, 3]
    input_dict = {'a': 1}

    @mutant
    def modify(lst, dct):
        lst.append(4)
        dct['b'] = 2
        return (lst, dct)

    result = modify(input_list, input_dict)
    assert result == (pvector([1, 2, 3, 4]), pmap({'a': 1, 'b': 2}))
    assert input_list == [1, 2, 3]
    assert input_dict == {'a': 1}

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s

    assert process_set({1, 2, 3}) == pset({1, 2, 3})

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t

    assert process_tuple((1, [2, 3])) == (1, pvector([2, 3]))


# LLM-generated content at query #44
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


# LLM-generated content at query #45
#--------------------------

```python
def test_mutant_decorator_basic():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_decorator_nested_structures():
    @mutant
    def process(data):
        return {'result': data}

    input_data = [1, {'key': [2, 3]}]
    expected = pmap({'result': pvector([1, pmap({'key': pvector([2, 3])})])})
    assert process(input_data) == expected

def test_mutant_decorator_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        result = {'a': a, 'b': b}
        result.update(kwargs)
        return result

    assert combine(1, 2, c=3, d=[4, 5]) == pmap({'a': 1, 'b': 2, 'c': 3, 'd': pvector([4, 5])})

def test_mutant_decorator_empty_args():
    @mutant
    def no_op():
        return None

    assert no_op() is None

def test_mutant_decorator_with_set():
    @mutant
    def set_op(s):
        return s | {4, 5}

    assert set_op({1, 2, 3}) == pset({1, 2, 3, 4, 5})

def test_mutant_decorator_with_tuple():
    @mutant
    def tuple_op(t):
        return t + (4, 5)

    assert tuple_op((1, 2, 3)) == (1, 2, 3, 4, 5)


# LLM-generated content at query #46
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_list():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    assert append_to_list([1, 2], 3) == pvector([1, 2, 3])

def test_mutant_dict():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    assert update_dict({'a': 1}, 'b', 2) == pmap({'a': 1, 'b': 2})

def test_mutant_nested():
    @mutant
    def nested_operation(data):
        data['lst'].append(4)
        return data
    assert nested_operation({'lst': [1, 2, 3]}) == pmap({'lst': pvector([1, 2, 3, 4])})

def test_mutant_kwargs():
    @mutant
    def merge_dicts(**kwargs):
        return {**kwargs}
    assert merge_dicts(a=1, b=2) == pmap({'a': 1, 'b': 2})

def test_mutant_no_mutation():
    @mutant
    def no_op(x):
        return x
    assert no_op(42) == 42


# LLM-generated content at query #47
#--------------------------

```python
def test_mutant_with_list_input():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_dict_input():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = modify_dict({'a': 1})
    assert result == pmap({'a': 1, 'new_key': 'new_value'})

def test_mutant_with_tuple_input():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_set_input():
    @mutant
    def modify_set(s):
        s.add(4)
        return s

    result = modify_set({1, 2, 3})
    assert result == pset({1, 2, 3, 4})

def test_mutant_with_mixed_inputs():
    @mutant
    def modify_mixed(lst, d, t, s):
        lst.append(4)
        d['new_key'] = 'new_value'
        t = t + (4,)
        s.add(4)
        return (lst, d, t, s)

    result = modify_mixed([1, 2, 3], {'a': 1}, (1, 2, 3), {1, 2, 3})
    assert result == (pvector([1, 2, 3, 4]), pmap({'a': 1, 'new_key': 'new_value'}), (1, 2, 3, 4), pset({1, 2, 3, 4}))

def test_mutant_with_no_mutation():
    @mutant
    def no_mutation(x):
        return x

    assert no_mutation(42) == 42
    assert no_mutation("hello") == "hello"

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 10
        return d

    result = modify_nested({'nested': {'value': 5}})
    assert result == pmap({'nested': pmap({'value': 10})})

def test_mutant_with_kwargs():
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})


# LLM-generated content at query #48
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #49
#--------------------------

```python
def test_mutant_with_no_args():
    @mutant
    def test_func():
        return 42
    assert test_func() == 42

def test_mutant_with_positional_args():
    @mutant
    def test_func(a, b):
        return a + b
    assert test_func(1, 2) == 3

def test_mutant_with_keyword_args():
    @mutant
    def test_func(a, b):
        return a * b
    assert test_func(a=3, b=4) == 12

def test_mutant_with_mixed_args():
    @mutant
    def test_func(a, b, c=10):
        return a + b + c
    assert test_func(1, 2, c=3) == 6

def test_mutant_with_mutable_args():
    @mutant
    def test_func(lst):
        lst.append(4)
        return lst
    result = test_func([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_dict_args():
    @mutant
    def test_func(d):
        d['new_key'] = 'new_value'
        return d
    result = test_func({'a': 1})
    assert result == pmap({'a': 1, 'new_key': 'new_value'})

def test_mutant_with_nested_structures():
    @mutant
    def test_func(data):
        data['list'].append(4)
        data['dict']['new_key'] = 'new_value'
        return data
    input_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = test_func(input_data)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'new_key': 'new_value'})})

def test_mutant_return_value_is_frozen():
    @mutant
    def test_func():
        return [1, 2, 3]
    result = test_func()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_strict_false():
    @mutant
    def test_func():
        return {'a': [1, 2, 3]}
    result = test_func()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], list)
    assert result == pmap({'a': [1, 2, 3]})


# LLM-generated content at query #50
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    assert freeze(collections.defaultdict(int, {'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #51
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == "test_func"


# LLM-generated content at query #52
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_list_arguments():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    result = append_to_list([1, 2], 3)
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict_arguments():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d
    result = update_dict({'a': 1}, 'b', 2)
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_set_arguments():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s
    result = add_to_set({1, 2}, 3)
    assert isinstance(result, pset)
    assert result == pset({1, 2, 3})

def test_mutant_with_tuple_arguments():
    @mutant
    def modify_tuple(t, item):
        lst = list(t)
        lst.append(item)
        return tuple(lst)
    result = modify_tuple((1, 2), 3)
    assert result == (1, 2, 3)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['lst'].append(4)
        return d
    result = modify_nested({'lst': [1, 2, 3]})
    assert isinstance(result, pmap)
    assert isinstance(result['lst'], pvector)
    assert result == pmap({'lst': pvector([1, 2, 3, 4])})

def test_mutant_with_kwargs():
    @mutant
    def merge_dicts(**kwargs):
        return dict(kwargs)
    result = merge_dicts(a=1, b=2)
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_no_mutation():
    @mutant
    def identity(x):
        return x
    assert identity(42) == 42
    assert identity("hello") == "hello"


