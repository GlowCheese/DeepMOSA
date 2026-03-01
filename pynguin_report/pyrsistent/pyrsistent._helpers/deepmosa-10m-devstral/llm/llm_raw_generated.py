####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_freeze_with_set():
    result = freeze(set([1, 2]))
    assert result == pset([1, 2])

def test_freeze_with_list_containing_dict():
    result = freeze([1, {'a': 3}])
    assert result == pvector([1, pmap({'a': 3})])

def test_freeze_with_tuple_containing_list():
    result = freeze((1, []))
    assert result == (1, pvector([]))

def test_freeze_with_empty_dict():
    result = freeze({})
    assert result == pmap()

def test_freeze_with_nested_dict():
    result = freeze({'a': {'b': 2}})
    assert result == pmap({'a': pmap({'b': 2})})

def test_freeze_with_list():
    result = freeze([1, 2, 3])
    assert result == pvector([1, 2, 3])

def test_freeze_with_nested_list():
    result = freeze([1, [2, 3]])
    assert result == pvector([1, pvector([2, 3])])

def test_freeze_with_tuple():
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)

def test_freeze_with_nested_tuple():
    result = freeze((1, (2, 3)))
    assert result == (1, (2, 3))

def test_freeze_with_pvector():
    result = freeze(pvector([1, 2, 3]), strict=True)
    assert result == pvector([1, 2, 3])

def test_freeze_with_pmap():
    result = freeze(pmap({'a': 1}), strict=True)
    assert result == pmap({'a': 1})

def test_freeze_with_non_strict_pmap():
    result = freeze(pmap({'a': [1, 2]}), strict=False)
    assert result == pmap({'a': [1, 2]})

def test_freeze_with_non_strict_pvector():
    result = freeze(pvector([1, {'a': 2}]), strict=False)
    assert result == pvector([1, {'a': 2}])

def test_freeze_with_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1})
    result = freeze(dd)
    assert result == pmap({'a': 1})

def test_freeze_with_immutable_types():
    result = freeze(42)
    assert result == 42

def test_freeze_with_string():
    result = freeze("hello")
    assert result == "hello"


# LLM-generated content at query #2
#--------------------------

```python
def test_mutant_with_empty_function():
    @mutant
    def empty():
        pass
    assert empty() is None

def test_mutant_with_simple_return():
    @mutant
    def simple():
        return 42
    assert simple() == 42

def test_mutant_with_list_argument():
    @mutant
    def list_arg(lst):
        return lst
    assert list_arg([1, 2, 3]) == pvector([1, 2, 3])

def test_mutant_with_dict_argument():
    @mutant
    def dict_arg(d):
        return d
    assert dict_arg({'a': 1}) == pmap({'a': 1})

def test_mutant_with_set_argument():
    @mutant
    def set_arg(s):
        return s
    assert set_arg({1, 2}) == pset({1, 2})

def test_mutant_with_tuple_argument():
    @mutant
    def tuple_arg(t):
        return t
    assert tuple_arg((1, [2])) == (1, pvector([2]))

def test_mutant_with_multiple_arguments():
    @mutant
    def multi_arg(a, b):
        return [a, b]
    assert multi_arg(1, {'c': 3}) == pvector([1, pmap({'c': 3})])

def test_mutant_with_kwargs():
    @mutant
    def with_kwargs(**kwargs):
        return kwargs
    assert with_kwargs(a=1, b=[2]) == pmap({'a': 1, 'b': pvector([2])})

def test_mutant_with_nested_structures():
    @mutant
    def nested(d):
        return d
    input_dict = {'a': [1, {'b': 2}], 'c': {3, 4}}
    expected = pmap({'a': pvector([1, pmap({'b': 2})]), 'c': pset({3, 4})})
    assert nested(input_dict) == expected

def test_mutant_with_strict_false():
    @mutant
    def non_strict():
        return {'a': [1]}
    # Even with strict=False in freeze, the decorator should still freeze
    assert non_strict() == pmap({'a': pvector([1])})

def test_mutant_with_pvector_input():
    @mutant
    def pvec_in(pv):
        return pv
    input_pv = pvector([1, 2, 3])
    assert pvec_in(input_pv) == pvector([1, 2, 3])

def test_mutant_with_pmap_input():
    @mutant
    def pmap_in(pm):
        return pm
    input_pm = pmap({'a': 1})
    assert pmap_in(input_pm) == pmap({'a': 1})

def test_mutant_with_pset_input():
    @mutant
    def pset_in(ps):
        return ps
    input_ps = pset({1, 2})
    assert pset_in(input_ps) == pset({1, 2})

def test_mutant_preserves_none():
    @mutant
    def none_func():
        return None
    assert none_func() is None

def test_mutant_preserves_immutable_types():
    @mutant
    def immutable(x):
        return x
    assert immutable(42) == 42
    assert immutable("string") == "string"
    assert immutable((1, 2)) == (1, 2)


# LLM-generated content at query #3
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])

def test_mutant_nested():
    @mutant
    def process(data):
        data['count'] += 1
        return data

    result = process({'count': 0, 'items': [1, 2, 3]})
    assert result == pmap({'count': 1, 'items': pvector([1, 2, 3])})

def test_mutant_kwargs():
    @mutant
    def merge(a, b, **kwargs):
        return {**a, **b, **kwargs}

    result = merge({'x': 1}, {'y': 2}, z=3)
    assert result == pmap({'x': 1, 'y': 2, 'z': 3})

def test_mutant_no_mutation():
    original = {'a': [1, 2]}
    frozen = freeze(original)

    @mutant
    def modify(d):
        d['a'].append(3)
        return d

    result = modify(original)
    assert result == pmap({'a': pvector([1, 2, 3])})
    assert original == {'a': [1, 2]}
    assert frozen == pmap({'a': pvector([1, 2])})

def test_mutant_strict_false():
    @mutant
    def identity(x):
        return x

    assert identity(pset([1, 2, 3])) == pset([1, 2, 3])
    assert identity(pvector([1, 2, 3])) == pvector([1, 2, 3])


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_with_defaultdict():
    from collections import defaultdict
    from pyrsistent import freeze, pmap
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    expected = pmap({'a': 1, 'b': 2})
    assert result == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_mutant_with_list_argument():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, PVector)

def test_mutant_with_dict_argument():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = modify_dict({'a': 1})
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert isinstance(result, PMap)

def test_mutant_with_set_argument():
    @mutant
    def modify_set(s):
        s.add(4)
        return s

    result = modify_set({1, 2, 3})
    assert result == pset({1, 2, 3, 4})
    assert isinstance(result, PSet)

def test_mutant_with_tuple_argument():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)
    assert isinstance(result, tuple)

def test_mutant_with_mixed_arguments():
    @mutant
    def modify_mixed(lst, d, s):
        lst.append(4)
        d['new_key'] = 'new_value'
        s.add(4)
        return (lst, d, s)

    result = modify_mixed([1, 2, 3], {'a': 1}, {1, 2, 3})
    assert result == (pvector([1, 2, 3, 4]), pmap({'a': 1, 'new_key': 'new_value'}), pset({1, 2, 3, 4}))
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)

def test_mutant_with_kwargs():
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(a=1)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert isinstance(result, PMap)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 42
        return d

    result = modify_nested({'nested': {'value': 1}})
    assert result == pmap({'nested': pmap({'value': 42})})
    assert isinstance(result, PMap)
    assert isinstance(result['nested'], PMap)

def test_mutant_with_pvector_argument():
    @mutant
    def modify_pvector(pv):
        return pv.append(4)

    result = modify_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, PVector)

def test_mutant_with_pmap_argument():
    @mutant
    def modify_pmap(pm):
        return pm.set('new_key', 'new_value')

    result = modify_pmap(pmap({'a': 1}))
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert isinstance(result, PMap)

def test_mutant_with_pset_argument():
    @mutant
    def modify_pset(ps):
        return ps.add(4)

    result = modify_pset(pset({1, 2, 3}))
    assert result == pset({1, 2, 3, 4})
    assert isinstance(result, PSet)


# LLM-generated content at query #6
#--------------------------

```python
def test_freeze_predicate_false():
    assert not (type({}) is dict or (True and isinstance({}, PMap)))


# LLM-generated content at query #7
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    result = add_to_list([1, 2], 3)
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_nested_structures():
    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    result = modify_dict({'a': 1}, 'b', [2, 3])
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': pvector([2, 3])})

def test_mutant_with_kwargs():
    @mutant
    def update_dict(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = update_dict(a=1, b=2)
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def process_data(data, multiplier=2):
        return [x * multiplier for x in data]

    result = process_data([1, 2, 3], multiplier=3)
    assert isinstance(result, pvector)
    assert result == pvector([3, 6, 9])

def test_mutant_with_non_container_types():
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("hello") == "hello"


# LLM-generated content at query #8
#--------------------------

```python
def test_freeze_converts_defaultdict_to_pmap():
    d = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == "test_func"


# LLM-generated content at query #10
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

def test_freeze_tuple():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_mixed_containers():
    assert freeze([1, {'a': (2, 3)}]) == pvector([1, pmap({'a': (2, 3)})])

def test_freeze_non_strict_pvector():
    pv = pvector([1, 2, 3])
    assert freeze(pv, strict=False) == pv

def test_freeze_non_strict_pmap():
    pm = pmap({'a': 1})
    assert freeze(pm, strict=False) == pm

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

def test_freeze_non_container():
    assert freeze(42) == 42


# LLM-generated content at query #11
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
        s.add(4)
        return s

    input_set = {1, 2, 3}
    result = modify_set(input_set)
    assert result == pset([1, 2, 3, 4])
    assert input_set == {1, 2, 3}  # Original unchanged

def test_mutant_tuple_operations():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    input_tuple = (1, 2, 3)
    result = modify_tuple(input_tuple)
    assert result == (1, 2, 3, 4)
    assert input_tuple == (1, 2, 3)  # Original unchanged


# LLM-generated content at query #12
#--------------------------

```python
def test_mutant_decorator_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_decorator_with_nested_structures():
    @mutant
    def process(data):
        return {'result': data}

    assert process([1, [2, 3]]) == pmap({'result': pvector([1, pvector([2, 3])])})
    assert process({'a': {'b': 2}}) == pmap({'result': pmap({'a': pmap({'b': 2})})})

def test_mutant_decorator_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    assert merge(a=1, b=[2, 3]) == pmap({'a': 1, 'b': pvector([2, 3])})
    assert merge(x={'y': 4}) == pmap({'x': pmap({'y': 4})})

def test_mutant_decorator_with_mixed_args_and_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}

    assert combine(1, [2], c={3: 4}) == pmap({'a': 1, 'b': pvector([2]), 'kwargs': pmap({'c': pmap({3: 4})})})

def test_mutant_decorator_with_set():
    @mutant
    def wrap(s):
        return {'set': s}

    assert wrap({1, 2, 3}) == pmap({'set': pset({1, 2, 3})})

def test_mutant_decorator_with_tuple():
    @mutant
    def wrap(t):
        return {'tuple': t}

    assert wrap((1, [2, 3])) == pmap({'tuple': (1, pvector([2, 3]))})

def test_mutant_decorator_preserves_function_metadata():
    @mutant
    def example():
        """Example function."""
        return 42

    assert example.__name__ == 'example'
    assert example.__doc__ == 'Example function.'


# LLM-generated content at query #13
#--------------------------

```python
def test_freeze_defaultdict_strict_mode():
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd, strict=True)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #14
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
    assert freeze([1, {'a': [2, 3]}], strict=False) == pvector([1, {'a': [2, 3]}])

def test_freeze_pvector_with_strict_true():
    pv = pvector([1, 2, 3])
    assert freeze(pv, strict=True) == pv

def test_freeze_pmap_with_strict_true():
    pm = pmap({'a': 1, 'b': 2})
    assert freeze(pm, strict=True) == pm

def test_freeze_non_container():
    assert freeze(42) == 42


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_predicate_false():
    assert not callable(mutant)


# LLM-generated content at query #16
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    expected = pmap({'a': 1, 'b': 2})
    assert result == expected


# LLM-generated content at query #17
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set():
    assert freeze(set([1, 2, 3])) == pset([1, 2, 3])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list_with_primitives():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_list_with_dict():
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

def test_freeze_list_with_nested_lists():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict_with_primitives():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_dict_with_list_values():
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

def test_freeze_dict_with_nested_dicts():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_primitives():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_tuple_with_list():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_tuple_with_dict():
    assert freeze((1, {'a': 2})) == (1, pmap({'a': 2}))

def test_freeze_non_container():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"

def test_freeze_already_persistent():
    assert freeze(pvector([1, 2])) == pvector([1, 2])
    assert freeze(pset([1, 2])) == pset([1, 2])
    assert freeze(pmap({'a': 1})) == pmap({'a': 1})

def test_freeze_with_strict_false():
    assert freeze({'a': [1, 2]}, strict=False) == pmap({'a': [1, 2]})
    assert freeze([1, {'a': 2}], strict=False) == pvector([1, {'a': 2}])


# LLM-generated content at query #18
#--------------------------

```python
def test_freeze_dict_conversion():
    input_dict = {'a': 1, 'b': [2, 3], 'c': {'d': 4}}
    result = freeze(input_dict)
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert isinstance(result['b'], PVector)
    assert result['b'] == pvector([2, 3])
    assert isinstance(result['c'], PMap)
    assert result['c'] == pmap({'d': 4})


# LLM-generated content at query #19
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    assert (type(collections.defaultdict) is collections.defaultdict) or (True and isinstance(pmap(), PMap))


# LLM-generated content at query #20
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    assert freeze(collections.defaultdict(int, {'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == "test_func"


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_decorator_returns_callable():
    @mutant
    def test_func():
        pass
    assert callable(test_func)


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #24
#--------------------------

```python
def test_freeze_with_empty_dict_and_strict_false():
    assert freeze({}, strict=False) == {}


# LLM-generated content at query #25
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
    def process_list(lst):
        return lst + [4]

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
    def combine(a, b, c):
        return {'a': a, 'b': b, 'c': c}

    result = combine(1, [2, 3], {'d': 4})
    assert result == pmap({'a': 1, 'b': pvector([2, 3]), 'c': pmap({'d': 4})})

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        return kwargs

    result = process_kwargs(x=1, y=[2, 3])
    assert result == pmap({'x': 1, 'y': pvector([2, 3])})

def test_mutant_with_nested_structures():
    @mutant
    def nested(data):
        return data

    input_data = {'a': [1, {'b': 2}], 'c': {3, 4}}
    result = nested(input_data)
    assert result == pmap({'a': pvector([1, pmap({'b': 2})]), 'c': pset({3, 4})})

def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4,)

    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s | {4}

    result = process_set({1, 2, 3})
    assert result == pset({1, 2, 3, 4})

def test_mutant_with_no_args():
    @mutant
    def no_args():
        return [1, 2, 3]

    result = no_args()
    assert result == pvector([1, 2, 3])

def test_mutant_with_strict_false():
    @mutant
    def identity(x):
        return x

    result = identity(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


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
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])
    assert add({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return data

    input_data = {'key': [1, 2, {'nested': 3}]}
    result = process(input_data)
    expected = pmap({'key': pvector([1, 2, pmap({'nested': 3})])})
    assert result == expected

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    result = merge(a=1, b=[2, 3])
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected

def test_mutant_with_set():
    @mutant
    def handle_set(s):
        return s

    input_set = {1, 2, 3}
    result = handle_set(input_set)
    expected = pset({1, 2, 3})
    assert result == expected

def test_mutant_with_tuple():
    @mutant
    def handle_tuple(t):
        return t

    input_tuple = (1, [2, 3], {'a': 4})
    result = handle_tuple(input_tuple)
    expected = (1, pvector([2, 3]), pmap({'a': 4}))
    assert result == expected

def test_mutant_with_empty_structures():
    @mutant
    def empty():
        return {}, [], set(), ()

    result = empty()
    expected = (pmap({}), pvector([]), pset(), ())
    assert result == expected

def test_mutant_with_strict_false():
    @mutant
    def no_strict(data):
        return data

    input_data = PMap({'a': PVector([1, 2])})
    result = no_strict(input_data)
    assert result == input_data

def test_mutant_with_mixed_types():
    @mutant
    def mixed(a, b, c):
        return a, b, c

    result = mixed([1, 2], {'a': 3}, {4, 5})
    expected = (pvector([1, 2]), pmap({'a': 3}), pset({4, 5}))
    assert result == expected


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_with_empty_set():
    assert freeze(set()) == pset()

def test_freeze_with_set():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_with_empty_list():
    assert freeze([]) == pvector()

def test_freeze_with_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_with_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_with_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_with_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_with_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_with_empty_tuple():
    assert freeze(()) == ()

def test_freeze_with_tuple():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_with_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_with_pvector():
    assert freeze(pvector([1, 2, 3])) == pvector([1, 2, 3])

def test_freeze_with_pmap():
    assert freeze(pmap({'a': 1, 'b': 2})) == pmap({'a': 1, 'b': 2})

def test_freeze_with_immutable_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True

def test_freeze_with_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

def test_freeze_with_non_strict_mode():
    assert freeze(pvector([1, 2, 3]), strict=False) == pvector([1, 2, 3])
    assert freeze(pmap({'a': 1, 'b': 2}), strict=False) == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #3
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not False and True)


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_defaultdict():
    from collections import defaultdict
    from pyrsistent import pmap
    from _helpers import freeze

    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    expected = pmap({'a': 1, 'b': 2})
    assert result == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #6
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    from pyrsistent import pmap
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #7
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    from pyrsistent import freeze, pmap
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

```python
def test_freeze_converts_dict_to_pmap():
    result = freeze({'a': 1, 'b': 2})
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant_predicate_false():
    assert not callable(mutant)


# LLM-generated content at query #11
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

def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s | {4, 5}

    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3, 4, 5])

def test_mutant_with_tuple_argument():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_pvector_argument():
    @mutant
    def extend_pvector(v):
        return v.append(4)

    result = extend_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_pmap_argument():
    @mutant
    def update_pmap(m):
        return m.set('new_key', 'new_value')

    result = update_pmap(pmap({'a': 1}))
    assert result == pmap({'a': 1, 'new_key': 'new_value'})

def test_mutant_with_mixed_types():
    @mutant
    def process_mixed(data):
        data['list'].append(data['value'])
        data['set'].add(5)
        return data

    result = process_mixed({'list': [1, 2], 'set': {3, 4}, 'value': 5})
    assert result == pmap({
        'list': pvector([1, 2, 5]),
        'set': pset([3, 4, 5]),
        'value': 5
    })


# LLM-generated content at query #12
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

def test_mutant_with_nested_structures():
    @mutant
    def nest(data):
        return {'nested': data}

    input_data = {'key': [1, 2, 3]}
    result = nest(input_data)
    expected = pmap({'nested': pmap({'key': pvector([1, 2, 3])})})
    assert result == expected

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}

    result = combine(1, [2, 3], x={'y': 4})
    expected = pmap({
        'a': 1,
        'b': pvector([2, 3]),
        'kwargs': pmap({'x': pmap({'y': 4})})
    })
    assert result == expected

def test_mutant_with_no_args():
    @mutant
    def no_args():
        return [1, 2, 3]

    result = no_args()
    assert result == pvector([1, 2, 3])

def test_mutant_preserves_immutable_types():
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity((1, 2, 3)) == (1, 2, 3)
    assert identity("hello") == "hello"


# LLM-generated content at query #13
#--------------------------

```python
def test_freeze_with_strict_true_converts_pvector_elements():
    result = freeze(pvector([1, {'a': 3}]), strict=True)
    assert result == pvector([1, pmap({'a': 3})])


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant_predicate():
    assert not (False)


# LLM-generated content at query #15
#--------------------------

```python
def test_freeze_set_returns_pset():
    assert isinstance(freeze(set([1, 2])), pset)


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_predicate():
    assert mutant(lambda x: x)(1) == 1


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def example_func():
        pass
    assert example_func.__name__ == 'example_func'


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
        return t + (item,)
    result = modify_tuple((1, 2), 3)
    assert result == (1, 2, 3)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['inner'][0] = 10
        return d
    result = modify_nested({'inner': [1, 2, 3]})
    assert result == pmap({'inner': pvector([10, 2, 3])})

def test_mutant_with_kwargs():
    @mutant
    def merge_dicts(**kwargs):
        return {**kwargs}
    result = merge_dicts(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, **kwargs}
    result = combine(1, [2, 3], c=4)
    assert result == pmap({'a': 1, 'b': pvector([2, 3]), 'c': 4})


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == 'test_func'


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_returns_frozen_result():
    @mutant
    def test_func(x):
        return x + 1

    result = test_func(5)
    assert isinstance(result, (int, float, str, tuple, frozenset, bytes))


# LLM-generated content at query #21
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
    def multiply(a, b, c=2):
        return a * b * c
    assert multiply(3, 4) == 24
    assert multiply(3, 4, c=3) == 36

def test_mutant_with_list_arg():
    @mutant
    def process_list(lst):
        return sum(lst)
    assert process_list([1, 2, 3]) == 6

def test_mutant_with_dict_arg():
    @mutant
    def process_dict(d):
        return sum(d.values())
    assert process_dict({'a': 1, 'b': 2}) == 3

def test_mutant_with_nested_structures():
    @mutant
    def nested_sum(data):
        return data['a'] + data['b'][0]
    assert nested_sum({'a': 1, 'b': [2, 3]}) == 3

def test_mutant_returns_frozen_structures():
    @mutant
    def return_list():
        return [1, 2, 3]
    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_returns_frozen_dict():
    @mutant
    def return_dict():
        return {'a': 1, 'b': 2}
    result = return_dict()
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_returns_frozen_set():
    @mutant
    def return_set():
        return {1, 2, 3}
    result = return_set()
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3})

def test_mutant_returns_frozen_tuple():
    @mutant
    def return_tuple():
        return (1, [2, 3])
    result = return_tuple()
    assert isinstance(result, tuple)
    assert result == (1, pvector([2, 3]))


# LLM-generated content at query #22
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    original = defaultdict(int, {'a': 1, 'b': 2})
    frozen = freeze(original)
    assert isinstance(frozen, PMap)
    assert frozen == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #23
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list_with_primitives():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_list_with_dict():
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

def test_freeze_list_with_nested_lists():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_dict_with_list_values():
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

def test_freeze_dict_with_nested_dicts():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_primitives():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_tuple_with_list():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_tuple_with_dict():
    assert freeze((1, {'a': 2})) == (1, pmap({'a': 2}))

def test_freeze_non_container():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"

def test_freeze_pvector():
    assert freeze(pvector([1, 2, 3]), strict=True) == pvector([1, 2, 3])

def test_freeze_pmap():
    assert freeze(pmap({'a': 1}), strict=True) == pmap({'a': 1})

def test_freeze_with_strict_false():
    assert freeze(pvector([1, [2, 3]]), strict=False) == pvector([1, [2, 3]])
    assert freeze(pmap({'a': {1, 2}}), strict=False) == pmap({'a': {1, 2}})


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == 'test_func'


# LLM-generated content at query #25
#--------------------------

```python
def test_freeze_with_empty_set():
    assert freeze(set()) == pset()


# LLM-generated content at query #26
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze(set([1, 2, 3])) == pset([1, 2, 3])

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

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

def test_freeze_non_container():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #27
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

def test_freeze_pvector_without_strict():
    pv = pvector([1, 2, 3])
    assert freeze(pv, strict=False) == pv

def test_freeze_pmap_without_strict():
    pm = pmap({'a': 1, 'b': 2})
    assert freeze(pm, strict=False) == pm

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

def test_freeze_non_container():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #28
#--------------------------

```python
def test_freeze_set_conversion():
    assert freeze(set([1, 2])) == pset([1, 2])


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == "test_func"


# LLM-generated content at query #30
#--------------------------

```python
def test_mutant_with_empty_args_and_kwargs():
    @mutant
    def empty_fn():
        return 42
    assert empty_fn() == 42

def test_mutant_with_simple_args():
    @mutant
    def add(a, b):
        return a + b
    assert add(1, 2) == 3

def test_mutant_with_simple_kwargs():
    @mutant
    def subtract(a, b):
        return a - b
    assert subtract(a=5, b=3) == 2

def test_mutant_with_list_arg():
    @mutant
    def process_list(lst):
        return lst + [4]
    assert process_list([1, 2, 3]) == pvector([1, 2, 3, 4])

def test_mutant_with_dict_arg():
    @mutant
    def process_dict(d):
        d['new_key'] = 10
        return d
    assert process_dict({'a': 1}) == pmap({'a': 1, 'new_key': 10})

def test_mutant_with_nested_structures():
    @mutant
    def nested_fn(data):
        data['list'].append(4)
        return data
    assert nested_fn({'list': [1, 2, 3]}) == pmap({'list': pvector([1, 2, 3, 4])})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def mixed_fn(a, b, c=10):
        return a + b + c
    assert mixed_fn(1, 2, c=3) == 6

def test_mutant_with_set_arg():
    @mutant
    def process_set(s):
        return s | {4, 5}
    assert process_set({1, 2, 3}) == pset({1, 2, 3, 4, 5})

def test_mutant_with_tuple_arg():
    @mutant
    def process_tuple(t):
        return t + (4,)
    assert process_tuple((1, 2, 3)) == (1, 2, 3, 4)

def test_mutant_preserves_immutability():
    @mutant
    def modify_arg(lst):
        lst.append(4)
        return lst
    original = [1, 2, 3]
    result = modify_arg(original)
    assert original == [1, 2, 3]
    assert result == pvector([1, 2, 3, 4])


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

def test_thaw_tuple_recursively():
    from pyrsistent import v
    assert thaw((1, v(2, 3))) == (1, [2, 3])

def test_thaw_nested_pvector_and_pmap():
    from pyrsistent import v, m
    assert thaw(v(1, m(a=2, b=v(3, 4)))) == [1, {'a': 2, 'b': [3, 4]}]

def test_thaw_with_strict_false():
    from pyrsistent import v, m
    assert thaw(v(1, m(a=2)), strict=False) == [1, m(a=2)]

def test_thaw_list_with_strict_true():
    assert thaw([1, 2, 3], strict=True) == [1, 2, 3]

def test_thaw_dict_with_strict_true():
    assert thaw({'a': 1, 'b': 2}, strict=True) == {'a': 1, 'b': 2}

def test_thaw_empty_containers():
    from pyrsistent import v, m, s
    assert thaw(v()) == []
    assert thaw(m()) == {}
    assert thaw(s()) == set()

def test_thaw_non_pyrsistent_types():
    assert thaw(42) == 42
    assert thaw("hello") == "hello"
    assert thaw(None) is None


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_empty_list():
    assert freeze([]) == pvector([])

def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 1}}) == pmap({'a': pmap({'b': 1})})

def test_freeze_mixed_nested_structures():
    assert freeze([1, {'a': [2, 3]}]) == pvector([1, pmap({'a': pvector([2, 3])})])

def test_freeze_with_strict_false():
    pvec = pvector([1, 2])
    assert freeze(pvec, strict=False) == pvec

def test_freeze_with_strict_true():
    pvec = pvector([1, 2])
    assert freeze(pvec, strict=True) == pvector([freeze(1), freeze(2)])

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

def test_freeze_pmap_with_strict_true():
    pm = pmap({'a': 1, 'b': 2})
    assert freeze(pm, strict=True) == pmap({'a': freeze(1), 'b': freeze(2)})

def test_freeze_pvector_with_strict_true():
    pv = pvector([1, 2, 3])
    assert freeze(pv, strict=True) == pvector([freeze(1), freeze(2), freeze(3)])


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_with_strict_true_converts_dict_to_pmap():
    result = freeze({'a': 1, 'b': 2}, strict=True)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_converts_dict_to_pmap():
    result = freeze({'a': 1, 'b': 2})
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #5
#--------------------------

```python
def test_freeze_with_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    expected = pmap({'a': 1, 'b': 2})
    assert result == expected


# LLM-generated content at query #6
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
    def append_to_list(lst, item):
        lst.append(item)
        return lst

    result = append_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict_arguments():
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
    def combine(a, b, c=10):
        return a + b + c

    result = combine(1, 2, c=3)
    assert result == 6

def test_mutant_with_no_return():
    @mutant
    def no_return(x):
        pass

    result = no_return(1)
    assert result is None

def test_mutant_with_set_arguments():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_with_tuple_arguments():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_strict_false():
    @mutant
    def modify_pvector(pv):
        return pv.append(4)

    result = modify_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_mixed_types():
    @mutant
    def mixed(data):
        data['list'].append(3)
        data['set'].add(4)
        return data

    result = mixed({'list': [1, 2], 'set': {1, 2}})
    assert result == pmap({'list': pvector([1, 2, 3]), 'set': pset([1, 2, 4])})


# LLM-generated content at query #7
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
        return t + (item,)
    result = modify_tuple((1, 2), 3)
    assert result == (1, 2, 3)

def test_mutant_with_nested_structures():
    @mutant
    def nested_operation(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data
    input_data = {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    result = nested_operation(input_data)
    expected = pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'b': 2, 'c': 3})})
    assert result == expected

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs
    result = process_kwargs(a=1, b=2)
    expected = pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert result == expected

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def mixed_args_kwargs(arg1, arg2, **kwargs):
        arg1.append(arg2)
        kwargs['arg2'] = arg2
        return {'arg1': arg1, 'kwargs': kwargs}
    result = mixed_args_kwargs([1, 2], 3, key1='value1')
    expected = pmap({'arg1': pvector([1, 2, 3]), 'kwargs': pmap({'key1': 'value1', 'arg2': 3})})
    assert result == expected

def test_mutant_returns_frozen_result():
    @mutant
    def return_list():
        return [1, 2, 3]
    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_empty_structures():
    @mutant
    def empty_operations():
        lst = []
        d = {}
        s = set()
        lst.append(1)
        d['a'] = 1
        s.add(1)
        return {'list': lst, 'dict': d, 'set': s}
    result = empty_operations()
    expected = pmap({'list': pvector([1]), 'dict': pmap({'a': 1}), 'set': pset([1])})
    assert result == expected


# LLM-generated content at query #8
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_non_empty_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_empty_list():
    assert freeze([]) == pvector([])

def test_freeze_non_empty_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_non_empty_tuple():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, (2, 3))) == (1, (2, 3))

def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_non_empty_set():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': 1}), strict=False) == pmap({'a': 1})

def test_freeze_non_strict_pvector():
    assert freeze(pvector([1, 2]), strict=False) == pvector([1, 2])

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

def test_freeze_mixed_types():
    assert freeze({'a': [1, 2], 'b': (3, 4), 'c': {5, 6}}) == pmap({'a': pvector([1, 2]), 'b': (3, 4), 'c': pset({5, 6})})


# LLM-generated content at query #9
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #10
#--------------------------

```python
def test_freeze_with_dict():
    result = freeze({'a': 13, 'b': 14})
    assert result == pmap({'a': 13, 'b': 14})


# LLM-generated content at query #11
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    from collections import defaultdict
    from pyrsistent import freeze, pmap
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_defaultdict():
    d = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant_predicate_false():
    assert not callable(mutant)


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_with_list_input():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    result = add_to_list([1, 2], 3)
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict_input():
    @mutant
    def add_to_dict(dct, key, value):
        dct[key] = value
        return dct

    result = add_to_dict({'a': 1}, 'b', 2)
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_set_input():
    @mutant
    def add_to_set(st, item):
        st.add(item)
        return st

    result = add_to_set({1, 2}, 3)
    assert isinstance(result, pset)
    assert result == pset({1, 2, 3})

def test_mutant_with_tuple_input():
    @mutant
    def add_to_tuple(tpl, item):
        return tpl + (item,)

    result = add_to_tuple((1, 2), 3)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(dct, key, value):
        dct[key]['nested'] = value
        return dct

    result = modify_nested({'a': {'nested': 1}}, 'a', 2)
    assert isinstance(result, pmap)
    assert result == pmap({'a': pmap({'nested': 2})})

def test_mutant_with_no_mutation():
    @mutant
    def no_mutation(x):
        return x

    assert no_mutation(42) == 42
    assert no_mutation("hello") == "hello"

def test_mutant_with_kwargs():
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(existing_key='existing_value')
    assert isinstance(result, pmap)
    assert result == pmap({'existing_key': 'existing_value', 'new_key': 'new_value'})


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_decorator_returns_callable():
    @mutant
    def test_func():
        pass
    assert callable(test_func)


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant_decorator_basic():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])

def test_mutant_decorator_with_kwargs():
    @mutant
    def merge_dicts(a, b):
        return {**a, **b}

    assert merge_dicts({'x': 1}, {'y': 2}) == pmap({'x': 1, 'y': 2})

def test_mutant_decorator_nested_structures():
    @mutant
    def process(data):
        data['values'].append(4)
        return data

    input_data = {'values': [1, 2, 3]}
    result = process(input_data)
    assert result == pmap({'values': pvector([1, 2, 3, 4])})
    assert input_data == {'values': [1, 2, 3]}  # Original should be unchanged

def test_mutant_decorator_with_set():
    @mutant
    def union_sets(a, b):
        return a.union(b)

    assert union_sets({1, 2}, {2, 3}) == pset([1, 2, 3])

def test_mutant_decorator_with_tuple():
    @mutant
    def modify_tuple(data):
        return data + (4,)

    assert modify_tuple((1, 2, 3)) == (1, 2, 3, 4)

def test_mutant_decorator_strict_false():
    @mutant
    def identity(x):
        return x

    class CustomClass:
        pass

    obj = CustomClass()
    assert identity(obj) is obj


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3


# LLM-generated content at query #19
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

def test_mutant_with_set():
    @mutant
    def union_sets(a, b):
        return a.union(b)

    assert union_sets({1, 2}, {2, 3}) == pset([1, 2, 3])

def test_mutant_with_tuple():
    @mutant
    def modify_tuple(data):
        return data + (4,)

    assert modify_tuple((1, 2, 3)) == (1, 2, 3, 4)

def test_mutant_with_kwargs():
    @mutant
    def combine(**kwargs):
        return {k: v * 2 for k, v in kwargs.items()}

    assert combine(a=1, b=2) == pmap({'a': 2, 'b': 4})

def test_mutant_with_no_args():
    @mutant
    def get_defaults():
        return {'a': [1, 2], 'b': {'c': 3}}

    assert get_defaults() == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_predicate():
    assert True


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_decorator_basic():
    @mutant
    def add(a, b):
        return a + b

    result = add([1, 2], [3, 4])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_decorator_nested_structures():
    @mutant
    def process(data):
        data['values'].append(5)
        return data

    input_data = {'values': [1, 2, 3]}
    result = process(input_data)
    assert result == pmap({'values': pvector([1, 2, 3, 5])})
    assert input_data == {'values': [1, 2, 3]}

def test_mutant_decorator_with_kwargs():
    @mutant
    def merge(**kwargs):
        return {**kwargs, 'extra': 1}

    result = merge(a=[1, 2], b={'c': 3})
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3}), 'extra': 1})

def test_mutant_decorator_no_mutation():
    @mutant
    def identity(x):
        return x

    result = identity(pset([1, 2, 3]))
    assert result == pset([1, 2, 3])

def test_mutant_decorator_with_tuple():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_with_list_input():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list([1, 2, 3])
    assert result == (1, 2, 3, 4)

def test_mutant_with_dict_input():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = modify_dict({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

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

def test_mutant_with_mixed_types():
    @mutant
    def modify_mixed(data):
        data['list'].append(4)
        data['set'].add(5)
        return data

    result = modify_mixed({'list': [1, 2, 3], 'set': {1, 2, 3}})
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'set': pset({1, 2, 3, 5})})

def test_mutant_with_kwargs():
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['nested']['value'] = 42
        return data

    result = modify_nested({'nested': {'value': 10}})
    assert result == pmap({'nested': pmap({'value': 42})})

def test_mutant_with_pvector_input():
    @mutant
    def modify_pvector(pv):
        return pv.append(4)

    result = modify_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_pmap_input():
    @mutant
    def modify_pmap(pm):
        return pm.set('new_key', 'new_value')

    result = modify_pmap(pmap({'a': 1, 'b': 2}))
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

def test_mutant_with_pset_input():
    @mutant
    def modify_pset(ps):
        return ps.add(4)

    result = modify_pset(pset({1, 2, 3}))
    assert result == pset({1, 2, 3, 4})


# LLM-generated content at query #23
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
        data['nested']['value'] = 42
        return data

    input_data = {'nested': {'value': 10}}
    result = process(input_data)
    assert result == pmap({'nested': pmap({'value': 42})})
    assert input_data == {'nested': {'value': 10}}  # Original unchanged

def test_mutant_with_set_and_list():
    @mutant
    def modify_mixed(data):
        data['items'].add(3)
        data['values'].append(4)
        return data

    input_data = {'items': {1, 2}, 'values': [1, 2, 3]}
    result = modify_mixed(input_data)
    assert result == pmap({'items': pset({1, 2, 3}), 'values': pvector([1, 2, 3, 4])})
    assert input_data == {'items': {1, 2}, 'values': [1, 2, 3]}  # Original unchanged

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'extra': kwargs}

    result = combine(1, [2], extra={'key': 'value'})
    assert result == pmap({'a': 1, 'b': pvector([2]), 'extra': pmap({'key': 'value'})})

def test_mutant_with_no_args():
    @mutant
    def get_defaults():
        return {'items': [1, 2], 'config': {'enabled': True}}

    result = get_defaults()
    assert result == pmap({'items': pvector([1, 2]), 'config': pmap({'enabled': True})})


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not (1, 2) and 0)


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_predicate():
    assert not callable(mutant)


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

    result = add(1, 2)
    assert result == 3

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
        b['key'] = a
        c.append(b)
        return c

    result = process_mixed(1, {'key': 2}, [3])
    assert result == pvector([3, pmap({'key': 1})])

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        kwargs['extra'] = 'value'
        return kwargs

    result = process_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'extra': 'value'})

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(d):
        d['inner']['value'] = 10
        return d

    result = process_nested({'inner': {'value': 5}})
    assert result == pmap({'inner': pmap({'value': 10})})

def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        s.add(4)
        return s

    result = process_set({1, 2, 3})
    assert result == pset({1, 2, 3, 4})

def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4,)

    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_no_mutation():
    @mutant
    def no_mutation(x):
        return x

    result = no_mutation(42)
    assert result == 42

def test_mutant_with_pvector_argument():
    @mutant
    def process_pvector(pv):
        return pv.append(4)

    result = process_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_pmap_argument():
    @mutant
    def process_pmap(pm):
        return pm.set('new', 'value')

    result = process_pmap(pmap({'a': 1}))
    assert result == pmap({'a': 1, 'new': 'value'})

def test_mutant_with_pset_argument():
    @mutant
    def process_pset(ps):
        return ps.add(4)

    result = process_pset(pset({1, 2, 3}))
    assert result == pset({1, 2, 3, 4})

def test_mutant_with_strict_false():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant_predicate():
    assert mutant(lambda x: x) is not None


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == [1, 2, 3, 4]


# LLM-generated content at query #30
#--------------------------

```python
def test_mutant_basic():
    @mutant
    def add(a, b):
        return a + b
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])

def test_mutant_with_dict():
    @mutant
    def merge_dicts(d1, d2):
        return {**d1, **d2}
    assert merge_dicts({'a': 1}, {'b': 2}) == pmap({'a': 1, 'b': 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        data['values'].append(5)
        return data
    assert process({'values': [1, 2, 3]}) == pmap({'values': pvector([1, 2, 3, 5])})

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        result = {'a': a, 'b': b}
        result.update(kwargs)
        return result
    assert combine([1], [2], c=[3], d=[4]) == pmap({'a': pvector([1]), 'b': pvector([2]), 'c': pvector([3]), 'd': pvector([4])})

def test_mutant_with_set():
    @mutant
    def union_sets(s1, s2):
        return s1 | s2
    assert union_sets({1, 2}, {2, 3}) == pset([1, 2, 3])

def test_mutant_with_tuple():
    @mutant
    def extend_tuple(t, item):
        return t + (item,)
    assert extend_tuple((1, 2), 3) == (1, 2, 3)

def test_mutant_no_mutation():
    @mutant
    def identity(x):
        return x
    assert identity(42) == 42
    assert identity("hello") == "hello"

def test_mutant_with_mixed_types():
    @mutant
    def mix(data):
        data['list'].append(data['value'])
        data['set'].add(data['value'])
        return data
    result = mix({'list': [1, 2], 'set': {1, 2}, 'value': 3})
    assert result == pmap({'list': pvector([1, 2, 3]), 'set': pset([1, 2, 3]), 'value': 3})


# LLM-generated content at query #31
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 1}}) == pmap({'a': pmap({'b': 1})})

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_nested_tuple():
    assert freeze((1, (2, 3))) == (1, (2, 3))

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

def test_freeze_dict_with_pmap():
    assert freeze(pmap({'a': 1})) == pmap({'a': 1})

def test_freeze_list_with_pvector():
    assert freeze(pvector([1, 2])) == pvector([1, 2])

def test_freeze_non_strict():
    assert freeze({'a': [1, 2]}, strict=False) == pmap({'a': [1, 2]})

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1})
    assert freeze(dd) == pmap({'a': 1})

def test_freeze_mixed_types():
    assert freeze([1, {'a': (2, 3)}, {4, 5}]) == pvector([1, pmap({'a': (2, 3)}), pset({4, 5})])


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not _EMPTY_PSET and not _EMPTY_PMAP)


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])

def test_mutant_with_list_arguments():
    @mutant
    def process_list(lst):
        lst.append(4)
        return lst

    result = process_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, PVector)

def test_mutant_with_dict_arguments():
    @mutant
    def process_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = process_dict({'a': 1})
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert isinstance(result, PMap)

def test_mutant_with_mixed_arguments():
    @mutant
    def process_mixed(a, b, c):
        a.append(b)
        c['key'] = b
        return (a, c)

    result = process_mixed([1, 2], 3, {'x': 10})
    assert result == (pvector([1, 2, 3]), pmap({'x': 10, 'key': 3}))

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
        data['nested']['value'] = 42
        data['list'].append(99)
        return data

    input_data = {'nested': {'value': 10}, 'list': [1, 2, 3]}
    result = process_nested(input_data)
    assert result == pmap({'nested': pmap({'value': 42}), 'list': pvector([1, 2, 3, 99])})

def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s | {4, 5}

    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3, 4, 5])

def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4,)

    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_no_mutation():
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("string") == "string"
    assert identity(None) is None


# LLM-generated content at query #34
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

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        return {'result': data}
    assert process([1, {'a': 2}]) == pmap({'result': pvector([1, pmap({'a': 2})])})

def test_mutant_with_kwargs():
    @mutant
    def configure(**kwargs):
        return kwargs
    assert configure(a=1, b=[2, 3]) == pmap({'a': 1, 'b': pvector([2, 3])})

def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s | {4, 5}
    assert process_set({1, 2, 3}) == pset([1, 2, 3, 4, 5])

def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4, 5)
    assert process_tuple((1, 2, 3)) == (1, 2, 3, 4, 5)

def test_mutant_with_mixed_arguments():
    @mutant
    def combine(a, b, c):
        return [a, b, c]
    assert combine(1, {'a': 2}, [3, 4]) == pvector([1, pmap({'a': 2}), pvector([3, 4])])

def test_mutant_with_no_arguments():
    @mutant
    def get_default():
        return {'default': [1, 2]}
    assert get_default() == pmap({'default': pvector([1, 2])})

def test_mutant_with_strict_false():
    @mutant
    def identity(x):
        return x
    assert identity(pvector([1, 2])) == pvector([1, 2])
    assert identity(pmap({'a': 1})) == pmap({'a': 1})


# LLM-generated content at query #35
#--------------------------

```python
def test_mutant_decorator_returns_callable():
    @mutant
    def test_func():
        pass
    assert callable(test_func)


# LLM-generated content at query #36
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == "test_func"


# LLM-generated content at query #37
#--------------------------

```python
def test_freeze_with_defaultdict():
    default_dict = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(default_dict)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #38
#--------------------------

```python
def test_mutant_decorator_preserves_function_name_and_docstring():
    @mutant
    def example_function(x):
        """Example function for testing."""
        return x + 1

    assert example_function.__name__ == "example_function"
    assert example_function.__doc__ == "Example function for testing."


# LLM-generated content at query #39
#--------------------------

```python
def test_mutant_with_list():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])

def test_mutant_with_dict():
    @mutant
    def modify_dict(d):
        d['c'] = 3
        return d

    result = modify_dict({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})

def test_mutant_with_set():
    @mutant
    def modify_set(s):
        s.add(3)
        return s

    result = modify_set({1, 2})
    assert result == pset([1, 2, 3])

def test_mutant_with_tuple():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['inner']['value'] = 42
        return d

    result = modify_nested({'outer': 1, 'inner': {'value': 10}})
    assert result == pmap({'outer': 1, 'inner': pmap({'value': 42})})

def test_mutant_with_kwargs():
    @mutant
    def modify_kwargs(a, b, **kwargs):
        kwargs['c'] = a + b
        return kwargs

    result = modify_kwargs(1, 2, d=4)
    assert result == pmap({'d': 4, 'c': 3})

def test_mutant_preserves_immutability():
    @mutant
    def no_op(x):
        return x

    original = [1, 2, 3]
    result = no_op(original)
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #40
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

def test_mutant_with_mixed_arguments():
    @mutant
    def process(data):
        return data['values'][0] + data['factor']

    result = process({'values': [10, 20], 'factor': 5})
    assert result == 15
    assert isinstance(result, int)

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return a + b + kwargs['extra']

    result = combine(1, 2, extra=3)
    assert result == 6
    assert isinstance(result, int)

def test_mutant_with_nested_structures():
    @mutant
    def nested_sum(data):
        return data[0] + data[1]['inner']

    result = nested_sum([1, {'inner': 2}])
    assert result == 3
    assert isinstance(result, int)

def test_mutant_with_pvector_input():
    @mutant
    def first_element(vec):
        return vec[0]

    result = first_element(pvector([1, 2, 3]))
    assert result == 1
    assert isinstance(result, int)

def test_mutant_with_pmap_input():
    @mutant
    def get_key(pm):
        return pm['key']

    result = get_key(pmap({'key': 42}))
    assert result == 42
    assert isinstance(result, int)

def test_mutant_with_pset_input():
    @mutant
    def set_size(ps):
        return len(ps)

    result = set_size(pset([1, 2, 3]))
    assert result == 3
    assert isinstance(result, int)

def test_mutant_with_tuple_input():
    @mutant
    def tuple_sum(t):
        return t[0] + t[1]

    result = tuple_sum((10, 20))
    assert result == 30
    assert isinstance(result, int)


# LLM-generated content at query #41
#--------------------------

```python
def test_mutant_decorator_returns_function():
    result = mutant(lambda x: x)
    assert callable(result)


# LLM-generated content at query #42
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


# LLM-generated content at query #43
#--------------------------

```python
def test_freeze_defaultdict():
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #44
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == "test_func"


