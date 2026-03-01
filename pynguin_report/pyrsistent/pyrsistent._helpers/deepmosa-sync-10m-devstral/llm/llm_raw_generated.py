####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set():
    assert freeze(set([1, 2])) == pset([1, 2])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list():
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple():
    assert freeze((1, [])) == (1, pvector([]))

def test_freeze_nested_tuple():
    assert freeze((1, (2, []))) == (1, (2, pvector([])))

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict():
    assert freeze({'a': 3}) == pmap({'a': 3})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 3}}) == pmap({'a': pmap({'b': 3})})

def test_freeze_defaultdict():
    defaultdict = collections.defaultdict(int, {'a': 3})
    assert freeze(defaultdict) == pmap({'a': 3})

def test_freeze_pvector():
    pv = pvector([1, 2, 3])
    assert freeze(pv) == pvector([1, 2, 3])

def test_freeze_pmap():
    pm = pmap({'a': 3})
    assert freeze(pm) == pmap({'a': 3})

def test_freeze_pset():
    ps = pset([1, 2, 3])
    assert freeze(ps) == pset([1, 2, 3])

def test_freeze_non_strict():
    assert freeze([1, {'a': 3}], strict=False) == pvector([1, {'a': 3}])

def test_freeze_immutable():
    assert freeze(1) == 1
    assert freeze("string") == "string"


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    from pyrsistent import pmap
    from _helpers import freeze

    original = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(original)
    expected = pmap({'a': 1, 'b': 2})

    assert result == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_mutant_with_simple_types():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add("a", "b") == "ab"

def test_mutant_with_list_and_dict():
    @mutant
    def modify(data):
        data["list"].append(4)
        return data

    input_data = {"list": [1, 2, 3]}
    result = modify(input_data)
    assert result == {"list": pvector([1, 2, 3, 4])}
    assert input_data == {"list": [1, 2, 3]}

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        data["nested"]["value"] += 1
        return data

    input_data = {"nested": {"value": 5}}
    result = process(input_data)
    assert result == {"nested": pmap({"value": 6})}
    assert input_data == {"nested": {"value": 5}}

def test_mutant_with_set():
    @mutant
    def handle_set(data):
        data["set"].add(4)
        return data

    input_data = {"set": {1, 2, 3}}
    result = handle_set(input_data)
    assert result == {"set": pset([1, 2, 3, 4])}
    assert input_data == {"set": {1, 2, 3}}

def test_mutant_with_tuple():
    @mutant
    def handle_tuple(data):
        return data + (4,)

    input_data = (1, 2, 3)
    result = handle_tuple(input_data)
    assert result == (1, 2, 3, 4)
    assert input_data == (1, 2, 3)

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {"a": a, "b": b, "kwargs": kwargs}

    result = combine(1, 2, x=3, y=4)
    assert result == pmap({"a": 1, "b": 2, "kwargs": pmap({"x": 3, "y": 4})})

def test_mutant_with_no_args():
    @mutant
    def no_args():
        return [1, 2, 3]

    result = no_args()
    assert result == pvector([1, 2, 3])

def test_mutant_with_empty_structures():
    @mutant
    def empty():
        return {"list": [], "dict": {}, "set": set()}

    result = empty()
    assert result == pmap({"list": pvector([]), "dict": pmap({}), "set": pset([])})


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1})
    assert isinstance(freeze(d), PMap)


# LLM-generated content at query #5
#--------------------------

```python
def test_freeze_predicate_false():
    assert not (type(None) is dict or (True and isinstance(None, PMap)))


# LLM-generated content at query #6
#--------------------------

```python
def test_freeze_predicate_false():
    assert not (type(None) is dict or (True and isinstance(None, PMap)))


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
    def modify_tuple(t, index, value):
        lst = list(t)
        lst[index] = value
        return tuple(lst)
    result = modify_tuple((1, 2, 3), 1, 4)
    assert result == (1, 4, 3)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['lst'].append(4)
        return d
    result = modify_nested({'lst': [1, 2, 3]})
    assert result == pmap({'lst': pvector([1, 2, 3, 4])})

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}
    result = combine(1, 2, c=3, d=4)
    expected = pmap({'a': 1, 'b': 2, 'kwargs': pmap({'c': 3, 'd': 4})})
    assert result == expected

def test_mutant_preserves_immutability():
    @mutant
    def modify_list(lst):
        lst.append(1)
        return lst
    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]
    assert result == pvector([1, 2, 3, 1])


# LLM-generated content at query #8
#--------------------------

```python
def test_freeze_with_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #9
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    assert (type(collections.defaultdict) is collections.defaultdict) or (True and isinstance(pmap(), PMap))


# LLM-generated content at query #10
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #11
#--------------------------

```python
def test_mutant_with_list_argument():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, pvector)

def test_mutant_with_dict_argument():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = modify_dict({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert isinstance(result, pmap)

def test_mutant_with_set_argument():
    @mutant
    def modify_set(s):
        s.add(4)
        return s

    result = modify_set({1, 2, 3})
    assert result == pset({1, 2, 3, 4})
    assert isinstance(result, pset)

def test_mutant_with_tuple_argument():
    @mutant
    def modify_tuple(t):
        return t + (4,)

    result = modify_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)
    assert isinstance(result, tuple)

def test_mutant_with_mixed_arguments():
    @mutant
    def modify_mixed(lst, d, s, t):
        lst.append(4)
        d['new_key'] = 'new_value'
        s.add(5)
        return (lst, d, s, t)

    result = modify_mixed([1, 2, 3], {'a': 1}, {1, 2, 3}, (1, 2, 3))
    assert result == (pvector([1, 2, 3, 4]), pmap({'a': 1, 'new_key': 'new_value'}), pset({1, 2, 3, 5}), (1, 2, 3))
    assert isinstance(result[0], pvector)
    assert isinstance(result[1], pmap)
    assert isinstance(result[2], pset)
    assert isinstance(result[3], tuple)

def test_mutant_with_kwargs():
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert isinstance(result, pmap)

def test_mutant_with_no_mutation():
    @mutant
    def no_mutation(lst):
        return lst

    result = no_mutation([1, 2, 3])
    assert result == pvector([1, 2, 3])
    assert isinstance(result, pvector)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['nested_list'].append(4)
        return d

    result = modify_nested({'nested_list': [1, 2, 3]})
    assert result == pmap({'nested_list': pvector([1, 2, 3, 4])})
    assert isinstance(result['nested_list'], pvector)


# LLM-generated content at query #12
#--------------------------

```python
def test_mutant_with_empty_function():
    @mutant
    def empty_fn():
        pass
    assert empty_fn() is None

def test_mutant_with_list_argument():
    @mutant
    def process_list(lst):
        lst.append(4)
        return lst
    result = process_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, PVector)

def test_mutant_with_dict_argument():
    @mutant
    def process_dict(d):
        d['new_key'] = 'new_value'
        return d
    result = process_dict({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert isinstance(result, PMap)

def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        s.add(4)
        return s
    result = process_set({1, 2, 3})
    assert result == pset({1, 2, 3, 4})
    assert isinstance(result, PSet)

def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4,)
    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)
    assert isinstance(result, tuple)

def test_mutant_with_mixed_arguments():
    @mutant
    def process_mixed(lst, d, s, t):
        lst.append(4)
        d['new_key'] = 'new_value'
        s.add(4)
        return (lst, d, s, t)
    result = process_mixed([1, 2, 3], {'a': 1}, {1, 2, 3}, (1, 2, 3))
    assert result == (pvector([1, 2, 3, 4]), pmap({'a': 1, 'new_key': 'new_value'}), pset({1, 2, 3, 4}), (1, 2, 3))
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)
    assert isinstance(result[3], tuple)

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs
    result = process_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert isinstance(result, PMap)

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(d):
        d['nested_list'].append(4)
        return d
    result = process_nested({'nested_list': [1, 2, 3]})
    assert result == pmap({'nested_list': pvector([1, 2, 3, 4])})
    assert isinstance(result['nested_list'], PVector)

def test_mutant_with_strict_false():
    @mutant
    def process_list_non_strict(lst):
        lst.append(4)
        return lst
    result = process_list_non_strict(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, PVector)


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant_decorator_basic():
    @mutant
    def add(a, b):
        return a + b

    result = add([1, 2], [3, 4])
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, pvector)

def test_mutant_decorator_with_dict():
    @mutant
    def merge_dicts(d1, d2):
        return {**d1, **d2}

    result = merge_dicts({'a': 1}, {'b': 2})
    assert result == pmap({'a': 1, 'b': 2})
    assert isinstance(result, PMap)

def test_mutant_decorator_with_nested_structures():
    @mutant
    def process(data):
        return {'result': data}

    input_data = [1, {'nested': [2, 3]}]
    result = process(input_data)
    expected = pmap({'result': pvector([1, pmap({'nested': pvector([2, 3])})])})
    assert result == expected
    assert isinstance(result['result'][1]['nested'], PVector)

def test_mutant_decorator_with_kwargs():
    @mutant
    def combine(a, b=10):
        return {'a': a, 'b': b}

    result = combine([1, 2], b=[3, 4])
    expected = pmap({'a': pvector([1, 2]), 'b': pvector([3, 4])})
    assert result == expected
    assert isinstance(result['a'], PVector)

def test_mutant_decorator_returns_non_container():
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("hello") == "hello"

def test_mutant_decorator_with_tuple():
    @mutant
    def wrap_in_tuple(x):
        return (x, x)

    result = wrap_in_tuple([1, 2])
    assert result == (pvector([1, 2]), pvector([1, 2]))
    assert isinstance(result[0], pvector)

def test_mutant_decorator_with_set():
    @mutant
    def set_operation(s):
        return s | {1, 2, 3}

    result = set_operation({4, 5})
    assert result == pset({1, 2, 3, 4, 5})
    assert isinstance(result, PSet)


# LLM-generated content at query #15
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    d = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_with_list_arg():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])
    assert type(result) is PVector

def test_mutant_with_dict_arg():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = modify_dict({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert type(result) is PMap

def test_mutant_with_set_arg():
    @mutant
    def modify_set(s):
        s.add(4)
        return s

    result = modify_set({1, 2, 3})
    assert result == pset({1, 2, 3, 4})
    assert type(result) is PSet

def test_mutant_with_tuple_arg():
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
        data['dict']['new_key'] = 'new_value'
        return data

    input_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = modify_nested(input_data)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'new_key': 'new_value'})})
    assert type(result['list']) is PVector
    assert type(result['dict']) is PMap

def test_mutant_with_kwargs():
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert type(result) is PMap

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def modify_mixed(lst, d, **kwargs):
        lst.append(4)
        d['new_key'] = 'new_value'
        kwargs['kwarg_key'] = 'kwarg_value'
        return lst, d, kwargs

    result = modify_mixed([1, 2, 3], {'a': 1}, kwarg1='value1')
    assert result == (pvector([1, 2, 3, 4]), pmap({'a': 1, 'new_key': 'new_value'}), pmap({'kwarg1': 'value1', 'kwarg_key': 'kwarg_value'}))
    assert type(result[0]) is PVector
    assert type(result[1]) is PMap
    assert type(result[2]) is PMap

def test_mutant_with_no_mutation():
    @mutant
    def no_mutation(data):
        return data

    result = no_mutation([1, 2, 3])
    assert result == pvector([1, 2, 3])
    assert type(result) is PVector

def test_mutant_with_empty_structures():
    @mutant
    def modify_empty(lst, d, s):
        lst.append(1)
        d['key'] = 'value'
        s.add(1)
        return lst, d, s

    result = modify_empty([], {}, set())
    assert result == (pvector([1]), pmap({'key': 'value'}), pset({1}))
    assert type(result[0]) is PVector
    assert type(result[1]) is PMap
    assert type(result[2]) is PSet

def test_mutant_with_strict_false():
    @mutant
    def modify_with_pvector(pv):
        pv = pv.append(4)
        return pv

    result = modify_with_pvector(pvector([1, 2, 3]))
    assert result == pvector([1, 2, 3, 4])
    assert type(result) is PVector


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3]) == pvector([1, 2, 3])
    assert add({"a": 1}, {"b": 2}) == pmap({"a": 1, "b": 2})

def test_mutant_with_nested_structures():
    @mutant
    def process(data):
        data["values"].append(4)
        return data

    input_data = {"values": [1, 2, 3]}
    result = process(input_data)
    assert result == pmap({"values": pvector([1, 2, 3, 4])})
    assert input_data == {"values": [1, 2, 3]}

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    assert merge(a=1, b=[2, 3]) == pmap({"a": 1, "b": pvector([2, 3])})

def test_mutant_with_no_args():
    @mutant
    def get_defaults():
        return {"key": [1, 2]}

    assert get_defaults() == pmap({"key": pvector([1, 2])})

def test_mutant_with_mutable_return():
    @mutant
    def create_list():
        return [1, 2, 3]

    result = create_list()
    assert result == pvector([1, 2, 3])
    assert isinstance(result, PVector)

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s | {4, 5}

    assert process_set({1, 2, 3}) == pset([1, 2, 3, 4, 5])

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t + (4, 5)

    assert process_tuple((1, 2, 3)) == (1, 2, 3, 4, 5)

def test_mutant_with_strict_false():
    @mutant
    def identity(x):
        return x

    assert identity(PVector([1, 2, 3])) == PVector([1, 2, 3])
    assert identity(PMap({"a": 1})) == PMap({"a": 1})


# LLM-generated content at query #18
#--------------------------

```python
def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set():
    assert freeze(set([1, 2])) == pset([1, 2])

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_list():
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple():
    assert freeze((1, [])) == (1, pvector([]))

def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_dict():
    assert freeze({'a': 3}) == pmap({'a': 3})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 4}}) == pmap({'a': pmap({'b': 4})})

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': [1, 2]}), strict=False) == pmap({'a': [1, 2]})

def test_freeze_non_strict_pvector():
    assert freeze(pvector([1, {'a': 3}]), strict=False) == pvector([1, {'a': 3}])

def test_freeze_non_strict_pset():
    assert freeze(pset([1, 2]), strict=False) == pset([1, 2])

def test_freeze_non_strict_tuple():
    assert freeze((1, [2, 3]), strict=False) == (1, [2, 3])

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, a=3)
    assert freeze(dd) == pmap({'a': 3})

def test_freeze_non_container():
    assert freeze(42) == 42

def test_freeze_string():
    assert freeze("hello") == "hello"


# LLM-generated content at query #19
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
    assert freeze(pvector([1, 2, 3]), strict=True) == pvector([1, 2, 3])

def test_freeze_pmap_with_elements():
    assert freeze(pmap({'a': 1, 'b': 2}), strict=True) == pmap({'a': 1, 'b': 2})

def test_freeze_non_container():
    assert freeze(42) == 42

def test_freeze_non_container_with_strict_false():
    assert freeze(42, strict=False) == 42


# LLM-generated content at query #20
#--------------------------

```python
def test_freeze_defaultdict_conversion():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_with_list_arguments():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    result = add_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict_arguments():
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    result = add_to_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})

def test_mutant_with_tuple_arguments():
    @mutant
    def add_to_tuple(tpl, item):
        return tpl + (item,)

    result = add_to_tuple((1, 2), 3)
    assert result == (1, 2, 3)

def test_mutant_with_set_arguments():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset([1, 2, 3])

def test_mutant_with_mixed_arguments():
    @mutant
    def modify_mixed(lst, d, tpl, s, item):
        lst.append(item)
        d['new_key'] = item
        new_tpl = tpl + (item,)
        s.add(item)
        return lst, d, new_tpl, s

    result = modify_mixed([1], {'a': 1}, (1,), {1}, 2)
    assert result == (pvector([1, 2]), pmap({'a': 1, 'new_key': 2}), (1, 2), pset([1, 2]))

def test_mutant_with_kwargs():
    @mutant
    def modify_with_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_with_kwargs(a=1, b=2)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(lst):
        lst[0]['key'] = 'modified'
        return lst

    result = modify_nested([{'key': 'original'}])
    assert result == pvector([pmap({'key': 'modified'})])

def test_mutant_with_no_mutation():
    @mutant
    def no_mutation(x):
        return x

    assert no_mutation(42) == 42
    assert no_mutation("string") == "string"


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_predicate():
    assert not callable(mutant)


# LLM-generated content at query #24
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
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_tuple():
    assert freeze((1, (2, 3))) == (1, (2, 3))

def test_freeze_mixed_containers():
    assert freeze([1, {'a': (2, {3, 4})}]) == pvector([1, pmap({'a': (2, pset({3, 4}))})])

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

def test_freeze_pvector_with_strict_true():
    pv = pvector([1, [2, 3]])
    assert freeze(pv, strict=True) == pvector([1, pvector([2, 3])])

def test_freeze_pvector_with_strict_false():
    pv = pvector([1, [2, 3]])
    assert freeze(pv, strict=False) == pv

def test_freeze_pset():
    ps = pset({1, 2, 3})
    assert freeze(ps) == ps

def test_freeze_non_container():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #25
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

def test_freeze_pvector():
    assert freeze(pvector([1, 2, 3])) == pvector([1, 2, 3])

def test_freeze_pmap():
    assert freeze(pmap({'a': 1})) == pmap({'a': 1})

def test_freeze_non_container():
    assert freeze(42) == 42

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': [1, 2]}), strict=False) == pmap({'a': [1, 2]})

def test_freeze_non_strict_pvector():
    assert freeze(pvector([1, {'a': 2}]), strict=False) == pvector([1, {'a': 2}])


# LLM-generated content at query #26
#--------------------------

```python
def test_freeze_with_dict():
    result = freeze({'a': 1, 'b': [2, 3]})
    expected = pmap({'a': 1, 'b': pvector([2, 3])})
    assert result == expected


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_decorator_preserves_original_function():
    @mutant
    def test_func(x):
        return x + 1

    assert test_func.__name__ == 'test_func'
    assert test_func.__doc__ is None


# LLM-generated content at query #28
#--------------------------

```python
def test_freeze_with_strict_true_and_pmap_input():
    input_pmap = pmap({'a': [1, 2], 'b': {'c': 3}})
    result = freeze(input_pmap, strict=True)
    expected = pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})
    assert result == expected


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant_with_list_input():
    @mutant
    def add_item(lst, item):
        lst.append(item)
        return lst

    result = add_item([1, 2], 3)
    assert result == pvector([1, 2, 3])
    assert isinstance(result, PVector)

def test_mutant_with_dict_input():
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})
    assert isinstance(result, PMap)

def test_mutant_with_set_input():
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    result = add_to_set({1, 2}, 3)
    assert result == pset({1, 2, 3})
    assert isinstance(result, PSet)

def test_mutant_with_tuple_input():
    @mutant
    def modify_tuple(t, item):
        lst = list(t)
        lst.append(item)
        return tuple(lst)

    result = modify_tuple((1, 2), 3)
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)

def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(d):
        d['lst'].append(4)
        return d

    result = modify_nested({'lst': [1, 2, 3]})
    assert result == pmap({'lst': pvector([1, 2, 3, 4])})
    assert isinstance(result['lst'], PVector)

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, **kwargs}

    result = combine(1, 2, c=3, d=[4, 5])
    expected = pmap({'a': 1, 'b': 2, 'c': 3, 'd': pvector([4, 5])})
    assert result == expected
    assert isinstance(result['d'], PVector)

def test_mutant_with_no_mutation():
    @mutant
    def no_op(x):
        return x

    assert no_op(42) == 42
    assert no_op("hello") == "hello"
    assert no_op(None) is None

def test_mutant_with_multiple_args():
    @mutant
    def merge_lists(a, b):
        return a + b

    result = merge_lists([1, 2], [3, 4])
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, PVector)


# LLM-generated content at query #30
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not False and True)


# LLM-generated content at query #31
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add("a", "b") == "ab"


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


# LLM-generated content at query #34
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


# LLM-generated content at query #35
#--------------------------

```python
def test_mutant_with_empty_function():
    @mutant
    def empty_func():
        return None

    assert empty_func() is None

def test_mutant_with_simple_immutable_args():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3

def test_mutant_with_list_arg():
    @mutant
    def process_list(lst):
        return lst + [4]

    result = process_list([1, 2, 3])
    assert result == [1, 2, 3, 4]
    assert isinstance(result, tuple)

def test_mutant_with_dict_arg():
    @mutant
    def process_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = process_dict({'a': 1})
    assert result == {'a': 1, 'new_key': 'new_value'}
    assert isinstance(result, tuple)

def test_mutant_with_set_arg():
    @mutant
    def process_set(s):
        return s | {4}

    result = process_set({1, 2, 3})
    assert result == {1, 2, 3, 4}
    assert isinstance(result, tuple)

def test_mutant_with_tuple_arg():
    @mutant
    def process_tuple(t):
        return t + (4,)

    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)
    assert isinstance(result, tuple)

def test_mutant_with_mixed_args():
    @mutant
    def mixed(a, b, c):
        return (a, b, c)

    result = mixed([1, 2], {'a': 3}, {4, 5})
    assert result == ([1, 2], {'a': 3}, {4, 5})
    assert isinstance(result, tuple)
    assert isinstance(result[0], tuple)
    assert isinstance(result[1], tuple)
    assert isinstance(result[2], tuple)

def test_mutant_with_kwargs():
    @mutant
    def with_kwargs(a, b=2, c=3):
        return a + b + c

    assert with_kwargs(1) == 6
    assert with_kwargs(1, b=3) == 7
    assert with_kwargs(1, b=3, c=4) == 8

def test_mutant_with_nested_structures():
    @mutant
    def nested(data):
        return data

    input_data = {
        'list': [1, 2, {'a': 3}],
        'set': {4, 5},
        'tuple': (6, [7, 8])
    }
    result = nested(input_data)
    assert result['list'] == [1, 2, {'a': 3}]
    assert result['set'] == {4, 5}
    assert result['tuple'] == (6, [7, 8])
    assert isinstance(result, tuple)
    assert isinstance(result['list'], tuple)
    assert isinstance(result['list'][2], tuple)
    assert isinstance(result['set'], tuple)
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['tuple'][1], tuple)

def test_mutant_with_pyrsistent_types():
    @mutant
    def pyrsistent_types(pv, pm, ps):
        return (pv, pm, ps)

    from pyrsistent import pvector, pmap, pset
    result = pyrsistent_types(pvector([1, 2]), pmap({'a': 3}), pset({4, 5}))
    assert result == (pvector([1, 2]), pmap({'a': 3}), pset({4, 5}))
    assert isinstance(result, tuple)


# LLM-generated content at query #36
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

def test_freeze_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(d) == pmap({'a': 1, 'b': 2})

def test_freeze_pvector_with_strict():
    pv = pvector([1, 2, 3])
    assert freeze(pv, strict=True) == pvector([1, 2, 3])

def test_freeze_pmap_with_strict():
    pm = pmap({'a': 1, 'b': 2})
    assert freeze(pm, strict=True) == pmap({'a': 1, 'b': 2})

def test_freeze_pvector_without_strict():
    pv = pvector([1, 2, 3])
    assert freeze(pv, strict=False) == pv

def test_freeze_pmap_without_strict():
    pm = pmap({'a': 1, 'b': 2})
    assert freeze(pm, strict=False) == pm

def test_freeze_mixed_types():
    data = [1, {'a': [2, 3]}, (4, {5, 6})]
    expected = pvector([1, pmap({'a': pvector([2, 3])}), (4, pset([5, 6]))])
    assert freeze(data) == expected


# LLM-generated content at query #37
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == 'test_func'


# LLM-generated content at query #38
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

    input_data = {'values': [1, 2, 3]}
    result = process(input_data)
    assert result == {'values': pvector([1, 2, 3, 4])}
    assert input_data == {'values': [1, 2, 3]}

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return dict(kwargs)

    result = merge(a=[1, 2], b={'x': 3})
    assert result == {'a': pvector([1, 2]), 'b': pmap({'x': 3})}

def test_mutant_with_no_args():
    @mutant
    def get_defaults():
        return {'a': [1], 'b': {'x': 2}}

    result = get_defaults()
    assert result == {'a': pvector([1]), 'b': pmap({'x': 2})}

def test_mutant_with_mutable_args():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s | {4, 5}

    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3, 4, 5])


# LLM-generated content at query #39
#--------------------------

```python
def test_freeze_empty_list():
    assert freeze([]) == pvector([])

def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_empty_set():
    assert freeze(set()) == pset([])

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_list_with_elements():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_dict_with_elements():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_tuple_with_elements():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 1}}) == pmap({'a': pmap({'b': 1})})

def test_freeze_nested_mixed():
    assert freeze([1, {'a': [2, 3]}]) == pvector([1, pmap({'a': pvector([2, 3])})])

def test_freeze_with_pvector():
    assert freeze(pvector([1, 2, 3])) == pvector([1, 2, 3])

def test_freeze_with_pmap():
    assert freeze(pmap({'a': 1})) == pmap({'a': 1})

def test_freeze_with_pset():
    assert freeze(pset([1, 2, 3])) == pset([1, 2, 3])

def test_freeze_non_strict():
    assert freeze([1, {'a': [2, 3]}], strict=False) == pvector([1, {'a': [2, 3]}])

def test_freeze_defaultdict():
    d = collections.defaultdict(int, {'a': 1})
    assert freeze(d) == pmap({'a': 1})

def test_freeze_non_container():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #40
#--------------------------

```python
def test_mutant_predicate():
    assert not callable(mutant)


# LLM-generated content at query #41
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

def test_freeze_non_strict():
    assert freeze(pvector([1, [2, 3]]), strict=False) == pvector([1, [2, 3]])

def test_freeze_non_strict_dict():
    assert freeze(pmap({'a': [1, 2]}), strict=False) == pmap({'a': [1, 2]})


# LLM-generated content at query #42
#--------------------------

```python
def test_freeze_with_empty_dict_and_strict_false():
    assert freeze({}, strict=False) == {}


# LLM-generated content at query #43
#--------------------------

```python
def test_mutant_with_empty_function():
    @mutant
    def empty_fn():
        return None

    assert empty_fn() is None

def test_mutant_with_simple_args():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3

def test_mutant_with_list_arg():
    @mutant
    def process_list(lst):
        return lst + [4]

    assert process_list([1, 2, 3]) == pvector([1, 2, 3, 4])

def test_mutant_with_dict_arg():
    @mutant
    def process_dict(d):
        return {**d, 'new_key': 'new_value'}

    assert process_dict({'a': 1}) == pmap({'a': 1, 'new_key': 'new_value'})

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

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        return {**kwargs, 'new_key': 'new_value'}

    assert process_kwargs(a=1, b=2) == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        data['inner_list'].append(4)
        return data

    input_data = {'inner_list': [1, 2, 3]}
    result = process_nested(input_data)
    assert result == pmap({'inner_list': pvector([1, 2, 3, 4])})
    assert input_data == {'inner_list': [1, 2, 3]}  # Original should be unchanged

def test_mutant_with_pvector_arg():
    @mutant
    def process_pvector(pv):
        return pv.append(4)

    assert process_pvector(pvector([1, 2, 3])) == pvector([1, 2, 3, 4])

def test_mutant_with_pmap_arg():
    @mutant
    def process_pmap(pm):
        return pm.set('new_key', 'new_value')

    assert process_pmap(pmap({'a': 1})) == pmap({'a': 1, 'new_key': 'new_value'})

def test_mutant_with_pset_arg():
    @mutant
    def process_pset(ps):
        return ps.add(4)

    assert process_pset(pset({1, 2, 3})) == pset({1, 2, 3, 4})


# LLM-generated content at query #44
#--------------------------

```python
def test_mutant_predicate_false():
    assert not callable(mutant)


# LLM-generated content at query #45
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == 'test_func'


# LLM-generated content at query #46
#--------------------------

```python
def test_freeze_with_set_input():
    result = freeze(set([1, 2]))
    assert result == pset([1, 2])


# LLM-generated content at query #47
#--------------------------

```python
def test_freeze_predicate_false():
    assert not (type({}) is dict or (True and isinstance({}, PMap)))


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_freeze_with_empty_set():
    assert freeze(set()) == pset()

def test_freeze_with_non_empty_set():
    assert freeze(set([1, 2, 3])) == pset([1, 2, 3])

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
    assert freeze((1, (2, 3))) == (1, (2, 3))

def test_freeze_with_mixed_types():
    assert freeze([1, {'a': 2}, (3, 4), set([5, 6])]) == pvector([1, pmap({'a': 2}), (3, 4), pset([5, 6])])

def test_freeze_with_pvector():
    assert freeze(pvector([1, 2, 3]), strict=True) == pvector([1, 2, 3])

def test_freeze_with_pmap():
    assert freeze(pmap({'a': 1, 'b': 2}), strict=True) == pmap({'a': 1, 'b': 2})

def test_freeze_with_pset():
    assert freeze(pset([1, 2, 3])) == pset([1, 2, 3])

def test_freeze_with_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, a=1, b=2)
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

def test_freeze_with_non_strict_mode():
    assert freeze(pvector([1, [2, 3]]), strict=False) == pvector([1, [2, 3]])

def test_freeze_with_non_strict_mode_dict():
    assert freeze(pmap({'a': {'b': 2}}), strict=False) == pmap({'a': {'b': 2}})


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap()

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 1}}) == pmap({'a': pmap({'b': 1})})

def test_freeze_empty_list():
    assert freeze([]) == pvector()

def test_freeze_nested_list():
    assert freeze([{'a': 1}]) == pvector([pmap({'a': 1})])

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_nested_tuple():
    assert freeze(({'a': 1},)) == (pmap({'a': 1}),)

def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_set_with_elements():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_pmap_strict():
    assert freeze(pmap({'a': [1, 2]})) == pmap({'a': pvector([1, 2])})

def test_freeze_pvector_strict():
    assert freeze(pvector([{'a': 1}])) == pvector([pmap({'a': 1})])

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': [1, 2]}), strict=False) == pmap({'a': [1, 2]})

def test_freeze_non_strict_pvector():
    assert freeze(pvector([{'a': 1}]), strict=False) == pvector([{'a': 1}])

def test_freeze_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, a=1)
    assert freeze(dd) == pmap({'a': 1})

def test_freeze_mixed_types():
    assert freeze([1, {'a': (2, [3])}]) == pvector([1, pmap({'a': (2, pvector([3]))})])


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_with_strict_true_converts_pmap():
    pm = pmap({'a': 1})
    result = freeze(pm, strict=True)
    assert isinstance(result, PMap)


# LLM-generated content at query #4
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

def test_freeze_non_strict():
    assert freeze(pvector([1, [2, 3]]), strict=False) == pvector([1, [2, 3]])

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': [1, 2]}), strict=False) == pmap({'a': [1, 2]})


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
    def process_list(lst):
        lst.append(4)
        return lst
    assert process_list([1, 2, 3]) == pvector([1, 2, 3, 4])

def test_mutant_with_dict_argument():
    @mutant
    def process_dict(d):
        d['new_key'] = 'new_value'
        return d
    assert process_dict({'a': 1}) == pmap({'a': 1, 'new_key': 'new_value'})

def test_mutant_with_mixed_arguments():
    @mutant
    def process_mixed(a, b, c):
        return (a, b, c)
    assert process_mixed([1, 2], {'x': 3}, {4, 5}) == (pvector([1, 2]), pmap({'x': 3}), pset({4, 5}))

def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(**kwargs):
        return kwargs
    assert process_kwargs(x=1, y=[2, 3]) == pmap({'x': 1, 'y': pvector([2, 3])})

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        data['nested']['value'] = 10
        return data
    assert process_nested({'nested': {'value': 5}}) == pmap({'nested': pmap({'value': 10})})

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

def test_mutant_with_no_args():
    @mutant
    def no_args():
        return [1, 2, 3]
    assert no_args() == pvector([1, 2, 3])

def test_mutant_with_strict_false():
    @mutant
    def process_strict_false(data):
        return data
    result = process_strict_false(pvector([1, 2, 3]))
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #6
#--------------------------

```python
def test_freeze_with_empty_set():
    assert freeze(set()) == pset()

def test_freeze_with_non_empty_set():
    assert freeze(set([1, 2, 3])) == pset([1, 2, 3])

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
    assert freeze((1, (2, 3))) == (1, (2, 3))

def test_freeze_with_mixed_containers():
    assert freeze([1, {'a': (2, [3, 4])}]) == pvector([1, pmap({'a': (2, pvector([3, 4]))})])

def test_freeze_with_pvector():
    assert freeze(pvector([1, 2, 3]), strict=True) == pvector([1, 2, 3])

def test_freeze_with_pmap():
    assert freeze(pmap({'a': 1, 'b': 2}), strict=True) == pmap({'a': 1, 'b': 2})

def test_freeze_with_pset():
    assert freeze(pset([1, 2, 3])) == pset([1, 2, 3])

def test_freeze_with_defaultdict():
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

def test_freeze_with_non_strict_mode():
    assert freeze(pvector([1, [2, 3]]), strict=False) == pvector([1, [2, 3]])

def test_freeze_with_non_container():
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #7
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3

def test_mutant_with_list_input():
    @mutant
    def process_list(lst):
        lst.append(4)
        return lst

    result = process_list([1, 2, 3])
    assert result == (1, 2, 3, 4)

def test_mutant_with_dict_input():
    @mutant
    def process_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = process_dict({'a': 1, 'b': 2})
    assert result == {'a': 1, 'b': 2, 'new_key': 'new_value'}

def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        data['list'].append(4)
        data['nested_dict']['c'] = 3
        return data

    input_data = {'list': [1, 2, 3], 'nested_dict': {'a': 1, 'b': 2}}
    result = process_nested(input_data)
    assert result == {'list': (1, 2, 3, 4), 'nested_dict': {'a': 1, 'b': 2, 'c': 3}}

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        result = a + b
        for k, v in kwargs.items():
            result += v
        return result

    result = combine(1, 2, c=3, d=4)
    assert result == 10

def test_mutant_with_no_args():
    @mutant
    def get_constant():
        return [1, 2, 3]

    result = get_constant()
    assert result == (1, 2, 3)

def test_mutant_with_set_input():
    @mutant
    def process_set(s):
        s.add(4)
        return s

    result = process_set({1, 2, 3})
    assert result == {1, 2, 3, 4}

def test_mutant_with_tuple_input():
    @mutant
    def process_tuple(t):
        return t + (4,)

    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)


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
        return {**kwargs, 'extra': 1}

    result = merge(a=[1, 2], b={'c': 3})
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3}), 'extra': 1})

def test_mutant_with_no_args():
    @mutant
    def get_defaults():
        return {'a': [1, 2], 'b': {'c': 3}}

    result = get_defaults()
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s | {3, 4}

    result = process_set({1, 2})
    assert result == pset([1, 2, 3, 4])

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t + (3, 4)

    result = process_tuple((1, 2))
    assert result == (1, 2, 3, 4)


# LLM-generated content at query #9
#--------------------------

```python
def test_freeze_defaultdict():
    d = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #10
#--------------------------

```python
def test_freeze_converts_dict_to_pmap():
    result = freeze({'a': 1, 'b': 2})
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #11
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
    assert freeze([1, {'a': 2}, (3, [4])]) == pvector([1, pmap({'a': 2}), (3, pvector([4]))])

def test_freeze_pmap_strict():
    assert freeze(pmap({'a': 1}), strict=True) == pmap({'a': 1})

def test_freeze_pvector_strict():
    assert freeze(pvector([1, 2]), strict=True) == pvector([1, 2])

def test_freeze_non_strict_pmap():
    assert freeze(pmap({'a': 1}), strict=False) == pmap({'a': 1})

def test_freeze_non_strict_pvector():
    assert freeze(pvector([1, 2]), strict=False) == pvector([1, 2])

def test_freeze_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, {'a': 1})
    assert freeze(d) == pmap({'a': 1})

def test_freeze_nested_defaultdict():
    from collections import defaultdict
    d = defaultdict(int, {'a': defaultdict(int, {'b': 2})})
    assert freeze(d) == pmap({'a': pmap({'b': 2})})


# LLM-generated content at query #12
#--------------------------

```python
def test_mutant_with_list():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    result = add_to_list([1, 2], 3)
    assert result == pvector([1, 2, 3])

def test_mutant_with_dict():
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    result = add_to_dict({'a': 1}, 'b', 2)
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
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    input_data = {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    result = modify_nested(input_data)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'b': 2, 'c': 3})})

def test_mutant_with_no_mutation():
    @mutant
    def no_mutation(x):
        return x

    assert no_mutation(42) == 42
    assert no_mutation("hello") == "hello"


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant_with_simple_function():
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3]) == pvector([1, 2, 3])
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
        return [1, 2, 3]

    assert get_defaults() == pvector([1, 2, 3])

def test_mutant_with_set():
    @mutant
    def process_set(s):
        return s | {4, 5}

    assert process_set({1, 2, 3}) == pset([1, 2, 3, 4, 5])

def test_mutant_with_tuple():
    @mutant
    def process_tuple(t):
        return t + (4, 5)

    assert process_tuple((1, 2, 3)) == (1, 2, 3, 4, 5)

def test_mutant_preserves_immutability():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert result == pvector([1, 2, 3, 4])
    assert original == [1, 2, 3]


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == 'test_func'


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_predicate_false():
    assert not False


# LLM-generated content at query #16
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
        d['inner']['value'] = 42
        return d
    result = modify_nested({'inner': {'value': 0}})
    assert result == {'inner': {'value': 42}}

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return a + b + sum(kwargs.values())
    result = combine(1, 2, c=3, d=4)
    assert result == 10

def test_mutant_returns_frozen():
    @mutant
    def return_list():
        return [1, 2, 3]
    result = return_list()
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_pset():
    @mutant
    def process_set(s):
        return s | {4, 5}
    result = process_set({1, 2, 3})
    assert isinstance(result, pset)
    assert result == pset({1, 2, 3, 4, 5})

def test_mutant_with_pmap():
    @mutant
    def process_map(m):
        m['new_key'] = 'new_value'
        return m
    result = process_map({'a': 1})
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})


# LLM-generated content at query #17
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
    assert process({'a': {'b': [1, 2]}}) == pmap({'result': pmap({'a': pmap({'b': pvector([1, 2])})})})

def test_mutant_with_kwargs():
    @mutant
    def merge(**kwargs):
        return kwargs

    assert merge(a=1, b=[2, 3]) == pmap({'a': 1, 'b': pvector([2, 3])})
    assert merge(x={'y': [4, 5]}) == pmap({'x': pmap({'y': pvector([4, 5])})})

def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}

    assert combine(1, [2, 3], x=4, y=[5, 6]) == pmap({
        'a': 1,
        'b': pvector([2, 3]),
        'kwargs': pmap({'x': 4, 'y': pvector([5, 6])})
    })

def test_mutant_with_no_freeze_needed():
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("hello") == "hello"
    assert identity((1, 2, 3)) == (1, 2, 3)

def test_mutant_with_set():
    @mutant
    def set_op(s):
        return s | {1, 2, 3}

    assert set_op({4, 5}) == pset({1, 2, 3, 4, 5})
    assert set_op(pset({4, 5})) == pset({1, 2, 3, 4, 5})

def test_mutant_with_tuple():
    @mutant
    def tuple_op(t):
        return t + (1, 2)

    assert tuple_op((3, 4)) == (3, 4, 1, 2)
    assert tuple_op((3, [4, 5])) == (3, pvector([4, 5]), 1, 2)


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (False)


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_predicate_false():
    assert not (not False and True)


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_decorator_preserves_function_name():
    @mutant
    def test_func():
        pass
    assert test_func.__name__ == 'test_func'


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
        return {'result': data}

    assert process([1, [2, 3]]) == pmap({'result': pvector([1, pvector([2, 3])])})
    assert process({'a': {'b': 2}}) == pmap({'result': pmap({'a': pmap({'b': 2})})})

def test_mutant_with_set_input():
    @mutant
    def wrap_set(s):
        return {'set': s}

    assert wrap_set({1, 2, 3}) == pmap({'set': pset({1, 2, 3})})

def test_mutant_with_tuple_input():
    @mutant
    def wrap_tuple(t):
        return {'tuple': t}

    assert wrap_tuple((1, [2, 3])) == pmap({'tuple': (1, pvector([2, 3]))})

def test_mutant_with_kwargs():
    @mutant
    def combine(a, b, **kwargs):
        return {'a': a, 'b': b, 'kwargs': kwargs}

    result = combine(1, [2], c={3, 4}, d=(5, [6]))
    expected = pmap({
        'a': 1,
        'b': pvector([2]),
        'kwargs': pmap({
            'c': pset({3, 4}),
            'd': (5, pvector([6]))
        })
    })
    assert result == expected

def test_mutant_preserves_immutability():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert result == pvector([1, 2, 3, 4])
    assert original == [1, 2, 3]

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
        return pm.set('new_key', 'new_value')

    original = pmap({'a': 1})
    result = process_pmap(original)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert original == pmap({'a': 1})

def test_mutant_with_empty_structures():
    @mutant
    def empty():
        return {}, [], set(), ()

    result = empty()
    assert result == (pmap({}), pvector([]), pset(set()), ())

def test_mutant_with_non_container_types():
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("string") == "string"
    assert identity(None) is None


