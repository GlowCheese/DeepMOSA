####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_freeze():
    # Test basic types
    assert freeze(1) == 1
    assert freeze("hello") == "hello"
    assert freeze(None) is None

    # Test list to pvector conversion
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

    # Test dict to pmap conversion
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})
    assert freeze({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})

    # Test defaultdict to pmap conversion
    default_dict = collections.defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(default_dict) == pmap({'a': 1, 'b': 2})

    # Test set to pset conversion
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

    # Test tuple conversion
    assert freeze((1, 2, 3)) == (1, 2, 3)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

    # Test nested structures
    nested = {'a': [1, {'b': 2}], 'c': (3, [4, 5])}
    expected = pmap({'a': pvector([1, pmap({'b': 2})]), 'c': (3, pvector([4, 5]))})
    assert freeze(nested) == expected

    # Test strict mode
    pvec = pvector([1, 2, 3])
    assert freeze(pvec, strict=True) == pvector([1, 2, 3])
    assert freeze(pvec, strict=False) == pvec

    pmap_obj = pmap({'a': 1, 'b': 2})
    assert freeze(pmap_obj, strict=True) == pmap({'a': 1, 'b': 2})
    assert freeze(pmap_obj, strict=False) == pmap_obj

    # Test pset remains unchanged
    pset_obj = pset({1, 2, 3})
    assert freeze(pset_obj) == pset_obj


# LLM-generated content at query #2
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['nested']['value'] = 'changed'
        return data

    original = {'nested': {'value': 'original'}}
    result = modify_nested(original)
    assert original == {'nested': {'value': 'original'}}  # Original unchanged
    assert result == pmap({'nested': pmap({'value': 'changed'})})  # Return value is frozen

    # Test with kwargs
    @mutant
    def modify_kwargs(a, b, **kwargs):
        kwargs['c'] = 'new'
        return {'a': a, 'b': b, 'kwargs': kwargs}

    original_kwargs = {'c': 'old'}
    result = modify_kwargs(1, 2, **original_kwargs)
    assert original_kwargs == {'c': 'old'}  # Original unchanged
    assert result == pmap({'a': 1, 'b': 2, 'kwargs': pmap({'c': 'new'})})  # Return value is frozen

    # Test with strict=False
    @mutant
    def modify_with_strict_false(lst):
        return lst

    original = [1, 2, 3]
    result = modify_with_strict_false(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3])  # Still frozen (strict is default True)

    # Test return value is frozen
    @mutant
    def return_mutable():
        return {'key': [1, 2, 3]}

    result = return_mutable()
    assert isinstance(result, PMap)
    assert isinstance(result['key'], PVector)


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze():
    # Test basic types
    assert freeze(1) == 1
    assert freeze("hello") == "hello"
    assert freeze(None) is None

    # Test list to pvector
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

    # Test dict to pmap
    assert freeze({"a": 1}) == pmap({"a": 1})
    assert freeze({"a": [1, 2]}) == pmap({"a": pvector([1, 2])})

    # Test defaultdict to pmap
    assert freeze(collections.defaultdict(int, {"a": 1})) == pmap({"a": 1})

    # Test set to pset
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

    # Test tuple
    assert freeze((1, 2, 3)) == (1, 2, 3)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

    # Test nested structures
    nested = {"a": [1, {"b": 2}], "c": (3, [4, 5])}
    expected = pmap({"a": pvector([1, pmap({"b": 2})]), "c": (3, pvector([4, 5]))})
    assert freeze(nested) == expected

    # Test strict mode
    assert freeze(pvector([1, 2]), strict=True) == pvector([1, 2])
    assert freeze(pmap({"a": 1}), strict=True) == pmap({"a": 1})
    assert freeze(pvector([1, 2]), strict=False) == pvector([1, 2])
    assert freeze(pmap({"a": 1}), strict=False) == pmap({"a": 1})

    # Test non-strict mode with already frozen objects
    assert freeze(pvector([1, 2]), strict=False) == pvector([1, 2])
    assert freeze(pmap({"a": 1}), strict=False) == pmap({"a": 1})


# LLM-generated content at query #4
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list should be unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict should be unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = nested_operation(original_data)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original should be unchanged

    # Test with kwargs
    @mutant
    def add_to_set(s, value):
        s.add(value)
        return s

    original_set = {1, 2, 3}
    result = add_to_set(s=original_set, value=4)
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})
    assert original_set == {1, 2, 3}  # Original set should be unchanged


# LLM-generated content at query #5
#--------------------------

```python
def test_freeze():
    # Test freezing a list with nested structures
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

    # Test freezing a tuple with nested structures
    assert freeze((1, [])) == (1, pvector([]))

    # Test freezing a set
    assert freeze(set([1, 2])) == pset([1, 2])

    # Test freezing a dict
    assert freeze({'a': 1, 'b': [2, 3]}) == pmap({'a': 1, 'b': pvector([2, 3])})

    # Test freezing a defaultdict
    dd = collections.defaultdict(int, {'a': 1, 'b': [2, 3]})
    assert freeze(dd) == pmap({'a': 1, 'b': pvector([2, 3])})

    # Test freezing with strict=False
    assert freeze([1, {'a': 3}], strict=False) == pvector([1, {'a': 3}])

    # Test freezing already frozen structures
    pv = pvector([1, 2, 3])
    assert freeze(pv) == pv
    pm = pmap({'a': 1})
    assert freeze(pm) == pm
    ps = pset([1, 2])
    assert freeze(ps) == ps

    # Test freezing non-container types
    assert freeze(1) == 1
    assert freeze("hello") == "hello"


# LLM-generated content at query #6
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list should be unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict should be unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(10)
        data['set'].add(20)
        return data

    original_data = {'list': [1, 2], 'set': {3, 4}}
    result = nested_operation(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)
    assert result == pmap({'list': pvector([1, 2, 10]), 'set': pset({3, 4, 20})})
    assert original_data == {'list': [1, 2], 'set': {3, 4}}  # Original should be unchanged

    # Test with strict=False
    @mutant
    def non_strict_operation(lst):
        return lst + [5]

    result = non_strict_operation([1, 2])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 5])


# LLM-generated content at query #7
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    frozen_list = add_to_list(original_list, 4)

    assert isinstance(frozen_list, PVector)
    assert frozen_list == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list should be unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    frozen_dict = modify_dict(original_dict, 'b', 2)

    assert isinstance(frozen_dict, PMap)
    assert frozen_dict == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict should be unchanged

    @mutant
    def nested_operation(data):
        data['nested']['value'] = 10
        return data

    original_data = {'nested': {'value': 5}}
    frozen_data = nested_operation(original_data)

    assert isinstance(frozen_data, PMap)
    assert frozen_data == pmap({'nested': pmap({'value': 10})})
    assert original_data == {'nested': {'value': 5}}  # Original data should be unchanged

    # Test with kwargs
    @mutant
    def modify_with_kwargs(lst, **kwargs):
        lst.extend(kwargs['values'])
        return lst

    original_list = [1]
    frozen_list = modify_with_kwargs(original_list, values=[2, 3])

    assert isinstance(frozen_list, PVector)
    assert frozen_list == pvector([1, 2, 3])
    assert original_list == [1]  # Original list should be unchanged


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]
    assert result == pvector([1, 2, 3, 4])

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['nested'].append(5)
        return data

    original = {'nested': [1, 2]}
    result = modify_nested(original)
    assert original == {'nested': [1, 2]}
    assert result == pmap({'nested': pvector([1, 2, 5])})

    # Test with multiple arguments
    @mutant
    def combine_and_modify(a, b):
        a['key'] = 'modified'
        b.append('new')
        return {'a': a, 'b': b}

    dict_arg = {'key': 'original'}
    list_arg = ['old']
    result = combine_and_modify(dict_arg, list_arg)
    assert dict_arg == {'key': 'original'}
    assert list_arg == ['old']
    assert result == pmap({
        'a': pmap({'key': 'modified'}),
        'b': pvector(['old', 'new'])
    })

    # Test with keyword arguments
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['value'].add(10)
        return kwargs

    kwargs_arg = {'value': {1, 2}}
    result = modify_kwargs(**kwargs_arg)
    assert kwargs_arg == {'value': {1, 2}}
    assert result == pmap({'value': pset({1, 2, 10})})

    # Test return value is frozen
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def test_strict_false():
        return {'a': [1, 2]}

    result = test_strict_false()
    assert isinstance(result, dict)
    assert isinstance(result['a'], list)


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    @mutant
    def nested_operation(data):
        data['list'].append(1)
        data['dict']['key'] = 'new_value'
        return data

    # Test with list
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original should be unchanged

    # Test with dict
    original_dict = {'a': 1, 'b': 2}
    result = modify_dict(original_dict, 'c', 3)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})
    assert original_dict == {'a': 1, 'b': 2}  # Original should be unchanged

    # Test with nested structures
    original_data = {'list': [1, 2], 'dict': {'key': 'value'}}
    result = nested_operation(original_data)
    assert isinstance(result, PMap)
    assert result['list'] == pvector([1, 2, 1])
    assert result['dict'] == pmap({'key': 'new_value'})
    assert original_data == {'list': [1, 2], 'dict': {'key': 'value'}}  # Original should be unchanged

    # Test with kwargs
    @mutant
    def modify_with_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'a': 1}
    result = modify_with_kwargs(**original_kwargs)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert original_kwargs == {'a': 1}  # Original should be unchanged


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original = [1, 2, 3]
    result = add_to_list(original, 4)
    assert original == [1, 2, 3]  # Original should be unchanged
    assert result == pvector([1, 2, 3, 4])  # Return should be frozen

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 10
        return d

    original_dict = {'nested': {'value': 5}}
    result_dict = modify_nested(original_dict)
    assert original_dict == {'nested': {'value': 5}}  # Original unchanged
    assert result_dict == pmap({'nested': pmap({'value': 10})})  # Return frozen

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 20
        return kwargs

    original_kwargs = {'x': 10, 'y': 30}
    result_kwargs = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'x': 10, 'y': 30}  # Original unchanged
    assert result_kwargs == pmap({'x': 20, 'y': 30})  # Return frozen

    # Test with non-mutable operations
    @mutant
    def pure_function(x, y):
        return x + y

    assert pure_function(2, 3) == 5  # Should work normally

    # Test with already frozen input
    frozen_input = pvector([1, 2, 3])
    @mutant
    def process_frozen(v):
        return v.append(4)

    result = process_frozen(frozen_input)
    assert frozen_input == pvector([1, 2, 3])  # Original frozen input unchanged
    assert result == pvector([1, 2, 3, 4])  # Return should be new frozen vector


# LLM-generated content at query #11
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original should be unchanged

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 10
        return d

    original_dict = {'nested': {'value': 5}}
    result = modify_nested(original_dict)
    assert isinstance(result, PMap)
    assert result == pmap({'nested': pmap({'value': 10})})
    assert original_dict == {'nested': {'value': 5}}  # Original should be unchanged

    # Test with multiple arguments
    @mutant
    def combine(lst1, lst2):
        lst1.extend(lst2)
        return lst1

    list1 = [1, 2]
    list2 = [3, 4]
    result = combine(list1, list2)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert list1 == [1, 2]  # Original should be unchanged
    assert list2 == [3, 4]  # Original should be unchanged

    # Test with keyword arguments
    @mutant
    def update_dict(d, key='default', value=0):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, key='b', value=2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original should be unchanged

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with non-container types (should pass through)
    @mutant
    def add_numbers(a, b):
        return a + b

    assert add_numbers(1, 2) == 3


# LLM-generated content at query #12
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original = [1, 2, 3]
    result = add_to_list(original, 4)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])

    # Test with nested structures
    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': [1, 2]}
    result_dict = modify_dict(original_dict, 'a', [3, 4])
    assert original_dict == {'a': [1, 2]}  # Original unchanged
    assert result_dict == pmap({'a': pvector([3, 4])})

    # Test with kwargs
    @mutant
    def update_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'old_key': 'old_value'}
    result_kwargs = update_kwargs(**original_kwargs)
    assert original_kwargs == {'old_key': 'old_value'}  # Original unchanged
    assert result_kwargs == pmap({'old_key': 'old_value', 'new_key': 'new_value'})

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def add_to_list_non_strict(lst, item):
        lst.append(item)
        return lst

    original = [1, 2, 3]
    result = add_to_list_non_strict(original, 4)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    # Test with dict
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['nested']['value'] = 10
        return data

    original = {'list': [1, 2, 3], 'nested': {'value': 5}}
    result = modify_nested(original)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['nested'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'nested': pmap({'value': 10})})
    assert original == {'list': [1, 2, 3], 'nested': {'value': 5}}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    original_kwargs = {'x': 1, 'y': 2}
    result = modify_kwargs(**original_kwargs)
    assert isinstance(result, PMap)
    assert result == pmap({'x': 10, 'y': 2})
    assert original_kwargs == {'x': 1, 'y': 2}  # Original unchanged

    # Test return value freezing
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant():
    @mutant
    def test_func(a, b, c=None):
        a.append(4)
        b['d'] = 4
        if c is not None:
            c.add(5)
        return {'result': [a, b, c]}

    # Test with mutable arguments
    input_list = [1, 2, 3]
    input_dict = {'a': 1, 'b': 2}
    input_set = {1, 2, 3}

    result = test_func(input_list, input_dict, input_set)

    # Check that original arguments are not modified
    assert input_list == [1, 2, 3]
    assert input_dict == {'a': 1, 'b': 2}
    assert input_set == {1, 2, 3}

    # Check that result is frozen
    assert isinstance(result, PMap)
    assert isinstance(result['result'], PVector)
    assert len(result['result']) == 3
    assert isinstance(result['result'][0], PVector)
    assert isinstance(result['result'][1], PMap)
    assert isinstance(result['result'][2], PSet)

    # Check that the frozen result contains the expected values
    assert result['result'][0] == pvector([1, 2, 3, 4])
    assert result['result'][1] == pmap({'a': 1, 'b': 2, 'd': 4})
    assert result['result'][2] == pset({1, 2, 3, 5})

    # Test with None argument
    result_none = test_func(input_list, input_dict)
    assert result_none['result'][2] is None

    # Test with no mutation
    @mutant
    def no_mutation_func(x):
        return x

    test_value = [1, 2, 3]
    result_no_mut = no_mutation_func(test_value)
    assert result_no_mut == pvector([1, 2, 3])
    assert test_value == [1, 2, 3]


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant():
    # Test basic functionality with simple arguments
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])

    # Test with mutable arguments
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original list unchanged

    # Test with keyword arguments
    @mutant
    def combine(a, b, c=10):
        return a + b + c

    assert combine(1, 2, c=3) == 6

    # Test with nested structures
    @mutant
    def nested_operation(data):
        data['values'].append(5)
        return data

    input_dict = {'values': [1, 2, 3]}
    result = nested_operation(input_dict)
    assert result == pmap({'values': pvector([1, 2, 3, 5])})
    assert input_dict == {'values': [1, 2, 3]}  # Original dict unchanged

    # Test with tuple arguments
    @mutant
    def process_tuple(t):
        return t + (4,)

    input_tuple = (1, 2, 3)
    result = process_tuple(input_tuple)
    assert result == (1, 2, 3, 4)
    assert input_tuple == (1, 2, 3)  # Original tuple unchanged

    # Test with set arguments
    @mutant
    def process_set(s):
        return s | {4}

    input_set = {1, 2, 3}
    result = process_set(input_set)
    assert result == pset({1, 2, 3, 4})
    assert input_set == {1, 2, 3}  # Original set unchanged

    # Test with mixed argument types
    @mutant
    def mixed_args(a, b, c):
        b['key'] = a
        c.add(5)
        return (a, b, c)

    input_list = [1, 2]
    input_dict = {'key': 0}
    input_set = {1, 2, 3}
    result = mixed_args(10, input_dict, input_set)
    assert result == (10, pmap({'key': 10}), pset({1, 2, 3, 5}))
    assert input_list == [1, 2]
    assert input_dict == {'key': 0}
    assert input_set == {1, 2, 3}


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(1)
        data['dict']['new_key'] = 'new_value'
        return data

    original_data = {'list': [1, 2], 'dict': {'key': 'value'}}
    result = nested_operation(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'key': 'value', 'new_key': 'new_value'})})
    assert original_data == {'list': [1, 2], 'dict': {'key': 'value'}}  # Original data unchanged

    # Test with kwargs
    @mutant
    def kwargs_test(a, b=1):
        return {'a': a + 1, 'b': b + 1}

    result = kwargs_test(10, b=20)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 11, 'b': 21})


# LLM-generated content at query #17
#--------------------------

```python
def test_freeze():
    # Test basic types
    assert freeze(1) == 1
    assert freeze("hello") == "hello"
    assert freeze(None) is None

    # Test list to pvector conversion
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

    # Test dict to pmap conversion
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

    # Test defaultdict to pmap conversion
    default_dict = collections.defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(default_dict) == pmap({'a': 1, 'b': 2})

    # Test tuple conversion
    assert freeze((1, 2, 3)) == (1, 2, 3)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

    # Test set to pset conversion
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

    # Test nested structures
    nested = {
        'a': [1, 2, {'b': 3}],
        'c': (4, [5, 6])
    }
    expected = pmap({
        'a': pvector([1, 2, pmap({'b': 3})]),
        'c': (4, pvector([5, 6]))
    })
    assert freeze(nested) == expected

    # Test strict=False
    assert freeze([1, {'a': 2}], strict=False) == pvector([1, {'a': 2}])
    assert freeze({'a': [1, 2]}, strict=False) == pmap({'a': [1, 2]})

    # Test already frozen types
    pvec = pvector([1, 2, 3])
    assert freeze(pvec) == pvec
    pmap_obj = pmap({'a': 1})
    assert freeze(pmap_obj) == pmap_obj
    pset_obj = pset({1, 2})
    assert freeze(pset_obj) == pset_obj


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original = [1, 2, 3]
    result = add_to_list(original, 4)
    assert original == [1, 2, 3]
    assert result == pvector([1, 2, 3, 4])

    # Test with nested structures
    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': [1, 2]}
    result_dict = modify_dict(original_dict, 'a', [3, 4])
    assert original_dict == {'a': [1, 2]}
    assert result_dict == pmap({'a': pvector([3, 4])})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'key': 'value'}
    result_kwargs = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'key': 'value'}
    assert result_kwargs == pmap({'key': 'value', 'new_key': 'new_value'})

    # Test with mixed args and kwargs
    @mutant
    def mixed_modification(lst, d, **kwargs):
        lst.append(1)
        d['key'] = 'modified'
        kwargs['kwarg'] = 'modified'
        return lst, d, kwargs

    original_list = [1, 2]
    original_dict = {'key': 'original'}
    original_kwargs = {'kwarg': 'original'}
    result_list, result_dict, result_kwargs = mixed_modification(original_list, original_dict, **original_kwargs)
    assert original_list == [1, 2]
    assert original_dict == {'key': 'original'}
    assert original_kwargs == {'kwarg': 'original'}
    assert result_list == pvector([1, 2, 1])
    assert result_dict == pmap({'key': 'modified'})
    assert result_kwargs == pmap({'kwarg': 'modified'})

    # Test with non-mutable types
    @mutant
    def no_mutation(x):
        return x + 1

    assert no_mutation(5) == 6

    # Test with tuple
    @mutant
    def modify_tuple(t):
        return t + (1,)

    original_tuple = (1, 2)
    result_tuple = modify_tuple(original_tuple)
    assert original_tuple == (1, 2)
    assert result_tuple == (1, 2, 1)


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original should be unchanged

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 10
        return d

    original_dict = {'nested': {'value': 5}}
    result = modify_nested(original_dict)
    assert isinstance(result, PMap)
    assert result == pmap({'nested': pmap({'value': 10})})
    assert original_dict == {'nested': {'value': 5}}  # Original should be unchanged

    # Test with multiple arguments
    @mutant
    def combine_dicts(d1, d2):
        d1.update(d2)
        return d1

    dict1 = {'a': 1}
    dict2 = {'b': 2}
    result = combine_dicts(dict1, dict2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert dict1 == {'a': 1}  # Original should be unchanged
    assert dict2 == {'b': 2}  # Original should be unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'old_key': 'old_value'}
    result = modify_kwargs(**original_kwargs)
    assert isinstance(result, PMap)
    assert result == pmap({'old_key': 'old_value', 'new_key': 'new_value'})
    assert original_kwargs == {'old_key': 'old_value'}  # Original should be unchanged

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def no_strict(lst):
        return lst

    original_list = pvector([1, 2, 3])
    result = no_strict(original_list)
    assert isinstance(result, PVector)
    assert result == original_list


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['nested']['value'] = 'modified'
        return data

    original_dict = {'nested': {'value': 'original'}, 'other': [1, 2]}
    result_dict = modify_nested(original_dict)
    assert original_dict == {'nested': {'value': 'original'}, 'other': [1, 2]}
    assert result_dict == pmap({'nested': pmap({'value': 'modified'}), 'other': pvector([1, 2])})

    # Test with kwargs
    @mutant
    def modify_kwargs(a, b, **kwargs):
        kwargs['c'] = 'new'
        return {'a': a, 'b': b, 'kwargs': kwargs}

    original_kwargs = {'c': 'old', 'd': [1, 2]}
    result_kwargs = modify_kwargs(1, 2, **original_kwargs)
    assert original_kwargs == {'c': 'old', 'd': [1, 2]}
    assert result_kwargs == pmap({'a': 1, 'b': 2, 'kwargs': pmap({'c': 'new', 'd': pvector([1, 2])})})

    # Test with tuple (immutable)
    @mutant
    def process_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result_tuple = process_tuple(original_tuple)
    assert original_tuple == (1, 2, 3)
    assert result_tuple == (1, 2, 3, 4)

    # Test with set
    @mutant
    def process_set(s):
        return s | {4}

    original_set = {1, 2, 3}
    result_set = process_set(original_set)
    assert original_set == {1, 2, 3}
    assert result_set == pset({1, 2, 3, 4})

    # Test strict=False behavior
    @mutant
    def modify_list_non_strict(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list_non_strict(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])  # Frozen result

    # Test with dict mutation
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}  # Original unchanged
    assert result == pmap({'a': 1, 'b': 2})  # Frozen result

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = modify_nested(original)
    assert original == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original unchanged
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    original_kwargs = {'y': 20}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'y': 20}  # Original unchanged
    assert result == pmap({'y': 20, 'x': 10})

    # Test return value is frozen
    @mutant
    def return_new_list():
        return [1, 2, 3]

    result = return_new_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def non_strict_test(lst):
        return lst

    result = non_strict_test([1, 2, 3])
    assert result == [1, 2, 3]  # Not frozen when strict=False in freeze/thaw


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    # Test with list
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    # Test with dict
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_nested = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = modify_nested(original_nested)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original unchanged

    # Test with kwargs
    @mutant
    def process_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'key1': 'value1'}
    result = process_kwargs(**original_kwargs)
    assert isinstance(result, PMap)
    assert result == pmap({'key1': 'value1', 'new_key': 'new_value'})
    assert original_kwargs == {'key1': 'value1'}  # Original kwargs unchanged


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant():
    @mutant
    def test_func(a, b, c=None):
        a.append(1)
        b['key'] = 'value'
        if c is not None:
            c.add(5)
        return {'result': a, 'data': b, 'extra': c}

    # Test with list, dict, and set
    input_list = [1, 2, 3]
    input_dict = {'key': 'old_value'}
    input_set = {1, 2, 3}

    result = test_func(input_list, input_dict, input_set)

    # Check that inputs are unchanged
    assert input_list == [1, 2, 3]
    assert input_dict == {'key': 'old_value'}
    assert input_set == {1, 2, 3}

    # Check that result is frozen
    assert isinstance(result['result'], PVector)
    assert isinstance(result['data'], PMap)
    assert isinstance(result['extra'], PSet)

    # Check that result values are correct
    assert result['result'] == pvector([1, 2, 3, 1])
    assert result['data'] == pmap({'key': 'value'})
    assert result['extra'] == pset({1, 2, 3, 5})

    # Test with None for optional parameter
    result_none = test_func(input_list, input_dict)
    assert result_none['extra'] is None

    # Test with immutable inputs
    @mutant
    def test_immutable(a, b):
        return a + b

    assert test_immutable(1, 2) == 3


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)

    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list should remain unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result_dict = modify_dict(original_dict, 'b', 2)

    assert isinstance(result_dict, PMap)
    assert result_dict == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict should remain unchanged

    @mutant
    def nested_operations(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result_data = nested_operations(original_data)

    assert isinstance(result_data, PMap)
    assert isinstance(result_data['list'], PVector)
    assert isinstance(result_data['dict'], PMap)
    assert result_data == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original data should remain unchanged

    # Test with kwargs
    @mutant
    def add_to_set(s, value):
        s.add(value)
        return s

    original_set = {1, 2, 3}
    result_set = add_to_set(s=original_set, value=4)

    assert isinstance(result_set, PSet)
    assert result_set == pset({1, 2, 3, 4})
    assert original_set == {1, 2, 3}  # Original set should remain unchanged


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original should be unchanged
    assert result == pvector([1, 2, 3, 4])

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 100
        return d

    original_dict = {'nested': {'value': 50}}
    result = modify_nested(original_dict)
    assert original_dict == {'nested': {'value': 50}}  # Original should be unchanged
    assert result == pmap({'nested': pmap({'value': 100})})

    # Test with multiple arguments
    @mutant
    def combine_dicts(d1, d2):
        d1.update(d2)
        return d1

    dict1 = {'a': 1}
    dict2 = {'b': 2}
    result = combine_dicts(dict1, dict2)
    assert dict1 == {'a': 1}  # Original should be unchanged
    assert dict2 == {'b': 2}  # Original should be unchanged
    assert result == pmap({'a': 1, 'b': 2})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'key': 'value'}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'key': 'value'}  # Original should be unchanged
    assert result == pmap({'key': 'value', 'new_key': 'new_value'})

    # Test with return value freezing
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def test_strict_false():
        return {'a': [1, 2]}

    result = test_strict_false()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original = [1, 2, 3]
    result = add_to_list(original, 4)
    assert isinstance(result, PVector)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 100
        return d

    original_dict = {'nested': {'value': 50}}
    result_dict = modify_nested(original_dict)
    assert isinstance(result_dict, PMap)
    assert original_dict == {'nested': {'value': 50}}  # Original unchanged
    assert result_dict == pmap({'nested': pmap({'value': 100})})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    original_kwargs = {'x': 5, 'y': 20}
    result_kwargs = modify_kwargs(**original_kwargs)
    assert isinstance(result_kwargs, PMap)
    assert original_kwargs == {'x': 5, 'y': 20}  # Original unchanged
    assert result_kwargs == pmap({'x': 10, 'y': 20})

    # Test with mixed args and kwargs
    @mutant
    def mixed_modification(lst, d, **kwargs):
        lst.append(1)
        d['new_key'] = 'new_value'
        kwargs['z'] = 30
        return lst, d, kwargs

    original_list = [1, 2]
    original_dict = {'a': 1}
    original_kwargs = {'z': 10}
    result_list, result_dict, result_kwargs = mixed_modification(original_list, original_dict, **original_kwargs)

    assert isinstance(result_list, PVector)
    assert isinstance(result_dict, PMap)
    assert isinstance(result_kwargs, PMap)

    assert original_list == [1, 2]
    assert original_dict == {'a': 1}
    assert original_kwargs == {'z': 10}

    assert result_list == pvector([1, 2, 1])
    assert result_dict == pmap({'a': 1, 'new_key': 'new_value'})
    assert result_kwargs == pmap({'z': 30})

    # Test with immutable input
    @mutant
    def immutable_input(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result_tuple = immutable_input(original_tuple)
    assert isinstance(result_tuple, tuple)
    assert original_tuple == (1, 2, 3)
    assert result_tuple == (1, 2, 3, 4)


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    # Test with list
    input_list = [1, 2, 3]
    result = add_to_list(input_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original list unchanged

    # Test with dict
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    input_dict = {'a': 1}
    result = add_to_dict(input_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert input_dict == {'a': 1}  # Original dict unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    input_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = modify_nested(input_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert input_data == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'x': 10})

    # Test with tuple (should remain tuple)
    @mutant
    def modify_tuple(t):
        return t + (4,)

    input_tuple = (1, 2, 3)
    result = modify_tuple(input_tuple)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)
    assert input_tuple == (1, 2, 3)  # Original unchanged


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)

    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)

    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(10)
        data['set'].add(20)
        return data

    original_data = {'list': [1, 2], 'set': {3, 4}}
    result = nested_operation(original_data)

    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)
    assert result == pmap({'list': pvector([1, 2, 10]), 'set': pset({3, 4, 20})})
    assert original_data == {'list': [1, 2], 'set': {3, 4}}  # Original unchanged

    # Test with kwargs
    @mutant
    def kwargs_test(a, b, **kwargs):
        kwargs['c'] = a + b
        return kwargs

    result = kwargs_test(1, 2, d=3)
    assert isinstance(result, PMap)
    assert result == pmap({'d': 3, 'c': 3})


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant():
    # Test basic functionality with simple arguments
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])

    # Test with mutable arguments
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original list should be unchanged

    # Test with keyword arguments
    @mutant
    def combine(a, b, c=10):
        return a + b + c

    assert combine(1, 2, c=3) == 6

    # Test with nested structures
    @mutant
    def process_dict(d):
        d['new_key'] = 'new_value'
        return d

    input_dict = {'a': 1, 'b': [2, 3]}
    result = process_dict(input_dict)
    assert result == pmap({'a': 1, 'b': pvector([2, 3]), 'new_key': 'new_value'})
    assert input_dict == {'a': 1, 'b': [2, 3]}  # Original dict should be unchanged

    # Test with strict=False
    @mutant
    def no_strict_process(lst):
        return lst

    assert no_strict_process([1, 2, 3]) == pvector([1, 2, 3])

    # Test that the decorator preserves function metadata
    @mutant
    def example_function(x):
        """Example function for testing metadata preservation."""
        return x * 2

    assert example_function.__name__ == 'example_function'
    assert example_function.__doc__ == "Example function for testing metadata preservation."


# LLM-generated content at query #30
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    # Test with list
    input_list = [1, 2, 3]
    result = add_to_list(input_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original list unchanged

    # Test with dict
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    input_dict = {'a': 1}
    result = add_to_dict(input_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert input_dict == {'a': 1}  # Original dict unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    input_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = modify_nested(input_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert input_data == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'x': 10})

    # Test with tuple (should remain tuple)
    @mutant
    def modify_tuple(t):
        return t + (4,)

    input_tuple = (1, 2, 3)
    result = modify_tuple(input_tuple)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)
    assert input_tuple == (1, 2, 3)  # Original unchanged

    # Test with set (should become pset)
    @mutant
    def modify_set(s):
        return s | {4}

    input_set = {1, 2, 3}
    result = modify_set(input_set)
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})
    assert input_set == {1, 2, 3}  # Original unchanged


# LLM-generated content at query #31
#--------------------------

```python
def test_mutant():
    @mutant
    def example_function(a, b, c=None):
        a.append(1)
        b['key'] = 'value'
        if c is not None:
            c.add(2)
        return {'result': a, 'data': b, 'set': c}

    # Test with list, dict, and set
    input_list = [1, 2, 3]
    input_dict = {'key': 'old_value'}
    input_set = {3, 4}

    result = example_function(input_list, input_dict, input_set)

    # Check that original inputs are unchanged
    assert input_list == [1, 2, 3]
    assert input_dict == {'key': 'old_value'}
    assert input_set == {3, 4}

    # Check that result is frozen
    assert isinstance(result['result'], PVector)
    assert isinstance(result['data'], PMap)
    assert isinstance(result['set'], PSet)

    # Check that result values are correct
    assert result['result'] == pvector([1, 2, 3, 1])
    assert result['data'] == pmap({'key': 'value'})
    assert result['set'] == pset({3, 4, 2})

    # Test with None for optional parameter
    result_none = example_function([1], {'a': 1})
    assert isinstance(result_none['result'], PVector)
    assert isinstance(result_none['data'], PMap)
    assert result_none['set'] is None

    # Test with nested structures
    @mutant
    def nested_function(x):
        x['nested']['value'] = 10
        return x

    nested_input = {'nested': {'value': 5}}
    nested_result = nested_function(nested_input)
    assert nested_input == {'nested': {'value': 5}}
    assert isinstance(nested_result['nested'], PMap)
    assert nested_result['nested']['value'] == 10


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original = [1, 2, 3]
    result = add_to_list(original, 4)
    assert original == [1, 2, 3]
    assert result == pvector([1, 2, 3, 4])

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 10
        return d

    original_dict = {'nested': {'value': 5}}
    result_dict = modify_nested(original_dict)
    assert original_dict == {'nested': {'value': 5}}
    assert result_dict == pmap({'nested': pmap({'value': 10})})

    # Test with multiple arguments
    @mutant
    def combine_dicts(d1, d2):
        d1.update(d2)
        return d1

    dict1 = {'a': 1}
    dict2 = {'b': 2}
    result = combine_dicts(dict1, dict2)
    assert dict1 == {'a': 1}
    assert dict2 == {'b': 2}
    assert result == pmap({'a': 1, 'b': 2})

    # Test with keyword arguments
    @mutant
    def modify_with_kwargs(lst, value=5):
        lst.append(value)
        return lst

    original = [1, 2]
    result = modify_with_kwargs(original, value=3)
    assert original == [1, 2]
    assert result == pvector([1, 2, 3])

    # Test with no mutation
    @mutant
    def no_mutation(x):
        return x + 1

    assert no_mutation(5) == 6

    # Test with complex nested structures
    @mutant
    def complex_modification(data):
        data['list'].append(4)
        data['set'].add(5)
        data['nested']['value'] = 20
        return data

    original = {
        'list': [1, 2, 3],
        'set': {1, 2, 3},
        'nested': {'value': 10}
    }
    result = complex_modification(original)
    assert original == {
        'list': [1, 2, 3],
        'set': {1, 2, 3},
        'nested': {'value': 10}
    }
    assert result == pmap({
        'list': pvector([1, 2, 3, 4]),
        'set': pset({1, 2, 3, 5}),
        'nested': pmap({'value': 20})
    })


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    @mutant
    def nested_operation(data):
        data['list'].append(10)
        data['dict']['new_key'] = 'new_value'
        return data

    # Test list mutation
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original unchanged

    # Test dict mutation
    original_dict = {'a': 1, 'b': 2}
    result = modify_dict(original_dict, 'c', 3)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})
    assert original_dict == {'a': 1, 'b': 2}  # Original unchanged

    # Test nested mutation
    original_data = {'list': [1, 2], 'dict': {'key': 'value'}}
    result = nested_operation(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 10]), 'dict': pmap({'key': 'value', 'new_key': 'new_value'})})
    assert original_data == {'list': [1, 2], 'dict': {'key': 'value'}}  # Original unchanged

    # Test with kwargs
    @mutant
    def kwargs_test(a, b, **kwargs):
        kwargs['new_key'] = 'new_value'
        return {'a': a, 'b': b, 'kwargs': kwargs}

    result = kwargs_test(1, 2, x=10, y=20)
    assert isinstance(result, PMap)
    assert isinstance(result['kwargs'], PMap)
    assert result == pmap({'a': 1, 'b': 2, 'kwargs': pmap({'x': 10, 'y': 20, 'new_key': 'new_value'})})


# LLM-generated content at query #34
#--------------------------

```python
def test_mutant():
    # Test basic mutation prevention
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original unchanged

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 100
        return d

    original_dict = {'nested': {'value': 50}}
    result = modify_nested(original_dict)
    assert isinstance(result, PMap)
    assert result == pmap({'nested': pmap({'value': 100})})
    assert original_dict == {'nested': {'value': 50}}  # Original unchanged

    # Test with multiple arguments
    @mutant
    def combine(a, b):
        a.update(b)
        return a

    dict1 = {'x': 1}
    dict2 = {'y': 2}
    result = combine(dict1, dict2)
    assert isinstance(result, PMap)
    assert result == pmap({'x': 1, 'y': 2})
    assert dict1 == {'x': 1}  # Original unchanged
    assert dict2 == {'y': 2}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(existing='value')
    assert isinstance(result, PMap)
    assert result == pmap({'existing': 'value', 'new_key': 'new_value'})

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False behavior
    @mutant
    def test_strict_false(lst):
        return lst

    # This should still freeze because mutant always uses strict=True
    result = test_strict_false([1, 2, 3])
    assert isinstance(result, PVector)


# LLM-generated content at query #35
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])  # Frozen result

    # Test with dict
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    original_dict = {'a': 1}
    result_dict = modify_dict(original_dict)
    assert original_dict == {'a': 1}  # Original unchanged
    assert result_dict == pmap({'a': 1, 'new_key': 'new_value'})  # Frozen result

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(5)
        data['dict']['b'] = 2
        return data

    original_nested = {'list': [1, 2], 'dict': {'a': 1}}
    result_nested = modify_nested(original_nested)
    assert original_nested == {'list': [1, 2], 'dict': {'a': 1}}  # Original unchanged
    assert result_nested == pmap({
        'list': pvector([1, 2, 5]),
        'dict': pmap({'a': 1, 'b': 2})
    })

    # Test with kwargs
    @mutant
    def modify_kwargs(a, b=None):
        if b is not None:
            b['modified'] = True
        return a, b

    original_kwarg = {'key': 'value'}
    result_a, result_b = modify_kwargs([1, 2], b=original_kwarg)
    assert original_kwarg == {'key': 'value'}  # Original unchanged
    assert result_a == pvector([1, 2])
    assert result_b == pmap({'key': 'value', 'modified': True})

    # Test return value is frozen
    @mutant
    def return_mutable():
        return {'a': [1, 2, 3]}

    result = return_mutable()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)


# LLM-generated content at query #36
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list should remain unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict should remain unchanged

    @mutant
    def nested_operations(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = nested_operations(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original data should remain unchanged

    @mutant
    def no_mutation(value):
        return value + 1

    assert no_mutation(5) == 6


# LLM-generated content at query #37
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)

    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list should be unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)

    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict should be unchanged

    @mutant
    def modify_set(s, item):
        s.add(item)
        return s

    original_set = {1, 2}
    result = modify_set(original_set, 3)

    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3})
    assert original_set == {1, 2}  # Original set should be unchanged

    @mutant
    def modify_tuple(t, index, value):
        lst = list(t)
        lst[index] = value
        return tuple(lst)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple, 1, 4)

    assert result == (1, 4, 3)
    assert original_tuple == (1, 2, 3)  # Original tuple should be unchanged

    @mutant
    def nested_modification(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    result = nested_modification(original_data)

    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result['list'] == pvector([1, 2, 3, 4])
    assert result['dict'] == pmap({'a': 1, 'b': 2, 'c': 3})
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}  # Original should be unchanged


# LLM-generated content at query #38
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)

    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)

    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    @mutant
    def nested_operations(data):
        data['list'].append(1)
        data['dict']['new_key'] = 'value'
        return data

    original_data = {'list': [1, 2], 'dict': {'key': 'val'}}
    result = nested_operations(original_data)

    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'key': 'val', 'new_key': 'value'})})
    assert original_data == {'list': [1, 2], 'dict': {'key': 'val'}}  # Original unchanged

    # Test with kwargs
    @mutant
    def with_kwargs(a, b=1):
        return a + b

    result = with_kwargs(5, b=3)
    assert result == 8
    assert isinstance(result, int)  # Non-container types remain unchanged


# LLM-generated content at query #39
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['a'].append(4)
        return d

    original_dict = {'a': [1, 2, 3], 'b': 2}
    result_dict = modify_nested(original_dict)
    assert original_dict == {'a': [1, 2, 3], 'b': 2}  # Original unchanged
    assert result_dict == pmap({'a': pvector([1, 2, 3, 4]), 'b': 2})

    # Test with kwargs
    @mutant
    def modify_kwargs(a, b, **kwargs):
        kwargs['c'].append(4)
        return kwargs

    original_kwargs = {'c': [1, 2, 3], 'd': 2}
    result_kwargs = modify_kwargs(1, 2, **original_kwargs)
    assert original_kwargs == {'c': [1, 2, 3], 'd': 2}  # Original unchanged
    assert result_kwargs == pmap({'c': pvector([1, 2, 3, 4]), 'd': 2})

    # Test return value freezing
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with tuple
    @mutant
    def modify_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result_tuple = modify_tuple(original_tuple)
    assert original_tuple == (1, 2, 3)  # Original unchanged
    assert result_tuple == (1, 2, 3, 4)

    # Test with set
    @mutant
    def modify_set(s):
        return s | {4}

    original_set = {1, 2, 3}
    result_set = modify_set(original_set)
    assert original_set == {1, 2, 3}  # Original unchanged
    assert result_set == pset({1, 2, 3, 4})


# LLM-generated content at query #40
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original should be unchanged

    # Test with nested structures
    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1, 'b': [2, 3]}
    result = modify_dict(original_dict, 'c', 4)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': pvector([2, 3]), 'c': 4})
    assert original_dict == {'a': 1, 'b': [2, 3]}

    # Test with kwargs
    @mutant
    def update_dict(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = update_dict(a=1, b=2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

    # Test with mixed arguments
    @mutant
    def complex_operation(lst, d, val):
        lst.append(val)
        d['new_key'] = val
        return lst, d

    original_list = [1, 2]
    original_dict = {'x': 10}
    result_list, result_dict = complex_operation(original_list, original_dict, 3)
    assert isinstance(result_list, PVector)
    assert isinstance(result_dict, PMap)
    assert result_list == pvector([1, 2, 3])
    assert result_dict == pmap({'x': 10, 'new_key': 3})
    assert original_list == [1, 2]
    assert original_dict == {'x': 10}

    # Test return value is frozen
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #41
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['key'][0] = 'modified'
        return data

    original = {'key': ['value']}
    result = modify_nested(original)
    assert original == {'key': ['value']}  # Original unchanged
    assert result == pmap({'key': pvector(['modified'])})

    # Test with multiple arguments
    @mutant
    def combine_and_modify(a, b):
        a['new_key'] = b[0]
        return a

    original_a = {'existing': 1}
    original_b = [2, 3]
    result = combine_and_modify(original_a, original_b)
    assert original_a == {'existing': 1}
    assert original_b == [2, 3]
    assert result == pmap({'existing': 1, 'new_key': 2})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 'changed'
        return kwargs

    original = {'x': 1, 'y': 2}
    result = modify_kwargs(**original)
    assert original == {'x': 1, 'y': 2}
    assert result == pmap({'x': 'changed', 'y': 2})

    # Test return value is frozen
    @mutant
    def return_mutable():
        return {'a': [1, 2]}

    result = return_mutable()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)


# LLM-generated content at query #42
#--------------------------

```python
def test_mutant():
    # Test that mutant decorator freezes input arguments and return value
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    input_list = [1, 2, 3]
    result = add_to_list(input_list, 4)

    # Check that input list is not modified
    assert input_list == [1, 2, 3]

    # Check that result is frozen
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])

    # Test with dict
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    input_dict = {'a': 1}
    result = add_to_dict(input_dict, 'b', 2)

    # Check that input dict is not modified
    assert input_dict == {'a': 1}

    # Check that result is frozen
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    input_data = {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    result = modify_nested(input_data)

    # Check that input is not modified
    assert input_data == {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}

    # Check that result is properly frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'b': 2, 'c': 3})})

    # Test with kwargs
    @mutant
    def process_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    input_kwargs = {'key1': 'value1'}
    result = process_kwargs(**input_kwargs)

    # Check that input is not modified
    assert input_kwargs == {'key1': 'value1'}

    # Check that result is frozen
    assert isinstance(result, PMap)
    assert result == pmap({'key1': 'value1', 'new_key': 'new_value'})


# LLM-generated content at query #43
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    frozen_list = add_to_list(original_list, 4)

    assert isinstance(frozen_list, PVector)
    assert frozen_list == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list should be unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1, 'b': 2}
    frozen_dict = modify_dict(original_dict, 'c', 3)

    assert isinstance(frozen_dict, PMap)
    assert frozen_dict == pmap({'a': 1, 'b': 2, 'c': 3})
    assert original_dict == {'a': 1, 'b': 2}  # Original dict should be unchanged

    @mutant
    def nested_operations(data):
        data['list'].append(4)
        data['set'].add(4)
        return data

    original_data = {'list': [1, 2, 3], 'set': {1, 2, 3}}
    frozen_data = nested_operations(original_data)

    assert isinstance(frozen_data, PMap)
    assert isinstance(frozen_data['list'], PVector)
    assert isinstance(frozen_data['set'], PSet)
    assert frozen_data['list'] == pvector([1, 2, 3, 4])
    assert frozen_data['set'] == pset({1, 2, 3, 4})
    assert original_data == {'list': [1, 2, 3], 'set': {1, 2, 3}}  # Original should be unchanged

    # Test with strict=False
    @mutant
    def non_strict_test(lst):
        return lst

    mixed_list = [1, {'a': 2}, {3, 4}]
    result = non_strict_test(mixed_list)

    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)
    assert result == pvector([1, pmap({'a': 2}), pset({3, 4})])
    assert mixed_list == [1, {'a': 2}, {3, 4}]  # Original should be unchanged


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_mutant():
    # Test basic mutation prevention
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original should be unchanged

    # Test with dict mutation
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original should be unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['nested'].append(1)
        return data

    original_data = {'nested': [1, 2]}
    result = modify_nested(original_data)
    assert isinstance(result['nested'], PVector)
    assert result == pmap({'nested': pvector([1, 2, 1])})
    assert original_data == {'nested': [1, 2]}  # Original should be unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

    # Test return value is frozen
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def test_strict_false(lst):
        return lst

    result = test_strict_false([1, 2, 3])
    assert isinstance(result, PVector)  # Still frozen by default


# LLM-generated content at query #2
#--------------------------

```python
def test_mutant():
    # Test that mutant decorator freezes input arguments and return value
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    frozen_list = add_to_list(original_list, 4)

    # Check that original list is unchanged
    assert original_list == [1, 2, 3]

    # Check that returned list is frozen (pvector)
    assert isinstance(frozen_list, PVector)
    assert frozen_list == pvector([1, 2, 3, 4])

    # Test with dict
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    frozen_dict = add_to_dict(original_dict, 'b', 2)

    # Check that original dict is unchanged
    assert original_dict == {'a': 1}

    # Check that returned dict is frozen (pmap)
    assert isinstance(frozen_dict, PMap)
    assert frozen_dict == pmap({'a': 1, 'b': 2})

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_nested = {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    frozen_nested = modify_nested(original_nested)

    # Check that original nested structure is unchanged
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}

    # Check that returned nested structure is frozen
    assert isinstance(frozen_nested, PMap)
    assert isinstance(frozen_nested['list'], PVector)
    assert isinstance(frozen_nested['dict'], PMap)
    assert frozen_nested == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'b': 2, 'c': 3})})

    # Test with tuple (should remain tuple but with frozen contents)
    @mutant
    def modify_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    frozen_tuple = modify_tuple(original_tuple)

    # Check that original tuple is unchanged
    assert original_tuple == (1, 2, 3)

    # Check that returned tuple has frozen contents
    assert frozen_tuple == (1, 2, 3, 4)

    # Test with set (should remain set)
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    original_set = {1, 2, 3}
    frozen_set = add_to_set(original_set, 4)

    # Check that original set is unchanged
    assert original_set == {1, 2, 3}

    # Check that returned set is frozen (pset)
    assert isinstance(frozen_set, PSet)
    assert frozen_set == pset({1, 2, 3, 4})

    # Test with strict=False
    @mutant
    def no_strict_freeze(lst):
        return lst

    result = no_strict_freeze([1, 2, 3])
    # With strict=False, the list should remain a list (not frozen)
    assert isinstance(result, list)
    assert result == [1, 2, 3]


# LLM-generated content at query #3
#--------------------------

```python
def test_mutant():
    # Test with simple function that doesn't mutate inputs
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])

    # Test with function that would normally mutate inputs
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = append_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original should be unchanged
    assert result == pvector([1, 2, 3, 4])

    # Test with function that modifies dict
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}  # Original should be unchanged
    assert result == pmap({'a': 1, 'b': 2})

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['nested']['value'] = 10
        return data

    original = {'nested': {'value': 5}}
    result = modify_nested(original)
    assert original == {'nested': {'value': 5}}  # Original unchanged
    assert result == pmap({'nested': pmap({'value': 10})})

    # Test with kwargs
    @mutant
    def process_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'a': 1, 'b': 2}
    result = process_kwargs(**original_kwargs)
    assert original_kwargs == {'a': 1, 'b': 2}  # Original unchanged
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

    # Test with mixed args and kwargs
    @mutant
    def mixed_args_and_kwargs(arg1, arg2, **kwargs):
        arg1.append(arg2)
        kwargs['arg1'] = arg1
        return kwargs

    original_arg = [1, 2]
    original_kwargs = {'x': 10}
    result = mixed_args_and_kwargs(original_arg, 3, **original_kwargs)
    assert original_arg == [1, 2]  # Original unchanged
    assert original_kwargs == {'x': 10}  # Original unchanged
    assert result == pmap({'x': 10, 'arg1': pvector([1, 2, 3])})


# LLM-generated content at query #4
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Ensure original list is unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Ensure original dict is unchanged

    @mutant
    def nested_operations(data):
        data['list'].append(4)
        data['set'].add(4)
        return data

    original_data = {'list': [1, 2, 3], 'set': {1, 2, 3}}
    result = nested_operations(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'set': pset({1, 2, 3, 4})})
    assert original_data == {'list': [1, 2, 3], 'set': {1, 2, 3}}  # Ensure original data is unchanged

    @mutant
    def no_mutation(value):
        return value + 1

    assert no_mutation(5) == 6


# LLM-generated content at query #5
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original = [1, 2, 3]
    result = add_to_list(original, 4)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])

    # Test with dict mutation
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result_dict = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}  # Original unchanged
    assert result_dict == pmap({'a': 1, 'b': 2})

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_nested = {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    result_nested = modify_nested(original_nested)
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    assert result_nested == pmap({
        'list': pvector([1, 2, 3, 4]),
        'dict': pmap({'a': 1, 'b': 2, 'c': 3})
    })

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    original_kwargs = {'y': 20}
    result_kwargs = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'y': 20}
    assert result_kwargs == pmap({'y': 20, 'x': 10})

    # Test return value is frozen
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def strict_false_test(lst):
        return lst

    original = pvector([1, 2, 3])
    result = strict_false_test(original)
    assert result == original  # Should not double-freeze


# LLM-generated content at query #6
#--------------------------

```python
def test_mutant():
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original_list = [1, 2, 3]
    result = modify_list(original_list)
    assert original_list == [1, 2, 3]  # Original list should not be modified
    assert result == pvector([1, 2, 3, 4])  # Result should be a frozen pvector with the new element

    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    original_dict = {'a': 1, 'b': 2}
    result = modify_dict(original_dict)
    assert original_dict == {'a': 1, 'b': 2}  # Original dict should not be modified
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})  # Result should be a frozen pmap with the new key

    @mutant
    def modify_set(s):
        s.add(3)
        return s

    original_set = {1, 2}
    result = modify_set(original_set)
    assert original_set == {1, 2}  # Original set should not be modified
    assert result == pset({1, 2, 3})  # Result should be a frozen pset with the new element

    @mutant
    def modify_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple)
    assert original_tuple == (1, 2, 3)  # Original tuple should not be modified
    assert result == (1, 2, 3, 4)  # Result should be a new tuple with the new element

    @mutant
    def modify_nested(lst):
        lst[0]['a'] = 10
        return lst

    original_nested = [{'a': 1}, 2, 3]
    result = modify_nested(original_nested)
    assert original_nested == [{'a': 1}, 2, 3]  # Original nested structure should not be modified
    assert result == pvector([pmap({'a': 10}), 2, 3])  # Result should be a frozen structure with the modified value


# LLM-generated content at query #7
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen

    # Test with multiple arguments
    @mutant
    def modify_dict_and_list(d, lst):
        d['new_key'] = 'value'
        lst.append(5)
        return d, lst

    original_dict = {'a': 1}
    original_list = [1, 2]
    dict_result, list_result = modify_dict_and_list(original_dict, original_list)
    assert original_dict == {'a': 1}  # Original dict unchanged
    assert original_list == [1, 2]  # Original list unchanged
    assert dict_result == pmap({'a': 1, 'new_key': 'value'})
    assert list_result == pvector([1, 2, 5])

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(3)
        data['dict']['new'] = 'value'
        return data

    original = {'list': [1, 2], 'dict': {'a': 1}}
    result = modify_nested(original)
    assert original == {'list': [1, 2], 'dict': {'a': 1}}  # Original unchanged
    assert result == pmap({'list': pvector([1, 2, 3]), 'dict': pmap({'a': 1, 'new': 'value'})})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new'] = 'value'
        return kwargs

    original_kwargs = {'a': 1}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'a': 1}  # Original kwargs unchanged
    assert result == pmap({'a': 1, 'new': 'value'})

    # Test that return value is frozen even if function returns non-frozen
    @mutant
    def return_non_frozen():
        return [1, 2, 3]

    result = return_non_frozen()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with immutable input
    @mutant
    def immutable_input(x):
        return x + 1

    assert immutable_input(5) == 6


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    @mutant
    def nested_mutation(data):
        data['list'].append(1)
        data['dict']['new_key'] = 'new_value'
        return data

    original_data = {'list': [1, 2], 'dict': {'key': 'value'}}
    result = nested_mutation(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'key': 'value', 'new_key': 'new_value'})})
    assert original_data == {'list': [1, 2], 'dict': {'key': 'value'}}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_with_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'a': 1}
    result = modify_with_kwargs(**original_kwargs)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert original_kwargs == {'a': 1}  # Original kwargs unchanged


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original should be unchanged
    assert result == pvector([1, 2, 3, 4])  # Result should be frozen

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 'modified'
        return d

    original_dict = {'nested': {'value': 'original'}}
    result = modify_nested(original_dict)
    assert original_dict == {'nested': {'value': 'original'}}  # Original unchanged
    assert result == pmap({'nested': pmap({'value': 'modified'})})  # Result frozen

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['key'] = 'new_value'
        return kwargs

    original_kwargs = {'key': 'old_value', 'other': 'unchanged'}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'key': 'old_value', 'other': 'unchanged'}  # Original unchanged
    assert result == pmap({'key': 'new_value', 'other': 'unchanged'})  # Result frozen

    # Test with multiple arguments
    @mutant
    def combine_and_modify(lst1, lst2):
        combined = lst1 + lst2
        combined.append('extra')
        return combined

    list1 = [1, 2]
    list2 = [3, 4]
    result = combine_and_modify(list1, list2)
    assert list1 == [1, 2]  # Original unchanged
    assert list2 == [3, 4]  # Original unchanged
    assert result == pvector([1, 2, 3, 4, 'extra'])  # Result frozen

    # Test return value is frozen
    @mutant
    def return_dict():
        return {'a': [1, 2, 3]}

    result = return_dict()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original should be unchanged

    # Test with nested structures
    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        d['nested'].append(10)
        return d

    original_dict = {'a': 1, 'nested': [1, 2]}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'nested': pvector([1, 2, 10])})
    assert original_dict == {'a': 1, 'nested': [1, 2]}

    # Test with kwargs
    @mutant
    def update_set(s, item):
        s.add(item)
        return s

    original_set = {1, 2, 3}
    result = update_set(s=original_set, item=4)
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})
    assert original_set == {1, 2, 3}

    # Test with strict=False
    @mutant
    def no_strict_mutation(lst):
        lst.append(5)
        return lst

    original_list = PVector([1, 2])
    result = no_strict_mutation(original_list)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 5])
    assert original_list == pvector([1, 2])

    # Test return value freezing
    @mutant
    def return_new_list():
        return [1, 2, 3]

    result = return_new_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #11
#--------------------------

```python
def test_mutant():
    # Test that mutant decorator freezes input arguments and return value
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    # Test with list
    input_list = [1, 2, 3]
    result = add_to_list(input_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original list unchanged

    # Test with dict
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    input_dict = {'a': 1}
    result = add_to_dict(input_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert input_dict == {'a': 1}  # Original dict unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(1)
        data['dict']['new_key'] = 'new_value'
        return data

    input_data = {'list': [1, 2], 'dict': {'key': 'value'}}
    result = modify_nested(input_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'key': 'value', 'new_key': 'new_value'})})
    assert input_data == {'list': [1, 2], 'dict': {'key': 'value'}}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

    # Test with non-container types (should remain unchanged)
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("string") == "string"


# LLM-generated content at query #12
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original unchanged

    # Test with dict mutation
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['lst'].append(4)
        data['d']['x'] = 10
        return data

    original = {'lst': [1, 2, 3], 'd': {'x': 5}}
    result = modify_nested(original)
    assert isinstance(result['lst'], PVector)
    assert isinstance(result['d'], PMap)
    assert result == pmap({'lst': pvector([1, 2, 3, 4]), 'd': pmap({'x': 10})})
    assert original == {'lst': [1, 2, 3], 'd': {'x': 5}}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_with_kwargs(**kwargs):
        kwargs['a'] = kwargs.get('a', 0) + 1
        return kwargs

    result = modify_with_kwargs(a=1, b=2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 2, 'b': 2})

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with tuple (should remain tuple)
    @mutant
    def modify_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)
    assert original_tuple == (1, 2, 3)


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        return lst + [value]

    # Test with list
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    # Test with dict
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = add_to_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        return data

    original_data = {'list': [1, 2, 3]}
    result = modify_nested(original_data)
    assert isinstance(result['list'], PVector)
    assert result == pmap({'list': pvector([1, 2, 3, 4])})
    assert original_data == {'list': [1, 2, 3]}  # Original data unchanged

    # Test with kwargs
    @mutant
    def combine(a, b, c=5):
        return a + b + c

    result = combine(1, 2, c=3)
    assert result == 6
    assert isinstance(result, int)  # Non-container return value remains unchanged

    # Test with tuple
    @mutant
    def modify_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)
    assert original_tuple == (1, 2, 3)  # Original tuple unchanged


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])  # Frozen result

    # Test with nested structures
    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': [1, 2]}
    result = modify_dict(original_dict, 'a', [3, 4])
    assert original_dict == {'a': [1, 2]}  # Original unchanged
    assert result == pmap({'a': pvector([3, 4])})  # Frozen result

    # Test with kwargs
    @mutant
    def update_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'old_key': 'old_value'}
    result = update_kwargs(**original_kwargs)
    assert original_kwargs == {'old_key': 'old_value'}  # Original unchanged
    assert result == pmap({'old_key': 'old_value', 'new_key': 'new_value'})  # Frozen result

    # Test with multiple args
    @mutant
    def combine(lst1, lst2):
        lst1.extend(lst2)
        return lst1

    list1 = [1, 2]
    list2 = [3, 4]
    result = combine(list1, list2)
    assert list1 == [1, 2]  # Original unchanged
    assert list2 == [3, 4]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])  # Frozen result

    # Test with non-mutable operations
    @mutant
    def no_mutation(x):
        return x + 1

    assert no_mutation(5) == 6  # Simple case works


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original = [1, 2, 3]
    result = add_to_list(original, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original == [1, 2, 3]  # Original unchanged

    # Test with dict
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result_dict = update_dict(original_dict, 'b', 2)
    assert isinstance(result_dict, PMap)
    assert result_dict == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    nested = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result_nested = modify_nested(nested)
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['list'], PVector)
    assert isinstance(result_nested['dict'], PMap)
    assert result_nested == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert nested == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original unchanged

    # Test with kwargs
    @mutant
    def process_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'a': 1, 'b': 2}
    result_kwargs = process_kwargs(**original_kwargs)
    assert isinstance(result_kwargs, PMap)
    assert result_kwargs == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert original_kwargs == {'a': 1, 'b': 2}  # Original unchanged

    # Test with tuple (should remain tuple)
    @mutant
    def process_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result_tuple = process_tuple(original_tuple)
    assert isinstance(result_tuple, tuple)
    assert result_tuple == (1, 2, 3, 4)
    assert original_tuple == (1, 2, 3)  # Original unchanged

    # Test with set (should become pset)
    @mutant
    def process_set(s):
        return s | {4}

    original_set = {1, 2, 3}
    result_set = process_set(original_set)
    assert isinstance(result_set, PSet)
    assert result_set == pset({1, 2, 3, 4})
    assert original_set == {1, 2, 3}  # Original unchanged

    # Test with non-container types (should remain unchanged)
    @mutant
    def process_primitive(x):
        return x + 1

    assert process_primitive(5) == 6
    assert process_primitive("hello") == "hello1"


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])  # Result is frozen

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['a'].append(1)
        data['b']['c'] = 4
        return data

    original = {'a': [1, 2], 'b': {'c': 3}}
    result = modify_nested(original)
    assert original == {'a': [1, 2], 'b': {'c': 3}}  # Original unchanged
    assert result == pmap({'a': pvector([1, 2, 1]), 'b': pmap({'c': 4})})  # Result is frozen

    # Test with kwargs
    @mutant
    def modify_kwargs(a, b, **kwargs):
        a[0] = 99
        kwargs['x']['y'] = 88
        return {'a': a, 'b': b, 'kwargs': kwargs}

    original_a = [1, 2]
    original_b = {'c': 3}
    original_kwargs = {'x': {'y': 7}}
    result = modify_kwargs(original_a, original_b, **original_kwargs)
    assert original_a == [1, 2]  # Original unchanged
    assert original_b == {'c': 3}  # Original unchanged
    assert original_kwargs == {'x': {'y': 7}}  # Original unchanged
    assert result == pmap({
        'a': pvector([99, 2]),
        'b': pmap({'c': 3}),
        'kwargs': pmap({'x': pmap({'y': 88})})
    })  # Result is frozen

    # Test return value is frozen
    @mutant
    def return_new():
        return {'a': [1, 2, 3]}

    result = return_new()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)

    # Test strict mode
    @mutant
    def test_strict(data):
        return data

    # Should handle already frozen data
    frozen_data = pmap({'a': pvector([1, 2])})
    result = test_strict(frozen_data)
    assert result == frozen_data


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original should be unchanged
    assert result == pvector([1, 2, 3, 4])  # Result should be frozen

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 'modified'
        return d

    original_dict = {'nested': {'value': 'original'}}
    result = modify_nested(original_dict)
    assert original_dict == {'nested': {'value': 'original'}}  # Original unchanged
    assert result == pmap({'nested': pmap({'value': 'modified'})})  # Result frozen

    # Test with multiple arguments
    @mutant
    def combine_and_modify(lst1, lst2):
        combined = lst1 + lst2
        combined.append('new')
        return combined

    list1 = [1, 2]
    list2 = [3, 4]
    result = combine_and_modify(list1, list2)
    assert list1 == [1, 2] and list2 == [3, 4]  # Originals unchanged
    assert result == pvector([1, 2, 3, 4, 'new'])  # Result frozen

    # Test with keyword arguments
    @mutant
    def modify_with_kwargs(data, **updates):
        data.update(updates)
        return data

    original = {'a': 1}
    result = modify_with_kwargs(original, b=2, c=3)
    assert original == {'a': 1}  # Original unchanged
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})  # Result frozen

    # Test return value is frozen
    @mutant
    def return_mutable():
        return {'key': [1, 2, 3]}

    result = return_mutable()
    assert isinstance(result, pmap)
    assert isinstance(result['key'], pvector)

    # Test with tuple (should remain tuple)
    @mutant
    def modify_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple)
    assert original_tuple == (1, 2, 3)  # Original unchanged
    assert result == (1, 2, 3, 4)  # Tuples remain tuples

    # Test with set (should become pset)
    @mutant
    def modify_set(s):
        return s | {4, 5}

    original_set = {1, 2, 3}
    result = modify_set(original_set)
    assert original_set == {1, 2, 3}  # Original unchanged
    assert result == pset({1, 2, 3, 4, 5})  # Result is pset


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)

    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)

    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = nested_operation(original_data)

    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original unchanged

    # Test with strict=False
    @mutant
    def non_strict_operation(lst):
        return lst + [4]

    result = non_strict_operation([1, 2, 3])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['a']['b'] = 10
        return d

    original_dict = {'a': {'b': 5}}
    result = modify_nested(original_dict)
    assert original_dict == {'a': {'b': 5}}  # Original unchanged
    assert result == pmap({'a': pmap({'b': 10})})

    # Test with multiple arguments
    @mutant
    def combine_dicts(d1, d2):
        d1.update(d2)
        return d1

    dict1 = {'x': 1}
    dict2 = {'y': 2}
    result = combine_dicts(dict1, dict2)
    assert dict1 == {'x': 1}  # Original unchanged
    assert dict2 == {'y': 2}  # Original unchanged
    assert result == pmap({'x': 1, 'y': 2})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'a': 1}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'a': 1}  # Original unchanged
    assert result == pmap({'a': 1, 'new_key': 'new_value'})

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def test_strict_false(lst):
        return lst

    result = test_strict_false([1, 2, 3])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == [1, 2, 3, 4]  # Return value is frozen

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['a'].append(1)
        data['b']['c'] = 2
        return data

    original = {'a': [1, 2], 'b': {'c': 3}}
    result = modify_nested(original)
    assert original == {'a': [1, 2], 'b': {'c': 3}}  # Original unchanged
    assert result['a'] == [1, 2, 1]  # Nested list modified
    assert result['b']['c'] == 2  # Nested dict modified

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'].append(1)
        return kwargs

    original = {'x': [1, 2]}
    result = modify_kwargs(**original)
    assert original == {'x': [1, 2]}  # Original unchanged
    assert result['x'] == [1, 2, 1]  # Kwargs modified

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)  # Return value is frozen

    # Test with strict=False
    @mutant
    def modify_non_strict(lst):
        return lst

    original = [1, 2, 3]
    result = modify_non_strict(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == [1, 2, 3]  # Return value unchanged


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(1)
        data['dict']['new_key'] = 'new_value'
        return data

    original_data = {'list': [1, 2], 'dict': {'key': 'value'}}
    result = nested_operation(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'key': 'value', 'new_key': 'new_value'})})
    assert original_data == {'list': [1, 2], 'dict': {'key': 'value'}}  # Original data unchanged

    # Test with strict=False
    @mutant
    def non_strict_operation(lst):
        return lst

    result = non_strict_operation([1, 2, 3])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original should be unchanged
    assert result == pvector([1, 2, 3, 4])  # Return value should be frozen

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['a']['b'] = 2
        return d

    original_dict = {'a': {'b': 1}}
    result = modify_nested(original_dict)
    assert original_dict == {'a': {'b': 1}}  # Original should be unchanged
    assert result == pmap({'a': pmap({'b': 2})})  # Return value should be frozen

    # Test with multiple arguments
    @mutant
    def combine_dicts(d1, d2):
        d1.update(d2)
        return d1

    dict1 = {'a': 1}
    dict2 = {'b': 2}
    result = combine_dicts(dict1, dict2)
    assert dict1 == {'a': 1}  # Original should be unchanged
    assert dict2 == {'b': 2}  # Original should be unchanged
    assert result == pmap({'a': 1, 'b': 2})  # Return value should be frozen

    # Test with keyword arguments
    @mutant
    def modify_with_kwargs(lst, item=None):
        if item is not None:
            lst.append(item)
        return lst

    original_list = [1, 2]
    result = modify_with_kwargs(original_list, item=3)
    assert original_list == [1, 2]  # Original should be unchanged
    assert result == pvector([1, 2, 3])  # Return value should be frozen

    # Test that non-container types are handled correctly
    @mutant
    def simple_function(x, y):
        return x + y

    assert simple_function(1, 2) == 3  # Should work with non-container types


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Ensure original is unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Ensure original is unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(1)
        data['dict']['new_key'] = 'value'
        return data

    original_data = {'list': [1, 2], 'dict': {'key': 'val'}}
    result = nested_operation(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'key': 'val', 'new_key': 'value'})})
    assert original_data == {'list': [1, 2], 'dict': {'key': 'val'}}  # Ensure original is unchanged

    # Test with kwargs
    @mutant
    def kwargs_test(a, b=1):
        return {'a': a + 1, 'b': b + 1}

    result = kwargs_test(10, b=20)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 11, 'b': 21})


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)

    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)

    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}

    @mutant
    def nested_operations(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = nested_operations(original_data)

    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1}}

    @mutant
    def no_mutation(value):
        return value + 1

    assert no_mutation(5) == 6


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)

    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)

    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(4)
        data['dict']['new_key'] = 'new_value'
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'key': 'value'}}
    result = nested_operation(original_data)

    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'key': 'value', 'new_key': 'new_value'})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'key': 'value'}}  # Original data unchanged

    # Test with kwargs
    @mutant
    def modify_with_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'key': 'value'}
    result = modify_with_kwargs(**original_kwargs)

    assert isinstance(result, PMap)
    assert result == pmap({'key': 'value', 'new_key': 'new_value'})
    assert original_kwargs == {'key': 'value'}  # Original kwargs unchanged


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Ensure original list is unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Ensure original dict is unchanged

    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})
    assert original_set == {1, 2, 3}  # Ensure original set is unchanged

    @mutant
    def modify_tuple(t, index, value):
        lst = list(t)
        lst[index] = value
        return tuple(lst)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple, 1, 4)
    assert isinstance(result, tuple)
    assert result == (1, 4, 3)
    assert original_tuple == (1, 2, 3)  # Ensure original tuple is unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(4)
        data['dict']['new_key'] = 'new_value'
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'key': 'value'}}
    result = nested_operation(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'key': 'value', 'new_key': 'new_value'})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'key': 'value'}}  # Ensure original data is unchanged


# LLM-generated content at query #27
#--------------------------

```python
def test_freeze():
    # Test freezing a list with nested dict
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

    # Test freezing a tuple with nested list
    assert freeze((1, [])) == (1, pvector([]))

    # Test freezing a set
    assert freeze(set([1, 2])) == pset([1, 2])

    # Test freezing a dict
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

    # Test freezing a defaultdict
    dd = collections.defaultdict(int, {'a': 1})
    assert freeze(dd) == pmap({'a': 1})

    # Test freezing with strict=False
    assert freeze([1, {'a': [3, 4]}], strict=False) == pvector([1, {'a': [3, 4]}])

    # Test freezing a pvector (should remain unchanged when strict=True)
    pv = pvector([1, 2, 3])
    assert freeze(pv) == pv

    # Test freezing a pmap (should remain unchanged when strict=True)
    pm = pmap({'a': 1})
    assert freeze(pm) == pm

    # Test freezing a pset (should remain unchanged when strict=True)
    ps = pset([1, 2])
    assert freeze(ps) == ps

    # Test freezing a simple value (should remain unchanged)
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 100
        return d

    original_dict = {'nested': {'value': 50}}
    result = modify_nested(original_dict)
    assert original_dict == {'nested': {'value': 50}}  # Original unchanged
    assert result == pmap({'nested': pmap({'value': 100})})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 999
        return kwargs

    original_kwargs = {'x': 1, 'y': 2}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'x': 1, 'y': 2}  # Original unchanged
    assert result == pmap({'x': 999, 'y': 2})

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def no_strict(lst):
        return lst

    result = no_strict([1, 2, 3], strict=False)
    assert result == [1, 2, 3]  # Not frozen when strict=False


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Ensure original is unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Ensure original is unchanged

    @mutant
    def process_tuple(t, value):
        return t + (value,)

    original_tuple = (1, 2)
    result = process_tuple(original_tuple, 3)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)
    assert original_tuple == (1, 2)  # Ensure original is unchanged

    @mutant
    def process_set(s, value):
        s.add(value)
        return s

    original_set = {1, 2}
    result = process_set(original_set, 3)
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3})
    assert original_set == {1, 2}  # Ensure original is unchanged

    @mutant
    def nested_operations(data):
        data['list'].append(4)
        data['dict']['new_key'] = 'new_value'
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'key': 'value'}}
    result = nested_operations(original_data)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result['list'] == pvector([1, 2, 3, 4])
    assert result['dict'] == pmap({'key': 'value', 'new_key': 'new_value'})
    assert original_data == {'list': [1, 2, 3], 'dict': {'key': 'value'}}  # Ensure original is unchanged


# LLM-generated content at query #30
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original = [1, 2, 3]
    result = add_to_list(original, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original == [1, 2, 3]  # Original unchanged

    # Test with nested structures
    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        d['nested'].append(10)
        return d

    original_dict = {'a': 1, 'nested': [1, 2]}
    result_dict = modify_dict(original_dict, 'b', 2)
    assert isinstance(result_dict, PMap)
    assert result_dict == pmap({'a': 1, 'b': 2, 'nested': pvector([1, 2, 10])})
    assert original_dict == {'a': 1, 'nested': [1, 2]}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result_kwargs = modify_kwargs(a=1, b=2)
    assert isinstance(result_kwargs, PMap)
    assert result_kwargs == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def strict_false_test(lst):
        return lst

    result_strict = strict_false_test([1, 2, 3])
    assert isinstance(result_strict, PVector)
    assert result_strict == pvector([1, 2, 3])


# LLM-generated content at query #31
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)

    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Ensure original is unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)

    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Ensure original is unchanged

    @mutant
    def modify_set(s, value):
        s.add(value)
        return s

    original_set = {1, 2}
    result = modify_set(original_set, 3)

    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3})
    assert original_set == {1, 2}  # Ensure original is unchanged

    @mutant
    def modify_tuple(t, index, value):
        lst = list(t)
        lst[index] = value
        return tuple(lst)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple, 1, 4)

    assert isinstance(result, tuple)
    assert result == (1, 4, 3)
    assert original_tuple == (1, 2, 3)  # Ensure original is unchanged


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant():
    # Test basic functionality with simple arguments
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3]) == pvector([1, 2, 3])

    # Test with mutable arguments
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original list should be unchanged

    # Test with keyword arguments
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    input_dict = {'a': 1}
    result = update_dict(input_dict, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})
    assert input_dict == {'a': 1}  # Original dict should be unchanged

    # Test with nested structures
    @mutant
    def nested_operation(data):
        data['list'].append(1)
        data['dict']['new_key'] = 'new_value'
        return data

    input_data = {'list': [1, 2], 'dict': {'a': 1}}
    result = nested_operation(input_data)
    expected_result = pmap({
        'list': pvector([1, 2, 1]),
        'dict': pmap({'a': 1, 'new_key': 'new_value'})
    })
    assert result == expected_result
    assert input_data == {'list': [1, 2], 'dict': {'a': 1}}  # Original should be unchanged

    # Test with tuple arguments
    @mutant
    def modify_tuple(t):
        return t + (4,)

    input_tuple = (1, 2, 3)
    result = modify_tuple(input_tuple)
    assert result == (1, 2, 3, 4)
    assert input_tuple == (1, 2, 3)  # Original tuple should be unchanged

    # Test with set arguments
    @mutant
    def modify_set(s):
        return s | {4}

    input_set = {1, 2, 3}
    result = modify_set(input_set)
    assert result == pset({1, 2, 3, 4})
    assert input_set == {1, 2, 3}  # Original set should be unchanged


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant():
    @mutant
    def test_func(a, b, c=None):
        a.append(1)
        b['x'] = 2
        if c is not None:
            c.add(3)
        return {'a': a, 'b': b, 'c': c}

    # Test with list, dict, and set
    a = [1, 2]
    b = {'y': 1}
    c = {4, 5}
    result = test_func(a, b, c)

    # Check original objects are unchanged
    assert a == [1, 2]
    assert b == {'y': 1}
    assert c == {4, 5}

    # Check result is frozen
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    assert isinstance(result['c'], PSet)

    # Check result values
    assert result['a'] == pvector([1, 2, 1])
    assert result['b'] == pmap({'y': 1, 'x': 2})
    assert result['c'] == pset({4, 5, 3})

    # Test with None
    result_none = test_func([1], {'y': 1})
    assert result_none['c'] is None

    # Test with nested structures
    @mutant
    def nested_func(data):
        data['list'].append(1)
        data['dict']['x'] = 2
        return data

    original = {'list': [1, 2], 'dict': {'y': 1}}
    result_nested = nested_func(original)

    assert original == {'list': [1, 2], 'dict': {'y': 1}}
    assert result_nested['list'] == pvector([1, 2, 1])
    assert result_nested['dict'] == pmap({'y': 1, 'x': 2})


# LLM-generated content at query #34
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    input_list = [1, 2, 3]
    result = add_to_list(input_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original list unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    input_dict = {'a': 1}
    result = modify_dict(input_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert input_dict == {'a': 1}  # Original dict unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(1)
        data['dict']['new_key'] = 'value'
        return data

    input_data = {'list': [1, 2], 'dict': {'a': 1}}
    result = nested_operation(input_data)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'a': 1, 'new_key': 'value'})})
    assert input_data == {'list': [1, 2], 'dict': {'a': 1}}  # Original unchanged

    # Test with strict=False
    @mutant
    def strict_false_test(lst):
        return lst

    result = strict_false_test([1, 2, 3])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #35
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original = [1, 2, 3]
    result = add_to_list(original, 4)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])

    # Test with nested structures
    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1, 'b': [2, 3]}
    result_dict = modify_dict(original_dict, 'c', 4)
    assert original_dict == {'a': 1, 'b': [2, 3]}  # Original unchanged
    assert result_dict == pmap({'a': 1, 'b': pvector([2, 3]), 'c': 4})

    # Test with kwargs
    @mutant
    def update_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'key1': 'value1'}
    result_kwargs = update_kwargs(**original_kwargs)
    assert original_kwargs == {'key1': 'value1'}  # Original unchanged
    assert result_kwargs == pmap({'key1': 'value1', 'new_key': 'new_value'})

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def non_strict_test(lst):
        return lst

    mixed_list = [1, pvector([2, 3])]
    result = non_strict_test(mixed_list)
    assert result == pvector([1, pvector([2, 3])])


# LLM-generated content at query #36
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original list should not be modified
    assert result == pvector([1, 2, 3, 4])  # Result should be frozen

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}  # Original dict should not be modified
    assert result == pmap({'a': 1, 'b': 2})  # Result should be frozen

    @mutant
    def nested_modification(data):
        data['list'].append(1)
        data['dict']['new_key'] = 'new_value'
        return data

    original_data = {'list': [1, 2], 'dict': {'key': 'value'}}
    result = nested_modification(original_data)
    assert original_data == {'list': [1, 2], 'dict': {'key': 'value'}}  # Original should not be modified
    assert result == pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'key': 'value', 'new_key': 'new_value'})})

    @mutant
    def no_mutation(x):
        return x + 1

    assert no_mutation(5) == 6  # Non-mutable operations should work fine


# LLM-generated content at query #37
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    @mutant
    def nested_operations(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = nested_operations(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_with_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    result = modify_with_kwargs(a=1, b=2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'x': 10})


# LLM-generated content at query #38
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    # Test with list
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original list unchanged

    # Test with dict
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = add_to_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['nested_dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'nested_dict': {'a': 1}}
    result = modify_nested(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['nested_dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'nested_dict': pmap({'a': 1, 'c': 3})})
    assert original_data == {'list': [1, 2, 3], 'nested_dict': {'a': 1}}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = kwargs.get('x', 0) + 1
        return kwargs

    result = modify_kwargs(x=5, y=10)
    assert isinstance(result, PMap)
    assert result == pmap({'x': 6, 'y': 10})

    # Test with tuple (should remain tuple)
    @mutant
    def modify_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)
    assert original_tuple == (1, 2, 3)  # Original unchanged

    # Test with set (should become pset)
    @mutant
    def modify_set(s):
        return s | {4}

    original_set = {1, 2, 3}
    result = modify_set(original_set)
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})
    assert original_set == {1, 2, 3}  # Original unchanged


# LLM-generated content at query #39
#--------------------------

```python
def test_mutant():
    # Test basic functionality with simple arguments
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1, 2], [3, 4]) == pvector([1, 2, 3, 4])

    # Test with mutable arguments
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original list should be unchanged

    # Test with keyword arguments
    @mutant
    def combine(a, b, c=10):
        return a + b + c

    assert combine(1, 2, c=3) == 6

    # Test with nested structures
    @mutant
    def nested_operation(data):
        data['values'].append(4)
        return data

    input_data = {'values': [1, 2, 3]}
    result = nested_operation(input_data)
    assert result == pmap({'values': pvector([1, 2, 3, 4])})
    assert input_data == {'values': [1, 2, 3]}  # Original should be unchanged

    # Test with tuple arguments
    @mutant
    def process_tuple(t):
        return t + (4,)

    input_tuple = (1, 2, 3)
    result = process_tuple(input_tuple)
    assert result == (1, 2, 3, 4)
    assert input_tuple == (1, 2, 3)  # Original should be unchanged

    # Test with set arguments
    @mutant
    def process_set(s):
        return s | {4}

    input_set = {1, 2, 3}
    result = process_set(input_set)
    assert result == pset({1, 2, 3, 4})
    assert input_set == {1, 2, 3}  # Original should be unchanged

    # Test with strict=False
    @mutant
    def non_strict_operation(data):
        return data

    # This should not raise an error even with non-pyrsistent types
    result = non_strict_operation([1, 2, 3])
    assert isinstance(result, pvector)


# LLM-generated content at query #40
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])  # Result is frozen

    # Test with dictionary
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    original_dict = {'a': 1}
    result_dict = modify_dict(original_dict)
    assert original_dict == {'a': 1}  # Original unchanged
    assert result_dict == pmap({'a': 1, 'new_key': 'new_value'})  # Result is frozen

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(5)
        data['nested']['value'] = 10
        return data

    original_nested = {'list': [1, 2], 'nested': {'value': 5}}
    result_nested = modify_nested(original_nested)
    assert original_nested == {'list': [1, 2], 'nested': {'value': 5}}  # Original unchanged
    assert result_nested == pmap({
        'list': pvector([1, 2, 5]),
        'nested': pmap({'value': 10})
    })  # Result is frozen

    # Test with kwargs
    @mutant
    def modify_kwargs(a, b, **kwargs):
        kwargs['c'] = 3
        return {'a': a, 'b': b, 'kwargs': kwargs}

    result_kwargs = modify_kwargs(1, b=2, d=4)
    assert result_kwargs == pmap({
        'a': 1,
        'b': 2,
        'kwargs': pmap({'d': 4, 'c': 3})
    })

    # Test with non-container types (should remain unchanged)
    @mutant
    def no_change(x):
        return x

    assert no_change(5) == 5
    assert no_change("string") == "string"


# LLM-generated content at query #41
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original should be unchanged
    assert result == pvector([1, 2, 3, 4])  # Result should be frozen

    # Test with dict mutation
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}  # Original should be unchanged
    assert result == pmap({'a': 1, 'b': 2})  # Result should be frozen

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original = {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    result = modify_nested(original)
    assert original == {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}  # Original unchanged
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'b': 2, 'c': 3})})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    original_kwargs = {'y': 20}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'y': 20}  # Original unchanged
    assert result == pmap({'y': 20, 'x': 10})

    # Test return value freezing
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert result == pvector([1, 2, 3])
    assert isinstance(result, PVector)


# LLM-generated content at query #42
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original should be unchanged

    # Test with dictionary
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original should be unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = modify_nested(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original should be unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'x': 10})

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with tuple (should remain tuple)
    @mutant
    def process_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result = process_tuple(original_tuple)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)
    assert original_tuple == (1, 2, 3)  # Original should be unchanged


# LLM-generated content at query #43
#--------------------------

```python
def test_mutant():
    @mutant
    def test_func(a, b, c=None):
        a.append(4)
        b['d'] = 4
        if c is not None:
            c.add(4)
        return {'result': [a, b, c]}

    # Test with mutable inputs
    input_list = [1, 2, 3]
    input_dict = {'a': 1, 'b': 2}
    input_set = {1, 2, 3}

    result = test_func(input_list, input_dict, input_set)

    # Check that original inputs are unchanged
    assert input_list == [1, 2, 3]
    assert input_dict == {'a': 1, 'b': 2}
    assert input_set == {1, 2, 3}

    # Check that result is frozen
    assert isinstance(result['result'][0], PVector)
    assert isinstance(result['result'][1], PMap)
    assert isinstance(result['result'][2], PSet)

    # Check that result contains the modified values
    assert result['result'][0] == pvector([1, 2, 3, 4])
    assert result['result'][1] == pmap({'a': 1, 'b': 2, 'd': 4})
    assert result['result'][2] == pset({1, 2, 3, 4})

    # Test with None as optional argument
    result_none = test_func(input_list, input_dict)
    assert result_none['result'][2] is None

    # Test that the function itself is not modified
    assert test_func.__name__ == 'test_func'
    assert test_func.__doc__ is None


# LLM-generated content at query #44
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    # Test with list
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original list unchanged
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen

    # Test with dict
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}  # Original dict unchanged
    assert result == pmap({'a': 1, 'b': 2})  # Return value is frozen

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['lst'].append(4)
        data['d']['c'] = 3
        return data

    original_nested = {'lst': [1, 2, 3], 'd': {'a': 1}}
    result = modify_nested(original_nested)
    assert original_nested == {'lst': [1, 2, 3], 'd': {'a': 1}}  # Original unchanged
    assert result == pmap({'lst': pvector([1, 2, 3, 4]), 'd': pmap({'a': 1, 'c': 3})})  # Return value is frozen

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    original_kwargs = {'y': 20}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'y': 20}  # Original kwargs unchanged
    assert result == pmap({'y': 20, 'x': 10})  # Return value is frozen

    # Test with non-mutable arguments (should still work)
    @mutant
    def process_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result = process_tuple(original_tuple)
    assert original_tuple == (1, 2, 3)  # Original tuple unchanged
    assert result == (1, 2, 3, 4)  # Return value is frozen (though tuple is immutable)


# LLM-generated content at query #45
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)

    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)

    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}

    @mutant
    def nested_operations(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = nested_operations(original_data)

    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1}}

    @mutant
    def no_mutation(x):
        return x + 1

    assert no_mutation(5) == 6


# LLM-generated content at query #46
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Ensure original is unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = modify_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Ensure original is unchanged

    @mutant
    def modify_set(s, item):
        s.add(item)
        return s

    original_set = {1, 2}
    result = modify_set(original_set, 3)
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3})
    assert original_set == {1, 2}  # Ensure original is unchanged

    @mutant
    def modify_tuple(t, item):
        return t + (item,)

    original_tuple = (1, 2)
    result = modify_tuple(original_tuple, 3)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)
    assert original_tuple == (1, 2)  # Ensure original is unchanged

    # Test with nested structures
    @mutant
    def nested_modify(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_nested = {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    result = nested_modify(original_nested)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'b': 2, 'c': 3})})
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}  # Ensure original is unchanged


# LLM-generated content at query #47
#--------------------------

```python
def test_mutant():
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    input_list = [1, 2, 3]
    result = add_to_list(input_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original list unchanged

    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    input_dict = {'a': 1}
    result = modify_dict(input_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert input_dict == {'a': 1}  # Original dict unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    input_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = nested_operation(input_data)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result['list'] == pvector([1, 2, 3, 4])
    assert result['dict'] == pmap({'a': 1, 'c': 3})
    assert input_data == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original unchanged


