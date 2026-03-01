####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_freeze():
    # Test freezing a list
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

    # Test freezing a dict
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

    # Test freezing a defaultdict
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

    # Test freezing a set
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

    # Test freezing a tuple
    assert freeze((1, 2, 3)) == (1, 2, 3)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

    # Test freezing with nested structures
    nested = {'a': [1, {'b': 2}], 'c': {3, 4}}
    expected = pmap({'a': pvector([1, pmap({'b': 2})]), 'c': pset({3, 4})})
    assert freeze(nested) == expected

    # Test freezing with strict=False
    assert freeze([1, [2, 3]], strict=False) == pvector([1, [2, 3]])
    assert freeze({'a': [1, 2]}, strict=False) == pmap({'a': [1, 2]})

    # Test freezing pyrsistent types
    assert freeze(pvector([1, 2, 3])) == pvector([1, 2, 3])
    assert freeze(pmap({'a': 1})) == pmap({'a': 1})
    assert freeze(pset({1, 2})) == pset({1, 2})

    # Test freezing non-container types
    assert freeze(1) == 1
    assert freeze("hello") == "hello"
    assert freeze(None) is None


# LLM-generated content at query #2
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
    def modify_set(s, value):
        s.add(value)
        return s

    original_set = {1, 2}
    result = modify_set(original_set, 3)
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3})
    assert original_set == {1, 2}  # Original set unchanged

    @mutant
    def modify_tuple(t, value):
        return t + (value,)

    original_tuple = (1, 2)
    result = modify_tuple(original_tuple, 3)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)
    assert original_tuple == (1, 2)  # Original tuple unchanged

    @mutant
    def nested_modify(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = nested_modify(original_data)
    assert isinstance(result, PMap)
    assert result['list'] == pvector([1, 2, 3, 4])
    assert result['dict'] == pmap({'a': 1, 'c': 3})
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original data unchanged


# LLM-generated content at query #3
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
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

    # Test freezing a defaultdict
    dd = collections.defaultdict(int, {'a': [1, 2]})
    assert freeze(dd) == pmap({'a': pvector([1, 2])})

    # Test freezing with strict=False
    assert freeze([1, {'a': 3}], strict=False) == pvector([1, {'a': 3}])

    # Test freezing already frozen structures with strict=True
    pvec = pvector([1, 2])
    assert freeze(pvec, strict=True) == pvector([1, 2])

    # Test freezing already frozen structures with strict=False
    pvec = pvector([1, 2])
    assert freeze(pvec, strict=False) == pvector([1, 2])

    # Test freezing a nested tuple
    assert freeze((1, (2, [3]))) == (1, (2, pvector([3])))

    # Test freezing an empty list
    assert freeze([]) == pvector([])

    # Test freezing an empty dict
    assert freeze({}) == pmap({})

    # Test freezing an empty set
    assert freeze(set()) == pset(set())

    # Test freezing an empty tuple
    assert freeze(()) == ()

    # Test freezing a non-container type
    assert freeze(42) == 42


# LLM-generated content at query #4
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
    default_dict = collections.defaultdict(int, {'a': 1, 'b': [2, 3]})
    assert freeze(default_dict) == pmap({'a': 1, 'b': pvector([2, 3])})

    # Test freezing with strict=False
    assert freeze([1, {'a': 3}], strict=False) == pvector([1, {'a': 3}])

    # Test freezing a pvector (should remain unchanged if strict=True)
    pv = pvector([1, 2, 3])
    assert freeze(pv) == pv

    # Test freezing a pmap (should remain unchanged if strict=True)
    pm = pmap({'a': 1})
    assert freeze(pm) == pm

    # Test freezing a pset (should remain unchanged)
    ps = pset([1, 2])
    assert freeze(ps) == ps

    # Test freezing a non-container type (should remain unchanged)
    assert freeze(42) == 42
    assert freeze("hello") == "hello"


# LLM-generated content at query #5
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
        d['nested']['value'] = 10
        return d

    original_dict = {'nested': {'value': 5}}
    result = modify_nested(original_dict)
    assert original_dict == {'nested': {'value': 5}}  # Original unchanged
    assert result == pmap({'nested': pmap({'value': 10})})  # Result frozen

    # Test with kwargs
    @mutant
    def modify_kwargs(a, b=[]):
        b.append(a)
        return b

    original_kwarg = [1]
    result = modify_kwargs(2, b=original_kwarg)
    assert original_kwarg == [1]  # Original unchanged
    assert result == pvector([1, 2])  # Result frozen

    # Test return value is frozen
    @mutant
    def return_dict():
        return {'a': 1}

    result = return_dict()
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1})

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
    assert result == pvector([1, 2, 3, 4])  # Result frozen


# LLM-generated content at query #6
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
        data['nested']['value'] = 2
        return data

    original = {'list': [1, 2, 3], 'nested': {'value': 1}}
    result = modify_nested(original)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['nested'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'nested': pmap({'value': 2})})
    assert original == {'list': [1, 2, 3], 'nested': {'value': 1}}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    result = modify_kwargs(x=1, y=2)
    assert isinstance(result, PMap)
    assert result == pmap({'x': 10, 'y': 2})

    # Test return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with non-container types (should remain unchanged)
    @mutant
    def identity(x):
        return x

    assert identity(42) == 42
    assert identity("hello") == "hello"


# LLM-generated content at query #7
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
    assert original == [1, 2, 3]  # Original should be unchanged

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 100
        return d

    original_dict = {'nested': {'value': 50}}
    result_dict = modify_nested(original_dict)
    assert isinstance(result_dict, PMap)
    assert result_dict == pmap({'nested': pmap({'value': 100})})
    assert original_dict == {'nested': {'value': 50}}  # Original should be unchanged

    # Test with multiple arguments
    @mutant
    def combine(lst1, lst2):
        lst1.extend(lst2)
        return lst1

    list1 = [1, 2]
    list2 = [3, 4]
    result_combine = combine(list1, list2)
    assert isinstance(result_combine, PVector)
    assert result_combine == pvector([1, 2, 3, 4])
    assert list1 == [1, 2]  # Original should be unchanged
    assert list2 == [3, 4]  # Original should be unchanged

    # Test with keyword arguments
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_kwarg = {'a': 1}
    result_kwarg = update_dict(original_kwarg, 'b', 2)
    assert isinstance(result_kwarg, PMap)
    assert result_kwarg == pmap({'a': 1, 'b': 2})
    assert original_kwarg == {'a': 1}  # Original should be unchanged

    # Test return value is frozen
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result_return = return_mutable()
    assert isinstance(result_return, PVector)
    assert result_return == pvector([1, 2, 3])

    # Test with non-mutable arguments
    @mutant
    def process_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result_tuple = process_tuple(original_tuple)
    assert isinstance(result_tuple, tuple)
    assert result_tuple == (1, 2, 3, 4)
    assert original_tuple == (1, 2, 3)  # Original should be unchanged


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation with simple types
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add("a", "b") == "ab"

    # Test with mutable containers
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original should be unchanged

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['a'].append(4)
        return d

    input_dict = {'a': [1, 2, 3]}
    result = modify_nested(input_dict)
    assert result == pmap({'a': pvector([1, 2, 3, 4])})
    assert input_dict == {'a': [1, 2, 3]}  # Original should be unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(a, b, **kwargs):
        kwargs['c'].append(4)
        return kwargs

    input_kwargs = {'c': [1, 2, 3]}
    result = modify_kwargs(1, 2, **input_kwargs)
    assert result == pmap({'c': pvector([1, 2, 3, 4])})
    assert input_kwargs == {'c': [1, 2, 3]}  # Original should be unchanged

    # Test with strict=False
    @mutant
    def no_strict_modify(lst):
        return lst

    input_list = pvector([1, 2, 3])
    result = no_strict_modify(input_list)
    assert result == input_list  # Should not be double-frozen


# LLM-generated content at query #9
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
        data['a'].append(1)
        data['b']['c'] = 2
        return data

    original = {'a': [1, 2], 'b': {'c': 3}}
    result = modify_nested(original)
    assert original == {'a': [1, 2], 'b': {'c': 3}}  # Original unchanged
    assert result == pmap({'a': pvector([1, 2, 1]), 'b': pmap({'c': 2})})  # Return value is frozen

    # Test with kwargs
    @mutant
    def modify_with_kwargs(x, y=None):
        if y is not None:
            y.add(5)
        x['key'] = 'modified'
        return x, y

    original_x = {'key': 'value'}
    original_y = {1, 2, 3}
    result_x, result_y = modify_with_kwargs(original_x, y=original_y)
    assert original_x == {'key': 'value'}  # Original unchanged
    assert original_y == {1, 2, 3}  # Original unchanged
    assert result_x == pmap({'key': 'modified'})  # Return value is frozen
    assert result_y == pset({1, 2, 3, 5})  # Return value is frozen

    # Test with tuple
    @mutant
    def modify_tuple(t):
        return t + (4,)

    original = (1, 2, 3)
    result = modify_tuple(original)
    assert original == (1, 2, 3)  # Original unchanged
    assert result == (1, 2, 3, 4)  # Return value is frozen (tuples remain tuples)

    # Test with strict=False
    @mutant
    def no_strict_modify(lst):
        return lst

    original = pvector([1, 2, 3])
    result = no_strict_modify(original)
    assert result == original  # Should return as-is when already frozen


# LLM-generated content at query #10
#--------------------------

```python
def test_freeze():
    # Test freezing a list with nested structures
    test_list = [1, {'a': 3, 'b': [4, 5]}, [6, 7]]
    frozen = freeze(test_list)
    assert isinstance(frozen, PVector)
    assert isinstance(frozen[1], PMap)
    assert frozen[1]['a'] == 3
    assert isinstance(frozen[1]['b'], PVector)
    assert frozen[1]['b'] == pvector([4, 5])
    assert isinstance(frozen[2], PVector)
    assert frozen[2] == pvector([6, 7])

    # Test freezing a dict with nested structures
    test_dict = {'x': 1, 'y': [2, 3], 'z': {'a': 4}}
    frozen = freeze(test_dict)
    assert isinstance(frozen, PMap)
    assert frozen['x'] == 1
    assert isinstance(frozen['y'], PVector)
    assert frozen['y'] == pvector([2, 3])
    assert isinstance(frozen['z'], PMap)
    assert frozen['z']['a'] == 4

    # Test freezing a set
    test_set = {1, 2, 3}
    frozen = freeze(test_set)
    assert isinstance(frozen, PSet)
    assert frozen == pset({1, 2, 3})

    # Test freezing a tuple with nested structures
    test_tuple = (1, [2, 3], {'a': 4})
    frozen = freeze(test_tuple)
    assert isinstance(frozen, tuple)
    assert frozen[0] == 1
    assert isinstance(frozen[1], PVector)
    assert frozen[1] == pvector([2, 3])
    assert isinstance(frozen[2], PMap)
    assert frozen[2]['a'] == 4

    # Test freezing a defaultdict
    test_defaultdict = collections.defaultdict(int, {'a': 1, 'b': [2, 3]})
    frozen = freeze(test_defaultdict)
    assert isinstance(frozen, PMap)
    assert frozen['a'] == 1
    assert isinstance(frozen['b'], PVector)
    assert frozen['b'] == pvector([2, 3])

    # Test freezing with strict=False
    test_list_strict = [1, {'a': [2, 3]}]
    frozen_strict = freeze(test_list_strict, strict=True)
    frozen_not_strict = freeze(test_list_strict, strict=False)
    assert isinstance(frozen_strict[1]['a'], PVector)
    assert isinstance(frozen_not_strict[1]['a'], list)

    # Test freezing already frozen structures
    already_frozen = pvector([1, pmap({'a': 2})])
    frozen = freeze(already_frozen)
    assert frozen == already_frozen
    assert isinstance(frozen, PVector)
    assert isinstance(frozen[1], PMap)

    # Test freezing immutable types
    assert freeze(1) == 1
    assert freeze("hello") == "hello"
    assert freeze((1, 2, 3)) == (1, 2, 3)


# LLM-generated content at query #11
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
    result_dict = modify_dict(original_dict, 'b', [3, 4])
    assert original_dict == {'a': [1, 2]}  # Original unchanged
    assert result_dict == pmap({'a': pvector([1, 2]), 'b': pvector([3, 4])})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'key1': 'value1'}
    result_kwargs = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'key1': 'value1'}  # Original unchanged
    assert result_kwargs == pmap({'key1': 'value1', 'new_key': 'new_value'})

    # Test with multiple arguments
    @mutant
    def combine_and_modify(lst1, lst2):
        combined = lst1 + lst2
        combined.append(99)
        return combined

    list1 = [1, 2]
    list2 = [3, 4]
    result = combine_and_modify(list1, list2)
    assert list1 == [1, 2]  # Original unchanged
    assert list2 == [3, 4]  # Original unchanged
    assert result == pvector([1, 2, 3, 4, 99])


# LLM-generated content at query #12
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
    def modify_set(s, value):
        s.add(value)
        return s

    original_set = {1, 2}
    result = modify_set(original_set, 3)
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3})
    assert original_set == {1, 2}  # Original set should remain unchanged

    @mutant
    def modify_tuple(t, index, value):
        lst = list(t)
        lst[index] = value
        return tuple(lst)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple, 1, 99)
    assert result == (1, 99, 3)
    assert original_tuple == (1, 2, 3)  # Original tuple should remain unchanged

    @mutant
    def nested_mutation(data):
        data['list'].append(4)
        data['dict']['new_key'] = 'new_value'
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'key': 'value'}}
    result = nested_mutation(original_data)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'key': 'value', 'new_key': 'new_value'})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'key': 'value'}}  # Original data should remain unchanged


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

    original = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = modify_nested(original)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'c': 3})})
    assert original == {'list': [1, 2, 3], 'dict': {'a': 1}}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(a, b, **kwargs):
        kwargs['x'] = 10
        a.append(5)
        return {'a': a, 'b': b, 'kwargs': kwargs}

    original_a = [1, 2]
    original_b = {'key': 'value'}
    original_kwargs = {'y': 20}
    result = modify_kwargs(original_a, original_b, **original_kwargs)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['kwargs'], PMap)
    assert result == {'a': pvector([1, 2, 5]), 'b': pmap({'key': 'value'}), 'kwargs': pmap({'y': 20, 'x': 10})}
    assert original_a == [1, 2]
    assert original_b == {'key': 'value'}
    assert original_kwargs == {'y': 20}

    # Test return value is frozen
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False (should still work the same way)
    @mutant
    def test_strict_false(data):
        return data

    result = test_strict_false([1, 2, 3])
    assert isinstance(result, PVector)


# LLM-generated content at query #14
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
        data['list'].append(1)
        data['dict']['new_key'] = 'value'
        return data

    original_data = {'list': [1, 2], 'dict': {'key': 'val'}}
    result = nested_operation(original_data)

    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'key': 'val', 'new_key': 'value'})})
    assert original_data == {'list': [1, 2], 'dict': {'key': 'val'}}  # Original should be unchanged

    # Test with kwargs
    @mutant
    def kwargs_test(a, b, **kwargs):
        kwargs['new_key'] = 'new_value'
        a.append(1)
        return {'a': a, 'b': b, 'kwargs': kwargs}

    original_a = [1]
    original_b = 2
    original_kwargs = {'key': 'value'}
    result = kwargs_test(original_a, original_b, **original_kwargs)

    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert result == pmap({'a': pvector([1, 1]), 'b': 2, 'kwargs': pmap({'key': 'value', 'new_key': 'new_value'})})
    assert original_a == [1]
    assert original_kwargs == {'key': 'value'}


# LLM-generated content at query #15
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
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original dict should remain unchanged

    @mutant
    def modify_set(s, value):
        s.add(value)
        return s

    original_set = {1, 2}
    result = modify_set(original_set, 3)
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3})
    assert original_set == {1, 2}  # Original set should remain unchanged

    @mutant
    def modify_tuple(t, value):
        return t + (value,)

    original_tuple = (1, 2)
    result = modify_tuple(original_tuple, 3)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)
    assert original_tuple == (1, 2)  # Original tuple should remain unchanged

    @mutant
    def nested_mutation(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_data = {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    result = nested_mutation(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'b': 2, 'c': 3})})
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}  # Original should remain unchanged


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

    # Test freezing a defaultdict
    default_dict = collections.defaultdict(int, {'a': [1, 2]})
    assert freeze(default_dict) == pmap({'a': pvector([1, 2])})

    # Test freezing a pvector
    pv = pvector([1, 2, 3])
    assert freeze(pv) == pv

    # Test freezing a pmap
    pm = pmap({'a': 1})
    assert freeze(pm) == pm

    # Test freezing a pset
    ps = pset([1, 2])
    assert freeze(ps) == ps

    # Test freezing a tuple
    assert freeze((1, 2, 3)) == (1, 2, 3)

    # Test freezing a non-container
    assert freeze(42) == 42

    # Test strict=False
    assert freeze([1, {'a': [3, 4]}], strict=False) == pvector([1, {'a': [3, 4]}])


# LLM-generated content at query #2
#--------------------------

```python
def test_thaw():
    # Test thawing pvector to list
    assert thaw(pvector([1, 2, 3])) == [1, 2, 3]

    # Test thawing nested pvector
    assert thaw(pvector([1, pvector([2, 3])])) == [1, [2, 3]]

    # Test thawing pmap to dict
    assert thaw(pmap({'a': 1, 'b': 2})) == {'a': 1, 'b': 2}

    # Test thawing nested pmap
    assert thaw(pmap({'a': pvector([1, 2]), 'b': 3})) == {'a': [1, 2], 'b': 3}

    # Test thawing pset to set
    assert thaw(pset([1, 2, 3])) == {1, 2, 3}

    # Test thawing tuple
    assert thaw((1, pvector([2, 3]))) == (1, [2, 3])

    # Test thawing with strict=False
    assert thaw(pvector([1, 2, 3]), strict=False) == [1, 2, 3]
    assert thaw(pmap({'a': 1, 'b': 2}), strict=False) == {'a': 1, 'b': 2}

    # Test thawing non-pyrsistent objects
    assert thaw(1) == 1
    assert thaw("hello") == "hello"
    assert thaw([1, 2, 3]) == [1, 2, 3]  # list remains list when strict=False
    assert thaw({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}  # dict remains dict when strict=False

    # Test thawing mixed types
    mixed = pmap({
        'list': pvector([1, 2, 3]),
        'set': pset([4, 5, 6]),
        'nested': pmap({'a': pvector([7, 8])})
    })
    expected = {
        'list': [1, 2, 3],
        'set': {4, 5, 6},
        'nested': {'a': [7, 8]}
    }
    assert thaw(mixed) == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_thaw():
    # Test thawing a PVector
    pvec = pvector([1, 2, 3])
    assert thaw(pvec) == [1, 2, 3]

    # Test thawing a PVector with nested structures
    pvec_nested = pvector([1, pmap({'a': 2}), pset([3, 4])])
    assert thaw(pvec_nested) == [1, {'a': 2}, {3, 4}]

    # Test thawing a PMap
    pmap_obj = pmap({'a': 1, 'b': 2})
    assert thaw(pmap_obj) == {'a': 1, 'b': 2}

    # Test thawing a PMap with nested structures
    pmap_nested = pmap({'a': pvector([1, 2]), 'b': pset([3, 4])})
    assert thaw(pmap_nested) == {'a': [1, 2], 'b': {3, 4}}

    # Test thawing a PSet
    pset_obj = pset([1, 2, 3])
    assert thaw(pset_obj) == {1, 2, 3}

    # Test thawing a tuple
    tuple_obj = (1, pvector([2, 3]))
    assert thaw(tuple_obj) == (1, [2, 3])

    # Test thawing a non-pyrsistent object
    assert thaw(1) == 1
    assert thaw("string") == "string"

    # Test thawing with strict=False
    list_obj = [1, 2, 3]
    assert thaw(list_obj, strict=False) == [1, 2, 3]

    dict_obj = {'a': 1, 'b': 2}
    assert thaw(dict_obj, strict=False) == {'a': 1, 'b': 2}

    # Test thawing nested structures with strict=False
    nested_list = [1, {'a': pvector([2, 3])}]
    assert thaw(nested_list, strict=False) == [1, {'a': pvector([2, 3])}]


# LLM-generated content at query #4
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
    assert original_list == [1, 2, 3]  # Original list unchanged
    assert result == pvector([1, 2, 3, 4])

    # Test with dict
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = add_to_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert original_dict == {'a': 1}  # Original dict unchanged
    assert result == pmap({'a': 1, 'b': 2})

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data

    original_nested = {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    result = modify_nested(original_nested)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}  # Original unchanged
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'dict': pmap({'a': 1, 'b': 2, 'c': 3})})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    original_kwargs = {'y': 20}
    result = modify_kwargs(**original_kwargs)
    assert isinstance(result, PMap)
    assert original_kwargs == {'y': 20}  # Original kwargs unchanged
    assert result == pmap({'y': 20, 'x': 10})

    # Test with return value
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #5
#--------------------------

```python
def test_thaw():
    # Test thawing pvector to list
    assert thaw(pvector([1, 2, 3])) == [1, 2, 3]

    # Test thawing nested pvector
    assert thaw(pvector([1, pvector([2, 3])])) == [1, [2, 3]]

    # Test thawing pmap to dict
    assert thaw(pmap({'a': 1, 'b': 2})) == {'a': 1, 'b': 2}

    # Test thawing nested pmap
    assert thaw(pmap({'a': pvector([1, 2]), 'b': 3})) == {'a': [1, 2], 'b': 3}

    # Test thawing pset to set
    assert thaw(pset([1, 2, 3])) == {1, 2, 3}

    # Test thawing tuple
    assert thaw((1, 2, 3)) == (1, 2, 3)

    # Test thawing nested tuple
    assert thaw((1, pvector([2, 3]))) == (1, [2, 3])

    # Test thawing with strict=False
    assert thaw(pvector([1, 2, 3]), strict=False) == [1, 2, 3]
    assert thaw(pmap({'a': 1, 'b': 2}), strict=False) == {'a': 1, 'b': 2}
    assert thaw(pset([1, 2, 3]), strict=False) == {1, 2, 3}
    assert thaw((1, 2, 3), strict=False) == (1, 2, 3)

    # Test thawing mixed types
    mixed = pmap({
        'list': pvector([1, 2, 3]),
        'dict': pmap({'a': 1}),
        'set': pset([1, 2]),
        'tuple': (1, 2)
    })
    expected = {
        'list': [1, 2, 3],
        'dict': {'a': 1},
        'set': {1, 2},
        'tuple': (1, 2)
    }
    assert thaw(mixed) == expected

    # Test thawing non-pyrsistent types (should return as-is)
    assert thaw(1) == 1
    assert thaw("string") == "string"
    assert thaw(None) is None


# LLM-generated content at query #6
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation with simple types
    @mutant
    def add_one(x):
        return x + 1

    assert add_one(5) == 6
    assert isinstance(add_one(5), int)

    # Test with list mutation
    @mutant
    def append_item(lst, item):
        lst.append(item)
        return lst

    original_list = [1, 2, 3]
    result = append_item(original_list, 4)
    assert original_list == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])
    assert isinstance(result, PVector)

    # Test with dict mutation
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}  # Original unchanged
    assert result == pmap({'a': 1, 'b': 2})
    assert isinstance(result, PMap)

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
    def process_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'a': 1, 'b': 2}
    result = process_kwargs(**original_kwargs)
    assert original_kwargs == {'a': 1, 'b': 2}  # Original unchanged
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

    # Test with mixed args and kwargs
    @mutant
    def mixed_args_kwargs(arg1, arg2, **kwargs):
        arg1.append('modified')
        arg2['new_key'] = 'new_value'
        kwargs['kwarg_key'] = 'kwarg_value'
        return (arg1, arg2, kwargs)

    original_arg1 = ['original']
    original_arg2 = {'original': 'value'}
    original_kwargs = {'existing': 'kwarg'}

    result = mixed_args_kwargs(original_arg1, original_arg2, **original_kwargs)

    assert original_arg1 == ['original']  # Original unchanged
    assert original_arg2 == {'original': 'value'}  # Original unchanged
    assert original_kwargs == {'existing': 'kwarg'}  # Original unchanged

    result_arg1, result_arg2, result_kwargs = result
    assert result_arg1 == pvector(['original', 'modified'])
    assert result_arg2 == pmap({'original': 'value', 'new_key': 'new_value'})
    assert result_kwargs == pmap({'existing': 'kwarg', 'kwarg_key': 'kwarg_value'})

    # Test that return value is frozen
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
    assert original_tuple == (1, 2, 3)  # Original unchanged
    assert result == (1, 2, 3, 4)
    assert isinstance(result, tuple)


# LLM-generated content at query #7
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
        data['nested'].append(1)
        return data

    original_nested = {'nested': [1, 2]}
    result = modify_nested(original_nested)
    assert isinstance(result['nested'], PVector)
    assert result == pmap({'nested': pvector([1, 2, 1])})
    assert original_nested == {'nested': [1, 2]}  # Original unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    result = modify_kwargs(a=1, b=2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

    # Test with tuple (should remain tuple)
    @mutant
    def process_tuple(t):
        return t + (1,)

    original_tuple = (1, 2)
    result = process_tuple(original_tuple)
    assert isinstance(result, tuple)
    assert result == (1, 2, 1)
    assert original_tuple == (1, 2)  # Original unchanged


# LLM-generated content at query #8
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
    def add_to_set(s, value):
        s.add(value)
        return s

    original_set = {1, 2}
    result = add_to_set(original_set, 3)

    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3})
    assert original_set == {1, 2}  # Original set should remain unchanged

    @mutant
    def modify_tuple(t, index, value):
        lst = list(t)
        lst[index] = value
        return tuple(lst)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple, 1, 4)

    assert isinstance(result, tuple)
    assert result == (1, 4, 3)
    assert original_tuple == (1, 2, 3)  # Original tuple should remain unchanged


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
    assert original_list == [1, 2, 3]  # Original unchanged
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 10
        return d

    original_dict = {'nested': {'value': 5}}
    result = modify_nested(original_dict)
    assert original_dict == {'nested': {'value': 5}}  # Original unchanged
    assert result == pmap({'nested': pmap({'value': 10})})  # Return value is frozen

    # Test with multiple arguments
    @mutant
    def combine_and_modify(lst1, lst2):
        combined = lst1 + lst2
        combined.append(99)
        return combined

    list1 = [1, 2]
    list2 = [3, 4]
    result = combine_and_modify(list1, list2)
    assert list1 == [1, 2]  # Original unchanged
    assert list2 == [3, 4]  # Original unchanged
    assert result == pvector([1, 2, 3, 4, 99])  # Return value is frozen

    # Test with keyword arguments
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'key1': 'value1'}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'key1': 'value1'}  # Original unchanged
    assert result == pmap({'key1': 'value1', 'new_key': 'new_value'})  # Return value is frozen

    # Test that non-container types are returned as-is
    @mutant
    def simple_function(x):
        return x + 1

    assert simple_function(5) == 6


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
    assert original_list == [1, 2, 3]  # Original unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['nested']['value'] = 'modified'
        data['list'].append(4)
        return data

    original_data = {'nested': {'value': 'original'}, 'list': [1, 2, 3]}
    result = modify_nested(original_data)
    assert isinstance(result, PMap)
    assert result == pmap({'nested': pmap({'value': 'modified'}), 'list': pvector([1, 2, 3, 4])})
    assert original_data == {'nested': {'value': 'original'}, 'list': [1, 2, 3]}

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['a'] = kwargs.get('a', 0) + 1
        return kwargs

    original_kwargs = {'a': 1, 'b': 2}
    result = modify_kwargs(**original_kwargs)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 2, 'b': 2})
    assert original_kwargs == {'a': 1, 'b': 2}

    # Test with mixed args and kwargs
    @mutant
    def mixed_modification(lst, d, **kwargs):
        lst.append(1)
        d['new_key'] = 'new_value'
        kwargs['new_kwarg'] = True
        return {'list': lst, 'dict': d, 'kwargs': kwargs}

    original_list = [1, 2]
    original_dict = {'a': 1}
    original_kwargs = {'b': 2}
    result = mixed_modification(original_list, original_dict, **original_kwargs)
    assert isinstance(result, PMap)
    assert result['list'] == pvector([1, 2, 1])
    assert result['dict'] == pmap({'a': 1, 'new_key': 'new_value'})
    assert result['kwargs'] == pmap({'b': 2, 'new_kwarg': True})
    assert original_list == [1, 2]
    assert original_dict == {'a': 1}
    assert original_kwargs == {'b': 2}

    # Test with non-mutable types
    @mutant
    def no_mutation(x, y):
        return x + y

    assert no_mutation(1, 2) == 3
    assert no_mutation("a", "b") == "ab"

    # Test with tuple
    @mutant
    def modify_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple)
    assert result == (1, 2, 3, 4)
    assert original_tuple == (1, 2, 3)


# LLM-generated content at query #11
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
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

    # Test freezing a defaultdict
    dd = collections.defaultdict(list)
    dd['a'] = [1, 2]
    assert freeze(dd) == pmap({'a': pvector([1, 2])})

    # Test freezing with strict=False
    assert freeze([1, {'a': [3, 4]}], strict=False) == pvector([1, {'a': [3, 4]}])

    # Test freezing a pvector (should remain unchanged if strict=True)
    pv = pvector([1, 2, 3])
    assert freeze(pv) == pv

    # Test freezing a pmap (should remain unchanged if strict=True)
    pm = pmap({'a': 1})
    assert freeze(pm) == pm

    # Test freezing a pset (should remain unchanged if strict=True)
    ps = pset([1, 2])
    assert freeze(ps) == ps

    # Test freezing a simple value (should remain unchanged)
    assert freeze(42) == 42


# LLM-generated content at query #12
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

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['nested'].append(5)
        return data

    original = {'nested': [1, 2]}
    result = modify_nested(original)
    assert original == {'nested': [1, 2]}  # Original unchanged
    assert result == pmap({'nested': pvector([1, 2, 5])})  # Frozen result

    # Test with multiple arguments
    @mutant
    def combine_dicts(d1, d2):
        d1.update(d2)
        return d1

    original1 = {'a': 1}
    original2 = {'b': 2}
    result = combine_dicts(original1, original2)
    assert original1 == {'a': 1}  # Original unchanged
    assert original2 == {'b': 2}  # Original unchanged
    assert result == pmap({'a': 1, 'b': 2})  # Frozen result

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original = {'old_key': 'old_value'}
    result = modify_kwargs(**original)
    assert original == {'old_key': 'old_value'}  # Original unchanged
    assert result == pmap({'old_key': 'old_value', 'new_key': 'new_value'})  # Frozen result

    # Test with mixed arguments
    @mutant
    def mixed_args(arg1, arg2, kwarg1=None):
        arg1.append(arg2)
        if kwarg1:
            kwarg1['modified'] = True
        return arg1, kwarg1

    list_arg = [1, 2]
    dict_arg = {'key': 'value'}
    result_list, result_dict = mixed_args(list_arg, 3, kwarg1=dict_arg)
    assert list_arg == [1, 2]  # Original unchanged
    assert dict_arg == {'key': 'value'}  # Original unchanged
    assert result_list == pvector([1, 2, 3])  # Frozen result
    assert result_dict == pmap({'key': 'value', 'modified': True})  # Frozen result

    # Test return value freezing
    @mutant
    def return_mutable():
        return {'a': [1, 2, 3]}

    result = return_mutable()
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2, 3])


# LLM-generated content at query #13
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

    # Test freezing a PVector (strict=True)
    pv = pvector([1, {'a': 3}])
    assert freeze(pv) == pvector([1, pmap({'a': 3})])

    # Test freezing a PMap (strict=True)
    pm = pmap({'a': 1, 'b': [2, 3]})
    assert freeze(pm) == pmap({'a': 1, 'b': pvector([2, 3])})

    # Test freezing a PSet
    ps = pset([1, 2])
    assert freeze(ps) == pset([1, 2])

    # Test freezing a simple value
    assert freeze(42) == 42


# LLM-generated content at query #14
#--------------------------

```python
def test_freeze():
    # Test freezing a set
    assert freeze(set([1, 2])) == pset([1, 2])

    # Test freezing a list with nested dict
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

    # Test freezing a tuple with nested list
    assert freeze((1, [])) == (1, pvector([]))

    # Test freezing a dict
    assert freeze({'a': 1, 'b': [2, 3]}) == pmap({'a': 1, 'b': pvector([2, 3])})

    # Test freezing a defaultdict
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = [2, 3]
    assert freeze(dd) == pmap({'a': 1, 'b': pvector([2, 3])})

    # Test freezing a PVector (strict=True)
    pv = pvector([1, 2, {'a': 3}])
    assert freeze(pv) == pvector([1, 2, pmap({'a': 3})])

    # Test freezing a PMap (strict=True)
    pm = pmap({'a': 1, 'b': [2, 3]})
    assert freeze(pm) == pmap({'a': 1, 'b': pvector([2, 3])})

    # Test freezing a PSet
    ps = pset([1, 2, 3])
    assert freeze(ps) == pset([1, 2, 3])

    # Test freezing with strict=False
    assert freeze([1, {'a': 3}], strict=False) == pvector([1, {'a': 3}])

    # Test freezing immutable types (should return as-is)
    assert freeze(1) == 1
    assert freeze("hello") == "hello"
    assert freeze((1, 2, 3)) == (1, 2, 3)


# LLM-generated content at query #15
#--------------------------

```python
def test_freeze():
    # Test freezing a list
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

    # Test freezing a nested list
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

    # Test freezing a dict
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

    # Test freezing a nested dict
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

    # Test freezing a defaultdict
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

    # Test freezing a set
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

    # Test freezing a tuple
    assert freeze((1, 2, 3)) == (1, 2, 3)

    # Test freezing a nested tuple
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

    # Test freezing with strict=False
    assert freeze([1, [2, 3]], strict=False) == pvector([1, [2, 3]])

    # Test freezing a PVector
    pv = pvector([1, 2, 3])
    assert freeze(pv) == pv

    # Test freezing a PMap
    pm = pmap({'a': 1, 'b': 2})
    assert freeze(pm) == pm

    # Test freezing a PSet
    ps = pset({1, 2, 3})
    assert freeze(ps) == ps

    # Test freezing a mixed structure
    mixed = {'a': [1, 2], 'b': (3, 4), 'c': {5, 6}}
    expected = pmap({'a': pvector([1, 2]), 'b': (3, 4), 'c': pset({5, 6})})
    assert freeze(mixed) == expected


# LLM-generated content at query #16
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
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen

    # Test with dict mutation
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}  # Original unchanged
    assert result == pmap({'a': 1, 'b': 2})  # Return value is frozen

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

    # Test with mixed args and kwargs
    @mutant
    def mixed_mutation(lst, d, **kwargs):
        lst.append(1)
        d['new'] = 'value'
        kwargs['new_kwarg'] = 'kwarg_value'
        return {'list': lst, 'dict': d, 'kwargs': kwargs}

    original_list = [1, 2]
    original_dict = {'a': 1}
    original_kwargs = {'b': 2}
    result = mixed_mutation(original_list, original_dict, **original_kwargs)
    assert original_list == [1, 2]
    assert original_dict == {'a': 1}
    assert original_kwargs == {'b': 2}
    assert result == pmap({
        'list': pvector([1, 2, 1]),
        'dict': pmap({'a': 1, 'new': 'value'}),
        'kwargs': pmap({'b': 2, 'new_kwarg': 'kwarg_value'})
    })

    # Test that non-container types are handled correctly
    @mutant
    def simple_operation(x, y):
        return x + y

    assert simple_operation(1, 2) == 3


# LLM-generated content at query #17
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
    def nested_operations(data):
        data['list'].append(4)
        data['set'].add(4)
        return data

    original_data = {'list': [1, 2, 3], 'set': {1, 2, 3}}
    result = nested_operations(original_data)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)
    assert result['list'] == pvector([1, 2, 3, 4])
    assert result['set'] == pset({1, 2, 3, 4})
    assert original_data == {'list': [1, 2, 3], 'set': {1, 2, 3}}  # Ensure original is unchanged


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
    def nested_operation(data):
        data['list'].append(1)
        data['dict']['key'] = 'value'
        return data

    original_data = {'list': [1, 2], 'dict': {'key': 'old'}}
    result = nested_operation(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'key': 'value'})})
    assert original_data == {'list': [1, 2], 'dict': {'key': 'old'}}


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    dd = collections.defaultdict(list)
    dd['a'] = [1, 2]
    assert freeze(dd) == pmap({'a': pvector([1, 2])})

    # Test freezing with strict=False
    assert freeze([1, {'a': 3}], strict=False) == pvector([1, {'a': 3}])

    # Test freezing a pvector (should remain unchanged if strict=True)
    pv = pvector([1, 2, 3])
    assert freeze(pv) == pv

    # Test freezing a pmap (should remain unchanged if strict=True)
    pm = pmap({'a': 1})
    assert freeze(pm) == pm

    # Test freezing a pset (should remain unchanged)
    ps = pset([1, 2])
    assert freeze(ps) == ps

    # Test freezing a simple integer (should remain unchanged)
    assert freeze(42) == 42


# LLM-generated content at query #2
#--------------------------

```python
def test_thaw():
    # Test thawing PVector to list
    pvec = pvector([1, 2, 3])
    assert thaw(pvec) == [1, 2, 3]

    # Test thawing nested PVector
    nested_pvec = pvector([1, pvector([4, 5])])
    assert thaw(nested_pvec) == [1, [4, 5]]

    # Test thawing PMap to dict
    pmap_obj = pmap({'a': 1, 'b': 2})
    assert thaw(pmap_obj) == {'a': 1, 'b': 2}

    # Test thawing nested PMap
    nested_pmap = pmap({'a': 1, 'b': pmap({'c': 3})})
    assert thaw(nested_pmap) == {'a': 1, 'b': {'c': 3}}

    # Test thawing PSet to set
    pset_obj = pset({1, 2, 3})
    assert thaw(pset_obj) == {1, 2, 3}

    # Test thawing tuple
    tuple_obj = (1, pvector([2, 3]))
    assert thaw(tuple_obj) == (1, [2, 3])

    # Test thawing mixed types
    mixed = pmap({'a': pvector([1, 2]), 'b': pset({3, 4})})
    assert thaw(mixed) == {'a': [1, 2], 'b': {3, 4}}

    # Test thawing with strict=False
    assert thaw(pvector([1, 2]), strict=False) == [1, 2]
    assert thaw(pmap({'a': 1}), strict=False) == {'a': 1}

    # Test thawing non-pyrsistent objects
    assert thaw(42) == 42
    assert thaw("hello") == "hello"


# LLM-generated content at query #3
#--------------------------

```python
def test_thaw():
    # Test thawing pvector to list
    assert thaw(pvector([1, 2, 3])) == [1, 2, 3]
    assert thaw(pvector([1, pvector([2, 3])])) == [1, [2, 3]]

    # Test thawing pmap to dict
    assert thaw(pmap({'a': 1, 'b': 2})) == {'a': 1, 'b': 2}
    assert thaw(pmap({'a': pvector([1, 2])})) == {'a': [1, 2]}

    # Test thawing pset to set
    assert thaw(pset([1, 2, 3])) == {1, 2, 3}

    # Test thawing tuple
    assert thaw((1, 2, 3)) == (1, 2, 3)
    assert thaw((1, pvector([2, 3]))) == (1, [2, 3])

    # Test thawing with strict=False
    assert thaw(pvector([1, 2, 3]), strict=False) == [1, 2, 3]
    assert thaw([1, 2, 3], strict=False) == [1, 2, 3]

    # Test thawing mixed containers
    mixed = pmap({'a': pvector([1, 2]), 'b': pset([3, 4])})
    assert thaw(mixed) == {'a': [1, 2], 'b': {3, 4}}

    # Test thawing non-container types
    assert thaw(1) == 1
    assert thaw("hello") == "hello"
    assert thaw(None) is None


# LLM-generated content at query #4
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
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen

    # Test with dict
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}  # Original unchanged
    assert result == pmap({'a': 1, 'b': 2})  # Return value is frozen

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['nested']['value'] = 10
        return data

    original = {'list': [1, 2, 3], 'nested': {'value': 5}}
    result = modify_nested(original)
    assert original == {'list': [1, 2, 3], 'nested': {'value': 5}}  # Original unchanged
    assert result == pmap({'list': pvector([1, 2, 3, 4]), 'nested': pmap({'value': 10})})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 100
        return kwargs

    original_kwargs = {'x': 10, 'y': 20}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'x': 10, 'y': 20}  # Original unchanged
    assert result == pmap({'x': 100, 'y': 20})

    # Test with non-mutable operations
    @mutant
    def pure_function(a, b):
        return a + b

    assert pure_function(3, 4) == 7  # Should still work with pure functions

    # Test with mixed arguments
    @mutant
    def mixed_args(lst, d, x):
        lst.append(x)
        d['new'] = x
        return (lst, d)

    original_list = [1, 2]
    original_dict = {'a': 1}
    result = mixed_args(original_list, original_dict, 3)
    assert original_list == [1, 2]
    assert original_dict == {'a': 1}
    assert result == (pvector([1, 2, 3]), pmap({'a': 1, 'new': 3}))


# LLM-generated content at query #5
#--------------------------

```python
def test_thaw():
    # Test basic types
    assert thaw(1) == 1
    assert thaw("hello") == "hello"
    assert thaw(None) is None

    # Test PVector to list
    assert thaw(pvector([1, 2, 3])) == [1, 2, 3]
    assert thaw(pvector([1, pvector([2, 3])])) == [1, [2, 3]]

    # Test PMap to dict
    assert thaw(pmap({"a": 1, "b": 2})) == {"a": 1, "b": 2}
    assert thaw(pmap({"a": pvector([1, 2])})) == {"a": [1, 2]}

    # Test PSet to set
    assert thaw(pset([1, 2, 3])) == {1, 2, 3}

    # Test tuple
    assert thaw((1, 2, 3)) == (1, 2, 3)
    assert thaw((1, pvector([2, 3]))) == (1, [2, 3])

    # Test nested structures
    nested = pmap({
        "a": pvector([1, 2]),
        "b": pset([3, 4]),
        "c": (5, pvector([6, 7]))
    })
    expected = {
        "a": [1, 2],
        "b": {3, 4},
        "c": (5, [6, 7])
    }
    assert thaw(nested) == expected

    # Test strict mode
    assert thaw([1, 2], strict=True) == [1, 2]
    assert thaw({"a": 1}, strict=True) == {"a": 1}
    assert thaw([pmap({"a": 1})], strict=True) == [{"a": 1}]
    assert thaw(pvector([1, 2]), strict=False) == pvector([1, 2])
    assert thaw(pmap({"a": 1}), strict=False) == pmap({"a": 1})


# LLM-generated content at query #6
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
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 10
        return d

    original_dict = {'nested': {'value': 5}}
    result_dict = modify_nested(original_dict)
    assert original_dict == {'nested': {'value': 5}}  # Original unchanged
    assert result_dict == pmap({'nested': pmap({'value': 10})})  # Return value is frozen

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 100
        return kwargs

    original_kwargs = {'x': 1, 'y': 2}
    result_kwargs = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'x': 1, 'y': 2}  # Original unchanged
    assert result_kwargs == pmap({'x': 100, 'y': 2})  # Return value is frozen

    # Test with mixed args and kwargs
    @mutant
    def mixed_modification(lst, d, value):
        lst.append(value)
        d['new_key'] = value
        return lst, d

    original_list = [1, 2]
    original_dict = {'a': 3}
    result_list, result_dict = mixed_modification(original_list, original_dict, 4)
    assert original_list == [1, 2]  # Original list unchanged
    assert original_dict == {'a': 3}  # Original dict unchanged
    assert result_list == pvector([1, 2, 4])  # Return list is frozen
    assert result_dict == pmap({'a': 3, 'new_key': 4})  # Return dict is frozen

    # Test with immutable input
    @mutant
    def immutable_input(x):
        return x + 1

    assert immutable_input(5) == 6  # Works with non-container types


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
    assert result == pvector([1, 2, 3, 4])  # Result is frozen

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(5)
        data['nested']['value'] = 10
        return data

    original = {'list': [1, 2], 'nested': {'value': 5}}
    result = modify_nested(original)
    assert original == {'list': [1, 2], 'nested': {'value': 5}}  # Original unchanged
    assert result['list'] == pvector([1, 2, 5])
    assert result['nested']['value'] == 10

    # Test with kwargs
    @mutant
    def modify_kwargs(a, b, **kwargs):
        a.append(3)
        kwargs['c'].append(4)
        return {'a': a, 'b': b, 'kwargs': kwargs}

    original_a = [1]
    original_b = 2
    original_kwargs = {'c': [2]}
    result = modify_kwargs(original_a, original_b, **original_kwargs)
    assert original_a == [1]
    assert original_b == 2
    assert original_kwargs == {'c': [2]}
    assert result['a'] == pvector([1, 3])
    assert result['b'] == 2
    assert result['kwargs']['c'] == pvector([2, 4])

    # Test with strict=False
    @mutant
    def no_strict_modify(data):
        return data

    original = {'a': [1, 2]}
    result = no_strict_modify(original)
    assert result == original  # Should not be frozen when strict=False

    # Test return value is frozen
    @mutant
    def return_new():
        return {'new': [1, 2]}

    result = return_new()
    assert isinstance(result, dict)
    assert isinstance(result['new'], PVector)


# LLM-generated content at query #8
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
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen

    # Test with nested structures
    @mutant
    def modify_nested(d):
        d['nested']['value'] = 100
        return d

    original_dict = {'nested': {'value': 50}}
    result = modify_nested(original_dict)
    assert original_dict == {'nested': {'value': 50}}  # Original unchanged
    assert result == pmap({'nested': pmap({'value': 100})})  # Return value is frozen

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 999
        return kwargs

    original_kwargs = {'x': 1, 'y': 2}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'x': 1, 'y': 2}  # Original unchanged
    assert result == pmap({'x': 999, 'y': 2})  # Return value is frozen

    # Test with multiple arguments
    @mutant
    def combine_and_modify(a, b):
        a.update(b)
        return a

    original_a = {'x': 1}
    original_b = {'y': 2}
    result = combine_and_modify(original_a, original_b)
    assert original_a == {'x': 1}  # Original unchanged
    assert original_b == {'y': 2}  # Original unchanged
    assert result == pmap({'x': 1, 'y': 2})  # Return value is frozen

    # Test with non-container types (should pass through)
    @mutant
    def simple_math(x, y):
        return x + y

    assert simple_math(2, 3) == 5


# LLM-generated content at query #9
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

    original_dict = {'a': 1, 'b': 2}
    result = modify_dict(original_dict, 'c', 3)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})
    assert original_dict == {'a': 1, 'b': 2}  # Original dict should remain unchanged

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
    assert original_data == {'list': [1, 2, 3], 'dict': {'key': 'value'}}  # Original data should remain unchanged

    # Test with kwargs
    @mutant
    def modify_with_kwargs(lst=None, value=None):
        if lst is None:
            lst = []
        lst.append(value)
        return lst

    result = modify_with_kwargs(lst=[1, 2], value=3)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant():
    # Test basic mutation isolation
    @mutant
    def add_to_list(lst, value):
        lst.append(value)
        return lst

    original_list = [1, 2, 3]
    frozen_result = add_to_list(original_list, 4)
    assert isinstance(frozen_result, PVector)
    assert frozen_result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original should be unchanged

    # Test with dict mutation
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    frozen_dict = update_dict(original_dict, 'b', 2)
    assert isinstance(frozen_dict, PMap)
    assert frozen_dict == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Original should be unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['nested'].append(1)
        return data

    original_nested = {'nested': [1, 2]}
    result = modify_nested(original_nested)
    assert isinstance(result['nested'], PVector)
    assert result == pmap({'nested': pvector([1, 2, 1])})
    assert original_nested == {'nested': [1, 2]}

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'a': 1}
    result = modify_kwargs(**original_kwargs)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})
    assert original_kwargs == {'a': 1}

    # Test return value freezing
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with non-mutable operations
    @mutant
    def no_mutation(x):
        return x + 1

    assert no_mutation(5) == 6
    assert isinstance(no_mutation(5), int)  # Primitives remain unchanged


# LLM-generated content at query #11
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
    def combine_sets(s1, s2):
        return s1.union(s2)

    set1 = {1, 2}
    set2 = {3, 4}
    result = combine_sets(set1, set2)
    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})
    assert set1 == {1, 2} and set2 == {3, 4}  # Ensure originals are unchanged

    @mutant
    def process_tuple(t, item):
        return t + (item,)

    original_tuple = (1, 2)
    result = process_tuple(original_tuple, 3)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)
    assert original_tuple == (1, 2)  # Ensure original is unchanged

    @mutant
    def nested_operation(data):
        data['list'].append(1)
        data['dict']['key'] = 'value'
        return data

    original_data = {'list': [1, 2], 'dict': {'key': 'old'}}
    result = nested_operation(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert result == pmap({'list': pvector([1, 2, 1]), 'dict': pmap({'key': 'value'})})
    assert original_data == {'list': [1, 2], 'dict': {'key': 'old'}}  # Ensure original is unchanged


# LLM-generated content at query #12
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
    assert result == pvector([1, 2, 3, 4])  # Returned frozen

    # Test with multiple arguments
    @mutant
    def modify_dict_and_list(d, lst):
        d['new_key'] = 'value'
        lst.extend([4, 5])
        return d, lst

    original_dict = {'a': 1}
    original_list = [1, 2, 3]
    dict_result, list_result = modify_dict_and_list(original_dict, original_list)
    assert original_dict == {'a': 1}  # Original dict unchanged
    assert original_list == [1, 2, 3]  # Original list unchanged
    assert dict_result == pmap({'a': 1, 'new_key': 'value'})
    assert list_result == pvector([1, 2, 3, 4, 5])

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['nested']['value'] = 10
        data['list'].append(4)
        return data

    original = {'nested': {'value': 5}, 'list': [1, 2, 3]}
    result = modify_nested(original)
    assert original == {'nested': {'value': 5}, 'list': [1, 2, 3]}
    assert result == pmap({'nested': pmap({'value': 10}), 'list': pvector([1, 2, 3, 4])})

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'added'
        return kwargs

    original_kwargs = {'a': 1, 'b': 2}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'a': 1, 'b': 2}
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'added'})

    # Test with tuple (should remain tuple)
    @mutant
    def modify_tuple(t):
        return t + (4,)

    original = (1, 2, 3)
    result = modify_tuple(original)
    assert original == (1, 2, 3)
    assert result == (1, 2, 3, 4)

    # Test with set (should become pset)
    @mutant
    def modify_set(s):
        return s | {4}

    original = {1, 2, 3}
    result = modify_set(original)
    assert original == {1, 2, 3}
    assert result == pset({1, 2, 3, 4})

    # Test with non-container types (should remain unchanged)
    @mutant
    def pass_through(x):
        return x

    assert pass_through(42) == 42
    assert pass_through("string") == "string"
    assert pass_through(None) is None


# LLM-generated content at query #13
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
    def nested_operations(data):
        data['list'].append(1)
        data['set'].add(2)
        return data

    original_data = {'list': [1, 2], 'set': {3}}
    result = nested_operations(original_data)

    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)
    assert result['list'] == pvector([1, 2, 1])
    assert result['set'] == pset({3, 2})
    assert original_data == {'list': [1, 2], 'set': {3}}  # Original should be unchanged

    # Test with strict=False
    @mutant
    def non_strict_operations(data):
        return data

    result = non_strict_operations({'a': [1, 2]})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], list)  # Should not be frozen when strict=False in thaw


# LLM-generated content at query #14
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
    assert original_list == [1, 2, 3]  # Ensure original is unchanged

    # Test with dict
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = add_to_dict(original_dict, 'b', 2)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1}  # Ensure original is unchanged

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['lst'].append(4)
        data['d']['c'] = 3
        return data

    original_nested = {'lst': [1, 2, 3], 'd': {'a': 1}}
    result = modify_nested(original_nested)
    assert isinstance(result, PMap)
    assert isinstance(result['lst'], PVector)
    assert isinstance(result['d'], PMap)
    assert result == pmap({'lst': pvector([1, 2, 3, 4]), 'd': pmap({'a': 1, 'c': 3})})
    assert original_nested == {'lst': [1, 2, 3], 'd': {'a': 1}}  # Ensure original is unchanged

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    original_kwargs = {'y': 20}
    result = modify_kwargs(**original_kwargs)
    assert isinstance(result, PMap)
    assert result == pmap({'y': 20, 'x': 10})
    assert original_kwargs == {'y': 20}  # Ensure original is unchanged


# LLM-generated content at query #15
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
    def modify_set(s, value):
        s.add(value)
        return s

    original_set = {1, 2}
    result = modify_set(original_set, 3)

    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3})
    assert original_set == {1, 2}  # Original set should remain unchanged

    @mutant
    def modify_tuple(t, index, value):
        lst = list(t)
        lst[index] = value
        return tuple(lst)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple, 1, 4)

    assert isinstance(result, tuple)
    assert result == (1, 4, 3)
    assert original_tuple == (1, 2, 3)  # Original tuple should remain unchanged

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
    assert original_data == {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}  # Original should remain unchanged


# LLM-generated content at query #16
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

    # Test with dict mutation
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
    def modify_kwargs(**kwargs):
        kwargs['x'] = 10
        return kwargs

    kwargs_result = modify_kwargs(a=1, b=2)
    assert isinstance(kwargs_result, PMap)
    assert kwargs_result == pmap({'a': 1, 'b': 2, 'x': 10})

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
    result_tuple = process_tuple(original_tuple)
    assert isinstance(result_tuple, tuple)
    assert result_tuple == (1, 2, 3, 4)
    assert original_tuple == (1, 2, 3)


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
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': [1, 2]}
    result = modify_dict(original_dict, 'a', [3, 4])
    assert original_dict == {'a': [1, 2]}  # Original should be unchanged
    assert result == pmap({'a': pvector([3, 4])})  # Result should be frozen

    # Test with multiple arguments
    @mutant
    def combine_and_modify(lst1, lst2):
        combined = lst1 + lst2
        combined.append(99)
        return combined

    list1 = [1, 2]
    list2 = [3, 4]
    result = combine_and_modify(list1, list2)
    assert list1 == [1, 2]  # Original should be unchanged
    assert list2 == [3, 4]  # Original should be unchanged
    assert result == pvector([1, 2, 3, 4, 99])  # Result should be frozen

    # Test with keyword arguments
    @mutant
    def modify_with_kwargs(data, **updates):
        data.update(updates)
        return data

    original = {'x': 1}
    result = modify_with_kwargs(original, y=2, z=3)
    assert original == {'x': 1}  # Original should be unchanged
    assert result == pmap({'x': 1, 'y': 2, 'z': 3})  # Result should be frozen

    # Test return value is frozen
    @mutant
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False (if needed)
    @mutant
    def no_strict_operation(data):
        return data

    mixed = {'a': PVector([1, 2]), 'b': [3, 4]}
    result = no_strict_operation(mixed)
    # With strict=True (default), nested lists should be frozen
    assert result == pmap({'a': pvector([1, 2]), 'b': pvector([3, 4])})


# LLM-generated content at query #18
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
    def modify_nested(d):
        d['a'][0] = 10
        return d

    original_dict = {'a': [1, 2], 'b': 3}
    result_dict = modify_nested(original_dict)
    assert original_dict == {'a': [1, 2], 'b': 3}  # Original unchanged
    assert result_dict == pmap({'a': pvector([10, 2]), 'b': 3})  # Nested structure frozen

    # Test with kwargs
    @mutant
    def modify_kwargs(a, b, **kwargs):
        kwargs['c'].append(5)
        return kwargs

    original_kwargs = {'c': [1, 2], 'd': 3}
    result_kwargs = modify_kwargs(1, 2, **original_kwargs)
    assert original_kwargs == {'c': [1, 2], 'd': 3}  # Original unchanged
    assert result_kwargs == pmap({'c': pvector([1, 2, 5]), 'd': 3})  # Kwargs frozen

    # Test with strict=False
    @mutant
    def no_strict_modify(lst):
        return lst

    result_no_strict = no_strict_modify([1, 2, 3])
    assert result_no_strict == pvector([1, 2, 3])  # Still frozen (decorator uses strict=True by default)

    # Test return value is frozen
    @mutant
    def return_new():
        return {'a': [1, 2]}

    result_new = return_new()
    assert isinstance(result_new, PMap)
    assert isinstance(result_new['a'], PVector)

    # Test with immutable input
    @mutant
    def immutable_input(x):
        return x + 1

    assert immutable_input(5) == 6  # Works with non-container types


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
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen

    # Test with nested structures
    @mutant
    def modify_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': [1, 2]}
    result = modify_dict(original_dict, 'b', [3, 4])
    assert original_dict == {'a': [1, 2]}  # Original unchanged
    assert result == pmap({'a': pvector([1, 2]), 'b': pvector([3, 4])})  # Return value is frozen

    # Test with kwargs
    @mutant
    def update_dict(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'key1': 'value1'}
    result = update_dict(**original_kwargs)
    assert original_kwargs == {'key1': 'value1'}  # Original unchanged
    assert result == pmap({'key1': 'value1', 'new_key': 'new_value'})  # Return value is frozen

    # Test with tuple (should remain tuple)
    @mutant
    def modify_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple)
    assert original_tuple == (1, 2, 3)  # Original unchanged
    assert result == (1, 2, 3, 4)  # Return value is frozen (but tuple remains tuple)

    # Test with set (should become pset)
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s

    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)
    assert original_set == {1, 2, 3}  # Original unchanged
    assert result == pset({1, 2, 3, 4})  # Return value is frozen


# LLM-generated content at query #20
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
    def nested_operations(data):
        data['list'].append(1)
        data['set'].add(2)
        return data

    # Test with list
    original_list = [1, 2, 3]
    result = add_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]  # Original unchanged

    # Test with dict
    original_dict = {'a': 1, 'b': 2}
    result = modify_dict(original_dict, 'c', 3)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'c': 3})
    assert original_dict == {'a': 1, 'b': 2}  # Original unchanged

    # Test with nested structures
    original_data = {'list': [1, 2], 'set': {3, 4}}
    result = nested_operations(original_data)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)
    assert result == pmap({'list': pvector([1, 2, 1]), 'set': pset({3, 4, 2})})
    assert original_data == {'list': [1, 2], 'set': {3, 4}}  # Original unchanged

    # Test with immutable input
    immutable_input = (1, 2, 3)
    @mutant
    def process_tuple(t):
        return list(t)

    result = process_tuple(immutable_input)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])
    assert immutable_input == (1, 2, 3)  # Original unchanged


# LLM-generated content at query #21
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

    # Test with dictionary
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
    def return_mutable():
        return [1, 2, 3]

    result = return_mutable()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test with strict=False
    @mutant
    def no_strict(data):
        return data

    mixed = [1, {'a': 2}, pvector([3, 4])]
    result = no_strict(mixed)
    assert result == pvector([1, pmap({'a': 2}), pvector([3, 4])])


# LLM-generated content at query #22
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
        data['nested']['value'] = 'changed'
        return data

    original_nested = {'list': [1, 2, 3], 'nested': {'value': 'original'}}
    result_nested = modify_nested(original_nested)
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['list'], PVector)
    assert isinstance(result_nested['nested'], PMap)
    assert result_nested['list'] == pvector([1, 2, 3, 4])
    assert result_nested['nested']['value'] == 'changed'
    assert original_nested == {'list': [1, 2, 3], 'nested': {'value': 'original'}}

    # Test with kwargs
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'a': 1, 'b': 2}
    result_kwargs = modify_kwargs(**original_kwargs)
    assert isinstance(result_kwargs, PMap)
    assert result_kwargs == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})
    assert original_kwargs == {'a': 1, 'b': 2}

    # Test return value is frozen
    @mutant
    def return_mutable():
        return {'a': [1, 2, 3]}

    result = return_mutable()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant():
    @mutant
    def example_func(arg1, arg2=None):
        if arg2 is None:
            arg2 = []
        arg2.append(arg1)
        return arg2

    # Test with mutable arguments
    input_list = [1, 2, 3]
    result = example_func(input_list)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, [1, 2, 3]])

    # Test with keyword arguments
    input_dict = {'key': 'value'}
    result = example_func(input_list, arg2=input_dict)
    assert isinstance(result, PVector)
    assert result == pvector([{'key': 'value'}, [1, 2, 3]])

    # Test that original arguments are not modified
    original_list = [1, 2, 3]
    original_dict = {'key': 'value'}
    example_func(original_list, arg2=original_dict)
    assert original_list == [1, 2, 3]
    assert original_dict == {'key': 'value'}

    # Test with nested structures
    nested_list = [1, {'a': [2, 3]}]
    result = example_func(nested_list)
    assert isinstance(result, PVector)
    assert result == pvector([1, {'a': [2, 3]}, [1, {'a': [2, 3]}]])

    # Test with pyrsistent input
    pvec = pvector([1, 2, 3])
    result = example_func(pvec)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, pvector([1, 2, 3])])


# LLM-generated content at query #24
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
    assert result == pvector([1, 2, 3, 4])  # Return value is frozen

    # Test with dictionary
    @mutant
    def update_dict(d, key, value):
        d[key] = value
        return d

    original_dict = {'a': 1}
    result = update_dict(original_dict, 'b', 2)
    assert original_dict == {'a': 1}  # Original unchanged
    assert result == pmap({'a': 1, 'b': 2})  # Return value is frozen

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

    original_kwargs = {'x': 1, 'y': 2}
    result = modify_kwargs(**original_kwargs)
    assert original_kwargs == {'x': 1, 'y': 2}  # Original unchanged
    assert result == pmap({'x': 10, 'y': 2})

    # Test with tuple (should remain tuple)
    @mutant
    def process_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result = process_tuple(original_tuple)
    assert original_tuple == (1, 2, 3)
    assert result == (1, 2, 3, 4)

    # Test with set (should become pset)
    @mutant
    def process_set(s):
        return s | {4}

    original_set = {1, 2, 3}
    result = process_set(original_set)
    assert original_set == {1, 2, 3}
    assert result == pset({1, 2, 3, 4})

    # Test that already frozen inputs remain frozen
    @mutant
    def process_frozen(pv):
        return pv.append(4)

    original_pvector = pvector([1, 2, 3])
    result = process_frozen(original_pvector)
    assert original_pvector == pvector([1, 2, 3])
    assert result == pvector([1, 2, 3, 4])


# LLM-generated content at query #25
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
    assert original_list == [1, 2, 3]  # Original unchanged

    # Test with multiple arguments
    @mutant
    def modify_dict_and_list(d, lst):
        d['new_key'] = 'new_value'
        lst.append(100)
        return d, lst

    original_dict = {'a': 1}
    original_list = [1, 2]
    dict_result, list_result = modify_dict_and_list(original_dict, original_list)
    assert isinstance(dict_result, PMap)
    assert isinstance(list_result, PVector)
    assert dict_result == pmap({'a': 1, 'new_key': 'new_value'})
    assert list_result == pvector([1, 2, 100])
    assert original_dict == {'a': 1}
    assert original_list == [1, 2]

    # Test with keyword arguments
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs

    original_kwargs = {'a': 1, 'b': 2}
    result = modify_kwargs(**original_kwargs)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2, 'new_key': 'new_value'})

    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['nested']['value'] = 10
        data['list'].append(20)
        return data

    original = {'nested': {'value': 5}, 'list': [1, 2]}
    result = modify_nested(original)
    assert isinstance(result, PMap)
    assert isinstance(result['nested'], PMap)
    assert isinstance(result['list'], PVector)
    assert result == pmap({'nested': pmap({'value': 10}), 'list': pvector([1, 2, 20])})
    assert original == {'nested': {'value': 5}, 'list': [1, 2]}

    # Test return value freezing
    @mutant
    def return_mutable():
        return {'a': [1, 2, 3]}

    result = return_mutable()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert result == pmap({'a': pvector([1, 2, 3])})

    # Test with no arguments
    @mutant
    def no_args():
        return [1, 2, 3]

    result = no_args()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #26
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

    # Test with immutable input
    @mutant
    def process_tuple(t):
        return t + (4,)

    original_tuple = (1, 2, 3)
    result = process_tuple(original_tuple)
    assert result == (1, 2, 3, 4)
    assert original_tuple == (1, 2, 3)  # Original unchanged


# LLM-generated content at query #27
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
    def add_to_set(s, item):
        s.add(item)
        return s

    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)

    assert isinstance(result, PSet)
    assert result == pset({1, 2, 3, 4})
    assert original_set == {1, 2, 3}  # Original set should remain unchanged

    @mutant
    def modify_tuple(t, index, value):
        lst = list(t)
        lst[index] = value
        return tuple(lst)

    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple, 1, 4)

    assert isinstance(result, tuple)
    assert result == (1, 4, 3)
    assert original_tuple == (1, 2, 3)  # Original tuple should remain unchanged


# LLM-generated content at query #28
#--------------------------

```python
def test_freeze():
    # Test freezing a list
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

    # Test freezing a nested list
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

    # Test freezing a dict
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

    # Test freezing a nested dict
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

    # Test freezing a defaultdict
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    assert freeze(dd) == pmap({'a': 1, 'b': 2})

    # Test freezing a set
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

    # Test freezing a tuple
    assert freeze((1, 2, 3)) == (1, 2, 3)

    # Test freezing a nested tuple
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

    # Test freezing a PVector
    pv = pvector([1, 2, 3])
    assert freeze(pv) == pv

    # Test freezing a PMap
    pm = pmap({'a': 1, 'b': 2})
    assert freeze(pm) == pm

    # Test freezing a PSet
    ps = pset({1, 2, 3})
    assert freeze(ps) == ps

    # Test freezing with strict=False
    assert freeze([1, {'a': 3}], strict=False) == pvector([1, {'a': 3}])
    assert freeze({'a': [1, 2]}, strict=False) == pmap({'a': [1, 2]})

    # Test freezing mixed types
    mixed = [1, {'a': 2}, (3, 4), {5, 6}]
    expected = pvector([1, pmap({'a': 2}), (3, 4), pset({5, 6})])
    assert freeze(mixed) == expected

    # Test freezing immutable types
    assert freeze(1) == 1
    assert freeze("hello") == "hello"
    assert freeze(None) is None


