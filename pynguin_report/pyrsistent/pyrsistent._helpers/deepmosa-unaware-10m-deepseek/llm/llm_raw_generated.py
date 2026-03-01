####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_mutant():
    # Test with simple mutation of list argument
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    original_list = [1, 2, 3]
    result = append_to_list(original_list, 4)
    
    # Original list should not be modified
    assert original_list == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with dict mutation
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    original_dict = {'a': 1, 'b': 2}
    result = add_to_dict(original_dict, 'c', 3)
    
    # Original dict should not be modified
    assert original_dict == {'a': 1, 'b': 2}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with set mutation
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s
    
    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)
    
    # Original set should not be modified
    assert original_set == {1, 2, 3}
    # Result should be frozen (pset)
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test with multiple arguments
    @mutant
    def combine_and_modify(lst1, lst2):
        combined = lst1 + lst2
        combined.append('mutated')
        return combined
    
    list1 = [1, 2]
    list2 = [3, 4]
    result = combine_and_modify(list1, list2)
    
    assert list1 == [1, 2]
    assert list2 == [3, 4]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4, 'mutated']
    
    # Test with keyword arguments
    @mutant
    def modify_with_kwargs(d, **kwargs):
        for k, v in kwargs.items():
            d[k] = v
        return d
    
    original = {'x': 10}
    result = modify_with_kwargs(original, y=20, z=30)
    
    assert original == {'x': 10}
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 10, 'y': 20, 'z': 30}
    
    # Test that nested mutations are isolated
    @mutant
    def deeply_nested_mutation(data):
        data['list'][0] = 'mutated'
        data['dict']['inner'] = 'changed'
        return data
    
    original_data = {
        'list': ['original', 'values'],
        'dict': {'key': 'value'}
    }
    result = deeply_nested_mutation(original_data)
    
    # Original should remain unchanged
    assert original_data == {
        'list': ['original', 'values'],
        'dict': {'key': 'value'}
    }
    
    # Result should be fully frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert list(result['list']) == ['mutated', 'values']
    assert dict(result['dict']) == {'key': 'value', 'inner': 'changed'}
    
    # Test with no mutation
    @mutant
    def no_mutation(x, y):
        return x + y
    
    result = no_mutation(5, 3)
    assert result == 8  # Should return regular int (not frozen)
    
    # Test with tuple (should remain tuple but frozen recursively)
    @mutant
    def modify_tuple(t):
        # Tuples are immutable, but we can test the decorator works
        return t + ('extra',)
    
    original_tuple = (1, 2, [3, 4])
    result = modify_tuple(original_tuple)
    
    assert original_tuple == (1, 2, [3, 4])
    assert isinstance(result, tuple)
    # The list inside should be frozen
    assert isinstance(result[2], PVector)
    assert list(result[2]) == [3, 4]
    assert result == (1, 2, pvector([3, 4]), 'extra')


# LLM-generated content at query #2
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with mixed mutable inputs
    def modify_nested(data, new_val):
        data['list'].append(new_val)
        data['dict']['inner'] = new_val
        return data
    
    decorated = mutant(modify_nested)
    original = {'list': [1, 2], 'dict': {'a': 1}}
    result = decorated(original, 3)
    
    # Original should not be modified
    assert original == {'list': [1, 2], 'dict': {'a': 1}}
    # Result should be fully frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert list(result['list']) == [1, 2, 3]
    assert dict(result['dict']) == {'a': 1, 'inner': 3}
    
    # Test with keyword arguments
    def update_with_kwargs(d, **kwargs):
        for k, v in kwargs.items():
            d[k] = v
        return d
    
    decorated = mutant(update_with_kwargs)
    original = {'x': 1}
    result = decorated(original, y=2, z=3)
    
    assert original == {'x': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 1, 'y': 2, 'z': 3}
    
    # Test with tuple input (should remain tuple but frozen recursively)
    def modify_tuple_content(tpl):
        # Tuples are immutable, but can contain mutable elements
        lst = list(tpl)
        if isinstance(lst[0], list):
            lst[0].append('modified')
        return tuple(lst)
    
    decorated = mutant(modify_tuple_content)
    original = ([1, 2], 3)
    result = decorated(original)
    
    # Original should not be modified
    assert original[0] == [1, 2]
    # Result should have frozen inner list
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert list(result[0]) == [1, 2, 'modified']
    assert result[1] == 3
    
    # Test with set input
    def add_to_set(s, element):
        s.add(element)
        return s
    
    decorated = mutant(add_to_set)
    original = {1, 2, 3}
    result = decorated(original, 4)
    
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test that function metadata is preserved
    def example_func(x, y=1):
        """Example docstring"""
        return x + y
    
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring'
    
    # Test with no mutation (pure function)
    def pure_add(a, b):
        return a + b
    
    decorated = mutant(pure_add)
    result = decorated(2, 3)
    assert result == 5  # Should return regular int (not frozen)
    
    # Test with already frozen input
    frozen_input = pvector([1, 2, 3])
    def identity(x):
        return x
    
    decorated = mutant(identity)
    result = decorated(frozen_input)
    # Should return the same frozen input
    assert result is frozen_input


# LLM-generated content at query #3
#--------------------------

```python
def test_mutant():
    # Test with mutable list argument
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict argument
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine(a, b):
        return a + b
    
    decorated = mutant(combine)
    result = decorated([1, 2], [3, 4])
    
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with keyword arguments
    def update_dict(**kwargs):
        d = {}
        d.update(kwargs)
        return d
    
    decorated = mutant(update_dict)
    result = decorated(x=1, y=2)
    
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 1, 'y': 2}
    
    # Test with mixed args and kwargs
    def mixed(a, b, **kwargs):
        return {'a': a, 'b': b, **kwargs}
    
    decorated = mutant(mixed)
    result = decorated([1], {'x': 2}, c=3)
    
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1])
    assert result['b'] == pmap({'x': 2})
    assert result['c'] == 3
    
    # Test that function name is preserved
    def my_func():
        pass
    
    decorated = mutant(my_func)
    assert decorated.__name__ == 'my_func'
    
    # Test with set argument
    def add_to_set(s, item):
        s.add(item)
        return s
    
    decorated = mutant(add_to_set)
    original = {1, 2}
    result = decorated(original, 3)
    
    assert original == {1, 2}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}
    
    # Test with tuple argument (should remain tuple)
    def wrap_tuple(t):
        return t
    
    decorated = mutant(wrap_tuple)
    original = (1, [2, 3])
    result = decorated(original)
    
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)
    assert result[0] == 1
    assert list(result[1]) == [2, 3]


# LLM-generated content at query #4
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be mutated
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be mutated
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine(a, b, c):
        return [a, b, c]
    
    decorated = mutant(combine)
    result = decorated([1], {'x': 2}, (3, 4))
    
    # All results should be frozen
    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], tuple)
    
    # Test with keyword arguments
    def process_kwargs(**kwargs):
        kwargs['processed'] = True
        return kwargs
    
    decorated = mutant(process_kwargs)
    result = decorated(x=1, y=2)
    
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 1, 'y': 2, 'processed': True}
    
    # Test that function name is preserved
    def example_func():
        pass
    
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    
    # Test with nested mutation
    def deeply_mutate(data):
        data['list'][0] = 'mutated'
        data['dict']['inner'] = 'changed'
        return data
    
    decorated = mutant(deeply_mutate)
    original = {
        'list': ['original'],
        'dict': {'key': 'value'}
    }
    result = decorated(original)
    
    # Original should remain unchanged
    assert original['list'][0] == 'original'
    assert original['dict'] == {'key': 'value'}
    
    # Result should be fully frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    
    # Test with set input
    def add_to_set(s, element):
        s.add(element)
        return s
    
    decorated = mutant(add_to_set)
    original = {1, 2, 3}
    result = decorated(original, 4)
    
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test with tuple input (should remain tuple)
    def tuple_processor(t):
        return t
    
    decorated = mutant(tuple_processor)
    original = ([1, 2], {'a': 3})
    result = decorated(original)
    
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)


# LLM-generated content at query #5
#--------------------------

```python
def test_mutant():
    # Test with list mutation
    @mutant
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    original_list = [1, 2, 3]
    result = append_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    assert original_list == [1, 2, 3]
    
    # Test with dict mutation
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    original_dict = {'a': 1, 'b': 2}
    result = add_to_dict(original_dict, 'c', 3)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3}
    assert original_dict == {'a': 1, 'b': 2}
    
    # Test with set mutation
    @mutant
    def add_to_set(s, value):
        s.add(value)
        return s
    
    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    assert original_set == {1, 2, 3}
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'][0] = 100
        data['dict']['new'] = 'value'
        return data
    
    original_nested = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = modify_nested(original_nested)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert list(result['list']) == [100, 2, 3]
    assert dict(result['dict']) == {'a': 1, 'new': 'value'}
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1}}
    
    # Test with multiple arguments
    @mutant
    def combine_and_modify(lst1, lst2, d):
        combined = lst1 + lst2
        d['combined'] = combined
        return d
    
    list1 = [1, 2]
    list2 = [3, 4]
    original_dict = {'initial': 'value'}
    result = combine_and_modify(list1, list2, original_dict)
    assert isinstance(result, PMap)
    assert dict(result) == {'initial': 'value', 'combined': [1, 2, 3, 4]}
    assert list1 == [1, 2]
    assert list2 == [3, 4]
    assert original_dict == {'initial': 'value'}
    
    # Test with keyword arguments
    @mutant
    def modify_with_kwargs(data, multiplier=1):
        return [x * multiplier for x in data]
    
    original_data = [1, 2, 3]
    result = modify_with_kwargs(original_data, multiplier=2)
    assert isinstance(result, PVector)
    assert list(result) == [2, 4, 6]
    assert original_data == [1, 2, 3]
    
    # Test that function metadata is preserved
    @mutant
    def documented_function(x):
        """This is a test function."""
        return x * 2
    
    assert documented_function.__name__ == 'documented_function'
    assert documented_function.__doc__ == "This is a test function."
    
    # Test with tuple (should remain tuple but frozen inside)
    @mutant
    def modify_tuple_data(t):
        # Tuples are immutable, but we can return a modified version
        return (t[0] * 2, t[1] + 1)
    
    original_tuple = (1, 2)
    result = modify_tuple_data(original_tuple)
    assert isinstance(result, tuple)
    assert result == (2, 3)
    assert original_tuple == (1, 2)
    
    # Test with already frozen structures
    @mutant
    def process_frozen(data):
        # Should handle already frozen structures gracefully
        return data
    
    frozen_vector = pvector([1, 2, 3])
    frozen_map = pmap({'a': 1})
    result1 = process_frozen(frozen_vector)
    result2 = process_frozen(frozen_map)
    assert result1 is frozen_vector
    assert result2 is frozen_map


# LLM-generated content at query #6
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    # Result should contain expected values
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    # Result should contain expected values
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple args and kwargs
    def complex_function(a, b, c=0, d=0):
        a.append(99)
        b['key'] = 'value'
        return [a, b, c + d]
    
    decorated = mutant(complex_function)
    list_arg = [1, 2]
    dict_arg = {'x': 10}
    result = decorated(list_arg, dict_arg, c=5, d=3)
    
    # Originals should not be modified
    assert list_arg == [1, 2]
    assert dict_arg == {'x': 10}
    # Result should be fully frozen
    assert isinstance(result, PVector)
    assert len(result) == 3
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert list(result[0]) == [1, 2, 99]
    assert dict(result[1]) == {'x': 10, 'key': 'value'}
    assert result[2] == 8
    
    # Test with nested structures
    def modify_nested(data):
        data['list'][0] = 'modified'
        data['tuple'][0].append('mutated')
        return data
    
    decorated = mutant(modify_nested)
    original = {
        'list': ['original', 2, 3],
        'tuple': ([1, 2], 'static')
    }
    result = decorated(original)
    
    # Original should not be modified
    assert original['list'][0] == 'original'
    assert original['tuple'][0] == [1, 2]
    # Result should be frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    # The tuple should remain a tuple (not converted)
    assert isinstance(result['tuple'], tuple)
    # The list inside tuple should be frozen
    assert isinstance(result['tuple'][0], PVector)
    
    # Test that return value is frozen even when input is already immutable
    def return_same(x):
        return x
    
    decorated = mutant(return_same)
    # Test with already immutable types
    assert decorated(1) == 1
    assert decorated("string") == "string"
    assert decorated((1, 2, 3)) == (1, 2, 3)
    
    # Test with set
    def add_to_set(s, element):
        s.add(element)
        return s
    
    decorated = mutant(add_to_set)
    original_set = {1, 2, 3}
    result = decorated(original_set, 4)
    
    # Original should not be modified
    assert original_set == {1, 2, 3}
    # Result should be frozen (pset)
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test function metadata preservation
    def documented_func(x):
        """Test function with docstring."""
        return x
    
    decorated = mutant(documented_func)
    assert decorated.__name__ == 'documented_func'
    assert decorated.__doc__ == """Test function with docstring."""


# LLM-generated content at query #7
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    # Result should contain expected values
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated_dict = mutant(add_to_dict)
    original_dict = {'a': 1}
    result_dict = decorated_dict(original_dict, 'b', 2)
    
    # Original should not be modified
    assert original_dict == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result_dict, PMap)
    # Result should contain expected values
    assert dict(result_dict) == {'a': 1, 'b': 2}
    
    # Test with nested mutable structures
    def modify_nested(data):
        data['list'].append(3)
        data['dict']['c'] = 3
        return data
    
    decorated_nested = mutant(modify_nested)
    original_nested = {'list': [1, 2], 'dict': {'a': 1, 'b': 2}}
    result_nested = decorated_nested(original_nested)
    
    # Original should not be modified
    assert original_nested == {'list': [1, 2], 'dict': {'a': 1, 'b': 2}}
    # Result should be fully frozen
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['list'], PVector)
    assert isinstance(result_nested['dict'], PMap)
    # Result should contain expected modifications
    assert list(result_nested['list']) == [1, 2, 3]
    assert dict(result_nested['dict']) == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with keyword arguments
    def kwarg_function(a, b=[]):
        b.append(a)
        return b
    
    decorated_kwarg = mutant(kwarg_function)
    result_kwarg = decorated_kwarg(1, b=[0])
    
    # Result should be frozen
    assert isinstance(result_kwarg, PVector)
    assert list(result_kwarg) == [0, 1]
    
    # Test with multiple arguments
    def multi_arg(a, b, c):
        a.append(1)
        b['key'] = 'value'
        c.add(4)
        return a, b, c
    
    decorated_multi = mutant(multi_arg)
    result_multi = decorated_multi([], {}, {1, 2, 3})
    
    # Result should be a tuple of frozen structures
    assert isinstance(result_multi, tuple)
    assert isinstance(result_multi[0], PVector)
    assert isinstance(result_multi[1], PMap)
    assert isinstance(result_multi[2], PSet)
    assert list(result_multi[0]) == [1]
    assert dict(result_multi[1]) == {'key': 'value'}
    assert set(result_multi[2]) == {1, 2, 3, 4}
    
    # Test that function metadata is preserved
    def documented_func():
        """A test function"""
        return []
    
    decorated_doc = mutant(documented_func)
    assert decorated_doc.__name__ == 'documented_func'
    assert decorated_doc.__doc__ == 'A test function'
    
    # Test with no mutation
    def no_mutation(x):
        return x * 2
    
    decorated_no_mut = mutant(no_mutation)
    result_no_mut = decorated_no_mut(5)
    assert result_no_mut == 10
    
    # Test with already frozen input
    frozen_input = pvector([1, 2, 3])
    result_frozen = decorated(frozen_input, 4)
    assert isinstance(result_frozen, PVector)
    assert list(result_frozen) == [1, 2, 3, 4]


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated_dict = mutant(add_to_dict)
    original_dict = {'a': 1}
    result_dict = decorated_dict(original_dict, 'b', 2)
    
    # Original should not be modified
    assert original_dict == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result_dict, PMap)
    assert dict(result_dict) == {'a': 1, 'b': 2}
    
    # Test with nested structures
    def modify_nested(data):
        data['list'][0] = 'modified'
        data['inner']['value'] = 'changed'
        return data
    
    decorated_nested = mutant(modify_nested)
    original_nested = {
        'list': ['original', 2, 3],
        'inner': {'value': 'original'}
    }
    result_nested = decorated_nested(original_nested)
    
    # Original should not be modified
    assert original_nested['list'][0] == 'original'
    assert original_nested['inner']['value'] == 'original'
    # Result should be frozen with modifications
    assert isinstance(result_nested, PMap)
    assert result_nested['list'][0] == 'modified'
    assert result_nested['inner']['value'] == 'changed'
    
    # Test with positional and keyword arguments
    def mixed_args(a, b, c=10):
        a.append(b)
        return {'a': a, 'b': b, 'c': c}
    
    decorated_mixed = mutant(mixed_args)
    original_list = [1, 2]
    result_mixed = decorated_mixed(original_list, 3, c=20)
    
    # Original should not be modified
    assert original_list == [1, 2]
    # Result should be frozen
    assert isinstance(result_mixed, PMap)
    assert dict(result_mixed) == {'a': [1, 2, 3], 'b': 3, 'c': 20}
    
    # Test that function metadata is preserved
    def sample_func(x, y=5):
        """Sample function docstring"""
        return x + y
    
    decorated_sample = mutant(sample_func)
    assert decorated_sample.__name__ == 'sample_func'
    assert decorated_sample.__doc__ == 'Sample function docstring'
    
    # Test with set input
    def add_to_set(s, item):
        s.add(item)
        return s
    
    decorated_set = mutant(add_to_set)
    original_set = {1, 2, 3}
    result_set = decorated_set(original_set, 4)
    
    # Original should not be modified
    assert original_set == {1, 2, 3}
    # Result should be frozen (pset)
    assert isinstance(result_set, PSet)
    assert set(result_set) == {1, 2, 3, 4}
    
    # Test with tuple input (should remain tuple)
    def modify_tuple(t):
        # Tuples are immutable, but we can return a modified version
        return t + ('added',)
    
    decorated_tuple = mutant(modify_tuple)
    original_tuple = (1, 2, 3)
    result_tuple = decorated_tuple(original_tuple)
    
    # Original should not be modified (tuples are immutable anyway)
    assert original_tuple == (1, 2, 3)
    # Result should be a tuple
    assert isinstance(result_tuple, tuple)
    assert result_tuple == (1, 2, 3, 'added')
    
    # Test with no mutation
    def no_mutation(x):
        return x
    
    decorated_no_mut = mutant(no_mutation)
    test_input = [1, 2, 3]
    result_no_mut = decorated_no_mut(test_input)
    
    # Original should not be modified
    assert test_input == [1, 2, 3]
    # Result should be frozen
    assert isinstance(result_no_mut, PVector)
    assert list(result_no_mut) == [1, 2, 3]


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant():
    # Test with simple mutable list mutation
    @mutant
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    original_list = [1, 2, 3]
    result = append_to_list(original_list, 4)
    
    # Original list should not be modified
    assert original_list == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with dict mutation
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    original_dict = {'a': 1, 'b': 2}
    result = add_to_dict(original_dict, 'c', 3)
    
    # Original dict should not be modified
    assert original_dict == {'a': 1, 'b': 2}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with nested mutation
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['new'] = 'value'
        return data
    
    original_nested = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result = modify_nested(original_nested)
    
    # Original should remain unchanged
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1}}
    # Result should be fully frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert list(result['list']) == [1, 2, 3, 4]
    assert dict(result['dict']) == {'a': 1, 'new': 'value'}
    
    # Test with multiple arguments and keyword arguments
    @mutant
    def complex_operation(a, b, c=None):
        a.append(99)
        b['key'] = 'value'
        if c is not None:
            c.add(100)
        return a, b, c
    
    list_arg = [1, 2]
    dict_arg = {'x': 1}
    set_arg = {10, 20}
    
    result = complex_operation(list_arg, dict_arg, c=set_arg)
    
    # Originals unchanged
    assert list_arg == [1, 2]
    assert dict_arg == {'x': 1}
    assert set_arg == {10, 20}
    
    # Results frozen
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)
    assert list(result[0]) == [1, 2, 99]
    assert dict(result[1]) == {'x': 1, 'key': 'value'}
    assert set(result[2]) == {10, 20, 100}
    
    # Test with no mutation
    @mutant
    def no_mutation(x, y):
        return x + y
    
    result = no_mutation(5, 3)
    assert result == 8  # Should return regular int, not frozen
    
    # Test with function metadata preservation
    @mutant
    def documented_func(x):
        """A documented function."""
        return x
    
    assert documented_func.__name__ == 'documented_func'
    assert documented_func.__doc__ == "A documented function."
    
    # Test with empty arguments
    @mutant
    def empty_args():
        return {'empty': []}
    
    result = empty_args()
    assert isinstance(result, PMap)
    assert isinstance(result['empty'], PVector)
    assert list(result['empty']) == []


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original_list = [1, 2, 3]
    result = decorated(original_list, 4)
    
    # Original list should not be modified
    assert original_list == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original_dict = {'a': 1}
    result = decorated(original_dict, 'b', 2)
    
    # Original dict should not be modified
    assert original_dict == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple args and kwargs
    def complex_function(a, b, c=10):
        a.append(b)
        return {'a': a, 'b': b, 'c': c}
    
    decorated = mutant(complex_function)
    original_list = [1, 2]
    result = decorated(original_list, 3, c=20)
    
    # Original list should not be modified
    assert original_list == [1, 2]
    # Result should be frozen structure
    assert isinstance(result, PMap)
    assert dict(result) == {'a': [1, 2, 3], 'b': 3, 'c': 20}
    assert isinstance(result['a'], PVector)
    
    # Test with set input
    def add_to_set(s, value):
        s.add(value)
        return s
    
    decorated = mutant(add_to_set)
    original_set = {1, 2, 3}
    result = decorated(original_set, 4)
    
    # Original set should not be modified
    assert original_set == {1, 2, 3}
    # Result should be frozen (pset)
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test with tuple input (should remain tuple)
    def modify_tuple(t, value):
        return t + (value,)
    
    decorated = mutant(modify_tuple)
    original_tuple = (1, 2)
    result = decorated(original_tuple, 3)
    
    # Original tuple unchanged (tuples are immutable anyway)
    assert original_tuple == (1, 2)
    # Result should be a regular tuple
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)
    
    # Test that function name is preserved
    def sample_function():
        pass
    
    decorated = mutant(sample_function)
    assert decorated.__name__ == 'sample_function'
    
    # Test with nested mutable structures
    def modify_nested(data):
        data['list'].append(99)
        data['dict']['new'] = 'value'
        return data
    
    decorated = mutant(modify_nested)
    original_data = {'list': [1, 2], 'dict': {'a': 1}}
    result = decorated(original_data)
    
    # Original data should not be modified
    assert original_data == {'list': [1, 2], 'dict': {'a': 1}}
    # Result should be fully frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert list(result['list']) == [1, 2, 99]
    assert dict(result['dict']) == {'a': 1, 'new': 'value'}


# LLM-generated content at query #11
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple args and kwargs
    def complex_mutation(a, b, c=None):
        a.append(99)
        b['key'] = 'value'
        if c is not None:
            c.add(100)
        return a, b, c
    
    decorated = mutant(complex_mutation)
    list_arg = [1, 2]
    dict_arg = {'x': 1}
    set_arg = {10, 20}
    
    result = decorated(list_arg, dict_arg, c=set_arg)
    
    # Originals should not be modified
    assert list_arg == [1, 2]
    assert dict_arg == {'x': 1}
    assert set_arg == {10, 20}
    
    # Results should be frozen
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)
    
    # Test with no mutation
    def no_mutation(x, y):
        return x + y
    
    decorated = mutant(no_mutation)
    result = decorated(10, 20)
    assert result == 30  # Should return regular int, not frozen
    
    # Test with nested structures
    def modify_nested(data):
        data['list'][0] = 'modified'
        data['dict']['inner'] = 'changed'
        return data
    
    decorated = mutant(modify_nested)
    original = {
        'list': ['original', 2, 3],
        'dict': {'a': 1}
    }
    result = decorated(original)
    
    # Original should not be modified
    assert original['list'][0] == 'original'
    assert 'inner' not in original['dict']
    
    # Result should be fully frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    
    # Test function metadata preservation
    def sample_func(x, y=5):
        """Sample docstring"""
        return x + y
    
    decorated = mutant(sample_func)
    assert decorated.__name__ == 'sample_func'
    assert decorated.__doc__ == 'Sample docstring'
    
    # Test with tuple input (should remain tuple)
    def process_tuple(t):
        return t
    
    decorated = mutant(process_tuple)
    original = ([1, 2], {'a': 3})
    result = decorated(original)
    
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)


# LLM-generated content at query #12
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated_dict = mutant(add_to_dict)
    original_dict = {'a': 1}
    result_dict = decorated_dict(original_dict, 'b', 2)
    
    # Original should not be modified
    assert original_dict == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result_dict, PMap)
    assert dict(result_dict) == {'a': 1, 'b': 2}
    
    # Test with nested structures
    def modify_nested(data):
        data['list'][0] = 'modified'
        data['dict']['new'] = 'value'
        return data
    
    decorated_nested = mutant(modify_nested)
    original_nested = {
        'list': ['original'],
        'dict': {'key': 'old'}
    }
    result_nested = decorated_nested(original_nested)
    
    # Original should not be modified
    assert original_nested == {'list': ['original'], 'dict': {'key': 'old'}}
    # Result should be fully frozen
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['list'], PVector)
    assert isinstance(result_nested['dict'], PMap)
    assert result_nested['list'][0] == 'modified'
    assert result_nested['dict']['new'] == 'value'
    
    # Test with positional and keyword arguments
    def mixed_args(a, b, c=3):
        a.append(b)
        return {'a': a, 'b': b, 'c': c}
    
    decorated_mixed = mutant(mixed_args)
    original_list = [1]
    result_mixed = decorated_mixed(original_list, 2, c=4)
    
    assert original_list == [1]
    assert isinstance(result_mixed, PMap)
    assert dict(result_mixed) == {'a': [1, 2], 'b': 2, 'c': 4}
    
    # Test that function metadata is preserved
    def sample_func(x, y):
        """Sample function docstring"""
        return x + [y]
    
    decorated_sample = mutant(sample_func)
    assert decorated_sample.__name__ == 'sample_func'
    assert decorated_sample.__doc__ == 'Sample function docstring'
    
    # Test with set input
    def add_to_set(s, item):
        s.add(item)
        return s
    
    decorated_set = mutant(add_to_set)
    original_set = {1, 2}
    result_set = decorated_set(original_set, 3)
    
    assert original_set == {1, 2}
    assert isinstance(result_set, PSet)
    assert set(result_set) == {1, 2, 3}
    
    # Test with tuple input (should remain tuple)
    def process_tuple(t):
        return t + ('extra',)
    
    decorated_tuple = mutant(process_tuple)
    original_tuple = (1, 2)
    result_tuple = decorated_tuple(original_tuple)
    
    assert original_tuple == (1, 2)
    assert isinstance(result_tuple, tuple)
    assert result_tuple == (1, 2, 'extra')


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    # Result should contain the expected values
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated_dict = mutant(add_to_dict)
    original_dict = {'a': 1}
    result_dict = decorated_dict(original_dict, 'b', 2)
    
    # Original should not be modified
    assert original_dict == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result_dict, PMap)
    # Result should contain the expected values
    assert dict(result_dict) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine_and_modify(a, b):
        combined = a + b
        combined.append('modified')
        return combined
    
    decorated_combine = mutant(combine_and_modify)
    list1 = [1, 2]
    list2 = [3, 4]
    result_combined = decorated_combine(list1, list2)
    
    # Originals should not be modified
    assert list1 == [1, 2]
    assert list2 == [3, 4]
    # Result should be frozen
    assert isinstance(result_combined, PVector)
    # Result should contain the expected values
    assert list(result_combined) == [1, 2, 3, 4, 'modified']
    
    # Test with keyword arguments
    def update_with_kwargs(d, **kwargs):
        for k, v in kwargs.items():
            d[k] = v
        return d
    
    decorated_kwargs = mutant(update_with_kwargs)
    original_kwargs = {'x': 10}
    result_kwargs = decorated_kwargs(original_kwargs, y=20, z=30)
    
    # Original should not be modified
    assert original_kwargs == {'x': 10}
    # Result should be frozen
    assert isinstance(result_kwargs, PMap)
    # Result should contain the expected values
    assert dict(result_kwargs) == {'x': 10, 'y': 20, 'z': 30}
    
    # Test that nested mutations are isolated
    def deeply_nested_modification(data):
        data['list'][0] = 'mutated'
        data['inner']['value'] = 'changed'
        return data
    
    decorated_nested = mutant(deeply_nested_modification)
    original_nested = {
        'list': [[1, 2], [3, 4]],
        'inner': {'value': 'original'}
    }
    result_nested = decorated_nested(original_nested)
    
    # Original should not be modified
    assert original_nested['list'][0] == [1, 2]
    assert original_nested['inner']['value'] == 'original'
    # Result should be fully frozen
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['list'], PVector)
    assert isinstance(result_nested['list'][0], PVector)
    assert isinstance(result_nested['inner'], PMap)
    # Result should contain the modified values
    assert list(result_nested['list'][0]) == ['mutated', 2]
    assert result_nested['inner']['value'] == 'changed'
    
    # Test with no mutation (should still freeze)
    def no_mutation(x, y):
        return x + y
    
    decorated_no_mut = mutant(no_mutation)
    result_sum = decorated_no_mut([1], [2, 3])
    
    # Result should be frozen even though no mutation occurred
    assert isinstance(result_sum, PVector)
    assert list(result_sum) == [1, 2, 3]
    
    # Test function metadata preservation
    def example_func(a, b=1):
        """Example function docstring."""
        return a + [b]
    
    decorated_meta = mutant(example_func)
    
    # Should preserve function name
    assert decorated_meta.__name__ == 'example_func'
    # Should preserve docstring
    assert decorated_meta.__doc__ == "Example function docstring."


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple args and kwargs
    def complex_mutation(a, b, c=None):
        a.append(1)
        b['key'] = 'value'
        if c is not None:
            c.add(4)
        return a, b, c
    
    decorated = mutant(complex_mutation)
    list_arg = [1, 2]
    dict_arg = {'x': 1}
    set_arg = {1, 2, 3}
    
    result = decorated(list_arg, dict_arg, c=set_arg)
    
    # Originals should not be modified
    assert list_arg == [1, 2]
    assert dict_arg == {'x': 1}
    assert set_arg == {1, 2, 3}
    
    # Results should be frozen
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)
    
    # Test that function name is preserved
    def sample_func():
        pass
    
    decorated = mutant(sample_func)
    assert decorated.__name__ == 'sample_func'
    
    # Test with no mutation
    def no_mutation(x, y):
        return x + y
    
    decorated = mutant(no_mutation)
    result = decorated(1, 2)
    assert result == 3
    
    # Test with nested structures
    def modify_nested(data):
        data['list'][0] = 99
        data['inner']['key'] = 'modified'
        return data
    
    decorated = mutant(modify_nested)
    original = {
        'list': [1, 2, 3],
        'inner': {'key': 'original'}
    }
    result = decorated(original)
    
    # Original should not be modified
    assert original['list'] == [1, 2, 3]
    assert original['inner']['key'] == 'original'
    
    # Result should be fully frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['inner'], PMap)


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original_list = [1, 2, 3]
    result = decorated(original_list, 4)
    
    # Original list should not be modified
    assert original_list == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated_dict = mutant(add_to_dict)
    original_dict = {'a': 1}
    dict_result = decorated_dict(original_dict, 'b', 2)
    
    # Original dict should not be modified
    assert original_dict == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(dict_result, PMap)
    assert dict(dict_result) == {'a': 1, 'b': 2}
    
    # Test with multiple args and kwargs
    def complex_function(a, b, c=10):
        a.append(b)
        return {'a': a, 'b': b, 'c': c}
    
    decorated_complex = mutant(complex_function)
    list_arg = [1, 2]
    result = decorated_complex(list_arg, 3, c=20)
    
    # Original list should not be modified
    assert list_arg == [1, 2]
    # Result should be frozen structure
    assert isinstance(result, PMap)
    assert dict(result) == {'a': [1, 2, 3], 'b': 3, 'c': 20}
    assert isinstance(result['a'], PVector)
    
    # Test with nested mutable structures
    def modify_nested(data):
        data['list'].append('new')
        data['inner']['key'] = 'changed'
        return data
    
    decorated_nested = mutant(modify_nested)
    nested_input = {
        'list': [1, 2, 3],
        'inner': {'key': 'value'}
    }
    nested_result = decorated_nested(nested_input)
    
    # Original should not be modified
    assert nested_input == {'list': [1, 2, 3], 'inner': {'key': 'value'}}
    # Result should be fully frozen
    assert isinstance(nested_result, PMap)
    assert isinstance(nested_result['list'], PVector)
    assert isinstance(nested_result['inner'], PMap)
    assert dict(nested_result) == {
        'list': [1, 2, 3, 'new'],
        'inner': {'key': 'changed'}
    }
    
    # Test that function name is preserved
    def sample_function():
        pass
    
    decorated_sample = mutant(sample_function)
    assert decorated_sample.__name__ == 'sample_function'
    
    # Test with tuple input (should remain immutable)
    def process_tuple(t):
        # Tuples are immutable anyway, but let's verify behavior
        return t + (4,)
    
    decorated_tuple = mutant(process_tuple)
    tuple_input = (1, 2, 3)
    tuple_result = decorated_tuple(tuple_input)
    
    # Should return a tuple (not frozen to pvector)
    assert isinstance(tuple_result, tuple)
    assert tuple_result == (1, 2, 3, 4)
    
    # Test with set input
    def add_to_set(s, element):
        return s | {element}
    
    decorated_set = mutant(add_to_set)
    set_input = {1, 2, 3}
    set_result = decorated_set(set_input, 4)
    
    # Original set unchanged
    assert set_input == {1, 2, 3}
    # Result should be pset
    assert isinstance(set_result, PSet)
    assert set(set_result) == {1, 2, 3, 4}


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant():
    # Test with mutable list argument
    def append_one(lst):
        lst.append(1)
        return lst
    
    decorated = mutant(append_one)
    original = [2, 3]
    result = decorated(original)
    
    assert result == pvector([2, 3, 1])
    assert original == [2, 3]
    
    # Test with mutable dict argument
    def add_key(d):
        d['new'] = 'value'
        return d
    
    decorated = mutant(add_key)
    original = {'a': 1}
    result = decorated(original)
    
    assert result == pmap({'a': 1, 'new': 'value'})
    assert original == {'a': 1}
    
    # Test with multiple arguments
    def combine(a, b):
        a.extend(b)
        return a
    
    decorated = mutant(combine)
    result = decorated([1, 2], [3, 4])
    
    assert result == pvector([1, 2, 3, 4])
    
    # Test with keyword arguments
    def update_dict(**kwargs):
        kwargs['extra'] = True
        return kwargs
    
    decorated = mutant(update_dict)
    result = decorated(x=1, y=2)
    
    assert result == pmap({'x': 1, 'y': 2, 'extra': True})
    
    # Test that mutation inside function doesn't affect frozen args
    mutable_list = []
    
    def capture_reference(lst):
        mutable_list.append(lst)
        lst.append('mutated')
        return lst
    
    decorated = mutant(capture_reference)
    original = ['original']
    result = decorated(original)
    
    assert result == pvector(['original', 'mutated'])
    assert original == ['original']
    assert len(mutable_list) == 1
    assert isinstance(mutable_list[0], PVector)
    
    # Test with nested structures
    def modify_nested(data):
        data['list'][0] = 'changed'
        data['dict']['inner'] = 'modified'
        return data
    
    decorated = mutant(modify_nested)
    original = {
        'list': ['original'],
        'dict': {'key': 'value'}
    }
    result = decorated(original)
    
    expected = pmap({
        'list': pvector(['changed']),
        'dict': pmap({'key': 'value', 'inner': 'modified'})
    })
    assert result == expected
    assert original == {'list': ['original'], 'dict': {'key': 'value'}}
    
    # Test return value is frozen
    def return_mutable():
        return {'a': [1, 2, 3]}
    
    decorated = mutant(return_mutable)
    result = decorated()
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    
    # Test with set arguments
    def add_to_set(s):
        s.add(99)
        return s
    
    decorated = mutant(add_to_set)
    original = {1, 2, 3}
    result = decorated(original)
    
    assert result == pset({1, 2, 3, 99})
    assert original == {1, 2, 3}
    
    # Test with tuple arguments (should remain tuples)
    def modify_tuple(t):
        # Tuples are immutable, but can contain mutable elements
        return (list(t[0]), t[1])
    
    decorated = mutant(modify_tuple)
    result = decorated(([1, 2], 'hello'))
    
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert result[0] == pvector([1, 2])
    assert result[1] == 'hello'


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated_dict = mutant(add_to_dict)
    original_dict = {'a': 1}
    result_dict = decorated_dict(original_dict, 'b', 2)
    
    # Original should not be modified
    assert original_dict == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result_dict, PMap)
    assert dict(result_dict) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine_and_modify(lst1, lst2):
        combined = lst1 + lst2
        combined.append('mutated')
        return combined
    
    decorated_combine = mutant(combine_and_modify)
    list1 = [1, 2]
    list2 = [3, 4]
    result_combined = decorated_combine(list1, list2)
    
    # Originals should not be modified
    assert list1 == [1, 2]
    assert list2 == [3, 4]
    # Result should be frozen
    assert isinstance(result_combined, PVector)
    assert list(result_combined) == [1, 2, 3, 4, 'mutated']
    
    # Test with keyword arguments
    def update_with_kwargs(d, **kwargs):
        for k, v in kwargs.items():
            d[k] = v
        return d
    
    decorated_kwargs = mutant(update_with_kwargs)
    original_kw = {'x': 10}
    result_kw = decorated_kwargs(original_kw, y=20, z=30)
    
    # Original should not be modified
    assert original_kw == {'x': 10}
    # Result should be frozen
    assert isinstance(result_kw, PMap)
    assert dict(result_kw) == {'x': 10, 'y': 20, 'z': 30}
    
    # Test that function name is preserved
    def sample_function():
        pass
    
    decorated_sample = mutant(sample_function)
    assert decorated_sample.__name__ == 'sample_function'
    
    # Test with nested mutable structures
    def modify_nested(data):
        data['list'][0] = 'modified'
        data['inner']['key'] = 'changed'
        return data
    
    decorated_nested = mutant(modify_nested)
    original_nested = {
        'list': [1, 2, 3],
        'inner': {'key': 'value'}
    }
    result_nested = decorated_nested(original_nested)
    
    # Original should not be modified
    assert original_nested == {
        'list': [1, 2, 3],
        'inner': {'key': 'value'}
    }
    # Result should be fully frozen
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['list'], PVector)
    assert isinstance(result_nested['inner'], PMap)
    assert list(result_nested['list']) == ['modified', 2, 3]
    assert dict(result_nested['inner']) == {'key': 'changed'}
    
    # Test with no mutation (pure function)
    def pure_function(x, y):
        return x + y
    
    decorated_pure = mutant(pure_function)
    result_pure = decorated_pure(5, 3)
    # Should still work and return frozen result if applicable
    assert result_pure == 8
    
    # Test with set input
    def add_to_set(s, item):
        s.add(item)
        return s
    
    decorated_set = mutant(add_to_set)
    original_set = {1, 2, 3}
    result_set = decorated_set(original_set, 4)
    
    # Original should not be modified
    assert original_set == {1, 2, 3}
    # Result should be frozen (pset)
    assert isinstance(result_set, PSet)
    assert set(result_set) == {1, 2, 3, 4}


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    # Result should contain the expected values
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated_dict = mutant(add_to_dict)
    original_dict = {'a': 1}
    result_dict = decorated_dict(original_dict, 'b', 2)
    
    # Original should not be modified
    assert original_dict == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result_dict, PMap)
    # Result should contain the expected values
    assert dict(result_dict) == {'a': 1, 'b': 2}
    
    # Test with nested mutable structures
    def modify_nested(data):
        data['list'].append(3)
        data['dict']['c'] = 3
        return data
    
    decorated_nested = mutant(modify_nested)
    original_nested = {'list': [1, 2], 'dict': {'a': 1, 'b': 2}}
    result_nested = decorated_nested(original_nested)
    
    # Original should not be modified
    assert original_nested == {'list': [1, 2], 'dict': {'a': 1, 'b': 2}}
    # Result should be fully frozen
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['list'], PVector)
    assert isinstance(result_nested['dict'], PMap)
    # Result should contain the expected modifications
    assert list(result_nested['list']) == [1, 2, 3]
    assert dict(result_nested['dict']) == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with keyword arguments
    def kwarg_function(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs
    
    decorated_kwargs = mutant(kwarg_function)
    result_kwargs = decorated_kwargs(a=1, b=2)
    
    assert isinstance(result_kwargs, PMap)
    assert dict(result_kwargs) == {'a': 1, 'b': 2, 'new_key': 'new_value'}
    
    # Test with positional and keyword arguments
    def mixed_function(lst, d, value):
        lst.append(value)
        d['added'] = value
        return {'list': lst, 'dict': d}
    
    decorated_mixed = mutant(mixed_function)
    result_mixed = decorated_mixed([1], {'a': 1}, 2)
    
    assert isinstance(result_mixed, PMap)
    assert isinstance(result_mixed['list'], PVector)
    assert isinstance(result_mixed['dict'], PMap)
    assert list(result_mixed['list']) == [1, 2]
    assert dict(result_mixed['dict']) == {'a': 1, 'added': 2}
    
    # Test that function metadata is preserved
    def documented_function():
        """A test function."""
        return []
    
    decorated_doc = mutant(documented_function)
    assert decorated_doc.__name__ == 'documented_function'
    assert decorated_doc.__doc__ == """A test function."""
    
    # Test with immutable input (should remain unchanged)
    def identity(x):
        return x
    
    decorated_identity = mutant(identity)
    result_immutable = decorated_identity(42)
    assert result_immutable == 42
    
    # Test with tuple input (should be recursively frozen)
    def modify_tuple(t):
        # Tuples are immutable, but can contain mutable elements
        return t
    
    decorated_tuple = mutant(modify_tuple)
    result_tuple = decorated_tuple(([1, 2], {'a': 1}))
    
    assert isinstance(result_tuple, tuple)
    assert isinstance(result_tuple[0], PVector)
    assert isinstance(result_tuple[1], PMap)


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant():
    # Test with list mutation
    @mutant
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    original_list = [1, 2, 3]
    result = append_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original should not be modified
    assert isinstance(result, PVector)  # Result should be frozen
    assert list(result) == [1, 2, 3, 4]
    
    # Test with dict mutation
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    original_dict = {'a': 1, 'b': 2}
    result = add_to_dict(original_dict, 'c', 3)
    assert original_dict == {'a': 1, 'b': 2}  # Original should not be modified
    assert isinstance(result, PMap)  # Result should be frozen
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with set mutation
    @mutant
    def add_to_set(s, value):
        s.add(value)
        return s
    
    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)
    assert original_set == {1, 2, 3}  # Original should not be modified
    assert isinstance(result, PSet)  # Result should be frozen
    assert set(result) == {1, 2, 3, 4}
    
    # Test with multiple arguments
    @mutant
    def combine_and_modify(lst, d, s):
        lst.append('modified')
        d['new'] = 'value'
        s.add(99)
        return lst, d, s
    
    lst_arg = [1, 2]
    dict_arg = {'x': 10}
    set_arg = {5, 6}
    
    result = combine_and_modify(lst_arg, dict_arg, set_arg)
    
    # Originals unchanged
    assert lst_arg == [1, 2]
    assert dict_arg == {'x': 10}
    assert set_arg == {5, 6}
    
    # Results are frozen
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)
    
    # Test with keyword arguments
    @mutant
    def modify_with_kwargs(data, increment=1):
        if isinstance(data, dict):
            data['value'] += increment
        return data
    
    original_data = {'value': 5}
    result = modify_with_kwargs(original_data, increment=3)
    assert original_data == {'value': 5}
    assert isinstance(result, PMap)
    assert dict(result) == {'value': 8}
    
    # Test that function metadata is preserved
    @mutant
    def example_func(x, y=1):
        """Example function docstring."""
        return x + y
    
    assert example_func.__name__ == 'example_func'
    assert example_func.__doc__ == 'Example function docstring.'
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'][0] = 'changed'
        data['dict']['inner'] = 'modified'
        return data
    
    original_nested = {
        'list': ['original', 2, 3],
        'dict': {'key': 'value'}
    }
    
    result = modify_nested(original_nested)
    assert original_nested == {'list': ['original', 2, 3], 'dict': {'key': 'value'}}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    
    # Test with no mutation
    @mutant
    def pure_function(x, y):
        return x * y
    
    result = pure_function(3, 4)
    assert result == 12  # Should return regular int, not frozen
    
    # Test with tuple (should remain tuple but frozen recursively)
    @mutant
    def modify_in_tuple(t):
        # Tuples are immutable, but we can test they're handled correctly
        return (list(t[0]), t[1])
    
    original_tuple = ([1, 2], 3)
    result = modify_in_tuple(original_tuple)
    assert original_tuple == ([1, 2], 3)
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple args and kwargs
    def complex_function(a, b, c=10):
        a.append(b)
        return {'a': a, 'b': b, 'c': c}
    
    decorated = mutant(complex_function)
    original_list = [1, 2]
    result = decorated(original_list, 3, c=20)
    
    # Original should not be modified
    assert original_list == [1, 2]
    # Result should be frozen structure
    assert isinstance(result, PMap)
    assert dict(result) == {'a': pvector([1, 2, 3]), 'b': 3, 'c': 20}
    
    # Test with nested mutation
    def nested_mutation(data):
        data['list'].append('new')
        data['dict']['inner'] = 'modified'
        return data
    
    decorated = mutant(nested_mutation)
    original = {'list': [1, 2], 'dict': {'a': 1}}
    result = decorated(original)
    
    # Original should not be modified
    assert original == {'list': [1, 2], 'dict': {'a': 1}}
    # Result should be completely frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert list(result['list']) == [1, 2, 'new']
    assert dict(result['dict']) == {'a': 1, 'inner': 'modified'}
    
    # Test that function metadata is preserved
    def documented_func():
        """Test function docstring"""
        return []
    
    decorated = mutant(documented_func)
    assert decorated.__name__ == 'documented_func'
    assert decorated.__doc__ == "Test function docstring"
    
    # Test with no mutation (pure function)
    def pure_func(x, y):
        return x + y
    
    decorated = mutant(pure_func)
    result = decorated(10, 20)
    assert result == 30
    
    # Test with set input
    def modify_set(s, item):
        s.add(item)
        return s
    
    decorated = mutant(modify_set)
    original = {1, 2, 3}
    result = decorated(original, 4)
    
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test with tuple input (should remain tuple)
    def tuple_func(t):
        return t
    
    decorated = mutant(tuple_func)
    original = ([1, 2], {'a': 3})
    result = decorated(original)
    
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)


# LLM-generated content at query #2
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated_dict = mutant(add_to_dict)
    original_dict = {'a': 1}
    result_dict = decorated_dict(original_dict, 'b', 2)
    
    # Original should not be modified
    assert original_dict == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result_dict, PMap)
    assert dict(result_dict) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine(a, b):
        return a + b
    
    decorated_combine = mutant(combine)
    result_combine = decorated_combine([1, 2], [3, 4])
    
    assert isinstance(result_combine, PVector)
    assert list(result_combine) == [1, 2, 3, 4]
    
    # Test with keyword arguments
    def merge_dicts(**kwargs):
        result = {}
        for k, v in kwargs.items():
            result[k] = v
        return result
    
    decorated_merge = mutant(merge_dicts)
    result_merge = decorated_merge(x=1, y=2)
    
    assert isinstance(result_merge, PMap)
    assert dict(result_merge) == {'x': 1, 'y': 2}
    
    # Test that function name is preserved
    def my_function():
        pass
    
    decorated_name = mutant(my_function)
    assert decorated_name.__name__ == 'my_function'
    
    # Test with nested mutable structures
    def modify_nested(data):
        data['list'][0] = 'modified'
        return data
    
    decorated_nested = mutant(modify_nested)
    original_nested = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result_nested = decorated_nested(original_nested)
    
    # Original should not be modified
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1}}
    # Result should be fully frozen
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['list'], PVector)
    assert isinstance(result_nested['dict'], PMap)
    assert list(result_nested['list']) == ['modified', 2, 3]
    
    # Test with set input
    def add_to_set(s, element):
        s.add(element)
        return s
    
    decorated_set = mutant(add_to_set)
    original_set = {1, 2, 3}
    result_set = decorated_set(original_set, 4)
    
    # Original should not be modified
    assert original_set == {1, 2, 3}
    # Result should be frozen (pset)
    assert isinstance(result_set, PSet)
    assert set(result_set) == {1, 2, 3, 4}
    
    # Test with tuple input (should remain tuple)
    def tuple_identity(t):
        return t
    
    decorated_tuple = mutant(tuple_identity)
    original_tuple = ([1, 2], {'a': 3})
    result_tuple = decorated_tuple(original_tuple)
    
    # Should return frozen version with pvector and pmap inside tuple
    assert isinstance(result_tuple, tuple)
    assert isinstance(result_tuple[0], PVector)
    assert isinstance(result_tuple[1], PMap)
    assert list(result_tuple[0]) == [1, 2]
    assert dict(result_tuple[1]) == {'a': 3}


# LLM-generated content at query #3
#--------------------------

```python
def test_mutant():
    # Test with list mutation
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    original_list = [1, 2, 3]
    result = append_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    assert original_list == [1, 2, 3]
    
    # Test with dict mutation
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    original_dict = {'a': 1, 'b': 2}
    result = add_to_dict(original_dict, 'c', 3)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3}
    assert original_dict == {'a': 1, 'b': 2}
    
    # Test with set mutation
    @mutant
    def add_to_set(s, item):
        s.add(item)
        return s
    
    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    assert original_set == {1, 2, 3}
    
    # Test with tuple (should remain tuple)
    @mutant
    def process_tuple(t):
        return t + (4,)
    
    original_tuple = (1, 2, 3)
    result = process_tuple(original_tuple)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)
    assert original_tuple == (1, 2, 3)
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['c'] = 3
        return data
    
    original_nested = {
        'list': [1, 2, 3],
        'dict': {'a': 1, 'b': 2}
    }
    result = modify_nested(original_nested)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert list(result['list']) == [1, 2, 3, 4]
    assert dict(result['dict']) == {'a': 1, 'b': 2, 'c': 3}
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    
    # Test with multiple arguments
    @mutant
    def combine_and_modify(lst1, lst2, d):
        combined = lst1 + lst2
        d['combined'] = combined
        return d
    
    list1 = [1, 2]
    list2 = [3, 4]
    original_dict = {'a': 1}
    result = combine_and_modify(list1, list2, original_dict)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'combined': [1, 2, 3, 4]}
    assert list1 == [1, 2]
    assert list2 == [3, 4]
    assert original_dict == {'a': 1}
    
    # Test with keyword arguments
    @mutant
    def modify_with_kwargs(data, **kwargs):
        for k, v in kwargs.items():
            data[k] = v
        return data
    
    original_data = {'x': 1}
    result = modify_with_kwargs(original_data, y=2, z=3)
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 1, 'y': 2, 'z': 3}
    assert original_data == {'x': 1}
    
    # Test that function metadata is preserved
    @mutant
    def example_func(x, y=1):
        """Example function docstring."""
        return x + y
    
    assert example_func.__name__ == 'example_func'
    assert example_func.__doc__ == "Example function docstring."
    
    # Test with no mutation
    @mutant
    def no_mutation(x):
        return x
    
    original = [1, 2, 3]
    result = no_mutation(original)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]
    assert original == [1, 2, 3]


# LLM-generated content at query #4
#--------------------------

```python
def test_mutant():
    # Test with simple mutation of list argument
    @mutant
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    original_list = [1, 2, 3]
    result = append_to_list(original_list, 4)
    assert original_list == [1, 2, 3]  # Original should not be modified
    assert isinstance(result, PVector)  # Result should be frozen
    assert list(result) == [1, 2, 3, 4]
    
    # Test with dict mutation
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    original_dict = {'a': 1, 'b': 2}
    result = add_to_dict(original_dict, 'c', 3)
    assert original_dict == {'a': 1, 'b': 2}  # Original should not be modified
    assert isinstance(result, PMap)  # Result should be frozen
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'][0] = 99
        data['dict']['inner'] = 'modified'
        return data
    
    original_nested = {
        'list': [1, 2, 3],
        'dict': {'a': 1}
    }
    result = modify_nested(original_nested)
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1}}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert list(result['list']) == [99, 2, 3]
    assert dict(result['dict']) == {'a': 1, 'inner': 'modified'}
    
    # Test with keyword arguments
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['new_key'] = 'new_value'
        return kwargs
    
    result = modify_kwargs(existing='value')
    assert isinstance(result, PMap)
    assert dict(result) == {'existing': 'value', 'new_key': 'new_value'}
    
    # Test with positional and keyword arguments
    @mutant
    def mixed_args(pos, kw=None):
        if kw is None:
            kw = []
        kw.append(pos)
        return {'pos': pos, 'kw': kw}
    
    result = mixed_args(42, kw=[1, 2])
    assert isinstance(result, PMap)
    assert result['pos'] == 42
    assert list(result['kw']) == [1, 2, 42]
    
    # Test that function metadata is preserved
    @mutant
    def documented_func(x):
        """Test function documentation"""
        return x
    
    assert documented_func.__name__ == 'documented_func'
    assert documented_func.__doc__ == "Test function documentation"
    
    # Test with set mutation
    @mutant
    def add_to_set(s, element):
        s.add(element)
        return s
    
    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)
    assert original_set == {1, 2, 3}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test with tuple (should remain tuple)
    @mutant
    def process_tuple(t):
        # Tuples are immutable anyway, but test the decorator handles them
        return (list(t), len(t))
    
    original_tuple = (1, 2, 3)
    result = process_tuple(original_tuple)
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert list(result[0]) == [1, 2, 3]
    assert result[1] == 3


# LLM-generated content at query #5
#--------------------------

```python
def test_mutant():
    # Test with simple mutable arguments
    @mutant
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    original_list = [1, 2, 3]
    result = append_to_list(original_list, 4)
    
    # Original list should not be modified
    assert original_list == [1, 2, 3]
    # Result should be frozen
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with nested mutable structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['inner'] = 'modified'
        return data
    
    original_data = {'list': [1, 2, 3], 'dict': {'inner': 'original'}}
    result = modify_nested(original_data)
    
    # Original data should not be modified
    assert original_data == {'list': [1, 2, 3], 'dict': {'inner': 'original'}}
    # Result should be frozen with modifications
    assert isinstance(result, PMap)
    assert result['list'] == pvector([1, 2, 3, 4])
    assert result['dict'] == pmap({'inner': 'modified'})
    
    # Test with keyword arguments
    @mutant
    def modify_kwargs(**kwargs):
        kwargs['a'].append(1)
        return kwargs
    
    original_kwargs = {'a': [0], 'b': {'c': 2}}
    result = modify_kwargs(**original_kwargs)
    
    # Original kwargs should not be modified
    assert original_kwargs == {'a': [0], 'b': {'c': 2}}
    # Result should be frozen
    assert isinstance(result, PMap)
    assert result['a'] == pvector([0, 1])
    assert result['b'] == pmap({'c': 2})
    
    # Test with mixed args and kwargs
    @mutant
    def mixed_args(list_arg, dict_arg=None):
        list_arg.append('modified')
        if dict_arg:
            dict_arg['key'] = 'value'
        return list_arg, dict_arg
    
    original_list = ['a', 'b']
    original_dict = {'existing': True}
    result = mixed_args(original_list, dict_arg=original_dict)
    
    # Originals should not be modified
    assert original_list == ['a', 'b']
    assert original_dict == {'existing': True}
    # Result should be frozen
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert list(result[0]) == ['a', 'b', 'modified']
    assert dict(result[1]) == {'existing': True, 'key': 'value'}
    
    # Test with immutable arguments (should still work)
    @mutant
    def process_immutables(num, string, tup):
        return num + 1, string.upper(), tup + (4,)
    
    result = process_immutables(5, "hello", (1, 2, 3))
    assert result == (6, 'HELLO', (1, 2, 3, 4))
    
    # Test that function metadata is preserved
    @mutant
    def documented_func(x):
        """Test function documentation"""
        return x
    
    assert documented_func.__name__ == 'documented_func'
    assert documented_func.__doc__ == "Test function documentation"


# LLM-generated content at query #6
#--------------------------

```python
def test_mutant():
    # Test with mutable list argument
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen pvector
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict argument
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen pmap
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine(a, b):
        return a + b
    
    decorated = mutant(combine)
    result = decorated([1, 2], [3, 4])
    
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with keyword arguments
    def update_dict(d, **kwargs):
        d.update(kwargs)
        return d
    
    decorated = mutant(update_dict)
    original = {'x': 10}
    result = decorated(original, y=20, z=30)
    
    assert original == {'x': 10}
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 10, 'y': 20, 'z': 30}
    
    # Test with nested mutable structures
    def modify_nested(data):
        data['list'][0] = 99
        data['inner']['key'] = 'modified'
        return data
    
    decorated = mutant(modify_nested)
    original = {'list': [1, 2, 3], 'inner': {'key': 'original'}}
    result = decorated(original)
    
    assert original == {'list': [1, 2, 3], 'inner': {'key': 'original'}}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['inner'], PMap)
    assert list(result['list']) == [99, 2, 3]
    assert dict(result['inner']) == {'key': 'modified'}
    
    # Test that function metadata is preserved
    def example_func(x, y=1):
        """Example function"""
        return x + y
    
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example function'
    
    # Test with set argument
    def add_to_set(s, element):
        s.add(element)
        return s
    
    decorated = mutant(add_to_set)
    original = {1, 2, 3}
    result = decorated(original, 4)
    
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test with tuple argument (should remain tuple)
    def process_tuple(t):
        return t + (4,)
    
    decorated = mutant(process_tuple)
    result = decorated((1, 2, 3))
    
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)


# LLM-generated content at query #7
#--------------------------

```python
def test_mutant():
    # Test with mutable list argument
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict argument
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine(a, b, c):
        return [a, b, c]
    
    decorated = mutant(combine)
    result = decorated([1], {'x': 2}, (3, 4))
    
    # All results should be frozen
    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], tuple)
    
    # Test with keyword arguments
    def process_kwargs(**kwargs):
        kwargs['processed'] = True
        return kwargs
    
    decorated = mutant(process_kwargs)
    result = decorated(x=1, y=2)
    
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 1, 'y': 2, 'processed': True}
    
    # Test that function metadata is preserved
    def example_func(x):
        """Example docstring"""
        return x
    
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring'
    
    # Test with nested mutation
    def deeply_mutate(data):
        data['list'][0] = 'modified'
        data['dict']['inner'] = 'changed'
        return data
    
    decorated = mutant(deeply_mutate)
    original = {
        'list': ['original'],
        'dict': {'key': 'value'}
    }
    result = decorated(original)
    
    # Original should not be modified
    assert original['list'][0] == 'original'
    assert original['dict'] == {'key': 'value'}
    
    # Result should show modifications but be frozen
    assert isinstance(result, PMap)
    result_dict = dict(result)
    assert result_dict['list'][0] == 'modified'
    assert result_dict['dict']['inner'] == 'changed'
    
    # Test with set argument
    def add_to_set(s, element):
        s.add(element)
        return s
    
    decorated = mutant(add_to_set)
    original = {1, 2, 3}
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == {1, 2, 3}
    # Result should be frozen (pset)
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant():
    # Test with mutable list argument
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    decorated = mutant(append_to_list)
    original_list = [1, 2, 3]
    result = decorated(original_list, 4)
    
    # Original list should not be modified
    assert original_list == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict argument
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original_dict = {'a': 1}
    result = decorated(original_dict, 'b', 2)
    
    # Original dict should not be modified
    assert original_dict == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine_data(lst, d, value):
        lst.append(value)
        d['new'] = value
        return lst, d
    
    decorated = mutant(combine_data)
    lst = [1]
    d = {'old': 0}
    result = decorated(lst, d, 2)
    
    # Originals should not be modified
    assert lst == [1]
    assert d == {'old': 0}
    # Result should be frozen tuple with frozen elements
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert list(result[0]) == [1, 2]
    assert dict(result[1]) == {'old': 0, 'new': 2}
    
    # Test with keyword arguments
    def update_with_kwargs(d, **kwargs):
        for k, v in kwargs.items():
            d[k] = v
        return d
    
    decorated = mutant(update_with_kwargs)
    original = {'x': 1}
    result = decorated(original, y=2, z=3)
    
    assert original == {'x': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 1, 'y': 2, 'z': 3}
    
    # Test that function name is preserved
    def sample_function():
        pass
    
    decorated = mutant(sample_function)
    assert decorated.__name__ == 'sample_function'
    
    # Test with already frozen inputs
    def process_frozen(pvec, pmap):
        # These operations won't actually modify the inputs
        # since they're already frozen
        return pvec, pmap
    
    decorated = mutant(process_frozen)
    pvec = pvector([1, 2, 3])
    pmap_val = pmap({'a': 1})
    result = decorated(pvec, pmap_val)
    
    # Should return frozen versions
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    
    # Test with no mutation
    def no_mutation(x, y):
        return x + y
    
    decorated = mutant(no_mutation)
    result = decorated(1, 2)
    assert result == 3  # Should return regular int, not frozen
    
    # Test with set argument
    def add_to_set(s, item):
        s.add(item)
        return s
    
    decorated = mutant(add_to_set)
    original_set = {1, 2}
    result = decorated(original_set, 3)
    
    assert original_set == {1, 2}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant():
    # Test with mutable list argument
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict argument
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine(a, b, c):
        return [a, b, c]
    
    decorated = mutant(combine)
    result = decorated([1], {'x': 2}, (3, 4))
    
    assert isinstance(result, PVector)
    assert len(result) == 3
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], tuple)
    
    # Test with keyword arguments
    def process_kwargs(**kwargs):
        kwargs['processed'] = True
        return kwargs
    
    decorated = mutant(process_kwargs)
    result = decorated(x=1, y=2)
    
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 1, 'y': 2, 'processed': True}
    
    # Test that function metadata is preserved
    def example_func(x, y=1):
        """Example function"""
        return x + y
    
    decorated = mutant(example_func)
    
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example function'
    
    # Test with nested mutation
    def deeply_mutate(data):
        data['list'][0] = 'modified'
        data['inner']['value'] = 'changed'
        return data
    
    decorated = mutant(deeply_mutate)
    original = {
        'list': ['original'],
        'inner': {'value': 'original'}
    }
    result = decorated(original)
    
    # Original should not be modified
    assert original['list'][0] == 'original'
    assert original['inner']['value'] == 'original'
    # Result should be frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert result['list'][0] == 'modified'
    assert isinstance(result['inner'], PMap)
    assert result['inner']['value'] == 'changed'
    
    # Test with set argument
    def add_to_set(s, item):
        s.add(item)
        return s
    
    decorated = mutant(add_to_set)
    original = {1, 2, 3}
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == {1, 2, 3}
    # Result should be frozen (pset)
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test with tuple argument (should remain tuple)
    def process_tuple(t):
        # Tuples are immutable, but can contain mutable elements
        return t
    
    decorated = mutant(process_tuple)
    original = ([1, 2], {'a': 3})
    result = decorated(original)
    
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple args and kwargs
    def modify_multiple(a, b, c=0):
        a.append(99)
        b['key'] = 'value'
        return [a, b, c]
    
    decorated = mutant(modify_multiple)
    list_arg = [1, 2]
    dict_arg = {'x': 1}
    result = decorated(list_arg, dict_arg, c=5)
    
    # Originals should not be modified
    assert list_arg == [1, 2]
    assert dict_arg == {'x': 1}
    # Result should be frozen structure
    assert isinstance(result, PVector)
    assert len(result) == 3
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert list(result[0]) == [1, 2, 99]
    assert dict(result[1]) == {'x': 1, 'key': 'value'}
    assert result[2] == 5
    
    # Test with nested mutation
    def nested_mutation(data):
        data['list'].append(100)
        data['dict']['nested'] = 'modified'
        return data
    
    decorated = mutant(nested_mutation)
    original = {
        'list': [1, 2],
        'dict': {'a': 1}
    }
    result = decorated(original)
    
    # Original should not be modified
    assert original['list'] == [1, 2]
    assert original['dict'] == {'a': 1}
    # Result should be fully frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert list(result['list']) == [1, 2, 100]
    assert dict(result['dict']) == {'a': 1, 'nested': 'modified'}
    
    # Test that function metadata is preserved
    def example_func(x, y=10):
        """Example docstring"""
        return x + y
    
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring'
    
    # Test with no mutation
    def no_mutation(x, y):
        return x + y
    
    decorated = mutant(no_mutation)
    result = decorated(5, 3)
    assert result == 8  # Should return regular int (not frozen)
    
    # Test with tuple input (should remain tuple)
    def process_tuple(t):
        return t
    
    decorated = mutant(process_tuple)
    input_tuple = ([1, 2], {'a': 3})
    result = decorated(input_tuple)
    
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)


# LLM-generated content at query #11
#--------------------------

```python
def test_mutant():
    # Test with mutable list argument
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict argument
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine_structures(lst, d, s):
        lst.append('modified')
        d['new'] = 'value'
        return (lst, d, s)
    
    decorated = mutant(combine_structures)
    lst_arg = [1, 2]
    dict_arg = {'x': 10}
    set_arg = {1, 2, 3}
    result = decorated(lst_arg, dict_arg, set_arg)
    
    # Originals should not be modified
    assert lst_arg == [1, 2]
    assert dict_arg == {'x': 10}
    assert set_arg == {1, 2, 3}
    
    # Result should be frozen
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)
    
    # Test with keyword arguments
    def update_with_kwargs(d, **kwargs):
        for k, v in kwargs.items():
            d[k] = v
        return d
    
    decorated = mutant(update_with_kwargs)
    original = {'a': 1}
    result = decorated(original, b=2, c=3)
    
    assert original == {'a': 1}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3}
    
    # Test that function name is preserved
    def example_func():
        pass
    
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    
    # Test with nested mutable structures
    def modify_nested(data):
        data['list'][0] = 'changed'
        data['inner']['key'] = 'modified'
        return data
    
    decorated = mutant(modify_nested)
    original = {
        'list': [1, 2, 3],
        'inner': {'key': 'value'}
    }
    result = decorated(original)
    
    assert original['list'] == [1, 2, 3]
    assert original['inner'] == {'key': 'value'}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['inner'], PMap)
    
    # Test with no mutation
    def no_mutation(x, y):
        return x + y
    
    decorated = mutant(no_mutation)
    result = decorated(1, 2)
    assert result == 3
    
    # Test with set argument
    def add_to_set(s, element):
        s.add(element)
        return s
    
    decorated = mutant(add_to_set)
    original = {1, 2}
    result = decorated(original, 3)
    
    assert original == {1, 2}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze():
    # Test basic types remain unchanged
    assert freeze(5) == 5
    assert freeze("hello") == "hello"
    assert freeze(None) is None
    
    # Test list to pvector conversion
    result = freeze([1, 2, 3])
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]
    
    # Test nested list conversion
    result = freeze([1, [2, 3], 4])
    assert isinstance(result, PVector)
    assert isinstance(result[1], PVector)
    assert list(result[1]) == [2, 3]
    
    # Test dict to pmap conversion
    result = freeze({"a": 1, "b": 2})
    assert isinstance(result, PMap)
    assert dict(result) == {"a": 1, "b": 2}
    
    # Test nested dict conversion
    result = freeze({"a": {"b": 1}, "c": 2})
    assert isinstance(result, PMap)
    assert isinstance(result["a"], PMap)
    assert dict(result["a"]) == {"b": 1}
    
    # Test set to pset conversion
    result = freeze({1, 2, 3})
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}
    
    # Test tuple remains tuple but freezes contents
    result = freeze((1, [2, 3], {"a": 4}))
    assert isinstance(result, tuple)
    assert result[0] == 1
    assert isinstance(result[1], PVector)
    assert isinstance(result[2], PMap)
    
    # Test defaultdict conversion
    dd = collections.defaultdict(list, {"a": 1, "b": [2, 3]})
    result = freeze(dd)
    assert isinstance(result, PMap)
    assert dict(result) == {"a": 1, "b": pvector([2, 3])}
    
    # Test strict=False behavior with pyrsistent containers
    pv = pvector([1, 2, 3])
    pm = pmap({"a": 1})
    ps = pset([1, 2])
    
    # With strict=True (default), pyrsistent containers are recursively frozen
    result = freeze(pv, strict=True)
    assert result is pv  # Should return same object since already frozen
    
    result = freeze(pm, strict=True)
    assert result is pm  # Should return same object since already frozen
    
    # With strict=False, pyrsistent containers are not recursively frozen
    nested_pv = pvector([1, [2, 3]])
    result = freeze(nested_pv, strict=False)
    assert result is nested_pv  # Returns same object, no recursion
    
    # Test mixed nested structures
    complex_obj = {
        "list": [1, 2, {"inner": 3}],
        "tuple": (4, [5, 6]),
        "set": {7, 8, 9}
    }
    result = freeze(complex_obj)
    assert isinstance(result, PMap)
    assert isinstance(result["list"], PVector)
    assert isinstance(result["list"][2], PMap)
    assert isinstance(result["tuple"], tuple)
    assert isinstance(result["tuple"][1], PVector)
    assert isinstance(result["set"], PSet)
    
    # Test that dict keys are not frozen (they remain as-is)
    key_obj = [1, 2]
    result = freeze({key_obj: "value"})
    assert key_obj in result
    assert result[key_obj] == "value"
    assert isinstance(key_obj, list)  # Key remains mutable list
    
    # Test empty containers
    assert freeze([]) == pvector()
    assert freeze({}) == pmap()
    assert freeze(set()) == pset()
    assert freeze(()) == ()


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant():
    # Test with mutable list argument
    def append_one(lst):
        lst.append(1)
        return lst
    
    decorated = mutant(append_one)
    original = [2, 3]
    result = decorated(original)
    
    assert original == [2, 3]  # Original should not be mutated
    assert isinstance(result, PVector)  # Result should be frozen
    assert list(result) == [2, 3, 1]  # Result should have correct values
    
    # Test with mutable dict argument
    def add_key(d):
        d['new'] = 'value'
        return d
    
    decorated = mutant(add_key)
    original = {'a': 1}
    result = decorated(original)
    
    assert original == {'a': 1}  # Original should not be mutated
    assert isinstance(result, PMap)  # Result should be frozen
    assert dict(result) == {'a': 1, 'new': 'value'}  # Result should have correct values
    
    # Test with multiple arguments
    def combine(lst1, lst2):
        return lst1 + lst2
    
    decorated = mutant(combine)
    result = decorated([1, 2], [3, 4])
    
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with keyword arguments
    def merge_dicts(d1, d2):
        return {**d1, **d2}
    
    decorated = mutant(merge_dicts)
    result = decorated({'a': 1}, d2={'b': 2})
    
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test that function name is preserved
    def my_function():
        pass
    
    decorated = mutant(my_function)
    assert decorated.__name__ == 'my_function'
    
    # Test with nested mutable structures
    def modify_nested(data):
        data['list'][0] = 'modified'
        data['inner']['key'] = 'changed'
        return data
    
    decorated = mutant(modify_nested)
    original = {'list': ['original'], 'inner': {'key': 'original'}}
    result = decorated(original)
    
    assert original == {'list': ['original'], 'inner': {'key': 'original'}}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['inner'], PMap)
    assert result['list'][0] == 'modified'
    assert result['inner']['key'] == 'changed'
    
    # Test with set argument
    def add_to_set(s):
        s.add(4)
        return s
    
    decorated = mutant(add_to_set)
    original = {1, 2, 3}
    result = decorated(original)
    
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test with tuple argument (should remain tuple)
    def process_tuple(t):
        return t + ('extra',)
    
    decorated = mutant(process_tuple)
    result = decorated(('a', 'b'))
    
    assert isinstance(result, tuple)
    assert result == ('a', 'b', 'extra')
    
    # Test that returned immutable structures cannot be mutated
    result = decorated([1, 2])
    try:
        result.append(3)  # Should fail on PVector
        assert False, "Should not be able to mutate frozen result"
    except AttributeError:
        pass  # Expected
    
    # Test with no arguments
    def return_mutable():
        return [1, 2, 3]
    
    decorated = mutant(return_mutable)
    result = decorated()
    
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated_dict = mutant(add_to_dict)
    original_dict = {'a': 1}
    result_dict = decorated_dict(original_dict, 'b', 2)
    
    assert original_dict == {'a': 1}
    assert isinstance(result_dict, PMap)
    assert dict(result_dict) == {'a': 1, 'b': 2}
    
    # Test with nested structures
    def modify_nested(data):
        data['list'][0] = 99
        data['dict']['new'] = 'value'
        return data
    
    decorated_nested = mutant(modify_nested)
    original_nested = {'list': [1, 2, 3], 'dict': {'a': 1}}
    result_nested = decorated_nested(original_nested)
    
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1}}
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['list'], PVector)
    assert isinstance(result_nested['dict'], PMap)
    assert result_nested['list'][0] == 99
    assert result_nested['dict']['new'] == 'value'
    
    # Test with keyword arguments
    def kw_func(a, b=[]):
        b.append(a)
        return b
    
    decorated_kw = mutant(kw_func)
    result_kw = decorated_kw(5, b=[1, 2])
    
    assert isinstance(result_kw, PVector)
    assert list(result_kw) == [1, 2, 5]
    
    # Test with multiple arguments
    def multi_args(a, b, c):
        a.append(1)
        b['key'] = 'value'
        c.add(4)
        return a, b, c
    
    decorated_multi = mutant(multi_args)
    result_multi = decorated_multi([], {}, {1, 2, 3})
    
    assert isinstance(result_multi, tuple)
    assert isinstance(result_multi[0], PVector)
    assert isinstance(result_multi[1], PMap)
    assert isinstance(result_multi[2], PSet)
    assert list(result_multi[0]) == [1]
    assert dict(result_multi[1]) == {'key': 'value'}
    assert set(result_multi[2]) == {1, 2, 3, 4}
    
    # Test that function metadata is preserved
    def documented_func():
        """A test function"""
        return []
    
    decorated_doc = mutant(documented_func)
    assert decorated_doc.__name__ == 'documented_func'
    assert decorated_doc.__doc__ == 'A test function'
    
    # Test with no mutation
    def no_mutation(x):
        return x * 2
    
    decorated_no_mut = mutant(no_mutation)
    result_no_mut = decorated_no_mut(5)
    assert result_no_mut == 10


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant():
    # Test with list mutation
    @mutant
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    original_list = [1, 2, 3]
    result = append_to_list(original_list, 4)
    assert original_list == [1, 2, 3]
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with dict mutation
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    original_dict = {'a': 1, 'b': 2}
    result = add_to_dict(original_dict, 'c', 3)
    assert original_dict == {'a': 1, 'b': 2}
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3}
    
    # Test with set mutation
    @mutant
    def add_to_set(s, value):
        s.add(value)
        return s
    
    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)
    assert original_set == {1, 2, 3}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'][0] = 100
        data['dict']['inner'] = 'modified'
        return data
    
    original_nested = {
        'list': [1, 2, 3],
        'dict': {'a': 1, 'b': 2}
    }
    result = modify_nested(original_nested)
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1, 'b': 2}}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert list(result['list']) == [100, 2, 3]
    assert dict(result['dict']) == {'a': 1, 'b': 2, 'inner': 'modified'}
    
    # Test with multiple arguments
    @mutant
    def combine_structures(list_arg, dict_arg, set_arg):
        list_arg.append('modified')
        dict_arg['new'] = 'value'
        set_arg.add(999)
        return list_arg, dict_arg, set_arg
    
    list_arg = [1, 2]
    dict_arg = {'x': 10}
    set_arg = {100, 200}
    
    result = combine_structures(list_arg, dict_arg, set_arg)
    assert list_arg == [1, 2]
    assert dict_arg == {'x': 10}
    assert set_arg == {100, 200}
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)
    assert list(result[0]) == [1, 2, 'modified']
    assert dict(result[1]) == {'x': 10, 'new': 'value'}
    assert set(result[2]) == {100, 200, 999}
    
    # Test with keyword arguments
    @mutant
    def modify_with_kwargs(data, prefix=''):
        data['value'] = prefix + str(data.get('value', ''))
        return data
    
    original = {'value': 'test'}
    result = modify_with_kwargs(original, prefix='pre_')
    assert original == {'value': 'test'}
    assert isinstance(result, PMap)
    assert dict(result) == {'value': 'pre_test'}
    
    # Test that function metadata is preserved
    @mutant
    def example_func(x, y=1):
        """Example function documentation."""
        return x + y
    
    assert example_func.__name__ == 'example_func'
    assert example_func.__doc__ == "Example function documentation."
    
    # Test with no mutation
    @mutant
    def no_mutation(x):
        return x * 2
    
    result = no_mutation(5)
    assert result == 10
    
    # Test with tuple argument
    @mutant
    def process_tuple(t):
        return t + (4,)
    
    original_tuple = (1, 2, 3)
    result = process_tuple(original_tuple)
    assert original_tuple == (1, 2, 3)
    assert result == (1, 2, 3, 4)


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant():
    # Test with list mutation
    @mutant
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    original_list = [1, 2, 3]
    result = append_to_list(original_list, 4)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    assert original_list == [1, 2, 3]
    
    # Test with dict mutation
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    original_dict = {'a': 1, 'b': 2}
    result = add_to_dict(original_dict, 'c', 3)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2, 'c': 3}
    assert original_dict == {'a': 1, 'b': 2}
    
    # Test with set mutation
    @mutant
    def add_to_set(s, value):
        s.add(value)
        return s
    
    original_set = {1, 2, 3}
    result = add_to_set(original_set, 4)
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    assert original_set == {1, 2, 3}
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(4)
        data['dict']['new'] = 'value'
        return data
    
    original_nested = {
        'list': [1, 2, 3],
        'dict': {'a': 1}
    }
    result = modify_nested(original_nested)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert list(result['list']) == [1, 2, 3, 4]
    assert dict(result['dict']) == {'a': 1, 'new': 'value'}
    assert original_nested == {'list': [1, 2, 3], 'dict': {'a': 1}}
    
    # Test with multiple arguments
    @mutant
    def combine_and_modify(lst1, lst2, d):
        combined = lst1 + lst2
        d['combined_length'] = len(combined)
        return combined, d
    
    list1 = [1, 2]
    list2 = [3, 4]
    dict_arg = {'original': 'value'}
    
    result = combine_and_modify(list1, list2, dict_arg)
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert list(result[0]) == [1, 2, 3, 4]
    assert dict(result[1]) == {'original': 'value', 'combined_length': 4}
    assert list1 == [1, 2]
    assert list2 == [3, 4]
    assert dict_arg == {'original': 'value'}
    
    # Test with keyword arguments
    @mutant
    def modify_with_kwargs(data, multiplier=1):
        return [x * multiplier for x in data]
    
    original = [1, 2, 3]
    result = modify_with_kwargs(original, multiplier=2)
    assert isinstance(result, PVector)
    assert list(result) == [2, 4, 6]
    assert original == [1, 2, 3]
    
    # Test that function metadata is preserved
    @mutant
    def example_func(x, y=1):
        """Example function docstring."""
        return x + y
    
    assert example_func.__name__ == 'example_func'
    assert example_func.__doc__ == "Example function docstring."
    
    # Test with no mutation (pure function)
    @mutant
    def pure_add(a, b):
        return a + b
    
    result = pure_add(1, 2)
    assert result == 3
    
    # Test with tuple argument
    @mutant
    def extend_tuple(t, value):
        return t + (value,)
    
    original_tuple = (1, 2, 3)
    result = extend_tuple(original_tuple, 4)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)
    assert original_tuple == (1, 2, 3)


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant():
    # Test with mutable list argument
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict argument
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated_dict = mutant(add_to_dict)
    original_dict = {'a': 1}
    result_dict = decorated_dict(original_dict, 'b', 2)
    
    # Original should not be modified
    assert original_dict == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result_dict, PMap)
    assert dict(result_dict) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine_and_modify(lst1, lst2):
        combined = lst1 + lst2
        combined.append('mutated')
        return combined
    
    decorated_combine = mutant(combine_and_modify)
    list1 = [1, 2]
    list2 = [3, 4]
    result_combined = decorated_combine(list1, list2)
    
    assert list1 == [1, 2]
    assert list2 == [3, 4]
    assert isinstance(result_combined, PVector)
    assert list(result_combined) == [1, 2, 3, 4, 'mutated']
    
    # Test with keyword arguments
    def update_with_kwargs(d, **kwargs):
        for k, v in kwargs.items():
            d[k] = v
        return d
    
    decorated_kwargs = mutant(update_with_kwargs)
    original_kw = {'x': 10}
    result_kw = decorated_kwargs(original_kw, y=20, z=30)
    
    assert original_kw == {'x': 10}
    assert isinstance(result_kw, PMap)
    assert dict(result_kw) == {'x': 10, 'y': 20, 'z': 30}
    
    # Test that function metadata is preserved
    def example_func(a, b):
        """Example function docstring."""
        return [a, b]
    
    decorated_example = mutant(example_func)
    assert decorated_example.__name__ == 'example_func'
    assert decorated_example.__doc__ == 'Example function docstring.'
    
    # Test with already frozen input
    frozen_input = pvector([1, 2, 3])
    result_frozen = decorated(frozen_input, 4)
    assert isinstance(result_frozen, PVector)
    assert list(result_frozen) == [1, 2, 3, 4]
    
    # Test with no mutation (pure function)
    def pure_func(a, b):
        return a + b
    
    decorated_pure = mutant(pure_func)
    result_pure = decorated_pure(10, 20)
    assert result_pure == 30  # Should return unfrozen primitive
    
    # Test with nested structures
    def modify_nested(data):
        data['list'][0] = 'modified'
        data['inner']['key'] = 'changed'
        return data
    
    decorated_nested = mutant(modify_nested)
    nested_input = {
        'list': [1, 2, 3],
        'inner': {'key': 'value'}
    }
    result_nested = decorated_nested(nested_input)
    
    assert nested_input == {'list': [1, 2, 3], 'inner': {'key': 'value'}}
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['list'], PVector)
    assert isinstance(result_nested['inner'], PMap)
    assert result_nested['list'][0] == 'modified'
    assert result_nested['inner']['key'] == 'changed'


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple arguments
    def combine(a, b, c):
        return [a, b, c]
    
    decorated = mutant(combine)
    result = decorated([1], {'x': 2}, (3, 4))
    
    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], tuple)
    
    # Test with keyword arguments
    def process_kwargs(**kwargs):
        kwargs['processed'] = True
        return kwargs
    
    decorated = mutant(process_kwargs)
    result = decorated(x=1, y=2)
    
    assert isinstance(result, PMap)
    assert dict(result) == {'x': 1, 'y': 2, 'processed': True}
    
    # Test that function name is preserved
    def my_function():
        pass
    
    decorated = mutant(my_function)
    assert decorated.__name__ == 'my_function'
    
    # Test with nested mutation
    def nested_mutation(data):
        data['list'].append(99)
        data['dict']['inner'] = 'modified'
        return data
    
    decorated = mutant(nested_mutation)
    original = {'list': [1, 2], 'dict': {'a': 1}}
    result = decorated(original)
    
    # Original should not be modified
    assert original == {'list': [1, 2], 'dict': {'a': 1}}
    # Result should be fully frozen
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert list(result['list']) == [1, 2, 99]
    assert isinstance(result['dict'], PMap)
    assert dict(result['dict']) == {'a': 1, 'inner': 'modified'}
    
    # Test with set input
    def add_to_set(s, value):
        s.add(value)
        return s
    
    decorated = mutant(add_to_set)
    original = {1, 2, 3}
    result = decorated(original, 4)
    
    assert original == {1, 2, 3}
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3, 4}
    
    # Test with tuple input (should remain tuple)
    def wrap_tuple(t):
        return t
    
    decorated = mutant(wrap_tuple)
    original = (1, [2, 3])
    result = decorated(original)
    
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)
    assert list(result[1]) == [2, 3]


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant():
    # Test with mutable list input
    def append_to_list(lst, value):
        lst.append(value)
        return lst
    
    decorated = mutant(append_to_list)
    original = [1, 2, 3]
    result = decorated(original, 4)
    
    # Original should not be modified
    assert original == [1, 2, 3]
    # Result should be frozen (pvector)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]
    
    # Test with mutable dict input
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    decorated = mutant(add_to_dict)
    original = {'a': 1}
    result = decorated(original, 'b', 2)
    
    # Original should not be modified
    assert original == {'a': 1}
    # Result should be frozen (pmap)
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with multiple args and kwargs
    def complex_mutation(a, b, c=None):
        a.append(99)
        b['mutated'] = True
        if c is not None:
            c.add(100)
        return a, b, c
    
    decorated = mutant(complex_mutation)
    list_arg = [1, 2]
    dict_arg = {'x': 10}
    set_arg = {1, 2, 3}
    
    result = decorated(list_arg, dict_arg, c=set_arg)
    
    # Originals should not be modified
    assert list_arg == [1, 2]
    assert dict_arg == {'x': 10}
    assert set_arg == {1, 2, 3}
    
    # Results should be frozen
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)
    assert list(result[0]) == [1, 2, 99]
    assert dict(result[1]) == {'x': 10, 'mutated': True}
    assert set(result[2]) == {1, 2, 3, 100}
    
    # Test with no mutation
    def no_mutation(x, y):
        return x + y
    
    decorated = mutant(no_mutation)
    result = decorated(1, 2)
    assert result == 3  # Should return regular int
    
    # Test with tuple return (should remain tuple, not frozen)
    def return_tuple():
        return (1, [2, 3], {'a': 4})
    
    decorated = mutant(return_tuple)
    result = decorated()
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)  # Inner list should be frozen
    assert isinstance(result[2], PMap)     # Inner dict should be frozen
    
    # Test function metadata preservation
    def example_func(x, y=5):
        """Example docstring"""
        return x + y
    
    decorated = mutant(example_func)
    assert decorated.__name__ == 'example_func'
    assert decorated.__doc__ == 'Example docstring'


