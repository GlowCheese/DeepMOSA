####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_freeze():
    # Test freezing a simple list
    result = freeze([1, 2, 3])
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]
    
    # Test freezing a nested list with dict
    result = freeze([1, {'a': 3}])
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['a'] == 3
    
    # Test freezing a dict
    result = freeze({'a': 1, 'b': 2})
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2
    
    # Test freezing a dict with nested list
    result = freeze({'a': [1, 2]})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert list(result['a']) == [1, 2]
    
    # Test freezing a defaultdict
    dd = collections.defaultdict(int)
    dd['x'] = 10
    result = freeze(dd)
    assert isinstance(result, PMap)
    assert result['x'] == 10
    
    # Test freezing a set
    result = freeze({1, 2, 3})
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}
    
    # Test freezing a tuple
    result = freeze((1, [2, 3]))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)
    assert list(result[1]) == [2, 3]
    
    # Test freezing a tuple with dict
    result = freeze((1, {'a': 2}))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PMap)
    
    # Test freezing scalar values
    assert freeze(42) == 42
    assert freeze("string") == "string"
    assert freeze(None) is None
    
    # Test freezing deeply nested structure
    result = freeze({'a': [1, {'b': [2, 3]}]})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][1], PMap)
    assert isinstance(result['a'][1]['b'], PVector)
    
    # Test strict=False with PVector
    pv = pvector([1, 2])
    result = freeze(pv, strict=False)
    assert result is pv
    
    # Test strict=True with PVector
    pv = pvector([1, 2])
    result = freeze(pv, strict=True)
    assert isinstance(result, PVector)
    
    # Test strict=False with PMap
    pm = pmap({'a': 1})
    result = freeze(pm, strict=False)
    assert result is pm
    
    # Test strict=True with PMap
    pm = pmap({'a': 1})
    result = freeze(pm, strict=True)
    assert isinstance(result, PMap)
    
    # Test empty containers
    assert isinstance(freeze([]), PVector)
    assert isinstance(freeze({}), PMap)
    assert isinstance(freeze(set()), PSet)
    assert freeze(()) == ()


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze():
    # Test freezing dict
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})
    
    # Test freezing nested dict
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})
    
    # Test freezing list
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    
    # Test freezing nested list
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    
    # Test freezing list with dict
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])
    
    # Test freezing set
    assert freeze({1, 2}) == pset([1, 2])
    
    # Test freezing tuple
    assert freeze((1, 2)) == (1, 2)
    
    # Test freezing nested tuple
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))
    
    # Test freezing tuple with dict
    assert freeze((1, {'a': 3})) == (1, pmap({'a': 3}))
    
    # Test freezing defaultdict
    d = collections.defaultdict(int)
    d['a'] = 1
    assert freeze(d) == pmap({'a': 1})
    
    # Test freezing nested defaultdict
    d = collections.defaultdict(dict)
    d['a'] = {'b': 2}
    assert freeze(d) == pmap({'a': pmap({'b': 2})})
    
    # Test freezing primitive types
    assert freeze(42) == 42
    assert freeze("string") == "string"
    assert freeze(3.14) == 3.14
    assert freeze(None) is None
    
    # Test freezing with strict=False
    pv = pvector([1, 2, 3])
    assert freeze(pv, strict=False) is pv
    
    pm = pmap({'a': 1})
    assert freeze(pm, strict=False) is pm
    
    # Test freezing with strict=True (default)
    pv = pvector([1, 2, 3])
    result = freeze(pv, strict=True)
    assert result == pvector([1, 2, 3])
    
    pm = pmap({'a': 1})
    result = freeze(pm, strict=True)
    assert result == pmap({'a': 1})
    
    # Test complex nested structure
    complex_obj = {
        'list': [1, 2, {'nested': 'dict'}],
        'dict': {'key': [1, 2, 3]},
        'tuple': (1, [2, 3]),
        'set': {1, 2, 3}
    }
    frozen = freeze(complex_obj)
    assert frozen == pmap({
        'list': pvector([1, 2, pmap({'nested': 'dict'})]),
        'dict': pmap({'key': pvector([1, 2, 3])}),
        'tuple': (1, pvector([2, 3])),
        'set': pset([1, 2, 3])
    })
    
    # Test empty containers
    assert freeze([]) == pvector([])
    assert freeze({}) == pmap({})
    assert freeze(()) == ()
    assert freeze(set()) == pset([])
    
    # Test that dict keys are not frozen
    frozen = freeze({(1, 2): 'value'})
    assert frozen == pmap({(1, 2): 'value'})


# LLM-generated content at query #3
#--------------------------

```python
def test_thaw():
    from pyrsistent import s, m, v
    
    # Test thaw of pset to set
    assert thaw(s(1, 2)) == {1, 2}
    assert thaw(s(1, 2, 3)) == {1, 2, 3}
    
    # Test thaw of pvector with pmap to list with dict
    assert thaw(v(1, m(a=3))) == [1, {'a': 3}]
    
    # Test thaw of tuple with pvector
    assert thaw((1, v())) == (1, [])
    
    # Test thaw of nested structures
    assert thaw(m(x=v(1, 2, 3))) == {'x': [1, 2, 3]}
    assert thaw(v(m(a=1), m(b=2))) == [{'a': 1}, {'b': 2}]
    
    # Test thaw of simple types (should pass through)
    assert thaw(42) == 42
    assert thaw("string") == "string"
    assert thaw(None) is None
    
    # Test thaw of tuple with nested pyrsistent structures
    assert thaw((v(1, 2), m(x=3))) == ([1, 2], {'x': 3})
    
    # Test thaw with strict=False on regular dict and list
    assert thaw({'a': 1}, strict=False) == {'a': 1}
    assert thaw([1, 2, 3], strict=False) == [1, 2, 3]
    
    # Test thaw with strict=True on regular dict and list (should thaw recursively)
    assert thaw({'a': [1, 2]}, strict=True) == {'a': [1, 2]}
    assert thaw([v(1, 2)], strict=True) == [[1, 2]]
    
    # Test thaw of pmap with nested structures
    result = thaw(m(a=v(1, 2), b=m(c=3)))
    assert result == {'a': [1, 2], 'b': {'c': 3}}
    
    # Test thaw of empty containers
    assert thaw(s()) == set()
    assert thaw(v()) == []
    assert thaw(m()) == {}
    assert thaw(()) == ()
    
    # Test thaw of pset with various types
    pset_result = thaw(s(1, 2, 3, 4, 5))
    assert pset_result == {1, 2, 3, 4, 5}
    
    # Test thaw of complex nested structure
    complex_structure = m(
        data=v(1, 2, m(nested=v(3, 4))),
        metadata=m(version=1)
    )
    expected = {
        'data': [1, 2, {'nested': [3, 4]}],
        'metadata': {'version': 1}
    }
    assert thaw(complex_structure) == expected


# LLM-generated content at query #4
#--------------------------

def test_mutant():
    """Test the mutant decorator"""
    
    # Test basic mutant decorator functionality
    @mutant
    def modify_list(lst):
        lst = lst.append(4)
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, PVector)
    
    # Test that input is frozen (immutable)
    @mutant
    def try_modify_input(lst):
        # Since lst is frozen as pvector, we can't modify it in place
        return lst + pvector([99])
    
    original = [1, 2, 3]
    result = try_modify_input(original)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 99]
    assert original == [1, 2, 3]  # Original unchanged
    
    # Test with dict argument
    @mutant
    def modify_dict(d):
        return d.set('new_key', 'new_value')
    
    original_dict = {'a': 1}
    result = modify_dict(original_dict)
    assert isinstance(result, PMap)
    assert result['new_key'] == 'new_value'
    assert 'new_key' not in original_dict
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        return data.set('list', data['list'] + pvector([4]))
    
    nested = {'list': [1, 2, 3]}
    result = modify_nested(nested)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert list(result['list']) == [1, 2, 3, 4]
    
    # Test with multiple arguments
    @mutant
    def combine_args(lst, d):
        return lst + pvector([d['key']])
    
    result = combine_args([1, 2], {'key': 3})
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]
    
    # Test with keyword arguments
    @mutant
    def func_with_kwargs(lst, multiplier=2):
        return lst + pvector([multiplier])
    
    result = func_with_kwargs([1, 2], multiplier=5)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 5]
    
    # Test with set argument
    @mutant
    def modify_set(s):
        return s
    
    original_set = {1, 2, 3}
    result = modify_set(original_set)
    assert isinstance(result, PSet)
    
    # Test return value is frozen
    @mutant
    def return_mutable():
        return [1, 2, 3]
    
    result = return_mutable()
    assert isinstance(result, PVector)
    
    # Test with tuple argument
    @mutant
    def handle_tuple(t):
        return t
    
    result = handle_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)
    
    # Test function metadata is preserved
    @mutant
    def documented_function(x):
        """This is a documented function"""
        return x
    
    assert documented_function.__doc__ == "This is a documented function"
    assert documented_function.__name__ == "documented_function"


# LLM-generated content at query #5
#--------------------------

def test_mutant():
    """Test the mutant decorator"""
    
    # Test basic mutant behavior with list argument
    @mutant
    def modify_list(lst):
        lst = lst.append(4)
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, PVector)
    
    # Test that input is frozen and not modified
    @mutant
    def try_mutate_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    input_dict = {'a': 1}
    result = try_mutate_dict(input_dict)
    assert isinstance(result, PMap)
    assert input_dict == {'a': 1}  # original unchanged
    
    # Test with nested structures
    @mutant
    def process_nested(data):
        return data
    
    nested_input = {'list': [1, 2], 'set': {3, 4}}
    result = process_nested(nested_input)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)
    
    # Test with kwargs
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'x': 1})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    
    # Test function metadata is preserved
    @mutant
    def documented_func():
        """This is documented"""
        return []
    
    assert documented_func.__doc__ == "This is documented"
    assert documented_func.__name__ == "documented_func"
    
    # Test return value is frozen
    @mutant
    def return_mutable():
        return {'key': [1, 2, 3]}
    
    result = return_mutable()
    assert isinstance(result, PMap)
    assert isinstance(result['key'], PVector)
    
    # Test with multiple arguments
    @mutant
    def multi_arg(lst, dct, st):
        return {'list': lst, 'dict': dct, 'set': st}
    
    result = multi_arg([1], {'a': 1}, {2, 3})
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert isinstance(result['set'], PSet)


# LLM-generated content at query #6
#--------------------------

def test_mutant():
    """Test the mutant decorator for freezing arguments and return values"""
    
    # Test basic functionality with list argument
    @mutant
    def modify_list(lst):
        # Try to modify the list (should fail since it's frozen as pvector)
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]
    
    # Test with dict argument
    @mutant
    def modify_dict(d):
        return d
    
    result = modify_dict({'a': 1, 'b': 2})
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1, 'b': 2}
    
    # Test with nested structures
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'list': [1, 2], 'dict': {'x': 10}})
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    
    # Test with multiple arguments
    @mutant
    def combine_args(a, b):
        return [a, b]
    
    result = combine_args([1, 2], {'key': 'value'})
    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    
    # Test with keyword arguments
    @mutant
    def kwargs_func(a, b=None):
        return {'a': a, 'b': b}
    
    result = kwargs_func([1, 2], b=[3, 4])
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PVector)
    
    # Test that function name is preserved
    assert modify_list.__name__ == 'modify_list'
    
    # Test with set argument
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, PSet)
    assert set(result) == {1, 2, 3}
    
    # Test with tuple argument
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)


# LLM-generated content at query #7
#--------------------------

```python
def test_mutant():
    # Test basic mutant decorator functionality
    @mutant
    def modify_list(lst):
        lst = lst.append(4)
        return lst
    
    original = [1, 2, 3]
    result = modify_list(original)
    # Original should not be modified
    assert original == [1, 2, 3]
    assert isinstance(result, PVector)
    
    # Test with dict argument
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    original_dict = {'a': 1}
    result = modify_dict(original_dict)
    assert original_dict == {'a': 1}
    assert isinstance(result, PMap)
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(5)
        return data
    
    original_nested = {'list': [1, 2, 3]}
    result = modify_nested(original_nested)
    assert original_nested == {'list': [1, 2, 3]}
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    
    # Test with multiple arguments
    @mutant
    def combine(lst1, lst2):
        lst1 = lst1.append(10)
        lst2 = lst2.append(20)
        return [lst1, lst2]
    
    list1 = [1, 2]
    list2 = [3, 4]
    result = combine(list1, list2)
    assert list1 == [1, 2]
    assert list2 == [3, 4]
    assert isinstance(result, PVector)
    
    # Test with keyword arguments
    @mutant
    def modify_with_kwargs(data, extra=None):
        if extra:
            data['extra'] = extra
        return data
    
    original = {'key': 'value'}
    result = modify_with_kwargs(original, extra='test')
    assert original == {'key': 'value'}
    assert isinstance(result, PMap)
    assert result['extra'] == 'test'
    
    # Test that function name is preserved
    assert modify_list.__name__ == 'modify_list'
    
    # Test with set argument
    @mutant
    def modify_set(s):
        s = s.add(99)
        return s
    
    original_set = {1, 2, 3}
    result = modify_set(original_set)
    assert original_set == {1, 2, 3}
    assert isinstance(result, PSet)
    
    # Test with tuple argument
    @mutant
    def modify_tuple(t):
        return t + (99,)
    
    original_tuple = (1, 2, 3)
    result = modify_tuple(original_tuple)
    assert original_tuple == (1, 2, 3)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 99)


# LLM-generated content at query #8
#--------------------------

def test_mutant():
    """Test the mutant decorator"""
    
    # Test basic mutant behavior with list argument
    @mutant
    def modify_list(lst):
        lst = lst.append(4)
        return lst
    
    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3])
    
    # Test mutant with dict argument
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    result = modify_dict({'a': 1})
    assert isinstance(result, PMap)
    assert result['a'] == 1
    
    # Test mutant preserves original arguments
    original_list = [1, 2, 3]
    @mutant
    def try_modify(lst):
        return lst
    
    try_modify(original_list)
    assert original_list == [1, 2, 3]
    
    # Test mutant with nested structures
    @mutant
    def process_nested(data):
        return data
    
    nested_input = {'list': [1, 2], 'dict': {'key': 'value'}}
    result = process_nested(nested_input)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    
    # Test mutant with kwargs
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'x': 1})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    
    # Test mutant with set
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, PSet)
    
    # Test mutant with tuple
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)
    
    # Test mutant return value is frozen
    @mutant
    def return_mutable():
        return [1, 2, 3]
    
    result = return_mutable()
    assert isinstance(result, PVector)


# LLM-generated content at query #9
#--------------------------

def test_mutant():
    """Unit tests for the mutant decorator."""
    
    # Test 1: Basic mutant function that modifies input
    @mutant
    def modify_list(lst):
        lst = lst.assoc(0, 99)
        return lst
    
    original = [1, 2, 3]
    result = modify_list(original)
    assert result == pvector([99, 2, 3])
    assert original == [1, 2, 3]  # Original unchanged
    
    # Test 2: Mutant function with dict argument
    @mutant
    def modify_dict(d):
        d = d.set('key', 'modified')
        return d
    
    original_dict = {'key': 'original'}
    result_dict = modify_dict(original_dict)
    assert result_dict == pmap({'key': 'modified'})
    assert original_dict == {'key': 'original'}  # Original unchanged
    
    # Test 3: Mutant function with multiple arguments
    @mutant
    def combine_lists(lst1, lst2):
        combined = lst1 + lst2
        return combined
    
    list1 = [1, 2]
    list2 = [3, 4]
    result = combine_lists(list1, list2)
    assert len(result) == 4
    assert list1 == [1, 2]
    assert list2 == [3, 4]
    
    # Test 4: Mutant function with nested structures
    @mutant
    def modify_nested(data):
        return data
    
    nested = {'a': [1, 2, {'b': 3}]}
    result = modify_nested(nested)
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    
    # Test 5: Mutant function with kwargs
    @mutant
    def func_with_kwargs(a, b=None):
        if b is None:
            return a
        return a + b
    
    result = func_with_kwargs([1, 2], b=[3, 4])
    assert len(result) == 4
    
    # Test 6: Return value is frozen
    @mutant
    def return_list():
        return [1, 2, 3]
    
    result = return_list()
    assert isinstance(result, PVector)
    
    # Test 7: Function preserves return type through freeze
    @mutant
    def return_dict():
        return {'x': 1, 'y': 2}
    
    result = return_dict()
    assert isinstance(result, PMap)
    
    # Test 8: Mutant with set argument
    @mutant
    def process_set(s):
        return s
    
    original_set = {1, 2, 3}
    result = process_set(original_set)
    assert isinstance(result, PSet)
    assert set(result) == original_set
    
    # Test 9: Function name is preserved
    @mutant
    def named_function(x):
        return x
    
    assert named_function.__name__ == 'named_function'
    
    # Test 10: Mutant with tuple argument
    @mutant
    def process_tuple(t):
        return t
    
    original_tuple = (1, [2, 3])
    result = process_tuple(original_tuple)
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant():
    """Test the mutant decorator for freezing arguments and return values."""
    
    # Test basic mutation isolation
    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst
    
    original = [1, 2, 3]
    result = modify_list(original)
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 999]
    assert original == [1, 2, 3]  # original unchanged
    
    # Test with dict argument
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    original_dict = {'a': 1}
    result_dict = modify_dict(original_dict)
    assert isinstance(result_dict, PMap)
    assert result_dict['new_key'] == 'new_value'
    assert original_dict == {'a': 1}  # original unchanged
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        data['list'].append(100)
        data['nested']['value'] = 999
        return data
    
    original_nested = {'list': [1, 2], 'nested': {'value': 10}}
    result_nested = modify_nested(original_nested)
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['list'], PVector)
    assert original_nested == {'list': [1, 2], 'nested': {'value': 10}}
    
    # Test with multiple arguments
    @mutant
    def combine_args(lst, d, val):
        lst.append(val)
        d['combined'] = lst
        return {'result': d}
    
    arg_list = [1, 2]
    arg_dict = {'x': 10}
    result_multi = combine_args(arg_list, arg_dict, 5)
    assert isinstance(result_multi, PMap)
    assert arg_list == [1, 2]
    assert arg_dict == {'x': 10}
    
    # Test with kwargs
    @mutant
    def modify_with_kwargs(a, b=None):
        if b is None:
            b = {}
        b['modified'] = a
        return b
    
    result_kwargs = modify_with_kwargs([1, 2], b={'initial': 'value'})
    assert isinstance(result_kwargs, PMap)
    assert result_kwargs['initial'] == 'value'
    assert result_kwargs['modified'][0] == 1
    
    # Test function metadata is preserved
    @mutant
    def documented_func(x):
        """This is a documented function."""
        return x
    
    assert documented_func.__name__ == 'documented_func'
    assert documented_func.__doc__ == 'This is a documented function.'
    
    # Test with set argument
    @mutant
    def modify_set(s):
        s.add(999)
        return s
    
    result_set = modify_set({1, 2, 3})
    assert isinstance(result_set, PSet)
    assert 999 in result_set
    
    # Test with tuple argument
    @mutant
    def modify_tuple(t):
        return t + (999,)
    
    result_tuple = modify_tuple((1, 2, 3))
    assert isinstance(result_tuple, tuple)
    assert result_tuple == (1, 2, 3, 999)


# LLM-generated content at query #11
#--------------------------

def test_mutant():
    """Test the mutant decorator"""
    
    # Test basic mutation isolation with mutable arguments
    @mutant
    def modify_list(lst):
        lst = lst.append(4)
        return lst
    
    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3], "Original list should not be modified"
    
    # Test that arguments are frozen before function execution
    @mutant
    def try_mutate_arg(lst):
        # lst should be a pvector, not a list
        assert isinstance(lst, PVector)
        return lst.append(5)
    
    result = try_mutate_arg([1, 2, 3])
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 5]
    
    # Test with dict argument
    @mutant
    def modify_dict(d):
        assert isinstance(d, PMap)
        return d.set('new_key', 'new_value')
    
    original_dict = {'a': 1, 'b': 2}
    result = modify_dict(original_dict)
    assert original_dict == {'a': 1, 'b': 2}, "Original dict should not be modified"
    assert isinstance(result, PMap)
    assert result['new_key'] == 'new_value'
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        assert isinstance(data, PVector)
        assert isinstance(data[0], PMap)
        return data.append({'new': 'item'})
    
    nested = [{'a': 1}, {'b': 2}]
    result = modify_nested(nested)
    assert len(nested) == 2, "Original nested structure should not be modified"
    assert isinstance(result, PVector)
    
    # Test with kwargs
    @mutant
    def modify_with_kwargs(lst, d):
        assert isinstance(lst, PVector)
        assert isinstance(d, PMap)
        return lst.append(d)
    
    result = modify_with_kwargs([1, 2], {'key': 'value'})
    assert isinstance(result, PVector)
    
    # Test return value is frozen
    @mutant
    def return_mutable():
        return [1, 2, 3]
    
    result = return_mutable()
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]
    
    # Test with set argument
    @mutant
    def process_set(s):
        assert isinstance(s, PSet)
        return s.add(4)
    
    original_set = {1, 2, 3}
    result = process_set(original_set)
    assert original_set == {1, 2, 3}
    assert isinstance(result, PSet)
    
    # Test function metadata is preserved
    @mutant
    def documented_function():
        """This is a documented function"""
        return []
    
    assert documented_function.__doc__ == "This is a documented function"


# LLM-generated content at query #12
#--------------------------

def test_mutant():
    """Test the mutant decorator"""
    
    # Test basic functionality - mutant should freeze inputs and outputs
    @mutant
    def modify_list(lst):
        lst = lst.append(4)
        return lst
    
    original_list = [1, 2, 3]
    result = modify_list(original_list)
    # Original should not be modified (frozen)
    assert original_list == [1, 2, 3]
    # Result should be a pvector
    assert isinstance(result, PVector)
    
    # Test with dict argument
    @mutant
    def modify_dict(d):
        d = d.set('new_key', 'new_value')
        return d
    
    original_dict = {'key': 'value'}
    result = modify_dict(original_dict)
    assert original_dict == {'key': 'value'}
    assert isinstance(result, PMap)
    assert result['new_key'] == 'new_value'
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        return data
    
    nested_input = {'list': [1, 2, 3], 'set': {4, 5}}
    result = modify_nested(nested_input)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)
    
    # Test with kwargs
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'x': 1})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    
    # Test that decorated function preserves original function metadata
    @mutant
    def documented_function():
        """This is a documented function"""
        pass
    
    assert documented_function.__doc__ == "This is a documented function"
    assert documented_function.__name__ == "documented_function"
    
    # Test with tuple argument
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)
    
    # Test return value is frozen
    @mutant
    def return_mutable():
        return [1, 2, 3]
    
    result = return_mutable()
    assert isinstance(result, PVector)
    
    # Test with multiple arguments
    @mutant
    def multi_arg(a, b, c):
        return [a, b, c]
    
    result = multi_arg([1], {'x': 2}, {3, 4})
    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[2], PSet)


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant():
    """Test the mutant decorator freezes arguments and return values."""
    
    # Test basic functionality with mutable arguments
    @mutant
    def modify_list(lst):
        # Try to modify the input (should fail because it's frozen)
        lst.append(999)
        return lst
    
    original_list = [1, 2, 3]
    result = modify_list(original_list)
    
    # Result should be a pvector
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]
    # Original should be unchanged
    assert original_list == [1, 2, 3]
    
    # Test with dict arguments
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    original_dict = {'a': 1}
    result = modify_dict(original_dict)
    
    # Result should be a pmap
    assert isinstance(result, PMap)
    assert dict(result) == {'a': 1}
    # Original should be unchanged
    assert original_dict == {'a': 1}
    
    # Test with nested structures
    @mutant
    def modify_nested(data):
        return data
    
    nested_input = {'items': [1, 2, {'inner': 'value'}]}
    result = modify_nested(nested_input)
    
    # Result should be frozen recursively
    assert isinstance(result, PMap)
    assert isinstance(result['items'], PVector)
    assert isinstance(result['items'][2], PMap)
    
    # Test with kwargs
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'x': 10})
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    
    # Test that function name is preserved
    assert modify_list.__name__ == 'modify_list'
    
    # Test with set argument
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, PSet)
    
    # Test with tuple argument
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)


