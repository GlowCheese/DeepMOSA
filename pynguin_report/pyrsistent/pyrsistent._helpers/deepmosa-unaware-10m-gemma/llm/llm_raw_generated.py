####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - ensure return value is frozen
    @mutant
    def identity(x):
        return x

    assert isinstance(identity([1, 2]), PVector)
    assert isinstance(identity({'a': 1}), PMap)

    # Test 2: Ensure arguments are frozen upon entry
    class MutableContainer:
        def __init__(self, value):
            self.value = value
        def append(self, item):
            self.value.append(item)

    @mutant
    def check_args(arg_list, arg_dict):
        # Inside the function, args should already be frozen (PVector and PMap)
        assert isinstance(arg_list, PVector)
        assert isinstance(arg_dict, PMap)
        return arg_list

    mutable_list = [1, 2]
    mutable_dict = {'key': 'value'}
    
    result = check_args(mutable_list, arg_dict=mutable_dict)
    
    # Verify the original objects were not mutated by the decorator's freeze logic
    assert mutable_list == [1, 2]
    assert mutable_dict == {'key': 'value'}
    assert result == pvector([1, 2])

    # Test 3: Mutation inside the function does not affect the input arguments
    @mutant
    def attempt_mutation(arg_list):
        # Note: Even if we try to mutate via standard python list methods, 
        # because the decorator calls freeze() on args before fn(), 
        # arg_list is a PVector. PVector.append/extend don't exist or return new objects.
        # We test if the function can attempt to modify its local view without affecting caller.
        try:
            arg_list.append(3)
        except AttributeError:
            pass # PVector doesn't have append, which is good
        return arg_list

    original_list = [1, 2]
    res = attempt_mutation(original_list)
    assert original_list == [1, 2]
    assert res == pvector([1, 2])

    # Test 4: Deeply nested structures
    @mutant
    def deep_structure(data):
        return data

    nested = [1, {'a': [2, 3]}, (4, {'b': 5})]
    frozen_result = deep_structure(nested)
    
    assert isinstance(frozen_result, PVector)
    assert isinstance(frozen_result[1], PMap)
    assert isinstance(frozen_result[1]['a'], PVector)
    assert isinstance(frozen_result[2][1], PMap)

    # Test 5: Keyword arguments mutation isolation
    @mutant
    def kwarg_test(**kwargs):
        return kwargs

    kw_dict = {'data': [1, 2]}
    res_kw = kwarg_test(data=kw_dict['data'])
    assert isinstance(res_kw, PMap)
    assert isinstance(res_kw['data'], PVector)
    assert kw_dict == {'data': [1, 2]}

    # Test 6: Function with no arguments
    @mutant
    def no_args():
        return None

    assert no_args() is None
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest

def test_mutant():
    # Test Case 1: Basic functionality and freezing of return value
    @mutant
    def add(a, b):
        return a + b
    
    assert add(1, 2) == 3
    assert isinstance(add(1, 2), int)

    # Test Case 2: Verifying input arguments are frozen (list becomes pvector)
    @mutant
    def process_list(data):
        return data
    
    result = process_list([1, 2, 3])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test Case 3: Verifying keyword arguments are frozen (dict becomes pmap)
    @mutant
    def process_kwargs(**kwargs):
        return kwargs
    
    result = process_kwargs(a=1, b=[2, 3])
    assert isinstance(result, PMap)
    assert result['b'] == pvector([2, 3])

    # Test Case 4: Verifying deep freezing (nested structures)
    @mutant
    def identity_complex(data):
        return data
    
    complex_input = {"a": [1, {"b": 2}], "c": (3, 4)}
    result = identity_complex(complex_input)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][1], PMap)
    assert result['a'][1]['b'] == 2
    assert result['c'] == (3, 4)

    # Test Case 5: Verifying mutation attempt within function doesn't affect the returned frozen structure
    # Note: Since the decorator freezes inputs before passing them to fn, 
    # if fn tries to mutate a list, it's actually mutating a PVector which is immutable.
    @mutant
    def attempt_mutation(data):
        try:
            data.append(4)
        except (AttributeError, TypeError):
            pass # PVector doesn't have append or it raises error
        return data

    input_data = [1, 2, 3]
    result = attempt_mutation(input_data)
    assert result == pvector([1, 2, 3])
    assert len(result) == 3

    # Test Case 6: Verifying that the decorator preserves function metadata
    @mutant
    def documented_func():
        """This is a docstring."""
        return True
    
    assert documented_func.__doc__ == "This is a docstring."

    # Test Case 7: Complex nested mutation check
    @mutant
    def mutate_nested(data):
        # data is frozen, so we can't mutate the pmap/pvector directly via standard methods
        return data
    
    input_val = [{"key": "value"}]
    result = mutate_nested(input_val)
    assert isinstance(result, PVector)
    assert isinstance(result[0], PMap)
    assert result[0]['key'] == "value"
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - Verify return value is frozen
    @mutant
    def identity(x):
        return x

    assert isinstance(identity([1, 2]), PVector)
    assert isinstance(identity({'a': 1}), PMap)

    # Test 2: Argument freezing - Verify arguments are frozen before function execution
    # We use a list as an argument and check if it's converted to pvector inside the function
    @mutant
    def check_type(arg):
        return type(arg)

    assert check_type([1, 2]) is PVector
    assert check_type({'a': 1}) is PMap

    # Test 3: Keyword argument freezing - Verify kwargs are frozen
    @mutant
    def check_kwargs(**kwargs):
        return type(kwargs['data'])

    assert check_kwargs(data=[1, 2]) is PVector

    # Test 4: Recursive freezing - Verify nested structures are frozen
    @mutant
    def nested_check(obj):
        return obj

    input_data = [1, {'a': [2, 3]}, (4, [5])]
    result = nested_check(input_data)
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[1]['a'], PVector)
    assert isinstance(result[2][1], PVector)

    # Test 5: Mutation isolation (The core purpose of mutant)
    # The decorator freezes arguments, so even if the function tries to mutate 
    # the internal reference, it's working on a frozen copy.
    # We test that the original input remains unchanged and is not mutated by the function logic.
    
    mutable_list = [1, 2, 3]

    @mutant
    def attempt_mutation(l):
        # This would normally mutate 'l' if it weren't frozen
        # Since it's frozen, l is a PVector, and append/extend don't exist or change the original list.
        # However, we can try to use the function logic to see if the outer scope is safe.
        try:
            l.append(4) # This will fail on PVector in standard python, but we check for no side effects
        except AttributeError:
            pass
        return l

    attempt_mutation(mutable_list)
    assert mutable_list == [1, 2, 3]

    # Test 6: Verifying tuple recursion
    @mutant
    def tuple_test(t):
        return t

    input_tuple = (1, [2])
    result = tuple_test(input_tuple)
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)

    # Test 7: Verifying set behavior (Sets are not recursively frozen per docstring)
    @mutant
    def set_test(s):
        return s

    input_set = {1, 2, 3}
    result = set_test(input_set)
    assert isinstance(result, PSet)
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test case 1: Basic functionality and return value freezing
    @mutant
    def identity(x):
        return x

    assert identity([1, 2]) == pvector([1, 2])
    assert isinstance(identity({'a': 1}), PMap)

    # Test case 2: Mutation isolation (Input arguments are frozen)
    @mutant
    def mutate_input(data_list, data_dict):
        # Try to mutate the list
        try:
            data_list.append(3)
        except (AttributeError, TypeError):
            # PVector doesn't have append in same way or is immutable
            pass
            
        # Try to mutate the dict
        try:
            data_dict['new_key'] = 'new_val'
        except (TypeError, AttributeError):
            pass

        return data_list, data_dict

    initial_list = [1, 2]
    initial_dict = {'a': 1}
    
    result_list, result_dict = mutate_input(initial_list, initial_dict)

    # The inputs passed to the function were frozen, so they couldn't be mutated
    # and the returned values are also frozen.
    assert result_list == pvector([1, 2])
    assert result_dict == pmap({'a': 1})
    
    # Verify original objects in the outer scope remained untouched by any logic inside
    assert initial_list == [1, 2]
    assert initial_dict == {'a': 1}

    # Test case 3: Nested structures
    @mutant
    def nested_mutation(data):
        try:
            data[0]['inner'] = 'changed'
        except (TypeError, AttributeError):
            pass
        return data

    nested_input = [{'inner': 'original'}]
    result = nested_mutation(nested_input)
    
    assert result == pvector([pmap({'inner': 'original'})])
    assert nested_input[0]['inner'] == 'original'

    # Test case 4: Keyword arguments
    @mutant
    def kwarg_test(**kwargs):
        return kwargs

    result_kwarg = kwarg_test(a=[1], b={'c': 2})
    assert result_kwarg == pmap({'a': pvector([1]), 'b': pmap({'c': 2})})

    # Test case 5: Ensure decorator preserves metadata (wraps)
    @mutant
    def documented_func():
        """This is a docstring."""
        return True

    assert documented_func.__doc__ == "This is a docstring."
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic functionality and return value freezing
    @mutant
    def identity(x):
        return x

    assert isinstance(identity([1, 2]), PVector)
    assert isinstance(identity({'a': 1}), PMap)

    # Test mutation isolation: input arguments should remain frozen/immutable
    # even if the function attempts to mutate them (it can't because they are frozen)
    @mutant
    def mutator_function(data_list, data_dict):
        # This would normally raise TypeError on PVector/PMap
        # But since the decorator freezes them first, we check if it works
        try:
            data_list.append(3)
        except (AttributeError, TypeError):
            pass
        
        try:
            data_dict['new_key'] = 'new_val'
        except (AttributeError, TypeError):
            pass
        
        return data_list

    input_list = [1, 2]
    input_dict = {'a': 1}
    
    result = mutator_function(input_list, input_dict)
    
    # The result is frozen by the decorator
    assert isinstance(result, PVector)
    assert result == pvector([1, 2])

    # Test keyword arguments freezing
    @mutant
    def kwarg_test(key_val):
        return key_val

    result_kwarg = kwarg_test(key_val={'inner': [1]})
    assert isinstance(result_kwarg, PMap)
    assert result_kwarg['inner'] == pvector([1])

    # Test nested structure freezing
    @mutant
    def nested_structure(data):
        return data

    complex_input = [1, {'a': [2, 3]}, (4, {5})]
    result_nested = nested_structure(complex_input)
    
    assert isinstance(result_nested, PVector)
    assert isinstance(result_nested[1]['a'], PVector)
    assert isinstance(result_nested[2][1], PMap)
    assert result_nested[1]['a'] == pvector([2, 3])

    # Test that the decorator preserves function metadata
    @mutant
    def decorated_fn():
        """Docstring."""
        return True
    
    assert decorated_fn.__name__ == "decorated_fn"
    assert decorated_fn.__doc__ == "Docstring."

    # Test with complex pyrsistent types already passed in
    @mutant
    def identity_pyrsistent(p_obj):
        return p_obj

    already_frozen = pvector([pmap({'x': 1})])
    result_already_frozen = identity_pyrsistent(already_frozen)
    assert result_already_frozen == already_frozen
    assert isinstance(result_already_frozen[0]['x'], type(1)) # value is primitive, but structure is frozen
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic functionality: mutation inside function should not affect external objects
    # and return value should be frozen.
    
    @mutant
    def mutate_list(data):
        data.append(4)
        return data

    initial_list = [1, 2, 3]
    result = mutate_list(initial_list)

    # The input list should remain unchanged (it was frozen before entering fn)
    assert initial_list == [1, 2, 3]
    # The result should be a PVector and contain the mutation
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3, 4]

    # Test with nested structures (dict/list)
    @mutant
    def mutate_nested(data):
        data['inner'][0] = 'changed'
        return data

    initial_dict = {'inner': [1, 2]}
    result_dict = mutate_nested(initial_dict)

    assert initial_dict == {'inner': [1, 2]}
    assert isinstance(result_dict, PMap)
    assert result_dict['inner'][0] == 'changed'

    # Test with keyword arguments
    @mutant
    def mutate_kwargs(a, b):
        # b is a list passed via kwargs
        b.append('new')
        return {'a': a, 'b': b}

    arg_a = 10
    arg_b = [1]
    result_kwargs = mutate_kwargs(a=arg_a, b=arg_b)

    assert arg_b == [1]
    assert isinstance(result_kwargs, PMap)
    assert result_kwargs['b'] == [1, 'new']

    # Test that non-container types are passed through correctly but frozen (returns same value if immutable)
    @mutant
    def identity_test(x):
        return x

    assert identity_test(5) == 5
    assert identity_test("string") == "string"

    # Test that the decorator preserves metadata (wraps)
    @mutant
    def decorated_fn():
        """Docstring."""
        return None
    
    assert decorated_fn.__doc__ == "Docstring."
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - decorator freezes return value
    @mutant
    def identity(x):
        return x

    original_list = [1, 2, [3]]
    result = identity(original_list)
    
    assert isinstance(result, PVector)
    assert isinstance(result[2], PVector)
    assert result == pvector([1, 2, pvector([3])])

    # Test 2: Verifying input arguments are frozen
    @mutant
    def check_args(a, b, c=None):
        return (a, b, c)

    input_dict = {'key': [1, 2]}
    input_list = [1, {'inner': 2}]
    
    # We can't easily inspect 'a' inside the function without modifying it,
    # but we can verify that the output reflects frozen versions of inputs.
    res_a, res_b, res_c = check_args(input_dict, input_list, [10])
    
    assert isinstance(res_a, PMap)
    assert isinstance(res_a['key'], PVector)
    assert isinstance(res_b, PVector)
    assert isinstance(res_b[1], PMap)
    assert isinstance(res_c, PVector)

    # Test 3: Mutation within the function does not affect the frozen inputs
    @mutant
    def mutate_internal(mutable_list):
        # mutable_list is frozen by the decorator before entering fn
        # Therefore, .append() will fail or we can't change it if it's a PVector
        try:
            mutable_list.append(4)
        except (AttributeError, TypeError):
            pass
        return mutable_list

    my_list = [1, 2, 3]
    result_val = mutate_internal(my_list)
    # The input my_list was converted to pvector([1, 2, 3]) inside the decorator
    # and passed to the function. The result is also a pvector.
    assert result_val == pvector([1, 2, 3])

    # Test 4: Nested structures
    @mutant
    def nested_structure(data):
        return data

    complex_data = {
        "a": [1, {"b": 2}],
        "c": (3, [4])
    }
    result_complex = nested_structure(complex_data)
    
    assert isinstance(result_complex, PMap)
    assert isinstance(result_complex['a'], PVector)
    assert isinstance(result_complex['a'][1], PMap)
    assert isinstance(result_complex['c'][1], PVector)

    # Test 5: Verifying kwargs are frozen
    @mutant
    def check_kwargs(**kwargs):
        return kwargs

    result_kwargs = check_kwargs(x=[1, 2], y={'z': 3})
    assert isinstance(result_kwargs, PMap)
    assert isinstance(result_kwargs['x'], PVector)
    assert isinstance(result_kwargs['y'], PMap)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_mutant():
    # Test case 1: Basic mutation within a decorated function
    # The function modifies a list internally, but because of @mutant,
    # the returned value is frozen and the original input remains unchanged.
    
    shared_list = [1, 2, 3]
    
    @mutant
    def modify_list(l):
        # We attempt to mutate the list by appending an element
        # Note: if freeze works, 'l' is a PVector, which doesn't have .append()
        # So we have to use a trick or observe that the decorator 
        # forces it to be immutable.
        try:
            l.append(4)
        except AttributeError:
            # This block will execute because 'l' is frozen into a PVector
            pass
        return l

    result = modify_list(shared_list)
    
    # Verify result is frozen (PVector)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])
    # Verify original input was not mutated
    assert shared_list == [1, 2, 3]

    # Test case 2: Mutation of a dictionary value
    @mutant
    def mutate_dict(d):
        # Since d is frozen to PMap, we can't do d['a'] = 4
        # But if the function returns a new dict that we try to mutate...
        # The decorator freezes the return value.
        new_d = dict(d)
        new_d['b'] = 2
        return new_d

    input_dict = {'a': 1}
    result_dict = mutate_dict(input_dict)
    
    assert isinstance(result_dict, PMap)
    assert result_dict['a'] == 1
    assert result_dict['b'] == 2
    assert input_dict == {'a': 1}

    # Test case 3: Deeply nested structures
    @mutant
    def deep_mutate(data):
        # data is a PVector containing a PMap
        # We can't mutate, but we test if the returned structure is frozen
        return data

    nested_input = [1, {'key': 'value'}]
    result_nested = deep_mutate(nested_input)
    
    assert isinstance(result_nested, PVector)
    assert isinstance(result_nested[1], PMap)
    assert result_nested[1]['key'] == 'value'

    # Test case 4: Kwargs mutation prevention
    @mutant
    def mutate_kwargs(a, b):
        # Try to modify a dictionary passed as a kwarg
        try:
            b['new_key'] = 'new_val'
        except TypeError:
            pass # PMap doesn't support item assignment
        return a

    input_kwarg = {'existing': True}
    result_kwarg = mutate_kwargs(1, b=input_kwarg)
    
    assert isinstance(result_kwarg, PVector if False else int) # Should be 1
    assert input_kwarg == {'existing': True}

    # Test case 5: Ensuring return value is always frozen regardless of function logic
    @mutant
    def returns_mutable():
        return [1, 2, [3]]

    res = returns_mutable()
    assert isinstance(res, PVector)
    assert isinstance(res[2], PVector)
    assert res[2][0] == 3
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic functionality and freezing of return value
    @mutant
    def identity(x):
        return x

    assert identity([1, 2]) == pvector([1, 2])
    assert isinstance(identity({'a': 1}), PMap)

    # Test that arguments are frozen (cannot be mutated by the function body)
    # We use a list as an argument. If it wasn't frozen, we could append to it.
    # However, since freeze turns it into a pvector, .append() will raise AttributeError
    @mutant
    def attempt_mutation(data):
        try:
            data.append(3)
            return data
        except AttributeError:
            return "caught_mutation"

    assert attempt_mutation([1, 2]) == "caught_mutation"

    # Test complex nested structures
    @mutant
    def complex_func(a, b):
        return {"inner": [a, b]}

    input_list = [1, 2]
    input_dict = {"key": "val"}
    expected = pmap({"inner": pvector([pvector([1, 2]), pmap({"key": "val"})])})
    assert complex_func(input_list, b=input_dict) == expected

    # Test kwargs freezing
    @mutant
    def kwargs_test(**kwargs):
        return kwargs

    result = kwargs_test(x=[1], y={'a': 2})
    assert isinstance(result, PMap)
    assert result['x'] == pvector([1])
    assert result['y'] == pmap({'a': 2})

    # Test that the decorator preserves metadata (wraps)
    @mutant
    def documented_func():
        """Docstring."""
        return True
    
    assert documented_func.__doc__ == "Docstring."

    # Test with primitive types (should remain unchanged but still pass through freeze)
    @mutant
    def primitives(a, b):
        return a + b

    assert primitives(1, 2) == 3
    assert primitives("hello", " world") == "hello world"

    # Test with tuples (recursive freezing)
    @mutant
    def tuple_test(t):
        return t

    assert tuple_test((1, [2])) == (1, pvector([2]))
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test case 1: Basic mutation and freezing of return value
    def simple_mutation(data):
        data.append(4)
        return data

    input_list = [1, 2, 3]
    result = simple_mutation(input_list)
    
    assert isinstance(result, pvector)
    assert result == pmap({}) # This is a placeholder for logic; let's be precise:
    # The decorator freezes args and kwargs, so the function receives a PVector.
    # PVector does not have .append(). 
    # Wait, if the decorator calls freeze(e), then 'data' becomes a PVector.
    # PVector is immutable, so data.append(4) would raise an AttributeError.
    # Let's test a function that works with the frozen types.

    @mutant
    def add_to_dict(d, key, value):
        # d is frozen as PMap, so we can't do d[key] = value
        # We must return a new structure
        return d.set(key, value)

    initial_dict = {'a': 1}
    result_dict = add_to_dict(initial_dict, 'b', 2)
    
    assert isinstance(result_dict, PMap)
    assert result_dict['a'] == 1
    assert result_dict['b'] == 2
    # Verify original input wasn't mutated (though pmap is immutable anyway)
    assert initial_dict == {'a': 1}

    # Test case 2: Deep nesting and freezing of kwargs
    @mutant
    def nested_func(structure, extra):
        # structure is frozen, extra is frozen
        return structure

    nested_input = [1, {'x': 10}]
    extra_input = {'y': [1, 2]}
    result = nested_func(nested_input, extra={'y': [1, 2]})

    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['x'] == 10

    # Test case 3: Verifying that the decorator prevents mutation of arguments via error
    @mutant
    def attempt_mutation(lst):
        lst.append(99)
        return lst

    input_list = [1, 2]
    with pytest.raises(AttributeError):
        # Because the decorator calls freeze(e), input_list becomes a PVector
        # PVector does not have an 'append' method.
        attempt_mutation(input_list)

    # Test case 4: Verifying kwargs freezing
    @mutant
    def check_kwargs(val):
        return val

    # If we pass a list in kwargs, it should be frozen
    result = check_kwargs(val=[1, 2])
    assert isinstance(result, PVector)

    # Test case 5: Identity behavior for primitives
    @mutant
    def identity(x):
        return x

    assert identity(5) == 5
    assert identity("string") == "string"
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - decorator freezes return value
    @mutant
    def identity(x):
        return x

    mutable_list = [1, 2, 3]
    result = identity(mutable_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])
    # Ensure original input wasn't modified by the decorator's internal freeze logic
    assert isinstance(mutable_list, list)

    # Test 2: Mutation isolation - function attempts to mutate input but fails because inputs are frozen
    @mutant
    def mutating_func(data):
        try:
            # This will raise an error or fail to mutate the original object 
            # because 'data' is passed as a PVector (frozen)
            data.append(4)
            return True
        except (AttributeError, TypeError):
            return False

    my_list = [1, 2, 3]
    # The decorator freezes args before passing to fn. 
    # PVector does not have .append() like a list.
    success = mutating_func(my_list)
    assert success is False
    assert my_list == [1, 2, 3]

    # Test 3: Nested structures and kwargs
    @mutant
    def complex_func(a, b=None):
        return {"inner": a, "extra": b}

    input_dict = {"key": [1, 2]}
    input_kwargs = {"b": {"nested": 3}}
    
    result = complex_func(input_dict, b=input_kwargs["b"])
    
    assert isinstance(result, PMap)
    assert result['inner'] == pmap({'key': pvector([1, 2])})
    assert result['extra'] == pmap({'nested': 3})

    # Test 4: Checking that the decorator handles multiple arguments and kwargs correctly
    @mutant
    def multi_arg(x, y, z):
        return [x, y, z]

    res = multi_arg([1], {2: 3}, (4,))
    assert res == pvector([pvector([1]), pmap({2: 3}), (4,)])

    # Test 5: Verify that the function signature/metadata is preserved
    @mutant
    def annotated_fn(x: int) -> int:
        """Docstring."""
        return x

    assert annotated_fn.__name__ == "annotated_fn"
    assert annotated_fn.__doc__ == "Docstring."
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic mutation isolation and freezing of return value
    def incrementer(x):
        # This function attempts to mutate a list passed as an argument
        # Note: in reality, freeze() returns a PVector which is immutable, 
        # so we test if the decorator prevents the original input from being affected 
        # and ensures the return value is frozen.
        mutable_list = [1, 2]
        mutable_list.append(3)
        return x + len(mutable_list)

    @mutant
    def add_and_mutate(val, data):
        data.append(4)
        return val + sum(data)

    # Test Case 1: Return value is frozen (PVector/PMap)
    @mutant
    def return_list():
        return [1, 2, {'a': 3}]
    
    res = return_list()
    assert isinstance(res, PVector)
    assert isinstance(res[2], PMap)
    assert res[2]['a'] == 3

    # Test Case 2: Arguments are frozen (cannot be mutated by the function)
    input_list = [10, 20]
    @mutant
    def mutate_arg(l):
        try:
            l.append(30)
        except (AttributeError, TypeError):
            # PVector does not have append, so it should fail or be immutable
            pass
        return l

    res_list = mutate_arg(input_list)
    assert isinstance(res_list, PVector)
    assert len(input_list) == 2  # Original list remains unchanged
    assert res_list[0] == 10

    # Test Case 3: Keyword arguments are frozen
    @mutant
    def mutate_kwargs(k_arg):
        return k_arg

    input_dict = {'key': 'value'}
    res_kw = mutate_kwargs(k_arg=input_dict)
    assert isinstance(res_kw, PMap)
    assert res_kw['key'] == 'value'
    assert input_dict == {'key': 'value'} # Original dict remains unchanged

    # Test Case 4: Deeply nested structures
    @mutant
    def deep_structure():
        return [[{'z': [1]}]]
    
    res_deep = deep_structure()
    assert isinstance(res_deep, PVector)
    assert isinstance(res_deep[0][0]['z'], PVector)
    assert res_deep[0][0]['z'][0] == 1

    # Test Case 5: Verify functionality with standard types
    @mutant
    def simple_add(a, b):
        return a + b
    
    assert simple_add(1, 2) == 3
    assert isinstance(simple_add(1, 2), int)

    # Test Case 6: Functionality with tuple (recursive freezing)
    @mutant
    def tuple_test():
        return (1, [2, 3])
    
    res_tuple = tuple_test()
    assert isinstance(res_tuple, tuple)
    assert isinstance(res_tuple[1], PVector)

    # Test Case 7: Checking that the decorator preserves function metadata
    @mutant
    def named_func():
        """Docstring."""
        return True
    
    assert named_func.__name__ == "named_func"
    assert named_func.__doc__ == "Docstring."

    # Test Case 8: Complex interaction of all types
    @mutant
    def complex_func(a, b):
        return {'a': a, 'b': b}

    original_arg = [1, 2]
    result = complex_func(a=original_arg, b={'c': 3})
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    assert result['b']['c'] == 3
    assert len(original_arg) == 2 # Ensure original list wasn't mutated
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_mutant():
    # Test case 1: Basic mutation check
    # The decorator should freeze input arguments so they cannot be mutated by fn,
    # and it should also freeze the return value.
    
    def increment_list(data):
        # data is a list passed in. If frozen, this append will fail or 
        # if we are just testing if the returned value is frozen:
        new_data = list(data)
        new_data.append(4)
        return new_data

    decorated_increment = mutant(increment_list)
    
    input_list = [1, 2, 3]
    result = decorated_increment(input_list)
    
    # Assert return value is frozen (PVector instead of list)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    # Assert original input was not mutated in a way that affects the function's logic
    # (though freeze makes it immutable anyway)
    assert input_list == [1, 2, 3]

    # Test case 2: Nested structures and keyword arguments
    def complex_fn(a, b=None):
        res = {'x': a}
        if b:
            res['y'] = b
        return res

    decorated_complex = mutant(complex_fn)
    
    input_dict = {'val': [1, 2]}
    # kwargs is also frozen by the decorator
    result_complex = decorated_complex(a=input_dict, b=[3, 4])
    
    assert isinstance(result_complex, PMap)
    assert result_complex['x'] == pmap({'val': pvector([1, 2])})
    assert result_complex['y'] == pvector([3, 4])

    # Test case 3: Verifying immutability of arguments within the function
    def attempt_mutation(mutable_list):
        try:
            mutable_list.append(99)
        except (AttributeError, TypeError):
            # This is expected because mutable_list is frozen to PVector
            pass
        return mutable_list

    decorated_attempt = mutant(attempt_mutation)
    arg_list = [10, 20]
    result_attempt = decorated_attempt(arg_list)
    
    assert isinstance(result_attempt, PVector)
    assert result_attempt == pvector([10, 20])
    assert len(result_attempt) == 2

    # Test case 4: Identity check for primitives
    @mutant
    def identity_fn(x):
        return x

    assert identity_fn(5) == 5
    assert identity_fn("hello") == "hello"

    # Test case 5: Deeply nested structures
    def deep_nesting(d):
        return d

    @mutant
    def wrap_deep(d):
        return d

    nested_input = {
        'a': [1, {'b': 2}],
        'c': (3, 4)
    }
    
    result_deep = wrap_deep(nested_input)
    
    assert isinstance(result_deep, PMap)
    assert isinstance(result_deep['a'], PVector)
    assert isinstance(result_deep['a'][1], PMap)
    assert isinstance(result_deep['c'], tuple)
    assert result_deep['c'][0] == 3
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_mutant():
    # Setup mutable objects to be passed as arguments
    mutable_list = [1, 2, [3]]
    mutable_dict = {'a': 1, 'b': [2]}
    
    # A function that attempts to mutate its inputs and returns a value
    @mutant
    def mutating_function(arg_list, arg_dict, return_val):
        # Attempting mutation on the arguments passed in
        # Because mutant() calls freeze(), these are actually PVector/PMap
        try:
            arg_list.append(4)
        except (AttributeError, TypeError):
            pass
        
        try:
            arg_dict['c'] = 3
        except (AttributeError, TypeError):
            pass
        
        # Return a mutable structure
        return {'result': return_val, 'inner_list': [10]}

    # Execute the decorated function
    result = mutating_function(mutable_list, mutable_dict, 99)

    # Assertions on the returned value (should be frozen/immutable)
    assert isinstance(result, PMap)
    assert result['result'] == 99
    assert isinstance(result['inner_list'], PVector)
    assert result['inner_list'][0] == 10

    # Assertions on the inputs (should remain unchanged despite mutation attempts)
    # Note: In a real scenario, if we pass the original list, the decorator freezes it.
    # The function receives a PVector, so arg_list.append(4) would fail or not affect 
    # the original 'mutable_list' reference.
    assert mutable_list == [1, 2, [3]]
    assert mutable_dict == {'a': 1, 'b': [2]}

    # Verify deep freezing of kwargs
    @mutant
    def check_kwargs(key_val):
        return key_val

    kwarg_input = {'data': [1, 2]}
    result_kwarg = check_kwargs(data=kwarg_input)
    
    assert isinstance(result_kwarg, PMap)
    assert isinstance(result_kwarg['data'], PVector)
    assert result_kwarg['data'][0] == 1

    # Verify that non-mutable types remain untouched
    @mutant
    def simple_func(x):
        return x

    assert simple_func(5) == 5
    assert simple_func("string") == "string"
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_mutant():
    # Test case 1: Basic functionality and freezing of return value
    @mutant
    def identity(x):
        return x

    mutable_list = [1, 2, 3]
    result = identity(mutable_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])
    # Ensure original input wasn't mutated by the function itself (though identity doesn't mutate)
    assert mutable_list == [1, 2, 3]

    # Test case 2: Mutation inside the function is isolated from the input arguments
    @mutant
    def mutator(data):
        # Convert back to mutable to perform mutation
        mutable = thaw(data)
        mutable.append(4)
        return mutable

    input_list = [1, 2, 3]
    result = mutator(input_list)
    
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]  # Original remains unchanged

    # Test case 3: Nested structures and keyword arguments
    @mutant
    def nested_mutator(outer, inner_val):
        mutable_outer = thaw(outer)
        mutable_inner = thaw(inner_val)
        mutable_outer[0]['new_key'] = 'new_value'
        mutable_inner.append('added')
        return mutable_outer, mutable_inner

    input_dict = [{'a': 1}]
    input_list = [10]
    
    res_dict, res_list = nested_mutator(input_dict, input_list)
    
    assert isinstance(res_dict, PVector)
    assert isinstance(res_dict[0], PMap)
    assert res_dict[0]['new_key'] == 'new_value'
    assert res_list == pvector([10, 'added'])
    
    # Verify inputs were not mutated
    assert input_dict == [{'a': 1}]
    assert input_list == [10]

    # Test case 4: Verifying that kwargs are also frozen
    @mutant
    def kwarg_check(**kwargs):
        return kwargs

    input_kwargs = {'key': [1, 2]}
    result_kwargs = kwarg_check(**input_kwargs)
    
    assert isinstance(result_kwargs['key'], PVector)
    assert result_kwargs['key'] == pvector([1, 2])

    # Test case 5: Ensuring complex nested mutation inside the function doesn't leak
    @mutant
    def complex_mutation(data):
        # data is frozen, so we must thaw to mutate
        mutable = thaw(data)
        if isinstance(mutable, list):
            for item in mutable:
                if isinstance(item, dict):
                    item['mutated'] = True
        return mutable

    complex_input = [{'a': 1}, {'b': 2}]
    result = complex_mutation(complex_input)
    
    assert result[0]['mutated'] is True
    assert complex_input[0] == {'a': 1} # Original input remains pristine
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test case 1: Basic mutation within a decorated function
    # The decorator should freeze inputs and return a frozen result.
    # We'll use a list that we attempt to mutate inside the function.
    
    @mutant
    def modify_list(l):
        # Even though we mutate 'l' locally, the returned value is frozen.
        # Since 'l' was frozen by the decorator before entering, 
        # attempting to append would actually raise an error if we tried l.append().
        # However, we can return a new mutated version.
        new_list = list(l)
        new_list.append(4)
        return new_list

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])

    # Test case 2: Dictionary mutation and nested structures
    @mutant
    def modify_dict(d):
        # Create a mutable copy to perform operations
        new_dict = dict(d)
        new_dict['new_key'] = [10, 20]
        return new_dict

    input_dict = {'a': 1}
    result_dict = modify_dict(input_dict)
    
    assert isinstance(result_dict, PMap)
    assert result_dict['new_key'] == pvector([10, 20])

    # Test case 3: Keyword arguments
    @mutant
    def modify_kwargs(a, b):
        return {'a': a, 'b': b}

    result_kwargs = modify_kwargs(a=[1], b={'x': 2})
    assert isinstance(result_kwargs, PMap)
    assert result_kwargs['a'] == pvector([1])
    assert result_kwargs['b'] == pmap({'x': 2})

    # Test case 4: Verifying that the input itself is frozen upon entry.
    # If the function tries to mutate an input argument using a mutable method,
    # it should fail because 'freeze' was applied to args before the fn call.
    
    @mutant
    def attempt_mutation(l):
        try:
            l.append(99)
            return l
        except (AttributeError, TypeError):
            return "failed"

    input_list_to_fail = [1, 2]
    result_fail = attempt_mutation(input_list_to_fail)
    assert result_fail == "failed"

    # Test case 5: Deep nesting
    @mutant
    def deep_structure(data):
        return data

    nested_input = [1, {'inner': [2, 3]}, (4, 5)]
    result_deep = deep_structure(nested_input)
    
    assert isinstance(result_deep, PVector)
    assert isinstance(result_deep[1], PMap)
    assert isinstance(result_deep[1]['inner'], PVector)
    assert result_deep[2] == (4, 5) # tuple remains tuple per freeze logic
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic functionality and freezing of return value
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert isinstance(add([1], [2]), PVector)

    # Test that arguments are frozen (cannot be mutated by the function)
    @mutant
    def mutate_list(l):
        try:
            l.append(4)
        except AttributeError:
            # PVector does not have append, it has append which returns a new object
            # But mutation via index assignment should fail
            pass
        try:
            l[0] = 99
        except TypeError:
            pass
        return l

    input_list = [1, 2, 3]
    result = mutate_list(input_list)
    assert result == pvector([1, 2, 3])
    assert isinstance(result, PVector)

    # Test complex nested structures
    @mutant
    def process_complex(data):
        return data

    complex_data = {
        "a": [1, 2, {"b": 3}],
        "c": (4, 5),
        "d": {6, 7}
    }
    
    result = process_complex(complex_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result["a"], PVector)
    assert isinstance(result["a"][2], PMap)
    assert result["a"][2]["b"] == 3
    assert isinstance(result["c"], tuple)
    assert isinstance(result["c"][0], int)
    assert isinstance(result["d"], PSet)

    # Test keyword arguments freezing
    @mutant
    def process_kwargs(**kwargs):
        return kwargs

    kwarg_input = {"key": [1, 2]}
    result_kwargs = process_kwargs(**kwarg_input)
    assert isinstance(result_kwargs, PMap)
    assert isinstance(result_kwargs["key"], PVector)

    # Test identity/no-op for simple types
    @mutant
    def identity(x):
        return x

    assert identity(10) == 10
    assert identity("string") == "string"
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - mutation of input is prevented via freezing
    # and return value is frozen.
    def increment_list(l):
        l.append(4)
        return l

    input_list = [1, 2, 3]
    # Note: mutant decorator freezes inputs. Since lists are frozen to pvectors, 
    # the .append() call inside the function will actually fail if it tries 
    # to mutate a PVector (which is immutable), or we test that the 
    # original input remains unchanged if we passed a mutable list.
    
    @mutant
    def mutate_dict(d):
        d['new_key'] = 'new_value'
        return d

    original_dict = {'a': 1}
    result = mutate_dict(original_dict)

    # The input dict should not have been modified because it was frozen to a PMap
    assert 'new_key' not in original_dict
    # The result should be a pmap (frozen)
    assert isinstance(result, PMap)
    assert result['a'] == 1

    # Test 2: Nested structures
    @mutant
    def complex_mutation(data):
        # Attempting to mutate nested list inside a dict
        # Since freeze is recursive, data['list'] becomes a pvector
        data['inner']['val'] = 99
        return data

    complex_input = {'inner': {'val': 1}, 'list': [1, 2]}
    result = complex_mutation(complex_input)

    assert complex_input['inner']['val'] == 1
    assert isinstance(result['inner'], PMap)
    assert result['inner']['val'] == 99
    assert isinstance(result['list'], PVector)

    # Test 3: Keyword arguments
    @mutant
    def mutate_kwargs(**kwargs):
        kwargs['added'] = True
        return kwargs

    kwarg_input = {'existing': 'value'}
    result_kwargs = mutate_kwargs(existing='value')
    
    assert 'added' not in kwarg_input
    assert isinstance(result_kwargs, PMap)
    assert result_kwargs['added'] is True

    # Test 4: Verify immutability of the returned object (it must be frozen)
    @mutant
    def returns_list():
        return [1, 2, [3]]

    result_val = returns_list()
    assert isinstance(result_val, PVector)
    assert isinstance(result_val[2], PVector)

    # Test 5: Check that non-container types pass through correctly
    @mutant
    def identity(x):
        return x

    assert identity(10) == 10
    assert identity("string") == "string"
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_mutant():
    # Test case 1: Simple mutation inside a decorated function
    # The function modifies a list, but because of @mutant, the returned value is frozen
    @mutant
    def increment_list(l):
        l.append(4)
        return l

    input_list = [1, 2, 3]
    result = increment_list(input_list)
    
    # The original input should remain unchanged (if we assume the decorator freezes args)
    # Note: In the provided implementation, freeze(e) is called on args.
    # Therefore, 'l' inside the function becomes a PVector.
    # PVectors do not have an .append() method in pyrsistent, they return a new object.
    # However, let's test based on the logic provided in the snippet.
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3]) # Since append doesn't work on PVector, it stays same

    # Test case 2: Testing mutation of a mutable object passed as an argument
    # We use a dictionary to see if the returned value is frozen.
    @mutant
    def update_dict(d, key, value):
        # Note: Since args are frozen, d becomes a PMap. 
        # Calling d.set() or similar would be needed for pyrsistent.
        # If we try to mutate it like a dict (e.g., d[key] = value), 
        # it will raise an error because PMap is immutable.
        # The decorator's purpose is to prevent mutation from leaking out.
        new_d = d.set(key, value)
        return new_d

    initial_dict = {'a': 1}
    result_map = update_dict(initial_dict, 'b', 2)
    
    assert isinstance(result_map, PMap)
    assert result_map['a'] == 1
    assert result_map['b'] == 2

    # Test case 3: Checking kwargs are also frozen
    @mutant
    def check_kwargs(data):
        return data

    result_kwarg = check_kwargs(data=[1, 2])
    assert isinstance(result_kwarg, PVector)
    assert result_kwarg == pvector([1, 2])

    # Test case 4: Verifying that the decorator handles nested structures
    @mutant
    def complex_structure(obj):
        return obj

    nested_input = {"list": [1, {"inner": 2}], "tuple": (3, 4)}
    result_complex = complex_structure(nested_input)

    assert isinstance(result_complex, PMap)
    assert isinstance(result_complex["list"], PVector)
    assert isinstance(result_complex["list"][1], PMap)
    assert result_complex["list"][1]["inner"] == 2
    assert result_complex["tuple"] == (3, 4)

    # Test case 5: Ensuring the function's return value is frozen even if not modified
    @mutant
    def returns_list():
        return [10, 20]

    result_simple = returns_list()
    assert isinstance(result_simple, PVector)
    assert result_simple == pvector([10, 20])
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_mutant():
    # Test Case 1: Basic functionality - decorator freezes inputs and return value
    @mutant
    def identity(x):
        return x

    mutable_list = [1, 2, 3]
    result = identity(mutable_list)

    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])
    # Ensure the original input was not mutated by the decorator's internal logic
    assert isinstance(mutable_list, list)

    # Test Case 2: Deeply nested mutation isolation
    @mutant
    def mutate_internal(data):
        # This function tries to mutate an element inside a list if it were allowed
        # Since the decorator freezes inputs, data is already a PVector
        new_data = data.append(4) # PVector.append returns a new object
        return new_data

    input_data = [1, {'a': 2}]
    # Note: freeze converts input to pvector([1, pmap({'a': 2})])
    result_data = mutate_internal(input_data)
    
    assert isinstance(result_data, PVector)
    assert result_data[1]['a'] == 2
    assert len(result_data) == 3

    # Test Case 3: Keyword arguments freezing
    @mutant
    def check_kwargs(val=None):
        return val

    kwarg_dict = {'key': [1, 2]}
    result_kwarg = check_kwargs(val=kwarg_dict)
    
    assert isinstance(result_kwarg, PMap)
    assert result_kwarg['key'] == pvector([1, 2])

    # Test Case 4: Verifying that the function itself cannot mutate arguments
    @mutant
    def attempt_mutation(l):
        try:
            l[0] = 99
        except TypeError:
            pass # Expected because l is a PVector (immutable)
        return l

    original_list = [1, 2, 3]
    result_val = attempt_mutation(original_list)
    assert result_val[0] == 1
    assert original_list[0] == 1

    # Test Case 5: Tuple recursion
    @mutant
    def tuple_test(t):
        return t

    input_tuple = (1, [2, 3])
    result_tuple = tuple_test(input_tuple)
    assert isinstance(result_tuple, tuple)
    assert isinstance(result_tuple[1], PVector)
    assert result_tuple[1][0] == 2

    # Test Case 6: Set behavior (not recursive per implementation)
    @mutant
    def set_test(s):
        return s

    input_set = {1, 2, frozenset([3])}
    result_set = set_test(input_set)
    assert isinstance(result_set, PSet)
    # Check that the elements themselves are not frozen recursively if they are sets/frozensets
    # (though frozenset is immutable anyway, we check the container type)
    assert 1 in result_set
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - decorator freezes return value
    @mutant
    def identity(x):
        return x

    data = [1, 2, {"a": 3}]
    result = identity(data)
    
    assert isinstance(result, PVector)
    assert isinstance(result[2], PMap)
    assert result[2]["a"] == 3

    # Test 2: Ensures input arguments are frozen (mutation attempt inside function)
    @mutant
    def mutating_func(mutable_list, mutable_dict):
        # Attempt to mutate inputs
        mutable_list.append(4)
        mutable_dict["new"] = "value"
        return mutable_list

    original_list = [1, 2, 3]
    original_dict = {"a": 1}
    
    # The decorator freezes args before they reach the function body.
    # Therefore, 'mutable_list' inside the function is a PVector.
    # PVector does not have an .append() method like list (it uses .append which returns a new object).
    # However, even if we use methods that don't crash, the original input remains untouched.
    
    try:
        result = mutating_func(original_list, original_dict)
        # Check that the original objects were not mutated
        assert original_list == [1, 2, 3]
        assert original_dict == {"a": 1}
    except AttributeError:
        # If .append() failed because it's now a PVector, that also proves freezing worked
        pass

    # Test 3: Complex nested structures
    @mutant
    def complex_structure(structure):
        return structure

    nested = {
        "a": [1, 2, {"b": 3}],
        "c": (4, 5, [6])
    }
    
    result = complex_structure(nested)
    
    assert isinstance(result, PMap)
    assert isinstance(result["a"], PVector)
    assert isinstance(result["a"][2], PMap)
    assert isinstance(result["c"], tuple)
    assert isinstance(result["c"][2], PVector)

    # Test 4: Keyword arguments are also frozen
    @mutant
    def kwarg_test(**kwargs):
        return kwargs

    result = kwarg_test(a=[1, 2], b={'x': 10})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)

    # Test 5: Verify that freezing preserves values but changes types to immutable ones
    @mutant
    def type_check(val):
        return val

    assert isinstance(type_check([1, 2]), PVector)
    assert isinstance(type_check({"a": [1]}), PMap)
    assert isinstance(type_check((1, [2])), tuple)
    assert isinstance(type_check({1, 2}), PSet)
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_mutant():
    # Test case 1: Basic mutation check (Decorator should freeze inputs and return frozen output)
    def simple_increment(x):
        # This function attempts to mutate a list passed as an argument
        x.append(2)
        return x

    input_list = [1]
    decorated_inc = mutant(simple_increment)
    
    # The decorator freezes inputs, so 'x' inside simple_increment is a PVector.
    # PVector does not have an .append() method that mutates in-place like list.
    # However, the decorator implementation uses: freeze(fn(*[freeze(e) for e in args], ...))
    # If we pass a list, it becomes a PVector. 
    # Calling x.append(2) on a PVector will raise an AttributeError.
    with pytest.raises(AttributeError):
        decorated_inc(input_list)

    # Test case 2: Verifying return value is frozen
    def returns_mutable(x):
        return [x, {'a': 1}]

    decorated_ret = mutant(returns_mutable)
    result = decorated_ret(10)
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[0] == 10
    assert result[1]['a'] == 1

    # Test case 3: Verifying keyword arguments are frozen
    def check_kwargs(val=None):
        return val

    decorated_kw = mutant(check_kwargs)
    result_kw = decorated_kw(val=[1, 2])
    assert isinstance(result_kw, PVector)

    # Test case 4: Deeply nested structures
    def deep_structure(data):
        return data

    complex_input = {"a": [1, {"b": 2}], "c": (3, 4)}
    decorated_deep = mutant(deep_structure)
    result_deep = decorated_deep(complex_input)

    assert isinstance(result_deep, PMap)
    assert isinstance(result_deep['a'], PVector)
    assert isinstance(result_deep['a'][1], PMap)
    assert result_deep['c'] == (3, 4)
    assert isinstance(result_deep['c'][0], int) # tuple elements are not frozen unless they are containers

    # Test case 5: Ensuring the original input remains untouched if it were mutable
    mutable_dict = {'key': 'value'}
    def identity(d):
        return d
    
    decorated_id = mutant(identity)
    _ = decorated_id(mutable_dict)
    assert mutable_dict == {'key': 'value'} # Original stays same

    # Test case 6: Testing functionality with defaultdict (should be converted to pmap)
    import collections
    dd = collections.defaultdict(list, {'a': [1]})
    def return_data(d):
        return d
    
    decorated_dd = mutant(return_data)
    result_dd = decorated_dd(dd)
    assert isinstance(result_dd, PMap)
    assert result_dd['a'] == pvector([1])
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - mutation inside function is isolated from original input
    mutable_list = [1, 2, 3]
    mutable_dict = {'a': 1}

    @mutant
    def mutate_inputs(l, d):
        l.append(4)
        d['b'] = 2
        return l, d

    result_l, result_d = mutate_inputs(mutable_list, mutable_dict)

    # Verify input remains unchanged (frozen by decorator)
    assert mutable_list == [1, 2, 3]
    assert mutable_dict == {'a': 1}

    # Verify result is frozen/immutable and contains the changes
    assert isinstance(result_l, PVector)
    assert list(result_l) == [1, 2, 3, 4]
    assert isinstance(result_d, PMap)
    assert result_d['b'] == 2

    # Test 2: Deep mutation in nested structures
    nested_list = [{'key': 'val'}]

    @mutant
    def mutate_nested(lst):
        lst[0]['key'] = 'changed'
        return lst

    result_nested = mutate_nested(nested_list)

    # Original should be unchanged (the decorator freezes the input before function runs)
    assert nested_list == [{'key': 'val'}]
    # Result should be frozen and updated
    assert result_nested[0]['key'] == 'changed'
    assert isinstance(result_nested, PVector)

    # Test 3: Kwargs mutation
    @mutant
    def mutate_kwargs(**kwargs):
        kwargs['new_key'] = 'new_val'
        return kwargs

    kwarg_dict = {'old_key': 'old_val'}
    result_kwargs = mutate_kwargs(old_key='old_val') # passed via decorator logic essentially

    # Note: The decorator freezes kwargs items. 
    # Let's test passing a mutable object as a kwarg specifically.
    mutable_kwarg_obj = {'inner': 1}
    
    @mutant
    def mutate_kwarg_param(obj):
        obj['inner'] = 2
        return obj

    result_kwarg_param = mutate_kwarg_param(obj=mutable_kwarg_obj)
    assert mutable_kwarg_obj == {'inner': 1}
    assert result_kwarg_param['inner'] == 2
    assert isinstance(result_kwarg_param, PMap

    # Test 4: Function returning multiple frozen types
    @mutant
    def return_complex():
        return [1, {2: 3}, (4, 5)]

    res = return_complex()
    assert isinstance(res, PVector)
    assert isinstance(res[1], PMap)
    assert isinstance(res[2], tuple)
    assert res[1][2] == 3
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - ensures return value is frozen
    @mutant
    def identity(x):
        return x

    assert isinstance(identity([1, 2]), PVector)
    assert isinstance(identity({'a': 1}), PMap)

    # Test 2: Verifies that arguments are frozen upon entry
    # We use a mutable object and check if it's converted inside the function
    @mutant
    def check_frozen(arg):
        return arg

    result = check_frozen([1, 2])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2])

    # Test 3: Verifies that keyword arguments are frozen
    @mutant
    def check_kwargs(kwarg_val):
        return kwarg_val

    result_kwarg = check_kwargs(kwarg_val={'key': 'value'})
    assert isinstance(result_kwarg, PMap)
    assert result_kwarg['key'] == 'value'

    # Test 4: Verifies deep freezing (nested structures)
    @mutant
    def nested_func():
        return [[1, 2], {'a': [3, 4]}]

    result_nested = nested_func()
    assert isinstance(result_nested, PVector)
    assert isinstance(result_nested[0], PVector)
    assert isinstance(result_nested[1], PMap)
    assert isinstance(result_nested[1]['a'], PVector)

    # Test 5: Verifies that mutation of the input inside the function is isolated
    # (Since arguments are frozen, attempting to mutate them via list methods 
    # should either fail or not affect the original outside the scope)
    input_list = [1, 2, 3]

    @mutant
    def attempt_mutation(l):
        try:
            l.append(4)
        except (AttributeError, TypeError):
            # PVector does not have append (it has append which returns a new object)
            # But even if it did, the original input_list remains unchanged
            pass
        return l

    result = attempt_mutation(input_list)
    assert result == pvector([1, 2, 3])
    assert input_list == [1, 2, 3]

    # Test 6: Verifies tuple recursion (tuples remain tuples but elements are frozen)
    @mutant
    def tuple_test(t):
        return t

    result_tuple = tuple_test((1, [2, 3]))
    assert isinstance(result_tuple, tuple)
    assert isinstance(result_tuple[1], PVector)

    # Test 7: Verifies set behavior (sets are not recursively frozen)
    @mutant
    def set_test(s):
        return s

    result_set = set_test({1, 2, 3})
    assert isinstance(result_set, PSet)
    assert result_set == pset({1, 2, 3})
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test 1: Basic functionality and freezing of return value
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert isinstance(add(1, 2), int)

    # Test 2: Verify arguments are frozen (input mutation protection)
    # We use a list as an argument which is mutable
    @mutant
    def process_list(l):
        # Inside the function, l should be a PVector
        assert isinstance(l, PVector)
        return l

    input_list = [1, 2, 3]
    result = process_list(input_list)
    
    # The original input list should remain a standard list (unaffected by the function's internal logic)
    assert isinstance(input_list, list)
    # The result returned by the decorator must be frozen
    assert isinstance(result, PVector)

    # Test 3: Verify nested structures are frozen
    @mutant
    def complex_structure(d, l):
        return {"data": d, "items": l}

    input_dict = {"key": [1, 2]}
    input_list = [{"nested": 10}]
    
    result = complex_structure(input_dict, input_list)
    
    assert isinstance(result, PMap)
    assert isinstance(result["data"], PMap)
    assert isinstance(result["items"], PVector)
    assert isinstance(result["items"][0], PMap)
    assert result["data"]["key"] == pvector([1, 2])

    # Test 4: Verify kwargs are frozen
    @mutant
    def check_kwargs(**kwargs):
        return kwargs

    input_kwargs = {"a": [1, 2], "b": {"c": 3}}
    result = check_kwargs(**input_kwargs)
    
    assert isinstance(result, PMap)
    assert isinstance(result["a"], PVector)
    assert isinstance(result["b"], PMap)

    # Test 5: Verify that mutation attempt inside the function doesn't affect original inputs
    # (Note: Since arguments are frozen into PVector/PMap, they cannot be mutated in-place anyway)
    @mutant
    def try_mutate(v):
        # v is a PVector, so v[0] = 99 would raise an error. 
        # We check that the function can operate on the frozen version.
        return v

    original_v = [1, 2, 3]
    result_v = try_mutate(original_v)
    assert result_v == pvector([1, 2, 3])
    assert original_v == [1, 2, 3]

    # Test 6: Verify identity for primitives
    @mutant
    def identity(x):
        return x

    assert identity(5) == 5
    assert identity("string") == "string"
    assert identity(None) is None
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic functionality and freezing of return value
    @mutant
    def identity(x):
        return x

    assert identity([1, 2]) == pvector([1, 2])
    assert isinstance(identity({'a': 1}), PMap)

    # Test mutation isolation (input arguments are frozen)
    @mutant
    def mutate_list(l):
        # Attempting to mutate the input list should fail or be impossible 
        # because 'l' is a pvector, not a list.
        try:
            l.append(3)
        except (AttributeError, TypeError):
            pass
        return l

    result = mutate_list([1, 2])
    assert result == pvector([1, 2])
    assert len(result) == 2

    # Test nested structure mutation prevention
    @mutant
    def mutate_nested_dict(d):
        # d is a PMap. Keys/values are frozen.
        try:
            # This would fail on a PMap anyway, but we verify the structure remains immutable
            d['new_key'] = 'new_val'
        except (AttributeError, TypeError):
            pass
        return d

    input_dict = {'a': [1, 2]}
    result_dict = mutate_nested_dict(input_dict)
    assert result_dict == pmap({'a': pvector([1, 2])})
    assert 'new_key' not in result_dict

    # Test kwargs freezing
    @mutant
    def check_kwargs(val=None):
        return val

    assert check_kwargs(val=[10]) == pvector([10])

    # Test complex nested structure preservation and freezing
    @mutant
    def complex_fn(data):
        return data

    complex_input = {
        'a': [1, {'b': 2}],
        'c': (3, 4),
        'd': {5, 6}
    }
    
    expected_output = pmap({
        'a': pvector([1, pmap({'b': 2})]),
        'c': (3, 4),
        'd': pset({5, 6})
    })
    
    assert complex_fn(complex_input) == expected_output

    # Test that the decorator preserves metadata (wraps)
    @mutant
    def documented_fn():
        """Docstring."""
        return True
    
    assert documented_fn.__doc__ == "Docstring."

    # Verify that even if a function tries to modify an object, 
    # the returned value is always frozen.
    @mutant
    def sneaky_mutation(l):
        # If 'l' were a standard list, we could do l.append(4)
        # Since it's a pvector, append returns a new object, but if 
        # the function returned the original reference, mutant ensures it is frozen.
        return l

    original_list = [1, 2]
    res = sneaky_mutation(original_list)
    assert isinstance(res, PVector)
    assert res == pvector([1, 2])
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - decorator freezes return value
    @mutant
    def identity(x):
        return x

    input_list = [1, 2, 3]
    result = identity(input_list)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test 2: Verifying mutation protection for arguments
    # We use a list as an argument. The decorator freezes it before the function runs.
    # If the function tries to mutate 'args', it's mutating a PVector, not the original list.
    mutation_tracker = {"mutated": False}

    @mutant
    def attempt_mutation(mutable_list, mutable_dict):
        try:
            # Attempting to mutate the frozen structure (PVector/PMap) 
            # will either fail or create a new object without affecting the original.
            mutable_list[0] = 999
        except Exception:
            pass
        
        try:
            mutable_dict['key'] = 'new_value'
        except Exception:
            pass
            
        return mutable_list, mutable_dict

    original_list = [1, 2, 3]
    original_dict = {'a': 1}
    
    res_list, res_dict = attempt_mutation(original_list, mutable_dict=original_dict)

    # The original objects should remain unchanged
    assert original_list == [1, 2, 3]
    assert original_dict == {'a': 1}
    
    # The returned objects should be frozen
    assert isinstance(res_list, PVector)
    assert isinstance(res_dict, PMap)

    # Test 3: Complex nested structures
    @mutant
    def complex_func(data):
        return data

    nested_data = {
        'a': [1, {'b': 2}],
        'c': (3, 4)
    }
    
    result_nested = complex_func(nested_data)
    
    assert isinstance(result_nested, PMap)
    assert isinstance(result_nested['a'], PVector)
    assert isinstance(result_nested['a'][1], PMap)
    assert result_nested['c'] == (3, 4)
    assert isinstance(result_nested['c'][0], int)

    # Test 4: Kwargs freezing
    @mutant
    def check_kwargs(**kwargs):
        return kwargs

    kwarg_input = {'x': [10]}
    res_kwargs = check_kwargs(**kwarg_input)
    
    assert isinstance(res_kwargs, PMap)
    assert isinstance(res_kwargs['x'], PVector)
    assert kwarg_input['x'] == [10] # Original remains mutable list

    # Test 5: Functionality with sets (not recursively frozen per spec)
    @mutant
    def set_func(s):
        return s

    input_set = {1, 2, 3}
    result_set = set_func(input_set)
    assert isinstance(result_set, PSet)
    assert result_set == pset({1, 2, 3})
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test basic functionality and freezing of return value
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert add([1], [2]) == pvector([1, 2])

    # Test that arguments are frozen upon entry
    @mutant
    def mutate_args(data_list, data_dict):
        # Attempting to mutate the input (if they were standard lists/dicts)
        # Since mutant freezes them, we can't actually mutate the original objects 
        # in a way that affects the caller, but we test if the function receives P-types.
        data_list.append(4) 
        data_dict['new'] = 5
        return data_list, data_dict

    input_list = [1, 2, 3]
    input_dict = {'a': 1}
    
    result_list, result_dict = mutate_args(input_list, input_dict)
    
    # The return value must be frozen (PVector and PMap)
    assert isinstance(result_list, PVector)
    assert isinstance(result_dict, PMap)
    assert result_list == pvector([1, 2, 3, 4])
    assert result_dict == pmap({'a': 1, 'new': 5})

    # Test that the original objects passed to the function remain unchanged 
    # because the decorator freezes them before the function body executes.
    assert input_list == [1, 2, 3]
    assert input_dict == {'a': 1}

    # Test kwargs freezing
    @mutant
    def check_kwargs(val=None):
        return val

    assert check_kwargs(val=[10]) == pvector([10])

    # Test deep nesting
    @mutant
    def deep_nesting(data):
        return data

    nested_input = {"a": [1, {"b": 2}], "c": (3, 4)}
    result = deep_nesting(nested_input)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][1], PMap)
    assert result['a'][1]['b'] == 2
    assert result['c'] == (3, 4)

    # Test with existing pyrsistent objects
    @mutant
    def identity(o):
        return o

    pv = pvector([1, 2])
    pm = pmap({'x': 1})
    ps = pset([1, 2])

    assert identity(pv) == pv
    assert identity(pm) == pm
    # Note: freeze/thaw logic for PSet returns a standard set via the implementation details
    assert identity(ps) == {1, 2}
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - ensures return value is frozen
    @mutant
    def identity(x):
        return x

    assert isinstance(identity([1, 2]), PVector)
    assert isinstance(identity({'a': 1}), PMap)

    # Test 2: Ensures input arguments are frozen
    @mutant
    def check_types(arg_list, arg_dict):
        return type(arg_list), type(arg_dict)

    list_type, dict_type = check_types([1, 2], {'a': 3})
    assert list_type is PVector
    assert dict_type is PMap

    # Test 3: Ensures keyword arguments are frozen
    @mutant
    def check_kwargs(kwarg_list):
        return type(kwarg_list)

    kwarg_type = check_kwargs(kwarg_list=[1, 2])
    assert kwarg_type is PVector

    # Test 4: Ensures mutation within the function does not affect the outside world 
    # (via the fact that inputs are frozen and cannot be mutated in place)
    @mutant
    def attempting_mutation(mutable_list):
        try:
            mutable_list.append(3)
        except (AttributeError, TypeError):
            # PVector does not have .append() like list, 
            # or it returns a new object, so the original reference is safe.
            pass
        return mutable_list

    original_list = [1, 2]
    result = attempting_mutation(original_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2])
    assert original_list == [1, 2]  # Original remains untouched

    # Test 5: Deep nesting mutation prevention
    @mutant
    def deep_mutation(nested_dict):
        try:
            nested_dict['key'] = 'new_value'
        except (TypeError, AttributeError):
            pass
        return nested_dict

    original_nested = {'inner': {'a': 1}}
    result = deep_mutation(original_nested)
    
    assert isinstance(result, PMap)
    assert result['inner'] == pmap({'a': 1})
    assert original_nested == {'inner': {'a': 1}}

    # Test 6: Function metadata preservation (wraps)
    @mutant
    def decorated_fn():
        """Docstring for test."""
        return True

    assert decorated_fn.__doc__ == "Docstring for test."
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic functionality and frozen return value
    @mutant
    def identity(x):
        return x

    assert isinstance(identity([1, 2]), PVector)
    assert isinstance(identity({'a': 1}), PMap)

    # Test that arguments are frozen upon entry
    @mutant
    def check_args_frozen(data_list, data_dict):
        assert isinstance(data_list, PVector)
        assert isinstance(data_dict, PMap)
        return True

    assert check_args_frozen([1, 2], {'a': 3}) is True

    # Test that mutation inside the function does not affect the original object
    # (Since the function receives frozen versions, it cannot mutate the original list/dict)
    @mutant
    def attempt_mutation(mutable_list):
        # Even if we try to append, we are working on a PVector copy
        try:
            mutable_list.append(4)
        except (AttributeError, TypeError):
            pass
        return mutable_list

    original_list = [1, 2, 3]
    result = attempt_mutation(original_list)
    assert result == pvector([1, 2, 3])
    assert original_list == [1, 2, 3]

    # Test keyword arguments freezing
    @mutant
    def check_kwargs_frozen(**kwargs):
        assert isinstance(kwargs, PMap)
        return True

    assert check_kwargs_frozen(a=[1], b={'x': 2}) is True

    # Test nested structures
    @mutant
    def complex_structure(data):
        assert isinstance(data, PVector)
        assert isinstance(data[0], PMap)
        assert isinstance(data[0]['inner'], PVector)
        return data

    nested_input = [{'inner': [10]}]
    result = complex_structure(nested_input)
    assert result == pvector([pmap({'inner': pvector([10])})])

    # Test decorator preserves metadata (wraps)
    @mutant
    def decorated_fn():
        """Docstring."""
        return None
    
    assert decorated_fn.__doc__ == "Docstring."
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test basic functionality and freezing of args/kwargs
    def simple_fn(a, b, c=None):
        return {"a": a, "b": b, "default": c}

    # Case 1: Basic types (int)
    res = simple_fn(1, 2)
    assert res == pmap({"a": 1, "b": 2, "default": None})
    assert isinstance(res, PMap)

    # Case 2: Nested mutable structures in args and kwargs
    def nested_fn(data, extra):
        # Inside the function, data and extra are already frozen
        return {"data": data, "extra": extra}

    input_list = [1, {"key": "val"}]
    input_dict = {"item": [10, 20]}
    
    res = nested_fn(input_list, extra=input_dict)
    
    expected = pmap({
        "data": pvector([1, pmap({"key": "val"})]),
        "extra": pmap({"item": pvector([10, 20])})
    })
    assert res == expected

    # Case 3: Verifying that mutations inside the function do not affect the original input
    # because inputs are frozen before the function body executes.
    class MutableWrapper:
        def __init__(self, value):
            self.value = value
        def __getitem__(self, key):
            return self.value[key]
        def items(self):
            return self.value.items()

    def mutation_fn(mutable_dict):
        # Since mutable_dict is frozen by @mutant, we can't actually mutate it 
        # via standard dict assignment; if we try to treat it like a dict,
        # the decorator has already converted it to a PMap.
        return mutable_dict

    original_dict = {"a": [1, 2]}
    res = mutation_fn(original_dict)
    assert isinstance(res, PMap)
    assert res["a"] == pvector([1, 2])

    # Case 4: Verifying return value is frozen even if function returns a standard list/dict
    def return_mutable(x):
        return {"result": [x, {"inner": 1}]}

    res = return_mutable(5)
    assert isinstance(res, PMap)
    assert isinstance(res["result"], PVector)
    assert isinstance(res["result"][1], PMap)

    # Case 5: Verifying that the decorator preserves metadata (wraps)
    def decorated_fn(x):
        """Docstring."""
        return x
    
    decorated_fn = mutant(decorated_fn)
    assert decorated_fn.__doc__ == "Docstring."

    # Case 6: Complex nested structure
    def complex_structure(x):
        return x

    complex_input = {
        "list": [1, (2, 3), {"a": 4}],
        "set": {5, 6},
        "tuple": (7, [8])
    }
    
    res = complex_structure(complex_input)
    assert res["list"][0] == 1
    assert res["list"][1] == (2, 3)
    assert isinstance(res["list"][2], PMap)
    assert isinstance(res["set"], PSet)
    assert res["tuple"][1] == pvector([8])
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest

def test_mutant():
    # Test Case 1: Basic functionality - decorator freezes return value
    @mutant
    def identity(x):
        return x

    mutable_list = [1, 2, 3]
    result = identity(mutable_list)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test Case 2: Decorator freezes arguments (args and kwargs)
    @mutant
    def check_args(a, b, c=None):
        return a, b, c

    arg_list = [10]
    kwarg_dict = {'key': 'value'}
    res_a, res_b, res_c = check_args(arg_list, 20, c=kwarg_dict)

    assert isinstance(res_a, PVector)
    assert isinstance(res_c, PMap)
    assert res_a[0] == 10
    assert res_c['key'] == 'value'

    # Test Case 3: Deep freezing of nested structures
    @mutant
    def deep_structure(data):
        return data

    nested_input = [
        {"inner_list": [1, {"a": 2}]},
        (3, 4)
    ]
    result_nested = deep_structure(nested_input)

    assert isinstance(result_nested, PVector)
    assert isinstance(result_nested[0], PMap)
    assert isinstance(result_nested[0]['inner_list'], PVector)
    assert isinstance(result_nested[0]['inner_list'][1], PMap)
    assert isinstance(result_nested[1], tuple)
    assert result_nested[1][0] == 3

    # Test Case 4: Mutation isolation (The function body can mutate its local copies, 
    # but the returned value and original inputs are frozen)
    @mutant
    def mutation_attempt(data):
        # The decorator freezes 'data' before fn is called.
        # In Python, even if we try to modify a list inside the function,
        # the mutated version is what gets returned as a PVector.
        local_copy = list(data)
        local_copy.append(99)
        return local_copy

    original_input = [1, 2]
    result_mutated = mutation_attempt(original_input)
    
    assert result_mutated == pvector([1, 2, 99])
    # Verify the original input was not affected (though mutant decorator freezes it anyway)
    assert original_input == [1, 2]

    # Test Case 5: Empty containers
    @mutant
    def empty_containers():
        return [], {}, set()

    res_v, res_m, res_s = empty_containers()
    assert isinstance(res_v, PVector) and len(res_v) == 0
    assert isinstance(res_m, PMap) and len(res_m) == 0
    assert isinstance(res_s, PSet) and len(res_s) == 0
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_mutant():
    # Setup a mutable object to track mutations
    mutable_list = [1, 2, 3]
    mutable_dict = {'a': 1}

    # Case 1: Function that mutates its arguments
    @mutant
    def mutating_fn(x, y):
        # Attempting to mutate the input list/dict directly
        # Note: because mutant freezes args first, x and y will be PVector/PMap
        # Therefore, direct mutation like x.append(4) would raise an AttributeError
        # But we test that the return value is frozen and inputs remain effectively immutable
        x_as_list = list(x)
        x_as_list.append(4)
        return x_as_list

    result1 = mutating_fn(mutable_list, mutable_dict)
    assert isinstance(result1, PVector)
    assert result1 == pvector([1, 2, 3, 4])
    # Verify original input was not affected (it should be frozen by the decorator before fn runs)
    assert mutable_list == [1, 2, 3]

    # Case 2: Function that mutates a dictionary via kwargs
    @mutant
    def mutating_kwargs_fn(**kwargs):
        d = dict(kwargs)
        d['new_key'] = 'new_val'
        return d

    result2 = mutating_kwargs_fn(data=mutable_dict)
    assert isinstance(result2, PMap)
    assert result2['new_key'] == 'new_val'
    # Original dict should remain unchanged
    assert 'new_key' not in mutable_dict

    # Case 3: Nested structures
    @mutant
    def nested_fn(data):
        return data

    nested_input = [1, {'inner': [2, 3]}]
    result3 = nested_fn(nested_input)
    assert isinstance(result3, PVector)
    assert isinstance(result3[1], PMap)
    assert isinstance(result3[1]['inner'], PVector)
    assert result3[1]['inner'][0] == 2

    # Case 4: Verifying non-container types remain unchanged
    @mutant
    def identity_fn(x):
        return x

    assert identity_fn(5) == 5
    assert identity_fn("string") == "string"

    # Case 5: Check that mutation of the return value is impossible if it returns a mutable object
    @mutant
    def returning_mutable():
        return [1, 2] # The decorator will freeze this to pvector([1, 2])

    result4 = returning_mutable()
    assert isinstance(result4, PVector)
    with pytest.raises(AttributeError):
        result4.append(3)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - decorator freezes return value
    @mutant
    def identity(x):
        return x
    
    assert isinstance(identity([1, 2]), PVector)
    assert isinstance(identity({'a': 1}), PMap)

    # Test 2: Mutation isolation - function body cannot mutate input arguments
    # We use a list and attempt to append to it inside the function
    @mutant
    def attempt_mutation(mutable_list):
        mutable_list.append(4)
        return mutable_list

    original_list = [1, 2, 3]
    result = attempt_mutation(original_list)
    
    # The input 'original_list' should remain unchanged because it was frozen before fn called
    assert original_list == [1, 2, 3]
    # The result is a PVector containing the appended value (as seen by the function)
    assert result == pvector([1, 2, 3, 4])

    # Test 3: Kwargs freezing
    @mutant
    def check_kwargs(data=None):
        return data

    res_kwargs = check_kwargs(data={'key': 'value'})
    assert isinstance(res_kwargs, PMap)
    assert res_kwargs['key'] == 'value'

    # Test 4: Nested structures
    @mutant
    def nested_mutation(structure):
        structure[0]['inner'] = 'changed'
        return structure

    complex_struct = [{'inner': 'original'}]
    result_struct = nested_mutation(complex_struct)
    
    # The original input should be untouched
    assert complex_struct[0]['inner'] == 'original'
    # The returned value is frozen, so the mutation inside the function 
    # (which operates on a frozen copy passed in) is reflected in the return PVector
    assert result_struct[0]['inner'] == 'changed'
    assert isinstance(result_struct, PVector)

    # Test 5: Tuple preservation and freezing
    @mutant
    def tuple_test(t):
        return t

    res_tuple = tuple_test((1, [2]))
    assert isinstance(res_tuple, tuple)
    assert isinstance(res_tuple[1], PVector)

    # Test 6: No mutation possible on frozen objects (Side effect check)
    @mutant
    def side_effect_check(d):
        try:
            d['new'] = 'value'
        except Exception:
            pass # PMap will raise TypeError on item assignment
        return d

    input_dict = {'existing': True}
    result_dict = side_effect_check(input_dict)
    assert 'new' not in result_dict
    assert isinstance(result_dict, PMap)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test 1: basic functionality - decorator freezes return value
    @mutant
    def identity(x):
        return x

    res = identity([1, 2, 3])
    assert isinstance(res, PVector)
    assert res == pvector([1, 2, 3])

    # Test 2: decorator freezes input arguments (args and kwargs)
    @mutant
    def check_types(arg_list, arg_dict, arg_kwarg):
        return (type(arg_list), type(arg_dict), type(arg_kwarg))

    types = check_types([1, 2], {'a': 3}, arg_kwarg=[4, 5])
    assert types == (PVector, PMap, PVector)

    # Test 3: deep freezing of nested structures
    @mutant
    def complex_structure(data):
        return data

    nested = [{"a": [1, 2]}, {"b": (3, 4)}]
    res_nested = complex_structure(nested)
    assert isinstance(res_nested, PVector)
    assert isinstance(res_nested[0], PMap)
    assert isinstance(res_nested[0]["a"], PVector)
    assert isinstance(res_nested[1]["b"], tuple)

    # Test 4: Mutation isolation (verifying the function works even if it tries to mutate,
    # although since inputs are frozen, actual mutation of args would raise an error.
    # The decorator's purpose is to ensure that even if the original object was mutable,
    # the function sees a frozen version).
    
    mutable_list = [1, 2, 3]

    @mutant
    def attempt_mutation(l):
        try:
            l.append(4)
            return l
        except (AttributeError, TypeError):
            # pyrsistent PVector doesn't have .append() like list does
            return l

    res_mutation = attempt_mutation(mutable_list)
    assert isinstance(res_mutation, PVector)
    assert len(res_mutation) == 3
    assert mutable_list == [1, 2, 3] # Original remains untouched

    # Test 5: Verifying kwargs are frozen
    @mutant
    def check_kwargs(key_val):
        return type(key_val)

    assert check_kwargs(key_val={'inner': [1]}) == PMap

    # Test 6: Verify sets are handled (sets are not recursively frozen by freeze)
    @mutant
    def set_test(s):
        return s

    res_set = set_test({1, 2, 3})
    assert isinstance(res_set, PSet)
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_mutant():
    # Test case 1: Basic mutation inside function should be isolated from input
    def mutating_fn(data):
        # data is a list, we attempt to mutate it
        # Since mutant() freezes args, this will actually fail if we try .append() 
        # on a PVector, but let's test the logic of what the decorator does.
        # The decorator freezes input, so 'data' becomes a PVector.
        # We can't mutate PVector in place via append, but we can return a mutated version.
        new_data = data.append(4) 
        return new_data

    initial_list = [1, 2, 3]
    result = mutating_fn(initial_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    assert initial_list == [1, 2, 3] # Original remains untouched

    # Test case 2: Mutation of dict values inside function
    def mutating_dict(d):
        # d is a PMap. We can't mutate in place, but we return new structure.
        new_d = d.set('new_key', 'value')
        return new_d

    initial_dict = {'a': 1}
    result_dict = mutating_dict(initial_dict)
    
    assert isinstance(result_dict, PMap)
    assert result_dict['new_key'] == 'value'
    assert initial_dict == {'a': 1}

    # Test case 3: Testing kwargs freezing and mutation
    def mutating_kwargs(a, b):
        # b is a dict. We try to add a key.
        # Because of mutant, b is frozen. We must use set/new object.
        b_updated = b.set('added', True)
        return b_updated

    initial_kwargs = {'existing': 123}
    result_kwargs = mutating_kwargs(1, b=initial_kwargs)
    
    assert isinstance(result_kwargs, PMap)
    assert result_kwargs['added'] is True
    assert initial_kwargs == {'existing': 123}

    # Test case 4: Deeply nested structures
    def deep_mutation(structure):
        # structure is pvector([pmap({'inner': [1]])])
        # We return a version where the innermost list has an element added.
        # Note: since it's frozen, we have to navigate via pyrsistent API
        inner_map = structure[0]
        inner_list = inner_map['inner']
        new_inner_list = inner_list.append(2)
        new_inner_map = inner_map.set('inner', new_inner_list)
        return structure.set(0, new_inner_map)

    nested_input = [{'inner': [1]}]
    result_nested = deep_mutation(nested_input)
    
    assert result_nested[0]['inner'] == pvector([1, 2])
    assert nested_input[0]['inner'] == [1]

    # Test case 5: Verify return value is always frozen even if function returns normal list
    @mutant
    def returns_list():
        return [1, 2, {'a': 3}]

    result_unfrozen = returns_list()
    assert isinstance(result_unfrozen, PVector)
    assert isinstance(result_unfrozen[2], PMap)
    assert result_unfrozen[2]['a'] == 3

    # Test case 6: Verify identity of immutable types remains same
    @mutant
    def returns_int():
        return 42

    assert returns_int() == 42
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_freeze():
    # Test primitive types
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) is True
    assert freeze(None) is None

    # Test lists and pvectors
    assert isinstance(freeze([1, 2, 3]), PVector)
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([]) == pvector([])

    # Test dicts and pmaps
    assert isinstance(freeze({'a': 1}), PMap)
    assert freeze({'a': 1, 'b': [2, 3]}) == pmap({'a': 1, 'b': pvector([2, 3])})
    assert freeze({}) == pmap({})

    # Test tuples (recursive freezing of elements)
    assert freeze((1, [2])) == (1, pvector([2]))
    assert freeze((1, {'a': 2})) == (1, pmap({'a': 2}))

    # Test sets and pssets (not recursive per docstring)
    assert isinstance(freeze({1, 2}), PSet)
    assert freeze({1, 2}) == pset([1, 2])
    # Note: set elements are not recursively frozen by design in the code provided
    
    # Test defaultdict
    import collections
    dd = collections.defaultdict(list, {'a': [1, 2]})
    frozen_dd = freeze(dd)
    assert isinstance(frozen_dd, PMap)
    assert frozen_dd['a'] == pvector([1, 2])

    # Test strict=False behavior (does not recurse into containers)
    # Based on implementation: if strict=False, it only checks typ is list/dict/etc.
    # But the recursive calls inside the branches use freeze(v, strict).
    # Therefore, strict=False affects how PMap/PVector are handled as inputs 
    # but doesn't stop recursion for native types.
    assert freeze([1, [2]], strict=False) == pvector([1, pvector([2])])

    # Test complex nested structure
    complex_obj = {
        'a': [1, 2, {'c': 3}],
        'b': (4, 5),
        'd': {6, 7}
    }
    expected = pmap({
        'a': pvector([1, 2, pmap({'c': 3})]),
        'b': (4, 5),
        'd': pset([6, 7])
    })
    assert freeze(complex_obj) == expected

def test_mutant():
    @mutant
    def adder(a, b):
        return a + b

    assert adder(1, 2) == 3
    assert isinstance(adder([1], [2]), PVector)

    @mutant
    def complex_fn(data):
        # data is frozen, so we can't mutate it directly without error if it were a real list,
        # but here the decorator freezes it into a pvector.
        return data

    result = complex_fn([1, 2])
    assert isinstance(result, PVector)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test basic functionality: decorator freezes inputs and return value
    @mutant
    def identity(x):
        return x

    mutable_list = [1, 2, 3]
    result = identity(mutable_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    # Test that function cannot mutate input arguments because they are frozen
    @mutant
    def attempt_mutate(lst):
        try:
            lst[0] = 99
            return False
        except Exception:
            return True

    assert attempt_mutate([1, 2, 3]) is True

    # Test mutation of nested structures within kwargs
    @mutant
    def check_kwargs(data):
        return data

    mutable_dict = {"a": [1, 2]}
    result_dict = check_kwargs(data=mutable_dict)
    assert isinstance(result_dict, PMap)
    assert isinstance(result_dict['a'], PVector)
    assert result_dict['a'][0] == 1

    # Test that the function itself can perform logic but results are frozen
    @mutant
    def complex_logic(d):
        # This works because we create NEW objects, not mutating the frozen ones
        new_val = d['x'] + 1
        return {'x': new_val}

    input_data = {'x': 10}
    output = complex_logic(input_data)
    assert output == pmap({'x': 11})
    assert isinstance(output, PMap)

    # Test with multiple arguments and mixed types
    @mutant
    def multi_arg(a, b, c):
        return [a, b, c]

    res = multi_arg([1], {2: 3}, (4,))
    assert res == pvector([pvector([1]), pmap({2: 3}), (4,)])

    # Test that the decorator preserves metadata (wraps)
    @mutant
    def decorated_fn():
        """Docstring."""
        return None
    
    assert decorated_fn.__doc__ == "Docstring."
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - decorator freezes return value
    @mutant
    def identity(x):
        return x

    original_list = [1, 2, [3]]
    result = identity(original_list)
    
    assert isinstance(result, PVector)
    assert isinstance(result[2], PVector)
    assert result == pvector([1, 2, pvector([3])])

    # Test 2: Verifying that input arguments are frozen inside the function
    @mutant
    def check_frozen(arg_list, arg_dict):
        # Inside the function, inputs should already be pyrsistent types
        is_list_frozen = isinstance(arg_list, PVector)
        is_dict_frozen = isinstance(arg_dict, PMap)
        return is_list_frozen and is_dict_frozen

    assert check_frozen([1, 2], {'a': 1}) is True

    # Test 3: Verifying keyword arguments are frozen
    @mutant
    def check_kwargs(val=None):
        return isinstance(val, PVector)

    assert check_kwargs(val=[1, 2]) is True

    # Test 4: Ensuring deep mutation isolation (the decorated function receives frozen objects)
    # This test checks if the decorator works on nested structures
    @mutant
    def deep_structure(data):
        # data should be fully recursively frozen
        return data

    nested_input = {"a": [1, {"b": 2}], "c": (3, 4)}
    result = deep_structure(nested_input)

    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][1], PMap)
    assert result['a'][1]['b'] == 2
    assert result['c'] == (3, 4) # Tuple remains tuple but elements are frozen

    # Test 5: Testing with simple types (no change expected)
    @mutant
    def simple_type(x):
        return x

    assert simple_type(10) == 10
    assert simple_type("string") == "string"

    # Test 6: Ensuring the decorator preserves function metadata
    @mutant
    def metadata_test():
        """Docstring test."""
        return True
    
    assert metadata_test.__doc__ == "Docstring test."

    # Test 7: Verifying that mutation of the original input after calling does not affect result
    # (Though the decorator freezes inputs *before* they reach the function, 
    # we verify the returned value is an immutable snapshot)
    @mutant
    def capture_state(mutable_list):
        return mutable_list

    my_list = [1, 2, 3]
    captured = capture_state(my_list)
    my_list.append(4)
    
    assert len(captured) == 3
    assert captured == pvector([1, 2, 3])
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test Case 1: Basic functionality and return value freezing
    @mutant
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    assert isinstance(add(1, 2), int)

    # Test Case 2: Verifying arguments are frozen (input mutation isolation)
    @mutant
    def mutate_and_check(data_list, data_dict):
        # Attempt to mutate the inputs
        if isinstance(data_list, list):
            data_list.append(4)
        if isinstance(data_dict, dict):
            data_dict['new_key'] = 'new_val'
        return data_list

    initial_list = [1, 2, 3]
    initial_dict = {'a': 1}
    
    # The decorator freezes inputs before the function runs.
    # Therefore, inside the function, data_list is a PVector.
    # PVector does not have an .append() method that mutates in place (it returns a new one),
    # and if we tried to use list-specific mutation on a PVector, it would raise AttributeError.
    # However, let's test if the return value is frozen and the input remains unchanged 
    # as seen by the caller.
    
    result = mutate_and_check(initial_list, initial_dict)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3]) # The internal mutation attempt on a PVector would fail or not affect the original list
    assert initial_list == [1, 2, 3]
    assert initial_dict == {'a': 1}

    # Test Case 3: Verifying nested structures are frozen
    @mutant
    def complex_structure(nested):
        return nested

    input_data = {"a": [1, 2], "b": {"c": 3}}
    result = complex_structure(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    assert result['a'][0] == 1
    assert result['b']['c'] == 3

    # Test Case 4: Verifying kwargs are frozen
    @mutant
    def check_kwargs(**kwargs):
        return kwargs

    result = check_kwargs(x=[1], y={'z': 2})
    assert isinstance(result, PMap)
    assert isinstance(result['x'], PVector)
    assert isinstance(result['y'], PMap)

    # Test Case 5: Verifying tuple recursion (tuples are not converted to pvector, but their contents are)
    @mutant
    def tuple_test(t):
        return t

    input_tuple = (1, [2, 3])
    result = tuple_test(input_tuple)
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest

def test_mutant():
    # Test case 1: Basic mutation isolation
    # The function attempts to mutate a list passed as an argument.
    # If mutant works, the original list should remain unchanged because it was frozen.
    original_list = [1, 2, 3]
    
    @mutant
    def mutating_func(data):
        # In a real scenario, if data were a standard list, this would mutate it.
        # But mutant freezes the input first.
        # To test if mutation 'leaks', we check if the original object is modified.
        # Since freeze converts list to pvector, internal mutation of elements 
        # or structural change isn't possible on the original object.
        return data

    result = mutating_func(original_list)
    
    assert result == pvector([1, 2, 3])
    assert original_list == [1, 2, 3]
    assert isinstance(result, PVector)

    # Test case 2: Mutation of nested structures within the function
    @mutant
    def nested_mutation_func(data):
        # data is a pmap here. We can't mutate it, but we can return a modified version.
        # The decorator ensures the return value is also frozen.
        new_val = data.set('new_key', 'new_value')
        return new_val

    input_dict = {'a': 1}
    result_dict = nested_mutation_func(input_dict)
    
    assert result_dict == pmap({'a': 1, 'new_key': 'new_value'})
    assert isinstance(result_dict, PMap)
    assert input_dict == {'a': 1}

    # Test case 3: Keyword arguments mutation
    @mutant
    def kwarg_mutation_func(item):
        return item

    input_kwargs = {'data': [10, 20]}
    result_kwargs = kwarg_mutation_func(data=[10, 20])
    
    assert result_kwargs == pvector([10, 20])
    # Note: The original list passed as a kwarg is not mutated because it was frozen.

    # Test case 4: Deeply nested structures
    @mutant
    def deep_structure_func(data):
        return data

    deep_input = [1, {'a': [2, 3]}, (4, 5)]
    result_deep = deep_structure_func(deep_input)

    assert result_deep == pvector([1, pmap({'a': pvector([2, 3])}), (4, pvector([5]))])
    # Note: tuple elements are frozen recursively as per freeze implementation
    assert isinstance(result_deep[1]['a'], PVector)
    assert isinstance(result_deep[2][0], int)

    # Test case 5: Verifying that the return value is always frozen
    @mutant
    def returns_mutable(data):
        return [data, {'key': 'val'}]

    res = returns_mutable(1)
    assert isinstance(res, PVector)
    assert isinstance(res[1], PMap)
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic functionality: mutation inside should not affect original
    # and return value should be frozen.
    
    def increment_list(lst):
        # Attempt to mutate the list in place
        lst.append(4)
        return lst

    original_list = [1, 2, 3]
    result = increment_list(original_list)

    # The original input should remain unchanged (frozen by decorator)
    assert original_list == [1, 2, 3]
    # The return value should be frozen (PVector) and contain the mutation
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3] # Because the input was frozen before the fn ran

    # Test with dictionary mutation
    @mutant
    def mutate_dict(d):
        d['new_key'] = 'new_value'
        return d

    original_dict = {'a': 1}
    result_dict = mutate_dict(original_dict)

    assert original_dict == {'a': 1}
    assert isinstance(result_dict, PMap)
    assert result_dict['new_key'] == 'new_value'

    # Test with keyword arguments
    @mutant
    def mutate_kwargs(val, extra=None):
        # This is a bit of a trick: the decorator freezes kwargs too.
        # We can't mutate 'extra' if it's already frozen, but we can return things.
        return val

    assert isinstance(mutate_kwargs(1, extra={'a': 1}), PMap) == False # Just checking type safety
    
    # Test complex nested structure
    @mutant
    def complex_mutation(data):
        # data is frozen, so we can't mutate it, but let's see if return is frozen
        return data

    nested_input = [1, {'a': [2, 3]}, (4, 5)]
    result_nested = complex_mutation(nested_input)

    assert isinstance(result_nested, PVector)
    assert isinstance(result_nested[1], PMap)
    assert isinstance(result_nested[1]['a'], PVector)
    assert isinstance(result_nested[2], tuple)
    assert result_nested[2][0] == 4

    # Test that it preserves function metadata (wraps)
    @mutant
    def identity(x):
        """Docstring."""
        return x
    
    assert identity.__doc__ == "Docstring."

    # Test with multiple arguments and kwargs
    @mutant
    def multi_arg(a, b, c=None):
        return a

    assert multi_arg(1, 2, c=3) == 1
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic functionality and freezing of return value
    @mutant
    def identity(x):
        return x

    assert identity([1, 2]) == pvector([1, 2])
    assert isinstance(identity({'a': 1}), PMap)

    # Test that arguments are frozen (cannot be mutated by the function)
    @mutant
    def mutator_func(data):
        # Attempt to mutate a list if it weren't frozen
        # Since it is frozen, this would raise TypeError on standard lists 
        # but here we check if the input received is already pvector
        data.append(3)
        return data

    # If mutant works, the input [1, 2] becomes pvector([1, 2])
    # PVector append returns a new object and doesn't mutate in place
    result = mutator_func([1, 2])
    assert result == pvector([1, 2, 3])

    # Test nested structures
    @mutant
    def nested_func(d, l):
        return {**d, 'new': l}

    input_dict = {'a': [1]}
    input_list = [2, 3]
    expected = pmap({'a': pvector([1]), 'new': pvector([2, 3])})
    assert nested_func(input_dict, input_list) == expected

    # Test kwargs freezing
    @mutant
    def kwarg_func(val=None):
        return val

    assert kwarg_func(val=[10]) == pvector([10])

    # Test that the decorator preserves metadata (wraps)
    @mutant
    def documented_func():
        """Docstring."""
        return True
    
    assert documented_func.__doc__ == "Docstring."

    # Test side effect isolation: 
    # The function itself might try to mutate, but it only mutates its local frozen copy.
    # We verify the output is a frozen version of the logic's intent.
    @mutant
    def complex_logic(items):
        # items is actually a PVector here due to mutant
        new_items = items
        # This line attempts an in-place mutation which fails on PVector
        try:
            new_items.append(4)
        except (TypeError, AttributeError):
            pass 
        return new_items

    assert complex_logic([1, 2]) == pvector([1, 2])
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic functionality and freezing of return value
    @mutant
    def identity(x):
        return x

    assert identity([1, 2]) == pvector([1, 2])
    assert isinstance(identity({'a': 1}), PMap)

    # Test that arguments are frozen (cannot be mutated by the function)
    @mutant
    def mutator(data_list, data_dict):
        # Attempting to mutate input
        try:
            data_list.append(3)
        except (TypeError, AttributeError):
            pass
        
        try:
            data_dict['new_key'] = 'new_val'
        except (TypeError, AttributeError):
            pass
            
        return data_list

    input_list = [1, 2]
    input_dict = {'a': 1}
    result = mutator(input_list, input_dict)
    
    # The result should be a pvector and the original inputs should not have been mutated if they were converted
    assert result == pvector([1, 2])
    assert isinstance(result, PVector)

    # Test nested structures
    @mutant
    def nested_func(structure):
        return structure

    nested_input = [1, {'a': [2, 3]}, (4, 5)]
    expected_output = pvector([1, pmap({'a': pvector([2, 3])}), (4, 5)])
    assert nested_func(nested_input) == expected_output

    # Test keyword arguments
    @mutant
    def kwarg_func(a, b):
        return {'res': a + b}

    assert kwarg_func(a=10, b=20) == pmap({'res': 30})

    # Test that the decorator preserves metadata (wraps)
    @mutant
    def documented_func():
        """Docstring."""
        return True
    
    assert documented_func.__doc__ == "Docstring."

    # Test complex mutation attempt inside function
    @mutant
    def nested_mutation(data):
        # If we try to mutate a value that was frozen into a PMap
        if isinstance(data, PMap) and 'val' in data:
            # This line will fail if strict freezing worked because PMap is immutable
            # We catch it to ensure the test passes but confirms immutability
            try:
                data['val'] = 99
            except Exception:
                pass
        return data

    initial_data = {'val': 1}
    result_data = nested_mutation(initial_data)
    assert result_data['val'] == 1
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest

def test_freeze():
    # Test simple types
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True
    assert freeze(None) is None

    # Test lists/pvectors
    assert isinstance(freeze([1, 2]), PVector)
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([]) == pvector([])

    # Test dicts/pmaps
    assert isinstance(freeze({'a': 1}), PMap)
    assert freeze({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})
    assert freeze({}) == pmap({})

    # Test defaultdict
    dd = collections.defaultdict(list, {'a': [1], 'b': [2]})
    assert isinstance(freeze(dd), PMap)
    assert freeze(dd) == pmap({'a': pvector([1]), 'b': pvector([2])})

    # Test sets/psets (not recursive)
    assert isinstance(freeze({1, 2}), PSet)
    assert freeze({1, 2}) == pset([1, 2])
    
    # Test tuples (recursive)
    assert isinstance(freeze((1, [2])), tuple)
    assert freeze((1, [2], {'a': 3})) == (1, pvector([2]), pmap({'a': 3}))

    # Test strict=False behavior for PMap/PVector (should not recurse into them if already frozen)
    pv = pvector([1, [2]])
    # If strict is False, the inner list remains a list if it's not explicitly traversed by type check
    # However, per implementation: if typ is PVector and strict=True, it maps freeze.
    # If strict=False, PVector doesn't trigger the 'list' block, so it returns itself.
    assert freeze(pv, strict=False) == pv
    
    pm = pmap({'a': [1]})
    # With strict=False, the dict/pmap block only triggers if type is dict. 
    # Since pm is PMap, it skips the first two 'if' blocks and hits nothing, returning itself.
    assert freeze(pm, strict=False) == pm

def test_thaw():
    # Test pvector to list
    assert thaw(pvector([1, [2]])) == [1, [2]]
    assert thaw([1, [2]]) == [1, [2]]
    
    # Test pmap to dict
    assert thaw(pmap({'a': 1, 'b': [2]})) == {'a': 1, 'b': [2]}
    assert thaw({'a': 1, 'b': [2]}) == {'a': 1, 'b': [2]}

    # Test pset to set
    assert thaw(pset([1, 2])) == {1, 2}

    # Test tuple recursion
    assert thaw((1, pvector([2]))) == (1, [2])

def test_mutant():
    @mutant
    def add_to_list(l, val):
        # This is a trick: we can't mutate the original because it was frozen
        # But we can try to append to the local version.
        # The decorator freezes inputs, so 'l' is a PVector. 
        # PVector.append returns a new object; it doesn't mutate in-place.
        return l.append(val)

    original_list = [1, 2]
    result = add_to_list(original_list, 3)
    
    assert result == pvector([1, 2, 3])
    assert original_list == [1, 2] # Original remains untouched
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - decorator freezes return value
    @mutant
    def identity(x):
        return x

    assert isinstance(identity([1, 2]), PVector)
    assert isinstance(identity({'a': 1}), PMap)

    # Test 2: Verifying arguments are frozen upon entry
    class MutableList(list):
        pass

    @mutant
    def check_args(arg_list, arg_dict):
        # Inside the function, args should already be frozen
        assert isinstance(arg_list, PVector)
        assert isinstance(arg_dict, PMap)
        return arg_list

    m_list = MutableList([1, 2])
    m_dict = {'key': 'value'}
    result = check_args(m_list, m_dict)
    
    assert isinstance(result, PVector)
    assert result[0] == 1

    # Test 3: Verifying kwargs are frozen
    @mutant
    def check_kwargs(**kwargs):
        assert isinstance(kwargs, PMap)
        return kwargs

    kwarg_res = check_kwargs(a=[1], b={'nested': [2]})
    assert kwarg_res['a'] == 1
    assert kwarg_res['b']['nested'] == 2

    # Test 4: Verifying mutation inside the function does not affect the original object
    # (Because the function works on a frozen copy of the inputs)
    shared_list = [1, 2, 3]

    @mutant
    def mutate_internal(l):
        # We attempt to mutate the argument. 
        # Since 'l' is a PVector, l.append() doesn't exist or returns a new object.
        # However, if we try to use standard list methods on what we think is a list:
        try:
            # This will fail because l is a PVector, not a list
            l.append(4)
        except AttributeError:
            pass
        return l

    result = mutate_internal(shared_list)
    assert result == pvector([1, 2, 3])
    assert shared_list == [1, 2, 3]

    # Test 5: Deeply nested structures
    @mutant
    def deep_structure(data):
        return data

    nested = {"a": [1, {"b": 2}], "c": (3, 4)}
    frozen_result = deep_structure(nested)
    
    assert isinstance(frozen_result, PMap)
    assert isinstance(frozen_result['a'], PVector)
    assert isinstance(frozen_result['a'][1], PMap)
    assert frozen_result['a'][1]['b'] == 2
    assert frozen_result['c'] == (3, 4)

    # Test 6: Functionality with tuples (recursive freezing/thawing)
    @mutant
    def tuple_test(t):
        return t

    input_tuple = (1, [2, 3])
    output_tuple = tuple_test(input_tuple)
    assert isinstance(output_tuple, tuple)
    assert isinstance(output_tuple[1], PVector)
```


