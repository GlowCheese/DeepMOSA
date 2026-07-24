####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_freeze():
    # Test basic types
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True
    assert freeze(None) is None

    # Test lists to pvector
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([{'a': 1}]) == pvector([pmap({'a': 1})])

    # Test dicts to pmap
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})
    assert freeze({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})
    
    # Test defaultdict to pmap
    from collections import defaultdict
    dd = defaultdict(list, {'a': [1], 'b': [2]})
    assert freeze(dd) == pmap({'a': pvector([1]), 'b': pvector([2])})

    # Test sets to pset (not recursive)
    assert freeze({1, 2, 3}) == pset([1, 2, 3])
    # Note: as per docstring, set elements are not recursively frozen
    # If we had a set containing a list (which is unhashable in standard python), 
    # we wouldn't even be able to create the input. 
    # But we can test that it returns a pset.
    assert isinstance(freeze({1, 2}), PSet)

    # Test tuples (recursive)
    assert freeze((1, 2)) == (1, 2)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))
    assert freeze((1, {'a': 2})) == (1, pmap({'a': 2}))

    # Test strict=False behavior for PMap/PVector
    # If strict is False, PMap/PVector should not be recursively traversed
    pm = pmap({'a': [1, 2]})
    # With strict=True (default), it recurses into the list
    assert freeze(pm) == pmap({'a': pvector([1, 2])})
    # With strict=False, it should treat the PMap as an opaque object and not recurse into its values
    # However, the current implementation of freeze(o, strict=False) for PMap 
    # actually doesn't trigger the 'if typ is dict' block, so it returns 'o' as is.
    assert freeze(pm, strict=False) == pm

    # Test deep nesting
    deep_input = [1, {'a': (2, [3, {'d': 4}])}]
    expected_output = pvector([1, pmap({'a': (2, pvector([3, pmap({'d': 4}])]))})])
    assert freeze(deep_input) == expected_output

def test_thaw():
    # Test pvector to list
    assert thaw(pvector([1, 2])) == [1, 2]
    assert thaw(pvector([1, pvector([2])])) == [1, [2]]
    
    # Test pmap to dict
    assert thaw(pmap({'a': 1})) == {'a': 1}
    assert thaw(pmap({'a': pmap({'b': 2})})) == {'a': {'b': 2}}
    
    # Test pset to set
    assert thaw(pset([1, 2])) == {1, 2}
    
    # Test tuple recursion
    assert thaw((1, pvector([2]))) == (1, [2])

def test_mutant():
    @mutant
    def add_to_list(lst):
        # This function receives a frozen pvector
        # We can't mutate it in place, but we test if the return is frozen
        return lst

    input_list = [1, 2]
    result = add_to_list(input_list)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2])

    @mutant
    def modify_dict(d):
        return d

    input_dict = {'a': 1}
    result_dict = modify_dict(input_dict)
    assert isinstance(result_dict, PMap)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_freeze():
    # Test simple primitives
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) is True
    assert freeze(None) is None

    # Test list to pvector
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([{"a": 1}]) == pvector([pmap({"a": 1})])

    # Test dict to pmap
    assert freeze({"a": 1, "b": 2}) == pmap({"a": 1, "b": 2})
    assert freeze({"a": [1, 2], "b": {"c": 3}}) == pmap({"a": pvector([1, 2]), "b": pmap({"c": 3})})
    
    # Test defaultdict to pmap
    import collections
    dd = collections.defaultdict(list, {"a": [1], "b": [2]})
    assert freeze(dd) == pmap({"a": pvector([1]), "b": pvector([2])})

    # Test set to pset
    assert freeze({1, 2, 3}) == pset([1, 2, 3])
    # Note: sets are not recursively frozen per docstring
    assert freeze({(1, [2])}) == pset({(1, [2])}) 

    # Test tuple recursion
    assert freeze((1, 2)) == (1, 2)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))
    assert freeze((1, {"a": 2})) == (1, pmap({"a": 2}))

    # Test already frozen structures (strict=True)
    pv = pvector([1, pmap({"x": 10})])
    assert freeze(pv) == pv
    pm = pmap({"y": pvector([5])})
    assert freeze(pm) == pm

    # Test strict=False behavior for PMap/PVector
    # In strict=False, PMap/PVector are treated as primitives (not recursed)
    pv_inner = pvector([1, 2])
    assert freeze(pv_inner, strict=False) == pv_inner
    
    pm_inner = pmap({"a": 1})
    assert freeze(pm_inner, strict=False) == pm_inner

    # Test deep nesting
    complex_data = [
        {"key": [1, 2, {"deep": True}]},
        (10, {10: 10}),
        {1, 2, 3}
    ]
    expected = pvector([
        pmap({"key": pvector([1, 2, pmap({"deep": True}])])}),
        (10, pmap({10: 10})),
        pset([1, 2, 3])
    ])
    assert freeze(complex_data) == expected

def test_mutant_decorator():
    @mutant
    def add_to_list(lst):
        # Even if we try to mutate, the input is frozen
        # In a real scenario, mutation would fail or be ignored
        return lst

    input_list = [1, 2, 3]
    result = add_to_list(input_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

    @mutant
    def update_dict(d, key, value):
        # This function receives frozen inputs
        # We simulate a return that is also frozen
        return {key: value}

    result_dict = update_dict({"old": 1}, "new", 2)
    assert isinstance(result_dict, PMap)
    assert result_dict["new"] == 2
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_thaw():
    # Test basic types
    assert thaw(1) == 1
    assert thaw("string") == "string"
    assert thaw(True) == True
    assert thaw(None) is None

    # Test pset to set
    assert thaw(pset([1, 2, 3])) == {1, 2, 3}
    assert thaw({1, 2, 3}) == {1, 2, 3}

    # Test pvector to list (simple)
    assert thaw(pvector([1, 2])) == [1, 2]
    assert thaw([1, 2]) == [1, 2]

    # Test pmap to dict (simple)
    assert thaw(pmap({'a': 1, 'b': 2})) == {'a': 1, 'b': 2}
    assert thaw({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}

    # Test tuple (recursive)
    assert thaw((1, pvector([2]))) == (1, [2])
    assert thaw((1, pmap({'a': 2}))) == (1, {'a': 2})

    # Test complex nested structures
    nested_structure = pvector([
        pmap({'key': pset([1, 2]), 'list': pvector([3, 4])}),
        {'simple': 5}
    ])
    expected_structure = [
        {'key': {1, 2}, 'list': [3, 4]},
        {'simple': 5}
    ]
    assert thaw(nested_structure) == expected_structure

    # Test strict=False behavior for lists/dicts
    # In strict=False, list/dict should not be recursed upon
    nested_strict_false = [pvector([1])]
    assert thaw(nested_strict_false, strict=False) == [pvector([1])]

    # Test defaultdict conversion (if used in input)
    import collections
    dd = collections.defaultdict(list, {'a': [1, 2]})
    assert thaw(dd) == {'a': [1, 2]}
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test case 1: Basic mutation isolation
    # The function modifies a list internally, but the decorator should return a pvector
    # and the original list passed in should remain unchanged (frozen)
    mutable_list = [1, 2, 3]
    
    @mutant
    def modify_list(l):
        l.append(4)
        return l

    result = modify_list(mutable_list)
    
    assert result == pvector([1, 2, 3, 4])
    assert mutable_list == [1, 2, 3]  # Original remains unchanged
    assert isinstance(result, pvector)

    # Test case 2: Dictionary mutation and keyword arguments
    mutable_dict = {'a': 1}
    
    @mutant
    def modify_dict(d, extra_key='b'):
        d['extra'] = 2
        return pmap({'a': d['a'], 'extra': d['extra'], 'extra_key': extra_key})

    result = modify_dict(mutable_dict, extra_key='c')
    
    assert result == pmap({'a': 1, 'extra': 2, 'extra_key': 'c'})
    assert mutable_dict == {'a': 1}
    assert isinstance(result, pmap)

    # Test case 3: Nested structures
    nested_structure = {'inner': [10, 20]}
    
    @mutant
    def modify_nested(data):
        data['inner'].append(30)
        return data

    result = modify_nested(nested_structure)
    
    assert result == pmap({'inner': pvector([10, 20, 30])})
    assert nested_structure == {'inner': [10, 20]}

    # Test case 4: Verify that kwargs are also frozen
    @mutant
    def check_kwargs_frozen(k_arg):
        return k_arg

    mutable_kwarg = {'key': 'value'}
    result_kwarg = check_kwargs_frozen(k_arg=mutable_kwarg)
    
    assert result_kwarg == pmap({'key': 'value'})
    assert isinstance(result_kwarg, pmap)

    # Test case 5: Identity behavior for primitives
    @mutant
    def identity(x):
        return x

    assert identity(5) == 5
    assert identity("string") == "string"
    assert identity(True) is True
```


# LLM-generated content at query #5
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
    # We use a list as an argument and check if the function sees a PVector
    @mutant
    def check_type(arg):
        return type(arg)

    assert check_type([1, 2]) is PVector
    assert check_type({'a': 1}) is PMap

    # Test 3: Ensure keyword arguments are frozen
    @mutant
    def check_kwargs(kwarg_val):
        return type(kwarg_val)

    assert check_kwargs(kwarg_val=[1, 2]) is PVector

    # Test 4: Ensure mutation within the function does not affect the original input
    # (Though the decorator freezes them, we test that the return is a new frozen structure)
    @mutant
    def mutate_and_return(mutable_list, mutable_dict):
        # Even if we try to mutate the local references (which are actually frozen PVectors/PMaps)
        # the decorator ensures the returned value is a fresh frozen version.
        # Note: PVector/PMap cannot be mutated in-place like lists, 
        # but we check if the returned structure is correctly frozen.
        return [mutable_list, mutable_dict]

    original_list = [1, 2]
    original_dict = {'key': 'value'}
    result = mutate_and_return(original_list, mutable_dict=original_dict)
    
    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['key'] == 'value'

    # Test 5: Nested structures
    @mutant
    def nested_structure(data):
        return data

    complex_data = [1, {'a': [2, 3]}, (4, 5)]
    result = nested_structure(complex_data)
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[1]['a'], PVector)
    assert isinstance(result[2], tuple)
    assert isinstance(result[2][0], int)

    # Test 6: Verify that the decorator preserves function metadata
    @mutant
    def documented_func():
        """Docstring."""
        return True
    
    assert documented_func.__doc__ == "Docstring."

    # Test 7: Test with multiple arguments and mixed types
    @mutant
    def multi_arg(a, b, c=None):
        return [a, b, c]

    result = multi_arg(1, [2], c={'x': 3})
    assert result == pvector([1, pvector([2]), pmap({'x': 3})])
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_freeze():
    # Test primitive types
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True
    assert freeze(None) is None

    # Test list to pvector
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([{"a": 1}]) == pvector([pmap({"a": 1})])

    # Test dict to pmap
    assert freeze({"a": 1, "b": 2}) == pmap({"a": 1, "b": 2})
    assert freeze({"a": [1, 2], "b": {"c": 3}}) == pmap({"a": pvector([1, 2]), "b": pmap({"c": 3})})
    
    # Test defaultdict to pmap
    import collections
    dd = collections.defaultdict(list, {"a": [1], "b": [2]})
    assert freeze(dd) == pmap({"a": pvector([1]), "b": pvector([2])})

    # Test set to pset
    assert freeze({1, 2, 3}) == pset([1, 2, 3])
    # Note: per docstring, sets are not recursively frozen
    assert freeze({(1, [2])}) == pset({(1, [2])}) 

    # Test tuple (recursive)
    assert freeze((1, 2)) == (1, 2)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))
    assert freeze((1, {"a": 2})) == (1, pmap({"a": 2}))

    # Test existing pyrsistent objects (idempotency/strict mode)
    pv = pvector([1, 2])
    pm = pmap({"a": 1})
    assert freeze(pv) == pv
    assert freeze(pm) == pm
    
    # Test strict=False behavior for PMap/PVector
    # When strict=False, PMap/PVector shouldn't be traversed/re-frozen
    # (Based on the logic: if typ is dict or (strict and isinstance(o, PMap)))
    assert freeze(pm, strict=False) == pm
    assert freeze(pv, strict=False) == pv

def test_mutant_decorator():
    @mutant
    def add_to_list(l, x):
        # This is a bit of a trick: since the decorator freezes args,
        # we can't actually mutate the original list, but we test if it returns frozen
        return l + [x]

    original_list = [1, 2]
    result = add_to_list(original_list, 3)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

@pytest.mark.parametrize("input_val, expected", [
    ([1, 2], pvector([1, 2])),
    ({"a": 1}, pmap({"a": 1})),
    ((1, [2]), (1, pvector([2]))),
    ({1, 2}, pset([1, 2])),
])
def test_freeze_parametrized(input_val, expected):
    assert freeze(input_val) == expected
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

    assert isinstance(identity([1, 2]), PVector)
    assert isinstance(identity({'a': 1}), PMap)

    # Test 2: Verifying arguments are frozen
    # We use a mutable list as an argument. 
    # If mutant works, the function receives a PVector.
    # Even if the function modifies its local reference, the original is protected.
    
    class Tracker:
        def __init__(self):
            self.modified = False

    @mutant
    def modify_list(lst):
        # Attempting to mutate the argument if it were a standard list
        # Since it is frozen to PVector, standard mutation like .append() 
        # would fail or not affect the original object.
        # But we check if the input received by the function is a PVector.
        return lst

    input_list = [1, 2, 3]
    result = modify_list(input_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])
    assert isinstance(input_list, list)  # Original remains a list

    # Test 3: Deep freezing of nested structures
    @mutant
    def nested_func(data):
        return data

    nested_input = [1, {"key": [2, 3]}, (4, 5)]
    result = nested_func(nested_input)

    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[1]['key'], PVector)
    assert isinstance(result[2], tuple)
    assert isinstance(result[2][0], int)

    # Test 4: Keyword arguments are frozen
    @mutant
    def kwarg_func(a, b):
        return {'a': a, 'b': b}

    result = kwarg_func(a=[1], b={'x': 2})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)

    # Test 5: Ensuring mutation inside the function doesn't affect the outer scope
    # Because the arguments are frozen, any attempt to use methods like .append() 
    # on a PVector will either fail or return a new object, leaving the original intact.
    
    shared_list = [10, 20]

    @mutant
    def try_mutate(l):
        try:
            l.append(30)
        except AttributeError:
            # PVector does not have .append() (it has .append which returns a new PVector)
            pass
        return l

    try_mutate(shared_list)
    assert shared_list == [10, 20]

    # Test 6: Verification of return value type for complex dicts
    @mutant
    def return_dict():
        return {"list": [1, 2], "set": {3, 4}}

    res = return_dict()
    assert isinstance(res, PMap)
    assert isinstance(res["list"], PVector)
    assert isinstance(res["set"], PSet)
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - decorator freezes return value
    @mutant
    def identity(x):
        return x

    assert isinstance(identity([1, 2]), PVector)
    assert identity([1, 2]) == pvector([1, 2])

    # Test 2: Verifying arguments are frozen upon entry
    class MutableList(list):
        pass

    @mutant
    def check_args(arg):
        # Inside the function, arg should already be frozen (PVector)
        return arg

    input_list = MutableList([1, 2, 3])
    result = check_args(input_list)
    assert isinstance(result, PVector)
    assert isinstance(input_list, PVector) # The decorator freezes args before fn execution

    # Test 3: Verifying keyword arguments are frozen
    @mutant
    def check_kwargs(val):
        return val

    result_kwarg = check_kwargs(val={'a': 1})
    assert isinstance(result_kwarg, PMap)
    assert result_kwarg['a'] == 1

    # Test 4: Verifying nested mutation isolation
    # The decorator ensures that even if the function tries to mutate the input,
    # the input (being frozen) will raise an error or the function receives a copy.
    @mutant
    def attempt_mutation(data):
        try:
            # This will fail if data is a PVector because PVector is immutable
            data[0] = 99
        except (TypeError, Exception):
            pass
        return data

    original_data = [1, 2, 3]
    result = attempt_mutation(original_data)
    assert result == pvector([1, 2, 3])
    assert result[0] != 99

    # Test 5: Complex nested structures
    @mutant
    def complex_fn(structure):
        return structure

    complex_input = {
        'a': [1, 2, {'b': 3}],
        'c': (4, 5)
    }
    result_complex = complex_fn(complex_input)
    
    assert isinstance(result_complex, PMap)
    assert isinstance(result_complex['a'], PVector)
    assert isinstance(result_complex['a'][2], PMap)
    assert isinstance(result_complex['c'], tuple)
    assert result_complex['a'][2]['b'] == 3

    # Test 6: Ensuring the decorator works with multiple args and kwargs
    @mutant
    def multi_arg(a, b, c=None):
        return a

    assert multi_arg(1, [2], c={'x': 10}) == 1
    assert isinstance(multi_arg(1, [2], c={'x': 10}), PMap) # Error in logic check? No, return is 1 (int)
    
    # Re-verifying return value freeze logic:
    @mutant
    def returns_list(x):
        return [x]
    
    res = returns_list(1)
    assert isinstance(res, PVector)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test Case 1: Basic mutation inside function is isolated from input
    def mutating_fn(data):
        # data is frozen, so we can't mutate it directly if it's a pvector,
        # but if we pass a list, the decorator freezes it to a pvector first.
        # We check if the original list passed to the function remains unchanged.
        # However, since the decorator freezes args, we test if the function 
        # can return a mutated version of the data without affecting the caller's scope.
        mutable_list = [1, 2, 3]
        mutable_list.append(4)
        return mutable_list

    original_list = [1, 2, 3]
    result = mutating_fn(original_list)
    
    assert result == pvector([1, 2, 3, 4])
    assert original_list == [1, 2, 3]

    # Test Case 2: Checking dict mutation and freezing of return value
    def dict_mutator(d):
        # d is frozen as a PMap, so we can't do d['a'] = 2
        # But we can return a new structure.
        new_dict = dict(d)
        new_dict['new_key'] = 'new_val'
        return new_dict

    input_dict = {'a': 1}
    result_dict = dict_mutator(input_dict)
    
    assert result_dict == pmap({'a': 1, 'new_key': 'new_val'})
    assert input_dict == {'a': 1}

    # Test Case 3: Deeply nested structures
    @mutant
    def deep_mutation(structure):
        # The decorator freezes structure. 
        # If structure is a list containing a dict, it becomes pvector([pmap(...)])
        # We try to return a modified version.
        return structure

    nested_input = [1, {'a': [2, 3]}]
    result_nested = deep_mutation(nested_input)
    
    assert isinstance(result_nested, PVector)
    assert isinstance(result_nested[1], PMap)
    assert isinstance(result_nested[1]['a'], PVector)
    assert result_nested[1]['a'][0] == 2

    # Test Case 4: Keyword arguments
    @mutant
    def kwarg_test(a, b):
        return {'a': a, 'b': b}

    res_kwarg = kwarg_test(a=[1], b={'x': 10})
    assert res_kwarg == pmap({'a': pvector([1]), 'b': pmap({'x': 10})})

    # Test Case 5: Verifying that the function cannot mutate the input arguments
    # because they are frozen before the function body executes.
    def attempt_mutation(l):
        try:
            l.append(4)
            return True
        except (AttributeError, TypeError):
            return False

    decorated_attempt = mutant(attempt_mutation)
    
    # If we pass a list, it is converted to PVector. PVector has no .append()
    # This confirms the decorator's freezing mechanism is active.
    assert decorated_attempt([1, 2, 3]) is False
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality and return value freezing
    @mutant
    def identity(x):
        return x

    assert isinstance(identity([1, 2]), PVector)
    assert isinstance(identity({'a': 1}), PMap)

    # Test 2: Argument freezing (ensuring input arguments are frozen)
    @mutant
    def check_args(a, b):
        # We check if the types inside the function are frozen
        return type(a), type(b)

    arg_list = [1, 2, 3]
    arg_dict = {'key': 'value'}
    
    a_type, b_type = check_args(arg_list, arg_dict)
    assert a_type is PVector
    assert b_type is PMap

    # Test 3: Keyword argument freezing
    @mutant
    def check_kwargs(**kwargs):
        return type(kwargs['data'])

    assert check_kwargs(data=[1, 2]) is PVector

    # Test 4: Deep freezing (nested structures)
    @mutant
    def deep_structure(data):
        return data

    nested_input = [1, {'inner': [2, 3]}]
    result = deep_structure(nested_input)
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[1]['inner'], PVector)

    # Test 5: Mutation isolation (the function can mutate its local copies)
    @mutant
    def mutate_local(mutable_list):
        # mutable_list is actually a PVector here due to mutant decorator
        # but if we passed a standard list, it becomes a PVector.
        # We can't mutate PVector in-place like a list, but we can 
        # demonstrate that the decorator handles the conversion.
        local_list = list(mutable_list)
        local_list.append(4)
        return local_list

    original_list = [1, 2, 3]
    result = mutate_local(original_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    # The original input (if it were a mutable list passed to a non-mutant) 
    # is protected because the function receives a frozen version.

    # Test 6: Tuple recursion
    @mutant
    def tuple_test(t):
        return t

    result_tuple = tuple_test((1, [2]))
    assert isinstance(result_tuple, tuple)
    assert isinstance(result_tuple[1], PVector)

    # Test 7: Set handling (not recursive per docstring)
    @mutant
    def set_test(s):
        return s

    result_set = set_test({1, 2, 3})
    assert isinstance(result_set, PSet)
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic functionality: decorator freezes inputs and return value
    def simple_fn(x, y):
        return x + y
    
    mutant_simple = mutant(simple_fn)
    
    # Test with standard types
    assert mutant_simple(1, 2) == 3
    
    # Test with mutable inputs: the decorator should freeze them
    def check_frozen(data_list, data_dict):
        # The decorator freezes inputs, so data_list should be a PVector
        # and data_dict should be a PMap
        return isinstance(data_list, PVector) and isinstance(data_dict, PMap)

    assert mutant(check_frozen)([1, 2], {'a': 1}) is True

    # Test mutation isolation: verify that even if the function tries to 
    # mutate its local copy, the returned value is a frozen version
    def mutation_fn(l):
        # l is frozen, so we cannot mutate it directly like l.append(2)
        # but if we return a new list, the decorator freezes the result
        new_l = list(l)
        new_l.append(4)
        return new_l

    result = mutant(mutation_fn)([1, 2, 3])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])

    # Test complex nested structures
    def complex_fn(structure):
        return structure

    nested_input = [1, {'a': [2, 3]}, (4, 5)]
    result = mutant(complex_fn)(nested_input)
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[1]['a'], PVector)
    assert isinstance(result[2], tuple)
    assert result[2][0] == 4

    # Test keyword arguments
    def kwarg_fn(a, b=None):
        return {'a': a, 'b': b}

    res_kwarg = mutant(kwarg_fn)(a=1, b=[2])
    assert isinstance(res_kwarg, PMap)
    assert res_kwarg['a'] == 1
    assert res_kwarg['b'] == pvector([2])

    # Test strict=False behavior via the underlying freeze/thaw logic 
    # (Implicitly tested by mutant's use of freeze)
    def identity_fn(x):
        return x

    # Ensure that even if we pass something already frozen, it stays frozen
    already_frozen = pvector([1, pmap({'x': 2})])
    assert mutant(identity_fn)(already_frozen) == already_frozen
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - mutation of internal state is allowed, 
    # but arguments are frozen and return value is frozen.
    
    def identity_and_mutate(x, y):
        # x and y are frozen, so we can't mutate them directly if they are containers.
        # But we can mutate a local list.
        local_list = [1, 2, 3]
        local_list.append(4)
        # The function returns a list, which mutant will freeze into a pvector.
        return [x, y, local_list]

    result = identity_and_mutate(1, 2)
    
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == pvector([1, 2, 3, 4])

    # Test 2: Ensuring arguments are frozen
    # If we pass a mutable list, the decorator should freeze it before the function runs.
    
    def check_arg_type(mutable_list):
        # Inside the function, mutable_list should already be a PVector
        return type(mutable_list)

    assert check_arg_type([1, 2]) is PVector

    # Test 3: Testing dictionary/kwargs freezing
    
    def check_kwargs(data):
        # data should be a PMap
        return type(data)

    assert check_kwargs(data={'a': 1}) is PMap

    # Test 4: Deep nesting
    
    @mutant
    def deep_nest(structure):
        return structure

    input_data = {"a": [1, {"b": 2}], "c": (3, 4)}
    result = deep_nest(input_data)

    expected = pmap({
        "a": pvector([1, pmap({"b": 2})]),
        "c": (3, 4)
    })
    assert result == expected
    assert isinstance(result["a"][1], PMap)
    assert isinstance(result["a"], PVector)

    # Test 5: Verifying that the decorator preserves metadata (wraps)
    
    @mutant
    def documented_func():
        """This is a docstring."""
        return True

    assert documented_func.__doc__ == "This is a docstring."

    # Test 6: Verifying that mutation of input arguments is impossible 
    # because they are frozen before the function body executes.
    
    def attempt_mutation(mutable_list):
        try:
            mutable_list.append(99)
        except (AttributeError, TypeError):
            # PVector does not have .append()
            pass
        return mutable_list

    decorated_attempt = mutant(attempt_mutation)
    
    initial_list = [1, 2]
    result = decorated_attempt(initial_list)
    
    # The original list should remain [1, 2] because it was frozen 
    # before the function body could call .append()
    assert initial_list == [1, 2]
    assert result == pvector([1, 2])
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - mutation inside function is isolated
    # The function modifies its input list, but the decorator should return a frozen version
    # and the original external list should remain unchanged if we were to check it,
    # but specifically, the return value is a PVector.
    
    def mutating_fn(lst):
        lst.append(4)
        return lst

    input_list = [1, 2, 3]
    result = mutant(mutating_fn)(input_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])
    # Note: The decorator freezes args before passing them to fn.
    # So 'lst' inside fn is actually a PVector. PVector.append is not an in-place mutation
    # but rather returns a new PVector. However, the decorator's purpose is to 
    # ensure the return value is frozen and args are frozen.

    # Test 2: Nested structures
    def nested_mutation_fn(data):
        # data is a pmap because of the decorator
        # We can't mutate pmap in place, but we can return a new structure
        return data.set('new_key', [1, 2])

    input_dict = {'a': 1}
    result = mutant(nested_mutation_fn)(input_dict)
    
    assert isinstance(result, PMap)
    assert result['new_key'] == pvector([1, 2])

    # Test 3: Kwargs freezing
    def kwargs_fn(a, b=None):
        return {'a': a, 'b': b}

    result = mutant(kwargs_fn)(a=[1], b={'x': 10})
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1])
    assert result['b'] == pmap({'x': 10})

    # Test 4: Verify that the function receives frozen arguments
    def type_check_fn(arg):
        return type(arg)

    assert type_check_fn.__wrapped__ is not None # Check decorator doesn't destroy metadata
    assert mutant(type_check_fn)([1, 2]) == PVector
    assert mutant(type_check_fn)({'a': 1}) == PMap

    # Test 5: Complex nested mutation
    def complex_fn(data):
        # Even if we try to treat it like a list, it's a PVector
        # We return a modified structure
        return data[0]

    input_data = [[1, 2], {'key': 'val'}]
    result = mutant(complex_fn)(input_data)
    assert result == pvector([1, 2])

    # Test 6: Verify strictness/identity of non-containers
    def identity_fn(x):
        return x

    assert mutant(identity_fn)(123) == 123
    assert mutant(identity_fn)("string") == "string"
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_freeze():
    # Test primitives
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) is True
    assert freeze(None) is None

    # Test list to pvector
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([1, {'a': 2}]) == pvector([1, pmap({'a': 2})])

    # Test dict to pmap
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})
    assert freeze({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})
    
    # Test defaultdict to pmap
    import collections
    dd = collections.defaultdict(list, {'a': [1], 'b': [2]})
    assert freeze(dd) == pmap({'a': pvector([1]), 'b': pvector([2])})

    # Test set to pset
    assert freeze({1, 2, 3}) == pset([1, 2, 3])
    # Sets are not recursive per docstring
    assert freeze({(1, [2])}) == pset({(1, [2])}) 
    # Note: The docstring says "sets and dict keys are not recursively frozen"
    # but the code calls freeze(v) for dict values. 
    # However, for sets, it returns pset(o) without mapping freeze over elements.

    # Test tuple to tuple (recursive)
    assert freeze((1, 2)) == (1, 2)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))
    assert freeze((1, {'a': [2]})) == (1, pmap({'a': pvector([2])}))

    # Test idempotency (freezing already frozen objects)
    pv = pvector([1, pmap({'a': 2})])
    assert freeze(pv) == pv
    pm = pmap({'a': pvector([1])})
    assert freeze(pm) == pm

    # Test strict=False behavior
    # When strict=False, dicts/lists/tuples are not processed recursively via type check
    # but the logic relies on 'typ is list' etc.
    # If we pass a PMap but strict=False, it skips the first 'if' block for PMap
    # and falls through to return the object as is.
    pm_already = pmap({'a': [1]})
    assert freeze(pm_already, strict=False) == pm_already
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_mutant():
    # Test case 1: Basic functionality - ensuring return value is frozen
    @mutant
    def simple_func(x):
        return [x, x]
    
    result = simple_func(1)
    assert isinstance(result, PVector)
    assert result == pvector([1, 1])

    # Test case 2: Ensure input arguments are frozen
    # We pass a mutable list; the decorator should freeze it before the function runs
    @mutant
    def check_frozen(data):
        # If frozen, data should be a PVector, not a list
        return isinstance(data, PVector)
    
    assert check_frozen([1, 2, 3]) is True

    # Test case 3: Nested structures
    @mutant
    def nested_func(data):
        return data
    
    input_data = {"a": [1, {"b": 2}], "c": (3, 4)}
    result = nested_func(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][1], PMap)
    assert isinstance(result['c'], tuple)
    assert result['a'][1]['b'] == 2

    # Test case 4: Keyword arguments
    @mutant
    def kwarg_func(key_val):
        return key_val
    
    result = kwarg_func(key_val={'inner': [1]})
    assert isinstance(result, PMap)
    assert isinstance(result['inner'], PVector)

    # Test case 5: Verifying mutation isolation (The decorator freezes inputs)
    # Even if the function tries to treat it like a list, it's interacting with a PVector
    @mutant
    def mutation_attempt(data):
        # Since data is frozen, any attempt to use list methods like .append() 
        # on the input will fail if it were a list, but here it's a PVector.
        # We check if the original list passed from the outside was protected.
        return data

    original_list = [1, 2]
    result = mutation_attempt(original_list)
    
    # The function receives a PVector. 
    assert isinstance(result, PVector)
    # The original object remains unchanged (standard Python behavior, 
    # but mutant ensures the function's internal view is immutable).
    assert original_list == [1, 2]

    # Test case 6: Deeply nested mutation check
    @mutant
    def deep_check(data):
        return data

    complex_input = [1, {'a': [2, 3]}, (4, {'b': 5})]
    result = deep_check(complex_input)

    assert isinstance(result[1]['a'], PVector)
    assert isinstance(result[2][1], PMap)
    assert result[1]['a'][1] == 3
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest

def test_freeze():
    # Test primitives
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True
    assert freeze(None) is None

    # Test lists and pvectors
    assert isinstance(freeze([1, 2, 3]), PVector)
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([1, {'a': 2}]) == pvector([1, pmap({'a': 2})])

    # Test dicts and pmaps
    assert isinstance(freeze({'a': 1, 'b': 2}), PMap)
    assert freeze({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})
    
    # Test defaultdict
    dd = collections.defaultdict(list, {'a': [1], 'b': [2]})
    assert isinstance(freeze(dd), PMap)
    assert freeze(dd) == pmap({'a': pvector([1]), 'b': pvector([2])})

    # Test sets and pssets
    assert isinstance(freeze({1, 2, 3}), PSet)
    assert freeze({1, 2}) == pset([1, 2])
    # Note: per docstring, set elements are not recursively frozen
    # but since sets can't contain mutable types, this is a boundary case.

    # Test tuples
    assert isinstance(freeze((1, 2)), tuple)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

    # Test strict=False behavior (keys/elements not frozen)
    # Note: The implementation of freeze(o, strict=False) mainly affects 
    # how PMap/PVector are handled if they are passed in as the root.
    # If strict=False, PMap is not recursively frozen.
    pm = pmap({'a': [1, 2]})
    assert freeze(pm, strict=False) == pm
    
    # Test deep nesting
    complex_obj = [
        {'key': [1, 2, (3, 4)]},
        {1, 2, 3},
        (5, {'6': 7})
    ]
    expected = pvector([
        pmap({'key': pvector([1, 2, (3, 4)])}),
        pset({1, 2, 3}),
        (5, pmap({'6': 7}))
    ])
    assert freeze(complex_obj) == expected

def test_mutant_decorator():
    @mutant
    def add_to_list(l, val):
        # This is a bit of a paradox because the decorator freezes the input,
        # so 'l' is a PVector. PVector.append returns a new object.
        # The decorator ensures the returned value is also frozen.
        return l.append(val)

    input_list = [1, 2]
    result = add_to_list(input_list, 3)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])
    # Verify original input was not mutated (though impossible with PVector)
    assert input_list == [1, 2]

    @mutant
    def update_dict(d, k, v):
        # Since d is frozen, we can't mutate it, we must return a new one.
        # The decorator handles the freezing of the returned pmap.
        new_d = d.set(k, v)
        return new_d

    input_dict = {'a': 1}
    result = update_dict(input_dict, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_thaw():
    # Test basic types
    assert thaw(1) == 1
    assert thaw("string") == "string"
    assert thaw(True) == True
    assert thaw(None) == None

    # Test PSet to set
    ps = pset([1, 2, 3])
    assert thaw(ps) == {1, 2, 3}
    assert isinstance(thaw(ps), set)

    # Test PVector to list
    pv = pvector([1, 2, [3, 4]])
    assert thaw(pv) == [1, 2, [3, 4]]
    assert isinstance(thaw(pv), list)

    # Test PMap to dict
    pm = pmap({'a': 1, 'b': {'c': 2}})
    assert thaw(pm) == {'a': 1, 'b': {'c': 2}}
    assert isinstance(thaw(pm), dict)

    # Test nested structures
    nested_pv = pvector([pmap({'a': pvector([1, 2])}), tuple([3, 4])])
    expected_nested = [{'a': [1, 2]}, (3, 4)]
    assert thaw(nested_pv) == expected_nested

    # Test tuple recursion
    tup = (pvector([1]), pmap({'a': 2}))
    assert thaw(tup) == ([1], {'a': 2})
    assert isinstance(thaw(tup), tuple)

    # Test strict=False behavior (should not recurse into dict/list/tuple)
    # Note: The implementation of thaw uses 'strict' to decide whether to 
    # apply thaw to elements of native containers.
    native_list = [pvector([1])]
    # with strict=True (default), it should recurse into the list elements
    assert thaw(native_list) == [[1]]
    # with strict=False, it should leave the pvector as is inside the list
    assert thaw(native_list, strict=False) == [pvector([1])]

    native_dict = {'a': pvector([1])}
    assert thaw(native_dict) == {'a': [1]}
    assert thaw(native_dict, strict=False) == {'a': pvector([1])}

    native_tuple = (pvector([1]),)
    assert thaw(native_tuple) == ([1],)
    assert thaw(native_tuple, strict=False) == (pvector([1]),)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_thaw():
    # Test basic types
    assert thaw(1) == 1
    assert thaw("hello") == "hello"
    assert thaw(True) == True

    # Test pset/set
    assert thaw(pset([1, 2, 3])) == {1, 2, 3}
    assert thaw({1, 2, 3}) == {1, 2, 3}

    # Test pvector/list
    assert thaw(pvector([1, 2, 3])) == [1, 2, 3]
    assert thaw([1, 2, 3]) == [1, 2, 3]
    assert thaw([1, [2, 3]]) == [1, [2, 3]]
    assert thaw(pvector([pvector([1]), pmap({'a': 2})])) == [[1], {'a': 2}]

    # Test pmap/dict
    assert thaw(pmap({'a': 1, 'b': 2})) == {'a': 1, 'b': 2}
    assert thaw({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}
    assert thaw(pmap({'a': pvector([1, 2]), 'b': {'c': 3}})) == {'a': [1, 2], 'b': {'c': 3}}

    # Test tuple (recursive)
    assert thaw((1, 2, 3)) == (1, 2, 3)
    assert thaw((1, pvector([2, 3]))) == (1, [2, 3])
    assert thaw((pmap({'a': 1}),)) == ({'a': 1},)

    # Test complex nested structure
    complex_structure = pmap({
        'list': pvector([1, pmap({'inner': 2})]),
        'tuple': (pset([3, 4]),),
        'dict': {'deep': [pvector([5])]}
    })
    expected_structure = {
        'list': [1, {'inner': 2}],
        'tuple': ({3, 4},),
        'dict': {'deep': [[5]]}
    }
    assert thaw(complex_structure) == expected_structure

    # Test strict=False behavior
    # When strict=False, dicts/lists should not be recursively thawed
    nested_dict = {'a': pmap({'b': 1})}
    assert thaw(nested_dict, strict=False) == {'a': pmap({'b': 1})}
    
    nested_list = [pvector([1])]
    assert thaw(nested_list, strict=False) == [pvector([1])]
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_mutant():
    # Test case 1: Basic functionality and return value freezing
    def simple_fn(x, y):
        return {"result": x + y}
    
    decorated_fn = mutant(simple_fn)
    result = decorated_fn(1, 2)
    
    assert isinstance(result, PMap)
    assert result["result"] == 3

    # Test case 2: Verifying arguments are frozen upon entry
    # We use a mutable object inside a list to check if it was frozen
    mutable_list = [1, 2, {"a": 3}]
    
    def check_args_frozen(arg_list):
        # If mutant works, arg_list should be a pvector containing a pmap
        assert isinstance(arg_list, PVector)
        assert isinstance(arg_list[2], PMap)
        return arg_list

    decorated_check = mutant(check_args_frozen)
    result_args = decorated_check(mutable_list)
    assert result_args[2]["a"] == 3

    # Test case 3: Verifying keyword arguments are frozen
    def check_kwargs(data):
        assert isinstance(data, PMap)
        return data

    decorated_kwargs = mutant(check_kwargs)
    result_kwargs = decorated_kwargs(data={"key": [1, 2]})
    assert isinstance(result_kwargs["key"], PVector)

    # Test case 4: Deep mutation isolation
    # The decorator ensures that even if the function tries to mutate the input,
    # the input (being frozen) cannot be mutated in the caller's scope.
    # However, since freeze creates a new object, we test if the returned
    # value is a frozen version of what the function produced.
    
    def mutate_and_return(data):
        # In a real scenario, if 'data' were a standard dict, we could do data['a'] = 2
        # But mutant freezes 'data' before the function runs.
        # We check if the function receives a PMap.
        return data

    decorated_mutate = mutant(mutate_and_return)
    input_dict = {"a": 1}
    result_mutate = decorated_mutate(input_dict)
    
    assert isinstance(result_mutate, PMap)
    assert result_mutate["a"] == 1

    # Test case 5: Complex nested structures
    def complex_fn(structure):
        return structure

    decorated_complex = mutant(complex_fn)
    complex_input = {
        "a": [1, 2, {"b": 3}],
        "c": (4, 5)
    }
    result_complex = decorated_complex(complex_input)

    assert isinstance(result_complex, PMap)
    assert isinstance(result_complex["a"], PVector)
    assert isinstance(result_complex["a"][2], PMap)
    assert isinstance(result_complex["c"], tuple)
    assert result_complex["a"][2]["b"] == 3
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_thaw():
    # Test basic types
    assert thaw(1) == 1
    assert thaw("string") == "string"
    assert thaw(True) == True
    assert thaw(None) == None

    # Test PVector to list (recursive)
    v = pvector([1, pvector([2, 3]), pmap({'a': 4})])
    expected_v = [1, [2, 3], {'a': 4}]
    assert thaw(v) == expected_v

    # Test list to list (recursive, strict=True)
    l = [1, [2, 3], {'a': 4}]
    assert thaw(l) == [1, [2, 3], {'a': 4}]

    # Test PMap to dict (recursive values)
    m = pmap({'a': 1, 'b': pmap({'c': 2}), 'd': pvector([3, 4])})
    expected_m = {'a': 1, 'b': {'c': 2}, 'd': [3, 4]}
    assert thaw(m) == expected_m

    # Test dict to dict (recursive values, strict=True)
    d = {'a': 1, 'b': {'c': 2}, 'd': [3, 4]}
    assert thaw(d) == {'a': 1, 'b': {'c': 2}, 'd': [3, 4]}

    # Test PSet to set
    s = pset([1, 2, 3])
    assert thaw(s) == {1, 2, 3}

    # Test tuple (recursive)
    t = (1, pvector([2, 3]), pmap({'a': 4}))
    expected_t = (1, [2, 3], {'a': 4})
    assert thaw(t) == expected_t

    # Test strict=False behavior
    # With strict=False, list/dict types are not traversed
    l_non_strict = [1, [2, 3]]
    assert thaw(l_non_strict, strict=False) == [1, [2, 3]]
    
    d_non_strict = {'a': {'b': 1}}
    assert thaw(d_non_strict, strict=False) == {'a': {'b': 1}}

    # Test complex nested structure
    complex_struct = pmap({
        'list': pvector([pset([1]), pmap({'inner': 2})]),
        'tuple': (pvector([3]),),
        'dict': {'nested': pmap({'deep': 4})}
    })
    expected_complex = {
        'list': [{1}, {'inner': 2}],
        'tuple': ([3],),
        'dict': {'nested': {'deep': 4}}
    }
    assert thaw(complex_struct) == expected_complex
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest

def test_mutant():
    # Test case 1: Basic function with no mutation, verifying freezing of return value
    @mutant
    def identity(x):
        return x

    assert identity([1, 2, 3]) == pvector([1, 2, 3])
    assert identity({'a': 1}) == pmap({'a': 1})

    # Test case 2: Function that attempts to mutate its input
    # Because mutant freezes args, the input becomes a PVector/PMap which 
    # does not support in-place mutation like list.append() or dict.update().
    # However, we can test if the return value is correctly frozen even if 
    # the function logic tries to return a mutable object.
    @mutant
    def returns_mutable(x):
        return [x, {'key': 'val'}]

    result = returns_mutable(10)
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['key'] == 'val'

    # Test case 3: Verifying kwargs are frozen
    @mutant
    def check_kwargs(a=None):
        return a

    assert check_kwargs(a={'nested': [1]}) == pmap({'nested': pvector([1])})

    # Test case 4: Verifying deep nesting
    @mutant
    def deep_nest(data):
        return data

    complex_input = [1, {'a': [2, {'b': 3}]}, (4, 5)]
    expected_output = pvector([1, pmap({'a': pvector([2, pmap({'b': 3}])])}), (4, 5)])
    assert deep_nest(complex_input) == expected_output

    # Test case 5: Verifying that the decorator preserves function metadata
    @mutant
    def decorated_func():
        """Docstring."""
        return True
    
    assert decorated_func.__doc__ == "Docstring."

    # Test case 6: Testing with tuple elements (should be recursively frozen)
    @mutant
    def tuple_test(t):
        return t

    assert tuple_test(([1],)) == (pvector([1]),)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality and freezing of return value
    @mutant
    def identity(x):
        return x

    # Input is a mutable list, but the decorated function returns a frozen pvector
    input_list = [1, 2, 3]
    result = identity(input_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])
    # Ensure the original input remains untouched by the decorator's logic
    assert isinstance(input_list, list)

    # Test 2: Mutation isolation (internal mutation doesn't affect external state)
    # Note: The decorator freezes args, so the function receives a PVector.
    # We test if the function can still return a mutated version of its logic
    # while the decorator ensures the output is frozen.
    @mutant
    def adder(v, amount):
        # v is frozen (PVector), so we can't mutate it in place.
        # But we check if the result is frozen.
        return v[0] + amount

    assert adder(pvector([10]), 5) == 15

    # Test 3: Complex nested structures
    @mutant
    def complex_func(data):
        return data

    nested_data = {
        'a': [1, 2, {'b': 3}],
        'c': (4, 5)
    }
    
    result = complex_func(nested_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][2], PMap)
    assert result['a'][2]['b'] == 3
    assert result['c'] == (4, 5)

    # Test 4: Keyword arguments freezing
    @mutant
    def kwarg_test(key_val):
        return key_val

    kwarg_input = {'inner': [1, 2]}
    result_kwarg = kwarg_test(key_val=kwarg_input)
    
    assert isinstance(result_kwarg, PMap)
    assert isinstance(result_kwarg['inner'], PVector)

    # Test 5: Verifying that the decorator handles multiple args and kwargs
    @mutant
    def multi_arg(a, b, c=None):
        return a + b + (c if c else 0)

    assert multi_arg(1, 2, c=3) == 6
    assert multi_arg(pvector([1]), pvector([2]), c=pvector([3])) == 6 # This would fail if not frozen/thawed correctly
    # Actually, the decorator freezes args. 1 + 2 + 3 works because freeze(1) is 1.
    # But if we pass pvector, the result is frozen.
    
    # Test 6: Testing that the function cannot mutate the input because it's frozen
    @mutant
    def attempt_mutation(v):
        try:
            v[0] = 99
            return v
        except TypeError:
            return v

    input_v = pvector([1, 2, 3])
    result_v = attempt_mutation(input_v)
    assert result_v[0] == 1
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - mutation is prevented by freezing inputs
    # and the return value is frozen.
    def increment_list_element(lst, index):
        # This function attempts to mutate the input list
        lst[index] = lst[index] + 1
        return lst

    # We use a list as input. If mutant works, the input to 'increment_list_element'
    # becomes a PVector, which is immutable.
    # The decorator will cause an error if we try to mutate it, 
    # or we can observe the behavior.
    
    # Note: Since the decorator freezes args, the original list passed to the 
    # function call is not mutated, but the function itself receives a PVector.
    # If the function tries to mutate a PVector, it will raise a TypeError.
    
    with pytest.raises(TypeError):
        @mutant
        def mutate_me(v):
            v[0] = 99
            return v
        
        mutate_me([1, 2, 3])

    # Test 2: Verify return value is frozen
    @mutant
    def return_mutable_list():
        return [1, 2, {'a': 3}]

    result = return_mutable_list()
    assert isinstance(result, PVector)
    assert isinstance(result[2], PMap)
    assert result[2]['a'] == 3

    # Test 3: Verify kwargs are frozen
    @mutant
    def check_kwargs(data):
        return data

    result_kwarg = check_kwargs(data={'key': [1, 2]})
    assert isinstance(result_kwarg, PMap)
    assert isinstance(result_kwarg['key'], PVector)

    # Test 4: Verify nested structures are frozen
    @mutant
    def nested_structure():
        return [[{1: [2]}]]

    result_nested = nested_structure()
    assert isinstance(result_nested, PVector)
    assert isinstance(result_nested[0], PVector)
    assert isinstance(result_nested[0][0], PMap)
    assert isinstance(result_nested[0][0][1], PVector)

    # Test 5: Verify non-container types remain unchanged
    @mutant
    def simple_types(a, b):
        return a + b

    assert simple_types(1, 2) == 3
    assert simple_types("hello", " world") == "hello world"

    # Test 6: Verify identity of frozen elements
    @mutant
    def identity_check(x):
        return x

    assert identity_check(10) == 10
    assert identity_check(None) is None
```


# LLM-generated content at query #9
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

    # Test 2: Input arguments are frozen
    @mutant
    def check_types(arg_list, arg_dict):
        # Inside the function, args should be pyrsistent types
        assert isinstance(arg_list, PVector)
        assert isinstance(arg_dict, PMap)
        return arg_list

    check_types([1, 2], {'key': 'val'})

    # Test 3: Mutation isolation (Internal mutation does not affect original input)
    # Note: The decorator freezes inputs, so the function receives immutable objects.
    # To test "isolation", we check if the function can attempt to mutate but 
    # the decorator's freezing mechanism prevents the outside world from seeing 
    # side effects on the original mutable objects passed in.
    
    mutable_list = [1, 2, 3]
    
    @mutant
    def mutate_internal(lst):
        # We can't mutate the PVector directly because it's immutable.
        # But we test that the returned value is a new frozen structure.
        return lst.append(4) or lst

    result = mutate_internal(mutable_list)
    
    # The original list should remain unchanged because the decorator 
    # froze it into a PVector before the function body executed.
    assert mutable_list == [1, 2, 3]
    assert result == pvector([1, 2, 3])

    # Test 4: Keyword arguments are frozen
    @mutant
    def check_kwargs(**kwargs):
        assert isinstance(kwargs, PMap)
        return kwargs

    check_kwargs(a=[1], b={'inner': 2})

    # Test 5: Deep freezing of nested structures
    @mutant
    def deep_structure(data):
        return data

    nested_input = [1, {'a': [2, 3]}, (4, 5)]
    result = deep_structure(nested_input)
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[1]['a'], PVector)
    assert isinstance(result[2], tuple)
    assert isinstance(result[2][0], int)

    # Test 6: Functionality with no arguments
    @mutant
    def no_args():
        return 10

    assert no_args() == 10
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_thaw():
    # Test simple primitives
    assert thaw(1) == 1
    assert thaw("string") == "string"
    assert thaw(True) == True
    assert thaw(None) is None

    # Test PSet to set
    ps = pset([1, 2, 3])
    assert thaw(ps) == {1, 2, 3}
    assert isinstance(thaw(ps), set)

    # Test PVector to list (recursive)
    pv = pvector([1, pvector([2, 3]), pmap({'a': 4})])
    expected_pv = [1, [2, 3], {'a': 4}]
    assert thaw(pv) == expected_pv
    assert isinstance(thaw(pv), list)

    # Test PMap to dict (recursive)
    pm = pmap({'a': 1, 'b': pmap({'c': 2}), 'd': pvector([3, 4])})
    expected_pm = {'a': 1, 'b': {'c': 2}, 'd': [3, 4]}
    assert thaw(pm) == expected_pm
    assert isinstance(thaw(pm), dict)

    # Test tuple (recursive)
    tup = (1, pvector([2]), pmap({'a': 3}))
    expected_tup = (1, [2], {'a': 3})
    assert thaw(tup) == expected_tup
    assert isinstance(thaw(tup), tuple)

    # Test standard python containers (strict=True)
    std_list = [1, {'a': 2}]
    assert thaw(std_list) == [1, {'a': 2}]
    
    std_dict = {'x': [1, 2]}
    assert thaw(std_dict) == {'x': [1, 2]}

    # Test strict=False (should not recurse into dict/list contents)
    # If strict is False, it shouldn't call thaw on elements of a standard list
    # However, the implementation of thaw(list) with strict=False 
    # depends on the type check 'typ is list'. 
    # If typ is list and strict is False, it skips the list logic.
    
    # Test deep nesting mix
    complex_structure = pmap({
        'list': pvector([pset([1]), pmap({'inner': pvector([5])})]),
        'tuple': (pmap({'key': 1}),)
    })
    expected_complex = {
        'list': [{1}, {'inner': [5]}],
        'tuple': ({'key': 1},)
    }
    assert thaw(complex_structure) == expected_complex
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
from pyrsistent import pmap, pvector, pset

def test_thaw():
    # Test primitive types
    assert thaw(1) == 1
    assert thaw("string") == "string"
    assert thaw(True) == True
    assert thaw(None) is None

    # Test PSet to set
    ps = pset([1, 2, 3])
    assert thaw(ps) == {1, 2, 3}
    assert isinstance(thaw(ps), set)

    # Test PVector to list (simple)
    pv = pvector([1, 2, 3])
    assert thaw(pv) == [1, 2, 3]
    assert isinstance(thaw(pv), list)

    # Test PVector to list (nested)
    pv_nested = pvector([1, pvector([2, 3]), pmap({'a': 4})])
    assert thaw(pv_nested) == [1, [2, 3], {'a': 4}]

    # Test PMap to dict (simple)
    pm = pmap({'a': 1, 'b': 2})
    assert thaw(pm) == {'a': 1, 'b': 2}
    assert isinstance(thaw(pm), dict)

    # Test PMap to dict (nested)
    pm_nested = pmap({'a': pmap({'inner': 1}), 'b': pvector([1, 2])})
    assert thaw(pm_nested) == {'a': {'inner': 1}, 'b': [1, 2]}

    # Test tuple recursion
    t = (1, pvector([2, 3]), pmap({'a': 4}))
    assert thaw(t) == (1, [2, 3], {'a': 4})

    # Test standard Python containers (strict=True behavior)
    d = {'a': [1, 2], 'b': {'c': 3}}
    assert thaw(d) == {'a': [1, 2], 'b': {'c': 3}}
    
    l = [1, [2, 3], {'a': 4}]
    assert thaw(l) == [1, [2, 3], {'a': 4}]

    # Test strict=False (should not recurse into dicts/lists)
    # When strict=False, standard dicts/lists are returned as-is even if they contain pyrsistent objects
    pm_with_pv = pmap({'a': pvector([1, 2])})
    # thaw(pm_with_pv, strict=False) -> returns dict, but value remains PVector
    result = thaw(pm_with_pv, strict=False)
    assert isinstance(result, dict)
    assert isinstance(result['a'], PVector)

    l_with_pm = [pmap({'a': 1})]
    result_l = thaw(l_with_pm, strict=False)
    assert isinstance(result_l, list)
    assert isinstance(result_l[0], PMap)
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - decorator freezes return value
    @mutant
    def identity(x):
        return x
    
    assert isinstance(identity([1, 2]), PVector)
    assert identity([1, 2]) == pvector([1, 2])

    # Test 2: Deep freezing of return values
    @mutant
    def nested_list():
        return [{"a": [1, 2]}]
    
    result = nested_list()
    assert isinstance(result, PVector)
    assert isinstance(result[0], PMap)
    assert isinstance(result[0]['a'], PVector)
    assert result[0]['a'][0] == 1

    # Test 3: Arguments are frozen before being passed to the function
    # We check if the function receives frozen versions of the inputs
    @mutant
    def check_types(arg_list, arg_dict):
        assert isinstance(arg_list, PVector)
        assert isinstance(arg_dict, PMap)
        return arg_list

    check_types([1, 2], {"key": "value"})

    # Test 4: Keyword arguments are frozen
    @mutant
    def check_kwargs(**kwargs):
        assert isinstance(kwargs['data'], PMap)
        return kwargs['data']

    check_kwargs(data={'inner': [1]})

    # Test 5: Mutation inside the function does not affect the input (due to freezing)
    # Note: Since arguments are frozen, attempts to mutate them via standard 
    # list/dict methods will actually fail or be impossible on the frozen objects.
    @mutant
    def mutation_attempt(mutable_list):
        # This would raise an error if we tried to use .append() on a PVector
        # because PVector is immutable. The decorator ensures the input is PVector.
        try:
            mutable_list.append(3)
        except AttributeError:
            pass 
        return mutable_list

    result = mutation_attempt([1, 2])
    assert result == pvector([1, 2])

    # Test 6: Tuples are recursively frozen
    @mutant
    def tuple_test(t):
        return t

    result = tuple_test((1, [2, 3]))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)

    # Test 7: Integration with complex nested structures
    @mutant
    def complex_structure(data):
        return data

    input_data = {
        "a": [1, {"b": 2}],
        "c": {3, 4}
    }
    expected_output = pmap({
        "a": pvector([1, pmap({"b": 2})]),
        "c": pset({3, 4})
    })
    
    assert complex_structure(input_data) == expected_output
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality and return value freezing
    @mutant
    def identity(x):
        return x

    assert isinstance(identity([1, 2]), PVector)
    assert isinstance(identity({'a': 1}), PMap)

    # Test 2: Verifying arguments are frozen inside the function
    @mutant
    def check_types(arg_list, arg_dict, arg_kwarg):
        # Inside the function, inputs should be immutable pyrsistent types
        is_vector = isinstance(arg_list, PVector)
        is_map = isinstance(arg_dict, PMap)
        is_kwarg_map = isinstance(arg_kwarg, PMap)
        return is_vector and is_map and is_kwarg_map

    assert check_types([1, 2], {'a': 1}, a=1) is True

    # Test 3: Verifying mutation of internal state does not affect the original input
    # (The decorator freezes inputs, so even if we try to mutate the 'mutable' 
    # objects passed in, the function sees the frozen version)
    @mutant
    def mutate_internal(mutable_list):
        # mutable_list is actually a PVector here because of the decorator
        # We can't use .append() on PVector, but we can try to use it on a list 
        # if we were able to bypass the freeze. Since we can't, we check 
        # that the function's logic operates on a frozen copy.
        return mutable_list

    original_list = [1, 2, 3]
    result = mutate_internal(original_list)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])
    assert original_list == [1, 2, 3]

    # Test 4: Nested structures
    @mutant
    def nested_structure(data):
        return data

    nested_input = [1, {'a': [2, 3]}, (4, 5)]
    result = nested_structure(nested_input)
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[1]['a'], PVector)
    assert isinstance(result[2], tuple)
    assert result[2][0] == 4

    # Test 5: Keyword arguments freezing
    @mutant
    def kwarg_test(**kwargs):
        return kwargs

    result_kwargs = kwarg_test(a=[1], b={'c': 2})
    assert isinstance(result_kwargs, PMap)
    assert isinstance(result_kwargs['a'], PVector)
    assert isinstance(result_kwargs['b'], PMap)

    # Test 6: Function with no arguments
    @mutant
    def no_args():
        return 1

    assert no_args() == 1
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest

def test_mutant():
    # Test basic functionality and freezing of return value
    @mutant
    def identity(x):
        return x

    # Test with simple value
    assert identity(10) == 10
    
    # Test with list (should return pvector)
    assert isinstance(identity([1, 2]), PVector)
    assert identity([1, 2]) == pvector([1, 2])

    # Test with nested dict (should return pmap)
    nested_input = {'a': [1, 2], 'b': {'c': 3}}
    result = identity(nested_input)
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    assert result['a'][0] == 1

    # Test with keyword arguments
    @mutant
    def kwarg_test(a, b=None):
        return {'a': a, 'b': b}

    res_kwarg = kwarg_test(1, b=[2, 3])
    assert isinstance(res_kwarg, PMap)
    assert res_kwarg['b'] == pvector([2, 3])

    # Test mutation isolation (the decorator freezes inputs before fn runs)
    # We use a list as an argument. The function modifies the list.
    # Since the decorator calls freeze(e) on args, the function receives a PVector.
    # PVector is immutable, so the function cannot mutate the original list.
    
    mutable_list = [1, 2, 3]
    
    @mutant
    def mutate_list(l):
        # This would normally fail if l was a list and we tried to append,
        # but here we test if the input 'l' is effectively frozen.
        # We can't use l.append because PVector doesn't have it, 
        # but we can check if the original list remains untouched.
        try:
            l.append(4)
        except AttributeError:
            pass # Expected behavior as l is a PVector
        return l

    mutate_list(mutable_list)
    assert mutable_list == [1, 2, 3]

    # Test deep mutation attempt
    @mutant
    def mutate_nested(d):
        # d is a PMap. We can't mutate it directly via d['a'] = 5.
        # But we want to ensure that even if the function tries to 
        # manipulate its local reference, the structural integrity 
        # of the returned value is handled by the decorator.
        return d

    input_dict = {'a': [1]}
    result_dict = mutate_nested(input_dict)
    assert isinstance(result_dict, PMap)
    assert result_dict['a'] == pvector([1])

    # Test tuple recursion
    @mutant
    def tuple_test(t):
        return t

    assert isinstance(tuple_test((1, [2])), tuple)
    assert isinstance(tuple_test((1, [2]))[1], PVector)
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_mutant():
    # Test 1: Basic functionality - mutation of input is prevented via freezing
    # We use a list as an argument. The decorator freezes it into a PVector.
    # A PVector cannot be mutated via .append() or similar in-place methods.
    
    call_count = 0
    
    @mutant
    def mutate_input(data):
        nonlocal call_count
        call_count += 1
        # Since 'data' is frozen into a PVector, we cannot use data.append(4)
        # However, we can try to see if the original object passed in was frozen.
        return data

    original_list = [1, 2, 3]
    result = mutate_input(original_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])
    # Verify the returned value is frozen
    assert not isinstance(result, list)

    # Test 2: Verify that the decorator freezes kwargs as well
    @mutant
    def check_kwargs(**kwargs):
        return kwargs

    kwarg_result = check_kwargs(a=[1], b={'key': 'val'})
    assert isinstance(kwarg_result, PMap)
    assert kwarg_result['a'] == pvector([1])
    assert kwarg_result['b'] == pmap({'key': 'val'})

    # Test 3: Verify the return value is frozen even if the function returns a mutable type
    @mutant
    def return_mutable():
        return {'internal': [1, 2]}

    returned_val = return_mutable()
    assert isinstance(returned_val, PMap)
    assert isinstance(returned_val['internal'], PVector)

    # Test 4: Verify recursion and nested structures
    @mutant
    def complex_structure(data):
        return data

    complex_input = [1, {'a': [2, 3]}, (4, {5})]
    result_complex = complex_structure(complex_input)
    
    assert isinstance(result_complex, PVector)
    assert isinstance(result_complex[1], PMap)
    assert isinstance(result_complex[1]['a'], PVector)
    assert isinstance(result_complex[2], tuple)
    assert isinstance(result_complex[2][1], PMap)

    # Test 5: Verify that the function can still execute logic, just with frozen inputs
    @mutant
    def add_to_list_logic(v, item):
        # This function cannot mutate 'v' because 'v' is a PVector.
        # It must return a new structure.
        new_v = v.append(item)
        return new_v

    v_input = pvector([1, 2])
    result_add = add_to_list_logic(v_input, 3)
    assert result_add == pvector([1, 2, 3])

    # Test 6: Verify strict=True behavior (default) on types that are already P-types
    @mutant
    def identity(o):
        return o

    already_frozen = pvector([pmap({'x': 1})])
    result_already_frozen = identity(already_frozen)
    assert isinstance(result_already_frozen, PVector)
    assert isinstance(result_already_frozen[0], PMap)
```


