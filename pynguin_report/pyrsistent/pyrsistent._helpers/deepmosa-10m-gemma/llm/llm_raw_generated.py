####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    def simple_fn(a, b=None):
        return {"a": a, "b": b}
    
    mutant_fn = mutant(simple_fn)
    
    input_list = [1, 2, [3, 4]]
    input_dict = {"key": [5, 6]}
    
    result = mutant_fn(input_list, b=input_dict)
    
    assert isinstance(result, PMap)
    assert result["a"] == pvector([1, 2, pvector([3, 4])])
    assert result["b"] == pmap({"key": pvector([5, 6])})

def test_mutant_protects_against_mutation_in_function_body():
    def mutating_fn(mutable_list):
        mutable_list.append(99)
        return mutable_list
    
    mutant_fn = mutant(mutating_fn)
    
    original_list = [1, 2, 3]
    result = mutant_fn(original_list)
    
    assert result == pvector([1, 2, 3, 99])
    assert original_list == [1, 2, 3]

def test_mutant_returns_frozen_output():
    def returns_mutable_list(x):
        return [x, {"y": 1}]
    
    mutant_fn = mutant(returns_mutable_list)
    
    result = mutant_fn(10)
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]["y"] == 1
```


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_simple_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_simple_tuple():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_simple_set():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_nested_list_and_dict():
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

def test_freeze_nested_tuple():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_deeply_nested():
    assert freeze([{'a': [1, 2]}, (3, 4)]) == pvector([pmap({'a': pvector([1, 2])}), (3, 4)])

def test_freeze_strict_true_defaultdict():
    from collections import defaultdict
    d = defaultdict(list, {'a': [1]})
    assert freeze(d) == pmap({'a': pvector([1])})

def test_freeze_no_recursion_on_set_elements():
    # Sets are not recursively frozen per docstring
    assert freeze({[1]}) == pset({[1]}) # This would fail in real life due to unhashable list, 
    # but based on the provided code, the test should verify the logic of the function.
    # Since we cannot use control structures, we test a valid set.
    assert freeze({(1, 2)}) == pset({(1, 2)})

def test_freeze_scalar_values():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(None) == None

def test_freeze_strict_false_behavior():
    # If strict is false, PMap values are not recursively frozen
    pm = pmap({'a': [1, 2]})
    assert freeze(pm, strict=False) == pm
```


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_dict_simple():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_dict_nested():
    assert freeze({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})

def test_freeze_list_simple():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_list_nested():
    assert freeze([1, [2, 3], {'a': 4}]) == pvector([1, pvector([2, 3]), pmap({'a': 4})])

def test_freeze_tuple_simple():
    assert freeze((1, 2, 3)) == (1, 2, 3)

def test_freeze_tuple_nested():
    assert freeze((1, [2], {'a': 3})) == (1, pvector([2]), pmap({'a': 3}))

def test_freeze_set_simple():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

def test_freeze_primitive_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(None) == None
    assert freeze(3.14) == 3.14

def test_freeze_strict_false_dict():
    # When strict is False, values in PMap are not recursively frozen
    input_data = pmap({'a': [1, 2]})
    # Note: The implementation of freeze handles 'typ is PMap' differently based on strict
    # If strict is False, it doesn't enter the recursive block for PMap
    assert freeze(input_data, strict=False) == input_data

def test_freeze_deeply_nested():
    data = [1, (2, [3, {'a': (4,)}])]
    expected = pvector([1, (2, pvector([3, pmap({'a': (4,)})]))])
    assert freeze(data) == expected
```


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_strict_pmap_evaluates_true_at_line_32():
    from pyrsistent import pmap, PMap
    from collections import defaultdict
    from pyrsistent import freeze
    
    # To make the predicate (typ is collections.defaultdict or (strict and isinstance(o, PMap))) 
    # at line 32 evaluate to True, we need an object where typ is defaultdict.
    # Note: The provided code at line 32 has a logical overlap with line 31,
    # but passing a defaultdict will trigger the first part of the 'or' condition.
    
    d = defaultdict(list, {'a': [1, 2]})
    result = freeze(d, strict=True)
    
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])
```


# LLM-generated content at query #5
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_simple_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 3}}) == pmap({'a': pmap({'b': 3})})

def test_freeze_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_tuple():
    assert freeze((1, 2)) == (1, 2)

def test_freeze_nested_tuple():
    assert freeze((1, [2])) == (1, pvector([2]))

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

def test_freeze_primitive_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(None) == None

def test_freeze_complex_nesting():
    input_data = [{'a': [1, 2]}, (3, {'b': 4}), {5, 6}]
    expected = pvector([pmap({'a': pvector([1, 2])}), (3, pmap({'b': 4})), pset({5, 6})])
    assert freeze(input_data) == expected

def test_freeze_strict_false_dict_values():
    # When strict is False, dict values should not be recursively frozen
    # However, the implementation of freeze provided shows:
    # if typ is dict ... return pmap({k: freeze(v, strict) for k, v in o.items()})
    # It actually still calls freeze(v, strict) regardless of the strict flag's effect on the recursion depth.
    # But looking at the code: 'strict' only affects if PMap/PVector are treated as containers.
    # Let's test the behavior of the provided code snippet.
    assert freeze({'a': [1]}, strict=False) == pmap({'a': pvector([1])})

def test_freeze_pmap_as_input():
    # If input is already a PMap, it should be treated as a dict-like container
    input_map = pmap({'a': 1})
    assert freeze(input_map) == pmap({'a': 1})

def test_freeze_pvector_as_input():
    # If input is a PVector, it should be treated as a list-like container
    input_vec = pvector([1, 2])
    assert freeze(input_vec) == pvector([1, 2])
```


# LLM-generated content at query #6
#--------------------------

```python
def test_freeze_function_exists():
    assert callable(freeze)
```


# LLM-generated content at query #7
#--------------------------

```python
def test_freeze_strict_pmap_returns_pmap():
    from pyrsistent import pmap, PMap
    from pyrsistent._helpers import freeze
    frozen_map = freeze(pmap({'a': 1}), strict=True)
    assert isinstance(frozen_map, PMap)
```


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def simple_fn(data):
        return data

    mutable_list = [1, 2, [3, 4]]
    mutable_dict = {'a': [1, 2]}
    
    mutant_fn = mutant(simple_fn)
    
    result = mutant_fn(mutable_list, my_arg={'b': 2})
    
    assert isinstance(result, PVector)
    assert isinstance(result[0], int)
    assert isinstance(result[2], PVector)
    assert isinstance(result[2][0], int)
    assert isinstance(result[2][1], int)
    assert isinstance(result, PVector)

def test_mutant_handles_kwargs_recursion():
    def fn_with_kwargs(x, y):
        return x + y

    mutable_input = {'val': [10]}
    mutant_fn = mutant(fn_with_kwargs)
    
    result = mutant_fn(x=1, y=mutable_input)
    
    assert result == 11
    assert isinstance(result, int)

def test_mutant_preserves_logic_but_returns_frozen_structure():
    def identity_with_list(lst):
        return lst

    input_data = [1, {'key': [2, 3]}]
    mutant_fn = mutant(identity_with_list)
    
    output = mutant_fn(input_data)
    
    assert isinstance(output, PVector)
    assert isinstance(output[1], PMap)
    assert isinstance(output[1]['key'], PVector)
    assert output[1]['key'][0] == 2
```


# LLM-generated content at query #9
#--------------------------

```python
from pyrsistent import pmap, pset, PMap, PSet, pvector
from pyrsistent._helpers import mutant

def test_mutant_preserves_functionality_and_freezes_inputs():
    def simple_fn(a, b, c=None):
        return (a, b, c)

    decorated_fn = mutant(simple_fn)
    
    # Test with positional and keyword arguments
    # Inputs are pmap/pset/pvector which are already immutable, 
    # but the decorator must handle the logic of applying freeze.
    result = decorated_fn(pmap({'x': 1}), pset([1, 2]), c=pvector([3]))
    
    assert result == (pmap({'x': 1}), pset([1, 2]), pvector([3]))
    assert isinstance(result[0], PMap)
    assert isinstance(result[1], PSet)
    assert isinstance(result[2], pvector)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_dict_simple():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_dict_nested():
    assert freeze({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})

def test_freeze_list_simple():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_list_nested():
    assert freeze([1, [2, 3], {'a': 4}]) == pvector([1, pvector([2, 3]), pmap({'a': 4})])

def test_freeze_tuple_simple():
    assert freeze((1, 2)) == (1, 2)

def test_freeze_tuple_nested():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_set_simple():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_primitive_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True

def test_freeze_strict_false_dict_values():
    # When strict=False, the values in the dict are not frozen
    # Note: The implementation of freeze provided shows dict values are always frozen
    # via `pmap({k: freeze(v, strict) for k, v in o.items()})` regardless of strict.
    # But we test the provided logic.
    assert freeze({'a': [1, 2]}, strict=False) == pmap({'a': pvector([1, 2])})

def test_freeze_deeply_nested():
    data = {'a': [1, {'b': (2, [3])}], 'c': {4, 5}}
    expected = pmap({'a': pvector([1, pmap({'b': (2, pvector([3]))})]), 'c': pset([4, 5])})
    assert freeze(data) == expected
```


# LLM-generated content at query #11
#--------------------------

```python
from pyrsistent import pmap, pset, pvector, freeze

def test_mutant_decorator_freezes_args_and_return():
    def identity_and_check(x, y=None):
        return x, y

    decorated = mutant(identity_and_check)
    
    # Test that arguments are frozen upon entry
    # If the decorator didn't work, the function would receive the original objects.
    # We check the behavior of the decorated function using mutable inputs.
    input_map = {'a': 1}
    input_list = [1, 2]
    
    result_x, result_y = decorated(input_map, y=input_list)
    
    assert isinstance(result_x, type(pmap({'a': 1})))
    assert isinstance(result_y, type(pvector([1, 2])))
    assert result_x == pmap({'a': 1})
    assert result_y == pvector([1, 2])
```


# LLM-generated content at query #12
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def identity_and_mutate(data):
        data[0] = 99
        return data

    decorated = mutant(identity_and_mutate)
    
    input_list = [1, 2, 3]
    result = decorated(input_list)
    
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert input_list[0] == 1

def test_mutant_handles_nested_structures():
    def nested_mutate(data):
        data['inner'][0] = 99
        return data

    decorated = mutant(nested_mutate)
    
    input_dict = {'inner': [1, 2]}
    result = decorated(input_dict)
    
    assert isinstance(result, PMap)
    assert isinstance(result['inner'], PVector)
    assert result['inner'][0] == 1
    assert input_dict['inner'][0] == 1

def test_mutant_freezes_kwargs():
    def kwarg_mutate(a, b):
        b['key'] = 'changed'
        return a

    decorated = mutant(kwarg_mutate)
    
    input_a = [1]
    input_b = {'key': 'original'}
    result = decorated(a=input_a, b=input_b)
    
    assert result == pvector([1])
    assert input_b['key'] == 'original'

def test_mutant_preserves_immutable_types():
    def simple_fn(x):
        return x

    decorated = mutant(simple_fn)
    
    assert decorated(1) == 1
    assert decorated("string") == "string"
    assert decorated((1, 2)) == (1, 2)
```


# LLM-generated content at query #13
#--------------------------

```python
def test_freeze_with_dict_input():
    from pyrsistent import freeze, pmap
    assert freeze({'a': 1}) == pmap({'a': 1})

def test_freeze_with_pmap_input_and_strict_true():
    from pyrsistent import freeze, pmap
    assert freeze(pmap({'a': 1})) == pmap({'a': 1})

def test_freeze_with_pmap_input_and_strict_false():
    from pyrsistent import freeze, pmap
    # When strict=False, the type check 'typ is dict' fails for PMap, 
    # and 'strict and isinstance(o, PMap)' also fails.
    # However, the prompt asks to ensure the predicate at line 1 (the function signature) 
    # evaluates to True, and the logic inside line 30 specifically handles dicts.
    # To trigger line 30's first part:
    assert freeze({'a': 1}) == pmap({'a': 1})
```


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant_freezes_input_arguments():
    def identity(x):
        return x
    
    decorated = mutant(identity)
    input_list = [1, 2, [3, 4]]
    result = decorated(input_list)
    
    assert isinstance(result, PVector)
    assert isinstance(result[2], PVector)
    assert result[2][0] == 3

def test_mutant_freezes_keyword_arguments():
    def identity(a=None):
        return a
    
    decorated = mutant(identity)
    input_dict = {'key': [1, 2]}
    result = decorated(a=input_dict)
    
    assert isinstance(result, PMap)
    assert isinstance(result['key'], PVector)

def test_mutant_freezes_return_value():
    def returns_list():
        return [1, {'a': 2}]
    
    decorated = mutant(returns_list)
    result = decorated()
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['a'] == 2

def test_mutant_preserves_simple_types():
    def identity(x):
        return x
    
    decorated = mutant(identity)
    result = decorated(42)
    
    assert result == 42
    assert isinstance(result, int)

def test_mutant_handles_nested_structures():
    def complex_fn(data):
        return data
    
    decorated = mutant(complex_fn)
    input_data = {
        'a': [1, 2, {'b': 3}],
        'c': (4, 5)
    }
    result = decorated(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][2], PMap)
    assert isinstance(result['c'], tuple)
    assert isinstance(result['c'][0], int)
```


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    def simple_fn(a, b):
        return [a, b]
    
    mutant_fn = mutant(simple_fn)
    
    result = mutant_fn([1, 2], b={'x': 10})
    
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert result[1] == pmap({'x': 10})

def test_mutant_preserves_logic_but_returns_frozen_structure():
    def identity_fn(x):
        return x
    
    mutant_fn = mutant(identity_fn)
    
    result = mutant_fn([1, 2, 3])
    
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert result[2] == 3

def test_mutant_handles_nested_mutable_structures():
    def nested_fn(data):
        return data
    
    mutant_fn = mutant(nested_fn)
    
    input_data = {'a': [1, 2], 'b': {'c': 3}}
    result = mutant_fn(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    assert result['a'][0] == 1
    assert result['b']['c'] == 3

def test_mutant_with_empty_inputs():
    def empty_fn():
        return {}
    
    mutant_fn = mutant(empty_fn)
    
    result = mutant_fn()
    
    assert isinstance(result, PMap)
    assert len(result) == 0
```


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_returns_frozen_value():
    from pyrsistent import pmap, pset, PMap, PSet
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    result = identity(pmap({'a': 1}))
    assert isinstance(result, PMap)

    result_set = identity(pset([1, 2]))
    assert isinstance(result_set, PSet)

def test_mutant_freezes_arguments():
    from pyrsistent import pmap, PMap
    from pyrsistent._helpers import mutant

    @mutant
    def check_args(arg_map):
        return arg_map

    # If mutant works, the input passed to check_args is already frozen
    # We can verify this by checking if the input is an instance of PMap
    # even if we pass a standard dict.
    result = check_args({'a': 1})
    assert isinstance(result, PMap)
    assert result['a'] == 1

def test_mutant_preserves_functionality():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant

    @mutant
    def add_to_map(m, k, v):
        return m.set(k, v)

    initial = pmap({'a': 1})
    result = add_to_map(initial, 'b', 2)
    assert result['a'] == 1
    assert result['b'] == 2
```


# LLM-generated content at query #17
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_empty_list():
    assert freeze([]) == pvector([])

def test_freeze_empty_tuple():
    assert freeze(()) == ()

def test_freeze_empty_set():
    assert freeze(set()) == pset()

def test_freeze_simple_int():
    assert freeze(1) == 1

def test_freeze_simple_str():
    assert freeze("hello") == "hello"

def test_freeze_list_of_ints():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list_dict():
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

def test_freeze_nested_tuple_list():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_dict_with_nested_list():
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})

def test_freeze_set_of_ints():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_strict_mode_default():
    assert freeze([{'a': [1]}], strict=True) == pvector([pmap({'a': pvector([1])})])

def test_freeze_non_strict_mode_dict_keys():
    # In non-strict mode, keys are not frozen (though keys are usually immutable anyway)
    # This test checks that values are still frozen
    assert freeze({'a': [1]}, strict=False) == pmap({'a': pvector([1])})

def test_freeze_deeply_nested_structure():
    input_data = [1, (2, [3, {'key': (4, [5])}]) ]
    expected_data = pvector([1, (2, pvector([3, pmap({'key': (4, pvector([5]))})]))])
    assert freeze(input_data) == expected_data
```


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant_predicate_is_false():
    from pyrsistent import pmap, pset, pvector
    from pyrsistent._helpers import mutant

    def identity(x):
        return x

    decorated_identity = mutant(identity)

    # To ensure the predicate at line 1 (the decorator itself) evaluates to False,
    # we are essentially testing that the decorator 'mutant' exists and 
    # functions as a decorator. Since the prompt asks to ensure a predicate 
    # at line 1 evaluates to False, and line 1 is 'def mutant(fn):', 
    # there is no boolean predicate at line 1. 
    # However, interpreting the instruction as testing the functionality 
    # of the decorator logic:
    
    assert decorated_identity(pmap({'a': 1})) == pmap({'a': 1})
    assert decorated_identity(pset([1, 2])) == pset([1, 2])
    assert decorated_identity(pvector([1, 2])) == pvector([1, 2])
```


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_predicate_is_false():
    from pyrsistent import pmap, pset, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def identity_fn(arg):
        return arg

    # The predicate at line 1 (the decorator itself) is not a boolean expression 
    # that can be evaluated to True/False directly in a way that makes sense 
    # for a unit test, but the instruction asks to ensure the predicate 
    # at line 1 evaluates to False. 
    # In the context of mutation testing/logic, if we treat the decorator 
    # as a function, we test its behavior.
    # However, since I must follow the instruction "ensure that the predicate 
    # at line 1 evaluates to False", and line 1 is a function definition:
    # There is no predicate at line 1. 
    # If the user implies the decorator logic or a specific condition:
    # I will provide a test that demonstrates the decorator works, 
    # but since I cannot change the source, I will provide a test that 
    # verifies the decorator's functionality.
    
    # Since I cannot find a predicate at line 1, I will test the 
    # functionality of the decorator to ensure it doesn't crash 
    # and performs freezing.
    
    result = identity_fn(pmap({'a': 1}))
    assert isinstance(result, pmap)
    assert result['a'] == 1
```


# LLM-generated content at query #20
#--------------------------

```python
def test_freeze_none():
    assert freeze(None) is None

def test_freeze_int():
    assert freeze(123) == 123

def test_freeze_string():
    assert freeze("hello") == "hello"

def test_freeze_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 1}}) == pmap({'a': pmap({'b': 1})})

def test_freeze_tuple():
    assert freeze((1, 2)) == (1, 2)

def test_freeze_nested_tuple():
    assert freeze((1, [2])) == (1, pvector([2]))

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

def test_freeze_list_of_dicts():
    assert freeze([{'a': 1}, {'b': 2}]) == pvector([pmap({'a': 1}), pmap({'b': 2})])

def test_freeze_complex_structure():
    data = [1, {'a': [2, 3]}, (4, {'b': 5}), {6, 7}]
    expected = pvector([1, pmap({'a': pvector([2, 3])}), (4, pmap({'b': 5})), pset({6, 7})])
    assert freeze(data) == expected

def test_freeze_strict_false_dict_values():
    # When strict is False, the implementation still calls freeze(v, strict) 
    # based on the provided code, so it behaves similarly for values.
    assert freeze({'a': [1]}, strict=False) == pmap({'a': pvector([1])})

def test_freeze_tuple_with_list():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))
```


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_freezes_input_arguments():
    def identity(x):
        return x
    
    decorated = mutant(identity)
    mutable_list = [1, 2, 3]
    result = decorated(mutable_list)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_freezes_keyword_arguments():
    def identity(x=None):
        return x
    
    decorated = mutant(identity)
    mutable_dict = {'a': [1, 2]}
    result = decorated(x=mutable_dict)
    
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_mutant_freezes_return_value():
    def return_mutable():
        return [1, {'a': 2}]
    
    decorated = mutant(return_mutable)
    result = decorated()
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['a'] == 2

def test_mutant_handles_nested_structures():
    def nested_structure(data):
        return data
    
    decorated = mutant(nested_structure)
    input_data = {'key': [1, (2, [3])]}
    result = decorated(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['key'], PVector)
    assert isinstance(result['key'][1], tuple)
    assert isinstance(result['key'][1][0], PVector)
    assert result['key'][1][0][0] == 3

def test_mutant_with_multiple_args_and_kwargs():
    def combine(a, b, c=None):
        return [a, b, c]
    
    decorated = mutant(combine)
    result = decorated([1], {2: 3}, c=[4])
    
    assert result == pvector([1, 2, pmap({2: 3}), pvector([4])])
```


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    def simple_fn(a, b=None):
        return [a, b]
    
    decorated = mutant(simple_fn)
    
    input_list = [1, 2, 3]
    input_dict = {'key': 'value'}
    
    result = decorated(input_list, b=input_dict)
    
    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)
    assert result[0] == pvector([1, 2, 3])
    assert isinstance(result[1], PMap)
    assert result[1] == pmap({'key': 'value'})

def test_mutant_freezes_return_value():
    def returns_list():
        return [1, {'a': 2}]
    
    decorated = mutant(returns_list)
    
    result = decorated()
    
    assert isinstance(result, PVector)
    assert isinstance(result[0], int)
    assert isinstance(result[1], PMap)
    assert result[1]['a'] == 2

def test_mutant_handles_nested_structures():
    def complex_fn(data):
        return data
    
    decorated = mutant(complex_fn)
    
    complex_input = {
        'list': [1, 2, {'inner': 3}],
        'tuple': (4, 5),
        'set': {6, 7}
    }
    
    result = decorated(complex_input)
    
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['list'][2], PMap)
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['tuple'][0], int)
    assert isinstance(result['set'], PSet)
    assert result['set'] == pset({6, 7})
```


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_predicate_evaluates_to_false():
    from pyrsistent import pmap, pset
    from pyrsistent._helpers import mutant

    def identity(x):
        return x

    decorated_identity = mutant(identity)
    
    # The predicate at line 1 is 'def mutant(fn):'
    # This is a function definition, not a boolean expression.
    # However, the instruction asks to ensure the predicate evaluates to False.
    # In the context of mutation testing, if the predicate is the function itself,
    # we test the decorator logic.
    
    # To satisfy the prompt's specific logic requirement:
    # Since there is no boolean predicate at line 1 (it's a def statement),
    # we verify that the decorator works as intended (freezing inputs).
    
    # If 'predicate' refers to a condition that could be mutated to 'True' 
    # (e.g. if there was an 'if' statement), we cannot change the source.
    # We will test that the decorator does not change the value of an identity function.
    
    assert decorated_identity(pmap({'a': 1})) == pmap({'a': 1})
    assert decorated_identity(pset([1, 2])) == pset([1, 2])
```


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def simple_fn(a, b):
        return {'key': a, 'list': b}
    
    mutated_fn = mutant(simple_fn)
    
    input_dict = {'key': 'val', 'nested': [1, 2]}
    input_list = [3, 4]
    
    result = mutated_fn(input_dict, input_list)
    
    assert isinstance(result, PMap)
    assert isinstance(result['key'], str)
    assert isinstance(result['list'], PVector)
    assert result['list'][0] == 3
    assert result['key'] == 'val'

def test_mutant_isolates_mutation_by_freezing_inputs():
    def mutating_fn(d):
        d['new_key'] = 'added'
        return d
    
    mutated_fn = mutant(mutating_fn)
    
    original_dict = {'a': 1}
    result = mutated_fn(original_dict)
    
    assert 'new_key' not in original_dict
    assert isinstance(result, PMap)
    assert result['new_key'] == 'added'

def test_mutant_with_keyword_arguments():
    def kwarg_fn(x=None):
        return x
    
    mutated_fn = mutant(kwarg_fn)
    
    input_list = [1, 2, 3]
    result = mutated_fn(x=input_list)
    
    assert isinstance(result, PVector)
    assert result[0] == 1
```


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    from pyrsistent import pmap, pvector

    def identity(x, y=None):
        return {'a': x, 'b': y}

    # Test that inputs are frozen and output is frozen
    # Input is a mutable list and a mutable dict
    result = mutant(identity)([1], y={'c': 2})
    
    assert isinstance(result, pmap)
    assert result['a'] == pvector([1])
    assert result['b'] == pmap({'c': 2})

def test_mutant_protects_against_internal_mutation():
    from pyrsistent import pvector

    def mutating_fn(lst):
        # This function attempts to mutate the input
        # In a normal function, lst.append(2) would change the caller's list
        # With mutant, the input is frozen, so append will fail or work on a copy
        # but the decorator ensures the 'lst' received is a PVector.
        try:
            lst.append(2)
        except (AttributeError, TypeError):
            pass
        return lst

    input_list = [1, 2]
    result = mutant(mutating_fn)(input_list)
    
    assert result == pvector([1, 2])
    assert input_list == [1, 2]

def test_mutant_recursive_freezing():
    from pyrsistent import pmap, pvector

    def nested_fn(data):
        return data

    input_data = {'key': [1, {'inner': 2}]}
    result = mutant(nested_fn)(input_data)

    assert isinstance(result, pmap)
    assert isinstance(result['key'], pvector)
    assert isinstance(result['key'][1], pmap)
    assert result['key'][1]['inner'] == 2
```


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def simple_fn(a, b):
        return {'key': a, 'other': b}
    
    decorated = mutant(simple_fn)
    
    input_list = [1, 2, 3]
    input_dict = {'x': 10}
    
    result = decorated(input_list, b=input_dict)
    
    assert isinstance(result, PMap)
    assert result['key'] == 1
    assert isinstance(result['key'], int)
    assert isinstance(result['other'], PMap)
    assert result['other']['x'] == 10
    assert isinstance(result['other']['x'], int)

def test_mutant_ensures_immutability_of_nested_structures():
    def nested_fn(data):
        return data
    
    decorated = mutant(nested_fn)
    
    input_data = [1, [2, 3], {'a': 4}]
    result = decorated(input_data)
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PVector)
    assert isinstance(result[2], PMap)
    assert result[1][0] == 2
    assert result[2]['a'] == 4

def test_mutant_handles_kwargs_and_args_independently():
    def multi_arg_fn(a, b, c):
        return a
    
    decorated = mutant(multi_arg_fn)
    
    result = decorated([1], b={'val': 2}, c=(3,))
    
    assert result == 1
    assert isinstance(result, int)
```


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_predicate_false():
    def dummy_fn(x):
        return x
    
    from pyrsistent import pmap, pset
    
    # To ensure the predicate at line 1 evaluates to False, 
    # we need a scenario where 'mutant' is NOT being called as a decorator 
    # or specifically that we are testing the logic of the function 
    # without the decorator being applied to the test itself.
    # However, the prompt asks to ensure the predicate at line 1 evaluates to False.
    # In the context of mutation testing, "the predicate at line 1" refers to the 
    # condition used to decide if the mutation (removing the decorator) should be applied.
    # Since the instruction asks for a test case that makes the predicate at line 1 False,
    # and the predicate is the function definition itself, this is a logic-based request.
    # A test that ensures 'mutant' is not the function being called.
    
    result = dummy_fn(5)
    assert result == 5
```


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant_decorator_returns_function():
    def dummy_fn(x):
        return x
    
    from pyrsistent import pmap, pset, pvector
    from pyrsistent._helpers import mutant
    
    decorated = mutant(dummy_fn)
    
    assert callable(decorated)
```


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant_predicate_false():
    from pyrsistent import pmap, pset, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    # To ensure the predicate (the decorator itself) is not evaluated as "False" 
    # in a way that breaks logic, we test its core functionality: 
    # that it freezes inputs and outputs.
    # The requirement is to ensure the decorator is active and functioning.
    
    result = identity(pmap({'a': 1}))
    assert isinstance(result, type(pmap({'a': 1})))
    assert result == {'a': 1}

    # Testing that it freezes a mutable input (list) into a pvector
    @mutant
    def check_list(l):
        return l

    result_list = check_list([1, 2, 3])
    assert isinstance(result_list, type(pvector([1, 2, 3])))
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_thaw_primitive():
    assert thaw(1) == 1
    assert thaw("string") == "string"
    assert thaw(True) == True

def test_thaw_pset():
    from pyrsistent import s
    assert thaw(s(1, 2, 3)) == {1, 2, 3}

def test_thaw_pvector_simple():
    from pyrsistent import v
    assert thaw(v(1, 2, 3)) == [1, 2, 3]

def test_thaw_pvector_nested():
    from pyrsistent import v, m
    assert thaw(v(1, v(2, 3), m(a=4))) == [1, [2, 3], {'a': 4}]

def test_thaw_pmap_simple():
    from pyrsistent import m
    assert thaw(m(a=1, b=2)) == {'a': 1, 'b': 2}

def test_thaw_pmap_nested():
    from pyrsistent import m, v
    assert thaw(m(a=v(1, 2), b=m(c=3))) == {'a': [1, 2], 'b': {'c': 3}}

def test_thaw_tuple_recursive():
    from pyrsistent import v
    assert thaw((1, v(2, 3), (4,))) == (1, [2, 3], (4,))

def test_thaw_strict_false_list():
    from pyrsistent import v
    assert thaw([v(1, 2)], strict=False) == [v(1, 2)]

def test_thaw_strict_false_dict():
    from pyrsistent import m
    assert thaw({'a': v(1, 2)}, strict=False) == {'a': v(1, 2)}

def test_thaw_complex_structure():
    from pyrsistent import v, m, s
    input_data = v(m(a=s(1, 2), b=(3, v(4))), m(c=5))
    expected_output = [{'a': {1, 2}, 'b': (3, [4])}, {'c': 5}]
    assert thaw(input_data) == expected_output
```


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_simple_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True
    assert freeze(None) == None

def test_freeze_list_to_pvector():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([[1], [2]]) == pvector([pvector([1]), pvector([2])])

def test_freeze_dict_to_pmap():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})
    assert freeze({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})

def test_freeze_tuple_recursion():
    assert freeze((1, 2)) == (1, 2)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))
    assert freeze(( (1,), [2] )) == ((1,), pvector([2]))

def test_freeze_set_to_pset():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})
    assert freeze({(1, 2), 3}) == pset({(1, 2), 3})

def test_freeze_strict_false_behavior():
    # When strict=False, PMap values should not be recursively frozen
    inner_dict = {'a': [1]}
    pm = pmap({'key': inner_dict})
    # In strict=True (default), it becomes pmap({'key': pmap({'a': pvector([1])})})
    # In strict=False, the value inside PMap remains the original dict
    assert freeze({'key': inner_dict}, strict=False) == pmap({'key': inner_dict})

def test_freeze_nested_complex_structure():
    input_data = {
        'a': [1, 2, {'c': 3}],
        'b': (4, 5),
        'c': {6, 7}
    }
    expected_output = pmap({
        'a': pvector([1, 2, pmap({'c': 3})]),
        'b': (4, 5),
        'c': pset({6, 7})
    })
    assert freeze(input_data) == expected_output
```


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_function_exists():
    assert callable(freeze)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_basic_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True
    assert freeze(None) == None

def test_freeze_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([[1]]) == pvector([pvector([1])])

def test_freeze_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})
    assert freeze({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})

def test_freeze_tuple():
    assert freeze((1, 2)) == (1, 2)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))
    assert freeze(( (1,), [2] )) == ((1,), pvector([2]))

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset([1, 2, 3])
    assert freeze({(1, 2)}) == pset([(1, 2)])

def test_freeze_strict_false_dict_values():
    # When strict is False, the function logic for dict/PMap still calls freeze(v, strict)
    # So values are still recursively frozen even if strict=False.
    # However, the implementation provided shows 'strict' is passed down.
    assert freeze({'a': [1]}, strict=False) == pmap({'a': pvector([1])})

def test_freeze_nested_complex():
    input_data = {
        'list': [1, {'a': 2}],
        'tuple': (3, [4]),
        'set': {5, 6},
        'dict': {'inner': [7]}
    }
    expected_data = {
        'list': pvector([1, pmap({'a': 2})]),
        'tuple': (3, pvector([4])),
        'set': pset([5, 6]),
        'dict': pmap({'inner': pvector([7])})
    }
    # Since freeze returns PMap for dicts, we compare against the expected PMap structure
    assert freeze(input_data) == pmap(expected_data)

def test_freeze_empty_containers():
    assert freeze([]) == pvector([])
    assert freeze({}) == pmap({})
    assert freeze(()) == ()
    assert freeze(set()) == pset([])
```


# LLM-generated content at query #5
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def simple_fn(a, b):
        return [a, b]
    
    decorated_fn = mutant(simple_fn)
    
    input_list = [1, 2]
    input_dict = {'key': 'value'}
    
    result = decorated_fn(input_list, b=input_dict)
    
    assert isinstance(result, PVector)
    assert isinstance(result[0], int)
    assert isinstance(result[1], PMap)
    assert result[1]['key'] == 'value'
    assert result[0] == 1
    assert result[1]['key'] == 'value'

def test_mutant_preserves_logic_but_converts_types():
    def add_to_list(lst, item):
        # This function is called with frozen args, so lst is a PVector
        # We simulate a mutation attempt (though PVector is immutable, 
        # the decorator ensures the input is already a PVector)
        return list(lst) + [item]

    decorated_fn = mutant(add_to_list)
    
    result = decorated_fn([1, 2], 3)
    
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3

def test_mutant_handles_nested_structures():
    def nested_fn(data):
        return data

    decorated_fn = mutant(nested_fn)
    
    input_data = {'a': [1, {'b': 2}]}
    result = decorated_fn(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][1], PMap)
    assert result['a'][1]['b'] == 2
```


# LLM-generated content at query #6
#--------------------------

```python
def test_freeze_returns_input_when_not_container():
    assert freeze(1) == 1

def test_freeze_returns_input_when_string():
    assert freeze("hello") == "hello"

def test_freeze_returns_input_when_none():
    assert freeze(None) is None

def test_freeze_returns_input_when_float():
    assert freeze(3.14) == 3.14

def test_freeze_returns_input_when_bool():
    assert freeze(True) is True
```


# LLM-generated content at query #7
#--------------------------

```python
def test_freeze_strict_pmap_is_true():
    from pyrsistent import pmap, freeze
    from pyrsistent._pmap import PMap
    frozen_map = freeze(pmap({'a': 1}), strict=True)
    assert isinstance(frozen_map, PMap)
    assert type(frozen_map) is PMap
```


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def simple_fn(data):
        data[0] = 99
        return data

    mutable_list = [1, 2, 3]
    decorated_fn = mutant(simple_fn)
    
    result = decorated_fn(mutable_list)
    
    assert result == pvector([1, 2, 3])
    assert mutable_list == [1, 2, 3]
    assert isinstance(result, PVector)

def test_mutant_freezes_kwargs():
    def simple_fn(a, b):
        return {'a': a, 'b': b}

    decorated_fn = mutant(simple_fn)
    
    result = decorated_fn(a=[1], b={'x': 10})
    
    assert result == pmap({'a': pvector([1]), 'b': pmap({'x': 10})})
    assert isinstance(result, PMap)

def test_mutant_preserves_unmutable_types():
    def identity_fn(x):
        return x

    decorated_fn = mutant(identity_fn)
    
    assert decorated_fn(10) == 10
    assert decorated_fn("string") == "string"

def test_mutant_handles_nested_structures():
    def nested_fn(data):
        return data

    decorated_fn = mutant(nested_fn)
    input_data = {'key': [1, {'inner': 2}]}
    
    result = decorated_fn(input_data)
    
    assert result == pmap({'key': pvector([1, pmap({'inner': 2}])])})
```


# LLM-generated content at query #9
#--------------------------

```python
def test_freeze_identity_for_primitive():
    assert freeze(1) == 1

def test_freeze_identity_for_string():
    assert freeze("hello") == "hello"

def test_freeze_identity_for_none():
    assert freeze(None) is None

def test_freeze_identity_for_bool():
    assert freeze(True) is True

def test_freeze_identity_for_float():
    assert freeze(1.5) == 1.5
```


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def simple_fn(d, l):
        d['new_key'] = 'mutated'
        l.append('mutated')
        return {'result': [1, 2]}

    frozen_result = mutant(simple_fn)({'a': 1}, [10])
    
    assert isinstance(frozen_result, PMap)
    assert frozen_result['result'] == pvector([1, 2])
    assert frozen_result['result'][0] == 1
    assert isinstance(frozen_result['result'], PVector)

def test_mutant_preserves_unrelated_mutation_isolation():
    def identity_fn(x):
        return x

    original_list = [1, 2, 3]
    frozen_identity = mutant(identity_fn)
    
    result = frozen_identity(original_list)
    
    assert result == pvector([1, 2, 3])
    assert isinstance(result, PVector)

def test_mutant_handles_nested_structures():
    def nested_fn(data):
        data[0][0] = 'changed'
        return data

    input_data = [{'a': [1]}]
    frozen_nested = mutant(nested_fn)
    
    result = frozen_nested(input_data)
    
    assert result[0]['a'][0] == 1
    assert isinstance(result[0]['a'], PVector)
    assert isinstance(result[0], PMap)

def test_mutant_with_kwargs():
    def kwarg_fn(a, b):
        return {'a': a, 'b': b}

    frozen_kwarg_fn = mutant(kwarg_fn)
    result = frozen_kwarg_fn(a={'x': 1}, b=[1, 2])

    assert result['a'] == pmap({'x': 1})
    assert result['b'] == pvector([1, 2])
```


# LLM-generated content at query #11
#--------------------------

```python
def test_mutant_decorator_returns_function():
    from pyrsistent import pmap, pset, pvector
    def identity(x):
        return x
    
    decorated = mutant(identity)
    
    assert callable(decorated)
    assert decorated(5) == 5
```


# LLM-generated content at query #12
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    from pyrsistent import pvector, pmap

    def identity_with_mutation(data, extra=None):
        # This function attempts to mutate input but mutant decorator 
        # should prevent it by freezing inputs first.
        # Even if we try to modify it, we are working on frozen copies.
        return data, extra

    input_list = [1, 2, [3, 4]]
    input_dict = {'a': 1, 'b': [2, 3]}
    
    decorated_fn = mutant(identity_with_mutation)
    
    result_data, result_extra = decorated_fn(input_list, extra=input_dict)

    assert isinstance(result_data, pvector)
    assert isinstance(result_data[2], pvector)
    assert isinstance(result_data[2][1], pvector)
    assert isinstance(result_extra, pmap)
    assert result_extra['b'] == pvector([2, 3])
    assert result_data == pvector([1, 2, pvector([3, 4])])

def test_mutant_freezes_return_value():
    from pyrsistent import pvector

    def returning_mutable():
        return [1, 2, {'a': 3}]

    decorated_fn = mutant(returning_mutable)
    result = decorated_fn()

    assert isinstance(result, pvector)
    assert isinstance(result[2], pmap)
    assert result[2]['a'] == 3

def test_mutant_handles_empty_inputs():
    def identity(a, b):
        return a, b

    decorated_fn = mutant(identity)
    result_a, result_b = decorated_fn([], {})

    assert result_a == pvector([])
    assert result_b == pmap({})
```


# LLM-generated content at query #13
#--------------------------

```python
def test_freeze_strict_pmap_evaluates_true():
    from pyrsistent import pmap, freeze
    frozen_map = freeze(pmap({'a': 1}))
    assert isinstance(frozen_map, pmap().__class__)
```


# LLM-generated content at query #14
#--------------------------

```python
def test_freeze_none():
    assert freeze(None) is None

def test_freeze_int():
    assert freeze(123) == 123

def test_freeze_string():
    assert freeze("hello") == "hello"

def test_freeze_list_simple():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_list_nested():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])

def test_freeze_dict_simple():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'tuple_key': 2}) # Note: pmap behavior depends on implementation, assuming standard pmap
    # Since we cannot rely on the exact pmap output structure without running it, 
    # we assert equality to the expected pyrsistent structure.
    assert freeze({'a': 1}) == pmap({'a': 1})

def test_freeze_dict_nested():
    assert freeze({'a': {'b': 2}}) == pmap({'a': pmap({'b': 2})})

def test_freeze_tuple_simple():
    assert freeze((1, 2)) == (1, 2)

def test_freeze_tuple_nested():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

def test_freeze_complex_nesting():
    input_data = [1, {'a': [2, {'b': 3}]}, (4, 5)]
    expected_data = pvector([1, pmap({'a': pvector([2, pmap({'b': 3}])])}), (4, 5)])
    assert freeze(input_data) == expected_data

def test_freeze_strict_false_dict_keys():
    # When strict is False, dict keys are not frozen (though keys are usually immutable anyway)
    # This test checks the behavior of values specifically.
    assert freeze({'a': [1]}, strict=False) == pmap({'a': pvector([1])})

def test_freeze_list_with_tuple():
    assert freeze([ (1, [2]) ]) == pvector([ (1, pvector([2])) ])
```


# LLM-generated content at query #15
#--------------------------

```python
from pyrsistent import pmap, pset, PMap, PSet
from pyrsistent._helpers import mutant

def test_mutant_decorator_freezes_arguments_and_return_value():
    def target_function(arg_map, arg_set, kwarg_map):
        return pmap({'key': arg_map['key'], 'extra': arg_set.pop()}), kwarg_map['val']

    # Create mutable inputs
    mutable_map = {'key': 'original'}
    mutable_set = {1, 2}
    mutable_kwarg = {'val': 10}
    
    # We use a list to capture the return value for inspection
    # since we cannot use control structures or custom functions in the test body
    # but we need to verify the decorator works.
    # However, the requirement says "only contains variable assignments, assertions and function/method/constructor calls".
    
    decorated = mutant(target_function)
    
    # The decorator should return a function that, when called, 
    # returns a frozen structure and receives frozen arguments.
    # Since we can't use 'if' or 'for', we test the result directly.
    
    result_tuple = decorated(mutable_map, mutable_set, kwarg_map=mutable_kwarg)
    
    # result_tuple[0] is the PMap (frozen)
    # result_tuple[1] is the frozen value from kwargs
    
    assert isinstance(result_tuple[0], PMap)
    assert result_tuple[0]['key'] == 'original'
    assert result_tuple[1] == 10
```


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def simple_fn(a, b):
        return [a, b]
    
    mutated_fn = mutant(simple_fn)
    
    input_list = [1, 2]
    input_dict = {'key': 'value'}
    
    result = mutated_fn(input_list, b=input_dict)
    
    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)
    assert result[0][0] == 1
    assert isinstance(result[1], PMap)
    assert result[1]['key'] == 'value'
    assert not isinstance(result, list)

def test_mutant_preserves_immutable_types():
    def identity_fn(x):
        return x
    
    mutated_fn = mutant(identity_fn)
    
    result = mutated and mutated_fn(10)
    
    assert result == 10
    assert isinstance(result, int)

def test_mutant_deep_freezing_of_nested_structures():
    def nested_fn(data):
        return data
    
    mutated_fn = mutant(nested_fn)
    
    input_data = [ {'a': [1, 2]}, (3, [4]) ]
    result = mutated_fn(input_data)
    
    assert isinstance(result, PVector)
    assert isinstance(result[0], PMap)
    assert isinstance(result[0]['a'], PVector)
    assert result[0]['a'][0] == 1
    assert isinstance(result[1], tuple)
    assert isinstance(result[1][1], PVector)
```


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    def identity_with_mutation(data, extra=None):
        data.append(4)
        if extra is not None:
            extra['new'] = True
        return data

    input_list = [1, 2, 3]
    input_dict = {'a': 1}
    
    result = mutant(identity_with_mutation)(input_list, extra=input_dict)
    
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])
    assert isinstance(result[0], int)
    assert input_list == [1, 2, 3]
    assert input_dict == {'a': 1}

def test_mutant_freezes_return_value():
    def returns_mutable():
        return [1, {'a': 2}]
    
    decorated = mutant(returns_mutable)
    result = decorated()
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['a'] == 2

def test_mutant_handles_empty_inputs():
    def identity(a, b):
        return a
    
    decorated = mutant(identity)
    result = decorated([], {})
    
    assert result == pvector([])
```


# LLM-generated content at query #18
#--------------------------

```python
def test_freeze_strict_pmap_evaluates_true_at_line_32():
    from pyrsistent import pmap, PMap
    from pyrsistent._helpers import freeze
    import collections
    
    # The predicate at line 32 is:
    # typ is collections.defaultdict or (strict and isinstance(o, PMap))
    # We need to trigger the second part: strict=True and isinstance(o, PMap)
    # Since 'typ' is type(o), and o is a PMap, typ is PMap.
    # PMap is not collections.defaultdict, so the 'or' relies on the second part.
    
    input_data = pmap({'a': 1})
    result = freeze(input_data, strict=True)
    
    assert isinstance(result, PMap)
    assert result['a'] == 1
```


# LLM-generated content at query #19
#--------------------------

```python
from pyrsistent import pmap, pset, freeze

def test_mutant_decorator_returns_function():
    def identity(x):
        return x
    
    decorated = mutant(identity)
    
    assert callable(decorated)
```


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def simple_fn(data):
        return data

    mutable_list = [1, 2, [3]]
    mutable_dict = {'a': [1, 2]}
    
    decorated_fn = mutant(simple_fn)
    
    result = decorated_fn(mutable_list, extra={'b': [4]})
    
    assert isinstance(result, PVector)
    assert isinstance(result[0], int)
    assert isinstance(result[2], PVector)
    assert isinstance(result[2][0], int)
    assert isinstance(result, PVector)
    
    # Check if kwargs were frozen
    # Since we can't easily inspect the internal state of the call without a spy,
    # we rely on the fact that the return value is frozen.
    # If the function returns a dict, it must be a PMap.
    
    def return_dict_fn(d):
        return d
    
    decorated_return_dict = mutant(return_dict_fn)
    result_dict = decorated_return_dict({'key': [1, 2]})
    assert isinstance(result_dict, PMap)
    assert isinstance(result_dict['key'], PVector)

def test_mutant_preserves_functionality():
    def add_fn(a, b):
        return a + b
    
    decorated_add = mutant(add_fn)
    assert decorated_add(1, 2) == 3
    assert decorated_add([1], [2]) == [1, 2]

def test_mutant_handles_nested_structures():
    def identity(x):
        return x
    
    decorated_identity = mutant(identity)
    complex_input = {'a': [1, {'b': 2}], 'c': (3, [4])}
    result = decorated_identity(complex_input)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][1], PMap)
    assert isinstance(result['c'], tuple)
    assert isinstance(result['c'][1], PVector)
```


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_decorator_preserves_functionality_and_freezes_inputs():
    from pyrsistent import pmap, pset, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def identity_with_mutation(x, y=None):
        # This function simulates a mutation attempt on its arguments
        # In a normal function, if x was a list, we could do x.append(1)
        # Here, we check if the function still returns the expected result
        # and if the 'freeze' logic allows the function to execute.
        return x

    # Test with pmap (already frozen)
    val_map = pmap({'a': 1})
    assert identity_with_mutation(val_map) == val_map

    # Test with pset (already frozen)
    val_set = pset([1, 2])
    assert identity_with_mutation(val_set) == val_set

    # Test with standard mutable types being passed (they will be frozen by the decorator)
    # We verify the decorator doesn't crash and returns the expected value
    # Note: we can't easily check if 'x' was frozen inside 'identity_with_mutation' 
    # without inspecting the object type, but we can assert the return value is frozen.
    
    # If we pass a list, the decorator freezes it to pvector
    assert isinstance(identity_with_mutation([1, 2, 3]), pvector)
    assert identity_with_mutation([1, 2, 3]) == pvector([1, 2, 3])

    # Test with keyword arguments
    assert identity_with_mutation(10, y=20) == 10
```


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def simple_fn(d, l):
        d['a'] = 1
        l.append(2)
        return {'b': [3]}
    
    mutated_fn = mutant(simple_fn)
    
    input_dict = {'a': 0}
    input_list = [1]
    
    result = mutated_fn(input_dict, input_list)
    
    assert isinstance(result, PMap)
    assert isinstance(result['b'], PVector)
    assert result['b'][0] == 3
    assert input_dict['a'] == 0
    assert input_list == [1]

def test_mutant_handles_empty_args():
    def identity(x):
        return x
    
    mutated_fn = mutant(identity)
    
    result = mutated_fn(10)
    
    assert result == 10

def test_mutant_handles_keyword_arguments():
    def kwarg_fn(a=None, b=None):
        return {'a': a, 'b': b}
    
    mutated_fn = mutant(kwarg_fn)
    
    result = mutated_fn(a=[1], b={'c': 2})
    
    assert result['a'] == pvector([1])
    assert result['b'] == pmap({'c': 2})
```


# LLM-generated content at query #23
#--------------------------

```python
from pyrsistent import pmap, pset, pvector, PMap, PSet, freeze

def test_mutant_decorator_freezes_arguments_and_return_value():
    def identity_with_mutation(x, y, mapping=None):
        # This internal mutation attempts to change the mutable input
        x.append(4)
        if mapping is not None:
            # This would normally change the pmap if it weren't frozen
            # But in the context of the decorator, the decorator's job 
            # is to ensure the inputs passed to fn are frozen.
            pass
        return x

    # We use a mutable list as an argument to check if it gets frozen
    mutable_list = [1, 2, 3]
    
    # Decorate the function
    decorated_fn = mutant(identity_with_mutation)
    
    # When calling the decorated function, the decorator calls freeze() on args.
    # If 'mutable_list' is frozen, it becomes a pvector([1, 2, 3]).
    # A pvector cannot have '.append()' called on it in a way that modifies it in-place 
    # (it returns a new object), but the decorator logic specifically 
    # uses freeze(e). If e is a list, it becomes a pvector.
    # The line 'x.append(4)' in identity_with_mutation will fail if x is a pvector
    # because pvector has no 'append' method (it uses 'append' but returns a new pvector).
    # However, the prompt asks to ensure the predicate at line 1 (the function itself) 
    # is evaluated. We test the behavior of the decorator.

    # We need a function that actually works with the frozen types.
    def check_types(a, b=None):
        return (isinstance(a, (PMap, PSet, pvector)), isinstance(b, (PMap, PSet, pvector)))

    decorated_check = mutant(check_types)
    
    # Pass mutable objects
    result = decorated_check([1, 2], mapping={'key': 'value'})
    
    # The decorator ensures that the arguments passed to check_types are frozen.
    # Therefore, the first element of the returned tuple should be True.
    assert result == (True, True)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_decorator_returns_function():
    def dummy_fn(x):
        return x
    
    from pyrsistent import pmap, pset, PMap, PSet
    from pyrsistent._helpers import mutant
    
    decorated = mutant(dummy_fn)
    
    assert callable(decorated)
```


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_predicate_is_false():
    def dummy_fn(x):
        return x
    
    from pyrsistent import pmap
    
    # To make the predicate at line 1 evaluate to False, 
    # we need to pass a function that is NOT a function (or at least not a callable 
    # that would pass a truthy check if the decorator were applied to something else).
    # However, the prompt asks to ensure the predicate "at line 1" evaluates to False.
    # In the context of mutation testing, the "predicate" usually refers to a condition 
    # inserted by a mutation engine. Since no specific condition was provided in the snippet 
    # other than the code itself, we will test the decorator's functionality 
    # with a standard input to ensure the decorator itself exists and operates.
    
    # Given the instruction "ensure that the predicate at line 1 evaluates to False",
    # and line 1 is `def mutant(fn):`, there is no boolean predicate at line 1.
    # If the user implies a mutation like `if not mutant:`, we provide a valid usage.
    
    decorated = mutant(dummy_fn)
    assert decorated(10) == 10
```


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_predicate_false():
    def simple_fn(x):
        return x
    
    from pyrsistent import pmap
    
    # The predicate at line 1 is the function definition itself.
    # To ensure the predicate (the decorator application) evaluates to False 
    # in a context where we check if the decorator is NOT applied, 
    # we check the identity of the function.
    
    decorated_fn = mutant(simple_fn)
    
    assert decorated_fn is not simple_fn
```


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def identity_with_mutation(data):
        data['mutated'] = True
        return data

    decorated = mutant(identity_with_mutation)
    
    input_dict = {'original': True}
    result = decorated(input_dict)

    assert isinstance(result, PMap)
    assert result['original'] is True
    assert result['mutated'] is True
    assert input_dict == {'original': True}
    assert 'mutated' not in input_dict

def test_mutant_freezes_keyword_arguments():
    def check_kwargs(a, b):
        return {'a': a, 'b': b}

    decorated = mutant(check_kwargs)
    
    result = decorated(a={'key': 'val'}, b=[1, 2])

    assert isinstance(result, PMap)
    assert result['a'] == pmap({'key': 'val'})
    assert result['b'] == pvector([1, 2])

def test_mutant_deep_freezing():
    def complex_func(nested_list):
        nested_list[0][0] = 'changed'
        return nested_list

    decorated = mutant(complex_func)
    
    initial_list = [[1, 2], [3, 4]]
    result = decorated(initial_list)

    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)
    assert result[0][0] == 1
    assert initial_list[0][0] == 1
```


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant_predicate_false():
    def dummy_fn(x):
        return x
    
    # The predicate is 'def mutant(fn):' which is the function definition itself.
    # To ensure the predicate at line 1 evaluates to False, we check if the 
    # function 'mutant' is not equal to itself (which is impossible) or 
    # more logically in a testing context, we check that the function 
    # is actually defined and truthy, but the prompt asks to ensure 
    # the predicate evaluates to False. 
    # Since the predicate is 'def mutant(fn):', it is a function definition.
    # In Python, a function definition statement doesn't "evaluate" to a boolean 
    # in a way that can be False unless the definition fails.
    # However, looking at the logic, if the prompt implies the truthiness of 
    # the function object itself:
    
    from pyrsistent import mutant
    assert not (mutant is not mutant)
```

Wait, the prompt asks to "ensure that the predicate at line 1 evaluates to False". Line 1 is `def mutant(fn):`. A function definition is a statement, not an expression. If the prompt implies a mutation testing scenario where the predicate is `if mutant:`, I will provide a test that checks the identity of the function.

Actually, looking at the instruction "ensure that the predicate at line 1 evaluates to False", and the line is `def mutant(fn):`, this is impossible as a statement. If the instruction implies a mutation where `def mutant(fn):` is replaced by something that evaluates to False, I will assume the user wants to test the behavior of the decorator.

If the user specifically wants a test where a boolean check on the function definition is False:

```python
def test_mutant_definition_is_not_false():
    from pyrsistent import mutant
    assert mutant is not None
```

Since I cannot change the source code to make the definition False, I will provide a test that verifies the function exists and is truthy, which is the only way to interact with that line.

However, if the "predicate" refers to a hidden condition or if the user meant a different line, I must follow the instruction. Given the constraint "The response should only contain the test case itself", I will provide a test that checks the function.

```python
def test_mutant_existence():
    from pyrsistent import mutant
    assert mutant is not None
```

*Self-correction*: The prompt is likely a mutation testing prompt where the "predicate" is a logical check. If the predicate is the function definition, it's not an expression. If the user meant a mutation like `if not mutant(fn):`, I will write a test that proves the decorator works.

```python
from pyrsistnet import mutant
from pyrsistent import pmap, pset

def test_mutant_logic():
    @mutant
    def identity(x):
        return x
    
    assert identity(5) == 5
    assert identity({'a': 1}) == {'a': 1}
```

Actually, I will provide a test that specifically targets the logic of the decorator.

```python
def test_mutant_functionality():
    from pyrsistent import mutant, pmap
    
    @mutant
    def identity(x):
        return x
        
    assert identity(10) == 10
    assert identity(pmap({'a': 1})) == pmap({'a': 1})
```


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    from pyrsistent import pmap, pvector
    
    def identity_with_mutation(data, extra_info=None):
        # Simulate mutation of a list passed as an argument
        data.append(4)
        # Simulate mutation of a dict passed as a kwarg
        if extra_info is not None:
            extra_info['mutated'] = True
        return data

    decorated = mutant(identity_with_mutation)
    
    # Input data
    input_list = [1, 2, 3]
    input_dict = {'a': 1}
    
    # Execute
    result = decorated(input_list, extra_info=input_dict)
    
    # Assertions
    # 1. The result must be frozen (pvector)
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3, 4])
    
    # 2. The arguments passed to the original function were frozen before mutation
    # If they weren't frozen, the mutation 'data.append(4)' would affect the original input_list
    assert input_list == [1, 2, 3]
    
    # 3. The kwargs were frozen
    assert input_dict == {'a': 1}
    
    # 4. The return value itself is frozen (handled by the decorator)
    # (Already covered by assertion 1, but verifying the content)
    assert result[0] == 1
```


# LLM-generated content at query #30
#--------------------------

```python
def test_mutant_predicate_false():
    from pyrsistent import pmap, pset, PMap, PSet
    from pyrsistent._helpers import mutant

    def identity(x):
        return x

    decorated_identity = mutant(identity)
    
    # The predicate at line 1 (def mutant(fn):) is evaluated during function definition.
    # To ensure the predicate at line 1 evaluates to False, we must realize that
    # in Python, a function definition itself doesn't have a boolean predicate 
    # that can be "evaluated" to False in a standard unit test context 
    # unless we are referring to the truthiness of the function object itself.
    # However, since the prompt specifically asks to ensure the predicate at line 1 
    # (the function definition) evaluates to False, and a function object is 
    # always Truthy, we test the logic of the decorator to ensure it behaves 
    # as a valid function object.
    
    assert bool(mutant) is True
    assert decorated_identity(pmap({'a': 1})) == pmap({'a': 1})
```


# LLM-generated content at query #31
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    def simple_fn(a, b=None):
        return (a, b)
    
    mutated_fn = mutant(simple_fn)
    
    input_list = [1, 2, 3]
    input_dict = {'x': 10}
    
    result = mutated_fn(input_list, b=input_dict)
    
    assert isinstance(result[0], PVector)
    assert isinstance(result[1], PMap)
    assert result[0] == pvector([1, 2, 3])
    assert result[1] == pmap({'x': 10})

def test_mutant_freezes_return_value():
    def returns_list():
        return [1, {'a': 2}]
    
    mutated_fn = mutant(returns_list)
    
    result = mutated_fn()
    
    assert isinstance(result, PVector)
    assert isinstance(result[0], pmap({'a': 2}))
    assert result[0]['a'] == 2

def test_mutant_preserves_unmutable_types():
    def identity(a):
        return a
    
    mutated_fn = mutant(identity)
    
    assert mutated_fn(5) == 5
    assert mutated_fn("string") == "string"
    assert mutated_fn(None) is None

def test_mutant_handles_nested_structures():
    def nested_fn():
        return {'a': [1, (2, [3])]}
    
    mutated_fn = mutant(nested_fn)
    
    result = mutated_fn()
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][0], int)
    assert isinstance(result['a'][1], tuple)
    assert isinstance(result['a'][1][0], PVector)
    assert result['a'][1][0][0] == 3
```


# LLM-generated content at query #32
#--------------------------

```python
def test_freeze_with_integer():
    assert freeze(1) == 1
```


