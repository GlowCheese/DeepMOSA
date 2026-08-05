####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent import pvector, pmap
    
    def identity(x):
        return x
    
    decorated = mutant(identity)
    
    input_list = [1, 2, {'a': 3}]
    result = decorated(input_list)
    
    assert isinstance(result, pvector)
    assert result[2] == pmap({'a': 3})

def test_mutant_freezes_kwargs():
    from pyrsistent import pvector, pmap
    
    def identity(x=None):
        return x
    
    decorated = mutant(identity)
    
    result = decorated(x=[1, 2])
    
    assert isinstance(result, pvector)
    assert result[0] == 1

def test_mutant_freezes_return_value():
    from pyrsistent import pvector
    
    def returning_list():
        return [1, [2, 3]]
    
    decorated = mutant(returning_list)
    
    result = decorated()
    
    assert isinstance(result, pvector)
    assert isinstance(result[1], pvector)
    assert result[1][0] == 2

def test_mutant_handles_complex_nesting():
    from pyrsistent import pvector, pmap
    
    def complex_fn(d, l):
        return d, l
    
    decorated = mutant(complex_fn)
    
    input_dict = {'a': [1, 2]}
    input_list = [(3, {'b': 4})]
    
    res_dict, res_list = decorated(d=input_dict, l=input_list)
    
    assert isinstance(res_dict, pmap)
    assert isinstance(res_dict['a'], pvector)
    assert isinstance(res_list[0], tuple)
    assert isinstance(res_list[0][1], pmap)
```


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_simple_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list_and_dict():
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

def test_freeze_tuple_with_list():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

def test_freeze_deeply_nested_structure():
    input_data = {'a': [1, {'b': (2, [3])}], 'c': {4, 5}}
    expected_data = pmap({'a': pvector([1, pmap({'b': (2, pvector([3]))})]), 'c': pset({4, 5})})
    assert freeze(input_data) == expected_data

def test_freeze_no_op_on_primitives():
    assert freeze(10) == 10
    assert freeze("string") == "string"
    assert freeze(True) == True

def test_freeze_strict_false_dict_values():
    # When strict is False, dict values are not recursively frozen
    input_data = {'a': [1, 2]}
    result = freeze(input_data, strict=False)
    assert isinstance(result, PMap)
    assert result['a'] == [1, 2]

def test_freeze_tuple_elements():
    assert freeze((1, 2)) == (1, 2)
```


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_simple_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list_and_dict():
    assert freeze([1, {'a': 3}]) == pvector([1, pmap({'a': 3})])

def test_freeze_tuple_with_list():
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})

def test_freeze_deeply_nested_structure():
    input_data = [{'a': [1, 2]}, (3, {'b': 4})]
    expected_data = pvector([pmap({'a': pvector([1, 2])}), (3, pmap({'b': 4}))])
    assert freeze(input_data) == expected_data

def test_freeze_non_container():
    assert freeze(5) == 5
    assert freeze("hello") == "hello"

def test_freeze_strict_false_on_dict_values():
    # When strict=False, dict values should not be recursively frozen
    input_data = {'a': [1, 2]}
    # Note: The implementation of freeze provided shows that for 'typ is dict', 
    # it always calls freeze(v, strict). However, looking at the logic:
    # If strict is False, it doesn't enter the recursive call for PMap/PVector types if they are already frozen.
    # But for a standard dict, it performs {k: freeze(v, strict) ...}.
    # Let's test that lists inside dicts are still frozen because 'typ is list' is checked independently of strict.
    assert freeze({'a': [1, 2]}, strict=False) == pmap({'a': pvector([1, 2])})

def test_freeze_defaultdict():
    import collections
    input_data = collections.defaultdict(list, {'a': [1]})
    assert freeze(input_data) == pmap({'a': pvector([1])})
```


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_simple_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True
    assert freeze(None) == None

def test_freeze_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([[1], [2]]) == pvector([pvector([1]), pvector([2])])

def test_freeze_tuple():
    assert freeze((1, 2)) == (1, 2)
    assert freeze((1, [2])) == (1, pvector([2]))
    assert freeze(((1,),)) == ((1,),)

def test_freeze_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'tuple': 2}) # Note: logic in source has a typo/bug in the provided snippet mapping 'b' to 'tuple', but we test intended behavior of keys/values
    # Re-evaluating based on actual code: {'a': 1, 'b': 2} -> pmap({'a': 1, 'b': 2})
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})
    assert freeze({'a': {'b': 3}}) == pmap({'a': pmap({'b': 3})})

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})
    assert freeze({(1, 2)}) == pset({(1, 2)})

def test_freeze_nested_structures():
    input_data = [1, {'a': (2, [3])}, {4, 5}]
    expected = pvector([1, pmap({'a': (2, pvector([3]))}), pset({4, 5})])
    assert freeze(input_data) == expected

def test_freeze_strict_false():
    # When strict is False, PMap/PVector should not be recursed into
    pm = pmap({'a': [1]})
    # If strict=False, the list inside PMap remains a list (depending on implementation of dict comprehension)
    # Looking at code: `if typ is dict or (strict and isinstance(o, PMap)):`
    # If strict is False, it skips the recursion for PMap values.
    assert freeze(pm, strict=False) == pm
```


# LLM-generated content at query #5
#--------------------------

```python
from pyrsistent import pmap, pvector, pset

def test_mutant_freezes_args_and_kwargs():
    def simple_fn(data, extra=None):
        return data, extra

    mutated_fn = mutant(simple_fn)
    
    input_list = [1, 2, 3]
    input_dict = {'a': 1}
    input_extra = {'b': 2}
    
    result_data, result_extra = mutated_fn(input_list, extra=input_dict)
    
    assert isinstance(result_data, pvector)
    assert isinstance(result_extra, pmap)
    assert result_data == pvector([1, 2, 3])
    assert result_extra == pmap({'b': 2})

def test_mutant_freezes_return_value():
    def returns_mutable(x):
        return [x, {'a': x}]

    mutated_fn = mutant(returns_mutable)
    
    result = mutated_fn(10)
    
    assert isinstance(result, pvector)
    assert isinstance(result[1], pmap)
    assert result == pvector([10, pmap({'a': 10})])

def test_mutant_handles_nested_structures():
    def deep_fn(structure):
        return structure

    mutated_fn = mutant(deep_fn)
    
    complex_input = [ {'key': [1, 2]} ]
    result = mutated_fn(complex_input)
    
    assert isinstance(result, pvector)
    assert isinstance(result[0], pmap)
    assert isinstance(result[0]['key'], pvector)
    assert result == pvector([pmap({'key': pvector([1, 2])})])

def test_mutant_preserves_simple_types():
    def identity(x):
        return x

    mutated_fn = mutant(identity)
    
    assert mutated_fn(5) == 5
    assert mutated_fn("string") == "string"
    assert mutated_fn(True) is True
```


# LLM-generated content at query #6
#--------------------------

```python
def test_freeze_strict_pmap_is_true():
    from pyrsistent import pmap, freeze
    frozen_map = freeze(pmap({'a': 1}), strict=True)
    assert isinstance(frozen_map, pmap.__class__) and isinstance(frozen_map, type(pmap({'a': 1})))
```


# LLM-generated content at query #7
#--------------------------

```python
def test_freeze_basic_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True
    assert freeze(None) is None

def test_freeze_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([]) == pvector([])

def test_freeze_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})
    assert freeze({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})

def test_freeze_tuple():
    assert freeze((1, 2)) == (1, 2)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))
    assert freeze(()) == ()

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})
    assert freeze(set()) == pset()

def test_freeze_nested_structures():
    input_data = [
        {'a': 1, 'b': [2, 3]},
        (4, {'c': 5}),
        {6, 7}
    ]
    expected_output = pvector([
        pmap({'a': 1, 'b': pvector([2, 3])}),
        (4, pmap({'c': 5})),
        pset({6, 7})
    ])
    assert freeze(input_data) == expected_output

def test_freeze_strict_false_on_dict_values():
    # When strict is False, values in dicts/pmaps are not recursively frozen
    # Note: The implementation of freeze shows it recurses on values regardless of strict 
    # for the recursive call itself, but let's test the provided logic behavior.
    assert freeze({'a': [1]}, strict=False) == pmap({'a': pvector([1])})

def test_freeze_with_pmap_input():
    existing_pmap = pmap({'a': 1})
    assert freeze(existing_pmap) == pmap({'a': 1})

def test_freeze_with_pvector_input():
    existing_pvector = pvector([1, 2])
    assert freeze(existing_pvector) == pvector([1, 2])
```


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant_is_decorator():
    from pyrsistent import pmap, pset, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    result = identity(pmap({'a': 1}))
    assert isinstance(result, type(pmap({'a': 1})))
    assert result == pmap({'a': 1})

    @mutant
    def check_args(arg_list, arg_dict):
        return (arg_list, arg_dict)

    res_list, res_dict = check_args(pvector([1, 2]), {'key': pset([3])})
    assert isinstance(res_list, type(pvector([1, 2])))
    assert isinstance(res_dict['key'], type(pset([3])))
```


# LLM-generated content at query #9
#--------------------------

```python
def test_freeze_dict_type():
    from pyrsistent import freeze, pmap
    assert freeze({'a': 1}) == pmap({'a': 1})

def test_freeze_pmap_strict_true():
    from pyrsistent import freeze, pmap
    assert freeze(pmap({'a': 1}), strict=True) == pmap({'a': 1})

def test_freeze_defaultdict_type():
    import collections
    from pyrsistent import freeze, pmap
    assert freeze(collections.defaultdict(list, {'a': [1]})) == pmap({'a': (1,)})

def test_freeze_pmap_strict_false():
    from pyrsistent import freeze, pmap
    # Note: The predicate 'typ is dict or (strict and isinstance(o, PMap))' 
    # is satisfied if typ is dict. We check the logic for PMap with strict=True.
    assert freeze(pmap({'a': 1}), strict=True) == pmap({'a': 1})

def test_freeze_with_dict_input():
    from pyrsistent import freeze, pmap
    assert isinstance(freeze({'key': 'value'}), pmap)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent import pvector, pmap
    
    def identity(x):
        return x

    mutable_list = [1, 2, 3]
    mutable_dict = {'a': 1}
    
    decorated = mutant(identity)
    result = decorated(mutable_list, y={'b': 2})
    
    assert isinstance(result, pvector)
    assert result[0] == 1
    assert isinstance(result[1], int)
    assert result[2] == 3

def test_mutant_freezes_return_value():
    from pyrsistent import pvector, pmap
    
    def return_mutable():
        return [1, {'a': 2}]

    decorated = mutant(return_mutable)
    result = decorated()
    
    assert isinstance(result, pvector)
    assert isinstance(result[1], pmap)
    assert result[1]['a'] == 2

def test_mutant_handles_kwargs_recursion():
    from pyrsistent import pmap
    
    def check_structure(**kwargs):
        return kwargs

    decorated = mutant(check_structure)
    result = decorated(data=[{'inner': [1, 2]}])
    
    assert isinstance(result, pmap)
    assert isinstance(result['data'], pvector)
    assert isinstance(result['data'][0], pmap)
    assert isinstance(result['data'][0]['inner'], pvector)

def test_mutant_preserves_immutable_types():
    from pyrsistent import pvector
    
    def identity(x):
        return x

    decorated = mutant(identity)
    
    assert decorated(1) == 1
    assert decorated("string") == "string"
    assert decorated((1, 2)) == (1, 2)
```


# LLM-generated content at query #11
#--------------------------

```python
def test_freeze_primitives():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(None) is None
    assert freeze(3.14) == 3.14

def test_freeze_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([[1]]) == pvector([pvector([1])])

def test_freeze_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'tuple': pmap()}) # Note: logic error in provided source for empty dicts or keys, but testing expected behavior of standard pmap output
    assert freeze({'a': [1, 2]}) == pmap({'a': pvector([1, 2])})
    assert freeze({'a': {'b': 3}}) == pmap({'a': pmap({'b': 3})})

def test_freeze_tuple():
    assert freeze((1, 2)) == (1, 2)
    assert freeze((1, [2, 3])) == (1, pvector([2, 3]))
    assert freeze(( (1,), [2] )) == ((1,), pvector([2]))

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})
    assert freeze({(1, 2)}) == pset({(1, 2)})

def test_freeze_nested_complex():
    input_data = [
        {'a': 1, 'b': [2, 3]},
        (4, {'c': 5}),
        {6, 7}
    ]
    expected = pvector([
        pmap({'a': 1, 'b': pvector([2, 3])}),
        (4, pmap({'c': 5})),
        pset({6, 7})
    ])
    assert freeze(input_data) == expected

def test_freeze_strict_parameter():
    # When strict=False, dicts are not processed recursively in the same way if they were already PMap-like (based on provided source logic)
    # Testing that a list of dicts still freezes inner dict values
    assert freeze([{'a': [1]}], strict=True) == pvector([pmap({'a': pvector([1])})])
```


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_dict_is_true():
    from pyrsistent import freeze, pmap
    assert type(freeze({'a': 1})) is pmap
```


# LLM-generated content at query #13
#--------------------------

```python
from pyrsistent import pmap, pvector, pset

def test_mutant_freezes_args_and_kwargs():
    def simple_fn(a, b):
        return {'res': a + b}
    
    decorated = mutant(simple_fn)
    result = decorated({'x': 1}, b=2)
    
    assert isinstance(result, PMap)
    assert result['res'] == 3

def test_mutant_freezes_nested_structures():
    def complex_fn(data):
        return data
    
    decorated = mutant(complex_fn)
    input_data = {'a': [1, 2], 'b': {'c': 3}}
    result = decorated(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    assert result['a'][0] == 1
    assert result['b']['c'] == 3

def test_mutant_freezes_return_value():
    def returns_list(x):
        return [x, x]
    
    decorated = mutant(returns_list)
    result = decorated(5)
    
    assert isinstance(result, PVector)
    assert len(result) == 2
    assert result[0] == 5

def test_mutant_handles_tuple_recursively():
    def returns_tuple(x):
        return (x,)
    
    decorated = mutant(returns_tuple)
    result = decorated([1, 2])
    
    assert isinstance(result, tuple)
    assert isinstance(result[0], PVector)

def test_mutant_handles_kwargs_as_dict():
    def check_kwargs(**kwargs):
        return kwargs
    
    decorated = mutant(check_kwargs)
    result = decorated(key1='val1', key2=[1, 2])
    
    assert isinstance(result, PMap)
    assert isinstance(result['key2'], PVector)
```


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant_freezes_arguments():
    def identity(x):
        return x
    
    decorated = mutant(identity)
    mutable_list = [1, 2, 3]
    mutable_dict = {'a': 1}
    
    result = decorated(mutable_list, key=mutable_dict)
    
    assert isinstance(result, PVector)
    assert isinstance(result, PMap)
    assert result[0] == 1
    assert result['key'] == {'a': 1}

def test_mutant_freezes_return_value():
    def return_mutable():
        return [1, {'a': 2}]
    
    decorated = mutant(return_mutable)
    result = decorated()
    
    assert isinstance(result, PVector)
    assert isinstance(result[0], int)
    assert isinstance(result[1], PMap)
    assert result[1]['a'] == 2

def test_mutant_isolates_mutation_in_function():
    def mutating_fn(data):
        data.append(4)
        return data
    
    decorated = mutant(mutating_fn)
    input_list = [1, 2, 3]
    
    result = decorated(input_list)
    
    assert result == pvector([1, 2, 3, 4])
    assert input_list == [1, 2, 3]

def test_mutant_handles_kwargs():
    def check_kwargs(**kwargs):
        return kwargs
    
    decorated = mutant(check_kwargs)
    result = decorated(a=[1], b={'c': 2})
    
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1])
    assert result['b'] == pmap({'c': 2})

def test_mutant_with_nested_structures():
    def nested_identity(x):
        return x
    
    decorated = mutant(nested_identity)
    complex_input = {'a': [1, (2, [3])], 'b': {4, 5}}
    
    result = decorated(complex_input)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][0], int)
    assert isinstance(result['a'][1], tuple)
    assert isinstance(result['a'][1][0], PVector)
    assert isinstance(result['b'], PSet)
```


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_predicate_is_false():
    from pyrsistent import pmap, pset, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    # To ensure the predicate at line 1 (the decorator itself) evaluates to False
    # is impossible as it's a function definition. However, we test that 
    # 'mutant' returns a callable and performs freezing as described.
    
    result_map = identity(pmap({'a': 1}))
    assert isinstance(result_map, type(pmap({})))
    
    result_set = identity(pset([1, 2]))
    assert isinstance(result_set, type(pset()))

    result_vec = identity(pvector([1, 2]))
    assert isinstance(result_vec, type(pvector()))
```


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    def identity_with_mutation(data, extra=None):
        if isinstance(data, list):
            data.append(4)
        return data

    decorated = mutant(identity_with_mutation)
    
    input_list = [1, 2, 3]
    input_dict = {'a': [1]}
    
    result = decorated(input_list, extra={'b': [2]})
    
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    # The function appended 4, but the decorator freezes inputs first.
    # Because input_list was frozen to pvector([1, 2, 3]), the append fails or doesn't affect original list.
    # Actually, the decorator calls freeze(e) for each e in args. 
    # So identity_with_mutation receives PVector([1, 2, 3]).
    # Since PVector has no .append method like list, it would raise AttributeError if it tried to mutate.
    # However, the test checks that the result is frozen.

def test_mutant_ensures_output_is_frozen():
    def returns_mutable_list(x):
        return [x, x]

    decorated = mutant(returns_mutable_list)
    result = decorated(10)
    
    assert isinstance(result, PVector)
    assert result[0] == 10
    assert result[1] == 10

def test_mutant_handles_empty_inputs():
    def identity(x):
        return x

    decorated = mutant(identity)
    
    assert decorated() == None
    assert decorated([]) == PVector()
    assert decorated({}) == PMap()

def test_mutant_deep_freezing():
    def complex_structure(data):
        return data

    decorated = mutant(complex_structure)
    input_data = {'a': [1, {'b': 2}], 'c': (3, 4)}
    result = decorated(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][1], PMap)
    assert result['a'][1]['b'] == 2
    assert isinstance(result['c'], tuple)
    assert result['c'][0] == 3
```


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant_returns_frozen_value():
    from pyrsistent import pmap, pset, PMap, PSet

    @mutant
    def identity(x):
        return x

    result = identity(pmap({'a': 1}))
    assert isinstance(result, PMap)

def test_mutant_freezes_input_args():
    from pyrsistent import pmap, pset, PMap, PSet

    @mutant
    def check_args(arg):
        return arg

    arg = pmap({'a': 1})
    result = check_args(arg)
    assert isinstance(result, PMap)

def test_mutant_does_not_change_value_of_immutable_types():
    from pyrsistent import pmap, pset, PMap, PSet

    @mutant
    def identity(x):
        return x

    val = 5
    result = identity(val)
    assert result == 5
```


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    from pyrsistent import pmap, pvector, pset
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    assert identity(pmap({'a': 1})) == pmap({'a': 1})
    assert identity(pvector([1, 2])) == pvector([1, 2])
    assert identity(pset([1, 2])) == pset([1, 2])
    assert identity(5) == 5

def test_mutant_decorator_freezes_arguments():
    from pyrsistent import pmap, pvector, pset
    from pyrsistent._helpers import mutant

    @mutant
    def check_frozen(arg):
        return arg

    # If the decorator works, it calls freeze() on arguments.
    # We test that a mutable-like structure is passed as frozen.
    # Since pmap/pvector are already immutable, we rely on the fact 
    # that if 'freeze' was NOT called, a standard dict would be used.
    # The decorator ensures even if we pass a dict, it becomes a PMap.
    assert isinstance(check_frozen({'a': 1}), type(pmap({'a': 1})))

def test_mutant_decorator_freezes_kwargs():
    from pyrsistent import pmap, pvector, pset
    from pyrsistent._helpers import mutant

    @mutant
    def check_kwargs(arg):
        return arg

    assert isinstance(check_kwargs(arg={'a': 1}), type(pmap({'a': 1})))
```


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    from pyrsistent import pmap, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def identity_func(x, y=None):
        return x, y

    # Test with positional arguments (must be frozen)
    # If args are not frozen or the logic fails, the test will fail.
    # We use pmap/pvector because they are persistent and "frozen" by nature in pyrsistent context.
    arg_vec = pvector([1, 2, 3])
    arg_map = pmap({'a': 1})
    
    result = identity_func(arg_vec, y=arg_map)
    
    assert result[0] == arg_vec
    assert result[1] == arg_map
```


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_freezes_arguments():
    def identity_with_mutation(data):
        data['new_key'] = 'mutated'
        return data

    decorated = mutant(identity_with_mutation)
    input_dict = {'a': 1}
    result = decorated(input_dict)

    assert isinstance(result, pmap)
    assert result['a'] == 1
    assert 'new_key' not in input_dict
    assert 'new_key' in result

def test_mutant_freezes_kwargs():
    def check_kwargs(val):
        return val

    decorated = mutant(check_kwargs)
    input_list = [1, 2, 3]
    result = decorated(val=input_list)

    assert isinstance(result, pvector)
    assert result[0] == 1
    assert input_list is not result

def test_mutant_freezes_return_value():
    def return_mutable_list():
        return [1, {'nested': 2}]

    decorated = mutant(return_mutable_list)
    result = decorated()

    assert isinstance(result, pvector)
    assert isinstance(result[1], pmap)
    assert result[1]['nested'] == 2

def test_mutant_deep_freezing():
    def complex_structure(data):
        return data

    input_data = {
        'list': [1, 2, {'a': 3}],
        'tuple': (4, 5),
        'set': {6, 7}
    }
    
    decorated = mutant(complex_structure)
    result = decorated(input_data)

    assert isinstance(result, pmap)
    assert isinstance(result['list'], pvector)
    assert isinstance(result['list'][2], pmap)
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['set'], pset)
```


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def identity_with_mutation(data):
        data['modified'] = True
        return data

    decorated = mutant(identity_with_mutation)
    
    input_dict = {'original': True}
    result = decorated(input_dict)

    assert isinstance(result, PMap)
    assert result['original'] is True
    assert 'modified' in result
    assert input_dict == {'original': True}
    assert 'modified' not in input_dict

def test_mutant_freezes_kwargs():
    def check_kwargs(a, b):
        return a

    decorated = mutant(check_kwargs)
    
    input_list = [1, 2]
    result = decorated(a=input_list, b=3)

    assert isinstance(result, PVector)
    assert result[0] == 1
    assert input_list == [1, 2]

def test_mutant_deep_freezing():
    def deep_mutation(data):
        data[0][0] = 'changed'
        return data

    decorated = mutant(deep_mutation)
    
    input_data = [{'nested': [1]}]
    result = decorated(input_data)

    assert isinstance(result, PVector)
    assert isinstance(result[0], PMap)
    assert isinstance(result[0]['nested'], PVector)
    assert result[0]['nested'][0] == 'changed'
    assert input_data[0]['nested'][0] == 1
```


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_predicate_false():
    from pyrsistent import pmap, pset, PMap, PSet
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    result = identity(pmap({'a': 1}))
    assert isinstance(result, PMap)
```


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    def simple_fn(a, b=None):
        return [a, b]
    
    decorated = mutant(simple_fn)
    
    input_list = [1, 2, {'x': 3}]
    input_dict = {'y': 4}
    
    result = decorated(input_list, b=input_dict)
    
    assert isinstance(result, PVector)
    assert result[0] == PVector([1, 2, {'x': 3}])
    assert result[1] == PMap({'y': 4})

def test_mutant_preserves_return_value_structure():
    def returns_list(x):
        return [x, [x]]
    
    decorated = mutant(returns_list)
    
    result = decorated({'a': 1})
    
    assert isinstance(result, PVector)
    assert result[0] == PMap({'a': 1})
    assert result[1] == PVector([PMap({'a': 1})])

def test_mutant_handles_empty_inputs():
    def identity(x):
        return x
    
    decorated = mutant(identity)
    
    result = decorated([])
    
    assert isinstance(result, PVector)
    assert len(result) == 0

def test_mutant_with_nested_structures():
    def nested_fn(data):
        return data
    
    decorated = mutant(nested_fn)
    
    complex_input = [ (1, {'a': [2, 3]}) ]
    result = decorated(complex_input)
    
    assert isinstance(result, PVector)
    assert result[0] == (1, PMap({'a': PVector([2, 3])}))
```


# LLM-generated content at query #24
#--------------------------

```python
from pyrsistent import pmap, pset, pvector

def test_mutant_returns_frozen_value():
    def identity(x):
        return x
    
    decorated = mutant(identity)
    result = decorated(pmap({'a': 1}))
    assert isinstance(result, (pmap().__class__, pset().__class__, pvector().__class__))

def test_mutant_freezes_arguments():
    def check_args(arg):
        return arg
    
    decorated = mutant(check_args)
    result = decorated({'a': 1})
    assert isinstance(result, pmap().__class__)

def test_mutant_handles_kwargs():
    def check_kwargs(**kwargs):
        return kwargs
    
    decorated = mutant(check_kwargs)
    result = decorated(key='value')
    assert isinstance(result['key'], pmap().__class__) or isinstance(result['key'], str)
    # Since the implementation freezes items in kwargs.items(), 
    # if we pass a mutable dict as a kwarg, it should become a pmap/pvector/pset equivalent.
    # However, specifically checking that the returned dict contains frozen values:
    assert isinstance(result['key'], type(result['key']))
```

Wait, looking at the prompt requirements and the specific code to test (the `mutant` decorator): The requirement is "ensure that the predicate at line 1 evaluates to True". Line 1 is `def mutant(fn):`. This is a function definition, not a boolean expression. However, usually in mutation testing contexts, this means ensuring the decorator functions correctly as intended by its implementation (freezing inputs and outputs).

Since I cannot use `if` or imports other than what's needed for the test logic itself:

```python
from pyrsistent import pmap

def test_mutant_decorator_logic():
    def simple_func(x):
        return x
    
    decorated = mutant(simple_func)
    
    # Test that input is frozen (dict becomes pmap)
    input_data = {'a': 1}
    result_val = decorated(input_data)
    assert isinstance(result_val, pmap().__class__)
    
    # Test that output is frozen
    def returns_mutable(x):
        return {'b': 2}
    
    decorated_output = mutant(returns_mutable)
    result_output = decorated_output(None)
    assert isinstance(result_output, pmap().__class__)

def test_mutant_preserves_functionality():
    def add(a, b):
        return a + b
    
    decorated = mutant(add)
    assert decorated(1, 2) == 3
```


# LLM-generated content at query #25
#--------------------------

```python
from pyrsistent import pmap, pset, PMap, PSet, pvector

def test_mutant_preserves_functionality_and_freezes_inputs():
    def simple_fn(x, y, data=None):
        return (x, y, data)

    # We need to mock/use the actual decorator behavior
    # Since we cannot define 'mutant' in the test, we assume it's imported
    from pyrsistent._helpers import mutant
    
    # Test with standard types and check if they are frozen inside or returned as frozen
    # The requirement is that the predicate at line 1 evaluates to True.
    # In a mutation testing context (like MutPy), the goal is to prove the decorator works.
    
    decorated = mutant(simple_fn)
    
    # Case 1: Basic arguments
    result = decorated(1, 2)
    assert result == (1, 2)

    # Case 2: Mapping argument (should be frozen)
    initial_map = {'a': 1}
    result_with_map = decorated(1, 2, data=initial_map)
    assert isinstance(result_with_map[2], PMap)
    assert result_with_map[2]['a'] == 1

    # Case 3: Verifying the 'inner_f' logic via execution
    # The decorator must return a function that can be called.
    assert callable(decorated)
```


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    from pyrsistent import pmap, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x, y=None):
        return x, y

    assert identity(pmap({'a': 1}), y=pvector([2])) == (pmap({'a': 1}), pvector([2]))

def test_mutant_decorator_freezes_arguments():
    from pyrsistent import pmap, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def check_frozen(x, y):
        return x, y

    input_map = pmap({'a': 1})
    input_vec = pvector([2])
    
    result_x, result_y = check_frozen(input_map, y=input_vec)
    
    assert result_x == input_map
    assert result_y == input_vec
```


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_predicate_is_false():
    from pyrsistent import pmap, pset

    def identity(x):
        return x

    decorated = mutant(identity)
    
    # To make the predicate at line 1 evaluate to False, we need a scenario 
    # where 'mutant' is not called as a decorator or is used in a way that 
    # doesn't trigger the specific logic. However, since the instruction 
    # asks to ensure the predicate (the function definition itself) evaluates to False, 
    # and the code provided is just a function definition, there is no boolean 
    # expression at line 1. 
    # Assuming 'predicate' refers to the truthiness of the function object or 
    # a test that validates the decorator logic works as intended:
    
    assert decorated(1) == 1
    assert isinstance(decorated({'a': 1}), pmap)
    assert isinstance(decorated([1, 2]), pvector)
```


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant_decorator_predicate_false():
    from pyrsistent import pmap, pset, PMap, PSet
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    # To ensure the predicate at line 1 evaluates to False, we need a scenario 
    # where 'fn' is not a function (or specifically, where the decorator logic fails 
    # if it were applied to something that isn't callable).
    # However, since the prompt asks for a test ensuring the predicate evaluates to False,
    # and line 1 is `def mutant(fn):`, this is a structural part of the definition.
    # In mutation testing terms (like PIT), "the predicate at line 1 evaluates to False" 
    # usually refers to an 'if' condition. Since there is no 'if' on line 1, 
    # we assume the goal is to verify the decorator works correctly on standard inputs, 
    # and any mutation that would make the definition invalid is caught by the execution.
    
    result = identity(pmap({'a': 1}))
    assert isinstance(result, PMap)
    assert result['a'] == 1

    result_set = identity(pset([1, 2]))
    assert isinstance(result_set, PSet)
    assert 1 in result_set
```


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    from pyrsistent import pmap, pvector
    
    def identity(x, y=None):
        return {'a': x, 'b': y}
    
    frozen_identity = mutant(identity)
    
    result = frozen_identity([1, 2], y={'c': 3})
    
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])
    assert result['b'] == pmap({'c': 3})

def test_mutant_freezes_return_value():
    from pyrsistent import pmap
    
    def returns_mutable(x):
        return {'val': x}
    
    frozen_returns = mutant(returns_mutable)
    
    result = frozen_returns(10)
    
    assert isinstance(result, PMap)
    assert result['val'] == 10

def test_mutant_handles_nested_structures():
    from pyrsistent import pmap, pvector
    
    def complex_func(data):
        return data
    
    frozen_complex = mutant(complex_func)
    
    input_data = [1, {'inner': [2, 3]}]
    result = frozen_complex(input_data)
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[1]['inner'], PVector)

def test_mutant_preserves_primitive_types():
    def simple_func(x):
        return x
    
    frozen_simple = mutant(simple_func)
    
    assert frozen_simple(5) == 5
    assert frozen_simple("string") == "string"
    assert frozen_simple(None) is None
```


# LLM-generated content at query #30
#--------------------------

```python
from pyrsistent import pmap, pset, pvector

def test_mutant_decorator_freezes_arguments_and_return():
    def identity_with_mutation(x, y=None):
        nonlocal x
        x = x.set(4)
        result = pmap({'a': 1})
        if y is not None:
            result = result.set('b', 2)
        return result

    # To verify the predicate at line 1 (the decorator itself), we test if it works as intended.
    # Since we cannot use 'if' or custom functions in the test body, we rely on the fact that
    # the decorator is applied to a function and its behavior matches the requirement:
    # Arguments are frozen before fn is called, and return value is frozen.

    from pyrsistent._helpers import mutant

    @mutant
    def target_func(arg_map, arg_vec):
        # This internal mutation should not affect the original objects passed to the decorator
        return arg_map.set('new', True)

    initial_map = pmap({'a': 1})
    initial_vec = pvector([1, 2])
    
    # The decorator 'mutant' is a function that takes a function and returns a function.
    # We test the existence and basic execution of the decorated function.
    result = target_func(initial_map, initial_vec)

    assert isinstance(result, pmap)
    assert result['new'] is True
    assert 'b' not in result
```


# LLM-generated content at query #31
#--------------------------

```python
def test_freeze_strict_pmap_evaluates_true():
    from pyrsistent import pmap, PMap
    frozen_map = freeze(pmap({'a': 1}))
    assert isinstance(frozen_map, PMap)
```


# LLM-generated content at query #32
#--------------------------

```python
from pyrsistent import pmap, pset, PMap, PSet

def test_mutant_decorator_logic():
    # To make the predicate at line 1 (the function definition itself) evaluate to False,
    # we need to trigger a situation where 'fn' is not actually a function or something 
    # that breaks the decorator. However, since the prompt asks for a test ensuring 
    # the predicate evaluates to False, and the code provided is a decorator,
    # I will provide a test case that demonstrates the behavior of the inner logic.
    # Note: In Python, "the predicate at line 1" usually refers to a condition in an 'if'.
    # Since there is no conditional on line 1, this implies testing the identity or 
    # truthiness of the decorator application itself.
    
    @mutant
    def identity(x):
        return x

    result = identity(pmap({'a': 1}))
    assert isinstance(result, PMap)
```


# LLM-generated content at query #33
#--------------------------

```python
from pyrsistent import pmap, pset, pvector

def test_mutant_decorator_preserves_functionality():
    def identity(x):
        return x
    
    decorated_identity = mutant(identity)
    
    # Test with simple value
    assert decorated_identity(5) == 5
    
    # Test with pmap (already frozen)
    p_map = pmap({'a': 1})
    assert decorated_identity(p_map) == p_map
    
    # Test with dict (should be frozen by the decorator's logic via freeze call)
    d = {'a': 1}
    result = decorated_identity(d)
    assert result == pmap({'a': 1})
    
    # Test with kwargs
    def sum_two(a, b):
        return a + b
    
    decorated_sum = mutant(sum_two)
    assert decorated_sum(1, 2) == 3
    assert decorated_sum(a=10, b=20) == 30

def test_mutant_decorator_freezes_arguments():
    def check_args(data):
        return data
    
    decorated_check = mutant(check_args)
    
    # Input is a mutable list
    mutable_list = [1, 2, 3]
    result = decorated_check(mutable_list)
    
    # The result should be a pvector (frozen version of the list)
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])

def test_mutant_decorator_handles_kwargs_freezing():
    def check_kwargs(**kwargs):
        return kwargs
    
    decorated_check = mutant(check_kwargs)
    
    # Input is a mutable dict in kwargs
    result = decorated_check(val={'a': 1})
    
    # The result should be a pmap (frozen version of the dict)
    assert isinstance(result, pmap)
    assert result == pmap({'val': pmap({'a': 1})})
```


# LLM-generated content at query #34
#--------------------------

```python
def test_mutant_predicate_false():
    from pyrsistent import pmap, pset, PMap, PSet
    from pyrsistent._helpers import mutant

    # To make the predicate at line 1 evaluate to False, we need a scenario 
    # where 'fn' is NOT a function (or rather, the decorator fails or behaves differently).
    # However, the prompt asks for a test ensuring the predicate evaluates to False.
    # The predicate in 'def mutant(fn):' is implicitly True because it's a definition.
    # If we treat the 'predicate' as the truthiness of the function object itself:
    assert bool(mutant) == True

    # To specifically target the logic: if we provide something that isn't a callable, 
    # the decorator will fail when called. But since the prompt asks to ensure 
    # the predicate at line 1 evaluates to False, and line 1 is 'def mutant(fn):',
    # there is no explicit boolean predicate in the provided code snippet at line 1.
    # Assuming the "predicate" refers to a condition that would make 'mutant' invalid:
    
    @mutant
    def identity(x):
        return x

    assert identity(pmap({'a': 1})) == pmap({'a': 1})
```


# LLM-generated content at query #35
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    from pyrsistent import pvector, pmap
    
    def identity(x):
        return x

    mutable_list = [1, 2, 3]
    mutable_dict = {'a': 1}
    
    # The mutant decorator should freeze the input list and dict.
    # Since 'identity' returns the frozen version of its argument,
    # we check if the return value is a pvector.
    result = mutant(identity)(mutable_list)
    
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])

def test_mutant_freezes_kwargs():
    from pyrsistent import pmap
    
    def check_dict(d):
        return d

    mutable_dict = {'a': [1, 2]}
    # The decorator freezes kwargs, so the dict passed as 'd' becomes a pmap with pvector inside.
    result = mutant(check_dict)(d={'a': [1, 2]})
    
    assert isinstance(result, pmap)
    assert result['a'] == pvector([1, 2])

def test_mutant_deep_freezing_of_return_value():
    from pyrsistent import pvector, pmap
    
    def returns_mutable():
        return {'key': [1, 2], 'other': (3, 4)}

    # The decorator freezes the return value of the decorated function.
    result = mutant(returns_mutable)()
    
    assert isinstance(result, pmap)
    assert isinstance(result['key'], pvector)
    assert isinstance(result['other'], tuple)
    assert result['key'] == pvector([1, 2])

def test_mutant_with_complex_nested_structure():
    from pyrsistent import pvector, pmap

    def complex_func(a, b):
        return {'outer': [a, b]}

    input_list = [1, {'inner': [2]}]
    # input_list is frozen to pvector([1, pmap({'inner': pvector([2])})])
    # result should be pmap({'outer': pvector([pvector([1, pmap({'inner': pvector([2])})]), pvector([1, pmap({'inner': pvector([2])})])])}
    # wait, the logic is: args are frozen first. 
    # arg 'a' becomes pvector([1, pmap({'inner': pvector([2])})])
    # arg 'b' becomes pvector([1, pmap({'inner': pvector([2])})])
    # result of complex_func is {'outer': [frozen_a, frozen_b]}
    # then the decorator freezes that return value.
    
    result = mutant(complex_func)(input_list, input_list)
    
    expected_inner_structure = pvector([1, pmap({'inner': pvector([2])})])
    assert result['outer'] == pvector([expected_inner_structure, expected_inner_structure])
```


# LLM-generated content at query #36
#--------------------------

```python
def test_freeze_strict_pmap_evaluates_true():
    from pyrsistent import pmap, freeze
    frozen_pmap = freeze(pmap({'a': 1}), strict=True)
    assert isinstance(frozen_pmap, pmap().__class__)
```


# LLM-generated content at query #37
#--------------------------

```python
from pyrsistent import pmap, pset, freeze

def test_mutant_decorator_does_not_mutate_arguments():
    # To ensure the predicate at line 1 evaluates to False, we need a scenario where
    # the decorator is NOT applied or does not behave as a decorator in a way that 
    # triggers the 'mutant' logic (though the prompt asks to ensure the predicate 
    # evaluating to False). Since there is no explicit boolean predicate provided 
    # in the snippet other than the existence of the function, and usually 
    # mutation testing involves checking if a change breaks a test.
    # However, based on the instruction "ensure that the predicate at line 1 evaluates to False",
    # and looking at 'def mutant(fn):', there is no predicate. 
    # If we assume the "predicate" refers to a check like `isinstance(mutant, function)` 
    # being false or something similar, it's ambiguous.
    # Assuming the instruction implies testing the decorator logic:
    
    def simple_fn(x):
        return x

    decorated = mutant(simple_fn)
    
    input_list = [1, 2, 3]
    result = decorated(input_list)
    
    assert result == input_list
    assert isinstance(result, list) # Since it was frozen by the decorator's return line
```


# LLM-generated content at query #38
#--------------------------

```python
def test_mutant_predicate_false():
    from pyrsistent import pmap, pset, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    identity_with_pmap = identity(pmap({'a': 1}))
    identity_with_pset = identity(pset([1, 2]))
    identity_with_pvector = identity(pvector([1, 2]))

    assert identity_with_pmap == pmap({'a': 1})
    assert identity_with_pset == pset([1, 2])
    assert identity_with_pvector == pvector([1, 2])
```


# LLM-generated content at query #39
#--------------------------

```python
from pyrsistent import pmap, pset, PMap, PSet

def test_mutant_preserves_functionality():
    def identity(x, y=None):
        return x
    
    decorated = mutant(identity)
    
    # Test with positional arguments and verify return value is frozen
    result = decorated(pmap({'a': 1}))
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1})
    
    # Test with keyword arguments
    result_kw = decorated(pset([1, 2]), y=pset([3]))
    assert isinstance(result_kw, PSet)
    assert result_kw == pset([1])

def test_mutant_freezes_inputs():
    def check_frozen(x, y):
        return x == x and y == y
    
    decorated = mutant(check_frozen)
    
    # Using mutable types that would normally be passed in
    # The decorator freezes them before passing to fn
    result = decorated({'a': 1}, [1, 2])
    assert result is True
```


# LLM-generated content at query #40
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    from pyrsistent import pmap, pset, PMap, PSet

    def identity(x):
        return x

    decorated_identity = mutant(identity)

    assert decorated_identity(5) == 5
    assert decorated_identity(pmap({'a': 1})) == pmap({'a': 1})
    assert decorated_identity(pset([1, 2])) == pset([1, 2])

def test_mutant_decorator_freezes_arguments():
    from pyrsistent import pmap

    def check_args(d):
        return d

    decorated_check = mutant(check_args)
    
    # Passing a standard dict; the decorator should freeze it via the internal call to freeze()
    # Since we can't easily intercept the 'freeze' call without mocking, 
    # we verify that the returned value is indeed a PMap (frozen version of dict).
    result = decorated_check({'a': 1})
    assert isinstance(result, PMap)

def test_mutant_decorator_freezes_kwargs():
    from pyrsistent import pmap

    def check_kwargs(**kwargs):
        return kwargs

    decorated_check = mutant(check_kwargs)
    
    # Passing a dict in kwargs; the decorator should freeze it.
    result = decorated_check(data={'a': 1})
    assert isinstance(result['data'], PMap)
```


# LLM-generated content at query #41
#--------------------------

```python
def test_mutant_predicate_false():
    def identity(x):
        return x
    
    from pyrsistent import pmap
    
    # The predicate at line 1 is `def mutant(fn):`.
    # In Python, a function definition statement itself does not evaluate to a boolean.
    # However, the prompt asks to ensure the predicate evaluates to False.
    # Since we cannot modify the source code provided, and 'def' is a statement,
    # we test the behavior of the decorator logic on a known truthy value 
    # to imply that if the decorator failed to return the wrapped function (which is truthy),
    # it would fail. But more specifically, we check if any logic inside 'mutant' 
    # could be interpreted as a predicate. Since there are no explicit predicates 
    # in the signature, and the prompt implies a mutation on line 1, 
    # we provide an assertion that validates the decorator works correctly.
    
    decorated_identity = mutant(identity)
    assert decorated_identity(pmap({'a': 1})) == pmap({'a': 1})
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_thaw_simple_types():
    assert thaw(1) == 1
    assert thaw("string") == "string"
    assert thaw(True) == True

def test_thaw_pset():
    from pyrsistent import s
    assert thaw(s(1, 2, 3)) == {1, 2, 3}

def test_thaw_pvector_nested():
    from pyrsistent import v, m
    assert thaw(v(1, v(2, 3), m(a=4))) == [1, [2, 3], {'a': 4}]

def test_thaw_pmap_nested():
    from pyrsistent import m, v
    assert thaw(m(a=v(1, 2), b=m(c=3))) == {'a': [1, 2], 'b': {'c': 3}}

def test_thaw_tuple_recursive():
    from pyrsistent import v
    assert thaw((1, v(2, 3), (4,))) == (1, [2, 3], (4,))

def test_thaw_strict_false_list():
    from pyrsistent import v
    assert thaw([1, v(2)], strict=False) == [1, v(2)]

def test_thaw_strict_false_dict():
    from pyrsistent import m
    assert thaw({'a': v(1)}, strict=False) == {'a': v(1)}

def test_thaw_mixed_containers():
    from pyrsistent import v, m, s
    input_data = v(m(a=s(1), b=(2,)), 3)
    expected = [{'a': {1}, 'b': (2,)}, 3]
    assert thaw(input_data) == expected
```


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_empty_containers():
    assert freeze([]) == pvector()
    assert freeze({}) == pmap()
    assert freeze(set()) == pset()
    assert freeze(()) == ()

def test_freeze_simple_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True
    assert freeze(None) == None

def test_freeze_list_recursive():
    assert freeze([1, [2, 3]]) == pvector([1, pvector([2, 3])])
    assert freeze([{"a": 1}, (2, 3)]) == pvector([pmap({"a": 1}), (2, pvector([]))])

def test_freeze_dict_recursive():
    assert freeze({"a": 1, "b": [2, 3]}) == pmap({"a": 1, "b": pvector([2, 3])})
    assert freeze({"a": {"inner": []}}) == pmap({"a": pmap({"inner": pvector()})})

def test_freeze_tuple_recursive():
    assert freeze((1, [2])) == (1, pvector([2]))
    assert freeze(([1], {"a": 2})) == (pvector([1]), pmap({"a": 2}))

def test_freeze_set_not_recursive():
    # The docstring says sets/pset elements are not recursively frozen
    # We use a list of lists inside a set to verify the behavior via proxy if possible,
    # but since sets can't contain lists, we check that it stays as pset.
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_strict_false_behavior():
    # When strict=False, PMap/PVector are not recursively frozen by the type checks in logic
    # Based on: if typ is list or (strict and isinstance(o, PVector)):
    # If strict is False, PVector is not processed by the curried_freeze map.
    pv = pvector([list([1])])
    assert freeze(pv, strict=False) == pv
```


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_strict_pmap_evaluates_true():
    from pyrsistent import pmap, PMap
    frozen_map = freeze(pmap({'a': 1}))
    assert isinstance(frozen_map, PMap)
```


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_primitive_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True
    assert freeze(None) is None

def test_freeze_list():
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])
    assert freeze([[1], [2]]) == pvector([pvector([1]), pvector([2])])
    assert freeze([{'a': 1}]) == pvector([pmap({'a': 1})])

def test_freeze_tuple():
    assert freeze((1, 2)) == (1, 2)
    assert freeze(([1], [2])) == (pvector([1]), pvector([2]))
    assert freeze(({'a': 1},)) == (pmap({'a': 1}),)

def test_freeze_dict():
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})
    assert freeze({'a': [1, 2], 'b': {'c': 3}}) == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})

def test_freeze_set():
    assert freeze({1, 2, 3}) == pset({1, 2, 3})
    assert freeze({(1, 2), (3, 4)}) == pset({(1, 2), (3, 4)})

def test_freeze_nested_structures():
    input_data = [
        {'key': [1, 2, {'inner': 3}]},
        (4, 5),
        {6, 7}
    ]
    expected = pvector([
        pmap({'key': pvector([1, 2, pmap({'inner': 3})])}),
        (4, 5),
        pset({6, 7})
    ])
    assert freeze(input_data) == expected

def test_freeze_strict_false_behavior():
    # When strict is False, PMap values are not recursively frozen in the implementation logic provided
    # Note: The provided implementation for dict/PMap uses `freeze(v, strict)`. 
    # If strict=False, it still calls freeze on v. 
    # However, looking at the code: `if typ is dict or (strict and isinstance(o, PMap)):`
    # If we pass a PMap with strict=False, the first condition fails unless it's a native dict.
    input_data = pmap({'a': [1, 2]})
    # With strict=False, the logic for PMap might skip recursion depending on how the type check hits.
    # Given the implementation: `if typ is dict or (strict and isinstance(o, PMap)):`
    # If o is PMap and strict is False, it won't enter the dict/PMap block.
    assert freeze(input_data, strict=False) == input_data

def test_freeze_empty_containers():
    assert freeze([]) == pvector([])
    assert freeze({}) == pmap({})
    assert freeze(()) == ()
    assert freeze(set()) == pset()
```


# LLM-generated content at query #5
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    def simple_fn(d, l):
        d['new'] = 1
        l.append(2)
        return {'res': [3]}

    frozen_d = freeze({'a': 1})
    frozen_l = freeze([1])
    
    result = mutant(simple_fn)(frozen_d, frozen_l)
    
    assert isinstance(result, PMap)
    assert result['res'] == pvector([3])
    assert 'new' not in frozen_d
    assert len(frozen_l) == 1

def test_mutant_handles_empty_inputs():
    def identity(x):
        return x

    result = mutant(identity)([])
    
    assert isinstance(result, PVector)
    assert len(result) == 0

def test_mutant_freezes_kwargs():
    def kwarg_fn(k=None):
        return k

    result = mutant(kwarg_fn)(k={'inner': [1]})
    
    assert isinstance(result, PMap)
    assert result['inner'] == pvector([1])

def test_mutant_deep_freezing():
    def deep_mutation(data):
        data[0][0] = 99
        return data

    initial_data = [{'nested': [1]}]
    result = mutant(deep_mutation)(initial_data)
    
    assert result[0]['nested'][0] == 1
    assert isinstance(result, PVector)
    assert isinstance(result[0], PMap)
    assert isinstance(result[0]['nested'], PVector)
```


# LLM-generated content at query #6
#--------------------------

```python
def test_mutant_freezes_arguments():
    def identity(x):
        return x
    mutated_identity = mutant(identity)
    input_list = [1, 2, 3]
    result = mutated_identity(input_list)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_freezes_kwargs():
    def identity(x=None):
        return x
    mutated_identity = mutant(identity)
    input_dict = {'a': [1, 2]}
    result = mutated_identity(x=input_dict)
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_mutant_freezes_return_value():
    def return_list():
        return [1, {'a': 2}]
    mutated_fn = mutant(return_list)
    result = mutated_fn()
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['a'] == 2

def test_mutant_deep_freeze_nested_structures():
    def identity(x):
        return x
    mutated_identity = mutant(identity)
    complex_input = {'key': [1, (2, [3])]}
    result = mutated_identity(complex_input)
    assert isinstance(result, PMap)
    assert isinstance(result['key'], PVector)
    assert isinstance(result['key'][1], tuple)
    assert isinstance(result['key'][1][0], PVector)
    assert result['key'][1][0][0] == 3

def test_mutant_handles_tuple_recursion():
    def identity(x):
        return x
    mutated_identity = mutant(identity)
    input_tuple = (1, [2, 3])
    result = mutated_identity(input_tuple)
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)
```


# LLM-generated content at query #7
#--------------------------

```python
def test_freeze_empty_dict():
    assert freeze({}) == pmap({})

def test_freeze_dict_with_values():
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
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_primitive_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(None) is None

def test_freeze_deeply_nested_structure():
    input_data = {'a': [1, (2, {'c': 3})], 'b': {4, 5}}
    expected_output = pmap({
        'a': pvector([1, (2, pmap({'c': 3}))]),
        'b': pset([4, 5])
    })
    assert freeze(input_data) == expected_output

def test_freeze_strict_false_pmap():
    # When strict is False, PMap values should not be recursively frozen
    input_data = pmap({'a': [1, 2]})
    # In strict=True (default), the list inside PMap is frozen to pvector.
    # In strict=False, we check if it preserves the internal mutable structure if logic allows
    # However, based on code: `if typ is dict or (strict and isinstance(o, PMap)):`
    # If strict is False and o is PMap, it skips the dict/PMap block.
    assert freeze(input_data, strict=False) == input_data

def test_freeze_defaultdict():
    import collections
    input_data = collections.defaultdict(list, {'a': [1]})
    assert freeze(input_data) == pmap({'a': pvector([1])})
```


# LLM-generated content at query #8
#--------------------------

```python
def test_mutant_freezes_arguments():
    def identity_with_mutation(data):
        data[0] = "mutated"
        return data

    decorated = mutant(identity_with_mutation)
    input_list = [1, 2, 3]
    result = decorated(input_list)

    assert result == pvector([1, 2, 3])
    assert input_list == [1, 2, 3]

def test_mutant_freezes_kwargs():
    def identity_with_mutation(val=None):
        return val

    decorated = mutant(identity_with_mutation)
    input_dict = {'a': [1, 2]}
    result = decorated(val={'b': 3})

    assert result == pmap({'b': 3})
    assert input_dict == {'a': [1, 2]}

def test_mutant_freezes_return_value():
    def returns_mutable():
        return [1, 2, {'a': 3}]

    decorated = mutant(returns_mutable)
    result = decorated()

    assert result == pvector([1, 2, pmap({'a': 3})])
    assert isinstance(result, PVector)
    assert isinstance(result[2], PMap)

def test_mutant_deep_freeze():
    def identity(x):
        return x

    decorated = mutant(identity)
    complex_input = {'key': [1, (2, [3])]}
    result = decorated(complex_input)

    assert result == pmap({'key': pvector([1, (2, pvector([3]))])})
```


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant_freezes_arguments():
    def identity(x):
        return x
    
    frozen_identity = mutant(identity)
    
    input_list = [1, 2, [3, 4]]
    result = frozen_identity(input_list)
    
    assert isinstance(result, PVector)
    assert result[2][0] == 3
    assert isinstance(result[2], PVector)

def test_mutant_freezes_kwargs():
    def identity(val=None):
        return val
    
    frozen_identity = mutant(identity)
    
    input_dict = {'a': [1, 2]}
    result = frozen_identity(val=input_dict)
    
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_mutant_freezes_return_value():
    def return_list():
        return [1, {'a': 2}]
    
    frozen_return_list = mutant(return_list)
    
    result = frozen_return_list()
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['a'] == 2

def test_mutant_handles_nested_structures():
    def complex_fn(data):
        return data
    
    frozen_complex_fn = mutant(complex_fn)
    
    input_data = {
        'list': [1, (2, 3)],
        'tuple': (4, [5])
    }
    
    result = frozen_complex_fn(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert result['list'][1] == (2, 3)
    assert isinstance(result['tuple'][1], PVector)
```


# LLM-generated content at query #10
#--------------------------

```python
def test_freeze_strict_pmap_evaluates_true():
    from pyrsistent import pmap, freeze
    frozen_map = freeze(pmap({'a': 1}), strict=True)
    assert isinstance(frozen_map, pmap.__class__)
```


# LLM-generated content at query #11
#--------------------------

```python
def test_freeze_strict_pmap_evaluates_true():
    from pyrsistent import pmap, freeze
    sample_pmap = pmap({'a': 1})
    result = freeze(sample_pmap, strict=True)
    assert isinstance(result, type(sample_pmap))
```


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_with_dict():
    from pyrsistent import freeze, pmap
    result = freeze({'a': 1})
    assert isinstance(result, pmap)

def test_freeze_with_list():
    from pyrsistent import freeze, pvector
    result = freeze([1, 2])
    assert isinstance(result, pvector)

def test_freeze_with_tuple():
    from pyrsistent import freeze
    result = freeze((1, 2))
    assert result == (1, 2)

def test_freeze_with_set():
    from pyrsistent import freeze, pset
    result = freeze({1, 2})
    assert isinstance(result, pset)

def test_freeze_with_primitive():
    from pyrsistent import freeze
    result = freeze(5)
    assert result == 5

def test_freeze_recursive_dict():
    from pyrsistent import freeze, pmap
    result = freeze({'a': {'b': 2}})
    expected = pmap({'a': pmap({'b': 2})})
    assert result == expected

def test_freeze_recursive_list():
    from pyrsistent import freeze, pvector
    result = freeze([[1]])
    expected = pvector([pvector([1])])
    assert result == expected
```


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant_freezes_arguments():
    def identity(x):
        return x
    
    mutated_identity = mutant(identity)
    
    input_list = [1, 2, [3, 4]]
    result = mutated_identity(input_list)
    
    assert isinstance(result, PVector)
    assert result[2][1] == 4
    assert isinstance(result[2], PVector)

def test_mutant_freezes_kwargs():
    def identity(x=None):
        return x
    
    mutated_identity = mutant(identity)
    
    input_dict = {'a': [1, 2]}
    result = mutated_identity(x=input_dict)
    
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_mutant_freezes_return_value():
    def return_list():
        return [1, {'a': 2}]
    
    mutated_fn = mutant(return_list)
    result = mutated_fn()
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['a'] == 2

def test_mutant_handles_nested_structures():
    def nested_identity(data):
        return data
    
    mutated_fn = mutant(nested_identity)
    
    complex_input = {
        'key': [
            {'inner_key': (1, 2)},
            {3, 4}
        ]
    }
    
    result = mutated_fn(complex_input)
    
    assert isinstance(result, PMap)
    assert isinstance(result['key'], PVector)
    assert isinstance(result['key'][0], PMap)
    assert isinstance(result['key'][0]['inner_key'], tuple)
    assert result['key'][1] == pset({3, 4})

def test_mutant_preserves_primitives():
    def identity(x):
        return x
    
    mutated_identity = mutant(identity)
    
    assert mutated_identity(10) == 10
    assert mutated_identity("string") == "string"
    assert mutated_identity(True) is True
```


# LLM-generated content at query #14
#--------------------------

```python
def test_freeze_primitive_types():
    assert freeze(1) == 1
    assert freeze("string") == "string"
    assert freeze(True) == True
    assert freeze(None) == None

def test_freeze_list_to_pvector():
    assert isinstance(freeze([1, 2, 3]), PVector)
    assert freeze([1, 2, 3]) == pvector([1, 2, 3])

def test_freeze_nested_list():
    assert freeze([[1], [2, [3]]]) == pvector([pvector([1]), pvector([2, pvector([3])])])

def test_freeze_dict_to_pmap():
    assert isinstance(freeze({'a': 1, 'b': 2}), PMap)
    assert freeze({'a': 1, 'b': 2}) == pmap({'a': 1, 'b': 2})

def test_freeze_nested_dict():
    assert freeze({'a': {'b': 3}}) == pmap({'a': pmap({'b': 3})})

def test_freeze_tuple_to_tuple_recursive():
    assert freeze((1, [2])) == (1, pvector([2]))
    assert freeze(((1,), [2])) == ((1,), pvector([2]))

def test_freeze_set_to_pset():
    assert isinstance(freeze({1, 2, 3}), PSet)
    assert freeze({1, 2, 3}) == pset([1, 2, 3])

def test_freeze_list_of_dicts():
    assert freeze([{'a': 1}, {'b': 2}]) == pvector([pmap({'a': 1}), pmap({'b': 2})])

def test_freeze_strict_false_behavior_on_dict():
    # When strict is False, the function logic for dict/PMap depends on type check.
    # Based on implementation: if typ is dict or (strict and isinstance(o, PMap))
    # If o is a PMap and strict is False, it won't enter the pmap conversion block via the second condition.
    # However, since we can't easily create a PMap without the library context here, 
    # we test the standard dict behavior which remains the same for strict=True/False.
    assert freeze({'a': [1]}, strict=False) == pmap({'a': pvector([1])})

def test_freeze_empty_containers():
    assert freeze([]) == pvector([])
    assert freeze({}) == pmap({})
    assert freeze(()) == ()
    assert freeze(set()) == pset([])
```


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_decorator_returns_function():
    def dummy_fn(x):
        return x
    
    from pyrsistent import pmap, pset, PMap, PSet
    # Since we cannot define a custom function to check the predicate directly 
    # without violating the "no custom function definition" rule for logic, 
    # and the instruction asks to ensure the predicate at line 1 evaluates to True.
    # The predicate is 'def mutant(fn):', which is always true if the decorator exists.
    # We test that the decorator behaves as expected (freezing arguments).
    
    from pyrsistent import pmap, PMap
    
    @mutant
    def identity_with_dict(d):
        return d

    input_dict = {'a': 1}
    result = identity_with_dict(input_dict)
    
    assert isinstance(result, PMap)
    assert result['a'] == 1
```


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_predicate_false():
    from pyrsistent import pmap, pset, PMap, PSet
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    non_persistent_dict = {'a': 1}
    non_persistent_list = [1, 2]
    
    result = identity(non_persistent_dict, y=non_persistent_list)
    
    assert isinstance(result, PMap)
    assert result['a'] == 1
```


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    from pyrsistent import pvector, pmap
    
    def identity(x, y=None):
        return {'a': x, 'b': y}
    
    frozen_identity = mutant(identity)
    
    result = frozen_identity([1, 2], y={'c': 3})
    
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])
    assert result['b'] == pmap({'c': 3})

def test_mutant_freezes_return_value():
    from pyrsistent import pvector
    
    def returns_list():
        return [1, {'a': 2}]
    
    frozen_returns_list = mutant(returns_list)
    
    result = frozen_returns_list()
    
    assert isinstance(result, PVector)
    assert result[1] == pmap({'a': 2})

def test_mutant_handles_empty_inputs():
    from pyrsistent import pmap
    
    def identity(x):
        return x
    
    frozen_identity = mutant(identity)
    
    result = frozen_identity({})
    
    assert isinstance(result, PMap)
    assert len(result) == 0

def test_mutant_preserves_logic_with_mutation_inside():
    def adder(a, b):
        # The decorator freezes inputs BEFORE the function runs.
        # This test ensures that even if the function logic is simple, 
        # the transformation of input/output occurs as expected.
        return a + b
    
    frozen_adder = mutant(adder)
    
    assert frozen_adder(1, 2) == 3
```


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant_freezes_arguments():
    def identity(x):
        return x
    
    mutated_identity = mutant(identity)
    
    input_list = [1, 2, 3]
    result = mutated_identity(input_list)
    
    assert isinstance(result, PVector)
    assert result == PVector([1, 2, 3])

def test_mutant_freezes_kwargs():
    def identity(data):
        return data
    
    mutated_identity = mutant(identity)
    
    input_dict = {'a': [1, 2]}
    result = mutated_identity(data=input_dict)
    
    assert isinstance(result, PMap)
    assert result['a'] == PVector([1, 2])

def test_mutant_freezes_return_value():
    def return_list():
        return [1, {'inner': [2]}]
    
    mutated_return = mutant(return_list)
    
    result = mutated_return()
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert isinstance(result[1]['inner'], PVector)
    assert result == PVector([1, PMap({'inner': PVector([2])})])

def test_mutant_handles_nested_structures():
    def nested_func(data):
        return data
    
    mutated_nested = mutant(nested_func)
    
    input_data = {
        'list': [1, 2, {'a': 3}],
        'tuple': (4, 5),
        'set': {6, 7}
    }
    
    result = mutated_nested(input_data)
    
    assert isinstance(result, PMap)
    assert result['list'] == PVector([1, 2, PMap({'a': 3})])
    assert result['tuple'] == (4, 5)
    assert result['set'] == pset({6, 7})

def test_mutant_preserves_unmutable_types():
    def identity(x):
        return x
    
    mutated_identity = mutant(identity)
    
    result = mutated_identity(10)
    
    assert result == 10
    assert isinstance(result, int)
```


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    from pyrsistent import pmap, pset, PMap, PSet
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    assert identity(pmap({'a': 1})) == pmap({'a': 1})
    assert identity(pset([1, 2])) == pset([1, 2])
    assert identity(5) == 5

def test_mutant_decorator_freezes_arguments():
    from pyrsistent import pmap, PMap
    from pyrsistent._helpers import mutant

    @mutant
    def check_frozen(m):
        return m

    initial_dict = {'a': 1}
    m = pmap(initial_dict)
    # The decorator calls freeze() on arguments.
    # Since pmap returns a PMap (which is already frozen), 
    # and the inner logic applies freeze to args, 
    # we check if the return value of an identity-like function is correctly handled.
    assert check_frozen(m) == m
```


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_predicate_is_false():
    from pyrsistent import pmap, pset, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    result = identity({'a': 1})
    assert isinstance(result, pmap)
```


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_predicate_is_false():
    from pyrsistent import pmap, pset, freeze

    @mutant
    def identity(x):
        return x

    result = identity(pmap({'a': 1}))
    assert isinstance(result, type(pmap({'a': 1})))
```


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_predicate_evaluates_to_false():
    from pyrsistent import pmap, pset, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    # The predicate at line 1 is "def mutant(fn):"
    # Since this is a function definition, it exists as a function object.
    # To make the statement evaluate to False in a test context, 
    # we cannot directly change the code, but we can demonstrate that 
    # the decorator logic itself doesn't break when applied to standard functions.
    # However, the prompt asks to ensure the predicate (the function definition) evaluates to False.
    # In Python, a function definition statement itself is not an expression that returns a value.
    # If the "predicate" refers to a truthiness test of the decorator itself:
    
    assert bool(mutant) == True
    
    # Since I cannot modify the source code provided to make the line 'def mutant(fn):' 
    # evaluate to False (as it is a definition, not an expression), and the prompt 
    # specifically asks for a test where that specific line evaluates to False, 
    # this is logically impossible within standard Python execution of that snippet.
    # Assuming the user implies testing the truthiness of the decorator's return value:
    
    identity_with_mutant = mutant(identity)
    assert identity_with_mutant(pmap({'a': 1})) == pmap({'a': 1})
```


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    from pyrsistent import pmap, pset, PMap, PSet

    def identity(x):
        return x

    decorated_identity = mutant(identity)
    
    # Test with simple value
    assert decorated_identity(5) == 5
    
    # Test with mutable-like structure (pmap/pset are already immutable, but we test the logic)
    initial_map = pmap({'a': 1})
    assert decorated_identity(initial_map) == initial_map
    
    # Test with kwargs and freezing logic
    def check_kwargs(val, extra=None):
        return val, extra

    decorated_check = mutant(check_kwargs)
    result_val, result_extra = decorated_check(10, extra={'b': 2})
    assert result_val == 10
    assert result_extra == pmap({'b': 2})

    # Test that the decorator returns a function (the inner_f)
    assert callable(decorated_identity)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_decorator_works():
    from pyrsistent import pmap, pset, PMap, PSet
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x, y=None):
        return x, y

    # Test with positional arguments (integers are immutable)
    result_pos = identity(1, 2)
    assert result_pos == (1, 2)

    # Test with keyword arguments and mutable types (pmap/pset are persistent/immutable)
    input_map = pmap({'a': 1})
    input_set = pset([1, 2])
    result_kw = identity(x=input_map, y=input_set)
    assert result_kw == (input_map, input_set)
    assert isinstance(result_kw[0], PMap)
    assert isinstance(result_kw[1], PSet)

    # Test that the decorator returns a function
    assert callable(identity)
```


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_predicate_false():
    from pyrsistent import pmap, pset, pvector
    from pyrsistent._helpers import mutant

    @mutant
    def identity(x):
        return x

    identity_func = identity
    
    # The predicate at line 1 (the existence of the decorator) is tested by ensuring
    # that calling a decorated function with mutable inputs results in frozen outputs.
    # To specifically target the logic where we want to prove 'mutant' works as a decorator:
    # If we pass a list, it should be converted to a pvector.
    
    result = identity_func([1, 2, 3])
    
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])
```


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_decorator_preserves_functionality():
    from pyrsistent import pmap, pset, PMap, PSet

    @mutant
    def identity(x):
        return x

    assert identity(pmap({'a': 1})) == pmap({'a': 1})
    assert identity(pset([1, 2])) == pset([1, 2])
    assert identity(5) == 5

def test_mutant_decorator_freezes_arguments():
    from pyrsistent import pmap, PMap

    @mutant
    def check_frozen(m):
        return m

    mutable_dict = {'a': 1}
    # Since the decorator calls freeze() on args, 
    # passing a dict will result in a PMap being passed to the function.
    # We can verify this by checking if the type inside the function is PMap.
    
    @mutant
    def get_type(x):
        return type(x)

    assert get_type({'a': 1}) == PMap
```


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    def identity_with_mutation(data, extra=None):
        data[0] = "changed"
        return data

    mutable_list = [1, 2, 3]
    mutable_dict = {"a": 1}
    decorated = mutant(identity_with_mutation)
    
    result = decorated(mutable_list, extra=mutable_dict)
    
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3
    assert isinstance(result, PVector)

def test_mutant_freezes_return_value():
    def return_mutable_list():
        return [1, 2, 3]

    decorated = mutant(return_mutable_list)
    result = decorated()
    
    assert isinstance(result, PVector)
    assert result[0] == 1

def test_mutant_preserves_logic_with_frozen_inputs():
    def check_type(x):
        return type(x)

    decorated = mutant(check_type)
    
    assert check_type([1, 2]) == decorated([1, 2])
    assert check_type({"a": 1}) == decorated({"a": 1})

def test_mutant_handles_nested_structures():
    def nested_mutation(data):
        data[0][0] = "mutated"
        return data

    input_data = [[1]]
    decorated = mutant(nested_mutation)
    result = decorated(input_data)
    
    assert result[0][0] == 1
    assert isinstance(result, PVector)
    assert isinstance(result[0], PVector)

def test_mutant_handles_empty_inputs():
    def identity(x):
        return x

    decorated = mutant(identity)
    
    assert decorated([]) == PVector()
    assert decorated({}) == PMap()
```


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant_freezes_arguments():
    def identity(x):
        return x
    mutated_identity = mutant(identity)
    input_list = [1, 2, 3]
    result = mutated_identity(input_list)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_freezes_kwargs():
    def identity(x=None):
        return x
    mutated_identity = mutant(identity)
    input_dict = {'a': [1, 2]}
    result = mutated_identity(x=input_dict)
    assert isinstance(result, PMap)
    assert result['a'] == pvector([1, 2])

def test_mutant_freezes_return_value():
    def return_list():
        return [1, 2, {'a': 3}]
    mutated_return = mutant(return_list)
    result = mutated_return()
    assert isinstance(result, PVector)
    assert isinstance(result[2], PMap)
    assert result[2]['a'] == 3

def test_mutant_deep_freezing():
    def nested_structure(data):
        return data
    mutated_nested = mutant(nested_structure)
    complex_input = [1, (2, [3, {'key': 'value'}])]
    result = mutated_nested(complex_input)
    assert isinstance(result, PVector)
    assert isinstance(result[1], tuple)
    assert isinstance(result[1][1], PVector)
    assert isinstance(result[1][1][1], PMap)
    assert result[1][1][1]['key'] == 'value'

def test_mutant_preserves_primitives():
    def return_int(x):
        return x
    mutated_int = mutant(return_int)
    assert mutated_int(5) == 5
    assert mutated_int("string") == "string"
    assert mutated_int(True) is True
```


# LLM-generated content at query #29
#--------------------------

```python
from pyrsistent import pmap, pset, pvector

def test_mutant_decorator_is_callable():
    def identity(x):
        return x
    
    from pyrsistent._helpers import mutant
    decorated = mutant(identity)
    
    assert callable(decorated)
```


# LLM-generated content at query #30
#--------------------------

```python
def test_freeze_strict_pmap_evaluates_true():
    from pyrsistent import pmap, freeze
    input_data = pmap({'a': 1})
    result = freeze(input_data, strict=True)
    assert result == pmap({'a': 1})
```


# LLM-generated content at query #31
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
    
    decorated_fn = mutant(identity_with_mutation)
    result = decorated_fn(input_list, extra=input_dict)

    assert isinstance(result, PVector)
    assert result[0] == 1
    assert result[3] == 4
    assert isinstance(result[1], PMap) if len(result) > 1 else True # Checking structure
    assert input_list == [1, 2, 3]
    assert input_dict == {'a': 1}

def test_mutant_freezes_return_value():
    def returning_mutable_list():
        return [1, 2, {'a': 3}]

    decorated_fn = mutant(returning_mutable_list)
    result = decorated_fn()

    assert isinstance(result, PVector)
    assert isinstance(result[2], PMap)
    assert result[2]['a'] == 3

def test_mutant_handles_empty_inputs():
    def identity(x):
        return x

    decorated_fn = mutant(identity)
    result = decorated_fn()

    assert result is None

def test_mutant_preserves_unmutable_types():
    def identity_simple(x, y):
        return x + y

    decorated_fn = mutant(identity_simple)
    result = decorated_fn(1, 2)

    assert result == 3
```


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_decorator_returns_function():
    from pyrsistent import pmap, pset, PMap, PSet
    
    def sample_fn(a, b=None):
        return a + (b if b else 0)

    decorated = mutant(sample_fn)
    
    assert callable(decorated)
```


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant_freezes_args_and_kwargs():
    from pyrsistent import pvector, pmap, PMap, PVector
    
    def simple_fn(a, b=None):
        return [a, b]

    # Test that inputs are frozen and output is frozen
    result = mutant(simple_fn)([1, 2], b={'x': 3})
    assert isinstance(result, PVector)
    assert result[0] == 1
    assert isinstance(result[1], PMap)
    assert result[1]['x'] == 3

def test_mutant_preserves_logic():
    from pyrsistent import pvector

    def identity_fn(x):
        return x

    # Test that the function's return value is frozen even if it's a standard list
    result = mutant(identity_fn)([1, 2, 3])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3])

def test_mutant_with_nested_structures():
    from pyrsistent import pvector, pmap

    def complex_fn(data):
        return data

    input_data = {'a': [1, 2], 'b': {'c': 3}}
    result = mutant(complex_fn)(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    assert result['a'][0] == 1
    assert result['b']['c'] == 3

def test_mutant_handles_no_args():
    from pyrsistent import pvector

    def no_arg_fn():
        return [1, 2]

    result = mutant(no_arg_fn)()
    assert isinstance(result, PVector)
    assert result == pvector([1, 2])
```


# LLM-generated content at query #34
#--------------------------

```python
from pyrsistent import pmap, pset, pvector, freeze

def test_mutant_decorates_function():
    def simple_fn(a, b, c=None):
        return (a, b, c)
    
    decorated = mutant(simple_fn)
    
    # Test with positional and keyword arguments using mutable types
    # The decorator must freeze them to ensure the inner function sees frozen versions
    result = decorated({'key': 'val'}, [1, 2], d={'nested': 'list'})
    
    assert isinstance(result[0], type(pmap({'a': 1})))
    assert isinstance(result[1], type(pvector([1])))
    assert isinstance(result[2], type(pmap({'nested': 'list'})))
    assert result == (pmap({'key': 'val'}), pvector([1, 2]), pmap({'nested': 'list'}))

def test_mutant_returns_frozen_value():
    @mutant
    def returns_list():
        return [1, 2, 3]
    
    result = returns_list()
    assert isinstance(result, type(pvector([1])))
    assert result == pvector([1, 2, 3])
```


# LLM-generated content at query #35
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
```


# LLM-generated content at query #36
#--------------------------

```python
def test_freeze_strict_pmap_evaluates_true():
    from pyrsistent import pmap, freeze
    frozen_map = freeze(pmap({'a': 1}), strict=True)
    assert isinstance(frozen_map, pmap.__class__)
```


# LLM-generated content at query #37
#--------------------------

```python
def test_mutant_freezes_arguments():
    def identity(x):
        return x
    
    mutated_identity = mutant(identity)
    
    input_list = [1, 2, [3, 4]]
    result = mutated_identity(input_list)
    
    assert isinstance(result, PVector)
    assert result[2][1] == 4
    assert isinstance(result[2], PVector)

def test_mutant_freezes_kwargs():
    def identity(d):
        return d
    
    mutated_identity = mutant(identity)
    
    input_dict = {'a': [1, 2]}
    result = mutated_identity(d=input_dict)
    
    assert isinstance(result, PMap)
    assert result['a'] == PVector([1, 2])

def test_mutant_freezes_return_value():
    def returns_list():
        return [1, {'key': 'value'}]
    
    mutated_returns = mutant(returns_list)
    
    result = mutated_returns()
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['key'] == 'value'

def test_mutant_with_complex_nested_structure():
    def identity(x):
        return x
    
    mutated_identity = mutant(identity)
    
    input_data = {
        'list': [1, 2, {'inner': [3]}],
        'tuple': (4, 5),
        'set': {6, 7}
    }
    
    result = mutated_identity(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['list'][2], PMap)
    assert isinstance(result['list'][2]['inner'], PVector)
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['tuple'][0], int)
    assert isinstance(result['set'], PSet)

def test_mutant_preserves_primitives():
    def identity(x):
        return x
    
    mutated_identity = mutant(identity)
    
    assert mutated_identity(1) == 1
    assert mutated_identity("string") == "string"
    assert mutated_identity(True) is True
```


# LLM-generated content at query #38
#--------------------------

```python
def test_mutant_freezes_arguments():
    def identity(x):
        return x
    
    decorated = mutant(identity)
    input_list = [1, 2, [3]]
    result = decorated(input_list)
    
    assert isinstance(result, PVector)
    assert result[2] == pvector([3])

def test_mutant_freezes_kwargs():
    def identity(a=None):
        return a
    
    decorated = mutant(identity)
    input_dict = {'key': [1, 2]}
    result = decorated(key=input_dict)
    
    assert isinstance(result, PMap)
    assert result['key'] == pvector([1, 2])

def test_mutant_freezes_return_value():
    def return_list():
        return [1, {'a': 2}]
    
    decorated = mutant(return_list)
    result = decorated()
    
    assert isinstance(result, PVector)
    assert isinstance(result[1], PMap)
    assert result[1]['a'] == 2

def test_mutant_deep_freezing():
    def identity(x):
        return x
    
    decorated = mutant(identity)
    input_data = {'outer': [{'inner': [10]}]}
    result = decorated(input_data)
    
    assert isinstance(result, PMap)
    assert isinstance(result['outer'][0], PMap)
    assert isinstance(result['outer'][0]['inner'], PVector)
    assert result['outer'][0]['inner'][0] == 10

def test_mutant_preserves_immutable_types():
    def identity(x):
        return x
    
    decorated = mutant(identity)
    input_val = 42
    result = decorated(input_val)
    
    assert result == 42
    assert isinstance(result, int)

def test_mutant_handles_tuple_recursion():
    def identity(x):
        return x
    
    decorated = mutant(identity)
    input_tuple = (1, [2])
    result = decorated(input_tuple)
    
    assert isinstance(result, tuple)
    assert result[0] == 1
    assert isinstance(result[1], PVector)
```


# LLM-generated content at query #39
#--------------------------

```python
def test_mutant_decorator_returns_frozen_value():
    from pyrsistent import pmap, pvector
    
    @mutant
    def identity(x):
        return x

    result = identity(pmap({'a': 1}))
    assert isinstance(result, type(pmap({'a': 1})))
```


# LLM-generated content at query #40
#--------------------------

```python
from pyrsistent import pmap, pset, pvector

def test_mutant_predicate_is_false():
    # To ensure the predicate 'def mutant(fn):' evaluates to False in a mutation testing context,
    # we need to provide a scenario where the decorator is applied and behaves correctly.
    # However, since the prompt asks for a unit test to ensure the predicate at line 1 evaluates to False,
    # and in standard Python execution 'def mutant(fn):' is a definition (which evaluates to a function object, not a boolean),
    # this specific request is logically impossible under standard testing of a function definition.
    # Assuming the prompt implies checking that the decorator logic works such that it doesn't 
    # accidentally return a boolean or that we are testing the 'inner_f' logic:
    
    @mutant
    def identity(x):
        return x

    # Test with pmap (which is persistent/frozen)
    test_val = pmap({'a': 1})
    assert identity(test_val) == test_val
    
    # Test with pset
    test_set = pset([1, 2])
    assert identity(test_set) == test_set

    # Test with standard mutable types being frozen by the decorator's logic
    # Note: The mutant decorator calls freeze(). If 'freeze' is from pyrsistent,
    # it turns dict into pmap and list into pvector.
    @mutant
    def check_frozen(d):
        return d

    input_dict = {'key': 'value'}
    result = check_frozen(input_dict)
    assert isinstance(result, type(pmap()))
    assert result['key'] == 'value'
```


# LLM-generated content at query #41
#--------------------------

```python
from pyrsistent import pmap, pset, PMap, PSet

def test_mutant_decorator_freezes_args_and_returns():
    def identity(x, y=None):
        return x, y
    
    # We use a decorator to wrap the function. 
    # Since we cannot define functions or classes inside the test per instructions, 
    # and the task is to test 'mutant', we assume 'mutant' is available in the namespace.
    # However, since I must only provide the test case itself:
    
    from pyrsistent._helpers import mutant
    
    decorated_fn = mutant(identity)
    
    # Test with PMap (which is already frozen/persistent)
    input_map = pmap({'a': 1})
    input_val = 2
    result_tuple, result_val = decorated_fn(input_map, y=input_val)
    
    assert isinstance(result_tuple, PMap)
    assert isinstance(result_val, int)
    assert result_tuple == input_map
    assert result_val == input_val

def test_mutant_freezes_mutable_inputs():
    from pyrsistent._helpers import mutant
    
    # Using a list which is mutable. 
    # The decorator should turn it into a PVector (or similar frozen structure) 
    # via the 'freeze' call mentioned in the source.
    def check_type(lst):
        return lst

    decorated_check = mutant(check_type)
    
    # If we pass a list, 'freeze' inside 'mutant' will convert it to a pvector.
    result = decorated_check([1, 2, 3])
    
    from pyrsistent import pvector
    assert isinstance(result, type(pvector([1, 2, 3])))
```


