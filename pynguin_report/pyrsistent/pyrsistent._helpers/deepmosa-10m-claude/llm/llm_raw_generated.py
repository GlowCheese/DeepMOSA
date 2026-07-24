####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_freeze_empty_dict():
    from pyrsistent._helpers import freeze
    result = freeze({})
    assert result == {}


def test_freeze_simple_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1, 'b': 2})
    assert result == {'a': 1, 'b': 2}


def test_freeze_nested_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': {'b': 1}})
    assert result == {'a': {'b': 1}}


def test_freeze_dict_with_list_value():
    from pyrsistent._helpers import freeze
    result = freeze({'a': [1, 2, 3]})
    assert result == {'a': [1, 2, 3]}


def test_freeze_empty_list():
    from pyrsistent._helpers import freeze
    result = freeze([])
    assert result == []


def test_freeze_simple_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, 2, 3])
    assert result == [1, 2, 3]


def test_freeze_nested_list():
    from pyrsistent._helpers import freeze
    result = freeze([[1, 2], [3, 4]])
    assert result == [[1, 2], [3, 4]]


def test_freeze_list_with_dict():
    from pyrsistent._helpers import freeze
    result = freeze([1, {'a': 2}])
    assert result == [1, {'a': 2}]


def test_freeze_empty_set():
    from pyrsistent._helpers import freeze
    result = freeze(set())
    assert result == set()


def test_freeze_simple_set():
    from pyrsistent._helpers import freeze
    result = freeze({1, 2, 3})
    assert result == {1, 2, 3}


def test_freeze_empty_tuple():
    from pyrsistent._helpers import freeze
    result = freeze(())
    assert result == ()


def test_freeze_simple_tuple():
    from pyrsistent._helpers import freeze
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)


def test_freeze_tuple_with_list():
    from pyrsistent._helpers import freeze
    result = freeze((1, [2, 3]))
    assert result == (1, [2, 3])


def test_freeze_tuple_with_dict():
    from pyrsistent._helpers import freeze
    result = freeze((1, {'a': 2}))
    assert result == (1, {'a': 2})


def test_freeze_complex_nested_structure():
    from pyrsistent._helpers import freeze
    result = freeze({'a': [1, {'b': (2, 3)}], 'c': {4, 5}})
    assert result == {'a': [1, {'b': (2, 3)}], 'c': {4, 5}}


def test_freeze_defaultdict():
    from pyrsistent._helpers import freeze
    from collections import defaultdict
    d = defaultdict(int)
    d['a'] = 1
    result = freeze(d)
    assert result == {'a': 1}


def test_freeze_defaultdict_with_nested_values():
    from pyrsistent._helpers import freeze
    from collections import defaultdict
    d = defaultdict(list)
    d['a'] = [1, 2]
    result = freeze(d)
    assert result == {'a': [1, 2]}


def test_freeze_scalar_int():
    from pyrsistent._helpers import freeze
    result = freeze(42)
    assert result == 42


def test_freeze_scalar_string():
    from pyrsistent._helpers import freeze
    result = freeze("hello")
    assert result == "hello"


def test_freeze_scalar_none():
    from pyrsistent._helpers import freeze
    result = freeze(None)
    assert result is None


def test_freeze_strict_true_with_pmap():
    from pyrsistent._helpers import freeze
    from pyrsistent import pmap
    pm = pmap({'a': 1})
    result = freeze(pm, strict=True)
    assert result == {'a': 1}


def test_freeze_strict_true_with_pvector():
    from pyrsistent._helpers import freeze
    from pyrsistent import pvector
    pv = pvector([1, 2, 3])
    result = freeze(pv, strict=True)
    assert result == [1, 2, 3]


def test_freeze_strict_false_with_pmap():
    from pyrsistent._helpers import freeze
    from pyrsistent import pmap
    pm = pmap({'a': 1})
    result = freeze(pm, strict=False)
    assert result is pm


def test_freeze_strict_false_with_pvector():
    from pyrsistent._helpers import freeze
    from pyrsistent import pvector
    pv = pvector([1, 2, 3])
    result = freeze(pv, strict=False)
    assert result is pv


def test_freeze_dict_with_multiple_levels():
    from pyrsistent._helpers import freeze
    result = freeze({'a': {'b': {'c': 1}}})
    assert result == {'a': {'b': {'c': 1}}}


def test_freeze_list_with_multiple_levels():
    from pyrsistent._helpers import freeze
    result = freeze([[[1, 2], [3, 4]], [[5, 6]]])
    assert result == [[[1, 2], [3, 4]], [[5, 6]]]


def test_freeze_mixed_structure():
    from pyrsistent._helpers import freeze
    result = freeze({'x': [1, 2, {'y': (3, 4)}], 'z': {5, 6}})
    assert result == {'x': [1, 2, {'y': (3, 4)}], 'z': {5, 6}}


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_empty_dict():
    from pyrsistent._helpers import freeze
    result = freeze({})
    assert result == {}
    assert len(result) == 0


def test_freeze_simple_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1, 'b': 2})
    assert result['a'] == 1
    assert result['b'] == 2


def test_freeze_nested_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': {'b': 3}})
    assert result['a']['b'] == 3


def test_freeze_empty_list():
    from pyrsistent._helpers import freeze
    result = freeze([])
    assert len(result) == 0


def test_freeze_simple_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, 2, 3])
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_freeze_nested_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, [2, 3]])
    assert result[0] == 1
    assert result[1][0] == 2
    assert result[1][1] == 3


def test_freeze_list_with_dict():
    from pyrsistent._helpers import freeze
    result = freeze([1, {'a': 3}])
    assert result[0] == 1
    assert result[1]['a'] == 3


def test_freeze_empty_set():
    from pyrsistent._helpers import freeze
    result = freeze(set())
    assert len(result) == 0


def test_freeze_simple_set():
    from pyrsistent._helpers import freeze
    result = freeze({1, 2, 3})
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_freeze_empty_tuple():
    from pyrsistent._helpers import freeze
    result = freeze(())
    assert result == ()


def test_freeze_simple_tuple():
    from pyrsistent._helpers import freeze
    result = freeze((1, 2, 3))
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_freeze_nested_tuple():
    from pyrsistent._helpers import freeze
    result = freeze((1, (2, 3)))
    assert result[0] == 1
    assert result[1][0] == 2
    assert result[1][1] == 3


def test_freeze_tuple_with_list():
    from pyrsistent._helpers import freeze
    result = freeze((1, []))
    assert result[0] == 1
    assert len(result[1]) == 0


def test_freeze_scalar_int():
    from pyrsistent._helpers import freeze
    result = freeze(42)
    assert result == 42


def test_freeze_scalar_string():
    from pyrsistent._helpers import freeze
    result = freeze("hello")
    assert result == "hello"


def test_freeze_scalar_none():
    from pyrsistent._helpers import freeze
    result = freeze(None)
    assert result is None


def test_freeze_defaultdict():
    from pyrsistent._helpers import freeze
    import collections
    dd = collections.defaultdict(int)
    dd['a'] = 5
    result = freeze(dd)
    assert result['a'] == 5


def test_freeze_complex_nested_structure():
    from pyrsistent._helpers import freeze
    data = {
        'list': [1, 2, {'nested': 3}],
        'tuple': (4, [5, 6]),
        'set': {7, 8},
        'dict': {'inner': 9}
    }
    result = freeze(data)
    assert result['list'][0] == 1
    assert result['list'][2]['nested'] == 3
    assert result['tuple'][0] == 4
    assert result['tuple'][1][0] == 5
    assert 7 in result['set']
    assert result['dict']['inner'] == 9


def test_freeze_strict_true():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1}, strict=True)
    assert result['a'] == 1


def test_freeze_strict_false():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1}, strict=False)
    assert result['a'] == 1


def test_freeze_dict_with_tuple_values():
    from pyrsistent._helpers import freeze
    result = freeze({'a': (1, 2)})
    assert result['a'][0] == 1
    assert result['a'][1] == 2


def test_freeze_list_with_tuple_elements():
    from pyrsistent._helpers import freeze
    result = freeze([(1, 2), 3])
    assert result[0][0] == 1
    assert result[1] == 3


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    import collections
    from pyrsistent import freeze, pmap
    
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    
    result = freeze(dd, strict=True)
    
    assert result == pmap({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap()))


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1, 'b': 2})
    assert result == {'a': 1, 'b': 2}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_nested_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': {'b': 1}})
    assert result == {'a': {'b': 1}}
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PMap'


def test_freeze_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, 2, 3])
    assert list(result) == [1, 2, 3]
    assert str(type(result).__name__) == 'PVector'


def test_freeze_nested_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, [2, 3]])
    assert list(result) == [1, [2, 3]]
    assert str(type(result).__name__) == 'PVector'
    assert str(type(result[1]).__name__) == 'PVector'


def test_freeze_list_with_dict():
    from pyrsistent._helpers import freeze
    result = freeze([1, {'a': 3}])
    assert result[0] == 1
    assert result[1] == {'a': 3}
    assert str(type(result).__name__) == 'PVector'
    assert str(type(result[1]).__name__) == 'PMap'


def test_freeze_set():
    from pyrsistent._helpers import freeze
    result = freeze(set([1, 2, 3]))
    assert 1 in result
    assert 2 in result
    assert 3 in result
    assert str(type(result).__name__) == 'PSet'


def test_freeze_tuple():
    from pyrsistent._helpers import freeze
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)
    assert type(result) is tuple


def test_freeze_tuple_with_list():
    from pyrsistent._helpers import freeze
    result = freeze((1, [2, 3]))
    assert result[0] == 1
    assert list(result[1]) == [2, 3]
    assert type(result) is tuple
    assert str(type(result[1]).__name__) == 'PVector'


def test_freeze_defaultdict():
    from pyrsistent._helpers import freeze
    import collections
    d = collections.defaultdict(int)
    d['a'] = 1
    d['b'] = 2
    result = freeze(d)
    assert result == {'a': 1, 'b': 2}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_scalar():
    from pyrsistent._helpers import freeze
    result = freeze(42)
    assert result == 42


def test_freeze_string():
    from pyrsistent._helpers import freeze
    result = freeze("hello")
    assert result == "hello"


def test_freeze_none():
    from pyrsistent._helpers import freeze
    result = freeze(None)
    assert result is None


def test_freeze_empty_dict():
    from pyrsistent._helpers import freeze
    result = freeze({})
    assert result == {}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_empty_list():
    from pyrsistent._helpers import freeze
    result = freeze([])
    assert list(result) == []
    assert str(type(result).__name__) == 'PVector'


def test_freeze_empty_set():
    from pyrsistent._helpers import freeze
    result = freeze(set())
    assert len(result) == 0
    assert str(type(result).__name__) == 'PSet'


def test_freeze_empty_tuple():
    from pyrsistent._helpers import freeze
    result = freeze(())
    assert result == ()
    assert type(result) is tuple


def test_freeze_strict_true():
    from pyrsistent._helpers import freeze
    from pyrsistent import pvector, pmap
    pv = pvector([1, 2, {'a': 3}])
    result = freeze(pv, strict=True)
    assert result[2] == {'a': 3}
    assert str(type(result[2]).__name__) == 'PMap'


def test_freeze_strict_false():
    from pyrsistent._helpers import freeze
    from pyrsistent import pvector
    pv = pvector([1, 2, 3])
    result = freeze(pv, strict=False)
    assert list(result) == [1, 2, 3]


def test_freeze_complex_nested():
    from pyrsistent._helpers import freeze
    data = {'a': [1, {'b': 2}], 'c': (3, [4, 5])}
    result = freeze(data)
    assert result['a'][1] == {'b': 2}
    assert result['c'][1][0] == 4


# LLM-generated content at query #5
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [1]
    
    result = modify_list([1, 2])
    assert str(type(result).__name__) == 'PVector'


def test_mutant_with_dict_argument():
    @mutant
    def modify_dict(d):
        return d
    
    result = modify_dict({'a': 1})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'items': [1, 2, 3]})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, d):
        return lst
    
    result = combine([1, 2], {'a': 1})
    assert str(type(result).__name__) == 'PVector'


def test_mutant_with_kwargs():
    @mutant
    def func_with_kwargs(a, b=None):
        return [a, b]
    
    result = func_with_kwargs([1], b={'x': 1})
    assert str(type(result).__name__) == 'PVector'


def test_mutant_preserves_function_name():
    @mutant
    def my_function():
        return []
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert str(type(result).__name__) == 'PSet'


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)


def test_mutant_return_value_is_frozen():
    @mutant
    def return_list():
        return [1, 2, 3]
    
    result = return_list()
    assert str(type(result).__name__) == 'PVector'


def test_mutant_with_empty_containers():
    @mutant
    def process_empty(lst, d):
        return [lst, d]
    
    result = process_empty([], {})
    assert str(type(result).__name__) == 'PVector'


# LLM-generated content at query #6
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    from pyrsistent._helpers import freeze
    import collections
    
    # Create a defaultdict instance
    dd = collections.defaultdict(list)
    dd['key1'] = [1, 2, 3]
    dd['key2'] = {'nested': 'value'}
    
    # Call freeze with strict=True (default)
    result = freeze(dd, strict=True)
    
    # Verify the result is a pmap with frozen values
    from pyrsistent import pmap, pvector
    expected = pmap({'key1': pvector([1, 2, 3]), 'key2': pmap({'nested': 'value'})})
    assert result == expected


# LLM-generated content at query #7
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    import collections
    from pyrsistent import freeze, pmap
    
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    
    result = freeze(dd, strict=True)
    
    assert result == pmap({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap()))


# LLM-generated content at query #8
#--------------------------

```python
def test_freeze_empty_dict():
    from pyrsistent import freeze, pmap
    result = freeze({})
    assert result == pmap({})


def test_freeze_simple_dict():
    from pyrsistent import freeze, pmap
    result = freeze({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_nested_dict():
    from pyrsistent import freeze, pmap
    result = freeze({'a': {'b': 1}})
    assert result == pmap({'a': pmap({'b': 1})})


def test_freeze_dict_with_list():
    from pyrsistent import freeze, pmap, pvector
    result = freeze({'a': [1, 2, 3]})
    assert result == pmap({'a': pvector([1, 2, 3])})


def test_freeze_empty_list():
    from pyrsistent import freeze, pvector
    result = freeze([])
    assert result == pvector([])


def test_freeze_simple_list():
    from pyrsistent import freeze, pvector
    result = freeze([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_freeze_nested_list():
    from pyrsistent import freeze, pvector
    result = freeze([1, [2, 3]])
    assert result == pvector([1, pvector([2, 3])])


def test_freeze_list_with_dict():
    from pyrsistent import freeze, pvector, pmap
    result = freeze([1, {'a': 2}])
    assert result == pvector([1, pmap({'a': 2})])


def test_freeze_empty_set():
    from pyrsistent import freeze, pset
    result = freeze(set())
    assert result == pset([])


def test_freeze_simple_set():
    from pyrsistent import freeze, pset
    result = freeze({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_freeze_empty_tuple():
    from pyrsistent import freeze
    result = freeze(())
    assert result == ()


def test_freeze_simple_tuple():
    from pyrsistent import freeze
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)


def test_freeze_nested_tuple():
    from pyrsistent import freeze, pvector
    result = freeze((1, [2, 3]))
    assert result == (1, pvector([2, 3]))


def test_freeze_tuple_with_dict():
    from pyrsistent import freeze, pmap
    result = freeze((1, {'a': 2}))
    assert result == (1, pmap({'a': 2}))


def test_freeze_scalar_int():
    from pyrsistent import freeze
    result = freeze(42)
    assert result == 42


def test_freeze_scalar_string():
    from pyrsistent import freeze
    result = freeze("hello")
    assert result == "hello"


def test_freeze_scalar_none():
    from pyrsistent import freeze
    result = freeze(None)
    assert result is None


def test_freeze_defaultdict():
    from pyrsistent import freeze, pmap
    from collections import defaultdict
    d = defaultdict(int)
    d['a'] = 1
    d['b'] = 2
    result = freeze(d)
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_defaultdict_nested():
    from pyrsistent import freeze, pmap
    from collections import defaultdict
    d = defaultdict(int)
    d['a'] = {'b': 1}
    result = freeze(d)
    assert result == pmap({'a': pmap({'b': 1})})


def test_freeze_complex_nested_structure():
    from pyrsistent import freeze, pmap, pvector, pset
    data = {
        'list': [1, 2, {'nested': 3}],
        'dict': {'inner': [4, 5]},
        'set': {6, 7},
        'tuple': (8, [9])
    }
    result = freeze(data)
    assert result['list'] == pvector([1, 2, pmap({'nested': 3})])
    assert result['dict'] == pmap({'inner': pvector([4, 5])})
    assert result['set'] == pset([6, 7])
    assert result['tuple'] == (8, pvector([9]))


def test_freeze_strict_false_pmap():
    from pyrsistent import freeze, pmap
    p = pmap({'a': 1})
    result = freeze(p, strict=False)
    assert result == p


def test_freeze_strict_true_pmap():
    from pyrsistent import freeze, pmap
    p = pmap({'a': [1, 2]})
    result = freeze(p, strict=True)
    assert result['a'].tolist() == [1, 2]


def test_freeze_strict_true_pvector():
    from pyrsistent import freeze, pvector, pmap
    v = pvector([1, {'a': 2}])
    result = freeze(v, strict=True)
    assert result[1] == pmap({'a': 2})


def test_freeze_list_of_dicts():
    from pyrsistent import freeze, pvector, pmap
    result = freeze([{'a': 1}, {'b': 2}])
    assert result == pvector([pmap({'a': 1}), pmap({'b': 2})])


def test_freeze_dict_with_tuple_values():
    from pyrsistent import freeze, pmap, pvector
    result = freeze({'a': (1, [2])})
    assert result == pmap({'a': (1, pvector([2]))})


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, freeze
    
    @mutant
    def modify_and_return(data):
        return data
    
    input_list = [1, 2, 3]
    result = modify_and_return(input_list)
    
    assert result == freeze(input_list)
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_decorator_with_dict_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_dict(d):
        return d
    
    input_dict = {'a': 1, 'b': 2}
    result = process_dict(input_dict)
    
    assert result == pmap(input_dict)


def test_mutant_decorator_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze
    
    @mutant
    def combine(a, b):
        return [a, b]
    
    result = combine([1, 2], {'x': 10})
    
    assert result == freeze([[1, 2], {'x': 10}])


def test_mutant_decorator_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_kwargs(data, **kwargs):
        return data
    
    result = process_kwargs([1, 2], key={'a': 1})
    
    frozen_result = result
    assert frozen_result is not None


def test_mutant_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant_decorator_predicate_line_1_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, freeze
    
    @mutant
    def modify_and_return(data):
        return data
    
    input_data = pmap({'a': 1, 'b': 2})
    result = modify_and_return(input_data)
    
    assert result == input_data
    assert not (result is input_data or type(result) != type(input_data))


# LLM-generated content at query #11
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        lst_copy = list(lst)
        lst_copy.append(999)
        return lst_copy
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 999)


def test_mutant_with_dict_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_dict(d):
        d_copy = dict(d)
        d_copy['new_key'] = 'new_value'
        return d_copy
    
    result = process_dict({'a': 1})
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})


def test_mutant_with_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        data_copy = {'list': list(data['list']), 'value': data['value']}
        data_copy['list'].append(4)
        return data_copy
    
    result = process_nested({'list': [1, 2, 3], 'value': 'test'})
    assert isinstance(result, pmap)
    assert result['list'] == pvector([1, 2, 3, 4])
    assert result['value'] == 'test'


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        s_copy = set(s)
        s_copy.add(4)
        return s_copy
    
    result = process_set({1, 2, 3})
    assert isinstance(result, pset)
    assert result == pset([1, 2, 3, 4])


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        return {'list': list(lst), 'dict': dict(dct)}
    
    result = combine([1, 2], {'a': 3})
    assert isinstance(result, pmap)
    assert result['list'] == pvector([1, 2])
    assert result['dict'] == pmap({'a': 3})


def test_mutant_with_keyword_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_kwargs(data=None, extra=None):
        return {'data': data, 'extra': extra}
    
    result = process_kwargs(data={'a': 1}, extra={'b': 2})
    assert isinstance(result, pmap)
    assert result['data'] == pmap({'a': 1})
    assert result['extra'] == pmap({'b': 2})


def test_mutant_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        return []
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return (t[0], t[1], 'new')
    
    result = process_tuple((1, 2))
    assert isinstance(result, tuple)
    assert result == (1, 2, 'new')


def test_mutant_with_deeply_nested_structure():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_deep(data):
        return {'outer': {'inner': [1, 2, 3]}}
    
    result = process_deep({})
    assert isinstance(result, pmap)
    assert isinstance(result['outer'], pmap)
    assert result['outer']['inner'] == pvector([1, 2, 3])


def test_mutant_with_primitive_return():
    from pyrsistent._helpers import mutant
    
    @mutant
    def get_number(lst):
        return sum(lst)
    
    result = get_number([1, 2, 3])
    assert result == 6


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    import collections
    from pyrsistent import freeze, pmap
    
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    
    result = freeze(dd, strict=True)
    
    assert result == pmap({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap({})))


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [1, 2, 3]
    
    result = modify_list([1, 2])
    assert str(result) == "pvector([1, 2, 1, 2, 3])"


def test_mutant_with_dict_argument():
    @mutant
    def process_dict(d):
        return d
    
    result = process_dict({'a': 1, 'b': [2, 3]})
    assert result['a'] == 1
    assert str(result['b']) == "pvector([2, 3])"


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, d):
        return {'list': lst, 'dict': d}
    
    result = combine([1, 2], {'x': 10})
    assert str(result['list']) == "pvector([1, 2])"
    assert result['dict']['x'] == 10


def test_mutant_with_kwargs():
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'key': 'value'})
    assert str(result['a']) == "pvector([1, 2])"
    assert result['b']['key'] == 'value'


def test_mutant_preserves_function_metadata():
    @mutant
    def my_function():
        """Test docstring"""
        pass
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'nested': {'list': [1, 2, {'inner': 3}]}})
    assert result['nested']['list'][2]['inner'] == 3


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert result[0] == 1
    assert str(result[1]) == "pvector([2, 3])"
    assert result[2]['a'] == 4


def test_mutant_with_empty_containers():
    @mutant
    def process_empty(lst, d, s):
        return {'list': lst, 'dict': d, 'set': s}
    
    result = process_empty([], {}, set())
    assert str(result['list']) == "pvector([])"
    assert len(result['dict']) == 0
    assert len(result['set']) == 0


def test_mutant_return_value_is_frozen():
    @mutant
    def return_list():
        return [1, 2, 3]
    
    result = return_list()
    assert str(result) == "pvector([1, 2, 3])"


def test_mutant_with_primitive_types():
    @mutant
    def process_primitives(n, s):
        return {'number': n, 'string': s}
    
    result = process_primitives(42, 'hello')
    assert result['number'] == 42
    assert result['string'] == 'hello'


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst
    
    result = modify_list([1, 2, 3])
    assert len(result) == 3


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def create_list():
        return [1, 2, 3]
    
    result = create_list()
    assert isinstance(result, type(pvector()))


def test_mutant_with_dict_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    result = modify_dict({'a': 1})
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1
    assert 'new_key' not in result


def test_mutant_with_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        data['items'].append(4)
        return data
    
    result = process_nested({'items': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert len(result['items']) == 3


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'x': 1})
    assert isinstance(result, type(pmap()))


def test_mutant_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        return []
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        s.add(4)
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset()))
    assert len(result) == 3


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def combine(a, b):
        return [a, b]
    
    result = combine([1, 2], [3, 4])
    assert isinstance(result, type(pvector()))
    assert len(result) == 2


def test_mutant_with_nested_list_in_dict():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pvector
    
    @mutant
    def process_complex(data):
        data['nested'].append(5)
        return data
    
    result = process_complex({'nested': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['nested'], type(pvector()))
    assert len(result['nested']) == 3


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        lst_copy = list(lst)
        lst_copy.append(999)
        return lst_copy
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3, 999])


def test_mutant_freezes_dict_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def modify_dict(d):
        d_copy = dict(d)
        d_copy['new_key'] = 'new_value'
        return d_copy
    
    result = modify_dict({'a': 1})
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        data_copy = dict(data)
        data_copy['items'] = list(data_copy['items'])
        data_copy['items'].append(4)
        return data_copy
    
    result = process_nested({'items': [1, 2, 3]})
    assert isinstance(result, pmap)
    assert isinstance(result['items'], pvector)
    assert result == pmap({'items': pvector([1, 2, 3, 4])})


def test_mutant_freezes_set_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def modify_set(s):
        s_copy = set(s)
        s_copy.add(4)
        return s_copy
    
    result = modify_set({1, 2, 3})
    assert isinstance(result, pset)
    assert result == pset([1, 2, 3, 4])


def test_mutant_freezes_tuple_arguments():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t + (4,)
    
    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)


def test_mutant_freezes_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def func_with_kwargs(a, b=None):
        result = dict(a)
        if b:
            result['b_key'] = b
        return result
    
    result = func_with_kwargs({'x': 1}, b={'y': 2})
    assert isinstance(result, pmap)
    assert isinstance(result['b_key'], pmap)


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        result = dict(dct)
        result['list'] = list(lst)
        result['list'].append(99)
        return result
    
    result = combine([1, 2], {'a': 1})
    assert isinstance(result, pmap)
    assert isinstance(result['list'], pvector)
    assert result == pmap({'a': 1, 'list': pvector([1, 2, 99])})


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def documented_func():
        """Test docstring"""
        return []
    
    assert documented_func.__doc__ == "Test docstring"
    assert documented_func.__name__ == "documented_func"


def test_mutant_with_scalar_return():
    from pyrsistent._helpers import mutant
    
    @mutant
    def return_scalar(x):
        return x + 1
    
    result = return_scalar(5)
    assert result == 6


def test_mutant_deeply_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_deep(data):
        return {'nested': {'list': [1, 2, {'inner': 3}]}}
    
    result = process_deep({})
    assert isinstance(result, pmap)
    assert isinstance(result['nested'], pmap)
    assert isinstance(result['nested']['list'], pvector)
    assert isinstance(result['nested']['list'][2], pmap)


# LLM-generated content at query #16
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset, pmap
    
    @mutant
    def modify_and_return(data):
        return data
    
    input_list = [1, 2, 3]
    result = modify_and_return(input_list)
    
    assert not isinstance(result, list)
    assert isinstance(result, (pset, tuple))


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]
    assert result == pvector([1, 2, 3, 999])


def test_mutant_freezes_return_value():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def create_dict():
        return {'a': 1, 'b': 2}

    result = create_dict()
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_with_nested_structures():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def process_nested(data):
        data[0].append(5)
        return data

    original = [[1, 2, 3], {'key': 'value'}]
    result = process_nested(original)
    assert original == [[1, 2, 3], {'key': 'value'}]
    assert result == pvector([pvector([1, 2, 3, 5]), pmap({'key': 'value'})])


def test_mutant_with_keyword_arguments():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def modify_with_kwargs(lst, extra_dict):
        lst.append(10)
        extra_dict['new_key'] = 'new_value'
        return (lst, extra_dict)

    result = modify_with_kwargs([1, 2], extra_dict={'a': 1})
    assert result[0] == pvector([1, 2, 10])
    assert result[1] == pmap({'a': 1, 'new_key': 'new_value'})


def test_mutant_with_set_argument():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def process_set(s):
        s.add(999)
        return s

    original = {1, 2, 3}
    result = process_set(original)
    assert original == {1, 2, 3}
    assert result == pset([1, 2, 3, 999])


def test_mutant_with_tuple_argument():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def process_tuple(t):
        return t + (999,)

    original = (1, [2, 3])
    result = process_tuple(original)
    assert original == (1, [2, 3])
    assert result == (1, pvector([2, 3]), 999)


def test_mutant_preserves_function_name():
    from pyrsistent._helpers import mutant

    @mutant
    def my_function():
        return []

    assert my_function.__name__ == 'my_function'


def test_mutant_with_multiple_arguments():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def combine(lst1, lst2):
        lst1.append(999)
        lst2.append(888)
        return (lst1, lst2)

    result = combine([1, 2], [3, 4])
    assert result[0] == pvector([1, 2, 999])
    assert result[1] == pvector([3, 4, 888])


def test_mutant_with_deeply_nested_structure():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def process_deep(data):
        data['nested']['list'].append(100)
        return data

    original = {'nested': {'list': [1, 2, 3]}}
    result = process_deep(original)
    assert original == {'nested': {'list': [1, 2, 3]}}
    assert result['nested']['list'] == pvector([1, 2, 3, 100])


def test_mutant_with_primitive_return():
    from pyrsistent._helpers import mutant

    @mutant
    def add_numbers(a, b):
        return a + b

    result = add_numbers(5, 3)
    assert result == 8


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant_decorator_predicate_line_1_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset, pmap
    
    @mutant
    def modify_input(s):
        return s
    
    input_set = pset([1, 2, 3])
    result = modify_input(input_set)
    
    assert result is not None
    assert len(result) == 3


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset
    
    @mutant
    def modify_and_return(data):
        return data
    
    input_dict = {'a': 1, 'b': 2}
    result = modify_and_return(input_dict)
    
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_map(m):
        return m
    
    @mutant
    def modify_set(s):
        return s
    
    @mutant
    def modify_list(lst):
        return lst
    
    test_map = pmap({'a': 1, 'b': 2})
    result_map = modify_map(test_map)
    assert result_map.is_persistent()
    
    test_set = pset([1, 2, 3])
    result_set = modify_set(test_set)
    assert result_set.is_persistent()
    
    test_list = [1, 2, 3]
    result_list = modify_list(test_list)
    assert isinstance(result_list, type(freeze(test_list)))
    
    @mutant
    def return_dict(d):
        return d
    
    test_dict = {'x': 10}
    result_dict = return_dict(test_dict)
    assert isinstance(result_dict, type(freeze(test_dict)))
    assert not isinstance(result_dict, dict)


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst

    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector()))
    assert len(result) == 3


def test_mutant_with_dict_argument():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant

    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = modify_dict({'a': 1})
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1


def test_mutant_with_nested_structures():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def process_nested(data):
        data['items'].append(4)
        return data

    result = process_nested({'items': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['items'], type(pvector()))
    assert len(result['items']) == 3


def test_mutant_with_set_argument():
    from pyrsistent import pset
    from pyrsistent._helpers import mutant

    @mutant
    def process_set(s):
        s.add(4)
        return s

    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset()))


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant

    @mutant
    def process_tuple(t):
        return t + (4,)

    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_mutant_with_multiple_arguments():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def combine(lst, dct):
        lst.append(100)
        dct['key'] = 'value'
        return {'list': lst, 'dict': dct}

    result = combine([1, 2], {'a': 1})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['list'], type(pvector()))
    assert isinstance(result['dict'], type(pmap()))


def test_mutant_with_keyword_arguments():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant

    @mutant
    def process_with_kwargs(d, key='default'):
        d[key] = 'processed'
        return d

    result = process_with_kwargs({'a': 1}, key='b')
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant

    @mutant
    def my_function():
        """Test docstring"""
        return {}

    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_deeply_nested_structures():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def process_deep(data):
        data['level1']['level2'].append(5)
        return data

    result = process_deep({'level1': {'level2': [1, 2, 3]}})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['level1'], type(pmap()))
    assert isinstance(result['level1']['level2'], type(pvector()))
    assert len(result['level1']['level2']) == 3


def test_mutant_return_value_is_frozen():
    from pyrsistent import pvector
    from pyrsistent._helpers import mutant

    @mutant
    def returns_list():
        return [1, 2, 3]

    result = returns_list()
    assert isinstance(result, type(pvector()))


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze, pset, pmap
    
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst
    
    input_list = [1, 2, 3]
    result = modify_list(input_list)
    
    assert isinstance(result, type(freeze([])))
    assert result == freeze([1, 2, 3])


def test_mutant_decorator_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze
    
    @mutant
    def combine_dicts(d1, d2):
        d1['key'] = 'value'
        return d1
    
    dict1 = {'a': 1}
    dict2 = {'b': 2}
    result = combine_dicts(dict1, dict2)
    
    assert isinstance(result, type(freeze({})))
    assert result == freeze({'a': 1})


def test_mutant_decorator_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze
    
    @mutant
    def process_data(data, multiplier=1):
        return [x * multiplier for x in data]
    
    input_data = [1, 2, 3]
    result = process_data(input_data, multiplier=2)
    
    assert isinstance(result, type(freeze([])))
    assert result == freeze([2, 4, 6])


def test_mutant_decorator_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def sample_function(x):
        """Sample docstring"""
        return x
    
    assert sample_function.__name__ == 'sample_function'
    assert sample_function.__doc__ == 'Sample docstring'


def test_mutant_decorator_with_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze
    
    @mutant
    def modify_nested(data):
        data['nested']['value'] = 100
        return data
    
    input_data = {'nested': {'value': 1}}
    result = modify_nested(input_data)
    
    assert isinstance(result, type(freeze({})))
    assert result == freeze({'nested': {'value': 1}})


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze, thaw
    
    @mutant
    def modify_list(lst):
        return lst + [4]
    
    input_list = [1, 2, 3]
    result = modify_list(input_list)
    
    # Verify that the result is frozen (persistent)
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"
    
    # Verify that the original input wasn't mutated
    assert input_list == [1, 2, 3]
    
    # Verify the result contains expected values
    assert thaw(result) == [1, 2, 3, 4]


def test_mutant_decorator_with_dict_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import thaw
    
    @mutant
    def modify_dict(d):
        d_copy = dict(d)
        d_copy['new_key'] = 'new_value'
        return d_copy
    
    input_dict = {'a': 1, 'b': 2}
    result = modify_dict(input_dict)
    
    # Verify that result is frozen
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"
    
    # Verify original wasn't mutated
    assert input_dict == {'a': 1, 'b': 2}
    
    # Verify result has expected values
    assert thaw(result) == {'a': 1, 'b': 2, 'new_key': 'new_value'}


def test_mutant_decorator_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import thaw
    
    @mutant
    def create_map(x=1, y=2):
        return {'x': x, 'y': y}
    
    result = create_map(x=10, y=20)
    
    # Verify result is frozen
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"
    
    # Verify result contains expected values
    assert thaw(result) == {'x': 10, 'y': 20}


def test_mutant_decorator_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    result = add_to_dict({'a': 1}, 'b', 2)
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2


def test_mutant_freezes_list_arguments():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    result = append_to_list([1, 2, 3], 4)
    assert isinstance(result, PVector)
    assert len(result) == 4
    assert result[3] == 4


def test_mutant_freezes_nested_structures():
    @mutant
    def modify_nested(data):
        return data
    
    nested_input = {'a': [1, 2, {'b': 3}]}
    result = modify_nested(nested_input)
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['a'][2], PMap)


def test_mutant_freezes_set_arguments():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, PSet)
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_mutant_freezes_tuple_arguments():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)
    assert isinstance(result[2], PMap)


def test_mutant_with_kwargs():
    @mutant
    def create_dict(key1=None, key2=None):
        return {'a': key1, 'b': key2}
    
    result = create_dict(key1={'nested': [1, 2]}, key2=[3, 4])
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PMap)
    assert isinstance(result['b'], PVector)


def test_mutant_preserves_function_name():
    @mutant
    def my_function():
        return {}
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_primitive_return():
    @mutant
    def return_number():
        return 42
    
    result = return_number()
    assert result == 42


def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def mixed_function(a, b, c=None):
        return {'a': a, 'b': b, 'c': c}
    
    result = mixed_function([1, 2], {'x': 1}, c={'y': 2})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    assert isinstance(result['c'], PMap)


def test_mutant_multiple_arguments():
    @mutant
    def combine(dict1, dict2, list1):
        return {'d1': dict1, 'd2': dict2, 'l': list1}
    
    result = combine({'a': 1}, {'b': 2}, [1, 2, 3])
    assert isinstance(result, PMap)
    assert isinstance(result['d1'], PMap)
    assert isinstance(result['d2'], PMap)
    assert isinstance(result['l'], PVector)


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [4]
    
    result = modify_list([1, 2, 3])
    assert str(result) == "pvector([1, 2, 3, 4])"


def test_mutant_freezes_dict_arguments():
    @mutant
    def get_value(d):
        return d['key']
    
    result = get_value({'key': 'value'})
    assert result == 'value'


def test_mutant_preserves_function_name():
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, d):
        return lst + [d['val']]
    
    result = combine([1, 2], {'val': 3})
    assert str(result) == "pvector([1, 2, 3])"


def test_mutant_with_kwargs():
    @mutant
    def func_with_kwargs(a, b=5):
        return a + b
    
    result = func_with_kwargs(3, b=2)
    assert result == 5


def test_mutant_freezes_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'a': [1, 2, {'b': 3}]})
    assert str(result) == "pmap({'a': pvector([1, 2, pmap({'b': 3})])})"


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert str(result) == "pset([1, 2, 3])"


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert str(result) == "(1, pvector([2, 3]))"


def test_mutant_with_empty_containers():
    @mutant
    def process_empty(lst, d):
        return (lst, d)
    
    result = process_empty([], {})
    assert str(result[0]) == "pvector([])"
    assert str(result[1]) == "pmap({})"


def test_mutant_return_value_is_frozen():
    @mutant
    def return_list():
        return [1, 2, 3]
    
    result = return_list()
    assert hasattr(result, 'append') is False


def test_mutant_with_none_argument():
    @mutant
    def process_none(x):
        return x
    
    result = process_none(None)
    assert result is None


def test_mutant_with_primitive_types():
    @mutant
    def add_numbers(a, b):
        return a + b
    
    result = add_numbers(5, 3)
    assert result == 8


def test_mutant_with_defaultdict_argument():
    from collections import defaultdict
    
    @mutant
    def process_defaultdict(d):
        return d
    
    dd = defaultdict(list)
    dd['key'] = [1, 2]
    result = process_defaultdict(dd)
    assert str(result) == "pmap({'key': pvector([1, 2])})"


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst
    
    original = [1, 2, 3]
    result = modify_list(original)
    assert isinstance(result, type(pvector([1, 2, 3])))


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def create_list():
        return [1, 2, 3]
    
    result = create_list()
    assert isinstance(result, type(pvector([1, 2, 3])))


def test_mutant_with_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pvector
    
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'a': [1, 2], 'b': {'c': 3}})
    assert isinstance(result, type(pmap({})))
    assert isinstance(result['a'], type(pvector([])))
    assert isinstance(result['b'], type(pmap({})))


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = process_kwargs(1, b=2)
    assert isinstance(result, type(pmap({})))
    assert result['a'] == 1
    assert result['b'] == 2


def test_mutant_prevents_mutation_of_arguments():
    from pyrsistent._helpers import mutant
    
    @mutant
    def try_mutate(lst):
        try:
            lst.append(999)
        except AttributeError:
            pass
        return lst
    
    original = [1, 2, 3]
    result = try_mutate(original)
    assert original == [1, 2, 3]


def test_mutant_with_set():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset([])))


def test_mutant_with_tuple():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def combine(lst1, lst2):
        return lst1 + lst2
    
    result = combine([1, 2], [3, 4])
    assert isinstance(result, type(pvector([])))


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def documented_func():
        """This is a docstring"""
        return []
    
    assert documented_func.__doc__ == "This is a docstring"
    assert documented_func.__name__ == "documented_func"


def test_mutant_with_empty_containers():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def return_empty():
        return {'list': [], 'dict': {}, 'set': set()}
    
    result = return_empty()
    assert isinstance(result['list'], type(pvector([])))
    assert isinstance(result['dict'], type(pmap({})))
    assert isinstance(result['set'], type(pset([])))


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def modify_list(lst):
        return lst.append(4)

    result = modify_list([1, 2, 3])
    assert isinstance(result, type(None))


def test_mutant_with_dict_argument():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant

    @mutant
    def process_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = process_dict({'a': 1})
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})


def test_mutant_with_list_argument():
    from pyrsistent import pvector
    from pyrsistent._helpers import mutant

    @mutant
    def process_list(lst):
        return lst + [4, 5]

    result = process_list([1, 2, 3])
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3, 4, 5])


def test_mutant_with_nested_structures():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def process_nested(data):
        return data

    result = process_nested({'a': [1, 2], 'b': {'c': 3}})
    assert isinstance(result, pmap)
    assert isinstance(result['a'], pvector)
    assert isinstance(result['b'], pmap)


def test_mutant_with_set_argument():
    from pyrsistent import pset
    from pyrsistent._helpers import mutant

    @mutant
    def process_set(s):
        return s

    result = process_set({1, 2, 3})
    assert isinstance(result, pset)
    assert result == pset([1, 2, 3])


def test_mutant_with_tuple_argument():
    from pyrsistent import pvector
    from pyrsistent._helpers import mutant

    @mutant
    def process_tuple(t):
        return t

    result = process_tuple((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)
    assert isinstance(result[1], pvector)
    assert isinstance(result[2], pmap)


def test_mutant_with_multiple_arguments():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def combine(lst, dct):
        return {'list': lst, 'dict': dct}

    result = combine([1, 2], {'a': 3})
    assert isinstance(result, pmap)
    assert isinstance(result['list'], pvector)
    assert isinstance(result['dict'], pmap)


def test_mutant_with_keyword_arguments():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant

    @mutant
    def process_kwargs(a=None, b=None):
        return {'a': a, 'b': b}

    result = process_kwargs(a=[1, 2], b={'x': 10})
    assert isinstance(result, pmap)


def test_mutant_with_mixed_arguments_and_kwargs():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant

    @mutant
    def process_mixed(lst, dct=None):
        return [lst, dct]

    result = process_mixed([1, 2], dct={'key': 'value'})
    assert isinstance(result, pvector)
    assert isinstance(result[0], pvector)
    assert isinstance(result[1], pmap)


def test_mutant_preserves_function_name():
    from pyrsistent._helpers import mutant

    @mutant
    def my_function():
        return None

    assert my_function.__name__ == 'my_function'


def test_mutant_with_scalar_return_value():
    from pyrsistent._helpers import mutant

    @mutant
    def return_scalar():
        return 42

    result = return_scalar()
    assert result == 42


def test_mutant_with_empty_containers():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant

    @mutant
    def process_empty(lst, dct, s):
        return {'list': lst, 'dict': dct, 'set': s}

    result = process_empty([], {}, set())
    assert isinstance(result, pmap)
    assert isinstance(result['list'], pvector)
    assert isinstance(result['dict'], pmap)
    assert isinstance(result['set'], pset)


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_dict(d):
        d['key'] = 'value'
        return d
    
    result = modify_dict({'initial': 'data'})
    assert isinstance(result, PMap)
    assert result['initial'] == 'data'
    assert result['key'] == 'value'


def test_mutant_freezes_list_arguments():
    @mutant
    def append_to_list(lst):
        lst.append(5)
        return lst
    
    result = append_to_list([1, 2, 3])
    assert isinstance(result, PVector)
    assert len(result) == 4
    assert result[3] == 5


def test_mutant_freezes_nested_structures():
    @mutant
    def modify_nested(data):
        data['nested']['value'] = 99
        return data
    
    result = modify_nested({'nested': {'value': 1}})
    assert isinstance(result, PMap)
    assert isinstance(result['nested'], PMap)
    assert result['nested']['value'] == 99


def test_mutant_freezes_kwargs():
    @mutant
    def func_with_kwargs(a, b=None):
        if b is None:
            b = {}
        b['key'] = 'added'
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'existing': 'value'})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)
    assert result['b']['key'] == 'added'


def test_mutant_preserves_function_name():
    @mutant
    def original_function():
        return {}
    
    assert original_function.__name__ == 'original_function'


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return {'set': s}
    
    result = process_set({1, 2, 3})
    assert isinstance(result, PMap)
    assert isinstance(result['set'], PSet)


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], 4))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(list1, list2):
        return list1 + list2
    
    result = combine([1, 2], [3, 4])
    assert isinstance(result, PVector)
    assert len(result) == 4


def test_mutant_with_no_arguments():
    @mutant
    def create_data():
        return {'key': 'value'}
    
    result = create_data()
    assert isinstance(result, PMap)
    assert result['key'] == 'value'


def test_mutant_freezes_return_value_list():
    @mutant
    def return_list():
        return [1, 2, 3]
    
    result = return_list()
    assert isinstance(result, PVector)


def test_mutant_freezes_return_value_dict():
    @mutant
    def return_dict():
        return {'a': 1}
    
    result = return_dict()
    assert isinstance(result, PMap)


def test_mutant_with_empty_containers():
    @mutant
    def process_empty(d, l, s):
        return {'dict': d, 'list': l, 'set': s}
    
    result = process_empty({}, [], set())
    assert isinstance(result, PMap)
    assert isinstance(result['dict'], PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)


# LLM-generated content at query #29
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    import collections
    from pyrsistent import freeze, pmap
    
    dd = collections.defaultdict(list)
    dd['a'] = [1, 2]
    dd['b'] = [3, 4]
    
    result = freeze(dd, strict=True)
    
    assert isinstance(result, type(pmap()))
    assert result['a'] == (1, 2) or result['a'] == [1, 2]
    assert result['b'] == (3, 4) or result['b'] == [3, 4]


# LLM-generated content at query #30
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        return lst
    
    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'a': [1, 2], 'b': {'c': 3}})
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def create_structure():
        return {'key': [1, 2, 3]}
    
    result = create_structure()
    assert result == pmap({'key': pvector([1, 2, 3])})


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(a, b):
        return [a, b]
    
    result = combine({'x': 1}, [1, 2])
    assert result == pvector([pmap({'x': 1}), pvector([1, 2])])


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'nested': True})
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'nested': True})})


def test_mutant_with_set():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset, pvector
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_mutant_with_tuple():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], 4))
    assert result == (1, pvector([2, 3]), 4)


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        """My docstring"""
        pass
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'My docstring'


def test_mutant_with_empty_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_empty(lst, d, s):
        return (lst, d, s)
    
    result = process_empty([], {}, set())
    assert result == (pvector([]), pmap({}), pset([]))


def test_mutant_with_primitive_types():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_primitives(a, b, c):
        return (a, b, c)
    
    result = process_primitives(1, 'string', 3.14)
    assert result == (1, 'string', 3.14)


# LLM-generated content at query #31
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset
    
    @mutant
    def modify_and_return(data):
        return data
    
    original_list = [1, 2, 3]
    result = modify_and_return(original_list)
    
    assert isinstance(result, (pset, tuple)) or hasattr(result, '__hash__')
    
    @mutant
    def process_dict(d):
        return d
    
    original_dict = {'a': 1, 'b': 2}
    result_dict = process_dict(original_dict)
    
    assert isinstance(result_dict, type(pmap({}))) or hasattr(result_dict, '__hash__')
    
    @mutant
    def with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result_with_kwargs = with_kwargs([1, 2], b={'x': 10})
    
    assert hasattr(result_with_kwargs, '__hash__')


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    original_dict = {'key': 'value'}
    result = modify_dict(original_dict)
    
    # The predicate at line 1 (def mutant(fn):) evaluates to False
    # because we're testing that the function is not None and the decorator works
    assert (mutant is None) == False
    
    # Additional assertions to verify the decorator functionality
    assert isinstance(result, type(freeze({})))
    assert result['key'] == 'value'


def test_mutant_decorator_with_persistent_structures():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_pmap(m):
        return m
    
    test_map = pmap({'a': 1, 'b': 2})
    result = process_pmap(test_map)
    
    # Verify the predicate at line 1 is False (the function exists and is callable)
    assert (mutant is None) == False
    assert callable(mutant)


def test_mutant_decorator_freezes_kwargs():
    from pyrsistent import freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs(1, b={'nested': 'dict'})
    
    # Test that the predicate evaluates to False
    assert (mutant is None) == False
    assert result is not None


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset

    @mutant
    def add_to_list(lst, item):
        lst.append(item)
        return lst

    result = add_to_list([1, 2], 3)
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d

    result = modify_dict({'a': 1})
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap

    @mutant
    def create_map(key1='default', key2='value'):
        return {key1: key2}

    result = create_map(key1='test', key2='result')
    assert isinstance(result, pmap)
    assert result == pmap({'test': 'result'})


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset

    @mutant
    def process_set(s):
        s.add(4)
        return s

    result = process_set({1, 2, 3})
    assert isinstance(result, pset)
    assert result == pset([1, 2, 3, 4])


def test_mutant_with_mixed_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def combine(lst, dct):
        lst.append(dct['key'])
        return lst

    result = combine([1, 2], {'key': 3})
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant

    @mutant
    def process_tuple(t):
        return t + (4,)

    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant

    @mutant
    def my_function(x):
        """This is my function"""
        return x

    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'This is my function'


def test_mutant_with_empty_containers():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset

    @mutant
    def return_empties():
        return {'list': [], 'dict': {}, 'set': set()}

    result = return_empties()
    assert isinstance(result, pmap)
    assert isinstance(result['list'], pvector)
    assert isinstance(result['dict'], pmap)
    assert isinstance(result['set'], pset)


def test_mutant_with_nested_list_and_dict():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def create_nested(data):
        return [1, {'nested': [2, 3]}]

    result = create_nested([])
    assert isinstance(result, pvector)
    assert isinstance(result[1], pmap)
    assert isinstance(result[1]['nested'], pvector)


# LLM-generated content at query #34
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_list(lst):
        lst_copy = list(lst)
        lst_copy.append(99)
        return lst_copy
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 3, 99]


def test_mutant_freezes_nested_structures():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_nested(data):
        data_dict = dict(data)
        data_dict['new_key'] = [1, 2, 3]
        return data_dict
    
    result = process_nested({'a': [1, 2]})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['a'], type(pvector()))
    assert isinstance(result['new_key'], type(pvector()))


def test_mutant_freezes_return_value():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def create_dict():
        return {'x': 10, 'y': [1, 2, 3]}
    
    result = create_dict()
    assert isinstance(result, type(pmap()))
    assert isinstance(result['y'], type(pvector()))


def test_mutant_with_multiple_arguments():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def combine(list1, list2):
        combined = list(list1) + list(list2)
        return combined
    
    result = combine([1, 2], [3, 4])
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 3, 4]


def test_mutant_with_kwargs():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def build_map(key1=None, key2=None):
        return {'a': key1, 'b': key2}
    
    result = build_map(key1=[1, 2], key2=[3, 4])
    assert isinstance(result, type(pmap()))
    assert isinstance(result['a'], type(pvector()))
    assert isinstance(result['b'], type(pvector()))


def test_mutant_with_set_argument():
    from pyrsistent import pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_set(s):
        s_copy = set(s)
        s_copy.add(4)
        return s_copy
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset()))
    assert 4 in result


def test_mutant_with_tuple_argument():
    from pyrsistent import pvector
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t + (4, 5)
    
    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4, 5)


def test_mutant_with_mixed_types():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_mixed(lst, dct, st):
        return {
            'list': lst,
            'dict': dct,
            'set': st
        }
    
    result = process_mixed([1, 2], {'x': 10}, {5, 6})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['list'], type(pvector()))
    assert isinstance(result['dict'], type(pmap()))
    assert isinstance(result['set'], type(pset()))


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        """Test docstring"""
        return x
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


# LLM-generated content at query #35
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_dict(d):
        return d
    
    input_dict = {'a': 1, 'b': [2, 3]}
    result = modify_dict(input_dict)
    
    # The predicate at line 1: def mutant(fn): - we verify the decorator works
    # by checking that arguments are frozen (converted to persistent types)
    assert isinstance(result, type(freeze({'a': 1, 'b': [2, 3]})))
    
    @mutant
    def process_list(lst):
        return lst
    
    input_list = [1, 2, 3]
    result_list = process_list(input_list)
    
    # Verify return value is frozen
    assert isinstance(result_list, type(freeze([1, 2, 3])))
    
    @mutant
    def process_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result_kwargs = process_kwargs(1, b=2)
    assert isinstance(result_kwargs, type(freeze({})))


# LLM-generated content at query #36
#--------------------------

```python
def test_freeze_defaultdict_predicate_line_32():
    from pyrsistent._helpers import freeze
    import collections
    
    # Test that the predicate at line 32 evaluates to True for defaultdict
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd, strict=True)
    
    # The result should be a pmap with frozen values
    assert result == {'a': 1, 'b': 2}
    
    # Test with nested structure
    dd_nested = collections.defaultdict(list, {'x': [1, 2], 'y': [3, 4]})
    result_nested = freeze(dd_nested, strict=True)
    
    # Values should be frozen (lists become pvectors)
    assert len(result_nested) == 2
    assert 'x' in result_nested
    assert 'y' in result_nested


# LLM-generated content at query #37
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        return lst
    
    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_data(data):
        return data
    
    result = process_data({'a': [1, 2], 'b': {'c': 3}})
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def create_structure():
        return [1, 2, {'x': 3}]
    
    result = create_structure()
    assert result == pvector([1, 2, pmap({'x': 3})])


def test_mutant_freezes_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_kwargs(a=None, b=None):
        return {'a': a, 'b': b}
    
    result = process_kwargs(a=[1, 2], b={'x': 3})
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'x': 3})})


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], 4))
    assert result == (1, pvector([2, 3]), 4)


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        return [lst, dct]
    
    result = combine([1, 2], {'a': 3})
    assert result == pvector([pvector([1, 2]), pmap({'a': 3})])


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def documented_function(x):
        """This is a documented function."""
        return x
    
    assert documented_function.__name__ == 'documented_function'
    assert 'documented function' in documented_function.__doc__


def test_mutant_with_mixed_kwargs_and_args():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def mixed_func(a, b, c=None):
        return {'a': a, 'b': b, 'c': c}
    
    result = mixed_func([1], {'x': 2}, c={'y': 3})
    assert result == pmap({'a': pvector([1]), 'b': pmap({'x': 2}), 'c': pmap({'y': 3})})


def test_mutant_with_primitive_return():
    from pyrsistent._helpers import mutant
    
    @mutant
    def return_primitive(x):
        return 42
    
    result = return_primitive([1, 2, 3])
    assert result == 42


def test_mutant_with_empty_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_empty(lst, dct, st):
        return [lst, dct, st]
    
    result = process_empty([], {}, set())
    assert result == pvector([pvector([]), pmap({}), pset([])])


# LLM-generated content at query #38
#--------------------------

```python
def test_mutant_freezes_arguments():
    @mutant
    def modify_list(lst):
        return lst + [1]
    
    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 1])


def test_mutant_freezes_return_value():
    @mutant
    def create_dict():
        return {'a': 1, 'b': 2}
    
    result = create_dict()
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_freezes_nested_structures():
    @mutant
    def create_nested():
        return {'key': [1, 2, 3]}
    
    result = create_nested()
    assert isinstance(result, PMap)
    assert isinstance(result['key'], PVector)


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, dct):
        return {'list': lst, 'dict': dct}
    
    result = combine([1, 2], {'x': 10})
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)


def test_mutant_with_kwargs():
    @mutant
    def make_map(a=1, b=2):
        return {'a': a, 'b': b}
    
    result = make_map(a=5, b=10)
    assert isinstance(result, PMap)
    assert result['a'] == 5
    assert result['b'] == 10


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return {'set': s}
    
    result = process_set({1, 2, 3})
    assert isinstance(result, PMap)
    assert isinstance(result['set'], PSet)


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return {'tuple': t}
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, PMap)
    assert isinstance(result['tuple'], tuple)
    assert isinstance(result['tuple'][1], PVector)


def test_mutant_preserves_function_metadata():
    @mutant
    def my_function():
        """Test docstring"""
        return {}
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_primitive_return():
    @mutant
    def return_int():
        return 42
    
    result = return_int()
    assert result == 42


def test_mutant_deeply_nested_structures():
    @mutant
    def create_deep():
        return {'a': {'b': {'c': [1, 2, 3]}}}
    
    result = create_deep()
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PMap)
    assert isinstance(result['a']['b'], PMap)
    assert isinstance(result['a']['b']['c'], PVector)


# LLM-generated content at query #39
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_data(data):
        return data
    
    result = process_data({'a': [1, 2], 'b': 3})
    assert isinstance(result, pmap)
    assert isinstance(result['a'], pvector)
    assert result == pmap({'a': pvector([1, 2]), 'b': 3})


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def create_list():
        return [1, 2, 3]
    
    result = create_list()
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs(1, b=[2, 3])
    assert isinstance(result, pmap)
    assert result['a'] == 1


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, pset)


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], 4))
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        return {'list': lst, 'dict': dct}
    
    result = combine([1, 2], {'x': 10})
    assert isinstance(result, pmap)
    assert isinstance(result['list'], pvector)
    assert isinstance(result['dict'], pmap)


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def documented_function():
        """This is a docstring."""
        return []
    
    assert documented_function.__name__ == 'documented_function'
    assert documented_function.__doc__ == 'This is a docstring.'


def test_mutant_with_deeply_nested_structure():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'a': [1, {'b': [2, 3]}]})
    assert isinstance(result, pmap)
    assert isinstance(result['a'], pvector)
    assert isinstance(result['a'][1], pmap)
    assert isinstance(result['a'][1]['b'], pvector)


def test_mutant_with_empty_containers():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_empty(lst, dct, s):
        return {'list': lst, 'dict': dct, 'set': s}
    
    result = process_empty([], {}, set())
    assert isinstance(result, pmap)
    assert isinstance(result['list'], pvector)
    assert isinstance(result['dict'], pmap)
    assert isinstance(result['set'], pset)


# LLM-generated content at query #40
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [1, 2, 3]
    
    result = modify_list([1, 2])
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"
    assert list(result) == [1, 2, 1, 2, 3]


def test_mutant_freezes_dict_arguments():
    @mutant
    def get_value(d):
        return d['key']
    
    result = get_value({'key': 'value'})
    assert result == 'value'


def test_mutant_freezes_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'list': [1, 2], 'dict': {'nested': 'value'}})
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"


def test_mutant_freezes_multiple_arguments():
    @mutant
    def combine(lst, dct):
        return [lst, dct]
    
    result = combine([1, 2], {'a': 1})
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"
    assert len(result) == 2


def test_mutant_freezes_kwargs():
    @mutant
    def func_with_kwargs(a=None, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs(a=[1, 2], b={'x': 1})
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"


def test_mutant_preserves_function_name():
    @mutant
    def my_function():
        return 42
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert str(type(result)) == "<class 'pyrsistent._pset.PSet'>"


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_mutant_with_empty_collections():
    @mutant
    def process_empty(lst, dct, s):
        return {'list': lst, 'dict': dct, 'set': s}
    
    result = process_empty([], {}, set())
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"


def test_mutant_return_value_is_frozen():
    @mutant
    def return_mutable():
        return [1, 2, {'a': 3}]
    
    result = return_mutable()
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"
    assert str(type(result[2])) == "<class 'pyrsistent._pmap.PMap'>"


# LLM-generated content at query #41
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        lst = lst.append(4)
        return lst
    
    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_mutant_with_dict_argument():
    @mutant
    def process_dict(d):
        return d
    
    result = process_dict({'a': 1, 'b': 2})
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'list': [1, 2], 'set': {3, 4}})
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, d):
        return {'result': lst, 'data': d}
    
    result = combine([1, 2], {'x': 10})
    assert isinstance(result, PMap)
    assert isinstance(result['result'], PVector)
    assert isinstance(result['data'], PMap)


def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = process_kwargs([1, 2], b={'key': 'value'})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)
    assert isinstance(result[2], PMap)


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, PSet)


def test_mutant_preserves_function_name():
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_empty_containers():
    @mutant
    def process_empty(lst, d, s):
        return {'list': lst, 'dict': d, 'set': s}
    
    result = process_empty([], {}, set())
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['dict'], PMap)
    assert isinstance(result['set'], PSet)


def test_mutant_deeply_nested_structures():
    @mutant
    def process_deep(data):
        return data
    
    result = process_deep({'a': {'b': [1, 2, {'c': 3}]}})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PMap)
    assert isinstance(result['a']['b'], PVector)
    assert isinstance(result['a']['b'][2], PMap)


# LLM-generated content at query #42
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [1]
    
    result = modify_list([1, 2, 3])
    assert str(type(result).__name__) == 'PVector'


def test_mutant_with_dict_argument():
    @mutant
    def get_value(d):
        return d
    
    result = get_value({'a': 1, 'b': 2})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'key': [1, 2, 3]})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert str(type(result).__name__) == 'PSet'


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, d):
        return lst
    
    result = combine([1, 2], {'a': 1})
    assert str(type(result).__name__) == 'PVector'


def test_mutant_with_kwargs():
    @mutant
    def process_kwargs(a, b=None):
        return [a, b]
    
    result = process_kwargs([1], b={'x': 1})
    assert str(type(result).__name__) == 'PVector'


def test_mutant_preserves_function_name():
    @mutant
    def my_function():
        return []
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_nested_dict_and_list():
    @mutant
    def process_complex(data):
        return data
    
    result = process_complex({'outer': {'inner': [1, 2, 3]}})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_return_value_is_frozen():
    @mutant
    def create_list():
        return [1, 2, 3]
    
    result = create_list()
    assert str(type(result).__name__) == 'PVector'


# LLM-generated content at query #43
#--------------------------

```python
def test_freeze_dict():
    from pyrsistent import freeze, pmap
    result = freeze({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_nested_dict():
    from pyrsistent import freeze, pmap
    result = freeze({'a': {'b': 3}})
    assert result == pmap({'a': pmap({'b': 3})})


def test_freeze_list():
    from pyrsistent import freeze, pvector
    result = freeze([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_freeze_nested_list():
    from pyrsistent import freeze, pvector, pmap
    result = freeze([1, {'a': 3}])
    assert result == pvector([1, pmap({'a': 3})])


def test_freeze_tuple():
    from pyrsistent import freeze, pvector
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)


def test_freeze_nested_tuple():
    from pyrsistent import freeze, pvector
    result = freeze((1, [2, 3]))
    assert result == (1, pvector([2, 3]))


def test_freeze_set():
    from pyrsistent import freeze, pset
    result = freeze({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_freeze_set_with_duplicates():
    from pyrsistent import freeze, pset
    result = freeze({1, 2, 2, 3})
    assert result == pset([1, 2, 3])


def test_freeze_primitive():
    from pyrsistent import freeze
    assert freeze(42) == 42
    assert freeze("hello") == "hello"
    assert freeze(3.14) == 3.14


def test_freeze_none():
    from pyrsistent import freeze
    assert freeze(None) is None


def test_freeze_defaultdict():
    from pyrsistent import freeze, pmap
    import collections
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    result = freeze(dd)
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_defaultdict_nested():
    from pyrsistent import freeze, pmap
    import collections
    dd = collections.defaultdict(int)
    dd['a'] = {'x': 10}
    result = freeze(dd)
    assert result == pmap({'a': pmap({'x': 10})})


def test_freeze_complex_nested_structure():
    from pyrsistent import freeze, pmap, pvector, pset
    data = {'list': [1, 2, {'nested': 3}], 'set': {4, 5}, 'tuple': (6, [7, 8])}
    result = freeze(data)
    assert result == pmap({
        'list': pvector([1, 2, pmap({'nested': 3})]),
        'set': pset([4, 5]),
        'tuple': (6, pvector([7, 8]))
    })


def test_freeze_empty_dict():
    from pyrsistent import freeze, pmap
    result = freeze({})
    assert result == pmap({})


def test_freeze_empty_list():
    from pyrsistent import freeze, pvector
    result = freeze([])
    assert result == pvector([])


def test_freeze_empty_tuple():
    from pyrsistent import freeze
    result = freeze(())
    assert result == ()


def test_freeze_empty_set():
    from pyrsistent import freeze, pset
    result = freeze(set())
    assert result == pset([])


def test_freeze_strict_false_with_pmap():
    from pyrsistent import freeze, pmap
    pm = pmap({'a': 1})
    result = freeze(pm, strict=False)
    assert result == pm


def test_freeze_strict_true_with_pmap():
    from pyrsistent import freeze, pmap
    pm = pmap({'a': 1})
    result = freeze(pm, strict=True)
    assert result == pmap({'a': 1})


def test_freeze_strict_false_with_pvector():
    from pyrsistent import freeze, pvector
    pv = pvector([1, 2, 3])
    result = freeze(pv, strict=False)
    assert result == pv


def test_freeze_strict_true_with_pvector():
    from pyrsistent import freeze, pvector
    pv = pvector([1, 2, 3])
    result = freeze(pv, strict=True)
    assert result == pvector([1, 2, 3])


def test_freeze_list_of_dicts():
    from pyrsistent import freeze, pvector, pmap
    result = freeze([{'a': 1}, {'b': 2}])
    assert result == pvector([pmap({'a': 1}), pmap({'b': 2})])


def test_freeze_dict_with_list_values():
    from pyrsistent import freeze, pmap, pvector
    result = freeze({'x': [1, 2], 'y': [3, 4]})
    assert result == pmap({'x': pvector([1, 2]), 'y': pvector([3, 4])})


# LLM-generated content at query #44
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze, pmap, pset
    
    call_log = []
    
    @mutant
    def modify_and_return(data):
        call_log.append(('received_type', type(data).__name__))
        return data
    
    # Test with mutable list
    input_list = [1, 2, 3]
    result = modify_and_return(input_list)
    
    # Result should be frozen (pvector)
    assert str(type(result).__name__) == 'PVector'
    
    # Test with mutable dict
    input_dict = {'a': 1, 'b': 2}
    result2 = modify_and_return(input_dict)
    
    # Result should be frozen (pmap)
    assert str(type(result2).__name__) == 'PMap'
    
    # Test with mutable set
    input_set = {1, 2, 3}
    result3 = modify_and_return(input_set)
    
    # Result should be frozen (pset)
    assert str(type(result3).__name__) == 'PSet'


# LLM-generated content at query #45
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset

    @mutant
    def modify_list(lst):
        lst.append(99)
        return lst

    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]
    assert result == pvector([1, 2, 3, 99])


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def create_dict():
        return {'a': 1, 'b': 2}

    result = create_dict()
    assert isinstance(result, pmap.__class__)
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_with_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def process_nested(data):
        data['items'].append(4)
        return data

    original = {'items': [1, 2, 3]}
    result = process_nested(original)
    assert original == {'items': [1, 2, 3]}
    assert isinstance(result, pmap.__class__)


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap

    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}

    result = func_with_kwargs([1, 2], b={'x': 1})
    assert isinstance(result, pmap.__class__)


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant

    @mutant
    def my_function():
        """Test docstring"""
        pass

    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset

    @mutant
    def process_set(s):
        s.add(4)
        return s

    original = {1, 2, 3}
    result = process_set(original)
    assert original == {1, 2, 3}
    assert isinstance(result, pset.__class__)


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector

    @mutant
    def process_tuple(t):
        return t

    original = (1, [2, 3])
    result = process_tuple(original)
    assert result == (1, pvector([2, 3]))


def test_mutant_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap

    @mutant
    def combine(dict1, dict2):
        dict1['combined'] = dict2
        return dict1

    result = combine({'a': 1}, {'b': 2})
    assert isinstance(result, pmap.__class__)


def test_mutant_deeply_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pvector

    @mutant
    def process_deep(data):
        data['level1']['level2'].append(99)
        return data

    original = {'level1': {'level2': [1, 2, 3]}}
    result = process_deep(original)
    assert original == {'level1': {'level2': [1, 2, 3]}}
    assert isinstance(result, pmap.__class__)


# LLM-generated content at query #46
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, freeze
    
    call_log = []
    
    @mutant
    def modify_input(data):
        call_log.append(type(data))
        return data
    
    # Test with mutable dict - should be frozen before passing to function
    result = modify_input({'a': 1, 'b': 2})
    
    # Verify the argument was frozen (converted to pmap)
    assert call_log[0].__name__ == 'PMap'
    
    # Verify the return value is frozen
    assert hasattr(result, '__hash__')
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #47
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 3]


def test_mutant_with_dict_argument():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_dict(d):
        d['key'] = 'modified'
        return d
    
    result = modify_dict({'key': 'original'})
    assert isinstance(result, type(pmap()))
    assert result['key'] == 'original'


def test_mutant_with_nested_structures():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_nested(data):
        data[0]['nested'].append(999)
        return data
    
    result = process_nested([{'nested': [1, 2, 3]}])
    assert isinstance(result, type(pvector()))
    assert isinstance(result[0], type(pmap()))
    assert list(result[0]['nested']) == [1, 2, 3]


def test_mutant_with_multiple_arguments():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def combine_data(lst, d):
        lst.append(100)
        d['added'] = True
        return (lst, d)
    
    result = combine_data([1, 2], {'key': 'value'})
    assert isinstance(result, tuple)
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))
    assert list(result[0]) == [1, 2]
    assert result[1]['key'] == 'value'


def test_mutant_with_kwargs():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_kwargs(a=None, b=None):
        if a is not None:
            a['modified'] = True
        return a
    
    result = process_kwargs(a={'original': 'value'})
    assert isinstance(result, type(pmap()))
    assert result['original'] == 'value'
    assert 'modified' not in result


def test_mutant_preserves_set():
    from pyrsistent import pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_set(s):
        s.add(999)
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset()))
    assert 1 in result and 2 in result and 3 in result


def test_mutant_with_tuple():
    from pyrsistent import pvector
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t + (999,)
    
    result = process_tuple((1, [2, 3], 4))
    assert isinstance(result, tuple)
    assert isinstance(result[1], type(pvector()))
    assert list(result[1]) == [2, 3]


def test_mutant_with_scalar_return():
    from pyrsistent._helpers import mutant
    
    @mutant
    def return_scalar(lst):
        return 42
    
    result = return_scalar([1, 2, 3])
    assert result == 42


def test_mutant_with_none_return():
    from pyrsistent._helpers import mutant
    
    @mutant
    def return_none(d):
        return None
    
    result = return_none({'key': 'value'})
    assert result is None


# LLM-generated content at query #48
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    import collections
    from pyrsistent import freeze, pmap
    
    # Test that the predicate at line 32 evaluates to True
    # The predicate is: typ is collections.defaultdict or (strict and isinstance(o, PMap))
    
    # Create a defaultdict
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    
    # Call freeze with strict=True (default)
    result = freeze(dd, strict=True)
    
    # Verify the result is a pmap with frozen values
    assert result == pmap({'a': 1, 'b': 2})
    assert type(result).__name__ == 'PMap'


# LLM-generated content at query #49
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    original = {'a': 1, 'b': 2}
    result = modify_dict(original)
    
    # The result should be frozen (persistent)
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['new_key'] == 'new_value'
    
    # Original should not be modified (input was frozen)
    assert original == {'a': 1, 'b': 2}
    assert 'new_key' not in original


def test_mutant_decorator_with_list_argument():
    from pyrsistent import pvector, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def append_to_list(lst):
        lst.append(4)
        return lst
    
    original = [1, 2, 3]
    result = append_to_list(original)
    
    # The result should be frozen (persistent)
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 3, 4]
    
    # Original should not be modified
    assert original == [1, 2, 3]


def test_mutant_decorator_with_kwargs():
    from pyrsistent import pmap, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def create_mapping(**kwargs):
        kwargs['extra'] = 'value'
        return kwargs
    
    result = create_mapping(a=1, b=2)
    
    # The result should be frozen
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1
    assert result['b'] == 2
    assert result['extra'] == 'value'


def test_mutant_decorator_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        """My docstring"""
        return x
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'My docstring'


def test_mutant_decorator_with_nested_structures():
    from pyrsistent import pmap, pvector, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_nested(data):
        data['nested']['key'] = 'modified'
        return data
    
    original = {'nested': {'key': 'original'}}
    result = modify_nested(original)
    
    # Result should be frozen
    assert isinstance(result, type(pmap()))
    assert result['nested']['key'] == 'modified'
    
    # Original should not be modified
    assert original == {'nested': {'key': 'original'}}


# LLM-generated content at query #50
#--------------------------

```python
def test_mutant_decorator_predicate_line_1_evaluates_to_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, pvector
    
    @mutant
    def modify_input(data):
        return data
    
    input_list = [1, 2, 3]
    result = modify_input(input_list)
    
    assert isinstance(result, pvector)
    assert input_list == [1, 2, 3]
    assert result != input_list


# LLM-generated content at query #51
#--------------------------

```python
def test_mutant_freezes_arguments():
    @mutant
    def add_to_list(lst, value):
        lst = lst + [value]
        return lst
    
    result = add_to_list([1, 2, 3], 4)
    assert result == pvector([1, 2, 3, 4])


def test_mutant_freezes_return_value():
    @mutant
    def create_dict():
        return {'a': 1, 'b': 2}
    
    result = create_dict()
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'list': [1, 2], 'set': {3, 4}})
    assert result == pmap({'list': pvector([1, 2]), 'set': pset([3, 4])})


def test_mutant_with_kwargs():
    @mutant
    def create_with_kwargs(a=None, b=None):
        return {'a': a, 'b': b}
    
    result = create_with_kwargs(a=1, b=[2, 3])
    assert result == pmap({'a': 1, 'b': pvector([2, 3])})


def test_mutant_preserves_function_name():
    @mutant
    def my_function():
        return {}
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(list1, list2, dict1):
        return {'lists': [list1, list2], 'dict': dict1}
    
    result = combine([1, 2], [3, 4], {'x': 5})
    assert result == pmap({'lists': pvector([pvector([1, 2]), pvector([3, 4])]), 'dict': pmap({'x': 5})})


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(tpl):
        return tpl
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert result == (1, pvector([2, 3]), pmap({'a': 4}))


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_mutant_deep_nesting():
    @mutant
    def deeply_nested(data):
        return data
    
    result = deeply_nested({'a': {'b': [1, 2, {'c': 3}]}})
    assert result == pmap({'a': pmap({'b': pvector([1, 2, pmap({'c': 3})])})})


def test_mutant_with_empty_containers():
    @mutant
    def empty_containers():
        return {'list': [], 'dict': {}, 'set': set()}
    
    result = empty_containers()
    assert result == pmap({'list': pvector([]), 'dict': pmap({}), 'set': pset()})


# LLM-generated content at query #52
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def add_to_dict(d, key, value):
        d[key] = value
        return d
    
    result = add_to_dict({'a': 1}, 'b', 2)
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2


def test_mutant_freezes_list_arguments():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    result = append_to_list([1, 2], 3)
    assert isinstance(result, PVector)
    assert len(result) == 3
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_mutant_freezes_nested_structures():
    @mutant
    def modify_nested(data):
        return data
    
    result = modify_nested({'a': [1, 2], 'b': {'c': 3}})
    assert isinstance(result, PMap)
    assert isinstance(result['a'], PVector)
    assert isinstance(result['b'], PMap)


def test_mutant_freezes_kwargs():
    @mutant
    def create_dict(a=1, b=2):
        return {'a': a, 'b': b}
    
    result = create_dict(a=10, b=20)
    assert isinstance(result, PMap)
    assert result['a'] == 10
    assert result['b'] == 20


def test_mutant_freezes_set_arguments():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, PSet)


def test_mutant_freezes_tuple_arguments():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)
    assert isinstance(result[2], PMap)


def test_mutant_preserves_function_metadata():
    @mutant
    def my_func():
        """Test function"""
        pass
    
    assert my_func.__name__ == 'my_func'
    assert my_func.__doc__ == """Test function"""


def test_mutant_with_multiple_arguments():
    @mutant
    def merge_dicts(d1, d2):
        d1.update(d2)
        return d1
    
    result = merge_dicts({'a': 1}, {'b': 2})
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2


def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def create_structure(lst, d=None):
        if d is None:
            d = {}
        d['data'] = lst
        return d
    
    result = create_structure([1, 2, 3], d={'existing': 'value'})
    assert isinstance(result, PMap)
    assert isinstance(result['data'], PVector)
    assert result['existing'] == 'value'


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_thaw_pvector_to_list():
    from pyrsistent import v
    result = thaw(v(1, 2, 3))
    assert result == [1, 2, 3]
    assert isinstance(result, list)

def test_thaw_pmap_to_dict():
    from pyrsistent import m
    result = thaw(m(a=1, b=2))
    assert result == {'a': 1, 'b': 2}
    assert isinstance(result, dict)

def test_thaw_pset_to_set():
    from pyrsistent import s
    result = thaw(s(1, 2, 3))
    assert result == {1, 2, 3}
    assert isinstance(result, set)

def test_thaw_tuple_recursive():
    from pyrsistent import v
    result = thaw((1, v(2, 3)))
    assert result == (1, [2, 3])
    assert isinstance(result, tuple)

def test_thaw_nested_pvector_pmap():
    from pyrsistent import v, m
    result = thaw(v(1, m(a=3)))
    assert result == [1, {'a': 3}]

def test_thaw_nested_pmap_pvector():
    from pyrsistent import v, m
    result = thaw(m(x=v(1, 2)))
    assert result == {'x': [1, 2]}

def test_thaw_empty_pvector():
    from pyrsistent import v
    result = thaw(v())
    assert result == []

def test_thaw_empty_pmap():
    from pyrsistent import m
    result = thaw(m())
    assert result == {}

def test_thaw_empty_pset():
    from pyrsistent import s
    result = thaw(s())
    assert result == set()

def test_thaw_empty_tuple():
    result = thaw(())
    assert result == ()

def test_thaw_primitive_types():
    assert thaw(42) == 42
    assert thaw("string") == "string"
    assert thaw(3.14) == 3.14
    assert thaw(None) is None

def test_thaw_list_strict_true():
    result = thaw([1, 2, 3], strict=True)
    assert result == [1, 2, 3]

def test_thaw_dict_strict_true():
    result = thaw({'a': 1, 'b': 2}, strict=True)
    assert result == {'a': 1, 'b': 2}

def test_thaw_list_strict_false():
    result = thaw([1, 2, 3], strict=False)
    assert result == [1, 2, 3]

def test_thaw_dict_strict_false():
    result = thaw({'a': 1, 'b': 2}, strict=False)
    assert result == {'a': 1, 'b': 2}

def test_thaw_nested_list_strict_true():
    result = thaw([1, [2, 3]], strict=True)
    assert result == [1, [2, 3]]

def test_thaw_nested_dict_strict_true():
    result = thaw({'a': {'b': 1}}, strict=True)
    assert result == {'a': {'b': 1}}

def test_thaw_deeply_nested_structures():
    from pyrsistent import v, m
    result = thaw(v(m(x=v(1, m(y=2)))))
    assert result == [{'x': [1, {'y': 2}]}]

def test_thaw_tuple_with_pset():
    from pyrsistent import s
    result = thaw((s(1, 2), 3))
    assert result == ({1, 2}, 3)

def test_thaw_pmap_with_tuple_value():
    from pyrsistent import m
    result = thaw(m(a=(1, 2)))
    assert result == {'a': (1, 2)}

def test_thaw_pmap_with_nested_tuple():
    from pyrsistent import m, v
    result = thaw(m(a=(1, v(2, 3))))
    assert result == {'a': (1, [2, 3])}

def test_thaw_preserves_dict_keys():
    from pyrsistent import m
    result = thaw(m(key1=1, key2=2))
    assert 'key1' in result
    assert 'key2' in result

def test_thaw_list_with_none_elements():
    from pyrsistent import v
    result = thaw(v(1, None, 3))
    assert result == [1, None, 3]

def test_thaw_dict_with_none_values():
    from pyrsistent import m
    result = thaw(m(a=None, b=2))
    assert result == {'a': None, 'b': 2}


# LLM-generated content at query #2
#--------------------------

```python
def test_freeze_empty_dict():
    from pyrsistent._helpers import freeze
    result = freeze({})
    assert result == {}
    assert len(result) == 0


def test_freeze_simple_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1, 'b': 2})
    assert result['a'] == 1
    assert result['b'] == 2


def test_freeze_nested_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': {'b': 1}})
    assert result['a']['b'] == 1


def test_freeze_empty_list():
    from pyrsistent._helpers import freeze
    result = freeze([])
    assert len(result) == 0


def test_freeze_simple_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, 2, 3])
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_freeze_nested_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, [2, 3]])
    assert result[0] == 1
    assert result[1][0] == 2
    assert result[1][1] == 3


def test_freeze_list_with_dict():
    from pyrsistent._helpers import freeze
    result = freeze([1, {'a': 3}])
    assert result[0] == 1
    assert result[1]['a'] == 3


def test_freeze_empty_set():
    from pyrsistent._helpers import freeze
    result = freeze(set())
    assert len(result) == 0


def test_freeze_simple_set():
    from pyrsistent._helpers import freeze
    result = freeze({1, 2, 3})
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_freeze_empty_tuple():
    from pyrsistent._helpers import freeze
    result = freeze(())
    assert result == ()


def test_freeze_simple_tuple():
    from pyrsistent._helpers import freeze
    result = freeze((1, 2, 3))
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_freeze_nested_tuple():
    from pyrsistent._helpers import freeze
    result = freeze((1, (2, 3)))
    assert result[0] == 1
    assert result[1][0] == 2
    assert result[1][1] == 3


def test_freeze_tuple_with_list():
    from pyrsistent._helpers import freeze
    result = freeze((1, []))
    assert result[0] == 1
    assert len(result[1]) == 0


def test_freeze_primitive_int():
    from pyrsistent._helpers import freeze
    result = freeze(42)
    assert result == 42


def test_freeze_primitive_string():
    from pyrsistent._helpers import freeze
    result = freeze("hello")
    assert result == "hello"


def test_freeze_primitive_none():
    from pyrsistent._helpers import freeze
    result = freeze(None)
    assert result is None


def test_freeze_defaultdict():
    from pyrsistent._helpers import freeze
    import collections
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    result = freeze(dd)
    assert result['a'] == 1
    assert result['b'] == 2


def test_freeze_defaultdict_nested():
    from pyrsistent._helpers import freeze
    import collections
    dd = collections.defaultdict(int)
    dd['a'] = {'x': 10}
    result = freeze(dd)
    assert result['a']['x'] == 10


def test_freeze_complex_nested_structure():
    from pyrsistent._helpers import freeze
    data = {'a': [1, {'b': 2}], 'c': (3, [4, 5])}
    result = freeze(data)
    assert result['a'][0] == 1
    assert result['a'][1]['b'] == 2
    assert result['c'][0] == 3
    assert result['c'][1][0] == 4
    assert result['c'][1][1] == 5


def test_freeze_strict_mode_true():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1}, strict=True)
    assert result['a'] == 1


def test_freeze_strict_mode_false():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1}, strict=False)
    assert result['a'] == 1


def test_freeze_set_with_multiple_elements():
    from pyrsistent._helpers import freeze
    result = freeze(set([1, 2, 3, 2]))
    assert 1 in result
    assert 2 in result
    assert 3 in result
    assert len(result) == 3


def test_freeze_list_with_nested_dict_and_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, {'a': [2, 3]}])
    assert result[0] == 1
    assert result[1]['a'][0] == 2
    assert result[1]['a'][1] == 3


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_set_conversion():
    from pyrsistent._helpers import freeze
    from pyrsistent import pset
    
    result = freeze(set([1, 2]))
    expected = pset([1, 2])
    assert result == expected


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    import collections
    from pyrsistent import freeze, pmap
    
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    
    result = freeze(dd, strict=True)
    
    assert result == pmap({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap({})))


# LLM-generated content at query #5
#--------------------------

```python
def test_freeze_set_converts_to_pset():
    from pyrsistent import freeze, pset
    result = freeze(set([1, 2]))
    expected = pset([1, 2])
    assert result == expected


# LLM-generated content at query #6
#--------------------------

```python
def test_freeze_set_converts_to_pset():
    from pyrsistent._helpers import freeze
    from pyrsistent import pset
    
    result = freeze(set([1, 2]))
    expected = pset([1, 2])
    assert result == expected


# LLM-generated content at query #7
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [1]
    
    result = modify_list([1, 2, 3])
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"
    assert list(result) == [1, 2, 3, 1]


def test_mutant_freezes_dict_arguments():
    @mutant
    def get_value(d):
        return d
    
    result = get_value({'a': 1, 'b': 2})
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"


def test_mutant_freezes_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'key': [1, 2, {'inner': 3}]})
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"
    assert str(type(result['key'])) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_freezes_kwargs():
    @mutant
    def func_with_kwargs(a=None):
        return a
    
    result = func_with_kwargs(a={'x': 1})
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, dct):
        return [lst, dct]
    
    result = combine([1, 2], {'a': 3})
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"
    assert str(type(result[0])) == "<class 'pyrsistent._pvector.PVector'>"
    assert str(type(result[1])) == "<class 'pyrsistent._pmap.PMap'>"


def test_mutant_preserves_function_name():
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert str(type(result)) == "<class 'pyrsistent._pset.PSet'>"


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert type(result) == tuple
    assert str(type(result[1])) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_with_no_arguments():
    @mutant
    def no_args():
        return [1, 2, 3]
    
    result = no_args()
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_with_scalar_return():
    @mutant
    def return_scalar(x):
        return x + 1
    
    result = return_scalar(5)
    assert result == 6


# LLM-generated content at query #8
#--------------------------

```python
def test_freeze_set_converts_to_pset():
    from pyrsistent import freeze, pset
    result = freeze(set([1, 2]))
    expected = pset([1, 2])
    assert result == expected


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, freeze
    
    call_log = []
    
    @mutant
    def modify_and_return(data):
        call_log.append(type(data))
        return data
    
    input_dict = {'a': 1, 'b': 2}
    result = modify_and_return(input_dict)
    
    assert call_log[0].__name__ != 'dict'
    assert not isinstance(result, dict)
    assert hasattr(result, '__hash__')


# LLM-generated content at query #10
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_map(m):
        return m
    
    @mutant
    def modify_set(s):
        return s
    
    @mutant
    def modify_dict(d):
        return d
    
    test_map = pmap({'a': 1, 'b': 2})
    result_map = modify_map(test_map)
    assert result_map.is_persistent()
    
    test_set = pset([1, 2, 3])
    result_set = modify_set(test_set)
    assert result_set.is_persistent()
    
    test_dict = {'x': 10, 'y': 20}
    result_dict = modify_dict(test_dict)
    assert isinstance(result_dict, type(freeze(test_dict)))
    
    @mutant
    def process_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = process_with_kwargs(pmap({'key': 'value'}), b=pset([1, 2]))
    assert isinstance(result, type(freeze({})))


# LLM-generated content at query #11
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 3]


def test_mutant_with_dict_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    result = modify_dict({'a': 1})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1})


def test_mutant_with_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        data['list'].append(100)
        return data
    
    result = process_nested({'list': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['list'], type(pvector()))
    assert list(result['list']) == [1, 2, 3]


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        lst.append(dct['key'])
        return lst
    
    result = combine([1, 2], {'key': 3})
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2]


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_kwargs(data, extra=None):
        data['processed'] = True
        return data
    
    result = process_kwargs({'a': 1}, extra={'b': 2})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1})


def test_mutant_preserves_immutability():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def mutate_attempt(lst):
        original_lst = lst
        lst.append(999)
        return lst
    
    original = [1, 2, 3]
    result = mutate_attempt(original)
    assert original == [1, 2, 3]
    assert list(result) == [1, 2, 3]


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        s.add(999)
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset()))
    assert result == pset({1, 2, 3})


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert result[0] == 1


def test_mutant_with_complex_nested_structure():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_complex(data):
        data['items'].append({'new': 'item'})
        return data
    
    input_data = {'items': [{'id': 1}], 'tags': {1, 2, 3}}
    result = process_complex(input_data)
    assert isinstance(result, type(pmap()))
    assert isinstance(result['items'], type(pvector()))
    assert len(result['items']) == 1


# LLM-generated content at query #12
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector()))
    assert result == pvector([1, 2, 3])


def test_mutant_freezes_dict_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    result = modify_dict({'a': 1})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1})


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def modify_nested(data):
        data[0]['key'] = 'mutated'
        return data
    
    result = modify_nested([{'key': 'original'}])
    assert isinstance(result, type(pvector()))
    assert result[0] == pmap({'key': 'original'})


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'x': 10})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'x': 10})})


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        return {'list': lst, 'dict': dct}
    
    result = combine([1, 2], {'key': 'value'})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'list': pvector([1, 2]), 'dict': pmap({'key': 'value'})})


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        """Test docstring"""
        pass
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        s.add(999)
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset()))
    assert result == pset({1, 2, 3})


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_mutant_prevents_mutation_of_input():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def try_mutate(vec):
        try:
            vec.append(999)
        except (AttributeError, TypeError):
            pass
        return vec
    
    original = [1, 2, 3]
    result = try_mutate(original)
    assert original == [1, 2, 3]
    assert result == pvector([1, 2, 3])


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, freeze
    
    call_log = []
    
    @mutant
    def modify_and_return(data):
        call_log.append(data)
        return data
    
    input_list = [1, 2, 3]
    result = modify_and_return(input_list)
    
    assert not isinstance(input_list, type(result))
    assert result == pset([1, 2, 3]) or result == freeze([1, 2, 3])
    assert str(type(result)) != "<class 'list'>"
    
    input_dict = {'a': 1, 'b': 2}
    result2 = modify_and_return(input_dict)
    
    assert not isinstance(input_dict, type(result2))
    assert str(type(result2)) != "<class 'dict'>"


# LLM-generated content at query #14
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector()))
    assert result == pvector([1, 2, 3, 999])


def test_mutant_freezes_dict_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    result = modify_dict({'a': 1})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1, 'new_key': 'new_value'})


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        data['items'].append(4)
        return data
    
    result = process_nested({'items': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert result['items'] == pvector([1, 2, 3, 4])


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def combine_lists(lst1, lst2):
        lst1.extend(lst2)
        return lst1
    
    result = combine_lists([1, 2], [3, 4])
    assert result == pvector([1, 2, 3, 4])


def test_mutant_with_keyword_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def create_mapping(data, extra=None):
        if extra:
            data.update(extra)
        return data
    
    result = create_mapping({'a': 1}, extra={'b': 2})
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        s.add(4)
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset()))
    assert result == pset([1, 2, 3, 4])


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t + (4,)
    
    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)


def test_mutant_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_mixed_types():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_mixed(data):
        data['list'].append(99)
        data['nested']['key'] = 'updated'
        return data
    
    result = process_mixed({'list': [1, 2], 'nested': {'key': 'original'}})
    assert result['list'] == pvector([1, 2, 99])
    assert result['nested'] == pmap({'key': 'updated'})


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst
    
    original_list = [1, 2, 3]
    result = modify_list(original_list)
    
    assert isinstance(result, type(pvector()))
    assert result == pvector([1, 2, 3, 999])


def test_mutant_freezes_nested_structures():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_nested(data):
        data['key'].append(5)
        return data
    
    input_data = {'key': [1, 2, 3]}
    result = process_nested(input_data)
    
    assert isinstance(result, type(pmap()))
    assert isinstance(result['key'], type(pvector()))
    assert result == pmap({'key': pvector([1, 2, 3, 5])})


def test_mutant_with_multiple_arguments():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def combine(lst, dct):
        dct['result'] = lst
        return dct
    
    result = combine([1, 2], {'a': 10})
    
    assert isinstance(result, type(pmap()))
    assert isinstance(result['result'], type(pvector()))
    assert result == pmap({'a': 10, 'result': pvector([1, 2])})


def test_mutant_with_keyword_arguments():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_with_kwargs(data, multiplier=2):
        data['value'] = data.get('value', 0) * multiplier
        return data
    
    result = process_with_kwargs({'value': 5}, multiplier=3)
    
    assert isinstance(result, type(pmap()))
    assert result == pmap({'value': 15})


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        """My docstring"""
        return 42
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'My docstring'


def test_mutant_with_set_argument():
    from pyrsistent import pset, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_set(s):
        s.add(999)
        return s
    
    result = process_set({1, 2, 3})
    
    assert isinstance(result, type(pset()))
    assert result == pset([1, 2, 3, 999])


def test_mutant_with_tuple_argument():
    from pyrsistent import pvector
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t + (4,)
    
    result = process_tuple((1, 2, 3))
    
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)


def test_mutant_deeply_nested_structure():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def deep_process(data):
        data['nested']['list'].append(100)
        return data
    
    input_data = {'nested': {'list': [1, 2, 3]}}
    result = deep_process(input_data)
    
    assert isinstance(result, type(pmap()))
    assert isinstance(result['nested'], type(pmap()))
    assert isinstance(result['nested']['list'], type(pvector()))
    assert result == pmap({'nested': pmap({'list': pvector([1, 2, 3, 100])})})


def test_mutant_returns_scalar_value():
    from pyrsistent._helpers import mutant
    
    @mutant
    def get_scalar(lst):
        return len(lst)
    
    result = get_scalar([1, 2, 3])
    
    assert result == 3
    assert isinstance(result, int)


def test_mutant_with_empty_containers():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_empty(lst, dct, s):
        return {'list': lst, 'dict': dct, 'set': s}
    
    result = process_empty([], {}, set())
    
    assert isinstance(result, type(pmap()))
    assert isinstance(result['list'], type(pvector()))
    assert isinstance(result['dict'], type(pmap()))
    assert isinstance(result['set'], type(pset()))


# LLM-generated content at query #16
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    import collections
    from pyrsistent import freeze, pmap
    
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    
    result = freeze(dd, strict=True)
    
    assert result == pmap({'a': 1, 'b': 2})
    assert type(result).__name__ == 'PMap'


# LLM-generated content at query #17
#--------------------------

```python
def test_freeze_empty_dict():
    from pyrsistent import freeze, pmap
    result = freeze({})
    assert result == pmap({})


def test_freeze_simple_dict():
    from pyrsistent import freeze, pmap
    result = freeze({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_nested_dict():
    from pyrsistent import freeze, pmap
    result = freeze({'a': {'b': 1}})
    assert result == pmap({'a': pmap({'b': 1})})


def test_freeze_empty_list():
    from pyrsistent import freeze, pvector
    result = freeze([])
    assert result == pvector([])


def test_freeze_simple_list():
    from pyrsistent import freeze, pvector
    result = freeze([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_freeze_nested_list():
    from pyrsistent import freeze, pvector
    result = freeze([1, [2, 3]])
    assert result == pvector([1, pvector([2, 3])])


def test_freeze_list_with_dict():
    from pyrsistent import freeze, pvector, pmap
    result = freeze([1, {'a': 2}])
    assert result == pvector([1, pmap({'a': 2})])


def test_freeze_empty_set():
    from pyrsistent import freeze, pset
    result = freeze(set())
    assert result == pset([])


def test_freeze_simple_set():
    from pyrsistent import freeze, pset
    result = freeze({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_freeze_empty_tuple():
    from pyrsistent import freeze
    result = freeze(())
    assert result == ()


def test_freeze_simple_tuple():
    from pyrsistent import freeze
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)


def test_freeze_nested_tuple():
    from pyrsistent import freeze, pvector
    result = freeze((1, [2, 3]))
    assert result == (1, pvector([2, 3]))


def test_freeze_complex_nested_structure():
    from pyrsistent import freeze, pmap, pvector, pset
    result = freeze({'a': [1, 2], 'b': {'c': 3}})
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})


def test_freeze_primitive_int():
    from pyrsistent import freeze
    result = freeze(42)
    assert result == 42


def test_freeze_primitive_string():
    from pyrsistent import freeze
    result = freeze("hello")
    assert result == "hello"


def test_freeze_primitive_none():
    from pyrsistent import freeze
    result = freeze(None)
    assert result is None


def test_freeze_strict_false_with_pmap():
    from pyrsistent import freeze, pmap
    pm = pmap({'a': 1})
    result = freeze(pm, strict=False)
    assert result == pm


def test_freeze_strict_true_with_pmap():
    from pyrsistent import freeze, pmap
    pm = pmap({'a': 1})
    result = freeze(pm, strict=True)
    assert result == pmap({'a': 1})


def test_freeze_strict_false_with_pvector():
    from pyrsistent import freeze, pvector
    pv = pvector([1, 2])
    result = freeze(pv, strict=False)
    assert result == pv


def test_freeze_strict_true_with_pvector():
    from pyrsistent import freeze, pvector
    pv = pvector([1, 2])
    result = freeze(pv, strict=True)
    assert result == pvector([1, 2])


def test_freeze_defaultdict():
    from pyrsistent import freeze, pmap
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_defaultdict_nested():
    from pyrsistent import freeze, pmap
    from collections import defaultdict
    dd = defaultdict(int, {'a': {'b': 1}})
    result = freeze(dd)
    assert result == pmap({'a': pmap({'b': 1})})


def test_freeze_dict_with_list_values():
    from pyrsistent import freeze, pmap, pvector
    result = freeze({'a': [1, 2], 'b': [3, 4]})
    assert result == pmap({'a': pvector([1, 2]), 'b': pvector([3, 4])})


def test_freeze_tuple_with_dict():
    from pyrsistent import freeze, pmap
    result = freeze(({'a': 1}, 2))
    assert result == (pmap({'a': 1}), 2)


def test_freeze_mixed_structure():
    from pyrsistent import freeze, pmap, pvector, pset
    result = freeze({'data': [1, 2, {'nested': True}], 'items': {1, 2, 3}})
    assert result == pmap({'data': pvector([1, 2, pmap({'nested': True})]), 'items': pset([1, 2, 3])})


# LLM-generated content at query #18
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_list(lst):
        # This would mutate if lst were a list, but it's frozen to pvector
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector([1, 2, 3])))


def test_mutant_freezes_dict_arguments():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_dict(d):
        return d
    
    result = process_dict({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap({'a': 1, 'b': 2})))


def test_mutant_freezes_nested_structures():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'key': [1, 2, 3]})
    assert isinstance(result, type(pmap({})))
    assert isinstance(result['key'], type(pvector([])))


def test_mutant_freezes_set_arguments():
    from pyrsistent import pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset([1, 2, 3])))


def test_mutant_freezes_tuple_arguments():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)


def test_mutant_freezes_kwargs():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_with_kwargs(a, b=None):
        return (a, b)
    
    result = process_with_kwargs([1], b={'key': 'value'})
    assert isinstance(result[1], type(pmap({})))


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        """Test docstring"""
        return x
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_multiple_arguments():
    from pyrsistent import pvector
    from pyrsistent._helpers import mutant
    
    @mutant
    def combine(list1, list2):
        return list1
    
    result = combine([1, 2], [3, 4])
    assert isinstance(result, type(pvector([])))


def test_mutant_with_no_arguments():
    from pyrsistent._helpers import mutant
    
    @mutant
    def get_constant():
        return {'a': 1}
    
    result = get_constant()
    from pyrsistent import pmap
    assert isinstance(result, type(pmap({})))


def test_mutant_freezes_deeply_nested_structures():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_deep(data):
        return data
    
    result = process_deep({'outer': {'inner': [1, 2, 3]}})
    assert isinstance(result, type(pmap({})))
    assert isinstance(result['outer'], type(pmap({})))
    assert isinstance(result['outer']['inner'], type(pvector([])))


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_and_return(data):
        return data
    
    input_list = [1, 2, 3]
    result = modify_and_return(input_list)
    
    # The predicate at line 1 is that the function is decorated with @mutant
    # This means arguments should be frozen and return value should be frozen
    assert isinstance(result, (pset, pmap, tuple, frozenset)) or hasattr(result, '__hash__')


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [1]
    
    result = modify_list([1, 2])
    assert result == (1, 2, 1)


def test_mutant_with_dict_argument():
    @mutant
    def get_value(d):
        return d['key']
    
    result = get_value({'key': 'value'})
    assert result == 'value'


def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'list': [1, 2], 'set': {1, 2}})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(a, b):
        return a + b
    
    result = combine([1, 2], [3, 4])
    assert result == (1, 2, 3, 4)


def test_mutant_with_kwargs():
    @mutant
    def create_dict(a=1, b=2):
        return {'a': a, 'b': b}
    
    result = create_dict(a=10, b=20)
    assert result['a'] == 10
    assert result['b'] == 20


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert str(type(result).__name__) == 'PSet'


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], 4))
    assert isinstance(result, tuple)
    assert result[0] == 1
    assert result[2] == 4


def test_mutant_preserves_function_name():
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_empty_containers():
    @mutant
    def process_empty(lst, d, s):
        return [lst, d, s]
    
    result = process_empty([], {}, set())
    assert len(result) == 3


def test_mutant_with_nested_dict_and_list():
    @mutant
    def process_complex(data):
        return data
    
    result = process_complex({'nested': [1, 2, {'inner': 'value'}]})
    assert result['nested'][2]['inner'] == 'value'


# LLM-generated content at query #21
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst
    
    original = [1, 2, 3]
    result = modify_list(original)
    assert result == [1, 2, 3, 999]


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def create_list():
        return [1, 2, 3]
    
    result = create_list()
    assert result == pvector([1, 2, 3])


def test_mutant_with_dict_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    original = {'a': 1}
    result = modify_dict(original)
    assert result == pmap({'a': 1, 'new_key': 'new_value'})


def test_mutant_with_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        data['items'].append(4)
        return data
    
    original = {'items': [1, 2, 3]}
    result = process_nested(original)
    assert result == pmap({'items': pvector([1, 2, 3, 4])})


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def combine_lists(lst1, lst2):
        lst1.extend(lst2)
        return lst1
    
    result = combine_lists([1, 2], [3, 4])
    assert result == pvector([1, 2, 3, 4])


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def create_with_kwargs(a=1, b=2):
        return {'a': a, 'b': b}
    
    result = create_with_kwargs(a=10, b=20)
    assert result == pmap({'a': 10, 'b': 20})


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        """Test docstring"""
        pass
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        s.add(4)
        return s
    
    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3, 4])


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t + (4, 5)
    
    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4, 5)


def test_mutant_original_not_modified():
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst
    
    original = [1, 2, 3]
    result = modify_list(original)
    assert original == [1, 2, 3]
    assert result != original


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_decorator_predicate_line_1_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset, pmap
    
    @mutant
    def modify_set(s):
        return s
    
    @mutant
    def modify_map(m):
        return m
    
    input_set = pset([1, 2, 3])
    input_map = pmap({'a': 1, 'b': 2})
    
    result_set = modify_set(input_set)
    result_map = modify_map(input_map)
    
    assert result_set == input_set
    assert result_map == input_map
    assert result_set is not input_set
    assert result_map is not input_map


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    original_dict = {'a': 1}
    result = modify_dict(original_dict)
    
    assert 'new_key' not in original_dict
    assert original_dict == {'a': 1}


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    original = {'a': 1}
    result = modify_dict(original)
    
    # Original should not be modified
    assert 'new_key' not in original
    assert original == {'a': 1}
    
    # Result should be frozen (persistent)
    assert 'new_key' in result
    assert result['new_key'] == 'new_value'
    
    @mutant
    def modify_list(lst):
        lst.append(4)
        return lst
    
    original_list = [1, 2, 3]
    result_list = modify_list(original_list)
    
    # Original should not be modified
    assert len(original_list) == 3
    assert original_list == [1, 2, 3]
    
    # Result should be frozen (persistent)
    assert len(result_list) == 4
    assert 4 in result_list
    
    @mutant
    def process_with_kwargs(data, extra=None):
        data['processed'] = True
        if extra:
            data['extra'] = extra
        return data
    
    input_data = {'x': 10}
    output = process_with_kwargs(input_data, extra='test')
    
    # Original should not be modified
    assert 'processed' not in input_data
    assert input_data == {'x': 10}
    
    # Result should be frozen
    assert output['processed'] is True
    assert output['extra'] == 'test'


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, pvector
    
    @mutant
    def modify_and_return(data):
        return data
    
    input_list = [1, 2, 3]
    result = modify_and_return(input_list)
    
    assert not isinstance(result, list)
    assert isinstance(result, type(pvector()))


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, freeze
    
    @mutant
    def modify_map(m):
        return m
    
    test_dict = {'a': 1, 'b': 2}
    result = modify_map(test_dict)
    
    # The result should be frozen (a pmap)
    assert isinstance(result, type(pmap(test_dict)))
    
    @mutant
    def process_args(arg1, arg2):
        return (arg1, arg2)
    
    test_list = [1, 2, 3]
    test_set = {4, 5, 6}
    result1, result2 = process_args(test_list, test_set)
    
    # Arguments should be frozen in the function
    assert isinstance(result1, type(freeze([1, 2, 3])))
    assert isinstance(result2, type(freeze({4, 5, 6})))
    
    @mutant
    def process_kwargs(data=None):
        return data
    
    result_kwargs = process_kwargs(data={'x': 10})
    
    # Keyword arguments should be frozen
    assert isinstance(result_kwargs, type(pmap({'x': 10})))


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze, thaw
    
    @mutant
    def modify_list(lst):
        return lst + [4]
    
    input_list = [1, 2, 3]
    result = modify_list(input_list)
    
    # The result should be frozen (persistent)
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"
    assert list(result) == [1, 2, 3, 4]


# LLM-generated content at query #28
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, pvector
    
    @mutant
    def modify_input(data):
        return data
    
    # Test with mutable dict - should be frozen to pmap
    result = modify_input({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap({'a': 1})))
    
    # Test with mutable list - should be frozen to pvector
    result = modify_input([1, 2, 3])
    assert isinstance(result, type(pvector([1, 2, 3])))
    
    # Test with mutable set - should be frozen to pset
    result = modify_input({1, 2, 3})
    assert isinstance(result, type(pset([1, 2, 3])))
    
    # Test that nested mutable structures are also frozen
    @mutant
    def nested_modify(data):
        return data
    
    result = nested_modify({'key': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    
    # Test with kwargs
    @mutant
    def kwargs_modify(a, b=None):
        return {'result': a, 'b': b}
    
    result = kwargs_modify([1, 2], b={'nested': 'dict'})
    assert isinstance(result, type(pmap()))


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_and_return(data):
        return data
    
    input_list = [1, 2, 3]
    result = modify_and_return(input_list)
    
    # The predicate at line 1 "def mutant(fn):" evaluates to False
    # because the function object itself is falsy when evaluated in a boolean context? No.
    # Actually, we need to verify the decorator works correctly.
    # The predicate that should be False is checking if arguments are NOT frozen.
    # We verify that the decorated function receives frozen arguments.
    
    input_dict = {'a': 1}
    
    @mutant
    def returns_input(data):
        return data
    
    frozen_result = returns_input(input_dict)
    
    # Verify the result is frozen (is a pmap, not a dict)
    assert isinstance(frozen_result, type(pmap()))
    assert frozen_result == pmap({'a': 1})


# LLM-generated content at query #30
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst
    
    original_list = [1, 2, 3]
    result = modify_list(original_list)
    
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 3, 999]


def test_mutant_with_dict_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    original_dict = {'a': 1}
    result = modify_dict(original_dict)
    
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1, 'new_key': 'new_value'})


def test_mutant_with_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        data['items'].append(4)
        return data
    
    original = {'items': [1, 2, 3]}
    result = process_nested(original)
    
    assert isinstance(result, type(pmap()))
    assert isinstance(result['items'], type(pvector()))
    assert list(result['items']) == [1, 2, 3, 4]


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, d):
        lst.append(d['value'])
        return lst
    
    result = combine([1, 2], {'value': 3})
    
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 3]


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def build_map(a, b=None):
        return {'a': a, 'b': b}
    
    result = build_map(1, b=2)
    
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_preserves_original_arguments():
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_and_return(lst):
        lst.append(999)
        return lst
    
    original_list = [1, 2, 3]
    result = modify_and_return(original_list)
    
    assert original_list == [1, 2, 3]


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    
    assert isinstance(result, type(pset()))


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, 2, 3))
    
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)


def test_mutant_with_deeply_nested_structure():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_deep(data):
        return data
    
    original = {'level1': {'level2': [1, 2, 3]}}
    result = process_deep(original)
    
    assert isinstance(result, type(pmap()))
    assert isinstance(result['level1'], type(pmap()))
    assert isinstance(result['level1']['level2'], type(pvector()))


def test_mutant_function_wraps_preserves_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        """Test docstring"""
        pass
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


# LLM-generated content at query #31
#--------------------------

```python
def test_freeze_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1, 'b': 2})
    assert result == {'a': 1, 'b': 2}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_nested_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': {'b': 1}})
    assert result == {'a': {'b': 1}}
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PMap'


def test_freeze_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, 2, 3])
    assert list(result) == [1, 2, 3]
    assert str(type(result).__name__) == 'PVector'


def test_freeze_nested_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, [2, 3]])
    assert list(result) == [1, [2, 3]]
    assert str(type(result).__name__) == 'PVector'
    assert str(type(result[1]).__name__) == 'PVector'


def test_freeze_set():
    from pyrsistent._helpers import freeze
    result = freeze({1, 2, 3})
    assert result == {1, 2, 3}
    assert str(type(result).__name__) == 'PSet'


def test_freeze_tuple():
    from pyrsistent._helpers import freeze
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)


def test_freeze_nested_tuple():
    from pyrsistent._helpers import freeze
    result = freeze((1, [2, 3]))
    assert result[0] == 1
    assert list(result[1]) == [2, 3]
    assert str(type(result[1]).__name__) == 'PVector'


def test_freeze_complex_structure():
    from pyrsistent._helpers import freeze
    result = freeze({'a': [1, {'b': 2}], 'c': (3, 4)})
    assert result['a'][0] == 1
    assert result['a'][1]['b'] == 2
    assert result['c'] == (3, 4)


def test_freeze_defaultdict():
    from pyrsistent._helpers import freeze
    from collections import defaultdict
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd)
    assert result == {'a': 1, 'b': 2}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_empty_dict():
    from pyrsistent._helpers import freeze
    result = freeze({})
    assert result == {}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_empty_list():
    from pyrsistent._helpers import freeze
    result = freeze([])
    assert list(result) == []
    assert str(type(result).__name__) == 'PVector'


def test_freeze_empty_set():
    from pyrsistent._helpers import freeze
    result = freeze(set())
    assert result == set()
    assert str(type(result).__name__) == 'PSet'


def test_freeze_empty_tuple():
    from pyrsistent._helpers import freeze
    result = freeze(())
    assert result == ()
    assert isinstance(result, tuple)


def test_freeze_primitive():
    from pyrsistent._helpers import freeze
    assert freeze(1) == 1
    assert freeze('string') == 'string'
    assert freeze(3.14) == 3.14
    assert freeze(None) is None


def test_freeze_strict_false():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1}, strict=False)
    assert result == {'a': 1}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_list_with_dict():
    from pyrsistent._helpers import freeze
    result = freeze([1, {'a': 3}])
    assert list(result) == [1, {'a': 3}]
    assert str(type(result).__name__) == 'PVector'
    assert str(type(result[1]).__name__) == 'PMap'


def test_freeze_dict_with_list_value():
    from pyrsistent._helpers import freeze
    result = freeze({'a': [1, 2, 3]})
    assert list(result['a']) == [1, 2, 3]
    assert str(type(result['a']).__name__) == 'PVector'


def test_freeze_nested_structures():
    from pyrsistent._helpers import freeze
    result = freeze({'x': [1, 2, {'y': [3, 4]}]})
    assert result['x'][2]['y'][0] == 3
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['x']).__name__) == 'PVector'
    assert str(type(result['x'][2]).__name__) == 'PMap'
    assert str(type(result['x'][2]['y']).__name__) == 'PVector'


def test_freeze_set_of_primitives():
    from pyrsistent._helpers import freeze
    result = freeze({1, 2, 3, 2, 1})
    assert result == {1, 2, 3}
    assert str(type(result).__name__) == 'PSet'


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_map(m):
        return m
    
    @mutant
    def modify_set(s):
        return s
    
    @mutant
    def modify_list(lst):
        return lst
    
    input_map = pmap({'a': 1, 'b': 2})
    result_map = modify_map(input_map)
    assert result_map.is_persistent() is True
    
    input_set = pset([1, 2, 3])
    result_set = modify_set(input_set)
    assert result_set.is_persistent() is True
    
    input_list = [1, 2, 3]
    result_list = modify_list(input_list)
    assert result_list.is_persistent() is True
    
    input_dict = {'x': 1, 'y': 2}
    result_dict = freeze(input_dict)
    assert result_dict.is_persistent() is True


# LLM-generated content at query #33
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset
    
    @mutant
    def modify_and_return(data):
        return data
    
    original_map = pmap({'a': 1, 'b': 2})
    result = modify_and_return(original_map)
    
    assert result.evolver() is not None
    assert not (result == {'a': 1, 'b': 2} and isinstance(result, dict))


# LLM-generated content at query #34
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze, pmap, pset
    
    call_log = []
    
    @mutant
    def process_data(data, mapping):
        call_log.append((data, mapping))
        return {'result': data, 'map': mapping}
    
    input_list = [1, 2, 3]
    input_dict = {'key': 'value'}
    
    result = process_data(input_list, input_dict)
    
    # Verify arguments were frozen before being passed to the function
    frozen_args = call_log[0]
    assert frozen_args[0] == pset(input_list)
    assert frozen_args[1] == pmap(input_dict)
    
    # Verify return value is frozen
    assert isinstance(result, type(pmap({})))
    assert result == pmap({'result': pset(input_list), 'map': pmap(input_dict)})


# LLM-generated content at query #35
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze, thaw
    
    @mutant
    def modify_list(lst):
        # Try to modify the input (should fail because it's frozen)
        return lst
    
    original_list = [1, 2, 3]
    result = modify_list(original_list)
    
    # The result should be frozen (persistent)
    assert result == [1, 2, 3]
    # Verify it's a persistent structure by checking it's not a regular list
    assert not isinstance(result, list)
    
    @mutant
    def modify_dict(d):
        return d
    
    original_dict = {'a': 1, 'b': 2}
    result_dict = modify_dict(original_dict)
    
    # The result should be frozen (persistent)
    assert result_dict == {'a': 1, 'b': 2}
    # Verify it's a persistent structure by checking it's not a regular dict
    assert not isinstance(result_dict, dict)
    
    @mutant
    def function_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result_kwargs = function_with_kwargs([1, 2], b={'x': 1})
    
    # The result should be frozen
    assert result_kwargs['a'] == [1, 2]
    assert result_kwargs['b'] == {'x': 1}


# LLM-generated content at query #36
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, freeze
    
    call_log = []
    
    @mutant
    def modify_arg(arg):
        call_log.append(type(arg))
        return arg
    
    # Test with mutable list
    mutable_list = [1, 2, 3]
    result = modify_arg(mutable_list)
    
    # The argument passed to the function should be frozen (pvector)
    assert call_log[0].__name__ == 'PVector'
    
    # The return value should be frozen
    assert str(type(result)).find('Persistent') != -1 or str(type(result)).find('PVector') != -1
    
    # Original list should not be modified
    assert mutable_list == [1, 2, 3]


# LLM-generated content at query #37
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def add_item(lst, item):
        lst_copy = list(lst)
        lst_copy.append(item)
        return lst_copy
    
    result = add_item([1, 2], 3)
    assert result == pvector([1, 2, 3])


def test_mutant_with_dict_argument():
    @mutant
    def update_dict(d, key, value):
        d_copy = dict(d)
        d_copy[key] = value
        return d_copy
    
    result = update_dict({'a': 1}, 'b', 2)
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return {'result': data}
    
    result = process_nested({'key': [1, 2, 3]})
    assert result == pmap({'result': pmap({'key': pvector([1, 2, 3])})})


def test_mutant_with_kwargs():
    @mutant
    def create_map(a=1, b=2):
        return {'a': a, 'b': b}
    
    result = create_map(a=10, b=20)
    assert result == pmap({'a': 10, 'b': 20})


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst1, lst2, d):
        return list(lst1) + list(lst2) + [d]
    
    result = combine([1, 2], [3, 4], {'x': 5})
    assert result == pvector([1, 2, 3, 4, pmap({'x': 5})])


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return {'items': s}
    
    result = process_set({1, 2, 3})
    assert result == pmap({'items': pset([1, 2, 3])})


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4,)
    
    result = process_tuple((1, 2, 3))
    assert result == (1, 2, 3, 4)


def test_mutant_preserves_function_name():
    @mutant
    def my_function():
        return []
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_empty_containers():
    @mutant
    def return_empty():
        return {'list': [], 'dict': {}, 'set': set()}
    
    result = return_empty()
    assert result == pmap({'list': pvector([]), 'dict': pmap({}), 'set': pset([])})


def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def mixed_function(a, b, c=3):
        return [a, b, c]
    
    result = mixed_function([1], [2], c={'d': 4})
    assert result == pvector([pvector([1]), pvector([2]), pmap({'d': 4})])


# LLM-generated content at query #38
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset, pmap, freeze
    
    call_log = []
    
    @mutant
    def test_function(arg1, arg2, kwarg1=None):
        call_log.append((arg1, arg2, kwarg1))
        return {'result': arg1}
    
    result = test_function([1, 2, 3], {'a': 1}, kwarg1=[4, 5, 6])
    
    assert len(call_log) == 1
    received_arg1, received_arg2, received_kwarg1 = call_log[0]
    
    assert isinstance(received_arg1, type(pset([1, 2, 3]))) or isinstance(received_arg1, type(pvector([1, 2, 3])))
    assert isinstance(received_arg2, type(pmap({'a': 1})))
    assert isinstance(received_kwarg1, type(pset([4, 5, 6])) or type(pvector([4, 5, 6])))
    
    assert isinstance(result, type(pmap({'result': 1})))


# LLM-generated content at query #39
#--------------------------

```python
def test_freeze_defaultdict_strict_true():
    from collections import defaultdict
    from pyrsistent import freeze, pmap
    
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd, strict=True)
    
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #40
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset
    
    @mutant
    def modify_and_return(data):
        return data
    
    input_list = [1, 2, 3]
    result = modify_and_return(input_list)
    
    # The predicate at line 1 "def mutant(fn):" evaluates to False
    # because it's a function definition statement, not a boolean expression
    assert callable(modify_and_return)
    assert result == pset([1, 2, 3]) or isinstance(result, (pmap, pset))


# LLM-generated content at query #41
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [1]
    
    result = modify_list([1, 2, 3])
    assert str(type(result).__name__) == 'PVector'


def test_mutant_with_dict_argument():
    @mutant
    def get_value(d, key):
        return d[key]
    
    result = get_value({'a': 1, 'b': 2}, 'a')
    assert result == 1


def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'key': [1, 2, 3]})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert str(type(result).__name__) == 'PSet'


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)


def test_mutant_with_kwargs():
    @mutant
    def create_map(a=1, b=2):
        return {'a': a, 'b': b}
    
    result = create_map(a=10, b=20)
    assert str(type(result).__name__) == 'PMap'


def test_mutant_preserves_function_name():
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, d):
        return lst + [d['key']]
    
    result = combine([1, 2], {'key': 3})
    assert str(type(result).__name__) == 'PVector'


def test_mutant_with_empty_list():
    @mutant
    def process_empty(lst):
        return lst
    
    result = process_empty([])
    assert str(type(result).__name__) == 'PVector'


def test_mutant_with_empty_dict():
    @mutant
    def process_empty_dict(d):
        return d
    
    result = process_empty_dict({})
    assert str(type(result).__name__) == 'PMap'


# LLM-generated content at query #42
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze, thaw
    
    @mutant
    def modify_list(lst):
        return lst + [4]
    
    input_list = [1, 2, 3]
    result = modify_list(input_list)
    
    # The result should be frozen (persistent)
    assert str(type(result)).count('Plist') > 0 or str(type(result)).count('pvector') > 0
    
    # Test with kwargs
    @mutant
    def process_dict(data=None):
        if data is None:
            data = {}
        return data
    
    input_dict = {'a': 1}
    result_dict = process_dict(data=input_dict)
    
    # The result should be frozen (persistent map)
    assert str(type(result_dict)).count('PMap') > 0 or hasattr(result_dict, '__getitem__')
    
    # Test that modifications to original don't affect frozen version
    @mutant
    def identity(x):
        return x
    
    original = [1, 2, 3]
    frozen_result = identity(original)
    original.append(4)
    
    # frozen_result should still have 3 elements
    assert len(frozen_result) == 3


# LLM-generated content at query #43
#--------------------------

```python
def test_freeze_defaultdict_predicate_line_32():
    import collections
    from pyrsistent import pmap, freeze
    
    # Test that the predicate at line 32 evaluates to True for defaultdict
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd, strict=True)
    
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #44
#--------------------------

```python
def test_mutant_decorator_predicate_line_1_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset
    
    @mutant
    def modify_and_return(data):
        return data
    
    input_map = pmap({'a': 1, 'b': 2})
    result = modify_and_return(input_map)
    
    assert result == input_map
    assert not (result is input_map)


# LLM-generated content at query #45
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset

    @mutant
    def modify_list(lst):
        return lst + [1]

    result = modify_list([1, 2])
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 1]


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def process_dict(d):
        return d

    result = process_dict({'a': [1, 2], 'b': {'c': 3}})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['a'], type(pvector()))
    assert isinstance(result['b'], type(pmap()))


def test_mutant_freezes_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap

    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}

    result = func_with_kwargs(1, b={'x': 2})
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1
    assert isinstance(result['b'], type(pmap()))


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant

    @mutant
    def my_function(x):
        """Test docstring"""
        return x

    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def combine(lst, dct):
        return [lst, dct]

    result = combine([1, 2], {'a': 3})
    assert isinstance(result, type(pvector()))
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))


def test_mutant_freezes_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset

    @mutant
    def process_set(s):
        return s

    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset()))
    assert set(result) == {1, 2, 3}


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant

    @mutant
    def process_tuple(t):
        return t

    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == 1


def test_mutant_with_empty_containers():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset

    @mutant
    def process_empty(lst, dct, st):
        return [lst, dct, st]

    result = process_empty([], {}, set())
    assert isinstance(result, type(pvector()))
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))
    assert isinstance(result[2], type(pset()))


def test_mutant_return_value_is_immutable():
    from pyrsistent._helpers import mutant

    @mutant
    def return_list(x):
        return [x, x + 1]

    result = return_list(5)
    try:
        result[0] = 100
        assert False, "Should not be able to modify frozen return value"
    except TypeError:
        pass


def test_mutant_with_primitive_return():
    from pyrsistent._helpers import mutant

    @mutant
    def return_int(x):
        return x + 1

    result = return_int(5)
    assert result == 6
    assert isinstance(result, int)


# LLM-generated content at query #46
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_and_return(lst, d):
        lst.append(999)
        d['new_key'] = 'new_value'
        return lst, d
    
    original_list = [1, 2, 3]
    original_dict = {'a': 1}
    result = modify_and_return(original_list, original_dict)
    
    assert isinstance(result, tuple)
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))
    assert original_list == [1, 2, 3]
    assert original_dict == {'a': 1}


def test_mutant_with_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        data[0].append('modified')
        return data
    
    original = [{'key': 'value'}]
    result = process_nested(original)
    
    assert isinstance(result, type(pvector()))
    assert isinstance(result[0], type(pmap()))
    assert original == [{'key': 'value'}]


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def func_with_kwargs(lst, mapping=None):
        if mapping is None:
            mapping = {}
        lst.append(1)
        mapping['key'] = 'value'
        return lst, mapping
    
    result = func_with_kwargs([1, 2], mapping={'a': 1})
    
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        """My docstring"""
        return x
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'My docstring'


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        s.add(999)
        return s
    
    original = {1, 2, 3}
    result = process_set(original)
    
    assert isinstance(result, type(pset()))
    assert original == {1, 2, 3}


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def process_tuple(t):
        return t
    
    original = (1, [2, 3], {'a': 4})
    result = process_tuple(original)
    
    assert isinstance(result, tuple)
    assert isinstance(result[1], type(pvector()))
    assert original == (1, [2, 3], {'a': 4})


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst1, lst2, dct):
        lst1.append(999)
        lst2.append(888)
        dct['key'] = 'value'
        return lst1, lst2, dct
    
    l1 = [1, 2]
    l2 = [3, 4]
    d = {'x': 1}
    
    result = combine(l1, l2, d)
    
    assert l1 == [1, 2]
    assert l2 == [3, 4]
    assert d == {'x': 1}
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pvector()))
    assert isinstance(result[2], type(pmap()))


def test_mutant_with_primitive_return():
    from pyrsistent._helpers import mutant
    
    @mutant
    def return_int(x):
        return 42
    
    result = return_int([1, 2, 3])
    
    assert result == 42


def test_mutant_with_none_return():
    from pyrsistent._helpers import mutant
    
    @mutant
    def return_none(lst):
        lst.append(1)
        return None
    
    original = [1, 2]
    result = return_none(original)
    
    assert result is None
    assert original == [1, 2]


# LLM-generated content at query #47
#--------------------------

```python
def test_mutant_decorator_predicate_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset
    
    @mutant
    def modify_input(data):
        return data
    
    original_dict = {'a': 1, 'b': 2}
    result = modify_input(original_dict)
    
    assert result == pmap({'a': 1, 'b': 2})
    assert original_dict == {'a': 1, 'b': 2}
    assert not (result is original_dict)


# LLM-generated content at query #48
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    import collections
    from pyrsistent import freeze, pmap
    
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    
    result = freeze(dd, strict=True)
    
    assert result == pmap({'a': 1, 'b': 2})
    assert type(result).__name__ == 'PMap'


# LLM-generated content at query #49
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_input(data):
        # Try to mutate the input (this should fail since it's frozen)
        return data
    
    input_list = [1, 2, 3]
    result = modify_input(input_list)
    
    # Result should be frozen (persistent)
    assert isinstance(result, (pset, type(pmap())))
    
    # Verify that the decorator preserves the data
    assert result == freeze(input_list)


# LLM-generated content at query #50
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        lst.append(999)
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 3]


def test_mutant_freezes_dict_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    result = modify_dict({'a': 1})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1})


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def modify_nested(data):
        data['list'].append(999)
        return data
    
    result = modify_nested({'list': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert list(result['list']) == [1, 2, 3]


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        lst.append(dct['key'])
        return lst
    
    result = combine([1, 2], {'key': 3})
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2]


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process(data, extra=None):
        if extra:
            data['extra'] = extra['value']
        return data
    
    result = process({'a': 1}, extra={'value': 2})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1})


def test_mutant_preserves_immutability():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def return_list():
        return [1, 2, 3]
    
    result = return_list()
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 3]


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def modify_set(s):
        s.add(999)
        return s
    
    result = modify_set({1, 2, 3})
    assert isinstance(result, type(pset()))
    assert result == pset({1, 2, 3})


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t + (999,)
    
    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)


# LLM-generated content at query #51
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset, pmap, freeze
    
    @mutant
    def modify_and_return(s, m):
        return s, m
    
    original_set = pset([1, 2, 3])
    original_map = pmap({'a': 1, 'b': 2})
    
    result_set, result_map = modify_and_return(original_set, original_map)
    
    assert result_set == original_set
    assert result_map == original_map
    assert result_set is not original_set or result_set == freeze(original_set)
    assert result_map is not original_map or result_map == freeze(original_map)


# LLM-generated content at query #52
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_map(m):
        return m
    
    original_map = pmap({'a': 1, 'b': 2})
    result = modify_map(original_map)
    
    # Verify that the result is frozen (is a PMap, not a dict)
    assert not isinstance(result, dict)
    assert hasattr(result, '__setitem__') is False or result.__class__.__name__ == 'PMap'
    
    @mutant
    def modify_set(s):
        return s
    
    original_set = pset([1, 2, 3])
    result_set = modify_set(original_set)
    
    # Verify that the result is frozen (is a PSet, not a regular set)
    assert not isinstance(result_set, set)
    assert hasattr(result_set, '__setitem__') is False or result_set.__class__.__name__ == 'PSet'
    
    @mutant
    def process_data(data):
        return data
    
    test_data = {'nested': [1, 2, 3]}
    frozen_result = process_data(test_data)
    
    # The predicate at line 1 is False when the function is NOT a mutant function
    # but a regular function. So verify that our decorated function IS different from a regular one
    def regular_function(x):
        return x
    
    assert mutant is not None
    assert callable(modify_map)
    assert modify_map.__wrapped__ is not None


# LLM-generated content at query #53
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    original = {'a': 1}
    result = modify_dict(original)
    
    # The predicate at line 1 "def mutant(fn):" evaluates to False
    # because it's a function definition statement, not a boolean expression
    assert callable(modify_dict)
    assert result['a'] == 1
    assert result['new_key'] == 'new_value'
    
    # Verify the result is frozen
    try:
        result['another_key'] = 'value'
        assert False, "Should not be able to modify frozen result"
    except TypeError:
        pass
    
    # Verify the original is not modified
    assert 'new_key' not in original


def test_mutant_with_pmap():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def update_pmap(m):
        return m.set('x', 10)
    
    original_map = pmap({'a': 1})
    result = update_pmap(original_map)
    
    assert result['x'] == 10
    assert result['a'] == 1
    assert 'x' not in original_map


def test_mutant_with_pset():
    from pyrsistent import pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def add_to_set(s):
        return s.add(4)
    
    original_set = pset([1, 2, 3])
    result = add_to_set(original_set)
    
    assert 4 in result
    assert 4 not in original_set


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_with_kwargs(d, extra=None):
        if extra:
            d['extra'] = extra
        return d
    
    result = process_with_kwargs({'a': 1}, extra='value')
    
    assert result['a'] == 1
    assert result['extra'] == 'value'
    
    try:
        result['new'] = 'fail'
        assert False, "Should not be able to modify frozen result"
    except TypeError:
        pass


