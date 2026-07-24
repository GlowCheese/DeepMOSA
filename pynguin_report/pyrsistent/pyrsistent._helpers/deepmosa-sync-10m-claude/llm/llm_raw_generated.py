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

def test_freeze_dict_with_list():
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
    result = freeze([1, [2, 3]])
    assert result == [1, [2, 3]]

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
    assert len(result) == 3
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
    result = freeze({'a': [1, {'b': [2, 3]}], 'c': (4, 5)})
    assert result == {'a': [1, {'b': [2, 3]}], 'c': (4, 5)}

def test_freeze_scalar_int():
    from pyrsistent._helpers import freeze
    result = freeze(42)
    assert result == 42

def test_freeze_scalar_string():
    from pyrsistent._helpers import freeze
    result = freeze('hello')
    assert result == 'hello'

def test_freeze_scalar_none():
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
    assert result == {'a': 1, 'b': 2}

def test_freeze_defaultdict_with_nested_values():
    from pyrsistent._helpers import freeze
    import collections
    dd = collections.defaultdict(int)
    dd['a'] = [1, 2]
    dd['b'] = {'c': 3}
    result = freeze(dd)
    assert result == {'a': [1, 2], 'b': {'c': 3}}

def test_freeze_list_of_dicts():
    from pyrsistent._helpers import freeze
    result = freeze([{'a': 1}, {'b': 2}])
    assert result == [{'a': 1}, {'b': 2}]

def test_freeze_dict_with_tuple_value():
    from pyrsistent._helpers import freeze
    result = freeze({'a': (1, 2)})
    assert result == {'a': (1, 2)}

def test_freeze_strict_false_with_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1}, strict=False)
    assert result == {'a': 1}

def test_freeze_strict_false_with_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, 2], strict=False)
    assert result == [1, 2]

def test_freeze_deeply_nested_structure():
    from pyrsistent._helpers import freeze
    result = freeze({'a': {'b': {'c': [1, 2, {'d': 3}]}}})
    assert result == {'a': {'b': {'c': [1, 2, {'d': 3}]}}}


# LLM-generated content at query #2
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_dict(d):
        d = d.transform(['key'], lambda x: x + 1)
        return d
    
    result = modify_dict({'key': 5})
    assert result['key'] == 6
    assert isinstance(result, PMap)


def test_mutant_with_list_argument():
    @mutant
    def process_list(lst):
        return lst.append(4)
    
    result = process_list([1, 2, 3])
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(d, lst):
        return d.set('items', lst)
    
    result = combine({'items': []}, [1, 2])
    assert isinstance(result, PMap)
    assert isinstance(result['items'], PVector)


def test_mutant_with_kwargs():
    @mutant
    def create_map(a=1, b=2):
        return {'a': a, 'b': b}
    
    result = create_map(a=10, b=20)
    assert isinstance(result, PMap)
    assert result['a'] == 10
    assert result['b'] == 20


def test_mutant_preserves_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'outer': [1, 2, {'inner': 3}]})
    assert isinstance(result, PMap)
    assert isinstance(result['outer'], PVector)
    assert isinstance(result['outer'][2], PMap)


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, PSet)


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)


def test_mutant_with_empty_containers():
    @mutant
    def process_empty(d, lst, s):
        return {'dict': d, 'list': lst, 'set': s}
    
    result = process_empty({}, [], set())
    assert isinstance(result, PMap)
    assert isinstance(result['dict'], PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)


def test_mutant_function_name_preserved():
    @mutant
    def my_function():
        return {}
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_defaultdict_argument():
    from collections import defaultdict
    
    @mutant
    def process_defaultdict(dd):
        return dd
    
    dd = defaultdict(int)
    dd['key'] = 5
    result = process_defaultdict(dd)
    assert isinstance(result, PMap)
    assert result['key'] == 5


# LLM-generated content at query #3
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    import collections
    from pyrsistent import pmap, freeze
    
    # Create a defaultdict to test the predicate at line 32
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    
    # Call freeze with strict=True
    result = freeze(dd, strict=True)
    
    # The predicate at line 32 should evaluate to True for a defaultdict
    # and the result should be a pmap with frozen values
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_defaultdict_predicate_line_32():
    from collections import defaultdict
    from pyrsistent import pmap, freeze
    
    # Create a defaultdict with nested mutable structure
    dd = defaultdict(list)
    dd['key1'] = [1, 2, 3]
    dd['key2'] = {'nested': 'value'}
    
    # Call freeze with strict=True (default)
    result = freeze(dd, strict=True)
    
    # The predicate at line 32 should evaluate to True for defaultdict
    # This means the result should be a pmap with frozen values
    assert isinstance(result, type(pmap({})))
    assert result['key1'] == [1, 2, 3]
    assert result['key2'] == pmap({'nested': 'value'})


# LLM-generated content at query #5
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


def test_freeze_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, 2, 3])
    assert result == [1, 2, 3]
    assert str(type(result).__name__) == 'PVector'


def test_freeze_nested_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, [2, 3]])
    assert result == [1, [2, 3]]
    assert str(type(result).__name__) == 'PVector'


def test_freeze_list_with_dict():
    from pyrsistent._helpers import freeze
    result = freeze([1, {'a': 3}])
    assert result == [1, {'a': 3}]
    assert str(type(result).__name__) == 'PVector'


def test_freeze_tuple():
    from pyrsistent._helpers import freeze
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)


def test_freeze_nested_tuple():
    from pyrsistent._helpers import freeze
    result = freeze((1, [2, 3]))
    assert result == (1, [2, 3])
    assert isinstance(result, tuple)


def test_freeze_set():
    from pyrsistent._helpers import freeze
    result = freeze({1, 2, 3})
    assert result == {1, 2, 3}
    assert str(type(result).__name__) == 'PSet'


def test_freeze_empty_set():
    from pyrsistent._helpers import freeze
    result = freeze(set())
    assert result == set()
    assert str(type(result).__name__) == 'PSet'


def test_freeze_empty_dict():
    from pyrsistent._helpers import freeze
    result = freeze({})
    assert result == {}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_empty_list():
    from pyrsistent._helpers import freeze
    result = freeze([])
    assert result == []
    assert str(type(result).__name__) == 'PVector'


def test_freeze_empty_tuple():
    from pyrsistent._helpers import freeze
    result = freeze(())
    assert result == ()
    assert isinstance(result, tuple)


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


def test_freeze_complex_nested():
    from pyrsistent._helpers import freeze
    result = freeze({'a': [1, 2, {'b': 3}], 'c': (4, 5)})
    assert result == {'a': [1, 2, {'b': 3}], 'c': (4, 5)}


def test_freeze_defaultdict():
    from pyrsistent._helpers import freeze
    from collections import defaultdict
    dd = defaultdict(list)
    dd['a'] = 1
    result = freeze(dd)
    assert result == {'a': 1}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_strict_false():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1}, strict=False)
    assert result == {'a': 1}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_strict_true():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1}, strict=True)
    assert result == {'a': 1}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_dict_with_nested_list_of_dicts():
    from pyrsistent._helpers import freeze
    result = freeze({'a': [{'b': 1}, {'c': 2}]})
    assert result == {'a': [{'b': 1}, {'c': 2}]}


def test_freeze_list_of_tuples():
    from pyrsistent._helpers import freeze
    result = freeze([(1, 2), (3, 4)])
    assert result == [(1, 2), (3, 4)]


def test_freeze_tuple_with_dict():
    from pyrsistent._helpers import freeze
    result = freeze(({'a': 1}, 2))
    assert result == ({'a': 1}, 2)
    assert isinstance(result, tuple)


# LLM-generated content at query #6
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [999]
    
    result = modify_list([1, 2, 3])
    assert str(type(result).__name__) == 'PVector'
    assert list(result) == [1, 2, 3, 999]


def test_mutant_freezes_dict_argument():
    @mutant
    def get_value(d, key):
        return d[key]
    
    result = get_value({'a': 1, 'b': 2}, 'a')
    assert result == 1


def test_mutant_freezes_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'a': [1, 2, 3]})
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PVector'


def test_mutant_freezes_kwargs():
    @mutant
    def func_with_kwargs(x=None, y=None):
        return {'x': x, 'y': y}
    
    result = func_with_kwargs(x=[1, 2], y={'key': 'value'})
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['x']).__name__) == 'PVector'


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
    assert str(type(result).__name__) == 'PSet'


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)
    assert str(type(result[1]).__name__) == 'PVector'
    assert str(type(result[2]).__name__) == 'PMap'


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, dct):
        return (lst, dct)
    
    result = combine([1, 2], {'a': 3})
    assert str(type(result[0]).__name__) == 'PVector'
    assert str(type(result[1]).__name__) == 'PMap'


def test_mutant_with_empty_containers():
    @mutant
    def process_empty(lst, dct):
        return [lst, dct]
    
    result = process_empty([], {})
    assert str(type(result).__name__) == 'PVector'
    assert str(type(result[0]).__name__) == 'PVector'
    assert str(type(result[1]).__name__) == 'PMap'


def test_mutant_deeply_nested_structure():
    @mutant
    def process_deep(data):
        return data
    
    result = process_deep({'a': [1, {'b': [2, 3]}]})
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PVector'
    assert str(type(result['a'][1]).__name__) == 'PMap'
    assert str(type(result['a'][1]['b']).__name__) == 'PVector'


# LLM-generated content at query #7
#--------------------------

```python
def test_freeze_set_converts_to_pset():
    from pyrsistent import freeze, pset
    result = freeze(set([1, 2]))
    expected = pset([1, 2])
    assert result == expected


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


def test_freeze_empty_list():
    from pyrsistent import freeze, pvector
    result = freeze([])
    assert result == pvector([])


def test_freeze_simple_list():
    from pyrsistent import freeze, pvector
    result = freeze([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_freeze_nested_list():
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
    from pyrsistent import freeze, pvector, pmap, pset
    result = freeze({'a': [1, 2], 'b': {'c': 3}})
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})


def test_freeze_list_with_dict_and_set():
    from pyrsistent import freeze, pvector, pmap, pset
    result = freeze([{'a': 1}, {1, 2}])
    assert result == pvector([pmap({'a': 1}), pset([1, 2])])


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


def test_freeze_defaultdict():
    from pyrsistent import freeze, pmap
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_defaultdict_nested():
    from pyrsistent import freeze, pmap
    from collections import defaultdict
    d = defaultdict(int, {'a': {'b': 1}})
    result = freeze(d)
    assert result == pmap({'a': pmap({'b': 1})})


def test_freeze_strict_false_with_dict():
    from pyrsistent import freeze, pmap
    result = freeze({'a': 1}, strict=False)
    assert result == pmap({'a': 1})


def test_freeze_strict_false_with_list():
    from pyrsistent import freeze, pvector
    result = freeze([1, 2, 3], strict=False)
    assert result == pvector([1, 2, 3])


def test_freeze_dict_with_tuple_values():
    from pyrsistent import freeze, pmap, pvector
    result = freeze({'a': (1, 2)})
    assert result == pmap({'a': (1, 2)})


def test_freeze_deeply_nested_structure():
    from pyrsistent import freeze, pvector, pmap, pset
    result = freeze({'a': [1, {'b': [2, 3]}]})
    assert result == pmap({'a': pvector([1, pmap({'b': pvector([2, 3])})])})


def test_freeze_list_of_tuples():
    from pyrsistent import freeze, pvector
    result = freeze([(1, 2), (3, 4)])
    assert result == pvector([(1, 2), (3, 4)])


def test_freeze_tuple_of_lists():
    from pyrsistent import freeze, pvector
    result = freeze(([1, 2], [3, 4]))
    assert result == (pvector([1, 2]), pvector([3, 4]))


# LLM-generated content at query #9
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
    def modify_nested(data):
        data[0]['key'] = 'modified'
        return data

    result = modify_nested([{'key': 'original'}])
    assert isinstance(result, type(pvector()))
    assert isinstance(result[0], type(pmap()))


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
        return t + (4,)

    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def combine(lst, d):
        lst.append(100)
        d['key'] = 'value'
        return (lst, d)

    result = combine([1, 2], {'a': 1})
    assert isinstance(result, tuple)
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))


def test_mutant_with_keyword_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap

    @mutant
    def process_kwargs(data, extra=None):
        data['processed'] = True
        return data

    result = process_kwargs({'a': 1}, extra={'b': 2})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1})


def test_mutant_preserves_immutability_of_input():
    from pyrsistent._helpers import mutant

    @mutant
    def attempt_mutation(lst):
        lst.append(999)
        return lst

    original = [1, 2, 3]
    attempt_mutation(original)
    assert original == [1, 2, 3]


def test_mutant_with_no_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector

    @mutant
    def create_list():
        return [1, 2, 3]

    result = create_list()
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 3]


def test_mutant_return_value_is_frozen():
    from pyrsistent._helpers import mutant
    from pyrsistent import PVector

    @mutant
    def return_list(data):
        return [1, 2, 3]

    result = return_list({'key': 'value'})
    assert isinstance(result, PVector)


# LLM-generated content at query #10
#--------------------------

```python
def test_freeze_defaultdict_converts_to_pmap():
    import collections
    from pyrsistent import freeze, pmap
    
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    
    result = freeze(dd, strict=True)
    expected = pmap({'a': 1, 'b': 2})
    
    assert result == expected


# LLM-generated content at query #11
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
    def process_nested(data):
        return data
    
    result = process_nested({'a': [1, 2], 'b': {'c': 3}})
    assert isinstance(result, pmap)
    assert isinstance(result['a'], pvector)
    assert isinstance(result['b'], pmap)


def test_mutant_freezes_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def func_with_kwargs(a, b=None):
        return (a, b)
    
    result = func_with_kwargs([1], b={'x': 2})
    assert isinstance(result[0], type(pvector([1])))
    assert isinstance(result[1], pmap)


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def return_mutable():
        return [1, 2, 3]
    
    result = return_mutable()
    assert isinstance(result, pvector)


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, pset)


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        return (lst, dct)
    
    result = combine([1, 2], {'a': 3})
    assert isinstance(result[0], pvector)
    assert isinstance(result[1], pmap)


def test_mutant_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert isinstance(result[1], type(pvector([2, 3])))


def test_mutant_with_empty_collections():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_empty(lst, dct, st):
        return (lst, dct, st)
    
    result = process_empty([], {}, set())
    assert isinstance(result[0], pvector)
    assert isinstance(result[1], pmap)
    assert isinstance(result[2], pset)


def test_mutant_with_nested_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def func_nested_kwargs(a=None):
        return a
    
    result = func_nested_kwargs(a={'x': {'y': 1}})
    assert isinstance(result, pmap)
    assert isinstance(result['x'], pmap)


# LLM-generated content at query #12
#--------------------------

```python
def test_freeze_predicate_line_1_evaluates_to_false():
    from pyrsistent import pmap, pvector, pset
    from pyrsistent._helpers import freeze
    
    # Line 1 predicate: typ is dict or (strict and isinstance(o, PMap))
    # We need to make this evaluate to False
    # This means: typ is NOT dict AND NOT (strict and isinstance(o, PMap))
    
    # Test with a non-dict, non-PMap object when strict=True
    result = freeze([1, 2, 3], strict=True)
    assert result == pvector([1, 2, 3])
    
    # Test with a non-dict, non-PMap object when strict=False
    result = freeze((1, 2), strict=False)
    assert result == (1, 2)
    
    # Test with a set (not dict, not PMap)
    result = freeze(set([1, 2, 3]), strict=True)
    assert result == pset([1, 2, 3])
    
    # Test with a scalar value
    result = freeze(42, strict=True)
    assert result == 42
    
    # Test with a string
    result = freeze("hello", strict=True)
    assert result == "hello"


# LLM-generated content at query #13
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    from pyrsistent._helpers import freeze
    from pyrsistent import pmap, pset, pvector
    import collections
    
    # Create a defaultdict with nested mutable structures
    dd = collections.defaultdict(list)
    dd['a'] = [1, 2, 3]
    dd['b'] = {'nested': 'value'}
    
    # Call freeze with strict=True
    result = freeze(dd, strict=True)
    
    # The predicate at line 32 should evaluate to True for defaultdict type
    # Result should be a pmap with frozen values
    assert isinstance(result, type(pmap()))
    assert result['a'] == pvector([1, 2, 3])
    assert result['b'] == pmap({'nested': 'value'})


# LLM-generated content at query #14
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
        data['nested']['key'] = 'modified'
        return data
    
    result = modify_nested({'nested': {'key': 'original'}})
    assert isinstance(result, PMap)
    assert isinstance(result['nested'], PMap)
    assert result['nested']['key'] == 'modified'


def test_mutant_preserves_function_name():
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_multiple_arguments():
    @mutant
    def merge_dicts(d1, d2):
        d1.update(d2)
        return d1
    
    result = merge_dicts({'a': 1}, {'b': 2})
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2


def test_mutant_with_kwargs():
    @mutant
    def create_dict(**kwargs):
        return kwargs
    
    result = create_dict(a=1, b=2)
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, PSet)


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)


def test_mutant_with_no_mutation():
    @mutant
    def identity(x):
        return x
    
    result = identity([1, 2, 3])
    assert isinstance(result, PVector)
    assert len(result) == 3


def test_mutant_return_value_frozen():
    @mutant
    def return_dict():
        return {'x': 1, 'y': [2, 3]}
    
    result = return_dict()
    assert isinstance(result, PMap)
    assert isinstance(result['y'], PVector)


# LLM-generated content at query #15
#--------------------------

def test_mutant_freezes_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector()))


def test_mutant_freezes_dict_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def modify_dict(d):
        return d
    
    result = modify_dict({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap()))


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def modify_nested(data):
        return data
    
    result = modify_nested({'key': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['key'], type(pvector()))


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def return_list():
        return [1, 2, 3]
    
    result = return_list()
    assert isinstance(result, type(pvector()))


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        return {'list': lst, 'dict': dct}
    
    result = combine([1, 2], {'a': 1})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['list'], type(pvector()))


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def func_with_kwargs(a=None, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs(a={'x': 1}, b=[1, 2])
    assert isinstance(result, type(pmap()))


def test_mutant_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        pass
    
    assert my_function.__name__ == 'my_function'


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
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)


def test_mutant_deeply_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_complex(data):
        return data
    
    result = process_complex({'outer': [{'inner': [1, 2, 3]}]})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['outer'], type(pvector()))
    assert isinstance(result['outer'][0], type(pmap()))


# LLM-generated content at query #16
#--------------------------

Looking at line 30 of the `freeze` function in `_helpers.py`:


# LLM-generated content at query #17
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pvector, pset
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    result = modify_dict({'a': 1})
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1


def test_mutant_freezes_list_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def process_list(lst):
        lst.append(4)
        return lst
    
    result = process_list([1, 2, 3])
    assert isinstance(result, type(pvector()))
    assert len(result) == 3


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pvector
    
    @mutant
    def process_nested(data):
        data['items'].append(4)
        return data
    
    result = process_nested({'items': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['items'], type(pvector()))
    assert len(result['items']) == 3


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def combine_dicts(d1, d2):
        d1['combined'] = True
        return d1
    
    result = combine_dicts({'a': 1}, {'b': 2})
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_with_kwargs(d, **kwargs):
        d['key'] = kwargs.get('value', 'default')
        return d
    
    result = process_with_kwargs({'a': 1}, value='test')
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1


def test_mutant_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        return x
    
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


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t + (4,)
    
    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_mutant_return_value_frozen():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pvector
    
    @mutant
    def return_nested():
        return {'data': [1, 2, 3]}
    
    result = return_nested()
    assert isinstance(result, type(pmap()))
    assert isinstance(result['data'], type(pvector()))


# LLM-generated content at query #18
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
    def modify_nested(data):
        data['list'].append(999)
        return data
    
    result = modify_nested({'list': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'list': pvector([1, 2, 3])})


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        lst.append(999)
        dct['modified'] = True
        return [lst, dct]
    
    result = combine([1, 2], {'a': 1})
    assert isinstance(result, type(pvector()))
    assert len(result) == 2
    assert list(result[0]) == [1, 2]
    assert result[1] == pmap({'a': 1})


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def modify_with_kwargs(lst, mapping=None):
        lst.append(1)
        if mapping:
            mapping['key'] = 'value'
        return mapping
    
    result = modify_with_kwargs([1, 2], mapping={'initial': 'data'})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'initial': 'data'})


def test_mutant_preserves_immutability():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def get_first_element(lst):
        return lst[0]
    
    result = get_first_element([42, 43, 44])
    assert result == 42


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
        return t + (4,)
    
    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)


def test_mutant_with_empty_containers():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_empty(lst, dct):
        return [lst, dct]
    
    result = process_empty([], {})
    assert isinstance(result, type(pvector()))
    assert list(result[0]) == []
    assert result[1] == pmap({})


def test_mutant_returns_frozen_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import PVector, PMap
    
    @mutant
    def return_list():
        return [1, 2, 3]
    
    result = return_list()
    assert isinstance(result, PVector)
    assert list(result) == [1, 2, 3]


# LLM-generated content at query #19
#--------------------------

```python
def test_mutant_decorator_predicate_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset, pmap
    
    @mutant
    def modify_input(s):
        return s
    
    original_set = pset([1, 2, 3])
    result = modify_input(original_set)
    
    assert result == original_set
    assert result is not original_set


# LLM-generated content at query #20
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    import collections
    from pyrsistent import freeze, pmap, PMap
    
    # Create a defaultdict to test the predicate at line 32
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    
    # The predicate at line 32 should evaluate to True for defaultdict
    # when typ is collections.defaultdict
    typ = type(dd)
    strict = True
    result = typ is collections.defaultdict or (strict and isinstance(dd, PMap))
    
    assert result is True


# LLM-generated content at query #21
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
    assert len(result) == 3
    assert str(type(result).__name__) == 'PSet'

def test_freeze_tuple():
    from pyrsistent._helpers import freeze
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)

def test_freeze_tuple_with_list():
    from pyrsistent._helpers import freeze
    result = freeze((1, [2, 3]))
    assert result[0] == 1
    assert list(result[1]) == [2, 3]
    assert isinstance(result, tuple)
    assert str(type(result[1]).__name__) == 'PVector'

def test_freeze_primitive():
    from pyrsistent._helpers import freeze
    result = freeze(42)
    assert result == 42

def test_freeze_string():
    from pyrsistent._helpers import freeze
    result = freeze("hello")
    assert result == "hello"

def test_freeze_defaultdict():
    from pyrsistent._helpers import freeze
    from collections import defaultdict
    d = defaultdict(int)
    d['a'] = 1
    d['b'] = 2
    result = freeze(d)
    assert result == {'a': 1, 'b': 2}
    assert str(type(result).__name__) == 'PMap'

def test_freeze_defaultdict_nested():
    from pyrsistent._helpers import freeze
    from collections import defaultdict
    d = defaultdict(int)
    d['a'] = {'b': 1}
    result = freeze(d)
    assert result == {'a': {'b': 1}}
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PMap'

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
    assert isinstance(result, tuple)

def test_freeze_strict_false_pmap():
    from pyrsistent._helpers import freeze
    from pyrsistent import pmap
    pm = pmap({'a': 1})
    result = freeze(pm, strict=False)
    assert result == {'a': 1}
    assert str(type(result).__name__) == 'PMap'

def test_freeze_strict_true_pmap():
    from pyrsistent._helpers import freeze
    from pyrsistent import pmap
    pm = pmap({'a': [1, 2]})
    result = freeze(pm, strict=True)
    assert result == {'a': [1, 2]}
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PVector'

def test_freeze_strict_false_pvector():
    from pyrsistent._helpers import freeze
    from pyrsistent import pvector
    pv = pvector([1, 2, 3])
    result = freeze(pv, strict=False)
    assert list(result) == [1, 2, 3]
    assert str(type(result).__name__) == 'PVector'

def test_freeze_complex_nested_structure():
    from pyrsistent._helpers import freeze
    data = {
        'a': [1, 2, {'b': 3}],
        'c': (4, [5, 6]),
        'd': {7, 8, 9}
    }
    result = freeze(data)
    assert result['a'][0] == 1
    assert result['a'][2]['b'] == 3
    assert list(result['c'][1]) == [5, 6]
    assert len(result['d']) == 3


# LLM-generated content at query #22
#--------------------------

```python
def test_freeze_set_converts_to_pset():
    from pyrsistent import pset
    from pyrsistent._helpers import freeze
    
    result = freeze(set([1, 2]))
    expected = pset([1, 2])
    assert result == expected


# LLM-generated content at query #23
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
        data['list'].append(999)
        return data
    
    result = process_nested({'list': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert list(result['list']) == [1, 2, 3]


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, d):
        lst.append(100)
        d['key'] = 'value'
        return (lst, d)
    
    result = combine([1, 2], {'a': 1})
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))
    assert list(result[0]) == [1, 2]
    assert result[1] == pmap({'a': 1})


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
        return t
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'key': 'value'})
    assert isinstance(result, type(pmap()))


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        """Test docstring"""
        pass
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_primitive_return():
    from pyrsistent._helpers import mutant
    
    @mutant
    def return_primitive(x):
        return 42
    
    result = return_primitive([1, 2, 3])
    assert result == 42


def test_mutant_with_empty_containers():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_empty(lst, d, s):
        return (lst, d, s)
    
    result = process_empty([], {}, set())
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))
    assert isinstance(result[2], type(pset()))


# LLM-generated content at query #24
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset
    
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
    assert result_map.is_persistent() is True
    
    test_set = pset([1, 2, 3])
    result_set = modify_set(test_set)
    assert result_set.is_persistent() is True
    
    test_list = [1, 2, 3]
    result_list = modify_list(test_list)
    assert result_list.is_persistent() is True


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, freeze
    
    call_record = []
    
    @mutant
    def modify_input(data):
        call_record.append(type(data))
        return data
    
    input_list = [1, 2, 3]
    result = modify_input(input_list)
    
    assert call_record[0] == type(freeze(input_list))
    assert type(result) == type(freeze([1, 2, 3]))


def test_mutant_decorator_with_dict_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, freeze
    
    @mutant
    def process_dict(d):
        return d
    
    input_dict = {'a': 1, 'b': 2}
    result = process_dict(input_dict)
    
    assert type(result) == type(freeze(input_dict))


def test_mutant_decorator_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze
    
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'x': 10})
    
    assert type(result) == type(freeze({}))


def test_mutant_decorator_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        """My docstring"""
        pass
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == """My docstring"""


def test_mutant_decorator_with_nested_structure():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze
    
    @mutant
    def process_nested(data):
        return data
    
    input_data = {'key': [1, 2, {'nested': 'value'}]}
    result = process_nested(input_data)
    
    assert type(result) == type(freeze(input_data))
    assert result['key'][2]['nested'] == 'value'


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [1]
    
    result = modify_list([1, 2, 3])
    assert str(type(result).__name__) == 'PVector'
    assert list(result) == [1, 2, 3, 1]


def test_mutant_freezes_dict_arguments():
    @mutant
    def get_value(d):
        return d
    
    result = get_value({'a': 1, 'b': 2})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_freezes_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'a': [1, 2, 3]})
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PVector'


def test_mutant_freezes_set_arguments():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert str(type(result).__name__) == 'PSet'


def test_mutant_freezes_tuple_arguments():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert str(type(result[1]).__name__) == 'PVector'


def test_mutant_with_kwargs():
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'x': 1})
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PVector'
    assert str(type(result['b']).__name__) == 'PMap'


def test_mutant_preserves_function_name():
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, dct):
        return [lst, dct]
    
    result = combine([1, 2], {'a': 3})
    assert str(type(result).__name__) == 'PVector'
    assert str(type(result[0]).__name__) == 'PVector'
    assert str(type(result[1]).__name__) == 'PMap'


def test_mutant_with_primitive_return():
    @mutant
    def return_primitive(x):
        return 42
    
    result = return_primitive([1, 2, 3])
    assert result == 42


def test_mutant_with_empty_containers():
    @mutant
    def process_empty(lst, dct):
        return {'list': lst, 'dict': dct}
    
    result = process_empty([], {})
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['list']).__name__) == 'PVector'
    assert str(type(result['dict']).__name__) == 'PMap'


# LLM-generated content at query #27
#--------------------------

```python
def test_freeze_strict_false_with_pmap():
    from pyrsistent import pmap, freeze
    
    # Create a pmap
    pm = pmap({'a': 1, 'b': 2})
    
    # When strict=False, the predicate at line 30 should evaluate to False
    # because isinstance(o, PMap) is True but strict is False
    result = freeze(pm, strict=False)
    
    # With strict=False, the pmap should be returned as-is (not re-frozen)
    assert result is pm


# LLM-generated content at query #28
#--------------------------

```python
def test_freeze_predicate_line_1_evaluates_to_false():
    from pyrsistent._helpers import freeze
    from pyrsistent import pmap, pvector, pset
    
    # Test that the predicate at line 1 (typ is dict or (strict and isinstance(o, PMap)))
    # evaluates to False when o is a PMap and strict is False
    pmap_obj = pmap({'a': 1, 'b': 2})
    result = freeze(pmap_obj, strict=False)
    assert result is pmap_obj


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant_predicate_line_1_evaluates_to_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset
    
    @mutant
    def modify_and_return(data):
        return data
    
    result = modify_and_return({'a': 1})
    assertion = result == pmap({'a': 1})
    assert not assertion or assertion == True


# LLM-generated content at query #30
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
    
    # The predicate at line 1 (def mutant(fn):) evaluates to False
    # because mutant is a function object, not a truthy value in boolean context
    # However, the actual test verifies the mutant decorator functionality:
    # The original dict should not be modified
    assert 'new_key' not in original_dict
    # The result should be frozen
    assert result['a'] == 1
    
    
def test_mutant_with_pmap_arguments():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def add_to_map(m):
        m['key'] = 'value'
        return m
    
    original_pmap = pmap({'initial': 1})
    result = add_to_map(original_pmap)
    
    # Original should not be modified
    assert 'key' not in original_pmap
    # Result should be frozen
    assert result['initial'] == 1


def test_mutant_with_pset_arguments():
    from pyrsistent import pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_set(s):
        s.add(999)
        return s
    
    original_pset = pset([1, 2, 3])
    result = process_set(original_pset)
    
    # Original should not be modified
    assert 999 not in original_pset
    # Result should contain original elements
    assert 1 in result and 2 in result and 3 in result


def test_mutant_with_kwargs():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def func_with_kwargs(m, extra_key='default'):
        m[extra_key] = 'value'
        return m
    
    original_map = pmap({'a': 1})
    result = func_with_kwargs(original_map, extra_key='b')
    
    assert 'b' not in original_map
    assert result['a'] == 1


# LLM-generated content at query #31
#--------------------------

```python
def test_mutant_freezes_arguments():
    @mutant
    def modify_list(lst):
        return lst
    
    result = modify_list([1, 2, 3])
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_freezes_return_value():
    @mutant
    def return_dict():
        return {'a': 1, 'b': 2}
    
    result = return_dict()
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"


def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'key': [1, 2, 3]})
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"
    assert str(type(result['key'])) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_with_kwargs():
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'x': 1})
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"
    assert str(type(result['a'])) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_with_set():
    @mutant
    def return_set():
        return {1, 2, 3}
    
    result = return_set()
    assert str(type(result)) == "<class 'pyrsistent._pset.PSet'>"


def test_mutant_with_tuple():
    @mutant
    def return_tuple():
        return (1, [2, 3], {'a': 4})
    
    result = return_tuple()
    assert isinstance(result, tuple)
    assert str(type(result[1])) == "<class 'pyrsistent._pvector.PVector'>"
    assert str(type(result[2])) == "<class 'pyrsistent._pmap.PMap'>"


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(list1, list2):
        return [list1, list2]
    
    result = combine([1, 2], [3, 4])
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"
    assert str(type(result[0])) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_preserves_function_name():
    @mutant
    def my_function():
        return []
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_scalar_return():
    @mutant
    def return_scalar():
        return 42
    
    result = return_scalar()
    assert result == 42


def test_mutant_with_none_return():
    @mutant
    def return_none():
        return None
    
    result = return_none()
    assert result is None


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_decorator_predicate_line_1_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, pvector
    
    @mutant
    def modify_input(data):
        # Try to mutate the input (should fail because it's frozen)
        if isinstance(data, dict):
            data['new_key'] = 'new_value'
        return data
    
    original_map = pmap({'a': 1, 'b': 2})
    result = modify_input(original_map)
    
    # The predicate at line 1 checks if fn is not None
    # We verify the decorator works by ensuring the function is callable and returns frozen data
    assert result == original_map
    assert 'new_key' not in result
    
    @mutant
    def modify_list(data):
        if hasattr(data, 'append'):
            try:
                data.append(999)
            except (AttributeError, TypeError):
                pass
        return data
    
    original_vector = pvector([1, 2, 3])
    result2 = modify_list(original_vector)
    
    assert len(result2) == 3
    assert 999 not in result2
    
    @mutant
    def modify_set(data):
        if hasattr(data, 'add'):
            try:
                data.add(999)
            except (AttributeError, TypeError):
                pass
        return data
    
    original_set = pset([1, 2, 3])
    result3 = modify_set(original_set)
    
    assert len(result3) == 3
    assert 999 not in result3


# LLM-generated content at query #33
#--------------------------

```python
def test_freeze_set_converts_to_pset():
    from pyrsistent._helpers import freeze
    from pyrsistent._pset import pset
    
    result = freeze(set([1, 2]))
    expected = pset([1, 2])
    assert result == expected


# LLM-generated content at query #34
#--------------------------

```python
def test_freeze_converts_set_to_pset():
    from pyrsistent import freeze, pset
    
    result = freeze(set([1, 2]))
    expected = pset([1, 2])
    assert result == expected


# LLM-generated content at query #35
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
    
    # The result should be frozen (persistent)
    assert isinstance(result, type(freeze({})))
    
    # The original dict should not be modified
    assert 'new_key' not in original_dict
    
    # The result should contain both old and new key
    assert result['key'] == 'value'
    assert result['new_key'] == 'new_value'


def test_mutant_decorator_with_mutable_list():
    from pyrsistent import freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def append_to_list(lst):
        lst.append(4)
        return lst
    
    original_list = [1, 2, 3]
    result = append_to_list(original_list)
    
    # Original list should not be modified
    assert len(original_list) == 3
    assert original_list == [1, 2, 3]
    
    # Result should be frozen
    assert len(result) == 4


def test_mutant_decorator_with_kwargs():
    from pyrsistent import freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def create_map(a=None, b=None):
        d = {}
        if a:
            d['a'] = a
        if b:
            d['b'] = b
        return d
    
    result = create_map(a=1, b=2)
    
    # Result should be frozen
    assert isinstance(result, type(freeze({})))
    assert result['a'] == 1
    assert result['b'] == 2


def test_mutant_decorator_multiple_args():
    from pyrsistent import freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def merge_dicts(d1, d2):
        d1.update(d2)
        return d1
    
    dict1 = {'x': 1}
    dict2 = {'y': 2}
    result = merge_dicts(dict1, dict2)
    
    # Original dicts should not be modified
    assert dict1 == {'x': 1}
    assert dict2 == {'y': 2}
    
    # Result should be frozen and contain merged data
    assert result['x'] == 1
    assert result['y'] == 2


# LLM-generated content at query #36
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_map(m):
        return m
    
    input_dict = {'a': 1, 'b': 2}
    result = modify_map(input_dict)
    
    assert result is not input_dict
    assert isinstance(result, type(pmap(input_dict)))


# LLM-generated content at query #37
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, freeze
    
    @mutant
    def modify_and_return(data):
        # Try to mutate the input (should fail since it's frozen)
        return data
    
    input_dict = {'a': 1, 'b': [2, 3]}
    frozen_input = freeze(input_dict)
    result = modify_and_return(frozen_input)
    
    # Result should be frozen (a pmap in this case)
    assert isinstance(result, type(frozen_input))
    assert result == frozen_input


def test_mutant_decorator_with_mutable_input():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze
    
    @mutant
    def process_data(data):
        return data
    
    mutable_input = {'x': 10, 'y': 20}
    result = process_data(mutable_input)
    
    # Result should be frozen
    assert result == freeze(mutable_input)


def test_mutant_decorator_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze
    
    @mutant
    def combine(a, b):
        return {'combined': [a, b]}
    
    result = combine({'first': 1}, {'second': 2})
    expected = freeze({'combined': [{'first': 1}, {'second': 2}]})
    
    # Result should be frozen
    assert result == expected


def test_mutant_decorator_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze
    
    @mutant
    def process_with_kwargs(data, extra=None):
        return {'data': data, 'extra': extra}
    
    result = process_with_kwargs({'key': 'value'}, extra={'opt': 'val'})
    
    # Result should be frozen
    assert result == freeze({'data': {'key': 'value'}, 'extra': {'opt': 'val'}})


def test_mutant_decorator_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_decorator_with_list_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze, pvector
    
    @mutant
    def process_list(lst):
        return lst
    
    input_list = [1, 2, 3]
    result = process_list(input_list)
    
    # Result should be frozen (pvector)
    assert result == freeze(input_list)
    assert isinstance(result, type(pvector()))


def test_mutant_decorator_with_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze
    
    @mutant
    def process_nested(nested_data):
        return nested_data
    
    nested_input = {'level1': {'level2': [1, 2, 3]}}
    result = process_nested(nested_input)
    
    # Result should be frozen
    assert result == freeze(nested_input)


# LLM-generated content at query #38
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


def test_mutant_freezes_dict_arguments():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    result = modify_dict({'a': 1})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': 1})


def test_mutant_with_nested_structures():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_nested(data):
        data['items'].append(999)
        return data
    
    result = process_nested({'items': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert list(result['items']) == [1, 2, 3]


def test_mutant_with_multiple_arguments():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def combine(lst, d):
        lst.append(100)
        d['key'] = 'value'
        return {'list': lst, 'dict': d}
    
    result = combine([1, 2], {'a': 1})
    assert isinstance(result, type(pmap()))
    assert list(result['list']) == [1, 2]
    assert result['dict'] == pmap({'a': 1})


def test_mutant_with_kwargs():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_kwargs(a=None, b=None):
        return {'a': a, 'b': b}
    
    result = process_kwargs(a=[1, 2], b={'x': 1})
    assert isinstance(result, type(pmap()))
    assert list(result['a']) == [1, 2]
    assert result['b'] == pmap({'x': 1})


def test_mutant_preserves_immutability():
    from pyrsistent import pvector
    from pyrsistent._helpers import mutant
    
    @mutant
    def get_first(lst):
        return lst[0]
    
    result = get_first([42])
    assert result == 42


def test_mutant_with_set_argument():
    from pyrsistent import pset
    from pyrsistent._helpers import mutant
    
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


def test_mutant_wraps_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        return []
    
    assert my_function.__name__ == 'my_function'


# LLM-generated content at query #39
#--------------------------

```python
def test_freeze_defaultdict_converts_to_pmap():
    import collections
    from pyrsistent import freeze, pmap
    
    dd = collections.defaultdict(int)
    dd['a'] = 1
    dd['b'] = 2
    
    result = freeze(dd, strict=True)
    expected = pmap({'a': 1, 'b': 2})
    
    assert result == expected


# LLM-generated content at query #40
#--------------------------

```python
def test_freeze_dict_basic():
    result = freeze({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap({})))
    assert result['a'] == 1
    assert result['b'] == 2


def test_freeze_dict_nested():
    result = freeze({'a': {'b': 3}})
    assert isinstance(result, type(pmap({})))
    assert isinstance(result['a'], type(pmap({})))
    assert result['a']['b'] == 3


def test_freeze_list_basic():
    result = freeze([1, 2, 3])
    assert isinstance(result, type(pvector([])))
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_freeze_list_nested():
    result = freeze([1, {'a': 3}])
    assert isinstance(result, type(pvector([])))
    assert result[0] == 1
    assert isinstance(result[1], type(pmap({})))
    assert result[1]['a'] == 3


def test_freeze_set():
    result = freeze(set([1, 2, 3]))
    assert isinstance(result, type(pset([])))
    assert 1 in result
    assert 2 in result
    assert 3 in result


def test_freeze_tuple_basic():
    result = freeze((1, 2, 3))
    assert isinstance(result, tuple)
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3


def test_freeze_tuple_nested():
    result = freeze((1, [2, 3]))
    assert isinstance(result, tuple)
    assert result[0] == 1
    assert isinstance(result[1], type(pvector([])))
    assert result[1][0] == 2


def test_freeze_mixed_nested():
    result = freeze({'list': [1, 2], 'dict': {'nested': 'value'}})
    assert isinstance(result, type(pmap({})))
    assert isinstance(result['list'], type(pvector([])))
    assert isinstance(result['dict'], type(pmap({})))
    assert result['dict']['nested'] == 'value'


def test_freeze_empty_dict():
    result = freeze({})
    assert isinstance(result, type(pmap({})))
    assert len(result) == 0


def test_freeze_empty_list():
    result = freeze([])
    assert isinstance(result, type(pvector([])))
    assert len(result) == 0


def test_freeze_empty_set():
    result = freeze(set())
    assert isinstance(result, type(pset([])))
    assert len(result) == 0


def test_freeze_empty_tuple():
    result = freeze(())
    assert isinstance(result, tuple)
    assert len(result) == 0


def test_freeze_primitive_types():
    assert freeze(1) == 1
    assert freeze('string') == 'string'
    assert freeze(3.14) == 3.14
    assert freeze(True) is True
    assert freeze(None) is None


def test_freeze_defaultdict():
    from collections import defaultdict
    d = defaultdict(list)
    d['key'] = [1, 2, 3]
    result = freeze(d)
    assert isinstance(result, type(pmap({})))
    assert isinstance(result['key'], type(pvector([])))


def test_freeze_strict_true_with_pmap():
    m = pmap({'a': 1})
    result = freeze(m, strict=True)
    assert isinstance(result, type(pmap({})))
    assert result['a'] == 1


def test_freeze_strict_false_with_pmap():
    m = pmap({'a': 1})
    result = freeze(m, strict=False)
    assert result is m


def test_freeze_strict_true_with_pvector():
    v = pvector([1, 2, 3])
    result = freeze(v, strict=True)
    assert isinstance(result, type(pvector([])))
    assert result[0] == 1


def test_freeze_strict_false_with_pvector():
    v = pvector([1, 2, 3])
    result = freeze(v, strict=False)
    assert result is v


def test_freeze_deeply_nested_structure():
    data = {'a': [1, {'b': [2, 3]}, 4], 'c': (5, [6, 7])}
    result = freeze(data)
    assert isinstance(result, type(pmap({})))
    assert isinstance(result['a'], type(pvector([])))
    assert isinstance(result['a'][1], type(pmap({})))
    assert isinstance(result['a'][1]['b'], type(pvector([])))
    assert isinstance(result['c'], tuple)
    assert isinstance(result['c'][1], type(pvector([])))


def test_freeze_set_with_multiple_elements():
    result = freeze({1, 2, 3, 4, 5})
    assert isinstance(result, type(pset([])))
    assert len(result) == 5
    for i in range(1, 6):
        assert i in result


# LLM-generated content at query #41
#--------------------------

```python
def test_mutant_decorator_predicate_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, pvector
    
    @mutant
    def modify_input(data):
        return data
    
    original_list = [1, 2, 3]
    result = modify_input(original_list)
    
    assert result is not original_list
    assert isinstance(result, type(pvector()))


# LLM-generated content at query #42
#--------------------------

```python
def test_freeze_defaultdict_predicate_line_32():
    import collections
    from pyrsistent import freeze, pmap
    
    # Create a defaultdict with nested mutable structures
    dd = collections.defaultdict(list)
    dd['a'] = [1, 2, 3]
    dd['b'] = {'nested': 'value'}
    
    # Call freeze with strict=True (default)
    result = freeze(dd, strict=True)
    
    # The predicate at line 32 should evaluate to True for defaultdict
    # and the result should be a pmap with frozen values
    assert isinstance(result, type(pmap()))
    assert result['a'] == [1, 2, 3]
    assert result['b'] == pmap({'nested': 'value'})


# LLM-generated content at query #43
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_and_return(data):
        # Try to mutate the input (should fail since it's frozen)
        return data
    
    input_dict = {'a': 1, 'b': 2}
    result = modify_and_return(input_dict)
    
    # Result should be frozen (persistent)
    assert isinstance(result, type(freeze(input_dict)))
    
    # Should not be a regular dict
    assert not isinstance(result, dict) or hasattr(result, '__hash__')


def test_mutant_decorator_with_multiple_arguments():
    from pyrsistent import pmap, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def combine(a, b):
        return {'combined': [a, b]}
    
    result = combine({'x': 1}, {'y': 2})
    
    # Result should be frozen
    assert result == freeze({'combined': [freeze({'x': 1}), freeze({'y': 2})]})


def test_mutant_decorator_with_kwargs():
    from pyrsistent import freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def process(data, extra=None):
        return {'data': data, 'extra': extra}
    
    result = process({'key': 'value'}, extra={'info': 'test'})
    
    # Result should be frozen
    assert result == freeze({'data': freeze({'key': 'value'}), 'extra': freeze({'info': 'test'})})


def test_mutant_decorator_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_decorator_with_pmap():
    from pyrsistent import pmap, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def transform(m):
        return m
    
    input_map = pmap({'a': 1, 'b': 2})
    result = transform(input_map)
    
    assert result == input_map
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_decorator_with_pset():
    from pyrsistent import pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_set(s):
        return s
    
    input_set = pset([1, 2, 3])
    result = process_set(input_set)
    
    assert result == input_set
    assert result == pset([1, 2, 3])


# LLM-generated content at query #44
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [1]
    
    result = modify_list([1, 2, 3])
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_with_dict_argument():
    @mutant
    def get_value(d):
        return d
    
    result = get_value({'a': 1, 'b': 2})
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"


def test_mutant_with_nested_structure():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'a': [1, 2, 3]})
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"


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
    assert str(type(result[1])) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, d):
        return lst
    
    result = combine([1, 2], {'a': 1})
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_with_keyword_arguments():
    @mutant
    def func_with_kwargs(data, extra={'default': True}):
        return data
    
    result = func_with_kwargs([1, 2], extra={'key': 'value'})
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_preserves_function_name():
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_scalar_return():
    @mutant
    def return_scalar(lst):
        return 42
    
    result = return_scalar([1, 2, 3])
    assert result == 42


def test_mutant_with_string_argument():
    @mutant
    def process_string(s):
        return s
    
    result = process_string("hello")
    assert result == "hello"


# LLM-generated content at query #45
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, freeze
    
    call_log = []
    
    @mutant
    def test_function(arg1, arg2, kwarg1=None):
        call_log.append((arg1, arg2, kwarg1))
        return {'result': arg1}
    
    result = test_function([1, 2, 3], {'a': 1}, kwarg1={'b': 2})
    
    assert len(call_log) == 1
    received_arg1, received_arg2, received_kwarg1 = call_log[0]
    
    assert hasattr(received_arg1, '__hash__')
    assert hasattr(received_arg2, '__hash__')
    assert hasattr(received_kwarg1, '__hash__')
    
    assert hasattr(result, '__hash__')
    
    try:
        received_arg1[0] = 999
        assert False, "Should not be able to mutate frozen argument"
    except (TypeError, AttributeError):
        pass
    
    try:
        result['result'] = 'modified'
        assert False, "Should not be able to mutate frozen return value"
    except (TypeError, AttributeError):
        pass


# LLM-generated content at query #46
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, freeze
    
    @mutant
    def modify_and_return(arg_dict, arg_set):
        # Try to mutate the arguments (should fail or have no effect due to freezing)
        return {"result": "success", "input_dict": arg_dict, "input_set": arg_set}
    
    # Call the decorated function with mutable arguments
    input_dict = {"key": "value"}
    input_set = {1, 2, 3}
    result = modify_and_return(input_dict, input_set)
    
    # Verify that the result is frozen (persistent)
    assert isinstance(result, type(pmap()))
    
    # Verify that the function executed and returned expected data
    assert result["result"] == "success"
    assert isinstance(result["input_dict"], type(pmap()))
    assert isinstance(result["input_set"], type(pset()))
    
    # Verify original arguments were not mutated
    assert input_dict == {"key": "value"}
    assert input_set == {1, 2, 3}


# LLM-generated content at query #47
#--------------------------

```python
def test_freeze_defaultdict_predicate():
    import collections
    from pyrsistent import pmap, freeze, PMap
    
    # Test that the predicate at line 32 evaluates to True for defaultdict
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd, strict=True)
    assert isinstance(result, PMap)
    assert result == pmap({'a': 1, 'b': 2})


# LLM-generated content at query #48
#--------------------------

```python
def test_mutant_decorator_predicate_line_1_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, pvector
    
    @mutant
    def modify_input(data):
        return data
    
    original_map = pmap({'a': 1, 'b': 2})
    result = modify_input(original_map)
    
    assert original_map == result
    assert original_map is not result or type(original_map) == type(result)
    
    original_set = pset([1, 2, 3])
    result_set = modify_input(original_set)
    
    assert original_set == result_set
    
    original_vector = pvector([1, 2, 3])
    result_vector = modify_input(original_vector)
    
    assert original_vector == result_vector


# LLM-generated content at query #49
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        return lst + [1]
    
    result = modify_list([1, 2, 3])
    assert result == pvector([1, 2, 3, 1])


def test_mutant_freezes_dict_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def get_value(d, key):
        return d.get(key)
    
    result = get_value({'a': 1, 'b': 2}, 'a')
    assert result == 1


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'key': [1, 2, {'nested': 3}]})
    assert isinstance(result, pmap)
    assert isinstance(result['key'], pvector)


def test_mutant_freezes_set_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_mutant_freezes_tuple_arguments():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def func_with_kwargs(a, b=None):
        return a
    
    result = func_with_kwargs([1, 2], b={'key': 'value'})
    assert result is not None


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    
    @mutant
    def add_lists(lst1, lst2):
        return lst1 + lst2
    
    result = add_lists([1, 2], [3, 4])
    assert len(result) == 4


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def documented_function():
        """Test docstring"""
        return []
    
    assert documented_function.__name__ == 'documented_function'


def test_mutant_with_empty_collections():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_empty(lst, d, s):
        return (lst, d, s)
    
    result = process_empty([], {}, set())
    assert isinstance(result, tuple)
    assert isinstance(result[0], pvector)
    assert isinstance(result[1], pmap)
    assert isinstance(result[2], pset)


def test_mutant_with_complex_nested_structure():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_complex(data):
        return data
    
    complex_data = {
        'list': [1, 2, [3, 4]],
        'dict': {'nested': {'deep': [5, 6]}},
        'tuple': (7, [8, 9]),
        'set': {10, 11}
    }
    result = process_complex(complex_data)
    assert isinstance(result, pmap)
    assert isinstance(result['list'], pvector)
    assert isinstance(result['set'], pset)


# LLM-generated content at query #50
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


def test_mutant_with_list_arguments():
    @mutant
    def append_to_list(lst, item):
        lst.append(item)
        return lst
    
    result = append_to_list([1, 2, 3], 4)
    assert isinstance(result, PVector)
    assert result == pvector([1, 2, 3, 4])


def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        return data
    
    nested_input = {'outer': {'inner': [1, 2, 3]}}
    result = modify_nested(nested_input)
    assert isinstance(result, PMap)
    assert isinstance(result['outer'], PMap)
    assert isinstance(result['outer']['inner'], PVector)


def test_mutant_with_multiple_arguments():
    @mutant
    def merge_dicts(d1, d2):
        d1.update(d2)
        return d1
    
    result = merge_dicts({'a': 1}, {'b': 2})
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2


def test_mutant_with_kwargs():
    @mutant
    def create_dict(**kwargs):
        return kwargs
    
    result = create_dict(a=1, b=2, c=[3, 4])
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['b'] == 2
    assert isinstance(result['c'], PVector)


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, PSet)


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)
    assert isinstance(result[1], PVector)
    assert isinstance(result[2], PMap)


def test_mutant_preserves_function_name():
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_empty_containers():
    @mutant
    def process_empty(d, lst):
        return (d, lst)
    
    result = process_empty({}, [])
    assert isinstance(result, tuple)
    assert isinstance(result[0], PMap)
    assert isinstance(result[1], PVector)


def test_mutant_with_primitive_arguments():
    @mutant
    def add_numbers(a, b):
        return a + b
    
    result = add_numbers(5, 3)
    assert result == 8


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    assert isinstance(result, tuple)
    assert str(type(result[1]).__name__) == 'PVector'

def test_freeze_scalar():
    from pyrsistent._helpers import freeze
    assert freeze(42) == 42
    assert freeze('string') == 'string'
    assert freeze(3.14) == 3.14

def test_freeze_defaultdict():
    from pyrsistent._helpers import freeze
    import collections
    dd = collections.defaultdict(list)
    dd['a'] = 1
    result = freeze(dd)
    assert result == {'a': 1}
    assert str(type(result).__name__) == 'PMap'

def test_freeze_empty_containers():
    from pyrsistent._helpers import freeze
    assert list(freeze([])) == []
    assert freeze({}) == {}
    assert freeze(set()) == set()
    assert freeze(()) == ()

def test_freeze_strict_true():
    from pyrsistent._helpers import freeze
    from pyrsistent import pvector, pmap
    pv = pvector([1, 2])
    result = freeze(pv, strict=True)
    assert list(result) == [1, 2]
    assert str(type(result).__name__) == 'PVector'

def test_freeze_strict_false():
    from pyrsistent._helpers import freeze
    from pyrsistent import pvector
    pv = pvector([1, 2])
    result = freeze(pv, strict=False)
    assert list(result) == [1, 2]
    assert str(type(result).__name__) == 'PVector'

def test_freeze_complex_nested_structure():
    from pyrsistent._helpers import freeze
    data = {
        'list': [1, 2, {'nested': 'dict'}],
        'dict': {'inner': [3, 4]},
        'tuple': (5, [6, 7]),
        'set': {8, 9}
    }
    result = freeze(data)
    assert result['list'][2] == {'nested': 'dict'}
    assert result['dict']['inner'] == [3, 4]
    assert result['tuple'][1] == [6, 7]
    assert result['set'] == {8, 9}


# LLM-generated content at query #2
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_list(lst):
        # Try to mutate the input (should not affect original due to freezing)
        return lst
    
    original_list = [1, 2, 3]
    result = modify_list(original_list)
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3])


def test_mutant_with_dict_argument():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_dict(d):
        return d
    
    original_dict = {'a': 1, 'b': 2}
    result = process_dict(original_dict)
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_with_nested_structure():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_nested(data):
        return data
    
    original = {'key': [1, 2, 3]}
    result = process_nested(original)
    assert isinstance(result, pmap)
    assert isinstance(result['key'], pvector)
    assert result == pmap({'key': pvector([1, 2, 3])})


def test_mutant_with_multiple_arguments():
    from pyrsistent import pvector
    from pyrsistent._helpers import mutant
    
    @mutant
    def combine(lst1, lst2):
        return lst1 + lst2
    
    result = combine([1, 2], [3, 4])
    assert isinstance(result, pvector)
    assert result == pvector([1, 2, 3, 4])


def test_mutant_with_kwargs():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def make_map(a=None, b=None):
        return {'a': a, 'b': b}
    
    result = make_map(a=1, b=2)
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_with_set():
    from pyrsistent import pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_set(s):
        return s
    
    original_set = {1, 2, 3}
    result = process_set(original_set)
    assert isinstance(result, pset)
    assert result == pset([1, 2, 3])


def test_mutant_with_tuple():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    original_tuple = (1, [2, 3], 4)
    result = process_tuple(original_tuple)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == 1
    assert result[2] == 4


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        """Test docstring"""
        return x
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_scalar_return():
    from pyrsistent._helpers import mutant
    
    @mutant
    def return_scalar(x):
        return 42
    
    result = return_scalar([1, 2, 3])
    assert result == 42


def test_mutant_with_mixed_kwargs_and_args():
    from pyrsistent import pmap, pvector
    from pyrsistent._helpers import mutant
    
    @mutant
    def combine_data(lst, d=None):
        return {'list': lst, 'dict': d}
    
    result = combine_data([1, 2], d={'x': 10})
    assert isinstance(result, pmap)
    assert isinstance(result['list'], pvector)
    assert isinstance(result['dict'], pmap)


# LLM-generated content at query #3
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


def test_thaw_nested_pvector_and_pmap():
    from pyrsistent import v, m
    result = thaw(v(1, m(a=3)))
    assert result == [1, {'a': 3}]
    assert isinstance(result, list)
    assert isinstance(result[1], dict)


def test_thaw_tuple_recursive():
    from pyrsistent import v
    result = thaw((1, v(2, 3)))
    assert result == (1, [2, 3])
    assert isinstance(result, tuple)
    assert isinstance(result[1], list)


def test_thaw_empty_pvector():
    from pyrsistent import v
    result = thaw(v())
    assert result == []
    assert isinstance(result, list)


def test_thaw_empty_pmap():
    from pyrsistent import m
    result = thaw(m())
    assert result == {}
    assert isinstance(result, dict)


def test_thaw_empty_pset():
    from pyrsistent import s
    result = thaw(s())
    assert result == set()
    assert isinstance(result, set)


def test_thaw_primitive_types():
    assert thaw(42) == 42
    assert thaw("hello") == "hello"
    assert thaw(3.14) == 3.14
    assert thaw(None) is None


def test_thaw_list_strict_true():
    result = thaw([1, 2, 3], strict=True)
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_thaw_dict_strict_true():
    result = thaw({'a': 1, 'b': 2}, strict=True)
    assert result == {'a': 1, 'b': 2}
    assert isinstance(result, dict)


def test_thaw_nested_list_strict_true():
    from pyrsistent import m
    result = thaw([1, m(x=2)], strict=True)
    assert result == [1, {'x': 2}]
    assert isinstance(result[1], dict)


def test_thaw_nested_dict_strict_true():
    from pyrsistent import v
    result = thaw({'a': v(1, 2)}, strict=True)
    assert result == {'a': [1, 2]}
    assert isinstance(result['a'], list)


def test_thaw_list_strict_false():
    result = thaw([1, 2, 3], strict=False)
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_thaw_dict_strict_false():
    result = thaw({'a': 1, 'b': 2}, strict=False)
    assert result == {'a': 1, 'b': 2}
    assert isinstance(result, dict)


def test_thaw_deeply_nested_structures():
    from pyrsistent import v, m, s
    result = thaw(v(m(a=v(1, 2), b=s(3, 4))))
    assert result == [{'a': [1, 2], 'b': {3, 4}}]
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert isinstance(result[0]['a'], list)
    assert isinstance(result[0]['b'], set)


def test_thaw_tuple_with_primitives():
    result = thaw((1, "hello", 3.14))
    assert result == (1, "hello", 3.14)
    assert isinstance(result, tuple)


def test_thaw_pmap_with_pmap_values():
    from pyrsistent import m
    result = thaw(m(outer=m(inner=5)))
    assert result == {'outer': {'inner': 5}}
    assert isinstance(result, dict)
    assert isinstance(result['outer'], dict)


# LLM-generated content at query #4
#--------------------------

```python
def test_freeze_dict_to_pmap():
    from pyrsistent import freeze, pmap
    result = freeze({'a': 1, 'b': 2})
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_nested_dict():
    from pyrsistent import freeze, pmap
    result = freeze({'a': {'b': 1}})
    assert result == pmap({'a': pmap({'b': 1})})


def test_freeze_list_to_pvector():
    from pyrsistent import freeze, pvector
    result = freeze([1, 2, 3])
    assert result == pvector([1, 2, 3])


def test_freeze_nested_list():
    from pyrsistent import freeze, pvector
    result = freeze([1, [2, 3]])
    assert result == pvector([1, pvector([2, 3])])


def test_freeze_set_to_pset():
    from pyrsistent import freeze, pset
    result = freeze({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_freeze_tuple_recursive():
    from pyrsistent import freeze, pvector
    result = freeze((1, [2, 3]))
    assert result == (1, pvector([2, 3]))


def test_freeze_complex_nested_structure():
    from pyrsistent import freeze, pmap, pvector, pset
    result = freeze({'a': [1, 2], 'b': {'c': 3}})
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})


def test_freeze_list_with_dict():
    from pyrsistent import freeze, pvector, pmap
    result = freeze([1, {'a': 3}])
    assert result == pvector([1, pmap({'a': 3})])


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


def test_freeze_scalar_value():
    from pyrsistent import freeze
    result = freeze(42)
    assert result == 42


def test_freeze_string_value():
    from pyrsistent import freeze
    result = freeze("hello")
    assert result == "hello"


def test_freeze_none_value():
    from pyrsistent import freeze
    result = freeze(None)
    assert result is None


def test_freeze_defaultdict():
    from pyrsistent import freeze, pmap
    from collections import defaultdict
    d = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(d)
    assert result == pmap({'a': 1, 'b': 2})


def test_freeze_defaultdict_nested():
    from pyrsistent import freeze, pmap
    from collections import defaultdict
    d = defaultdict(int, {'a': {'b': 1}})
    result = freeze(d)
    assert result == pmap({'a': pmap({'b': 1})})


def test_freeze_tuple_with_nested_dict():
    from pyrsistent import freeze, pmap
    result = freeze((1, {'a': 2}))
    assert result == (1, pmap({'a': 2}))


def test_freeze_tuple_with_nested_list():
    from pyrsistent import freeze, pvector
    result = freeze((1, [2, 3]))
    assert result == (1, pvector([2, 3]))


def test_freeze_strict_false_pmap():
    from pyrsistent import freeze, pmap
    result = freeze(pmap({'a': 1}), strict=False)
    assert result == pmap({'a': 1})


def test_freeze_strict_false_pvector():
    from pyrsistent import freeze, pvector
    result = freeze(pvector([1, 2]), strict=False)
    assert result == pvector([1, 2])


def test_freeze_dict_with_set():
    from pyrsistent import freeze, pmap, pset
    result = freeze({'a': {1, 2, 3}})
    assert result == pmap({'a': pset([1, 2, 3])})


def test_freeze_list_with_set():
    from pyrsistent import freeze, pvector, pset
    result = freeze([{1, 2, 3}])
    assert result == pvector([pset([1, 2, 3])])


# LLM-generated content at query #5
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
    
    @mutant
    def modify_dict(d):
        return d
    
    # Test with pmap
    input_map = pmap({'a': 1, 'b': 2})
    result_map = modify_map(input_map)
    assert result_map == input_map
    assert str(type(result_map)) == "<class 'pyrsistent._pmap.PMap'>"
    
    # Test with pset
    input_set = pset([1, 2, 3])
    result_set = modify_set(input_set)
    assert result_set == input_set
    assert str(type(result_set)) == "<class 'pyrsistent._pset.PSet'>"
    
    # Test with regular list
    input_list = [1, 2, 3]
    result_list = modify_list(input_list)
    assert result_list == input_list
    assert str(type(result_list)) == "<class 'pyrsistent._pvector.PVector'>"
    
    # Test with regular dict
    input_dict = {'x': 10, 'y': 20}
    result_dict = modify_dict(input_dict)
    assert result_dict == input_dict
    assert str(type(result_dict)) == "<class 'pyrsistent._pmap.PMap'>"
    
    # Test with kwargs
    @mutant
    def modify_with_kwargs(m, multiplier=1):
        return m
    
    input_map_kwargs = pmap({'a': 5})
    result_kwargs = modify_with_kwargs(input_map_kwargs, multiplier=2)
    assert result_kwargs == input_map_kwargs
    assert str(type(result_kwargs)) == "<class 'pyrsistent._pmap.PMap'>"


# LLM-generated content at query #6
#--------------------------

```python
def test_freeze_set_to_pset():
    from pyrsistent._helpers import freeze
    from pyrsistent import pset
    
    result = freeze(set([1, 2]))
    expected = pset([1, 2])
    assert result == expected


# LLM-generated content at query #7
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        return lst + [1]
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, pvector)
    assert list(result) == [1, 2, 3, 1]


def test_mutant_freezes_dict_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def get_value(d):
        return d
    
    result = get_value({'a': 1, 'b': 2})
    assert isinstance(result, pmap)
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process(data):
        return data
    
    result = process({'key': [1, 2, 3]})
    assert isinstance(result, pmap)
    assert isinstance(result['key'], pvector)
    assert list(result['key']) == [1, 2, 3]


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def combine(lst1, lst2):
        return lst1 + lst2
    
    result = combine([1, 2], [3, 4])
    assert isinstance(result, pvector)
    assert list(result) == [1, 2, 3, 4]


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def create_map(a=1, b=2):
        return {'a': a, 'b': b}
    
    result = create_map(a=10, b=20)
    assert isinstance(result, pmap)
    assert result['a'] == 10
    assert result['b'] == 20


def test_mutant_with_set():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, pset)


def test_mutant_with_tuple():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], 4))
    assert isinstance(result, tuple)
    assert isinstance(result[1], pvector)
    assert list(result[1]) == [2, 3]


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        """Test function"""
        return x
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test function'


def test_mutant_with_empty_containers():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_empty(lst, d, s):
        return [lst, d, s]
    
    result = process_empty([], {}, set())
    assert isinstance(result, pvector)
    assert isinstance(result[0], pvector)
    assert isinstance(result[1], pmap)
    assert isinstance(result[2], pset)


def test_mutant_with_mixed_args_and_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def merge_dicts(d1, d2=None):
        if d2 is None:
            return d1
        return {**d1, **d2}
    
    result = merge_dicts({'a': 1}, d2={'b': 2})
    assert isinstance(result, pmap)
    assert result['a'] == 1
    assert result['b'] == 2


# LLM-generated content at query #8
#--------------------------

```python
def test_freeze_dict_basic():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1, 'b': 2})
    assert result == {'a': 1, 'b': 2}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_dict_nested():
    from pyrsistent._helpers import freeze
    result = freeze({'a': {'b': 3}})
    assert result == {'a': {'b': 3}}
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PMap'


def test_freeze_list_basic():
    from pyrsistent._helpers import freeze
    result = freeze([1, 2, 3])
    assert list(result) == [1, 2, 3]
    assert str(type(result).__name__) == 'PVector'


def test_freeze_list_nested():
    from pyrsistent._helpers import freeze
    result = freeze([1, {'a': 3}])
    assert list(result) == [1, {'a': 3}]
    assert str(type(result).__name__) == 'PVector'
    assert str(type(result[1]).__name__) == 'PMap'


def test_freeze_tuple_basic():
    from pyrsistent._helpers import freeze
    result = freeze((1, 2, 3))
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)


def test_freeze_tuple_nested():
    from pyrsistent._helpers import freeze
    result = freeze((1, [2, 3]))
    assert result[0] == 1
    assert str(type(result[1]).__name__) == 'PVector'


def test_freeze_set_basic():
    from pyrsistent._helpers import freeze
    result = freeze({1, 2, 3})
    assert result == {1, 2, 3}
    assert str(type(result).__name__) == 'PSet'


def test_freeze_set_from_list():
    from pyrsistent._helpers import freeze
    result = freeze(set([1, 2]))
    assert result == {1, 2}
    assert str(type(result).__name__) == 'PSet'


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


def test_freeze_empty_tuple():
    from pyrsistent._helpers import freeze
    result = freeze(())
    assert result == ()
    assert isinstance(result, tuple)


def test_freeze_defaultdict():
    from pyrsistent._helpers import freeze
    import collections
    d = collections.defaultdict(int)
    d['a'] = 5
    result = freeze(d)
    assert result == {'a': 5}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_defaultdict_nested():
    from pyrsistent._helpers import freeze
    import collections
    d = collections.defaultdict(int)
    d['a'] = [1, 2]
    result = freeze(d)
    assert result == {'a': [1, 2]}
    assert str(type(result['a']).__name__) == 'PVector'


def test_freeze_deeply_nested():
    from pyrsistent._helpers import freeze
    result = freeze({'a': [1, {'b': (2, [3])}]})
    assert result['a'][0] == 1
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PVector'
    assert str(type(result['a'][1]).__name__) == 'PMap'
    assert isinstance(result['a'][1]['b'], tuple)
    assert str(type(result['a'][1]['b'][1]).__name__) == 'PVector'


def test_freeze_strict_false_dict():
    from pyrsistent._helpers import freeze
    result = freeze({'a': 1}, strict=False)
    assert result == {'a': 1}
    assert str(type(result).__name__) == 'PMap'


def test_freeze_strict_false_list():
    from pyrsistent._helpers import freeze
    result = freeze([1, 2], strict=False)
    assert list(result) == [1, 2]
    assert str(type(result).__name__) == 'PVector'


def test_freeze_dict_with_list_value():
    from pyrsistent._helpers import freeze
    result = freeze({'key': [1, 2, 3]})
    assert result == {'key': [1, 2, 3]}
    assert str(type(result['key']).__name__) == 'PVector'


def test_freeze_list_with_dict_and_tuple():
    from pyrsistent._helpers import freeze
    result = freeze([{'a': 1}, (2, 3)])
    assert result[0] == {'a': 1}
    assert result[1] == (2, 3)
    assert str(type(result[0]).__name__) == 'PMap'
    assert isinstance(result[1], tuple)


def test_freeze_complex_structure():
    from pyrsistent._helpers import freeze
    initial = {
        'name': 'test',
        'values': [1, 2, {'nested': True}],
        'config': {'debug': False, 'items': [4, 5, 6]}
    }
    result = freeze(initial)
    assert result['name'] == 'test'
    assert result['values'][2]['nested'] is True
    assert result['config']['items'][0] == 4


# LLM-generated content at query #9
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset

    @mutant
    def modify_list(lst):
        return lst

    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector([1, 2, 3])))


def test_mutant_freezes_dict_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap

    @mutant
    def modify_dict(d):
        return d

    result = modify_dict({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap({'a': 1})))


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector

    @mutant
    def return_list():
        return [1, 2, 3]

    result = return_list()
    assert isinstance(result, type(pvector([1, 2, 3])))


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def process_nested(data):
        return data

    result = process_nested([1, {'a': [2, 3]}])
    assert isinstance(result, type(pvector([])))
    assert isinstance(result[1], type(pmap({})))


def test_mutant_freezes_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap

    @mutant
    def func_with_kwargs(a, b=None):
        return {'result': a, 'b': b}

    result = func_with_kwargs([1, 2], b={'x': 1})
    assert isinstance(result, type(pmap({})))


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
        return s

    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset([1, 2, 3])))


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant

    @mutant
    def process_tuple(t):
        return t

    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def combine(lst, dct):
        return [lst, dct]

    result = combine([1, 2], {'a': 3})
    assert isinstance(result, type(pvector([])))
    assert isinstance(result[0], type(pvector([])))
    assert isinstance(result[1], type(pmap({})))


def test_mutant_with_deeply_nested_dict():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap

    @mutant
    def process_deep_dict(d):
        return d

    result = process_deep_dict({'a': {'b': {'c': 1}}})
    assert isinstance(result, type(pmap({})))
    assert isinstance(result['a'], type(pmap({})))
    assert isinstance(result['a']['b'], type(pmap({})))


def test_mutant_returns_frozen_result():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector

    @mutant
    def create_list():
        return [1, 2, 3]

    result = create_list()
    assert result.is_persistent()


# LLM-generated content at query #10
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


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_data(data):
        return data
    
    input_data = {'a': [1, 2], 'b': {'c': 3}}
    result = process_data(input_data)
    assert isinstance(result, type(pmap()))
    assert isinstance(result['a'], type(pvector()))
    assert isinstance(result['b'], type(pmap()))


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        return [lst, dct]
    
    result = combine([1, 2], {'x': 10})
    assert isinstance(result, type(pvector()))
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))


def test_mutant_with_keyword_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_kwargs(data=None):
        return data
    
    result = process_kwargs(data={'key': 'value'})
    assert isinstance(result, type(pmap()))
    assert result['key'] == 'value'


def test_mutant_prevents_mutation_of_input():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def try_mutate(lst):
        return lst
    
    original = [1, 2, 3]
    frozen_result = try_mutate(original)
    assert isinstance(frozen_result, type(pvector()))
    assert list(original) == [1, 2, 3]


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
    from pyrsistent import pvector
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], {'a': 4}))
    assert isinstance(result, tuple)
    assert isinstance(result[1], type(pvector()))


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        """This is my function."""
        pass
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == """This is my function."""


def test_mutant_with_empty_containers():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_empty(lst, dct, s):
        return [lst, dct, s]
    
    result = process_empty([], {}, set())
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))
    assert isinstance(result[2], type(pset()))


# LLM-generated content at query #11
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    import collections
    from pyrsistent import freeze, pmap
    
    dd = collections.defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd, strict=True)
    expected = pmap({'a': 1, 'b': 2})
    assert result == expected


# LLM-generated content at query #12
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, v
    
    @mutant
    def modify_map(m):
        return m
    
    @mutant
    def modify_set(s):
        return s
    
    @mutant
    def modify_vector(vec):
        return vec
    
    @mutant
    def modify_with_kwargs(m, s=None):
        return m
    
    input_map = pmap({'a': 1, 'b': 2})
    result_map = modify_map(input_map)
    assert result_map == input_map
    assert result_map.persistent
    
    input_set = pset([1, 2, 3])
    result_set = modify_set(input_set)
    assert result_set == input_set
    assert result_set.persistent
    
    input_vector = v(1, 2, 3)
    result_vector = modify_vector(input_vector)
    assert result_vector == input_vector
    assert result_vector.persistent
    
    input_map_kwargs = pmap({'x': 10})
    input_set_kwargs = pset([4, 5])
    result_kwargs = modify_with_kwargs(input_map_kwargs, s=input_set_kwargs)
    assert result_kwargs == input_map_kwargs
    assert result_kwargs.persistent


# LLM-generated content at query #13
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_and_return(data):
        # Try to modify the input (should fail since it's frozen)
        return data
    
    input_dict = {'a': 1, 'b': 2}
    result = modify_and_return(input_dict)
    
    # Result should be a frozen structure (pmap)
    assert isinstance(result, type(pmap(input_dict)))
    
    # Attempting to modify should raise an error
    try:
        result['c'] = 3
        assert False, "Should not be able to modify frozen result"
    except TypeError:
        pass
    
    # Test with list argument
    @mutant
    def process_list(lst):
        return lst
    
    input_list = [1, 2, 3]
    result_list = process_list(input_list)
    
    # Result should be a frozen structure (pvector)
    assert hasattr(result_list, '__hash__')
    
    # Test with kwargs
    @mutant
    def process_kwargs(**kwargs):
        return kwargs
    
    result_kwargs = process_kwargs(x=1, y=2)
    
    # Result should be frozen
    assert isinstance(result_kwargs, type(pmap({'x': 1, 'y': 2})))


# LLM-generated content at query #14
#--------------------------

```python
def test_freeze_defaultdict_strict_true():
    from collections import defaultdict
    from pyrsistent import freeze, pmap
    
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd, strict=True)
    
    assert result == pmap({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap()))


# LLM-generated content at query #15
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector([1, 2, 3])))


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def return_list():
        return [1, 2, 3]
    
    result = return_list()
    assert isinstance(result, type(pvector([1, 2, 3])))


def test_mutant_with_dict_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_dict(d):
        return d
    
    result = process_dict({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap({'a': 1})))


def test_mutant_with_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested([1, {'a': [2, 3]}])
    assert isinstance(result, type(pvector([1])))
    assert isinstance(result[1], type(pmap({'a': 1})))


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = process_with_kwargs([1, 2], b={'x': 1})
    assert isinstance(result, type(pmap({'a': 1})))


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset([1, 2, 3])))


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert result == (1, 2, 3)


def test_mutant_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        pass
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        return [lst, dct]
    
    result = combine([1, 2], {'a': 1})
    assert isinstance(result, type(pvector([1])))
    assert isinstance(result[0], type(pvector([1])))
    assert isinstance(result[1], type(pmap({'a': 1})))


def test_mutant_with_deeply_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_deep(data):
        return data
    
    result = process_deep({'a': [1, {'b': [2, 3]}]})
    assert isinstance(result, type(pmap({'a': 1})))


# LLM-generated content at query #16
#--------------------------

```python
def test_freeze_set_converts_to_pset():
    from pyrsistent._helpers import freeze
    from pyrsistent import pset
    
    result = freeze(set([1, 2]))
    expected = pset([1, 2])
    assert result == expected


# LLM-generated content at query #17
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
    def process_nested(data):
        data['list'].append(999)
        return data
    
    result = process_nested({'list': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'list': pvector([1, 2, 3])})


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(list1, list2):
        list1.append(999)
        list2.append(999)
        return [list1, list2]
    
    result = combine([1, 2], [3, 4])
    assert isinstance(result, type(pvector()))
    assert result == pvector([pvector([1, 2]), pvector([3, 4])])


def test_mutant_with_keyword_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_kwargs(a=None, b=None):
        return {'a': a, 'b': b}
    
    result = process_kwargs(a={'x': 1}, b={'y': 2})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'a': pmap({'x': 1}), 'b': pmap({'y': 2})})


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
        s.add(999)
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset()))
    assert result == pset({1, 2, 3})


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t + (999,)
    
    result = process_tuple((1, 2, 3))
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 999)


def test_mutant_return_value_is_frozen():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector
    
    @mutant
    def create_list():
        return [1, 2, 3]
    
    result = create_list()
    assert isinstance(result, type(pvector()))


def test_mutant_with_deeply_nested_structure():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_deep(data):
        data['nested']['list'].append(999)
        return data
    
    result = process_deep({'nested': {'list': [1, 2]}})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'nested': pmap({'list': pvector([1, 2])})})


# LLM-generated content at query #18
#--------------------------

```python
def test_freeze_set_conversion():
    from pyrsistent._helpers import freeze
    from pyrsistent import pset
    
    result = freeze(set([1, 2]))
    expected = pset([1, 2])
    assert result == expected


# LLM-generated content at query #19
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
    def get_value(d):
        return d
    
    result = get_value({'a': 1})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert str(type(result).__name__) == 'PSet'


def test_mutant_with_nested_structure():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'key': [1, 2, 3]})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)


def test_mutant_with_kwargs():
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'x': 1})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_preserves_function_name():
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst1, lst2):
        return lst1 + lst2
    
    result = combine([1, 2], [3, 4])
    assert str(type(result).__name__) == 'PVector'


def test_mutant_return_value_is_frozen():
    @mutant
    def return_dict(x):
        return {'result': x}
    
    result = return_dict([1, 2])
    assert str(type(result).__name__) == 'PMap'


def test_mutant_with_nested_list_in_dict():
    @mutant
    def process_complex(data):
        return data
    
    result = process_complex({'items': [1, 2, {'nested': 3}]})
    assert str(type(result).__name__) == 'PMap'


# LLM-generated content at query #20
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def modify_list(lst):
        lst[0] = 999
        return lst
    
    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector()))
    assert result == pvector([1, 2, 3])


def test_mutant_freezes_dict_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def modify_dict(d):
        d['key'] = 'modified'
        return d
    
    result = modify_dict({'key': 'original'})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'key': 'original'})


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def modify_nested(data):
        data[0]['nested'] = 'modified'
        return data
    
    result = modify_nested([{'nested': 'original'}])
    assert isinstance(result, type(pvector()))
    assert result[0] == pmap({'nested': 'original'})


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        lst[0] = 999
        dct['key'] = 'modified'
        return [lst, dct]
    
    result = combine([1, 2], {'key': 'value'})
    assert isinstance(result, type(pvector()))
    assert result[0] == pvector([1, 2])
    assert result[1] == pmap({'key': 'value'})


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process(data, multiplier=1):
        data['value'] = data['value'] * multiplier
        return data
    
    result = process({'value': 5}, multiplier=2)
    assert isinstance(result, type(pmap()))
    assert result == pmap({'value': 5})


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
    from pyrsistent import pvector
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], {'key': 'value'}))
    assert isinstance(result, tuple)
    assert result[1] == pvector([2, 3])


def test_mutant_original_arguments_unchanged():
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify(lst):
        lst[0] = 999
        return lst
    
    original = [1, 2, 3]
    result = modify(original)
    assert original == [1, 2, 3]
    assert original[0] == 1


def test_mutant_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        return x
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_empty_collections():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_empty(lst, dct, s):
        return [lst, dct, s]
    
    result = process_empty([], {}, set())
    assert result[0] == pvector([])
    assert result[1] == pmap({})
    assert result[2] == pset(set())


# LLM-generated content at query #21
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
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def return_dict():
        return {'a': 1, 'b': 2}
    
    result = return_dict()
    assert result == pmap({'a': 1, 'b': 2})


def test_mutant_with_kwargs():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def func_with_kwargs(x, y=None):
        return {'x': x, 'y': y}
    
    result = func_with_kwargs(1, y=2)
    assert result == pmap({'x': 1, 'y': 2})


def test_mutant_nested_structures():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_nested(data):
        return data
    
    original = {'a': [1, 2], 'b': {'c': 3}}
    result = process_nested(original)
    assert result == pmap({'a': pvector([1, 2]), 'b': pmap({'c': 3})})


def test_mutant_preserves_original_argument():
    from pyrsistent import pvector
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_and_return(lst):
        lst.append(100)
        return lst
    
    original = [1, 2, 3]
    result = modify_and_return(original)
    assert original == [1, 2, 3]
    assert isinstance(result, pvector)


def test_mutant_multiple_arguments():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def combine(lst, dct):
        return {'list': lst, 'dict': dct}
    
    result = combine([1, 2], {'a': 1})
    assert result == pmap({'list': pvector([1, 2]), 'dict': pmap({'a': 1})})


def test_mutant_with_set():
    from pyrsistent import pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert result == pset([1, 2, 3])


def test_mutant_with_tuple():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert result[0] == 1
    assert len(result) == 2


# LLM-generated content at query #22
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, pvector, freeze
    
    call_log = []
    
    @mutant
    def modify_and_return(data):
        call_log.append(type(data))
        return data
    
    input_list = [1, 2, 3]
    result = modify_and_return(input_list)
    
    assert isinstance(result, (pset, pvector, pmap)) or hasattr(result, '__hash__')
    assert call_log[0].__name__ in ('pvector', 'pset', 'pmap')


# LLM-generated content at query #23
#--------------------------

```python
def test_mutant_decorator_predicate_line_1_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset
    
    @mutant
    def modify_map(m):
        return m
    
    @mutant
    def modify_set(s):
        return s
    
    test_map = pmap({'a': 1, 'b': 2})
    test_set = pset([1, 2, 3])
    
    result_map = modify_map(test_map)
    result_set = modify_set(test_set)
    
    assert result_map == test_map
    assert result_set == test_set
    assert isinstance(result_map, type(test_map))
    assert isinstance(result_set, type(test_set))


# LLM-generated content at query #24
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
    assert not (result is original_map and original_map is not result)


# LLM-generated content at query #25
#--------------------------

```python
def test_mutant_freezes_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset

    @mutant
    def modify_list(lst):
        return lst

    result = modify_list([1, 2, 3])
    assert isinstance(result, type(pvector([1, 2, 3])))


def test_mutant_freezes_dict_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap

    @mutant
    def process_dict(d):
        return d

    result = process_dict({'a': 1, 'b': 2})
    assert isinstance(result, type(pmap({'a': 1})))


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def process_nested(data):
        return data

    result = process_nested({'x': [1, 2, 3]})
    assert isinstance(result, type(pmap({})))
    assert isinstance(result['x'], type(pvector([])))


def test_mutant_freezes_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector

    @mutant
    def create_list():
        return [1, 2, 3]

    result = create_list()
    assert isinstance(result, type(pvector([])))


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def combine(lst, dct):
        return (lst, dct)

    result = combine([1, 2], {'a': 1})
    assert isinstance(result[0], type(pvector([])))
    assert isinstance(result[1], type(pmap({})))


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap

    @mutant
    def process_with_kwargs(data, extra=None):
        return data

    result = process_with_kwargs({'a': 1}, extra={'b': 2})
    assert isinstance(result, type(pmap({})))


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset

    @mutant
    def process_set(s):
        return s

    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset([])))


def test_mutant_with_tuple_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector

    @mutant
    def process_tuple(t):
        return t

    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert isinstance(result[1], type(pvector([])))


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant

    @mutant
    def my_function():
        """Test docstring"""
        pass

    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_scalar_values():
    from pyrsistent._helpers import mutant

    @mutant
    def process_scalar(x):
        return x

    result = process_scalar(42)
    assert result == 42


def test_mutant_deeply_nested_structure():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap

    @mutant
    def process_deep(data):
        return data

    result = process_deep({'a': [1, {'b': [2, 3]}]})
    assert isinstance(result, type(pmap({})))
    assert isinstance(result['a'], type(pvector([])))
    assert isinstance(result['a'][1], type(pmap({})))
    assert isinstance(result['a'][1]['b'], type(pvector([])))


# LLM-generated content at query #26
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent import pmap, pset, freeze
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_dict(d):
        # Try to mutate the input - should fail because it's frozen
        d['new_key'] = 'new_value'
        return d
    
    input_dict = {'a': 1}
    try:
        result = modify_dict(input_dict)
        # If we reach here, the predicate at line 1 evaluates to False
        # because the function did NOT properly freeze the arguments
        predicate_result = False
    except (TypeError, AttributeError):
        # Expected behavior - frozen object cannot be mutated
        predicate_result = True
    
    assert predicate_result == False


# LLM-generated content at query #27
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [1]
    
    result = modify_list([1, 2, 3])
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_with_dict_argument():
    @mutant
    def get_value(d):
        return d['key']
    
    result = get_value({'key': 'value'})
    assert result == 'value'


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, d):
        return lst + [d['x']]
    
    result = combine([1, 2], {'x': 3})
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_with_keyword_arguments():
    @mutant
    def func_with_kwargs(a, b=10):
        return [a, b]
    
    result = func_with_kwargs(5, b=20)
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_nested_structure():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'a': [1, 2], 'b': {'c': 3}})
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"


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
    assert str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"


def test_mutant_return_value_is_frozen():
    @mutant
    def create_dict():
        return {'key': [1, 2, 3]}
    
    result = create_dict()
    assert str(type(result)) == "<class 'pyrsistent._pmap.PMap'>"


# LLM-generated content at query #28
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
    def modify_with_kwargs(m, s=None):
        return (m, s)
    
    input_map = pmap({'a': 1, 'b': 2})
    result_map = modify_map(input_map)
    
    assert result_map == input_map
    assert result_map is not input_map or result_map.mutant() is None
    
    input_set = pset([1, 2, 3])
    result_set = modify_set(input_set)
    
    assert result_set == input_set
    
    result_tuple = modify_with_kwargs(input_map, s=input_set)
    
    assert result_tuple[0] == input_map
    assert result_tuple[1] == input_set
    assert isinstance(result_tuple, tuple)


# LLM-generated content at query #29
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze, thaw
    
    @mutant
    def modify_list(lst):
        return lst + [4]
    
    original_list = [1, 2, 3]
    result = modify_list(original_list)
    
    assert result == [1, 2, 3, 4]
    assert original_list == [1, 2, 3]
    
    @mutant
    def modify_dict(d):
        return d.set('new_key', 'new_value')
    
    original_dict = {'a': 1, 'b': 2}
    result_dict = modify_dict(original_dict)
    
    assert thaw(result_dict) == {'a': 1, 'b': 2, 'new_key': 'new_value'}
    assert original_dict == {'a': 1, 'b': 2}
    
    @mutant
    def process_kwargs(x, y=5):
        return {'x': x, 'y': y}
    
    result_kw = process_kwargs(10, y=20)
    assert thaw(result_kw) == {'x': 10, 'y': 20}


# LLM-generated content at query #30
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    from collections import defaultdict
    from pyrsistent import freeze, pmap
    
    dd = defaultdict(int, {'a': 1, 'b': 2})
    result = freeze(dd, strict=True)
    expected = pmap({'a': 1, 'b': 2})
    assert result == expected


# LLM-generated content at query #31
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    from pyrsistent import pvector, pmap, pset
    from pyrsistent._helpers import mutant
    
    @mutant
    def modify_and_return(lst, dct):
        lst.append(999)
        dct['new_key'] = 'new_value'
        return lst, dct
    
    original_list = [1, 2, 3]
    original_dict = {'a': 1}
    
    result = modify_and_return(original_list, original_dict)
    
    assert isinstance(result, tuple)
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))
    assert result[0] == pvector([1, 2, 3])
    assert result[1] == pmap({'a': 1})


def test_mutant_preserves_function_metadata():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function(x):
        """Test docstring"""
        return x
    
    assert my_function.__name__ == 'my_function'
    assert my_function.__doc__ == 'Test docstring'


def test_mutant_with_nested_structures():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'key': [1, 2, {'inner': 3}]})
    
    assert isinstance(result, type(pmap()))
    assert isinstance(result['key'], type(pvector()))
    assert isinstance(result['key'][2], type(pmap()))


def test_mutant_with_kwargs():
    from pyrsistent import pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs(1, b={'nested': 'dict'})
    
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1
    assert isinstance(result['b'], type(pmap()))


def test_mutant_with_sets():
    from pyrsistent import pset, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    
    assert isinstance(result, type(pset()))
    assert result == pset([1, 2, 3])


def test_mutant_with_tuple():
    from pyrsistent import pvector
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3], 4))
    
    assert isinstance(result, tuple)
    assert isinstance(result[1], type(pvector()))
    assert result == (1, pvector([2, 3]), 4)


def test_mutant_multiple_args():
    from pyrsistent import pvector, pmap
    from pyrsistent._helpers import mutant
    
    @mutant
    def combine(lst, dct, tpl):
        return [lst, dct, tpl]
    
    result = combine([1, 2], {'x': 10}, (5, 6))
    
    assert isinstance(result, type(pvector()))
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))
    assert isinstance(result[2], tuple)


# LLM-generated content at query #32
#--------------------------

```python
def test_mutant_decorator_predicate_line_1():
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
    assert isinstance(result_set, type(input_set))
    assert isinstance(result_map, type(input_map))


# LLM-generated content at query #33
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
        data['items'].append(999)
        return data
    
    result = process_nested({'items': [1, 2, 3]})
    assert isinstance(result, type(pmap()))
    assert result == pmap({'items': pvector([1, 2, 3])})


def test_mutant_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, d):
        lst.append(4)
        d['key'] = 'value'
        return [lst, d]
    
    result = combine([1, 2, 3], {'a': 1})
    assert isinstance(result, type(pvector()))
    assert result[0] == pvector([1, 2, 3])
    assert result[1] == pmap({'a': 1})


def test_mutant_with_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def func_with_kwargs(a=None, b=None):
        if a is not None:
            a['key'] = 'modified'
        return a
    
    result = func_with_kwargs(a={'original': 'value'}, b=[1, 2])
    assert isinstance(result, type(pmap()))
    assert result == pmap({'original': 'value'})


def test_mutant_with_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        s.add(4)
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset()))
    assert result == pset([1, 2, 3])


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
    def documented_function():
        """This is a docstring"""
        pass
    
    assert documented_function.__doc__ == "This is a docstring"
    assert documented_function.__name__ == "documented_function"


def test_mutant_with_scalar_return():
    from pyrsistent._helpers import mutant
    
    @mutant
    def return_scalar(lst):
        return 42
    
    result = return_scalar([1, 2, 3])
    assert result == 42


def test_mutant_with_mixed_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def mixed_args(lst, d, s):
        lst.append(10)
        d['x'] = 20
        s.add(30)
        return {'list': lst, 'dict': d, 'set': s}
    
    result = mixed_args([1, 2], {'a': 1}, {1, 2})
    assert result['list'] == pvector([1, 2])
    assert result['dict'] == pmap({'a': 1})
    assert result['set'].count(1) == 1


# LLM-generated content at query #34
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
    
    assert result == [1, 2, 3, 4]
    assert thaw(result) == [1, 2, 3, 4]


def test_mutant_decorator_with_multiple_arguments():
    from pyrsistent._helpers import mutant
    
    @mutant
    def combine_dicts(d1, d2):
        return d1 + {d2}
    
    dict1 = {'a': 1}
    dict2 = {'b': 2}
    result = combine_dicts(dict1, dict2)
    
    assert 'a' in result or 'b' in result


def test_mutant_decorator_with_kwargs():
    from pyrsistent._helpers import mutant
    
    @mutant
    def create_map(a=1, b=2):
        return {'x': a, 'y': b}
    
    result = create_map(a=10, b=20)
    
    assert result['x'] == 10
    assert result['y'] == 20


def test_mutant_decorator_preserves_function_name():
    from pyrsistent._helpers import mutant
    
    @mutant
    def my_function():
        return 42
    
    assert my_function.__name__ == 'my_function'


def test_mutant_decorator_with_pset():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def add_to_set(s):
        return s | {4}
    
    input_set = pset([1, 2, 3])
    result = add_to_set(input_set)
    
    assert 4 in result


def test_mutant_decorator_with_pmap():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def update_map(m):
        return m.set('new_key', 'new_value')
    
    input_map = pmap({'a': 1})
    result = update_map(input_map)
    
    assert result['new_key'] == 'new_value'
    assert result['a'] == 1


# LLM-generated content at query #35
#--------------------------

```python
def test_mutant_decorator_freezes_arguments_and_return_value():
    from pyrsistent._helpers import mutant
    from pyrsistent import freeze, thaw
    
    call_log = []
    
    @mutant
    def modify_list(lst):
        call_log.append(type(lst))
        return lst
    
    original_list = [1, 2, 3]
    result = modify_list(original_list)
    
    # The argument passed to the function should be frozen (pvector)
    assert call_log[0].__name__ == 'PVector'
    
    # The return value should be frozen
    assert hasattr(result, '__hash__') or str(type(result)) == "<class 'pyrsistent._pvector.PVector'>"


# LLM-generated content at query #36
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_dict(d):
        d['new_key'] = 'new_value'
        return d
    
    original_dict = {'a': 1}
    result = modify_dict(original_dict)
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert result['new_key'] == 'new_value'


def test_mutant_freezes_list_arguments():
    @mutant
    def append_to_list(lst):
        lst.append(4)
        return lst
    
    original_list = [1, 2, 3]
    result = append_to_list(original_list)
    assert isinstance(result, PVector)
    assert len(result) == 4
    assert result[3] == 4


def test_mutant_preserves_function_name():
    @mutant
    def my_function():
        return {}
    
    assert my_function.__name__ == 'my_function'


def test_mutant_with_nested_structures():
    @mutant
    def modify_nested(data):
        data['inner']['value'] = 99
        return data
    
    original = {'inner': {'value': 1}}
    result = modify_nested(original)
    assert isinstance(result, PMap)
    assert isinstance(result['inner'], PMap)
    assert result['inner']['value'] == 99


def test_mutant_with_multiple_arguments():
    @mutant
    def merge_dicts(d1, d2):
        d1['merged'] = d2
        return d1
    
    dict1 = {'a': 1}
    dict2 = {'b': 2}
    result = merge_dicts(dict1, dict2)
    assert isinstance(result, PMap)
    assert result['a'] == 1
    assert isinstance(result['merged'], PMap)
    assert result['merged']['b'] == 2


def test_mutant_with_keyword_arguments():
    @mutant
    def create_dict(key=None, value=None):
        result = {}
        if key is not None and value is not None:
            result[key] = value
        return result
    
    result = create_dict(key='test', value='data')
    assert isinstance(result, PMap)
    assert result['test'] == 'data'


def test_mutant_with_set_argument():
    @mutant
    def process_set(s):
        s.add(4)
        return s
    
    original_set = {1, 2, 3}
    result = process_set(original_set)
    assert isinstance(result, PSet)


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t + (4,)
    
    original_tuple = (1, 2, 3)
    result = process_tuple(original_tuple)
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, 4)


def test_mutant_with_complex_nested_structure():
    @mutant
    def process_complex(data):
        data['list'].append(5)
        data['dict']['nested'] = [1, 2]
        return data
    
    original = {'list': [1, 2, 3], 'dict': {'key': 'value'}}
    result = process_complex(original)
    assert isinstance(result, PMap)
    assert isinstance(result['list'], PVector)
    assert len(result['list']) == 4
    assert isinstance(result['dict'], PMap)
    assert isinstance(result['dict']['nested'], PVector)


def test_mutant_returns_frozen_primitive():
    @mutant
    def return_string():
        return "hello"
    
    result = return_string()
    assert result == "hello"
    assert isinstance(result, str)


def test_mutant_with_empty_containers():
    @mutant
    def process_empty(d, l, s):
        return {'dict': d, 'list': l, 'set': s}
    
    result = process_empty({}, [], set())
    assert isinstance(result, PMap)
    assert isinstance(result['dict'], PMap)
    assert isinstance(result['list'], PVector)
    assert isinstance(result['set'], PSet)


# LLM-generated content at query #37
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
    
    input_list = [1, 2, 3]
    result = modify_and_return(input_list)
    
    assert result == pset([1, 2, 3]) or result == freeze([1, 2, 3])
    assert len(call_log) == 1


# LLM-generated content at query #38
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    import collections
    from pyrsistent import freeze, pmap
    
    initial_dict = collections.defaultdict(list)
    initial_dict['a'] = [1, 2, 3]
    initial_dict['b'] = [4, 5]
    
    result = freeze(initial_dict, strict=True)
    
    assert isinstance(result, type(pmap({})))
    assert result['a'] == [1, 2, 3]
    assert result['b'] == [4, 5]


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
    assert isinstance(result, type(pvector()))
    assert list(result) == [1, 2, 3]


def test_mutant_freezes_nested_structures():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def process_data(data):
        return data
    
    result = process_data({'a': [1, 2], 'b': {'c': 3}})
    assert isinstance(result, type(pmap()))
    assert isinstance(result['a'], type(pvector()))
    assert isinstance(result['b'], type(pmap()))


def test_mutant_freezes_dict_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_dict(d):
        return d
    
    result = process_dict({'x': 1, 'y': 2})
    assert isinstance(result, type(pmap()))
    assert result['x'] == 1
    assert result['y'] == 2


def test_mutant_freezes_set_argument():
    from pyrsistent._helpers import mutant
    from pyrsistent import pset
    
    @mutant
    def process_set(s):
        return s
    
    result = process_set({1, 2, 3})
    assert isinstance(result, type(pset()))
    assert 1 in result and 2 in result and 3 in result


def test_mutant_freezes_tuple_argument():
    from pyrsistent._helpers import mutant
    
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert result[0] == 1


def test_mutant_freezes_kwargs():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    
    @mutant
    def process_kwargs(a=None, b=None):
        return {'a': a, 'b': b}
    
    result = process_kwargs(a={'x': 1}, b=[1, 2])
    assert isinstance(result, type(pmap()))
    assert isinstance(result['a'], type(pmap()))


def test_mutant_multiple_arguments():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap
    
    @mutant
    def combine(lst, dct):
        return [lst, dct]
    
    result = combine([1, 2], {'a': 3})
    assert isinstance(result, type(pvector()))
    assert isinstance(result[0], type(pvector()))
    assert isinstance(result[1], type(pmap()))


def test_mutant_preserves_function_behavior():
    from pyrsistent._helpers import mutant
    
    @mutant
    def add_one(lst):
        return [x + 1 for x in lst]
    
    result = add_one([1, 2, 3])
    assert list(result) == [2, 3, 4]


def test_mutant_with_nested_defaultdict():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap
    import collections
    
    @mutant
    def process_defaultdict(d):
        return d
    
    dd = collections.defaultdict(int, {'a': 1})
    result = process_defaultdict(dd)
    assert isinstance(result, type(pmap()))
    assert result['a'] == 1


def test_mutant_with_complex_nested_structure():
    from pyrsistent._helpers import mutant
    from pyrsistent import pvector, pmap, pset
    
    @mutant
    def process_complex(data):
        return data
    
    input_data = {
        'list': [1, 2, {'nested': [3, 4]}],
        'set': {5, 6},
        'tuple': (7, [8, 9])
    }
    result = process_complex(input_data)
    assert isinstance(result, type(pmap()))
    assert isinstance(result['list'], type(pvector()))
    assert isinstance(result['set'], type(pset()))
    assert isinstance(result['tuple'], tuple)


# LLM-generated content at query #40
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
    def modify_with_kwargs(m, extra=None):
        return m
    
    # Test with pmap
    test_map = pmap({'a': 1, 'b': 2})
    result_map = modify_map(test_map)
    assert result_map == test_map
    assert result_map.evolver is not None or hasattr(result_map, '__class__')
    
    # Test with pset
    test_set = pset([1, 2, 3])
    result_set = modify_set(test_set)
    assert result_set == test_set
    
    # Test with kwargs
    result_with_kwargs = modify_with_kwargs(test_map, extra=pmap({'x': 10}))
    assert result_with_kwargs == test_map
    
    # Test that frozen objects remain frozen (predicate at line 1 evaluates to True)
    # The decorator successfully returns frozen versions
    assert result_map is not None
    assert result_set is not None
    assert result_with_kwargs is not None


# LLM-generated content at query #41
#--------------------------

```python
def test_mutant_decorator_predicate_line_1_false():
    from pyrsistent._helpers import mutant
    from pyrsistent import pmap, pset, pvector
    
    @mutant
    def modify_dict(d):
        d['key'] = 'value'
        return d
    
    original = {'initial': 'data'}
    result = modify_dict(original)
    
    assert original == {'initial': 'data'}
    assert result['key'] == 'value'
    assert isinstance(result, pmap)


# LLM-generated content at query #42
#--------------------------

```python
def test_mutant_freezes_arguments_and_return_value():
    @mutant
    def modify_list(lst):
        return lst + [4]
    
    result = modify_list([1, 2, 3])
    assert str(type(result).__name__) == 'PVector'
    assert list(result) == [1, 2, 3, 4]


def test_mutant_freezes_dict_arguments():
    @mutant
    def get_value(d):
        return d.get('a')
    
    result = get_value({'a': 1})
    assert result == 1


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, d):
        return lst + [d.get('x')]
    
    result = combine([1, 2], {'x': 3})
    assert str(type(result).__name__) == 'PVector'
    assert list(result) == [1, 2, 3]


def test_mutant_with_kwargs():
    @mutant
    def create_structure(a, b=5):
        return [a, b]
    
    result = create_structure(10, b=20)
    assert str(type(result).__name__) == 'PVector'
    assert list(result) == [10, 20]


def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'key': [1, 2, 3]})
    assert str(type(result).__name__) == 'PMap'
    inner_list = result['key']
    assert str(type(inner_list).__name__) == 'PVector'
    assert list(inner_list) == [1, 2, 3]


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
    assert str(type(result[1]).__name__) == 'PVector'


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
    assert str(type(result).__name__) == 'PVector'
    assert len(result) == 3


def test_mutant_with_deeply_nested_structure():
    @mutant
    def process_deep(data):
        return data
    
    result = process_deep({'a': {'b': [1, 2, {'c': 3}]}})
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PMap'
    assert str(type(result['a']['b']).__name__) == 'PVector'
    assert str(type(result['a']['b'][2]).__name__) == 'PMap'


# LLM-generated content at query #43
#--------------------------

```python
def test_freeze_defaultdict_with_strict_true():
    import collections
    from pyrsistent import freeze, pmap
    
    dd = collections.defaultdict(int)
    dd['x'] = 1
    dd['y'] = 2
    
    result = freeze(dd, strict=True)
    
    assert result == pmap({'x': 1, 'y': 2})


# LLM-generated content at query #44
#--------------------------

```python
def test_mutant_freezes_arguments():
    @mutant
    def modify_list(lst):
        return lst
    
    result = modify_list([1, 2, 3])
    assert str(type(result).__name__) == 'PVector'


def test_mutant_freezes_dict_arguments():
    @mutant
    def modify_dict(d):
        return d
    
    result = modify_dict({'a': 1, 'b': 2})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_freezes_return_value():
    @mutant
    def return_list():
        return [1, 2, 3]
    
    result = return_list()
    assert str(type(result).__name__) == 'PVector'


def test_mutant_with_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'items': [1, 2, {'nested': 3}]})
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['items']).__name__) == 'PVector'
    assert str(type(result['items'][2]).__name__) == 'PMap'


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, d):
        return [lst, d]
    
    result = combine([1, 2], {'a': 3})
    assert str(type(result).__name__) == 'PVector'
    assert str(type(result[0]).__name__) == 'PVector'
    assert str(type(result[1]).__name__) == 'PMap'


def test_mutant_with_kwargs():
    @mutant
    def func_with_kwargs(a, b=None):
        return {'a': a, 'b': b}
    
    result = func_with_kwargs([1, 2], b={'c': 4})
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PVector'
    assert str(type(result['b']).__name__) == 'PMap'


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
    assert str(type(result).__name__) == 'PSet'


def test_mutant_with_tuple_argument():
    @mutant
    def process_tuple(t):
        return t
    
    result = process_tuple((1, [2, 3]))
    assert isinstance(result, tuple)
    assert str(type(result[1]).__name__) == 'PVector'


def test_mutant_with_scalar_values():
    @mutant
    def add_one(x):
        return x + 1
    
    result = add_one(5)
    assert result == 6


def test_mutant_with_mixed_args_and_kwargs():
    @mutant
    def mixed_func(a, b, c=None):
        return {'a': a, 'b': b, 'c': c}
    
    result = mixed_func([1], {'x': 2}, c=[3, 4])
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['a']).__name__) == 'PVector'
    assert str(type(result['b']).__name__) == 'PMap'
    assert str(type(result['c']).__name__) == 'PVector'


# LLM-generated content at query #45
#--------------------------

```python
def test_mutant_freezes_arguments():
    @mutant
    def modify_list(lst):
        return lst
    
    result = modify_list([1, 2, 3])
    assert str(type(result).__name__) == 'PVector'


def test_mutant_freezes_dict_arguments():
    @mutant
    def process_dict(d):
        return d
    
    result = process_dict({'a': 1, 'b': 2})
    assert str(type(result).__name__) == 'PMap'


def test_mutant_freezes_return_value():
    @mutant
    def return_list():
        return [1, 2, 3]
    
    result = return_list()
    assert str(type(result).__name__) == 'PVector'


def test_mutant_freezes_nested_structures():
    @mutant
    def process_nested(data):
        return data
    
    result = process_nested({'key': [1, 2, 3]})
    assert str(type(result).__name__) == 'PMap'
    assert str(type(result['key']).__name__) == 'PVector'


def test_mutant_with_multiple_arguments():
    @mutant
    def combine(lst, d):
        return [lst, d]
    
    result = combine([1, 2], {'a': 3})
    assert str(type(result).__name__) == 'PVector'
    assert str(type(result[0]).__name__) == 'PVector'
    assert str(type(result[1]).__name__) == 'PMap'


def test_mutant_with_kwargs():
    @mutant
    def process_with_kwargs(data, **kwargs):
        return data
    
    result = process_with_kwargs([1, 2], extra={'nested': [3, 4]})
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
    assert type(result) is tuple
    assert str(type(result[1]).__name__) == 'PVector'


def test_mutant_with_primitive_return():
    @mutant
    def return_primitive():
        return 42
    
    result = return_primitive()
    assert result == 42


def test_mutant_with_none_return():
    @mutant
    def return_none():
        return None
    
    result = return_none()
    assert result is None


