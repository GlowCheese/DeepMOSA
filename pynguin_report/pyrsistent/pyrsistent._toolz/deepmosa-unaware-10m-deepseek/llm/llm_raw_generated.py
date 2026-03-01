####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], data) == 42
    
    # Test list access
    data = {'items': [{'name': 'apple'}, {'name': 'orange'}]}
    assert get_in(['items', 0, 'name'], data) == 'apple'
    
    # Test default value when key doesn't exist
    data = {'a': 1}
    assert get_in(['b'], data) is None
    assert get_in(['b'], data, default='missing') == 'missing'
    
    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested mixed structures
    data = {'users': [{'name': 'Alice', 'scores': [85, 90]}]}
    assert get_in(['users', 0, 'scores', 1], data) == 90
    
    # Test empty key list returns the collection itself
    data = {'a': 1}
    assert get_in([], data) == data
    
    # Test with empty collection
    assert get_in(['a'], {}) is None
    assert get_in(['a'], {}, default=0) == 0
    
    # Test with None values in structure
    data = {'a': None}
    assert get_in(['a'], data) is None
    assert get_in(['a', 'b'], data) is None
    
    # Test with integer keys in dictionary
    data = {1: {2: {3: 'deep'}}}
    assert get_in([1, 2, 3], data) == 'deep'


# LLM-generated content at query #2
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    assert get_in(["a", "b"], data) == {"c": 42}
    assert get_in(["a"], data) == {"b": {"c": 42}}

    # Test list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6

    # Test mixed nested structures
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2

    # Test default value when key not found
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="missing") == "missing"
    assert get_in(["b"], data, default=0) == 0

    # Test no_default=True raises appropriate errors
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection itself
    data = {"a": 1}
    assert get_in([], data) == data
    assert get_in([], data, default="default") == data

    # Test nested default behavior
    data = {"a": {"b": {}}}
    assert get_in(["a", "b", "c"], data) is None
    assert get_in(["a", "b", "c"], data, default=0) == 0

    # Test with None values in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data) is None

    # Test with empty collections
    assert get_in(["key"], {}) is None
    assert get_in([0], []) is None

    # Test complex nested structure
    data = {
        "users": [
            {"name": "Alice", "scores": [85, 92, 78]},
            {"name": "Bob", "scores": [88, 95, 82]},
        ]
    }
    assert get_in(["users", 0, "name"], data) == "Alice"
    assert get_in(["users", 1, "scores", 2], data) == 82
    assert get_in(["users", 2, "name"], data) is None
    assert get_in(["users", 0, "scores", 5], data) is None


# LLM-generated content at query #3
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    assert get_in(['a'], {'a': 1}) == 1
    assert get_in(['a', 'b'], {'a': {'b': 2}}) == 2
    assert get_in(['a', 'b', 'c'], {'a': {'b': {'c': 3}}}) == 3
    
    # Test nested list access
    assert get_in([0], [10]) == 10
    assert get_in([0, 1], [[1, 2]]) == 2
    assert get_in([0, 1, 2], [[[1, 2, 3]]]) == 3
    
    # Test mixed dictionary and list access
    assert get_in(['a', 0], {'a': [1, 2, 3]}) == 1
    assert get_in([0, 'b'], [{'b': 5}]) == 5
    
    # Test default value when key doesn't exist
    assert get_in(['x'], {'a': 1}) is None
    assert get_in(['x'], {'a': 1}, default='missing') == 'missing'
    assert get_in(['a', 'x'], {'a': {'b': 2}}, default=0) == 0
    assert get_in([5], [1, 2, 3], default='out of range') == 'out of range'
    
    # Test no_default=True raises appropriate exceptions
    try:
        get_in(['x'], {'a': 1}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in([5], [1, 2, 3], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    try:
        get_in(['a', 'b'], {'a': 1}, no_default=True)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test empty keys list returns the collection itself
    assert get_in([], {'a': 1}) == {'a': 1}
    assert get_in([], [1, 2, 3]) == [1, 2, 3]
    assert get_in([], 'hello') == 'hello'
    
    # Test with default for empty keys
    assert get_in([], {'a': 1}, default='default') == {'a': 1}
    
    # Test complex nested structure
    data = {
        'users': [
            {'name': 'Alice', 'scores': [85, 92, 78]},
            {'name': 'Bob', 'scores': [76, 88, 95]}
        ]
    }
    assert get_in(['users', 0, 'name'], data) == 'Alice'
    assert get_in(['users', 1, 'scores', 2], data) == 95
    assert get_in(['users', 0, 'scores', 5], data, default=0) == 0
    
    # Test with no_default on complex structure
    try:
        get_in(['users', 5, 'name'], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    assert get_in(['a'], {'a': 1}) == 1
    assert get_in(['a', 'b'], {'a': {'b': 2}}) == 2
    assert get_in(['a', 'b', 'c'], {'a': {'b': {'c': 3}}}) == 3
    
    # Test nested list access
    assert get_in([0], [1, 2, 3]) == 1
    assert get_in([1, 0], [[1, 2], [3, 4]]) == 3
    assert get_in([0, 1], [[1, 2], [3, 4]]) == 2
    
    # Test mixed dictionary and list access
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], data) == 1
    assert get_in(['a', 1, 'c'], data) == 2
    
    # Test default value when key doesn't exist
    assert get_in(['x'], {'a': 1}) is None
    assert get_in(['x'], {'a': 1}, default=0) == 0
    assert get_in(['a', 'x'], {'a': 1}, default='missing') == 'missing'
    
    # Test no_default=True raises appropriate exceptions
    try:
        get_in(['x'], {'a': 1}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in([5], [1, 2, 3], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    try:
        get_in(['a', 'b'], {'a': 1}, no_default=True)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test empty keys list returns the collection itself
    assert get_in([], {'a': 1}) == {'a': 1}
    assert get_in([], [1, 2, 3]) == [1, 2, 3]
    
    # Test with None as collection
    assert get_in(['a'], None, default='not found') == 'not found'
    
    # Test with custom default value
    assert get_in(['missing'], {}, default='custom') == 'custom'
    assert get_in([0, 'key'], [[1]], default='custom') == 'custom'
    
    # Test example from docstring
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #5
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}
    
    # Test list access
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], data) == 1
    assert get_in(['a', 1, 'c'], data) == 2
    
    # Test mixed nested structures
    data = {'a': {'b': [1, 2, {'c': 3}]}}
    assert get_in(['a', 'b', 0], data) == 1
    assert get_in(['a', 'b', 2, 'c'], data) == 3
    
    # Test default value when key doesn't exist
    data = {'a': 1}
    assert get_in(['b'], data) is None
    assert get_in(['b'], data, default='missing') == 'missing'
    assert get_in(['b'], data, default=0) == 0
    
    # Test no_default=True raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default=True raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test empty keys list returns the collection itself
    data = {'a': 1}
    assert get_in([], data) == data
    
    # Test with None values in the structure
    data = {'a': None}
    assert get_in(['a'], data) is None
    assert get_in(['a', 'b'], data) is None
    
    # Test with empty collections
    data = {}
    assert get_in(['a'], data) is None
    assert get_in(['a', 'b'], data, default='missing') == 'missing'
    
    # Test with list index out of bounds
    data = [1, 2, 3]
    assert get_in([5], data) is None
    assert get_in([5], data, default='out of range') == 'out of range'
    
    # Test with TypeError (trying to index into non-indexable)
    data = {'a': 123}
    assert get_in(['a', 'b'], data) is None
    
    # Test nested default behavior
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c', 'd'], data) is None
    assert get_in(['a', 'c', 'd'], data, default='nested missing') == 'nested missing'
    
    # Test with actual pyrsistent example from docstring
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #6
#--------------------------

```python
def test_get_in():
    # Test with nested dictionaries
    nested_dict = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], nested_dict) == 42
    assert get_in(['a', 'b'], nested_dict) == {'c': 42}
    assert get_in(['a'], nested_dict) == {'b': {'c': 42}}
    
    # Test with default value when key doesn't exist
    assert get_in(['a', 'b', 'd'], nested_dict) is None
    assert get_in(['a', 'x'], nested_dict, default='not found') == 'not found'
    assert get_in(['x'], nested_dict, default=0) == 0
    
    # Test with no_default=True
    try:
        get_in(['a', 'b', 'd'], nested_dict, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test with lists
    nested_list = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], nested_list) == 2
    assert get_in([1, 2], nested_list) == 6
    
    # Test with mixed structures
    mixed = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], mixed) == 1
    assert get_in(['a', 1, 'c'], mixed) == 2
    
    # Test with empty keys list
    assert get_in([], {'a': 1}) == {'a': 1}
    assert get_in([], [1, 2, 3]) == [1, 2, 3]
    
    # Test with index out of bounds in list
    assert get_in([0, 10], [[1, 2]]) is None
    try:
        get_in([0, 10], [[1, 2]], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test with wrong type access
    assert get_in(['a', 'b'], {'a': 123}) is None
    
    # Test with default parameter
    assert get_in(['x'], {}, default='default') == 'default'
    assert get_in([0], [], default='empty') == 'empty'
    
    # Test from docstring examples
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    coll = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], coll) == 42
    
    # Test list access
    coll = {'a': [1, 2, {'b': 3}]}
    assert get_in(['a', 2, 'b'], coll) == 3
    
    # Test default value when key doesn't exist
    coll = {'a': 1}
    assert get_in(['b'], coll) is None
    assert get_in(['b'], coll, default='not found') == 'not found'
    
    # Test no_default raises KeyError
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    coll = [1, 2, 3]
    try:
        get_in([5], coll, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested default
    coll = {'a': {'b': 1}}
    assert get_in(['a', 'c', 'd'], coll) is None
    assert get_in(['a', 'c', 'd'], coll, default=0) == 0
    
    # Test empty keys list returns the collection
    coll = {'a': 1}
    assert get_in([], coll) == coll
    
    # Test with mixed types
    coll = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], coll) == 1
    assert get_in(['a', 1, 'c'], coll) == 2
    
    # Test with None in collection
    coll = {'a': None}
    assert get_in(['a'], coll) is None
    assert get_in(['a', 'b'], coll) is None
    
    # Test with empty collection
    assert get_in(['a'], {}) is None
    assert get_in([0], []) is None


# LLM-generated content at query #8
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    coll = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], coll) == 1
    assert get_in(['a', 'b'], coll) == {'c': 1}
    assert get_in(['a'], coll) == {'b': {'c': 1}}
    
    # Test list access
    coll = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], coll) == 1
    assert get_in(['a', 1, 'c'], coll) == 2
    
    # Test mixed nested structures
    coll = {'a': [{'b': {'c': [1, 2, 3]}}]}
    assert get_in(['a', 0, 'b', 'c', 1], coll) == 2
    
    # Test default value when key doesn't exist
    coll = {'a': 1}
    assert get_in(['b'], coll) is None
    assert get_in(['b'], coll, default='missing') == 'missing'
    assert get_in(['b'], coll, default=0) == 0
    
    # Test no_default=True raises KeyError
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default=True raises IndexError for lists
    coll = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], coll, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test empty keys list returns the collection itself
    coll = {'a': 1}
    assert get_in([], coll) == coll
    
    # Test nested default behavior
    coll = {'a': {'b': 1}}
    assert get_in(['a', 'c'], coll) is None
    assert get_in(['a', 'c'], coll, default='missing') == 'missing'
    
    # Test with empty collection
    assert get_in(['a'], {}) is None
    assert get_in(['a'], {}, default=0) == 0
    
    # Test with None values in collection
    coll = {'a': None}
    assert get_in(['a'], coll) is None
    assert get_in(['a', 'b'], coll) is None
    
    # Test with integer keys in dictionary
    coll = {1: {2: {3: 'value'}}}
    assert get_in([1, 2, 3], coll) == 'value'


# LLM-generated content at query #9
#--------------------------

```python
def test_get_in():
    # Test with nested dictionaries
    nested_dict = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], nested_dict) == 42
    assert get_in(['a', 'b'], nested_dict) == {'c': 42}
    assert get_in(['a'], nested_dict) == {'b': {'c': 42}}
    
    # Test with lists and dictionaries mixed
    mixed = {'items': ['Apple', 'Orange'], 'counts': [5, 3]}
    assert get_in(['items', 0], mixed) == 'Apple'
    assert get_in(['items', 1], mixed) == 'Orange'
    assert get_in(['counts', 0], mixed) == 5
    
    # Test default value when key doesn't exist
    assert get_in(['x', 'y'], {'a': 1}) is None
    assert get_in(['x', 'y'], {'a': 1}, default='not found') == 'not found'
    assert get_in(['a', 'b', 'c'], {'a': {'b': {}}}) is None
    
    # Test no_default=True raises appropriate exceptions
    try:
        get_in(['x'], {}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in(['items', 10], {'items': []}, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test with empty keys list
    coll = {'a': 1, 'b': 2}
    assert get_in([], coll) == coll
    
    # Test with None as valid value
    coll_with_none = {'a': None, 'b': {'c': None}}
    assert get_in(['a'], coll_with_none) is None
    assert get_in(['b', 'c'], coll_with_none) is None
    
    # Test with integer keys in dictionary
    dict_with_int_keys = {1: {2: {3: 'deep'}}}
    assert get_in([1, 2, 3], dict_with_int_keys) == 'deep'
    
    # Test with default value and existing path
    assert get_in(['a', 'b'], {'a': {'b': 'value'}}, default='default') == 'value'
    
    # Test with TypeError (e.g., trying to index into a non-indexable object)
    assert get_in(['a', 'b'], {'a': 42}) is None


# LLM-generated content at query #10
#--------------------------

```python
def test_get_in():
    # Test with nested dictionaries
    nested_dict = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], nested_dict) == 42
    assert get_in(['a', 'b'], nested_dict) == {'c': 42}
    assert get_in(['a'], nested_dict) == {'b': {'c': 42}}
    
    # Test with default value when key doesn't exist
    assert get_in(['a', 'b', 'd'], nested_dict) is None
    assert get_in(['a', 'b', 'd'], nested_dict, default='not found') == 'not found'
    assert get_in(['x', 'y', 'z'], nested_dict, default=0) == 0
    
    # Test with no_default=True (should raise error)
    try:
        get_in(['a', 'b', 'd'], nested_dict, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in(['x'], nested_dict, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test with lists and mixed structures
    mixed = {'items': ['Apple', 'Orange'], 'nested': {'list': [1, 2, {'deep': 'value'}]}}
    assert get_in(['items', 0], mixed) == 'Apple'
    assert get_in(['items', 1], mixed) == 'Orange'
    assert get_in(['nested', 'list', 2, 'deep'], mixed) == 'value'
    
    # Test list index out of bounds
    assert get_in(['items', 5], mixed) is None
    assert get_in(['items', 5], mixed, default='missing') == 'missing'
    
    try:
        get_in(['items', 5], mixed, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test with empty collection
    assert get_in(['any', 'key'], {}) is None
    assert get_in([0], []) is None
    
    # Test with single level collection
    simple_dict = {'key': 'value'}
    assert get_in(['key'], simple_dict) == 'value'
    assert get_in(['missing'], simple_dict) is None
    
    # Test with numeric keys in dict
    dict_with_numeric_keys = {1: {2: {3: 'deep'}}}
    assert get_in([1, 2, 3], dict_with_numeric_keys) == 'deep'
    
    # Test with empty keys list (should return the collection itself)
    assert get_in([], nested_dict) == nested_dict
    assert get_in([], [1, 2, 3]) == [1, 2, 3]
    assert get_in([], 'string') == 'string'
    
    # Test with TypeError (e.g., trying to index a string)
    assert get_in([0, 1], 'abc') is None
    assert get_in([0, 1], 'abc', default='type error') == 'type error'
    
    # Test complex nested structure
    complex_struct = {
        'users': [
            {'name': 'Alice', 'scores': [85, 92, 78]},
            {'name': 'Bob', 'scores': [88, 95, 81]}
        ]
    }
    assert get_in(['users', 0, 'name'], complex_struct) == 'Alice'
    assert get_in(['users', 1, 'scores', 2], complex_struct) == 81
    assert get_in(['users', 2, 'name'], complex_struct) is None
    assert get_in(['users', 0, 'scores', 5], complex_struct) is None


# LLM-generated content at query #11
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 1}}}
    assert get_in(["a", "b", "c"], data) == 1
    
    # Test list access
    data = {"a": [{"b": 2}, {"c": 3}]}
    assert get_in(["a", 0, "b"], data) == 2
    assert get_in(["a", 1, "c"], data) == 3
    
    # Test mixed nested structures
    data = {"a": {"b": [1, 2, {"c": 3}]}}
    assert get_in(["a", "b", 2, "c"], data) == 3
    assert get_in(["a", "b", 1], data) == 2
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="missing") == "missing"
    
    # Test no_default=True raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default=True raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test empty keys list returns the collection itself
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test nested missing key with default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c", "d"], data, default="not found") == "not found"
    
    # Test with None value in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data, default="default") == "default"
    
    # Test with empty collection
    assert get_in(["key"], {}, default="empty") == "empty"
    
    # Test with integer keys in dict
    data = {1: {2: "value"}}
    assert get_in([1, 2], data) == "value"


# LLM-generated content at query #12
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    assert get_in(['a'], {'a': 1}) == 1
    assert get_in(['a', 'b'], {'a': {'b': 2}}) == 2
    assert get_in(['a', 'b', 'c'], {'a': {'b': {'c': 3}}}) == 3
    
    # Test list access
    assert get_in([0], [10]) == 10
    assert get_in([1, 0], [[], [20]]) == 20
    assert get_in([0, 1], [[30, 40]]) == 40
    
    # Test mixed nested structures
    data = {'a': [{'b': 5}, {'c': 6}]}
    assert get_in(['a', 0, 'b'], data) == 5
    assert get_in(['a', 1, 'c'], data) == 6
    
    # Test default value when key doesn't exist
    assert get_in(['x'], {}) is None
    assert get_in(['x'], {}, default='missing') == 'missing'
    assert get_in(['a', 'x'], {'a': {}}, default=0) == 0
    
    # Test no_default raises appropriate exceptions
    try:
        get_in(['x'], {}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in([5], [], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data
    
    # Test with None values
    data = {'a': None}
    assert get_in(['a'], data) is None
    
    # Test with falsy values
    data = {'a': 0, 'b': False, 'c': ''}
    assert get_in(['a'], data) == 0
    assert get_in(['b'], data) is False
    assert get_in(['c'], data) == ''
    
    # Test nested with missing intermediate key
    assert get_in(['a', 'b', 'c'], {'a': {}}) is None
    assert get_in(['a', 'b', 'c'], {'a': {}}, default='not found') == 'not found'
    
    # Test with integer keys in dict
    data = {1: {2: {3: 'deep'}}}
    assert get_in([1, 2, 3], data) == 'deep'
    
    # Test with tuple as key in dict
    data = {('a', 'b'): 'value'}
    assert get_in([('a', 'b')], data) == 'value'


# LLM-generated content at query #13
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 1}}}
    assert get_in(["a", "b", "c"], data) == 1
    
    # Test list access
    data = {"a": [{"b": 2}, {"c": 3}]}
    assert get_in(["a", 0, "b"], data) == 2
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="missing") == "missing"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested default behavior
    data = {"a": {"b": 1}}
    assert get_in(["a", "c"], data) is None
    assert get_in(["a", "c"], data, default=0) == 0
    
    # Test with empty keys list
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test with mixed types in path
    data = {"a": [{"b": 2}, {"c": 3}]}
    assert get_in(["a", 1, "c"], data) == 3
    
    # Test with non-existent nested path
    data = {"a": {"b": 1}}
    assert get_in(["a", "b", "c"], data) is None
    
    # Test with list index out of bounds (with default)
    data = [1, 2, 3]
    assert get_in([5], data) is None
    
    # Test with invalid intermediate type
    data = {"a": 1}
    assert get_in(["a", "b"], data) is None
    
    # Test with None as collection
    assert get_in(["a"], None) is None
    assert get_in(["a"], None, default="default") == "default"


# LLM-generated content at query #14
#--------------------------

```python
def test_get_in():
    # Test with nested dictionaries
    nested_dict = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], nested_dict) == 42
    assert get_in(['a', 'b'], nested_dict) == {'c': 42}
    assert get_in(['a'], nested_dict) == {'b': {'c': 42}}
    
    # Test with mixed nested structures (dicts and lists)
    mixed_structure = {'items': [{'name': 'apple', 'price': 1.0}, {'name': 'orange', 'price': 1.5}]}
    assert get_in(['items', 0, 'name'], mixed_structure) == 'apple'
    assert get_in(['items', 1, 'price'], mixed_structure) == 1.5
    
    # Test default value when key doesn't exist
    assert get_in(['x', 'y'], {'a': 1}) is None
    assert get_in(['x', 'y'], {'a': 1}, default='not found') == 'not found'
    assert get_in(['a', 'b'], {'a': 1}, default=0) == 0
    
    # Test no_default=True raises appropriate exceptions
    try:
        get_in(['x'], {}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in(['items', 5], {'items': []}, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test with empty keys list
    coll = {'a': 1, 'b': 2}
    assert get_in([], coll) == coll
    
    # Test with None as valid value
    coll_with_none = {'a': None}
    assert get_in(['a'], coll_with_none) is None
    
    # Test default doesn't apply when key exists with None value
    assert get_in(['a'], coll_with_none, default='default') is None
    
    # Test with list indexing only
    simple_list = [10, 20, 30]
    assert get_in([1], simple_list) == 20
    assert get_in([0], simple_list) == 10
    
    # Test with nested lists
    nested_list = [[1, 2], [3, 4]]
    assert get_in([0, 1], nested_list) == 2
    assert get_in([1, 0], nested_list) == 3
    
    # Test TypeError handling (e.g., trying to index into a non-indexable object)
    assert get_in([0], 'string', default='type error') == 'type error'
    
    # Test with complex default value
    complex_default = object()
    result = get_in(['non', 'existent'], {}, default=complex_default)
    assert result is complex_default


# LLM-generated content at query #15
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    assert get_in(['a'], {'a': 1}) == 1
    assert get_in(['a', 'b'], {'a': {'b': 2}}) == 2
    assert get_in(['a', 'b', 'c'], {'a': {'b': {'c': 3}}}) == 3
    
    # Test nested list access
    assert get_in([0], [1, 2, 3]) == 1
    assert get_in([1, 0], [[1, 2], [3, 4]]) == 3
    assert get_in([0, 1], [[1, 2], [3, 4]]) == 2
    
    # Test mixed dictionary and list access
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], data) == 1
    assert get_in(['a', 1, 'c'], data) == 2
    
    # Test default value when key doesn't exist
    assert get_in(['x'], {'a': 1}) is None
    assert get_in(['a', 'x'], {'a': {'b': 1}}) is None
    assert get_in(['a', 'b', 'c'], {'a': {'b': 1}}) is None
    assert get_in([5], [1, 2, 3]) is None
    assert get_in([0, 5], [[1, 2], [3, 4]]) is None
    
    # Test custom default value
    assert get_in(['x'], {'a': 1}, default='not found') == 'not found'
    assert get_in(['a', 'x'], {'a': {'b': 1}}, default=0) == 0
    assert get_in([5], [1, 2, 3], default='missing') == 'missing'
    
    # Test no_default=True raises appropriate exceptions
    try:
        get_in(['x'], {'a': 1}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in(['a', 'x'], {'a': {'b': 1}}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in([5], [1, 2, 3], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    try:
        get_in([0, 'x'], [1, 2, 3], no_default=True)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test empty keys list returns the collection itself
    assert get_in([], {'a': 1}) == {'a': 1}
    assert get_in([], [1, 2, 3]) == [1, 2, 3]
    assert get_in([], 'hello') == 'hello'
    
    # Test with None values in the structure
    assert get_in(['a'], {'a': None}) is None
    assert get_in(['a', 'b'], {'a': {'b': None}}) is None
    
    # Test with False values in the structure
    assert get_in(['a'], {'a': False}) is False
    assert get_in(['a', 'b'], {'a': {'b': False}}) is False
    
    # Test with zero values in the structure
    assert get_in(['a'], {'a': 0}) == 0
    assert get_in(['a', 'b'], {'a': {'b': 0}}) == 0
    
    # Test complex nested structure
    complex_data = {
        'users': [
            {
                'id': 1,
                'name': 'Alice',
                'orders': [
                    {'id': 'A1', 'total': 100},
                    {'id': 'A2', 'total': 200}
                ]
            },
            {
                'id': 2,
                'name': 'Bob',
                'orders': [
                    {'id': 'B1', 'total': 150}
                ]
            }
        ]
    }
    
    assert get_in(['users', 0, 'name'], complex_data) == 'Alice'
    assert get_in(['users', 1, 'orders', 0, 'total'], complex_data) == 150
    assert get_in(['users', 0, 'orders', 1, 'id'], complex_data) == 'A2'
    assert get_in(['users', 2, 'name'], complex_data) is None
    assert get_in(['users', 0, 'orders', 2], complex_data) is None
    assert get_in(['users', 0, 'address'], complex_data) is None


# LLM-generated content at query #16
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    coll = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], coll) == 42
    
    # Test list access
    coll = [[1, 2, 3], [4, 5, 6]]
    assert get_in([1, 2], coll) == 6
    
    # Test mixed nested structures
    coll = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], coll) == 2
    
    # Test default value when key doesn't exist
    coll = {'a': 1}
    assert get_in(['b'], coll) is None
    assert get_in(['b'], coll, default='missing') == 'missing'
    
    # Test no_default raises KeyError
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    coll = [1, 2, 3]
    try:
        get_in([5], coll, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test empty keys list returns the collection
    coll = {'a': 1}
    assert get_in([], coll) == coll
    
    # Test nested default behavior
    coll = {'a': {'b': 1}}
    assert get_in(['a', 'c'], coll) is None
    assert get_in(['a', 'c'], coll, default=0) == 0
    
    # Test with None in collection
    coll = {'a': None}
    assert get_in(['a'], coll) is None
    assert get_in(['a', 'b'], coll) is None
    
    # Test with empty collection
    assert get_in(['key'], {}) is None
    assert get_in([0], []) is None
    
    # Test complex nested structure
    coll = {
        'users': [
            {'name': 'Alice', 'scores': [85, 92, 78]},
            {'name': 'Bob', 'scores': [88, 95, 81]}
        ]
    }
    assert get_in(['users', 1, 'scores', 2], coll) == 81
    assert get_in(['users', 0, 'name'], coll) == 'Alice'
    
    # Test default with no_default=False (implicit)
    coll = {'a': 1}
    assert get_in(['b'], coll, default='default') == 'default'
    
    # Test that TypeError is caught and returns default
    coll = {'a': 1}
    assert get_in(['a', 'b'], coll) is None


# LLM-generated content at query #17
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], data) == 42
    
    # Test list access
    data = {'a': [{'b': 10}, {'c': 20}]}
    assert get_in(['a', 0, 'b'], data) == 10
    assert get_in(['a', 1, 'c'], data) == 20
    
    # Test default value when key doesn't exist
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default='missing') == 'missing'
    
    # Test no_default raises KeyError
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6
    
    # Test mixed nested structures
    data = {'a': [{'b': 1}, {'c': [2, 3, 4]}]}
    assert get_in(['a', 1, 'c', 2], data) == 4
    
    # Test empty keys list returns the collection itself
    data = {'a': 1}
    assert get_in([], data) == data
    
    # Test with None as valid value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    
    # Test default with empty collection
    assert get_in(['a'], {}, default='default') == 'default'
    
    # Test TypeError handling (e.g., trying to index into an integer)
    data = {'a': 42}
    assert get_in(['a', 'b'], data) is None


# LLM-generated content at query #18
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"a": [1, 2, {"b": 3}]}
    assert get_in(["a", 2, "b"], data) == 3
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested default behavior
    data = {"a": {"b": {}}}
    assert get_in(["a", "b", "c"], data) is None
    assert get_in(["a", "b", "c"], data, default=0) == 0
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test with mixed types
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2
    
    # Test TypeError handling (e.g., trying to index into an integer)
    data = {"a": 1}
    assert get_in(["a", "b"], data) is None
    assert get_in(["a", "b"], data, default="error") == "error"


# LLM-generated content at query #19
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    assert get_in(["a", "b"], data) == {"c": 42}
    assert get_in(["a"], data) == {"b": {"c": 42}}

    # Test list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6

    # Test mixed nested structures
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2

    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="missing") == "missing"
    assert get_in(["b"], data, default=0) == 0

    # Test no_default raises appropriate exception
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test no_default with list index error
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data

    # Test nested missing key with default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c", "d"], data) is None
    assert get_in(["a", "c", "d"], data, default="default") == "default"

    # Test with None values in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data) is None

    # Test with empty collections
    assert get_in(["key"], {}) is None
    assert get_in([0], []) is None

    # Test complex nested structure
    data = {
        "users": [
            {"name": "Alice", "scores": [85, 92, 78]},
            {"name": "Bob", "scores": [88, 95, 82]}
        ]
    }
    assert get_in(["users", 0, "name"], data) == "Alice"
    assert get_in(["users", 1, "scores", 2], data) == 82
    assert get_in(["users", 0, "scores", 5], data) is None
    assert get_in(["users", 2, "name"], data) is None


# LLM-generated content at query #20
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    assert get_in(["a", "b"], data) == {"c": 42}
    assert get_in(["a"], data) == {"b": {"c": 42}}

    # Test list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6

    # Test mixed nested structures
    data = {"a": [{"b": 10}, {"c": 20}]}
    assert get_in(["a", 0, "b"], data) == 10
    assert get_in(["a", 1, "c"], data) == 20

    # Test default value when key not found
    data = {"x": 1}
    assert get_in(["y"], data) is None
    assert get_in(["y"], data, default="missing") == "missing"
    assert get_in(["x", "y"], data, default=0) == 0

    # Test no_default=True raises appropriate exceptions
    data = {"x": 1}
    try:
        get_in(["y"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection itself
    data = {"a": 1}
    assert get_in([], data) == data

    # Test with None values in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data, default="default") == "default"

    # Test with integer keys in dictionary
    data = {1: {2: {3: "deep"}}}
    assert get_in([1, 2, 3], data) == "deep"

    # Test with custom default value
    data = {}
    assert get_in(["deep", "nested", "path"], data, default=[]) == []
    assert get_in(["deep", "nested", "path"], data, default={}) == {}

    # Test with TypeError scenario
    data = "not a collection"
    assert get_in([0], data, default="type error") == "type error"


# LLM-generated content at query #21
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    coll = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], coll) == 42
    assert get_in(['a', 'b'], coll) == {'c': 42}
    assert get_in(['a'], coll) == {'b': {'c': 42}}

    # Test list access
    coll = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], coll) == 2
    assert get_in([1, 2], coll) == 6

    # Test mixed nested structures
    coll = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], coll) == 1
    assert get_in(['a', 1, 'c'], coll) == 2

    # Test default value when key not found
    coll = {'a': 1}
    assert get_in(['b'], coll) is None
    assert get_in(['b'], coll, default='missing') == 'missing'
    assert get_in(['b'], coll, default=0) == 0

    # Test no_default=True raises KeyError
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test no_default=True raises IndexError for lists
    coll = [1, 2, 3]
    try:
        get_in([5], coll, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection
    coll = {'a': 1}
    assert get_in([], coll) == coll

    # Test nested default behavior
    coll = {'a': {'b': {}}}
    assert get_in(['a', 'b', 'c'], coll) is None
    assert get_in(['a', 'b', 'c'], coll, default='default') == 'default'

    # Test with None in collection
    coll = {'a': None}
    assert get_in(['a', 'b'], coll) is None

    # Test with empty collection
    assert get_in(['key'], {}) is None
    assert get_in([0], []) is None

    # Test complex nested structure
    coll = {
        'users': [
            {'name': 'Alice', 'scores': [85, 92, 78]},
            {'name': 'Bob', 'scores': [76, 88, 95]}
        ]
    }
    assert get_in(['users', 0, 'name'], coll) == 'Alice'
    assert get_in(['users', 1, 'scores', 2], coll) == 95
    assert get_in(['users', 0, 'scores', 1], coll) == 92
    assert get_in(['users', 2, 'name'], coll) is None
    assert get_in(['users', 0, 'age'], coll) is None


# LLM-generated content at query #22
#--------------------------

```python
def test_get_in():
    # Test with nested dictionaries
    nested_dict = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], nested_dict) == 42
    assert get_in(['a', 'b'], nested_dict) == {'c': 42}
    assert get_in(['a'], nested_dict) == {'b': {'c': 42}}
    
    # Test with default value when key doesn't exist
    assert get_in(['a', 'b', 'd'], nested_dict) is None
    assert get_in(['a', 'b', 'd'], nested_dict, default='not found') == 'not found'
    assert get_in(['x', 'y', 'z'], nested_dict, default=0) == 0
    
    # Test with no_default=True
    try:
        get_in(['a', 'b', 'd'], nested_dict, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test with mixed nested structures (dicts and lists)
    mixed = {'items': [{'name': 'apple', 'price': 1.0}, {'name': 'orange', 'price': 1.5}]}
    assert get_in(['items', 0, 'name'], mixed) == 'apple'
    assert get_in(['items', 1, 'price'], mixed) == 1.5
    
    # Test with list indices out of range
    assert get_in(['items', 5, 'name'], mixed) is None
    try:
        get_in(['items', 5, 'name'], mixed, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test with empty keys list
    assert get_in([], nested_dict) == nested_dict
    
    # Test with non-existent top-level key
    assert get_in(['x'], nested_dict) is None
    try:
        get_in(['x'], nested_dict, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test with TypeError (e.g., trying to index into a non-indexable value)
    assert get_in(['a', 'b', 'c', 'd'], nested_dict) is None
    
    # Test with example from docstring
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #23
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 1}}}
    assert get_in(["a", "b", "c"], data) == 1
    assert get_in(["a", "b"], data) == {"c": 1}
    assert get_in(["a"], data) == {"b": {"c": 1}}

    # Test list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6

    # Test mixed nested structures
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2

    # Test default value when key not found
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    assert get_in(["b"], data, default=0) == 0

    # Test no_default raises appropriate exception
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list
    data = {"a": 1}
    assert get_in([], data) == data

    # Test nested default behavior
    data = {"a": {"b": {}}}
    assert get_in(["a", "b", "c"], data) is None
    assert get_in(["a", "b", "c"], data, default="missing") == "missing"

    # Test with None values in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data) is None

    # Test with empty collections
    assert get_in(["key"], {}) is None
    assert get_in([0], []) is None

    # Test complex nested structure
    data = {
        "users": [
            {"name": "Alice", "scores": [85, 92, 78]},
            {"name": "Bob", "scores": [88, 95, 81]},
        ]
    }
    assert get_in(["users", 0, "name"], data) == "Alice"
    assert get_in(["users", 1, "scores", 2], data) == 81
    assert get_in(["users", 0, "scores", 1], data) == 92


# LLM-generated content at query #24
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    assert get_in(["a", "b"], data) == {"c": 42}
    assert get_in(["a"], data) == {"b": {"c": 42}}

    # Test list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6

    # Test mixed nested structures
    data = {"a": [{"b": 10}, {"c": 20}]}
    assert get_in(["a", 0, "b"], data) == 10
    assert get_in(["a", 1, "c"], data) == 20

    # Test default value when key doesn't exist
    data = {"x": 1}
    assert get_in(["y"], data) is None
    assert get_in(["y"], data, default="missing") == "missing"
    assert get_in(["x", "y"], data, default=0) == 0

    # Test no_default=True raises appropriate errors
    data = {"x": 1}
    try:
        get_in(["y"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection itself
    data = {"a": 1}
    assert get_in([], data) == data

    # Test with None as valid value
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["b"], data, default="default") == "default"

    # Test deeply nested structure
    data = {"level1": {"level2": {"level3": {"level4": "deep_value"}}}}
    assert get_in(["level1", "level2", "level3", "level4"], data) == "deep_value"
    assert get_in(["level1", "level2", "level3", "missing"], data, default=None) is None

    # Test with integer keys in dictionary
    data = {1: {2: {3: "number_keys"}}}
    assert get_in([1, 2, 3], data) == "number_keys"

    # Test TypeError handling (e.g., trying to index into a non-indexable type)
    data = {"a": 123}
    assert get_in(["a", "b"], data, default="type_error") == "type_error"


# LLM-generated content at query #25
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 1}}}
    assert get_in(["a", "b", "c"], data) == 1
    assert get_in(["a", "b"], data) == {"c": 1}
    assert get_in(["a"], data) == {"b": {"c": 1}}

    # Test list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6

    # Test mixed nested structures
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2

    # Test default value when key not found
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="not found") == "not found"
    assert get_in(["b"], data, default=0) == 0

    # Test no_default raises appropriate exceptions
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection itself
    data = {"a": 1}
    assert get_in([], data) == data
    assert get_in([], data, default="default") == data

    # Test nested default behavior
    data = {"a": {"b": {}}}
    assert get_in(["a", "b", "c"], data) is None
    assert get_in(["a", "b", "c"], data, default=0) == 0

    # Test with None values in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data) is None

    # Test with empty collections
    assert get_in(["key"], {}) is None
    assert get_in([0], []) is None

    # Test complex nested structure
    data = {
        "users": [
            {"name": "Alice", "scores": [85, 92, 78]},
            {"name": "Bob", "scores": [88, 79, 91]},
        ]
    }
    assert get_in(["users", 0, "name"], data) == "Alice"
    assert get_in(["users", 1, "scores", 2], data) == 91
    assert get_in(["users", 0, "scores", 1], data) == 92
    assert get_in(["users", 2, "name"], data) is None


# LLM-generated content at query #26
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    assert get_in(["a", "b"], data) == {"c": 42}
    assert get_in(["a"], data) == {"b": {"c": 42}}

    # Test list access
    data = {"items": [{"id": 1}, {"id": 2}]}
    assert get_in(["items", 0, "id"], data) == 1
    assert get_in(["items", 1, "id"], data) == 2

    # Test mixed nested structures
    data = {"a": [{"b": {"c": 10}}, {"b": {"c": 20}}]}
    assert get_in(["a", 0, "b", "c"], data) == 10
    assert get_in(["a", 1, "b", "c"], data) == 20

    # Test default value when key doesn't exist
    data = {"x": 1}
    assert get_in(["y"], data) is None
    assert get_in(["y"], data, default="missing") == "missing"
    assert get_in(["x", "y"], data, default=0) == 0

    # Test no_default=True raises appropriate exceptions
    data = {"x": 1}
    try:
        get_in(["y"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    data = {"items": []}
    try:
        get_in(["items", 0], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test with empty collection
    assert get_in(["key"], {}) is None
    assert get_in([0], []) is None

    # Test with nested empty structures
    data = {"a": {}}
    assert get_in(["a", "b"], data) is None
    assert get_in(["a", "b"], data, default=[]) == []

    # Test with integer keys in dict
    data = {1: {2: {3: "deep"}}}
    assert get_in([1, 2, 3], data) == "deep"

    # Test with None values
    data = {"a": None}
    try:
        get_in(["a", "b"], data, no_default=True)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    assert get_in(["a", "b"], data) is None

    # Test with default for nested missing key
    data = {"level1": {"level2": {"level3": "value"}}}
    assert get_in(["level1", "level2", "missing"], data, default="default") == "default"

    # Test with single key
    data = {"simple": "value"}
    assert get_in(["simple"], data) == "value"
    assert get_in(["simple"], data, no_default=True) == "value"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    assert get_in(['a'], {'a': 1}) == 1
    assert get_in(['a', 'b'], {'a': {'b': 2}}) == 2
    assert get_in(['a', 'b', 'c'], {'a': {'b': {'c': 3}}}) == 3
    
    # Test nested list access
    assert get_in([0], [10]) == 10
    assert get_in([0, 1], [[10, 20]]) == 20
    assert get_in([0, 1, 2], [[[10, 20, 30]]]) == 30
    
    # Test mixed dictionary and list access
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], data) == 1
    assert get_in(['a', 1, 'c'], data) == 2
    
    # Test default value when key doesn't exist
    assert get_in(['x'], {'a': 1}) is None
    assert get_in(['x'], {'a': 1}, default='missing') == 'missing'
    assert get_in(['a', 'x'], {'a': 1}, default=0) == 0
    
    # Test no_default=True raises appropriate errors
    try:
        get_in(['x'], {'a': 1}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in([5], [1, 2, 3], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test empty keys list returns the collection itself
    assert get_in([], {'a': 1}) == {'a': 1}
    assert get_in([], [1, 2, 3]) == [1, 2, 3]
    
    # Test with None as collection
    assert get_in(['a'], None) is None
    assert get_in(['a'], None, default='default') == 'default'
    
    # Test with empty collection
    assert get_in(['a'], {}) is None
    assert get_in([0], []) is None
    
    # Test complex nested structure
    complex_data = {
        'users': [
            {
                'name': 'Alice',
                'orders': [
                    {'id': 1, 'items': ['apple', 'banana']},
                    {'id': 2, 'items': ['orange']}
                ]
            }
        ]
    }
    assert get_in(['users', 0, 'orders', 1, 'items', 0], complex_data) == 'orange'
    assert get_in(['users', 0, 'orders', 2], complex_data) is None
    assert get_in(['users', 0, 'orders', 2], complex_data, default=[]) == []


# LLM-generated content at query #2
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 1}}}
    assert get_in(["a", "b", "c"], data) == 1
    assert get_in(["a", "b"], data) == {"c": 1}
    assert get_in(["a"], data) == {"b": {"c": 1}}

    # Test list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6

    # Test mixed nested structures
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2

    # Test default value when key not found
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="not found") == "not found"
    assert get_in(["b"], data, default=0) == 0

    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test no_default raises IndexError
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data

    # Test nested default behavior
    data = {"a": {"b": {}}}
    assert get_in(["a", "b", "c"], data) is None
    assert get_in(["a", "b", "c"], data, default=0) == 0

    # Test with None values in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data) is None

    # Test with empty collections
    assert get_in(["a"], {}) is None
    assert get_in([0], []) is None

    # Test complex nested structure
    data = {
        "users": [
            {"name": "Alice", "scores": [85, 92, 78]},
            {"name": "Bob", "scores": [88, 95, 82]}
        ]
    }
    assert get_in(["users", 0, "name"], data) == "Alice"
    assert get_in(["users", 1, "scores", 2], data) == 82
    assert get_in(["users", 0, "scores", 5], data) is None
    assert get_in(["users", 2, "name"], data) is None


# LLM-generated content at query #3
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 1}}}
    assert get_in(["a", "b", "c"], data) == 1
    
    # Test list access
    data = {"a": [{"b": 2}, {"c": 3}]}
    assert get_in(["a", 0, "b"], data) == 2
    assert get_in(["a", 1, "c"], data) == 3
    
    # Test mixed nested structures
    data = {"a": {"b": [1, 2, {"c": 3}]}}
    assert get_in(["a", "b", 2, "c"], data) == 3
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="missing") == "missing"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test nested default behavior
    data = {"a": {"b": 1}}
    assert get_in(["a", "c"], data) is None
    assert get_in(["a", "c"], data, default=0) == 0
    
    # Test with None in the path
    data = {"a": None}
    assert get_in(["a", "b"], data) is None
    
    # Test with empty collections
    assert get_in(["key"], {}) is None
    assert get_in([0], []) is None
    
    # Test with integer keys in dict
    data = {1: {2: "value"}}
    assert get_in([1, 2], data) == "value"
    
    # Test with string index in list (should return default)
    data = [1, 2, 3]
    assert get_in(["0"], data) is None


# LLM-generated content at query #4
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"a": [1, 2, {"b": 3}]}
    assert get_in(["a", 2, "b"], data) == 3
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested missing key with default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c", "d"], data, default="missing") == "missing"
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test mixed nested structures
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2
    
    # Test with None as valid value
    data = {"a": None}
    assert get_in(["a"], data) is None
    
    # Test accessing None as if it were a collection
    data = {"a": None}
    assert get_in(["a", "b"], data) is None
    
    # Test with empty collections
    data = {}
    assert get_in(["a"], data) is None
    assert get_in(["a"], data, default=[]) == []


# LLM-generated content at query #5
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    assert get_in(['a'], {'a': 1}) == 1
    assert get_in(['a', 'b'], {'a': {'b': 2}}) == 2
    assert get_in(['a', 'b', 'c'], {'a': {'b': {'c': 3}}}) == 3
    
    # Test nested list access
    assert get_in([0], [10]) == 10
    assert get_in([0, 1], [[10, 20]]) == 20
    assert get_in([0, 1, 2], [[[10, 20, 30]]]) == 30
    
    # Test mixed dictionary and list access
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], data) == 1
    assert get_in(['a', 1, 'c'], data) == 2
    
    # Test default value when key not found
    assert get_in(['x'], {}) is None
    assert get_in(['x'], {}, default=0) == 0
    assert get_in(['a', 'x'], {'a': {}}, default='missing') == 'missing'
    
    # Test no_default raises appropriate exceptions
    try:
        get_in(['x'], {}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in([0], [], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test with pyrsistent-like structure from docstring example
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0
    
    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data
    
    # Test with None values in structure
    data = {'a': None}
    assert get_in(['a'], data) is None
    
    # Test with integer keys in dictionary
    data = {1: 'one', 2: 'two'}
    assert get_in([1], data) == 'one'
    
    # Test with tuple keys
    data = {('a', 'b'): 'tuple_key'}
    assert get_in([('a', 'b')], data) == 'tuple_key'


# LLM-generated content at query #6
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    assert get_in(['a'], {'a': 1}) == 1
    assert get_in(['a', 'b'], {'a': {'b': 2}}) == 2
    assert get_in(['a', 'b', 'c'], {'a': {'b': {'c': 3}}}) == 3
    
    # Test nested list access
    assert get_in([0], [10]) == 10
    assert get_in([0, 1], [[1, 2]]) == 2
    assert get_in([0, 1, 2], [[[1, 2, 3]]]) == 3
    
    # Test mixed dictionary and list access
    assert get_in(['a', 0], {'a': [1, 2, 3]}) == 1
    assert get_in([0, 'b'], [{'b': 5}]) == 5
    
    # Test default value when key not found
    assert get_in(['x'], {'a': 1}) is None
    assert get_in(['x'], {'a': 1}, default='not found') == 'not found'
    assert get_in(['a', 'x'], {'a': {'b': 1}}, default=0) == 0
    
    # Test no_default parameter raises appropriate exceptions
    try:
        get_in(['x'], {'a': 1}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in([5], [1, 2, 3], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    try:
        get_in(['a', 'b'], {'a': 1}, no_default=True)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test empty keys list returns the collection itself
    assert get_in([], {'a': 1}) == {'a': 1}
    assert get_in([], [1, 2, 3]) == [1, 2, 3]
    
    # Test with None as collection
    assert get_in(['a'], None, default='default') == 'default'
    
    # Test with integer keys on dict (should use default)
    assert get_in([0], {'0': 'value'}, default='not found') == 'not found'
    
    # Test with string index on list (should use default)
    assert get_in(['0'], [1, 2, 3], default='not found') == 'not found'


# LLM-generated content at query #7
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"a": [1, 2, {"b": 3}]}
    assert get_in(["a", 2, "b"], data) == 3
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c"], data, default="missing") == "missing"
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test mixed nested structures
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2
    
    # Test with None in the path
    data = {"a": None}
    assert get_in(["a", "b"], data) is None
    
    # Test with empty collection
    assert get_in(["a"], {}, default="default") == "default"
    
    # Test with integer keys in dict
    data = {1: {2: "value"}}
    assert get_in([1, 2], data) == "value"


# LLM-generated content at query #8
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    coll = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], coll) == 42
    
    # Test with list in nested structure
    coll = {'items': [{'name': 'apple'}, {'name': 'orange'}]}
    assert get_in(['items', 0, 'name'], coll) == 'apple'
    
    # Test default value when key doesn't exist
    coll = {'a': {'b': 1}}
    assert get_in(['a', 'c'], coll) is None
    assert get_in(['a', 'c'], coll, default='not found') == 'not found'
    
    # Test no_default=True raises KeyError
    coll = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], coll, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test with list index out of bounds
    coll = {'items': ['a', 'b', 'c']}
    assert get_in(['items', 5], coll) is None
    
    # Test no_default=True raises IndexError for list
    coll = {'items': ['a', 'b', 'c']}
    try:
        get_in(['items', 5], coll, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test empty keys list returns the collection itself
    coll = {'a': 1}
    assert get_in([], coll) == coll
    
    # Test with mixed types
    coll = [{'a': [1, 2, {'b': 'value'}]}]
    assert get_in([0, 'a', 2, 'b'], coll) == 'value'
    
    # Test with default value for non-existent path
    coll = {'x': {'y': 10}}
    assert get_in(['x', 'z'], coll, default=99) == 99
    assert get_in(['a', 'b'], coll, default=99) == 99
    
    # Test with TypeError (e.g., trying to index into an integer)
    coll = {'a': 42}
    assert get_in(['a', 'b'], coll) is None
    
    # Test no_default=True raises TypeError
    coll = {'a': 42}
    try:
        get_in(['a', 'b'], coll, no_default=True)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"items": [{"id": 1}, {"id": 2}]}
    assert get_in(["items", 0, "id"], data) == 1
    assert get_in(["items", 1, "id"], data) == 2
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="not found") == "not found"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c"], data) is None
    assert get_in(["a", "c"], data, default=0) == 0
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test mixed dictionary and list access
    data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
    assert get_in(["users", 0, "name"], data) == "Alice"
    assert get_in(["users", 1, "age"], data) == 25
    
    # Test with None as intermediate value
    data = {"a": None}
    assert get_in(["a", "b"], data) is None
    
    # Test with empty collection
    assert get_in(["key"], {}) is None
    assert get_in([0], []) is None
    
    # Test with custom default value
    data = {"a": 1}
    assert get_in(["b"], data, default="DEFAULT") == "DEFAULT"
    
    # Test with integer keys in dictionary
    data = {1: {2: {3: "deep"}}}
    assert get_in([1, 2, 3], data) == "deep"


# LLM-generated content at query #10
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 1}}}
    assert get_in(["a", "b", "c"], data) == 1
    assert get_in(["a", "b"], data) == {"c": 1}
    assert get_in(["a"], data) == {"b": {"c": 1}}

    # Test list access
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2

    # Test mixed nested structures
    data = {"a": [{"b": {"c": [1, 2, 3]}}]}
    assert get_in(["a", 0, "b", "c", 2], data) == 3

    # Test default value when key not found
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="not found") == "not found"
    assert get_in(["b"], data, default=0) == 0

    # Test no_default=True raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test no_default=True raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection itself
    data = {"a": 1}
    assert get_in([], data) == data
    assert get_in([], data, default="default") == data

    # Test with None values in the structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data) is None

    # Test with empty collections
    data = {}
    assert get_in(["a"], data) is None
    assert get_in(["a"], data, default={}) == {}

    # Test with integer keys in dictionary
    data = {1: {2: {3: "value"}}}
    assert get_in([1, 2, 3], data) == "value"

    # Test with nested lists
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 1, 0], data) == 3
    assert get_in([1, 0, 1], data) == 6

    # Test default value with nested missing key
    data = {"a": {"b": 1}}
    assert get_in(["a", "c", "d"], data, default="missing") == "missing"

    # Test no_default with nested missing key
    data = {"a": {"b": 1}}
    try:
        get_in(["a", "c", "d"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    assert get_in(['a'], {'a': 1}) == 1
    assert get_in(['a', 'b'], {'a': {'b': 2}}) == 2
    assert get_in(['a', 'b', 'c'], {'a': {'b': {'c': 3}}}) == 3
    
    # Test nested list access
    assert get_in([0], [1, 2, 3]) == 1
    assert get_in([1, 0], [[1, 2], [3, 4]]) == 3
    assert get_in([0, 1], [[1, 2], [3, 4]]) == 2
    
    # Test mixed dictionary and list access
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], data) == 1
    assert get_in(['a', 1, 'c'], data) == 2
    
    # Test default value when key doesn't exist
    assert get_in(['x'], {'a': 1}) is None
    assert get_in(['a', 'x'], {'a': {'b': 1}}) is None
    assert get_in([5], [1, 2, 3]) is None
    assert get_in(['a', 'b', 'c'], {'a': {'b': {}}}) is None
    
    # Test custom default value
    assert get_in(['x'], {'a': 1}, default='not found') == 'not found'
    assert get_in(['a', 'x'], {'a': {'b': 1}}, default=0) == 0
    assert get_in([5], [1, 2, 3], default='missing') == 'missing'
    
    # Test no_default=True raises appropriate exceptions
    try:
        get_in(['x'], {'a': 1}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in(['a', 'x'], {'a': {'b': 1}}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in([5], [1, 2, 3], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test empty keys list returns the collection itself
    assert get_in([], {'a': 1}) == {'a': 1}
    assert get_in([], [1, 2, 3]) == [1, 2, 3]
    assert get_in([], 'hello') == 'hello'
    
    # Test with None collection
    assert get_in(['a'], None) is None
    assert get_in(['a'], None, default='default') == 'default'
    
    # Test example from docstring
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"a": [1, 2, {"b": 3}]}
    assert get_in(["a", 2, "b"], data) == 3
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c"], data, default="missing") == "missing"
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test mixed nested structures
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2
    
    # Test with None in the path
    data = {"a": None}
    assert get_in(["a", "b"], data) is None
    
    # Test with empty collection
    assert get_in(["a"], {}, default="not found") == "not found"
    
    # Test with integer keys in dict
    data = {1: {2: 3}}
    assert get_in([1, 2], data) == 3


# LLM-generated content at query #13
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 1}}}
    assert get_in(["a", "b", "c"], data) == 1
    assert get_in(["a", "b"], data) == {"c": 1}
    assert get_in(["a"], data) == {"b": {"c": 1}}

    # Test list access
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2

    # Test mixed nested structures
    data = {"a": {"b": [1, 2, {"c": 3}]}}
    assert get_in(["a", "b", 0], data) == 1
    assert get_in(["a", "b", 2, "c"], data) == 3

    # Test default value when key not found
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="not found") == "not found"
    assert get_in(["a", "b"], data, default=0) == 0

    # Test no_default parameter
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    try:
        get_in(["a", "b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test IndexError with no_default
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test TypeError with no_default
    data = {"a": 1}
    try:
        get_in(["a", "b"], data, no_default=True)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {"a": 1}
    assert get_in([], data) == data

    # Test with None as valid value
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data, default="default") == "default"

    # Test nested default behavior
    data = {"a": {"b": {}}}
    assert get_in(["a", "b", "c"], data) is None
    assert get_in(["a", "b", "c"], data, default=[]) == []

    # Test with integer keys in dict
    data = {1: {2: {3: "deep"}}}
    assert get_in([1, 2, 3], data) == "deep"

    # Test with list index out of bounds
    data = {"a": [1, 2, 3]}
    assert get_in(["a", 5], data) is None
    assert get_in(["a", 5], data, default="out of bounds") == "out of bounds"


# LLM-generated content at query #14
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"a": [1, 2, {"b": 3}]}
    assert get_in(["a", 2, "b"], data) == 3
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c"], data, default="missing") == "missing"
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test mixed list and dict access
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2
    
    # Test with None in the path
    data = {"a": None}
    assert get_in(["a", "b"], data) is None
    
    # Test with empty dict
    assert get_in(["key"], {}) is None
    
    # Test with empty list
    assert get_in([0], []) is None


# LLM-generated content at query #15
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 1}}}
    assert get_in(["a", "b", "c"], data) == 1
    assert get_in(["a", "b"], data) == {"c": 1}
    assert get_in(["a"], data) == {"b": {"c": 1}}

    # Test list access
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2

    # Test mixed nested structures
    data = {"a": {"b": [1, 2, {"c": 3}]}}
    assert get_in(["a", "b", 0], data) == 1
    assert get_in(["a", "b", 2, "c"], data) == 3

    # Test default value when key not found
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    assert get_in(["b"], data, default=[]) == []

    # Test no_default parameter
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test IndexError with no_default
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test TypeError with no_default
    data = {"a": 1}
    try:
        get_in(["a", "b"], data, no_default=True)
        assert False, "Should have raised TypeError"
    except (TypeError, KeyError):
        pass

    # Test empty keys list
    data = {"a": 1}
    assert get_in([], data) == data

    # Test nested default behavior
    data = {"a": {"b": {}}}
    assert get_in(["a", "b", "c"], data) is None
    assert get_in(["a", "b", "c"], data, default=0) == 0

    # Test with None values in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data, default="default") == "default"

    # Test with integer keys in dict
    data = {1: {2: {3: "value"}}}
    assert get_in([1, 2, 3], data) == "value"


# LLM-generated content at query #16
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    coll = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], coll) == 42
    
    # Test list access
    coll = {'a': [{'b': 10}, {'c': 20}]}
    assert get_in(['a', 0, 'b'], coll) == 10
    assert get_in(['a', 1, 'c'], coll) == 20
    
    # Test default value when key doesn't exist
    coll = {'a': {'b': 1}}
    assert get_in(['a', 'c'], coll) is None
    assert get_in(['a', 'c'], coll, default='not found') == 'not found'
    
    # Test no_default raises KeyError
    coll = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], coll, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    coll = {'a': [1, 2, 3]}
    try:
        get_in(['a', 5], coll, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested list access
    coll = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], coll) == 2
    assert get_in([1, 2], coll) == 6
    
    # Test mixed structures
    coll = {'a': [{'b': 1}, {'c': 2}], 'd': {'e': [3, 4, 5]}}
    assert get_in(['a', 0, 'b'], coll) == 1
    assert get_in(['d', 'e', 2], coll) == 5
    
    # Test empty keys list returns the collection itself
    coll = {'a': 1}
    assert get_in([], coll) == coll
    
    # Test with None as valid value
    coll = {'a': {'b': None}}
    assert get_in(['a', 'b'], coll) is None
    
    # Test default doesn't apply when key exists but value is None
    coll = {'a': {'b': None}}
    assert get_in(['a', 'b'], coll, default='default') is None
    
    # Test TypeError handling (e.g., trying to index into an integer)
    coll = {'a': 42}
    assert get_in(['a', 'b'], coll) is None
    assert get_in(['a', 'b'], coll, default='error') == 'error'


# LLM-generated content at query #17
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"a": [1, 2, {"b": 3}]}
    assert get_in(["a", 2, "b"], data) == 3
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c"], data, default="missing") == "missing"
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test mixed list and dict access
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2
    
    # Test with None in the path
    data = {"a": None}
    assert get_in(["a", "b"], data) is None
    
    # Test with empty collection
    assert get_in(["a"], {}, default="default") == "default"
    
    # Test with integer keys in dict
    data = {1: {2: {3: "value"}}}
    assert get_in([1, 2, 3], data) == "value"


# LLM-generated content at query #18
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"a": [1, 2, {"b": 3}]}
    assert get_in(["a", 2, "b"], data) == 3
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c"], data, default="missing") == "missing"
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test mixed nested structures
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2
    
    # Test with None in structure
    data = {"a": None}
    assert get_in(["a", "b"], data) is None
    
    # Test with empty collection
    assert get_in(["key"], {}, default="default") == "default"
    
    # Test with integer keys in dict
    data = {1: {2: {3: "value"}}}
    assert get_in([1, 2, 3], data) == "value"


# LLM-generated content at query #19
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    assert get_in(["a", "b"], data) == {"c": 42}
    assert get_in(["a"], data) == {"b": {"c": 42}}

    # Test list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6

    # Test mixed nested structures
    data = {"a": [{"b": 10}, {"c": 20}]}
    assert get_in(["a", 0, "b"], data) == 10
    assert get_in(["a", 1, "c"], data) == 20

    # Test default value when key not found
    data = {"x": 1}
    assert get_in(["y"], data) is None
    assert get_in(["y"], data, default="missing") == "missing"
    assert get_in(["x", "y"], data, default=[]) == []

    # Test no_default raises appropriate exceptions
    data = {"x": 1}
    try:
        get_in(["y"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    assert get_in([], [1, 2, 3]) == [1, 2, 3]

    # Test with None values in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data, default="default") == "default"

    # Test with empty collections
    assert get_in(["key"], {}, default="empty") == "empty"
    assert get_in([0], [], default="empty") == "empty"

    # Test nested empty collections
    data = {"a": {}}
    assert get_in(["a", "b"], data) is None
    assert get_in(["a", "b"], data, default=0) == 0


# LLM-generated content at query #20
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"items": ["apple", "banana", "cherry"]}
    assert get_in(["items", 1], data) == "banana"
    
    # Test mixed nested structures
    data = {"a": [{"b": 10}, {"b": 20}]}
    assert get_in(["a", 1, "b"], data) == 20
    
    # Test default value when key doesn't exist
    data = {"x": 1}
    assert get_in(["y"], data) is None
    assert get_in(["y"], data, default="missing") == "missing"
    
    # Test no_default raises KeyError
    data = {"x": 1}
    try:
        get_in(["y"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test nested missing key with default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c", "d"], data, default="not found") == "not found"
    
    # Test with None as valid value
    data = {"a": None}
    assert get_in(["a"], data) is None
    
    # Test accessing non-indexable type raises TypeError
    data = {"a": 42}
    try:
        get_in(["a", "b"], data, no_default=True)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    # Test with empty collection
    assert get_in(["key"], {}, default="empty") == "empty"
    
    # Test complex nested structure
    data = {
        "users": [
            {"name": "Alice", "scores": [85, 92, 78]},
            {"name": "Bob", "scores": [88, 95, 82]}
        ]
    }
    assert get_in(["users", 1, "scores", 2], data) == 82


# LLM-generated content at query #21
#--------------------------

```python
def test_get_in():
    # Test with nested dictionaries
    nested_dict = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], nested_dict) == 42
    assert get_in(['a', 'b'], nested_dict) == {'c': 42}
    assert get_in(['a'], nested_dict) == {'b': {'c': 42}}
    
    # Test with mixed nested structures (dicts and lists)
    mixed = {'items': [{'name': 'apple', 'price': 1.0}, {'name': 'orange', 'price': 1.5}]}
    assert get_in(['items', 0, 'name'], mixed) == 'apple'
    assert get_in(['items', 1, 'price'], mixed) == 1.5
    
    # Test default value when key doesn't exist
    assert get_in(['x', 'y', 'z'], {}) is None
    assert get_in(['a', 'b', 'd'], nested_dict) is None
    assert get_in(['items', 2], mixed) is None
    
    # Test custom default value
    assert get_in(['x', 'y'], {}, default='not found') == 'not found'
    assert get_in(['a', 'b', 'd'], nested_dict, default=0) == 0
    
    # Test no_default=True raises KeyError for missing keys
    try:
        get_in(['x'], {}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    try:
        get_in(['a', 'b', 'd'], nested_dict, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default=True raises IndexError for out of bounds list access
    try:
        get_in(['items', 5], mixed, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test with empty key list returns the collection itself
    assert get_in([], nested_dict) == nested_dict
    assert get_in([], mixed) == mixed
    
    # Test with None collection
    assert get_in(['a'], None) is None
    assert get_in(['a'], None, default='default') == 'default'
    
    # Test with non-iterable intermediate values
    assert get_in(['a', 'b'], {'a': 42}) is None
    assert get_in(['a', 'b'], {'a': 42}, no_default=True) is None
    
    # Test with example from docstring
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #22
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    assert get_in(["a", "b"], data) == {"c": 42}
    assert get_in(["a"], data) == {"b": {"c": 42}}

    # Test list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6

    # Test mixed nested structures
    data = {"a": [{"b": 10}, {"c": 20}]}
    assert get_in(["a", 0, "b"], data) == 10
    assert get_in(["a", 1, "c"], data) == 20

    # Test default value when key not found
    data = {"x": 1}
    assert get_in(["y"], data) is None
    assert get_in(["y"], data, default="not found") == "not found"
    assert get_in(["y"], data, default=0) == 0

    # Test no_default=True raises appropriate error
    data = {"x": 1}
    try:
        get_in(["y"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test no_default=True with list index error
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection itself
    data = {"a": 1}
    assert get_in([], data) == data
    assert get_in([], data, default="default") == data

    # Test nested default behavior
    data = {"a": {}}
    assert get_in(["a", "b"], data) is None
    assert get_in(["a", "b"], data, default="missing") == "missing"

    # Test with None values in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    try:
        get_in(["a", "b"], data, no_default=True)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    # Test with integer keys in dictionary
    data = {1: {2: {3: "value"}}}
    assert get_in([1, 2, 3], data) == "value"

    # Test with string index in list (should return default)
    data = [[1, 2, 3]]
    assert get_in([0, "a"], data) is None
    try:
        get_in([0, "a"], data, no_default=True)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], data) == 42
    assert get_in(['a', 'b'], data) == {'c': 42}
    assert get_in(['a'], data) == {'b': {'c': 42}}

    # Test list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6

    # Test mixed nested structures
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], data) == 1
    assert get_in(['a', 1, 'c'], data) == 2

    # Test default value when key doesn't exist
    data = {'a': 1}
    assert get_in(['b'], data) is None
    assert get_in(['b'], data, default='missing') == 'missing'
    assert get_in(['a', 'b'], data, default=0) == 0

    # Test no_default=True raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test no_default=True raises IndexError
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection itself
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with None values in collection
    data = {'a': None}
    assert get_in(['a'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test nested default behavior
    data = {'a': {'b': {}}}
    assert get_in(['a', 'b', 'c'], data) is None
    assert get_in(['a', 'b', 'c'], data, default=0) == 0

    # Test with integer keys in dictionary
    data = {1: {2: {3: 'deep'}}}
    assert get_in([1, 2, 3], data) == 'deep'

    # Test with tuple as key
    data = {('a', 'b'): 'value'}
    assert get_in([('a', 'b')], data) == 'value'


# LLM-generated content at query #24
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 1}}}
    assert get_in(["a", "b", "c"], data) == 1
    assert get_in(["a", "b"], data) == {"c": 1}
    assert get_in(["a"], data) == {"b": {"c": 1}}

    # Test list access
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2

    # Test mixed nested structures
    data = {"a": {"b": [1, 2, {"c": 3}]}}
    assert get_in(["a", "b", 0], data) == 1
    assert get_in(["a", "b", 2, "c"], data) == 3

    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="missing") == "missing"
    assert get_in(["b"], data, default=0) == 0

    # Test no_default=True raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test no_default=True raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection itself
    data = {"a": 1}
    assert get_in([], data) == data

    # Test nested missing key with default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c", "d"], data) is None
    assert get_in(["a", "c", "d"], data, default="missing") == "missing"

    # Test with None values in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data) is None

    # Test with empty collections
    assert get_in(["a"], {}) is None
    assert get_in([0], []) is None
    assert get_in([0], [], default="empty") == "empty"

    # Test complex nested structure
    data = {
        "users": [
            {"name": "Alice", "scores": [85, 92, 78]},
            {"name": "Bob", "scores": [88, 95, 82]}
        ]
    }
    assert get_in(["users", 0, "name"], data) == "Alice"
    assert get_in(["users", 1, "scores", 1], data) == 95
    assert get_in(["users", 0, "scores", 5], data) is None
    assert get_in(["users", 2, "name"], data) is None


# LLM-generated content at query #25
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"a": [1, 2, {"b": 3}]}
    assert get_in(["a", 2, "b"], data) == 3
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested missing key with default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c", "d"], data, default="missing") == "missing"
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test mixed nested structures
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2
    
    # Test with None as valid value
    data = {"a": None}
    assert get_in(["a"], data) is None
    
    # Test accessing list index
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6
    
    # Test TypeError handling (e.g., trying to index into an integer)
    data = {"a": 42}
    assert get_in(["a", "b"], data, default="error") == "error"


# LLM-generated content at query #26
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    coll = {'a': {'b': {'c': 42}}}
    assert get_in(['a', 'b', 'c'], coll) == 42
    assert get_in(['a', 'b'], coll) == {'c': 42}
    assert get_in(['a'], coll) == {'b': {'c': 42}}

    # Test list access
    coll = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], coll) == 2
    assert get_in([1, 2], coll) == 6

    # Test mixed nested structures
    coll = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], coll) == 1
    assert get_in(['a', 1, 'c'], coll) == 2

    # Test default value when key not found
    coll = {'a': 1}
    assert get_in(['b'], coll) is None
    assert get_in(['b'], coll, default='not found') == 'not found'
    assert get_in(['b'], coll, default=0) == 0

    # Test no_default raises KeyError
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test no_default raises IndexError
    coll = [1, 2, 3]
    try:
        get_in([5], coll, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection
    coll = {'a': 1}
    assert get_in([], coll) == coll

    # Test nested default behavior
    coll = {'a': {'b': {}}}
    assert get_in(['a', 'b', 'c'], coll) is None
    assert get_in(['a', 'b', 'c'], coll, default='missing') == 'missing'

    # Test with None values in collection
    coll = {'a': None}
    assert get_in(['a'], coll) is None
    assert get_in(['a', 'b'], coll) is None

    # Test with empty collection
    assert get_in(['key'], {}) is None
    assert get_in([0], []) is None

    # Test complex nested structure
    coll = {
        'users': [
            {'name': 'Alice', 'scores': [85, 92, 78]},
            {'name': 'Bob', 'scores': [88, 95, 82]}
        ]
    }
    assert get_in(['users', 0, 'name'], coll) == 'Alice'
    assert get_in(['users', 1, 'scores', 2], coll) == 82
    assert get_in(['users', 0, 'scores', 5], coll) is None
    assert get_in(['users', 2, 'name'], coll) is None


# LLM-generated content at query #27
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    assert get_in(["a", "b"], data) == {"c": 42}
    assert get_in(["a"], data) == {"b": {"c": 42}}

    # Test list access
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6

    # Test mixed nested structures
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2

    # Test default value when key not found
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="not found") == "not found"
    assert get_in(["a", "b"], data, default=0) == 0

    # Test no_default=True raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # Test no_default=True raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection itself
    data = {"a": 1}
    assert get_in([], data) == data
    assert get_in([], data, default="default") == data

    # Test with None values in structure
    data = {"a": None}
    assert get_in(["a"], data) is None
    assert get_in(["a", "b"], data, default="default") == "default"

    # Test with empty collections
    assert get_in(["key"], {}, default="default") == "default"
    assert get_in([0], [], default="default") == "default"

    # Test nested empty collections
    data = {"a": {}}
    assert get_in(["a", "b"], data, default="missing") == "missing"

    # Test with integer keys in dictionary
    data = {1: {2: {3: "deep"}}}
    assert get_in([1, 2, 3], data) == "deep"

    # Test with string index in list (should return default)
    data = [1, 2, 3]
    assert get_in(["0"], data, default="not found") == "not found"


# LLM-generated content at query #28
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"a": [1, 2, {"b": 3}]}
    assert get_in(["a", 2, "b"], data) == 3
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested missing key with default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c", "d"], data, default="missing") == "missing"
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test mixed list and dict access
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2
    
    # Test with None as valid value
    data = {"a": None}
    assert get_in(["a"], data) is None
    
    # Test accessing non-container returns default
    data = {"a": 1}
    assert get_in(["a", "b"], data) is None
    
    # Test with custom default value
    data = {"a": {"b": 1}}
    assert get_in(["a", "c"], data, default=0) == 0


# LLM-generated content at query #29
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {"a": {"b": {"c": 42}}}
    assert get_in(["a", "b", "c"], data) == 42
    
    # Test list access
    data = {"a": [1, 2, {"b": 3}]}
    assert get_in(["a", 2, "b"], data) == 3
    
    # Test default value when key doesn't exist
    data = {"a": 1}
    assert get_in(["b"], data) is None
    assert get_in(["b"], data, default="default") == "default"
    
    # Test no_default raises KeyError
    data = {"a": 1}
    try:
        get_in(["b"], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    
    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass
    
    # Test nested default
    data = {"a": {"b": 1}}
    assert get_in(["a", "c"], data, default="missing") == "missing"
    
    # Test empty keys list returns the collection
    data = {"a": 1}
    assert get_in([], data) == data
    
    # Test mixed list and dict access
    data = {"a": [{"b": 1}, {"c": 2}]}
    assert get_in(["a", 0, "b"], data) == 1
    assert get_in(["a", 1, "c"], data) == 2
    
    # Test with None in the path
    data = {"a": None}
    assert get_in(["a", "b"], data) is None
    
    # Test with empty collection
    assert get_in(["a"], {}) is None
    assert get_in([0], []) is None


