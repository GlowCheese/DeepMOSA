####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_get_in():
    # Test basic nested dictionary access
    data = {'a': {'b': {'c': 'value'}}}
    assert get_in(['a', 'b', 'c'], data) == 'value'
    
    # Test nested list access
    data = {'items': [1, 2, 3]}
    assert get_in(['items', 0], data) == 1
    assert get_in(['items', 2], data) == 3
    
    # Test mixed dict and list access
    data = {'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}}
    assert get_in(['purchase', 'items', 0], data) == 'Apple'
    assert get_in(['purchase', 'items', 1], data) == 'Orange'
    assert get_in(['purchase', 'costs', 0], data) == 0.50
    
    # Test single key access
    data = {'name': 'Alice'}
    assert get_in(['name'], data) == 'Alice'
    
    # Test empty keys returns the collection itself
    data = {'a': 1}
    assert get_in([], data) == data
    
    # Test missing key returns None by default
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['x'], data) is None
    assert get_in(['x', 'y'], data) is None
    
    # Test default value
    data = {'a': 1}
    assert get_in(['b'], data, default=0) == 0
    assert get_in(['b'], data, default='default') == 'default'
    
    # Test no_default=True raises KeyError
    data = {'a': 1}
    import pytest
    with pytest.raises(KeyError):
        get_in(['b'], data, no_default=True)
    
    # Test no_default=True raises IndexError for list
    data = [1, 2, 3]
    with pytest.raises(IndexError):
        get_in([10], data, no_default=True)
    
    # Test with None values in structure
    data = {'a': None}
    assert get_in(['a'], data) is None
    
    # Test TypeError for invalid access (e.g., indexing a dict with int)
    data = {'a': {'b': 1}}
    assert get_in(['a', 0], data) is None
    
    # Test with no_default=True raises TypeError
    with pytest.raises(TypeError):
        get_in(['a', 0], data, no_default=True)
    
    # Test complex nested structure
    data = {
        'users': [
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob', 'age': 25}
        ]
    }
    assert get_in(['users', 0, 'name'], data) == 'Alice'
    assert get_in(['users', 1, 'age'], data) == 25


# LLM-generated content at query #2
#--------------------------

```python
def test_get_in():
    # Test basic nested dictionary access
    data = {'a': {'b': {'c': 'value'}}}
    assert get_in(['a', 'b', 'c'], data) == 'value'
    
    # Test nested list access
    data = {'items': ['Apple', 'Orange', 'Banana']}
    assert get_in(['items', 0], data) == 'Apple'
    assert get_in(['items', 2], data) == 'Banana'
    
    # Test mixed dictionary and list access
    data = {'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}}
    assert get_in(['purchase', 'items', 0], data) == 'Apple'
    assert get_in(['purchase', 'costs', 1], data) == 1.25
    
    # Test single key
    data = {'name': 'Alice'}
    assert get_in(['name'], data) == 'Alice'
    
    # Test empty keys
    data = {'a': 'value'}
    assert get_in([], data) == data
    
    # Test missing key with default None
    data = {'a': 'value'}
    assert get_in(['b'], data) is None
    assert get_in(['a', 'b'], data) is None
    
    # Test missing key with custom default
    data = {'purchase': {'total': None}}
    assert get_in(['purchase', 'total'], data, 0) is None
    assert get_in(['purchase', 'items'], data, 0) == 0
    
    # Test missing index with default
    data = {'items': ['Apple']}
    assert get_in(['items', 10], data) is None
    assert get_in(['items', 10], data, 'default') == 'default'
    
    # Test no_default=True raises KeyError
    data = {}
    with pytest.raises(KeyError):
        get_in(['y'], data, no_default=True)
    
    # Test no_default=True raises IndexError
    data = {'items': [1, 2, 3]}
    with pytest.raises(IndexError):
        get_in(['items', 10], data, no_default=True)
    
    # Test no_default=True with valid path
    data = {'a': {'b': 'value'}}
    assert get_in(['a', 'b'], data, no_default=True) == 'value'
    
    # Test with None values in path
    data = {'a': None}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, 'default') == 'default'
    
    # Test with numeric keys in dictionary
    data = {0: {1: 'value'}}
    assert get_in([0, 1], data) == 'value'
    
    # Test with tuple as collection
    data = ({'a': 'value'}, {'b': 'value2'})
    assert get_in([0, 'a'], data) == 'value'
    assert get_in([1, 'b'], data) == 'value2'


# LLM-generated content at query #3
#--------------------------

```python
def test_get_in():
    # Test with nested dictionaries
    data = {'a': {'b': {'c': 'value'}}}
    assert get_in(['a', 'b', 'c'], data) == 'value'
    
    # Test with nested lists
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 1], data) == 4
    
    # Test with mixed dict and list
    data = {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}
    assert get_in(['items', 0], data) == 'Apple'
    assert get_in(['costs', 1], data) == 1.25
    
    # Test with single key
    data = {'name': 'Alice'}
    assert get_in(['name'], data) == 'Alice'
    
    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data
    
    # Test missing key returns default None
    data = {'a': 1}
    assert get_in(['b'], data) is None
    
    # Test missing key with custom default
    data = {'a': 1}
    assert get_in(['b'], data, default=0) == 0
    assert get_in(['b'], data, default='missing') == 'missing'
    
    # Test missing nested key returns default
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default=42) == 42
    
    # Test missing index returns default
    data = [1, 2, 3]
    assert get_in([10], data) is None
    assert get_in([10], data, default='out of range') == 'out of range'
    
    # Test no_default=True raises KeyError for missing dict key
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Should raise KeyError"
    except KeyError:
        pass
    
    # Test no_default=True raises IndexError for missing list index
    data = [1, 2, 3]
    try:
        get_in([10], data, no_default=True)
        assert False, "Should raise IndexError"
    except IndexError:
        pass
    
    # Test no_default=True raises KeyError for missing nested key
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Should raise KeyError"
    except KeyError:
        pass
    
    # Test with None value in structure (should return None, not default)
    data = {'a': None}
    assert get_in(['a'], data) is None
    
    # Test TypeError when accessing dict with integer key
    data = {'a': 1}
    assert get_in(['a', 0], data) is None
    assert get_in(['a', 0], data, default='error') == 'error'
    
    # Test TypeError when accessing list with string key
    data = [1, 2, 3]
    assert get_in([0, 'key'], data) is None
    assert get_in([0, 'key'], data, default='error') == 'error'
    
    # Test complex nested structure
    transaction = {
        'name': 'Alice',
        'purchase': {
            'items': ['Apple', 'Orange'],
            'costs': [0.50, 1.25]
        },
        'credit card': '5555-1234-1234-1234'
    }
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #4
#--------------------------

```python
def test_get_in():
    # Test with nested dictionaries
    data = {'a': {'b': {'c': 'value'}}}
    assert get_in(['a', 'b', 'c'], data) == 'value'
    
    # Test with nested lists
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    
    # Test with mixed dict and list
    data = {'items': [{'name': 'Apple'}, {'name': 'Orange'}]}
    assert get_in(['items', 0, 'name'], data) == 'Apple'
    assert get_in(['items', 1, 'name'], data) == 'Orange'
    
    # Test with single key
    data = {'name': 'Alice'}
    assert get_in(['name'], data) == 'Alice'
    
    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data
    
    # Test missing key returns default None
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['x', 'y', 'z'], data) is None
    
    # Test missing key with custom default
    data = {'a': 1}
    assert get_in(['b'], data, default=0) == 0
    assert get_in(['x', 'y'], data, default='not found') == 'not found'
    
    # Test missing index in list returns default
    data = [1, 2, 3]
    assert get_in([10], data) is None
    assert get_in([10], data, default='default') == 'default'
    
    # Test no_default=True raises KeyError
    data = {'a': 1}
    with pytest.raises(KeyError):
        get_in(['b'], data, no_default=True)
    
    # Test no_default=True raises IndexError
    data = [1, 2, 3]
    with pytest.raises(IndexError):
        get_in([10], data, no_default=True)
    
    # Test with TypeError (accessing nested structure incorrectly)
    data = {'a': 'string_value'}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 0], data) is None
    
    # Test no_default=True raises TypeError
    data = {'a': 'string_value'}
    with pytest.raises(TypeError):
        get_in(['a', 'b'], data, no_default=True)
    
    # Test with complex nested structure
    data = {
        'users': [
            {'name': 'Alice', 'scores': [10, 20, 30]},
            {'name': 'Bob', 'scores': [15, 25, 35]}
        ]
    }
    assert get_in(['users', 0, 'name'], data) == 'Alice'
    assert get_in(['users', 1, 'scores', 2], data) == 35
    assert get_in(['users', 0, 'scores', 0], data) == 10
    
    # Test with None values in structure
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    
    # Test empty collection
    assert get_in(['a'], {}) is None
    assert get_in([0], []) is None
    assert get_in(['a'], {}, default='empty') == 'empty'


# LLM-generated content at query #5
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 'value'}}}
    assert get_in(['a', 'b', 'c'], data) == 'value'
    
    # Test with nested list
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 1], data) == 4
    
    # Test with mixed dict and list
    data = {'items': [{'name': 'Apple'}, {'name': 'Orange'}]}
    assert get_in(['items', 0, 'name'], data) == 'Apple'
    assert get_in(['items', 1, 'name'], data) == 'Orange'
    
    # Test with single key
    data = {'name': 'Alice'}
    assert get_in(['name'], data) == 'Alice'
    
    # Test with empty keys
    data = {'a': 'value'}
    assert get_in([], data) == data
    
    # Test KeyError with default
    data = {'a': 'value'}
    assert get_in(['b'], data) is None
    assert get_in(['b'], data, 'default') == 'default'
    
    # Test IndexError with default
    data = [1, 2, 3]
    assert get_in([10], data) is None
    assert get_in([10], data, 'default') == 'default'
    
    # Test KeyError with no_default=True
    data = {'a': 'value'}
    with pytest.raises(KeyError):
        get_in(['b'], data, no_default=True)
    
    # Test IndexError with no_default=True
    data = [1, 2, 3]
    with pytest.raises(IndexError):
        get_in([10], data, no_default=True)
    
    # Test TypeError with no_default=True (accessing key on non-dict/list)
    data = {'a': 'string_value'}
    with pytest.raises(TypeError):
        get_in(['a', 'b'], data, no_default=True)
    
    # Test with None in path
    data = {'a': None}
    assert get_in(['a'], data) is None
    
    # Test with None in path trying to access further
    data = {'a': None}
    assert get_in(['a', 'b'], data) is None
    
    # Test with custom default value
    data = {}
    assert get_in(['missing'], data, default=42) == 42
    
    # Test nested missing key with default
    data = {'a': {'b': 'value'}}
    assert get_in(['a', 'c'], data, default='not_found') == 'not_found'
    
    # Test with numeric keys in dict
    data = {0: {1: 'value'}}
    assert get_in([0, 1], data) == 'value'


# LLM-generated content at query #6
#--------------------------

```python
def test_get_in():
    # Test basic nested dictionary access
    data = {'a': {'b': {'c': 'value'}}}
    assert get_in(['a', 'b', 'c'], data) == 'value'
    
    # Test nested list access
    data = {'items': ['Apple', 'Orange', 'Banana']}
    assert get_in(['items', 0], data) == 'Apple'
    assert get_in(['items', 1], data) == 'Orange'
    assert get_in(['items', 2], data) == 'Banana'
    
    # Test mixed dictionary and list access
    data = {'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}}
    assert get_in(['purchase', 'items', 0], data) == 'Apple'
    assert get_in(['purchase', 'costs', 1], data) == 1.25
    
    # Test simple key access
    data = {'name': 'Alice'}
    assert get_in(['name'], data) == 'Alice'
    
    # Test empty keys
    data = {'a': 'value'}
    assert get_in([], data) == data
    
    # Test default value when key doesn't exist
    data = {'a': {'b': 'value'}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['x', 'y'], data) is None
    
    # Test custom default value
    data = {'a': {'b': 'value'}}
    assert get_in(['a', 'c'], data, default=0) == 0
    assert get_in(['x'], data, default='default') == 'default'
    
    # Test list index out of bounds
    data = {'items': ['Apple', 'Orange']}
    assert get_in(['items', 10], data) is None
    assert get_in(['items', 10], data, default='missing') == 'missing'
    
    # Test no_default flag raises KeyError for missing dict key
    data = {'a': 'value'}
    with pytest.raises(KeyError):
        get_in(['b'], data, no_default=True)
    
    # Test no_default flag raises IndexError for missing list index
    data = {'items': ['Apple']}
    with pytest.raises(IndexError):
        get_in(['items', 10], data, no_default=True)
    
    # Test no_default flag raises KeyError for nested missing key
    data = {'a': {'b': 'value'}}
    with pytest.raises(KeyError):
        get_in(['a', 'c'], data, no_default=True)
    
    # Test TypeError when accessing non-subscriptable object
    data = {'a': 'string'}
    assert get_in(['a', 'b'], data) is None
    
    # Test TypeError with no_default
    data = {'a': 'string'}
    with pytest.raises(TypeError):
        get_in(['a', 'b'], data, no_default=True)
    
    # Test with nested lists
    data = [[1, 2], [3, 4], [5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([2, 0], data) == 5
    
    # Test with deeply nested structures
    data = {'level1': {'level2': {'level3': {'level4': 'deep'}}}}
    assert get_in(['level1', 'level2', 'level3', 'level4'], data) == 'deep'


# LLM-generated content at query #7
#--------------------------

```python
def test_get_in():
    # Test basic nested dictionary access
    data = {'a': {'b': {'c': 'value'}}}
    assert get_in(['a', 'b', 'c'], data) == 'value'
    
    # Test nested list access
    data = {'items': [1, 2, 3]}
    assert get_in(['items', 0], data) == 1
    assert get_in(['items', 2], data) == 3
    
    # Test mixed dictionary and list access
    data = {'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}}
    assert get_in(['purchase', 'items', 0], data) == 'Apple'
    assert get_in(['purchase', 'items', 1], data) == 'Orange'
    assert get_in(['purchase', 'costs', 0], data) == 0.50
    
    # Test single key access
    data = {'name': 'Alice'}
    assert get_in(['name'], data) == 'Alice'
    
    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data
    
    # Test missing key returns default None
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    
    # Test missing key with custom default
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=0) == 0
    assert get_in(['x', 'y'], data, default='missing') == 'missing'
    
    # Test missing index in list returns default None
    data = {'items': [1, 2, 3]}
    assert get_in(['items', 10], data) is None
    
    # Test missing index with custom default
    data = {'items': [1, 2, 3]}
    assert get_in(['items', 10], data, default=-1) == -1
    
    # Test no_default=True raises KeyError for missing key
    data = {'a': {}}
    with pytest.raises(KeyError):
        get_in(['a', 'missing'], data, no_default=True)
    
    # Test no_default=True raises IndexError for missing index
    data = {'items': [1, 2]}
    with pytest.raises(IndexError):
        get_in(['items', 10], data, no_default=True)
    
    # Test accessing None value
    data = {'a': None}
    assert get_in(['a'], data) is None
    
    # Test TypeError when accessing non-dict/list
    data = {'a': 'string'}
    assert get_in(['a', 0], data) is None
    
    # Test TypeError with no_default=True
    data = {'a': 'string'}
    with pytest.raises(TypeError):
        get_in(['a', 0], data, no_default=True)
    
    # Test deeply nested structure
    data = {'level1': {'level2': {'level3': {'level4': 'deep'}}}}
    assert get_in(['level1', 'level2', 'level3', 'level4'], data) == 'deep'
    
    # Test list of dicts
    data = [{'name': 'Alice'}, {'name': 'Bob'}]
    assert get_in([0, 'name'], data) == 'Alice'
    assert get_in([1, 'name'], data) == 'Bob'
    
    # Test with zero as default
    data = {'a': {}}
    assert get_in(['a', 'b'], data, default=0) == 0
    
    # Test with False as default
    data = {'a': {}}
    assert get_in(['a', 'b'], data, default=False) is False


# LLM-generated content at query #8
#--------------------------

```python
def test_get_in():
    # Test basic nested dictionary access
    data = {'a': {'b': {'c': 'value'}}}
    assert get_in(['a', 'b', 'c'], data) == 'value'
    
    # Test nested list access
    data = {'items': [1, 2, 3]}
    assert get_in(['items', 0], data) == 1
    assert get_in(['items', 2], data) == 3
    
    # Test mixed dictionary and list access
    data = {'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}}
    assert get_in(['purchase', 'items', 0], data) == 'Apple'
    assert get_in(['purchase', 'items', 1], data) == 'Orange'
    assert get_in(['purchase', 'costs', 1], data) == 1.25
    
    # Test single key access
    data = {'name': 'Alice'}
    assert get_in(['name'], data) == 'Alice'
    
    # Test empty keys list returns the collection itself
    data = {'a': 1}
    assert get_in([], data) == data
    
    # Test missing key returns default (None)
    data = {'a': {'b': 1}}
    assert get_in(['a', 'missing'], data) is None
    
    # Test missing key with custom default
    data = {'a': {'b': 1}}
    assert get_in(['a', 'missing'], data, default=0) == 0
    assert get_in(['missing'], data, default='not found') == 'not found'
    
    # Test missing index in list returns default
    data = {'items': [1, 2, 3]}
    assert get_in(['items', 10], data) is None
    assert get_in(['items', 10], data, default=-1) == -1
    
    # Test no_default=True raises KeyError for missing key
    data = {'a': 1}
    with pytest.raises(KeyError):
        get_in(['missing'], data, no_default=True)
    
    # Test no_default=True raises IndexError for missing index
    data = {'items': [1, 2, 3]}
    with pytest.raises(IndexError):
        get_in(['items', 10], data, no_default=True)
    
    # Test TypeError when accessing non-subscriptable object
    data = {'a': 42}
    assert get_in(['a', 'b'], data) is None
    
    # Test TypeError with no_default=True
    data = {'a': 42}
    with pytest.raises(TypeError):
        get_in(['a', 'b'], data, no_default=True)
    
    # Test with nested lists
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    
    # Test with complex nested structure
    data = {
        'users': [
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob', 'age': 25}
        ]
    }
    assert get_in(['users', 0, 'name'], data) == 'Alice'
    assert get_in(['users', 1, 'age'], data) == 25
    assert get_in(['users', 2, 'name'], data) is None


