####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}

    # Test with list
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([1, 1], data) == [3, 4]
    assert get_in([1], data) == [2, [3, 4]]

    # Test with default value
    assert get_in(['x'], data, default='not found') == 'not found'
    assert get_in(['x', 'y'], data, default='not found') == 'not found'
    assert get_in([5], data, default='not found') == 'not found'

    # Test with no_default=True
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with empty keys list
    assert get_in([], data) == data

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a'], data) == [1, {'b': 2}]

    # Test with default=None
    assert get_in(['x'], data) is None
    assert get_in(['a', 5], data) is None


# LLM-generated content at query #2
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with nested list
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 1, 1], data) == 4
    assert get_in([1, 0], data) == [5, 6]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, 2, {'b': 3}]}
    assert get_in(['a', 2, 'b'], data) == 3
    assert get_in(['a', 1], data) == 2
    assert get_in(['a', 3], data) is None
    assert get_in(['a', 3], data, default=0) == 0

    # Test with no_default=True
    data = {'a': {'b': 1}}
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in(['a', 'x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with non-existent keys and no default
    data = {'a': {'b': 1}}
    assert get_in(['x', 'y', 'z'], data) is None
    assert get_in(['a', 'x', 'y'], data) is None

    # Test with TypeError (e.g., trying to index a non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {
        'a': {
            'b': {
                'c': 1,
                'd': 2
            },
            'e': [3, 4, 5]
        },
        'f': 'value'
    }

    # Test successful nested access
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b', 'd'], data) == 2
    assert get_in(['a', 'e', 1], data) == 4
    assert get_in(['f'], data) == 'value'

    # Test with default value
    assert get_in(['a', 'b', 'x'], data) is None
    assert get_in(['a', 'b', 'x'], data, default='default') == 'default'
    assert get_in(['a', 'x', 'y'], data, default=0) == 0

    # Test with no_default=True
    try:
        get_in(['a', 'b', 'x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in(['a', 'e', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with empty keys list
    assert get_in([], data) == data

    # Test with list data structure
    list_data = [[1, 2], [3, 4], [5, 6]]
    assert get_in([1, 1], list_data) == 4
    assert get_in([2], list_data) == [5, 6]
    assert get_in([3], list_data) is None
    assert get_in([3], list_data, default='missing') == 'missing'

    # Test with mixed data structures
    mixed_data = {
        'items': ['apple', 'banana', 'cherry'],
        'prices': {
            'apple': 1.0,
            'banana': 0.5
        }
    }
    assert get_in(['items', 1], mixed_data) == 'banana'
    assert get_in(['prices', 'apple'], mixed_data) == 1.0
    assert get_in(['prices', 'cherry'], mixed_data) is None
    assert get_in(['prices', 'cherry'], mixed_data, default=0.0) == 0.0

    # Test with TypeError (non-subscriptable object)
    assert get_in(['a', 'b', 'c', 'd'], data) is None
    assert get_in(['a', 'b', 'c', 'd'], data, default='error') == 'error'


# LLM-generated content at query #4
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1], data) == [3, 4]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default=0) == 0

    # Test with no_default=True
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=0) == 0
    assert get_in(['a', 'c'], data, default=None) is None
    assert get_in(['a', 'c'], data, default='default') == 'default'


# LLM-generated content at query #5
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1], data) == [3, 4]
    assert get_in([2], data) is None

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1], data) == {'b': 2}

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with default value
    assert get_in(['x', 'y', 'z'], {}, default='default') == 'default'
    assert get_in(['x', 'y', 'z'], {}, default=None) is None

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 'string'}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='error') == 'error'


# LLM-generated content at query #6
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with non-existent nested keys
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default=0) == 0

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None


# LLM-generated content at query #7
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=99) == 99
    assert get_in(['a', 'c'], data) is None

    # Test no_default raises exception
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError with list
    data = {'a': [1, 2]}
    try:
        get_in(['a', 5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError with non-subscriptable
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test nested mixed structures
    data = {'a': [{'b': [1, 2, 3]}, {'c': 4}]}
    assert get_in(['a', 0, 'b', 1], data) == 2
    assert get_in(['a', 1, 'c'], data) == 4

    # Test with persistent data structure (assuming pyrsistent is available)
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #8
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'

    # Test with no_default=True
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with list
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2
    assert get_in(['a', 10], data) is None

    # Test with mixed types
    data = {'a': {'b': [1, 2, {'c': 3}]}}
    assert get_in(['a', 'b', 2, 'c'], data) == 3
    assert get_in(['a', 'b', 2, 'd'], data) is None

    # Test with empty keys
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None


# LLM-generated content at query #9
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1], data) == [3, 4]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default=0) == 0

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #10
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1], data) == [3, 4]
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default='default') == 'default'

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') is None
    assert get_in(['a', 'c'], data, default='default') == 'default'


# LLM-generated content at query #11
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with nested list
    data = [[[1, 2], 3], 4]
    assert get_in([0, 0, 1], data) == 2
    assert get_in([1], data) == 4
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1, 'c'], data) is None
    assert get_in(['a', 1, 'c'], data, default=0) == 0

    # Test with no_default=True
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) is None
    assert get_in(['a', 'c'], data, default=0) == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key doesn't exist
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=0) == 0

    # Test default value when key doesn't exist (no default provided)
    assert get_in(['a', 'c'], data) is None

    # Test with no_default=True raises KeyError
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with nested list and dictionary
    data = {'a': [{'b': 1}, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2

    # Test with IndexError in list
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default=0) == 0

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=0) == 0

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with single key
    data = {'a': 1}
    assert get_in(['a'], data) == 1


# LLM-generated content at query #13
#--------------------------

```python
def test_get_in():
    # Test basic nested dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default='not found') == 'not found'

    # Test default value when key not found (default None)
    assert get_in(['a', 'c'], data) is None

    # Test no_default raises KeyError
    with pytest.raises(KeyError):
        get_in(['a', 'c'], data, no_default=True)

    # Test IndexError for list out of bounds
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default='out of bounds') == 'out of bounds'

    # Test no_default raises IndexError
    with pytest.raises(IndexError):
        get_in(['a', 10], data, no_default=True)

    # Test TypeError for invalid key type
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default='type error') == 'type error'

    # Test no_default raises TypeError
    with pytest.raises(TypeError):
        get_in(['a', 'b'], data, no_default=True)

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structure (if available)
    try:
        from pyrsistent import freeze
        data = freeze({'a': {'b': 1}})
        assert get_in(['a', 'b'], data) == 1
    except ImportError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1, 'c'], data) is None

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with default value
    assert get_in(['x', 'y'], {}, default='not_found') == 'not_found'

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable)
    data = "string"
    assert get_in([0], data) == 's'
    assert get_in([1], data) == 't'
    assert get_in(['x'], data) is None


# LLM-generated content at query #15
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1, 'x'], data) is None
    assert get_in(['a', 1, 'x'], data, default=0) == 0

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([0], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': None}
    assert get_in(['a'], data) is None
    assert get_in(['a'], data, default=0) is None
    assert get_in(['x'], data, default=0) == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 'b', 'x'], data) is None

    # Test with list
    data = [1, 2, [3, 4]]
    assert get_in([2, 1], data) == 4
    assert get_in([0], data) == 1
    assert get_in([5], data) is None
    assert get_in([2, 5], data) is None

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1, 'x'], data) is None

    # Test no_default
    data = {'a': 1}
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([5], [1, 2], no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with default value
    assert get_in(['x'], {}, default=0) == 0
    assert get_in(['x', 'y'], {}, default=[]) == []
    assert get_in([5], [1, 2], default='out_of_bounds') == 'out_of_bounds'

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data
    assert get_in([], data, default='default') == data

    # Test with TypeError (non-subscriptable)
    data = 123
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='not_subscriptable') == 'not_subscriptable'


# LLM-generated content at query #17
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 'b', 'x'], data) is None
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([2], data, default='out_of_bounds') == 'out_of_bounds'

    # Test with mixed nested structure
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([0], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with empty keys list
    assert get_in([], data) == data

    # Test with default value
    assert get_in(['x', 'y', 'z'], {}, default='not_found') == 'not_found'


# LLM-generated content at query #18
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default return when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default='default') == 'default'

    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError for list out of bounds
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data) is None
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError for non-subscriptable types
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structures (assuming pyrsistent is available)
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #19
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3

    # Test with default value
    assert get_in(['x'], data, default='not found') == 'not found'
    assert get_in(['a', 'b', 'c'], data, default=0) == 0

    # Test with no_default=True
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2

    # Test with empty keys list
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default='error') == 'error'


# LLM-generated content at query #20
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with list
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([1, 1], data) == [3, 4]
    assert get_in([1], data) == [2, [3, 4]]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default=0) == 0

    # Test with no_default=True
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': 1}
    assert get_in(['b'], data, default='default') == 'default'


# LLM-generated content at query #21
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default='default') == 'default'

    # Test no_default raises exception
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError for list out of bounds
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 5], data) is None
    try:
        get_in(['a', 5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError for non-subscriptable object
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structure (assuming pyrsistent is available)
    try:
        from pyrsistent import freeze
        transaction = freeze({'name': 'Alice',
                             'purchase': {'items': ['Apple', 'Orange'],
                                         'costs': [0.50, 1.25]},
                             'credit card': '5555-1234-1234-1234'})
        assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
        assert get_in(['name'], transaction) == 'Alice'
        assert get_in(['purchase', 'total'], transaction) is None
        assert get_in(['purchase', 'total'], transaction, 0) == 0
    except ImportError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default='default') == 'default'

    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError for list out of bounds
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data) is None
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError for non-subscriptable types
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test nested mixed types
    data = {'a': [{'b': [1, 2, 3]}]}
    assert get_in(['a', 0, 'b', 1], data) == 2


# LLM-generated content at query #23
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test missing key with default
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default='default') == 'default'
    assert get_in(['a', 'c'], data) is None

    # Test missing key with no_default
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test missing index with default
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default='default') == 'default'
    assert get_in(['a', 10], data) is None

    # Test missing index with no_default
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError case (e.g., trying to index a non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default='default') == 'default'
    assert get_in(['a', 'b'], data) is None

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with nested mixed structures
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2


# LLM-generated content at query #24
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1], data) == [3, 4]
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 1, 'c'], data) is None
    assert get_in(['a', 1, 'c'], data, default='default') == 'default'

    # Test with no_default=True
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with non-existent path and no_default=False (default behavior)
    data = {'a': 1}
    assert get_in(['b'], data) is None
    assert get_in(['a', 'b'], data) is None

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'


# LLM-generated content at query #25
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([0], data) == [1, 2]
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None

    # Test no_default parameter
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([0], data, no_default=True)
        assert False, "Expected KeyError or TypeError"
    except (KeyError, TypeError):
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') is None
    assert get_in(['a', 'c'], data, default='default') == 'default'


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([1, 0], data) == 3
    assert get_in([0, 1], data) == 2
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['x'], data) is None

    # Test with no_default=True
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) is None
    assert get_in(['a', 'c'], data, default=0) == 0


# LLM-generated content at query #2
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None

    # Test with no_default=True
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) is None
    assert get_in(['a', 'c'], data, default=0) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default='not found') == 'not found'

    # Test default value when index out of range
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default='out of range') == 'out of range'

    # Test no_default raises KeyError
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test no_default raises IndexError
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with None default
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structures (assuming pyrsistent is available)
    try:
        from pyrsistent import freeze
        transaction = freeze({'name': 'Alice',
                             'purchase': {'items': ['Apple', 'Orange'],
                                         'costs': [0.50, 1.25]},
                             'credit card': '5555-1234-1234-1234'})
        assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
        assert get_in(['name'], transaction) == 'Alice'
        assert get_in(['purchase', 'total'], transaction) is None
        assert get_in(['purchase', 'total'], transaction, 0) == 0
    except ImportError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default='not found') == 'not found'

    # Test default value when index out of range
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default='out of range') == 'out of range'

    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test no_default raises IndexError
    data = [1, 2, 3]
    try:
        get_in([10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with None default
    data = {'a': 1}
    assert get_in(['b'], data) is None

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with mixed types
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2


# LLM-generated content at query #5
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1, 'c'], data) is None

    # Test with no_default=True
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with default value
    assert get_in(['x', 'y'], data, default='not_found') == 'not_found'

    # Test with empty keys list
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 'string'}
    assert get_in(['a', 0], data) is None
    assert get_in(['a', 0], data, default='default') == 'default'


# LLM-generated content at query #6
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}
    assert get_in(['x'], data) is None
    assert get_in(['a', 'x'], data) is None
    assert get_in(['a', 'b', 'x'], data) is None

    # Test with list
    data = [1, [2, [3, 4]]]
    assert get_in([1], data) == [2, [3, 4]]
    assert get_in([1, 1], data) == [3, 4]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([1, 2], data) is None

    # Test with default value
    assert get_in(['x'], {}, default=0) == 0
    assert get_in(['a', 'x'], {'a': 1}, default=[]) == []
    assert get_in([1], [1, 2], default='missing') == 2
    assert get_in([2], [1, 2], default='missing') == 'missing'

    # Test with no_default=True
    try:
        get_in(['x'], {}, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([2], [1, 2], no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None

    # Test with empty keys list
    assert get_in([], data) == data
    assert get_in([], {}, default=0) == {}
    assert get_in([], [1, 2, 3]) == [1, 2, 3]

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) == None
    assert get_in(['a', 'c'], data, default=0) == 0

    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError with list
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError when trying to index non-subscriptable
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structure (assuming pyrsistent is available)
    try:
        from pyrsistent import freeze
        transaction = freeze({'name': 'Alice',
                             'purchase': {'items': ['Apple', 'Orange'],
                                         'costs': [0.50, 1.25]},
                             'credit card': '5555-1234-1234-1234'})
        assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
        assert get_in(['name'], transaction) == 'Alice'
        assert get_in(['purchase', 'total'], transaction) is None
        assert get_in(['purchase', 'total'], transaction, 0) == 0
    except ImportError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with nested list
    data = [[[1, 2], 3], 4]
    assert get_in([0, 0, 1], data) == 2
    assert get_in([0, 1], data) == 3
    assert get_in([1], data) == 4
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with no_default=True
    data = {'a': 1}
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #9
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1], data) == [3, 4]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default=0) == 0

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #10
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=10) == 10

    # Test default value when index out of range
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 5], data, default=10) == 10

    # Test no_default raises KeyError
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test no_default raises IndexError
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with nested lists and dictionaries
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2

    # Test with persistent data structures (assuming pyrsistent is available)
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0

    # Test with TypeError (non-subscriptable object)
    data = {'a': 'string'}
    assert get_in(['a', 'b'], data, default=10) == 10


# LLM-generated content at query #11
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default='default') == 'default'

    # Test no_default raises KeyError
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError for list out of bounds
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data) is None
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError for non-subscriptable
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structures (assuming pyrsistent is available)
    try:
        from pyrsistent import freeze
        transaction = freeze({'name': 'Alice',
                            'purchase': {'items': ['Apple', 'Orange'],
                                        'costs': [0.50, 1.25]},
                            'credit card': '5555-1234-1234-1234'})
        assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
        assert get_in(['name'], transaction) == 'Alice'
        assert get_in(['purchase', 'total'], transaction) is None
        assert get_in(['purchase', 'total'], transaction, 0) == 0
    except ImportError:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default='default') == 'default'

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with non-existent nested keys
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default='default') == 'default'

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'


# LLM-generated content at query #13
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=42) == 42

    # Test default value when index out of range
    data = {'a': [1, 2]}
    assert get_in(['a', 5], data, default=42) == 42

    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test no_default raises IndexError
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with nested lists and dictionaries
    data = {'a': [{'b': 1}, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=42) == 42


# LLM-generated content at query #14
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1, 'x'], data) is None

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([0], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') is None
    assert get_in(['a', 'x'], data, default='default') == 'default'


# LLM-generated content at query #15
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x', 'y'], data) is None
    assert get_in(['x', 'y'], data, default='default') == 'default'

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2, 0], data) is None
    assert get_in([2, 0], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1, 'c'], data) is None

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'


# LLM-generated content at query #16
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with default value
    assert get_in(['b'], data, default='default') == 'default'

    # Test with empty keys list
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'


# LLM-generated content at query #17
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'

    # Test with nested list
    data = [[[1, 2], 3], 4]
    assert get_in([0, 0, 1], data) == 2
    assert get_in([1], data) == 4
    assert get_in([2], data) is None

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1, 'c'], data) is None

    # Test with no_default=True
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with IndexError
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with TypeError (non-subscriptable object)
    data = 123
    try:
        get_in([0], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with single key
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    assert get_in(['b'], data) is None


# LLM-generated content at query #18
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1, 'c'], data) is None

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with persistent data structure (assuming pyrsistent is available)
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0

    # Test with TypeError (non-subscriptable object)
    data = {'a': 'string'}
    assert get_in(['a', 0], data) is None
    assert get_in(['a', 0], data, default='default') == 'default'


# LLM-generated content at query #19
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key doesn't exist
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default='default') == 'default'

    # Test default value when key doesn't exist (no default provided)
    assert get_in(['a', 'c'], data) is None

    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test nested list access
    data = {'a': [[1, 2], [3, 4]]}
    assert get_in(['a', 1, 0], data) == 3

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test TypeError handling (e.g., trying to index a non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default='error') == 'error'


# LLM-generated content at query #20
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=42) == 42
    assert get_in(['a', 'c'], data) is None

    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError with list
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError with non-subscriptable
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structure (if available)
    try:
        from pyrsistent import freeze
        data = freeze({'a': {'b': 2}})
        assert get_in(['a', 'b'], data) == 2
    except ImportError:
        pass


# LLM-generated content at query #21
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default='default') == 'default'
    assert get_in(['a', 'c'], data) is None

    # Test no_default raises exception
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError with list
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test nested structure with mixed types
    data = {'a': [{'b': [1, 2, 3]}, {'c': 4}]}
    assert get_in(['a', 0, 'b', 1], data) == 2
    assert get_in(['a', 1, 'c'], data) == 4

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test TypeError handling (non-subscriptable)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default='default') == 'default'


# LLM-generated content at query #22
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default='default') == 'default'

    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError for list out of bounds
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 5], data) is None
    assert get_in(['a', 5], data, default='default') == 'default'

    # Test TypeError for non-subscriptable type
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test nested mixed types
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 0, 'b'], data) == 1
    assert get_in(['a', 1, 'c'], data) == 2

    # Test with persistent data structures (assuming pyrsistent is available)
    try:
        from pyrsistent import freeze
        transaction = freeze({'name': 'Alice',
                             'purchase': {'items': ['Apple', 'Orange'],
                                         'costs': [0.50, 1.25]},
                             'credit card': '5555-1234-1234-1234'})
        assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
        assert get_in(['name'], transaction) == 'Alice'
        assert get_in(['purchase', 'total'], transaction) is None
        assert get_in(['purchase', 'total'], transaction, 0) == 0
    except ImportError:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default='not found') == 'not found'
    assert get_in(['a', 'c'], data) is None

    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError with list
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test nested default
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data, default=0) == 0

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test TypeError with non-subscriptable
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default='error') == 'error'

    # Test multiple levels with mixed types
    data = {'a': [{'b': (1, 2, 3)}]}
    assert get_in(['a', 0, 'b', 1], data) == 2


# LLM-generated content at query #24
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with nested list
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 1, 1], data) == 4
    assert get_in([1, 0], data) == [5, 6]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [{'b': [1, 2, 3]}, {'c': [4, 5, 6]}]}
    assert get_in(['a', 0, 'b', 1], data) == 2
    assert get_in(['a', 1, 'c', 0], data) == 4
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default=0) == 0

    # Test with no_default=True
    data = {'a': {'b': 1}}
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in(['a', 'x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=0) == 0
    assert get_in(['x', 'y'], data, default=0) == 0

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #25
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1], data) == [3, 4]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default=0) == 0

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = 123
    assert get_in(['a'], data) is None
    assert get_in(['a'], data, default=0) == 0
    try:
        get_in(['a'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


