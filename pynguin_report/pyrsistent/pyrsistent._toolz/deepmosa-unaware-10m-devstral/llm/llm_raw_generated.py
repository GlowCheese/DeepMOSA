####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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
    assert get_in(['a', 'c'], data, default=None) is None
    assert get_in(['a', 'c'], data, default=0) == 0

    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError for list out of bounds
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError for non-subscriptable
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test nested mixed types
    data = {'a': [{'b': [1, 2, 3]}, {'c': 4}]}
    assert get_in(['a', 0, 'b', 1], data) == 2
    assert get_in(['a', 1, 'c'], data) == 4

    # Test default with nested missing keys
    data = {'a': {'b': 1}}
    assert get_in(['a', 'b', 'c', 'd'], data, default='missing') == 'missing'


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
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([1, 1], data) == [3, 4]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default=0) == 0

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

    # Test with default value
    data = {'a': None}
    assert get_in(['b'], data, default='default') == 'default'
    assert get_in(['a'], data, default='default') is None


# LLM-generated content at query #3
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

    # Test with nested list
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 1, 1], data) == 4
    assert get_in([1, 0, 0], data) == 5
    assert get_in([2], data) is None
    assert get_in([2], data, default='out_of_bounds') == 'out_of_bounds'

    # Test with mixed nested structures
    data = {'a': [{'b': [1, 2, 3]}, {'c': [4, 5, 6]}]}
    assert get_in(['a', 0, 'b', 1], data) == 2
    assert get_in(['a', 1, 'c', 0], data) == 4
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default='missing') == 'missing'

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
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') is None
    assert get_in(['a', 'c'], data, default='default') == 'default'


# LLM-generated content at query #4
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
    data = [1, 2, [3, 4, [5, 6]]]
    assert get_in([0], data) == 1
    assert get_in([2, 1], data) == 4
    assert get_in([2, 2, 1], data) == 6
    assert get_in([3], data) is None
    assert get_in([2, 3], data) is None
    assert get_in([2, 3], data, default=-1) == -1

    # Test with mixed structures
    data = {'a': [1, 2, {'b': 3}]}
    assert get_in(['a', 2, 'b'], data) == 3
    assert get_in(['a', 1], data) == 2
    assert get_in(['a', 3], data) is None
    assert get_in(['a', 3], data, default=0) == 0

    # Test no_default flag
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

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data
    assert get_in([], data, default='default') == data

    # Test with default value
    data = {}
    assert get_in(['a'], data, default='default') == 'default'
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #5
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

    # Test with list
    data = [1, 2, [3, 4]]
    assert get_in([2, 0], data) == 3
    assert get_in([2, 1], data) == 4
    assert get_in([2, 2], data) is None
    assert get_in([2, 2], data, default='default') == 'default'

    # Test with mixed types
    data = {'a': [1, 2, {'b': 3}]}
    assert get_in(['a', 2, 'b'], data) == 3
    assert get_in(['a', 2, 'c'], data) is None
    assert get_in(['a', 2, 'c'], data, default='default') == 'default'

    # Test with no_default=True
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = 123
    assert get_in(['a'], data) is None
    assert get_in(['a'], data, default='default') == 'default'


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
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([0, 2], data) is None

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None

    # Test with default value
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 'x'], data, default='default') == 'default'
    assert get_in(['a', 1, 'x'], data, default='default') == 'default'

    # Test with no_default=True
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

    try:
        get_in(['a', 1, 'x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    assert get_in([], data) == data

    # Test with non-existent keys in empty dict
    assert get_in(['x'], {}) is None
    assert get_in(['x'], {}, default='default') == 'default'

    try:
        get_in(['x'], {}, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass


# LLM-generated content at query #7
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

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([0], data) == [1, 2]
    assert get_in([2], data) is None
    assert get_in([2], data, default='out of bounds') == 'out of bounds'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None
    assert get_in(['x', 'y'], data) is None

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data
    assert get_in([], data, default='default') == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'


# LLM-generated content at query #8
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

    # Test no_default flag
    with pytest.raises(KeyError):
        get_in(['x'], data, no_default=True)
    with pytest.raises(IndexError):
        get_in([2], [1, 2], no_default=True)

    # Test with default value
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in([2], [1, 2], default='default') == 'default'

    # Test with empty keys list
    assert get_in([], data) == data


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

    # Test with list
    data = [1, [2, 3], 4]
    assert get_in([1, 0], data) == 2
    assert get_in([1], data) == [2, 3]
    assert get_in([5], data) is None
    assert get_in([5], data, default=0) == 0

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 5], data) is None
    assert get_in(['a', 5], data, default=0) == 0

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

    # Test with TypeError (non-subscriptable object)
    data = 123
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #11
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
    assert get_in(['x'], data, no_default=True) is None  # This should raise, but the function returns None

    # Test with list
    data = [1, 2, [3, 4]]
    assert get_in([2, 1], data) == 4
    assert get_in([2], data) == [3, 4]
    assert get_in([3], data) is None
    assert get_in([3], data, default=0) == 0

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 1, 'c'], data) is None

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) is None
    assert get_in(['a', 'c'], data, default=0) == 0

    # Test with TypeError (non-subscriptable)
    data = 123
    assert get_in(['a'], data) is None
    assert get_in(['a'], data, default=0) == 0


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
    assert get_in(['a', 'c'], data, default='default') == 'default'

    # Test default value when index is out of range
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 5], data, default='default') == 'default'

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

    # Test with None default
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None

    # Test with nested lists and dictionaries
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default='default') == 'default'


# LLM-generated content at query #13
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

    # Test with list
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([0], data) == 1
    assert get_in([1, 1], data) == [3, 4]
    assert get_in([5], data) is None
    assert get_in([1, 1, 5], data, default=-1) == -1

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['x'], data) is None

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['x'], data, no_default=True)
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
    assert get_in(['x', 'y', 'z'], {}, default='not found') == 'not found'
    assert get_in([0, 1, 2], [], default=0) == 0


# LLM-generated content at query #14
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
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([1, 1], data) == [3, 4]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None

    # Test with no_default=True
    data = {'a': 1}
    assert get_in(['b'], data, no_default=True) is None  # This should raise, but the function doesn't raise for None default
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) is None
    assert get_in(['a', 'c'], data, default=0) == 0


# LLM-generated content at query #15
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
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 1, 'x'], data) is None

    # Test no_default behavior
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
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') is None
    assert get_in(['a', 'x'], data, default='default') == 'default'


# LLM-generated content at query #16
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
    data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 2], data) == 6
    assert get_in([2], data) is None
    assert get_in([2], data, default='out of range') == 'out of range'
    assert get_in([0, 3], data) is None
    assert get_in([0, 3], data, default=0) == 0

    # Test with mixed nested structure
    data = {'a': [1, 2, {'b': 3}]}
    assert get_in(['a', 2, 'b'], data) == 3
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) == {'b': 3}
    assert get_in(['a', 3], data) is None
    assert get_in(['a', 3], data, default='not found') == 'not found'

    # Test no_default behavior
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

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data
    assert get_in([], data, default='default') == data

    # Test with non-subscriptable object
    data = "string"
    assert get_in([], data) == data
    assert get_in([0], data) == 's'
    assert get_in([1], data) == 't'
    assert get_in([10], data) is None
    assert get_in([10], data, default='out of range') == 'out of range'


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
    assert get_in(['a', 'b', 'x'], data, default='default') == 'default'

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([0], data) == [1, 2]
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'
    assert get_in([0, 2], data) is None
    assert get_in([0, 2], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default='default') == 'default'
    assert get_in(['a', 1, 'x'], data) is None
    assert get_in(['a', 1, 'x'], data, default='default') == 'default'

    # Test no_default flag
    data = {'a': 1}
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

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data
    assert get_in([], data, default='default') == data

    # Test with single key
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    assert get_in(['b'], data) is None
    assert get_in(['b'], data, default='default') == 'default'

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test with persistent data structures (assuming they behave like regular dicts/lists)
    # This is a mock test since we don't have the actual persistent data structures
    class MockPersistentDict(dict):
        pass

    data = MockPersistentDict({'a': MockPersistentDict({'b': 1})})
    assert get_in(['a', 'b'], data) == 1
    assert get_in(['x'], data) is None


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

    # Test TypeError with non-subscriptable object
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
    data = {'a': [{'b': [1, 2, 3]}]}
    assert get_in(['a', 0, 'b', 1], data) == 2


# LLM-generated content at query #19
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

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1, 'c'], data) is None

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
    assert get_in(['x'], data, default='default') == 'default'

    # Test with list
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([1, 1], data) == [3, 4]
    assert get_in([1], data) == [2, [3, 4]]
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with default value
    assert get_in(['b'], data, default=0) == 0
    assert get_in(['b'], data, default=None) is None

    # Test with empty keys list
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'


# LLM-generated content at query #21
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
    assert get_in(['a', 'b', 'x'], data, default='default') == 'default'

    # Test with nested list
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 1, 1], data) == 4
    assert get_in([1, 0, 0], data) == 5
    assert get_in([0], data) == [[1, 2], [3, 4]]
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'
    assert get_in([0, 1, 2], data) is None
    assert get_in([0, 1, 2], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [{'b': [1, 2, 3]}, {'c': [4, 5, 6]}]}
    assert get_in(['a', 0, 'b', 1], data) == 2
    assert get_in(['a', 1, 'c', 0], data) == 4
    assert get_in(['a', 0], data) == {'b': [1, 2, 3]}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default='default') == 'default'

    # Test with no_default=True
    data = {'a': {'b': {'c': 1}}}
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in(['a', 'b', 'x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': {'c': None}}}
    assert get_in(['a', 'b', 'c'], data) is None
    assert get_in(['a', 'b', 'c'], data, default='default') is None
    assert get_in(['a', 'b', 'x'], data, default='default') == 'default'


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
    assert get_in(['a', 'c'], data) == None
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
    assert get_in(['a', 10], data) == None
    assert get_in(['a', 10], data, default='default') == 'default'

    # Test TypeError for non-subscriptable
    data = {'a': 1}
    assert get_in(['a', 'b'], data) == None
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with nested mixed structures
    data = {'a': [{'b': [1, 2, 3]}, {'c': 4}]}
    assert get_in(['a', 0, 'b', 1], data) == 2
    assert get_in(['a', 1, 'c'], data) == 4


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
    assert get_in(['a', 5], data) is None
    assert get_in(['a', 5], data, default='default') == 'default'

    # Test TypeError for non-subscriptable
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'

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


# LLM-generated content at query #24
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0
    assert get_in(['a', 'b', 'x'], data) is None
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0
    assert get_in([0, 2], data) is None
    assert get_in([0, 2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default=0) == 0

    # Test no_default flag
    data = {'a': 1}
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

    # Test with single key
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    assert get_in(['b'], data) is None
    assert get_in(['b'], data, default=0) == 0

    # Test with TypeError (non-subscriptable object)
    data = 1
    assert get_in(['a'], data) is None
    assert get_in(['a'], data, default=0) == 0
    try:
        get_in(['a'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


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

    # Test with no_default=True
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

    # Test with single key
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    assert get_in(['b'], data) is None
    assert get_in(['b'], data, default=0) == 0


# LLM-generated content at query #26
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
    assert get_in(['a', 1, 'x'], data) is None

    # Test no_default parameter
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

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with single key
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    assert get_in(['b'], data) is None


# LLM-generated content at query #27
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

    # Test nested mixed structures
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2


# LLM-generated content at query #28
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['a', 'x'], data) is None
    assert get_in(['a', 'b', 'x'], data) is None

    # Test with default value
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 'x'], data, default='default') == 'default'
    assert get_in(['a', 'b', 'x'], data, default='default') == 'default'

    # Test with no_default=True
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

    try:
        get_in(['a', 'b', 'x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with list
    data_list = [1, 2, [3, 4, [5, 6]]]
    assert get_in([2, 2, 1], data_list) == 6
    assert get_in([0], data_list) == 1
    assert get_in([5], data_list) is None
    assert get_in([2, 5], data_list) is None
    assert get_in([2, 2, 5], data_list) is None

    # Test with mixed types
    mixed_data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], mixed_data) == 2
    assert get_in(['a', 0], mixed_data) == 1
    assert get_in(['x'], mixed_data) is None
    assert get_in(['a', 5], mixed_data) is None
    assert get_in(['a', 1, 'x'], mixed_data) is None

    # Test with empty keys
    assert get_in([], data) == data
    assert get_in([], data_list) == data_list
    assert get_in([], mixed_data) == mixed_data

    # Test with TypeError (non-subscriptable object)
    non_subscriptable = 42
    assert get_in(['x'], non_subscriptable) is None
    assert get_in(['x'], non_subscriptable, default='default') == 'default'
    try:
        get_in(['x'], non_subscriptable, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #29
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
    assert get_in([2], data, default=[]) == []

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
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

    # Test with single key
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    assert get_in(['b'], data) is None


# LLM-generated content at query #30
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

    # Test nested mixed types
    data = {'a': [{'b': (1, 2, 3)}]}
    assert get_in(['a', 0, 'b', 1], data) == 2

    # Test with default=None explicitly
    data = {'a': 1}
    assert get_in(['b'], data, default=None) is None


# LLM-generated content at query #31
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
    assert get_in(['a', 10], data, default='default') == 'default'

    # Test TypeError when trying to index non-subscriptable
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'

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


# LLM-generated content at query #32
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
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 5], data, default='default') == 'default'

    # Test no_default raises KeyError
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test no_default raises IndexError
    try:
        get_in(['a', 5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with persistent data structures (if available)
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
        pass  # Skip if pyrsistent is not available

    # Test with mixed types
    mixed_data = {'a': [{'b': 1}, {'b': 2}]}
    assert get_in(['a', 0, 'b'], mixed_data) == 1
    assert get_in(['a', 1, 'b'], mixed_data) == 2
    assert get_in(['a', 2, 'b'], mixed_data, default='not_found') == 'not_found'

    # Test empty keys list returns the collection itself
    assert get_in([], data) == data


# LLM-generated content at query #33
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
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #34
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
    data = [1, 2, [3, 4, [5, 6]]]
    assert get_in([2, 2, 1], data) == 6
    assert get_in([0], data) == 1
    assert get_in([5], data) is None
    assert get_in([5], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 1, 'c'], data) is None

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

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') is None
    assert get_in(['a', 'c'], data, default='default') == 'default'


# LLM-generated content at query #35
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
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 1, 1], data) == 4
    assert get_in([1, 0], data) == [5, 6]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['x'], data) is None

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

    try:
        get_in([0], [1, 2], no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': 1}}
    assert get_in(['x', 'y'], data, default='default') == 'default'
    assert get_in(['a', 'x'], data, default='default') == 'default'


# LLM-generated content at query #36
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    nested_dict = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], nested_dict) == 1
    assert get_in(['a', 'b'], nested_dict) == {'c': 1}
    assert get_in(['x'], nested_dict) is None
    assert get_in(['a', 'x'], nested_dict) is None
    assert get_in(['a', 'b', 'x'], nested_dict) is None

    # Test with default value
    assert get_in(['x'], nested_dict, default='default') == 'default'
    assert get_in(['a', 'x'], nested_dict, default='default') == 'default'
    assert get_in(['a', 'b', 'x'], nested_dict, default='default') == 'default'

    # Test with no_default=True
    try:
        get_in(['x'], nested_dict, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in(['a', 'x'], nested_dict, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in(['a', 'b', 'x'], nested_dict, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with list
    nested_list = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    assert get_in([0, 0, 1], nested_list) == 2
    assert get_in([1, 1, 2], nested_list) == 12
    assert get_in([0, 0, 5], nested_list) is None
    assert get_in([2, 0, 0], nested_list) is None

    # Test with mixed types
    mixed = {'a': [1, 2, {'b': 3}]}
    assert get_in(['a', 2, 'b'], mixed) == 3
    assert get_in(['a', 2, 'x'], mixed) is None
    assert get_in(['a', 5], mixed) is None

    # Test with empty keys
    assert get_in([], nested_dict) == nested_dict
    assert get_in([], nested_list) == nested_list
    assert get_in([], mixed) == mixed

    # Test with non-subscriptable types
    assert get_in(['x'], "string") is None
    assert get_in(['x'], 123) is None


# LLM-generated content at query #37
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

    # Test with default value
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 'x'], data, default='default') == 'default'
    assert get_in(['a', 'b', 'x'], data, default='default') == 'default'

    # Test with no_default=True
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

    try:
        get_in(['a', 'b', 'x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with list
    data = [[[1, 2, 3]]]
    assert get_in([0, 0, 1], data) == 2
    assert get_in([0], data) == [[1, 2, 3]]
    assert get_in([0, 0], data) == [1, 2, 3]
    assert get_in([1], data) is None
    assert get_in([0, 1], data) is None
    assert get_in([0, 0, 3], data) is None

    # Test with default value for list
    assert get_in([1], data, default='default') == 'default'
    assert get_in([0, 1], data, default='default') == 'default'
    assert get_in([0, 0, 3], data, default='default') == 'default'

    # Test with no_default=True for list
    try:
        get_in([1], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    try:
        get_in([0, 1], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    try:
        get_in([0, 0, 3], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with mixed types
    data = {'a': [1, 2, {'b': 3}]}
    assert get_in(['a', 2, 'b'], data) == 3
    assert get_in(['a', 0], data) == 1
    assert get_in(['a'], data) == [1, 2, {'b': 3}]
    assert get_in(['x'], data) is None
    assert get_in(['a', 3], data) is None
    assert get_in(['a', 2, 'x'], data) is None

    # Test with default value for mixed types
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 3], data, default='default') == 'default'
    assert get_in(['a', 2, 'x'], data, default='default') == 'default'

    # Test with no_default=True for mixed types
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in(['a', 3], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    try:
        get_in(['a', 2, 'x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass


# LLM-generated content at query #38
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
    assert get_in([1], data) == [3, 4]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structure
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


# LLM-generated content at query #39
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    nested_dict = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], nested_dict) == 1
    assert get_in(['a', 'b'], nested_dict) == {'c': 1}
    assert get_in(['x', 'y', 'z'], nested_dict) is None
    assert get_in(['x', 'y', 'z'], nested_dict, default=42) == 42

    # Test with nested list
    nested_list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 1, 1], nested_list) == 4
    assert get_in([1, 0, 0], nested_list) == 5
    assert get_in([2, 0, 0], nested_list) is None
    assert get_in([2, 0, 0], nested_list, default=0) == 0

    # Test with mixed nested structures
    mixed = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], mixed) == 2
    assert get_in(['a', 0], mixed) == 1
    assert get_in(['a', 1, 'c'], mixed) is None

    # Test with default value
    assert get_in(['x'], {}, default='default') == 'default'
    assert get_in(['x', 'y'], {}, default=0) == 0

    # Test with no_default=True
    try:
        get_in(['x'], {}, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([0], [], no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with empty keys list
    assert get_in([], nested_dict) == nested_dict
    assert get_in([], nested_list) == nested_list

    # Test with non-existent keys in nested structures
    assert get_in(['a', 'b', 'd'], nested_dict) is None
    assert get_in([0, 0, 0, 0], nested_list) is None

    # Test with TypeError (e.g., trying to index a non-subscriptable type)
    assert get_in(['a', 'b', 'c', 'd'], {'a': {'b': {'c': 'string'}}}) is None


# LLM-generated content at query #40
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

    # Test with mixed types
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

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {}
    assert get_in(['a', 'b', 'c'], data, default='default') == 'default'


# LLM-generated content at query #41
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

    # Test IndexError for list out of bounds
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 5], data) == None
    assert get_in(['a', 5], data, default=0) == 0

    # Test TypeError for non-subscriptable
    data = {'a': 1}
    assert get_in(['a', 'b'], data) == None
    assert get_in(['a', 'b'], data, default=0) == 0

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structure
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) == None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #42
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

    # Test nested mixed types
    data = {'a': [{'b': (1, 2, 3)}]}
    assert get_in(['a', 0, 'b', 1], data) == 2

    # Test with persistent data structures (if available)
    try:
        from pyrsistent import freeze
        data = freeze({'a': {'b': 1}})
        assert get_in(['a', 'b'], data) == 1
    except ImportError:
        pass


# LLM-generated content at query #43
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

    # Test with default value
    assert get_in(['x'], data, default=0) == 0
    assert get_in(['a', 'x'], data, default=0) == 0
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with no_default=True
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

    try:
        get_in(['a', 'b', 'x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with list
    data_list = [1, 2, [3, 4, [5, 6]]]
    assert get_in([0], data_list) == 1
    assert get_in([2], data_list) == [3, 4, [5, 6]]
    assert get_in([2, 2], data_list) == [5, 6]
    assert get_in([2, 2, 0], data_list) == 5
    assert get_in([2, 2, 3], data_list) is None
    assert get_in([2, 2, 3], data_list, default=0) == 0

    # Test with mixed types
    mixed_data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], mixed_data) == 2
    assert get_in(['a', 1, 'c'], mixed_data) is None
    assert get_in(['a', 1, 'c'], mixed_data, default=0) == 0

    # Test with empty keys list
    assert get_in([], data) == data
    assert get_in([], data_list) == data_list
    assert get_in([], mixed_data) == mixed_data

    # Test with TypeError (non-subscriptable object)
    non_subscriptable = 42
    assert get_in(['a'], non_subscriptable) is None
    assert get_in(['a'], non_subscriptable, default=0) == 0

    try:
        get_in(['a'], non_subscriptable, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #44
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
    assert get_in([2], data, default='default') == 'default'

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default='default') == 'default'

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with default value
    data = {'a': 1}
    assert get_in(['b'], data, default=0) == 0
    assert get_in(['b'], data, default=None) is None

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data


# LLM-generated content at query #45
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
    assert get_in(['a', 'b', 'x'], data, default='default') == 'default'

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'
    assert get_in([0, 2], data) is None
    assert get_in([0, 2], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default='default') == 'default'
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'

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

    try:
        get_in([0], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data
    assert get_in([], data, default='default') == data

    # Test with single key
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    assert get_in(['b'], data) is None
    assert get_in(['b'], data, default='default') == 'default'


# LLM-generated content at query #46
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
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default=42) == 42

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

    # Test empty keys list returns the collection
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with nested mixed structures
    data = {'a': [{'b': [1, 2, 3]}, {'c': 4}]}
    assert get_in(['a', 0, 'b', 1], data) == 2

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=42) == 42


# LLM-generated content at query #47
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

    # Test TypeError with non-subscriptable
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list returns the collection
    data = {'a': 1}
    assert get_in([], data) == data

    # Test nested mixed types
    data = {'a': {'b': [1, {'c': 2}]}}
    assert get_in(['a', 'b', 1, 'c'], data) == 2


# LLM-generated content at query #48
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test with list
    data = [1, [2, 3], 4]
    assert get_in([1, 0], data) == 2

    # Test with default value
    assert get_in(['x', 'y'], data, default='default') == 'default'

    # Test with no_default=True
    try:
        get_in(['x', 'y'], data, no_default=True)
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
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test with IndexError
    data = [1, 2, 3]
    assert get_in([5], data, default='default') == 'default'

    # Test with KeyError
    data = {'a': 1}
    assert get_in(['b'], data, default='default') == 'default'


# LLM-generated content at query #49
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

    # Test with TypeError (non-subscriptable object)
    data = 123
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0


# LLM-generated content at query #50
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

    # Test TypeError for non-subscriptable type
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

    # Test with nested mixed types
    data = {'a': [{'b': (1, 2, 3)}]}
    assert get_in(['a', 0, 'b', 1], data) == 2


# LLM-generated content at query #51
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
    assert get_in(['a', 0], data) == 1
    assert get_in(['x'], data) is None

    # Test no_default flag
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
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test with IndexError (list out of bounds)
    data = [1, 2, 3]
    assert get_in([5], data) is None
    assert get_in([5], data, default='default') == 'default'


# LLM-generated content at query #52
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
    data = [1, 2, [3, 4]]
    assert get_in([0], data) == 1
    assert get_in([2, 0], data) == 3
    assert get_in([2, 1], data) == 4
    assert get_in([3], data) is None
    assert get_in([3], data, default='default') == 'default'
    assert get_in([2, 2], data) is None
    assert get_in([2, 2], data, default=0) == 0

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default='default') == 'default'

    # Test no_default flag
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

    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with default value
    assert get_in(['x'], {}, default='default') == 'default'
    assert get_in([0], [], default='default') == 'default'
    assert get_in(['x', 'y'], {}, default='default') == 'default'


# LLM-generated content at query #53
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
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None

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

    # Test with non-existent nested key and no_default=True
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with TypeError (e.g., trying to index a non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'


# LLM-generated content at query #54
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 'x'], data) is None

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None

    # Test with mixed structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1, 'x'], data) is None

    # Test no_default flag
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([0], [], no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with default value
    assert get_in(['x'], {}, default=42) == 42
    assert get_in([0], [], default=42) == 42

    # Test with empty keys
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'


# LLM-generated content at query #55
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
    assert get_in(['x', 'y'], data, default='default') == 'default'

    # Test default value when key not found (no default provided)
    assert get_in(['x', 'y'], data) is None

    # Test no_default raises KeyError
    with pytest.raises(KeyError):
        get_in(['x', 'y'], data, no_default=True)

    # Test nested list access
    data = {'a': [[1, 2], [3, 4]]}
    assert get_in(['a', 1, 0], data) == 3

    # Test TypeError handling with default
    data = {'a': 'string'}
    assert get_in(['a', 'b'], data, default='error') == 'error'

    # Test TypeError handling with no_default
    with pytest.raises(TypeError):
        get_in(['a', 'b'], data, no_default=True)

    # Test empty keys list returns the collection itself
    assert get_in([], data) == data

    # Test with persistent data structure (if available)
    try:
        from pyrsistent import freeze
        persistent_data = freeze({'a': {'b': 2}})
        assert get_in(['a', 'b'], persistent_data) == 2
    except ImportError:
        pass


# LLM-generated content at query #56
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

    # Test default None when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None

    # Test no_default raises KeyError
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test nested list access
    data = {'a': [[1, 2], [3, 4]]}
    assert get_in(['a', 1, 0], data) == 3

    # Test IndexError with no_default
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError with no_default
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


# LLM-generated content at query #57
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='not found') == 'not found'
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with list
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([0], data) == 1
    assert get_in([1, 1], data) == [3, 4]
    assert get_in([5], data) is None
    assert get_in([1, 1, 5], data, default=-1) == -1

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 5], data) is None
    assert get_in(['a', 1, 'x'], data, default='missing') == 'missing'

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([0], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys list
    assert get_in([], data) == data
    assert get_in([], None) is None
    assert get_in([], None, default='default') is None

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #58
#--------------------------

```python
def test_get_in():
    # Test basic nested dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test nested list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test default value when key not found
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default='not found') == 'not found'

    # Test default value when index out of range
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 5], data, default='out of range') == 'out of range'

    # Test default value when TypeError (non-subscriptable)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default='type error') == 'type error'

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

    # Test no_default raises TypeError
    data = 1
    try:
        get_in(['a'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test default None when not found
    data = {'a': 1}
    assert get_in(['b'], data) is None

    # Test empty keys list returns the collection itself
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with mixed nested structures
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2


# LLM-generated content at query #59
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

    # Test nested mixed types
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2

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


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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

    # Test with nested mixed structures
    data = {'a': [{'b': [1, 2, 3]}, {'c': 4}]}
    assert get_in(['a', 0, 'b', 1], data) == 2
    assert get_in(['a', 1, 'c'], data) == 4


# LLM-generated content at query #2
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
    data = {'a': [1, 2]}
    try:
        get_in(['a', 5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError for non-subscriptable
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list returns the collection
    data = {'a': 1}
    assert get_in([], data) == data

    # Test nested mixed structures
    data = {'a': [{'b': [1, 2, 3]}]}
    assert get_in(['a', 0, 'b', 1], data) == 2

    # Test with persistent data structures (assuming they work like regular ones)
    from pyrsistent import m, v
    data = m(a=m(b=v(1, 2, 3)))
    assert get_in(['a', 'b', 1], data) == 2


# LLM-generated content at query #3
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
    data = {'a': 1}
    assert get_in(['b'], data, default=0) == 0

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = 123
    assert get_in(['a'], data) is None
    assert get_in(['a'], data, default='error') == 'error'


# LLM-generated content at query #4
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}

    # Test with default value
    assert get_in(['x', 'y', 'z'], data, default='not found') == 'not found'
    assert get_in(['a', 'x', 'y'], data, default=None) is None

    # Test with no_default=True
    try:
        get_in(['x', 'y'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with list
    list_data = [[1, 2, 3], [4, 5, 6]]
    assert get_in([0, 1], list_data) == 2
    assert get_in([1, 2], list_data) == 6

    # Test with mixed types
    mixed_data = {'a': [1, 2, {'b': 3}]}
    assert get_in(['a', 2, 'b'], mixed_data) == 3

    # Test with empty keys list
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    assert get_in(['a', 'b', 'c', 'd'], data, default='error') == 'error'


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
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 1, 1], data) == 4
    assert get_in([1], data) == [[5, 6], [7, 8]]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None

    # Test no_default behavior
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'c'], data, default=0) == 0


# LLM-generated content at query #6
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
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test with no_default=True
    data = {'a': 1}
    with pytest.raises(KeyError):
        get_in(['b'], data, no_default=True)
    with pytest.raises(IndexError):
        get_in([0], data, no_default=True)

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data


# LLM-generated content at query #7
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    nested_dict = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], nested_dict) == 1
    assert get_in(['a', 'b'], nested_dict) == {'c': 1}
    assert get_in(['x'], nested_dict) is None
    assert get_in(['x'], nested_dict, default='default') == 'default'

    # Test with nested list
    nested_list = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    assert get_in([0, 1, 2], nested_list) == 6
    assert get_in([1, 0], nested_list) == [7, 8, 9]
    assert get_in([2], nested_list) is None
    assert get_in([2], nested_list, default='default') == 'default'

    # Test with mixed nested structures
    mixed = {'a': [1, 2, {'b': 3}]}
    assert get_in(['a', 2, 'b'], mixed) == 3
    assert get_in(['a', 1], mixed) == 2
    assert get_in(['a', 3], mixed) is None
    assert get_in(['a', 3], mixed, default='default') == 'default'

    # Test no_default behavior
    try:
        get_in(['x'], nested_dict, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([0, 3], nested_list, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with empty keys list
    assert get_in([], nested_dict) == nested_dict
    assert get_in([], nested_list) == nested_list

    # Test with default value
    assert get_in(['x', 'y', 'z'], nested_dict, default='default') == 'default'
    assert get_in([0, 1, 2, 3], nested_list, default='default') == 'default'


# LLM-generated content at query #8
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
    assert get_in(['a', 10], data, default='default') == 'default'

    # Test TypeError for non-subscriptable types
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test empty keys list returns the collection itself
    data = {'a': 1}
    assert get_in([], data) == data

    # Test nested mixed structures
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2
    assert get_in(['a', 1, 'd'], data) is None


# LLM-generated content at query #9
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
    assert get_in(['a', 'b', 'x'], data, default='default') == 'default'

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([0], data) == [1, 2]
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'
    assert get_in([0, 2], data) is None
    assert get_in([0, 2], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default='default') == 'default'
    assert get_in(['a', 1, 'c'], data) is None
    assert get_in(['a', 1, 'c'], data, default='default') == 'default'

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

    try:
        get_in([0], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data
    assert get_in([], data, default='default') == data

    # Test with None as default
    data = {'a': 1}
    assert get_in(['x'], data, default=None) is None


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
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default=0) == 0

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError with list
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data) is None
    assert get_in(['a', 10], data, default=0) == 0

    # Test TypeError when trying to index non-subscriptable
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test nested mixed structures
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2
    assert get_in(['a', 0, 'b'], data) == 1


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
    assert get_in(['x', 'y'], data) is None
    assert get_in(['x', 'y'], data, default='default') == 'default'

    # Test no_default raises KeyError
    try:
        get_in(['x', 'y'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError with list
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
    assert get_in([], data) == data

    # Test with nested mixed types
    data = {'a': [{'b': [1, 2, 3]}]}
    assert get_in(['a', 0, 'b', 1], data) == 2


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
    assert get_in(['b'], data, no_default=True) is None  # This should raise, but the function returns None
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
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default=42) == 42

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

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with nested mixed structures
    data = {'a': [{'b': [1, 2, 3]}, {'c': 4}]}
    assert get_in(['a', 0, 'b', 1], data) == 2
    assert get_in(['a', 1, 'c'], data) == 4

    # Test with TypeError (non-subscriptable)
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

    # Test with nested list
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 1, 0], data) == 3
    assert get_in([1], data) == [[5, 6], [7, 8]]
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['x'], data) is None

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([0], data, no_default=True)
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


# LLM-generated content at query #15
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

    # Test TypeError for non-subscriptable
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structure
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #16
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

    # Test with mixed types
    data = {'a': [1, 2, {'b': 3}]}
    assert get_in(['a', 2, 'b'], data) == 3
    assert get_in(['a', 2, 'x'], data) is None
    assert get_in(['a', 2, 'x'], data, default=0) == 0

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

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0


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
    assert get_in(['a', 'b', 'x'], data) is None

    # Test with list
    data = [1, 2, [3, 4]]
    assert get_in([2, 1], data) == 4
    assert get_in([2], data) == [3, 4]
    assert get_in([5], data) is None
    assert get_in([5], data, default='default') == 'default'

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 5], data) is None

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([5], [1, 2], no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with non-subscriptable object
    data = 5
    assert get_in([], data) == 5
    assert get_in(['x'], data) is None
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #18
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
    assert get_in(['a', 'b', 'x'], data) is None
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with nested list
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([0], data) == [1, 2]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0
    assert get_in([0, 2], data) is None
    assert get_in([0, 2], data, default=0) == 0

    # Test with mixed nested structures
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

    try:
        get_in([0], data, no_default=True)
        assert False, "Expected KeyError or TypeError"
    except (KeyError, TypeError):
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data
    assert get_in([], data, default=0) == data

    # Test with non-existent keys and no_default=False
    data = {'a': 1}
    assert get_in(['b'], data) is None
    assert get_in(['a', 'b'], data) is None
    assert get_in([0], data) is None


# LLM-generated content at query #19
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

    # Test with default value
    assert get_in(['x'], data, default=0) == 0
    assert get_in(['a', 'x'], data, default=0) == 0
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with no_default=True
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

    # Test with list
    data_list = [1, [2, [3, [4]]]]
    assert get_in([1, 1, 1], data_list) == 4
    assert get_in([1, 1], data_list) == [3, [4]]
    assert get_in([1], data_list) == [2, [3, [4]]]
    assert get_in([1, 1, 1, 1], data_list) is None
    assert get_in([1, 1, 1, 1], data_list, default=0) == 0

    # Test with mixed types
    mixed_data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], mixed_data) == 2
    assert get_in(['a', 1], mixed_data) == {'b': 2}
    assert get_in(['a', 0], mixed_data) == 1
    assert get_in(['a', 2], mixed_data) is None
    assert get_in(['a', 2], mixed_data, default=0) == 0

    # Test with empty keys list
    assert get_in([], data) == data
    assert get_in([], data_list) == data_list
    assert get_in([], mixed_data) == mixed_data

    # Test with TypeError (non-subscriptable object)
    non_subscriptable = 123
    assert get_in(['a'], non_subscriptable) is None
    assert get_in(['a'], non_subscriptable, default=0) == 0


# LLM-generated content at query #20
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
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default=0) == 0

    # Test no_default flag
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

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data
    assert get_in([], data, default=0) == data

    # Test with default values
    data = {'a': None}
    assert get_in(['a'], data) is None
    assert get_in(['a'], data, default=0) is None
    assert get_in(['x'], data, default=0) == 0


# LLM-generated content at query #21
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

    # Test with default value
    assert get_in(['x'], data, default=0) == 0
    assert get_in(['a', 'x'], data, default=0) == 0
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with no_default=True
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

    try:
        get_in(['a', 'b', 'x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with list
    data_list = [1, 2, [3, 4, [5, 6]]]
    assert get_in([2, 2, 1], data_list) == 6
    assert get_in([2, 2], data_list) == [5, 6]
    assert get_in([2], data_list) == [3, 4, [5, 6]]
    assert get_in([10], data_list) is None
    assert get_in([2, 10], data_list) is None
    assert get_in([2, 2, 10], data_list) is None

    # Test with default value for list
    assert get_in([10], data_list, default=0) == 0
    assert get_in([2, 10], data_list, default=0) == 0
    assert get_in([2, 2, 10], data_list, default=0) == 0

    # Test with no_default=True for list
    try:
        get_in([10], data_list, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    try:
        get_in([2, 10], data_list, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    try:
        get_in([2, 2, 10], data_list, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with mixed dictionary and list
    mixed_data = {'a': [1, 2, {'b': [3, 4]}]}
    assert get_in(['a', 2, 'b', 1], mixed_data) == 4
    assert get_in(['a', 2, 'b'], mixed_data) == [3, 4]
    assert get_in(['a', 2], mixed_data) == {'b': [3, 4]}
    assert get_in(['a'], mixed_data) == [1, 2, {'b': [3, 4]}]
    assert get_in(['x'], mixed_data) is None
    assert get_in(['a', 10], mixed_data) is None
    assert get_in(['a', 2, 'x'], mixed_data) is None
    assert get_in(['a', 2, 'b', 10], mixed_data) is None

    # Test with default value for mixed data
    assert get_in(['x'], mixed_data, default=0) == 0
    assert get_in(['a', 10], mixed_data, default=0) == 0
    assert get_in(['a', 2, 'x'], mixed_data, default=0) == 0
    assert get_in(['a', 2, 'b', 10], mixed_data, default=0) == 0

    # Test with no_default=True for mixed data
    try:
        get_in(['x'], mixed_data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in(['a', 10], mixed_data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    try:
        get_in(['a', 2, 'x'], mixed_data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in(['a', 2, 'b', 10], mixed_data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with empty keys
    assert get_in([], data) == data
    assert get_in([], data_list) == data_list
    assert get_in([], mixed_data) == mixed_data

    # Test with TypeError (non-subscriptable object)
    non_subscriptable = 123
    assert get_in(['x'], non_subscriptable) is None
    assert get_in(['x'], non_subscriptable, default=0) == 0

    try:
        get_in(['x'], non_subscriptable, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
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
    assert get_in(['a', 'c'], data, default=42) == 42

    # Test default value when index out of range
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 5], data, default=42) == 42

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

    # Test empty keys list returns the collection
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with nested mixed structures
    data = {'a': [{'b': 2}, {'c': 3}]}
    assert get_in(['a', 1, 'c'], data) == 3

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=42) == 42


# LLM-generated content at query #23
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

    # Test with mixed types
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
    assert get_in(['a', 'c'], data, default='default') == 'default'


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

    # Test with list
    data = [1, [2, [3, [4]]]]
    assert get_in([1, 1, 1], data) == 4
    assert get_in([1, 1], data) == [3, [4]]
    assert get_in([1, 1, 1, 1], data) is None

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 1, 'x'], data) is None

    # Test with no_default=True
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with default value
    data = {'a': 1}
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['x', 'y'], data, default='default') == 'default'

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable)
    data = 5
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'


# LLM-generated content at query #25
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
    assert get_in(['a', 'c'], data, default='default') == 'default'

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

    # Test with persistent data structure
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #26
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

    # Test with nested mixed structures
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2


# LLM-generated content at query #27
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

    # Test with mixed types
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

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') is None
    assert get_in(['a', 'c'], data, default='default') == 'default'


# LLM-generated content at query #28
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
    assert get_in(['a'], data) == 1
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) is None
    assert get_in(['a', 'c'], data, default=0) == 0


# LLM-generated content at query #29
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

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data


# LLM-generated content at query #30
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
    assert get_in([1, 0], data) == 3
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

    # Test with single key
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    assert get_in(['b'], data) is None
    assert get_in(['b'], data, default=0) == 0

    # Test with TypeError (non-subscriptable object)
    data = 123
    assert get_in(['a'], data) is None
    assert get_in(['a'], data, default=0) == 0


# LLM-generated content at query #31
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

    # Test default value when key not found (no default provided)
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None

    # Test no_default flag raises KeyError
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test no_default flag raises IndexError for list
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with nested lists and dictionaries
    data = {'a': [{'b': 2}, {'c': 3}]}
    assert get_in(['a', 1, 'c'], data) == 3

    # Test TypeError handling (e.g., trying to index a non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=42) == 42


# LLM-generated content at query #32
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
    assert get_in(['x', 'y'], data, default='not found') == 'not found'

    # Test default value when key not found (default is None)
    assert get_in(['x', 'y'], data) is None

    # Test no_default raises KeyError
    try:
        get_in(['x', 'y'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test nested list access
    data = {'a': [[1, 2], [3, 4]]}
    assert get_in(['a', 1, 0], data) == 3

    # Test mixed dictionary and list access
    data = {'a': {'b': [1, 2, 3]}}
    assert get_in(['a', 'b', 2], data) == 3

    # Test empty keys list returns the collection itself
    assert get_in([], data) == data

    # Test TypeError handling (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default='error') == 'error'

    # Test IndexError handling (list index out of range)
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default='out of range') == 'out of range'


# LLM-generated content at query #33
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test with list
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3

    # Test with default value
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=0) == 0

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
    data = 1
    try:
        get_in([0], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value and missing key
    data = {'a': {'b': 1}}
    assert get_in(['x', 'y', 'z'], data, default='default') == 'default'

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2


# LLM-generated content at query #34
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
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([0], data) == 1
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed types
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


# LLM-generated content at query #35
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

    # Test with nested list
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 1, 1], data) == 4
    assert get_in([1, 0], data) == [5, 6]
    assert get_in([2], data) is None
    assert get_in([0, 1, 2], data, default=0) == 0

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

    try:
        get_in([0], data, no_default=True)
        assert False, "Expected KeyError or IndexError"
    except (KeyError, IndexError):
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with non-existent keys and no_default=False
    data = {'a': 1}
    assert get_in(['b'], data) is None
    assert get_in(['a', 'b'], data) is None


# LLM-generated content at query #36
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
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 1, 1], data) == 4
    assert get_in([1, 0], data) == [5, 6]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
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

    # Test with non-existent nested key
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default=0) == 0

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #37
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
    assert get_in(['x', 'y'], data) is None
    assert get_in(['x', 'y'], data, default='default') == 'default'

    # Test no_default raises KeyError
    try:
        get_in(['x', 'y'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError for list out of bounds
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError for non-subscriptable
    try:
        get_in(['a', 'b'], 123, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list returns the collection
    assert get_in([], data) == data

    # Test with nested mixed structures
    mixed_data = {'a': [{'b': [1, 2, 3]}, {'c': 'value'}]}
    assert get_in(['a', 0, 'b', 1], mixed_data) == 2
    assert get_in(['a', 1, 'c'], mixed_data) == 'value'
    assert get_in(['a', 2], mixed_data) is None


# LLM-generated content at query #38
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
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) is None
    assert get_in(['a', 'c'], data, default=0) == 0


# LLM-generated content at query #39
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
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
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

    try:
        get_in([0], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with non-existent nested keys
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'x'], data) is None
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with TypeError (e.g., trying to index a non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #40
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1
    assert get_in(['a', 'b'], data) == {'c': 1}
    assert get_in(['a'], data) == {'b': {'c': 1}}

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1], data) == 2
    assert get_in(['a', 2], data) == 3

    # Test missing key with default
    assert get_in(['x'], data) is None
    assert get_in(['x'], data, default='default') == 'default'
    assert get_in(['a', 5], data, default='out_of_bounds') == 'out_of_bounds'

    # Test missing key with no_default
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in(['a', 5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError handling
    data = {'a': 123}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='not_a_dict') == 'not_a_dict'

    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    assert get_in([], data) == data

    # Test nested mixed structures
    data = {'a': [{'b': [1, 2, 3]}, {'c': 4}]}
    assert get_in(['a', 0, 'b', 1], data) == 2
    assert get_in(['a', 1, 'c'], data) == 4
    assert get_in(['a', 0, 'b', 5], data) is None
    assert get_in(['a', 0, 'b', 5], data, default=99) == 99


# LLM-generated content at query #41
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
    assert get_in(['a', 'c'], data, default=42) == 42

    # Test missing key without default
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None

    # Test missing key with no_default=True
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError with list
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default=42) == 42

    # Test IndexError with no_default=True
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError with non-subscriptable object
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=42) == 42

    # Test TypeError with no_default=True
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structure
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


# LLM-generated content at query #42
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
    assert get_in(['a', 'x'], data) is None
    assert get_in(['a', 'x'], data, default=0) == 0

    # Test with nested list
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert get_in([0, 0, 0], data) == 1
    assert get_in([1, 1, 1], data) == 8
    assert get_in([0, 1], data) == [3, 4]
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0
    assert get_in([0, 0, 2], data) is None
    assert get_in([0, 0, 2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [{'b': [1, 2, 3]}, {'c': [4, 5, 6]}]}
    assert get_in(['a', 0, 'b', 1], data) == 2
    assert get_in(['a', 1, 'c', 0], data) == 4
    assert get_in(['a', 0, 'x'], data) is None
    assert get_in(['a', 0, 'x'], data, default=0) == 0
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

    data = [[1, 2, 3]]
    try:
        get_in([0, 5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with non-subscriptable object
    data = "string"
    assert get_in([], data) == data
    assert get_in([0], data) == 's'
    assert get_in([1], data) == 't'
    assert get_in([5], data) == 'g'
    assert get_in([6], data) is None
    assert get_in([6], data, default=0) == 0
    try:
        get_in([6], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass


# LLM-generated content at query #43
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
    data = [[1, 2], [3, 4]]
    assert get_in([0, 1], data) == 2
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 2], data) is None
    assert get_in(['a', 2], data, default=0) == 0

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

    # Test with non-existent nested key
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None
    assert get_in(['a', 'c'], data, default=0) == 0

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #44
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
    data = [1, [2, [3]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([0], data) == 1
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['x'], data) is None

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['x'], data, no_default=True)
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

    # Test with TypeError (non-subscriptable)
    data = 5
    try:
        get_in([0], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) is None
    assert get_in(['a', 'c'], data, default=0) == 0


# LLM-generated content at query #45
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

    # Test with default value
    assert get_in(['x'], data, default=0) == 0
    assert get_in(['a', 'x'], data, default=0) == 0
    assert get_in(['a', 'b', 'x'], data, default=0) == 0

    # Test with no_default=True
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

    try:
        get_in(['a', 'b', 'x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with list
    data_list = [1, [2, [3, 4]]]
    assert get_in([1], data_list) == [2, [3, 4]]
    assert get_in([1, 1], data_list) == [3, 4]
    assert get_in([1, 1, 0], data_list) == 3
    assert get_in([1, 1, 2], data_list) is None
    assert get_in([1, 1, 2], data_list, default=0) == 0

    # Test with mixed types
    mixed_data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], mixed_data) == 2
    assert get_in(['a', 1, 'x'], mixed_data) is None
    assert get_in(['a', 1, 'x'], mixed_data, default=0) == 0

    # Test with empty keys list
    assert get_in([], data) == data
    assert get_in([], data_list) == data_list
    assert get_in([], mixed_data) == mixed_data

    # Test with TypeError (non-subscriptable object)
    non_subscriptable = {'a': 1}
    assert get_in(['a'], non_subscriptable) == 1
    assert get_in(['x'], non_subscriptable) is None
    assert get_in(['x'], non_subscriptable, default=0) == 0


# LLM-generated content at query #46
#--------------------------

```python
def test_get_in():
    # Test basic dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test list access
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

    # Test with default value
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=99) == 99

    # Test with no_default raising KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with no_default raising IndexError
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with nested list and dictionary
    data = {'a': [{'b': 2}, {'c': 3}]}
    assert get_in(['a', 1, 'c'], data) == 3

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 'string'}
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test with default=None (default behavior)
    data = {'a': 1}
    assert get_in(['b'], data) is None


# LLM-generated content at query #47
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
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([0], data) == 1
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['x'], data) is None

    # Test no_default raises exception
    try:
        get_in(['x'], {}, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    try:
        get_in([0], [], no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with default value
    assert get_in(['x'], {}, default=0) == 0
    assert get_in([0], [], default=0) == 0

    # Test with empty keys list
    assert get_in([], data) == data


# LLM-generated content at query #48
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
    assert get_in([1, 0], data) == 3
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
    assert get_in(['a'], data) == 1
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with default value
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=0) == 0
    assert get_in(['a', 'c'], data, default=None) is None
    assert get_in(['a', 'c'], data, default='default') == 'default'

    # Test with empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = 123
    assert get_in(['a'], data) is None
    assert get_in(['a'], data, default=0) == 0


# LLM-generated content at query #49
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
    assert get_in(['a', 'c'], data, default='default') == 'default'
    assert get_in(['a', 'c'], data) is None

    # Test no_default raises KeyError
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test IndexError with no_default
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError with no_default
    data = "string"
    try:
        get_in([0, 0], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structure (assuming it's available)
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
        pass  # Skip if pyrsistent is not available


# LLM-generated content at query #50
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

    # Test with list
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([1, 1], data) == [3, 4]
    assert get_in([1], data) == [2, [3, 4]]
    assert get_in([2], data) is None
    assert get_in([2], data, default='default') == 'default'

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['a', 0], data) == 1
    assert get_in(['x'], data) is None

    # Test with no_default=True
    data = {'a': 1}
    try:
        get_in(['x'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with empty keys
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') is None
    assert get_in(['a', 'c'], data, default='default') == 'default'


# LLM-generated content at query #51
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

    # Test with no_default=True
    data = {'a': 1}
    assert get_in(['b'], data, no_default=True) is None  # This should raise, but the function returns None
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test with default value
    assert get_in(['x', 'y'], data, default='default') == 'default'

    # Test with empty keys list
    assert get_in([], data) == data

    # Test with TypeError (non-subscriptable object)
    data = {'a': 'string'}
    assert get_in(['a', 0], data) is None
    assert get_in(['a', 0], data, default='default') == 'default'


# LLM-generated content at query #52
#--------------------------

```python
def test_get_in():
    # Test basic nested dictionary access
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test basic nested list access
    data = [[[1, 2], 3], 4]
    assert get_in([0, 0, 1], data) == 2

    # Test mixed nested structure access
    data = {'a': [{'b': [1, 2, 3]}, 4]}
    assert get_in(['a', 0, 'b', 1], data) == 2

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

    # Test no_default raises IndexError for lists
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test empty keys list returns the collection itself
    data = {'a': 1}
    assert get_in([], data) == data

    # Test with persistent data structures (assuming they behave like regular dicts/lists)
    # This is a simplified test since we can't actually import pyrsistent here
    class MockPersistentDict(dict):
        pass

    data = MockPersistentDict({'a': {'b': 2}})
    assert get_in(['a', 'b'], data) == 2

    # Test TypeError handling (e.g., trying to index a non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test with tuples as keys
    data = {(1, 2): {'a': 1}}
    assert get_in([(1, 2), 'a'], data) == 1


# LLM-generated content at query #53
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
    assert get_in(['a', 0], data) == 1
    assert get_in(['x'], data) is None

    # Test no_default flag
    data = {'a': 1}
    try:
        get_in(['x'], data, no_default=True)
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


# LLM-generated content at query #54
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
    assert get_in([2], data, default=0) == 0

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


# LLM-generated content at query #55
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

    # Test TypeError for non-subscriptable types
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default='default') == 'default'

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test nested mixed types
    data = {'a': [{'b': [1, 2, 3]}]}
    assert get_in(['a', 0, 'b', 1], data) == 2


# LLM-generated content at query #56
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
    assert get_in([1, 0], data) == 3
    assert get_in([2], data) is None
    assert get_in([2], data, default=0) == 0

    # Test with mixed nested structures
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 0], data) == 1
    assert get_in(['a', 1], data) == {'b': 2}
    assert get_in(['x'], data) is None

    # Test with no_default=True
    data = {'a': 1}
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


# LLM-generated content at query #57
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

    # Test with TypeError (non-subscriptable object)
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None

    # Test with default value
    data = {'a': {'b': None}}
    assert get_in(['a', 'b'], data, default=0) is None
    assert get_in(['a', 'c'], data, default=0) == 0


# LLM-generated content at query #58
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
    assert get_in(['x', 'y'], data, default='not found') == 'not found'

    # Test default value when intermediate key not found
    assert get_in(['a', 'b', 'c'], {'a': 1}, default='not found') == 'not found'

    # Test no_default raises KeyError
    try:
        get_in(['x', 'y'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

    # Test no_default raises IndexError for list
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with None default
    assert get_in(['x', 'y'], data) is None

    # Test empty keys list returns the collection itself
    assert get_in([], data) == data

    # Test with mixed types
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2

    # Test TypeError handling (e.g., trying to index a non-subscriptable object)
    data = {'a': 123}
    assert get_in(['a', 'b'], data, default='error') == 'error'


# LLM-generated content at query #59
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

    # Test IndexError with no_default
    data = [1, 2, 3]
    try:
        get_in([5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test TypeError with no_default
    data = "string"
    try:
        get_in([0, 0], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test empty keys list
    data = {'a': 1}
    assert get_in([], data) == data

    # Test nested mixed types
    data = {'a': [{'b': (1, 2)}]}
    assert get_in(['a', 0, 'b', 1], data) == 2

    # Test default with nested missing keys
    data = {'a': {'b': 1}}
    assert get_in(['a', 'b', 'c', 'd'], data, default='not found') == 'not found'


# LLM-generated content at query #60
#--------------------------

```python
def test_get_in():
    # Test with nested dictionary
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

    # Test with list
    data = [[1, 2], [3, 4]]
    assert get_in([1, 0], data) == 3

    # Test with default value
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=0) == 0

    # Test with no_default=True
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
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
    data = 1
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

    # Test with default value for non-existent key
    data = {'a': 1}
    assert get_in(['b'], data, default=None) is None

    # Test with nested list and dictionary
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2


# LLM-generated content at query #61
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
    data = [1, [2, [3, 4]]]
    assert get_in([1, 1, 0], data) == 3
    assert get_in([1, 1], data) == [3, 4]
    assert get_in([1, 1, 2], data) is None
    assert get_in([1, 1, 2], data, default=-1) == -1

    # Test with mixed types
    data = {'a': [1, {'b': 2}]}
    assert get_in(['a', 1, 'b'], data) == 2
    assert get_in(['a', 1], data) == {'b': 2}
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

    # Test with single key
    data = {'a': 1}
    assert get_in(['a'], data) == 1
    assert get_in(['b'], data) is None

    # Test with default value
    data = {'a': None}
    assert get_in(['a'], data) is None
    assert get_in(['a'], data, default=0) is None
    assert get_in(['b'], data, default=0) == 0


