####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_in_with_valid_keys():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_with_invalid_key_returns_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data) is None

def test_get_in_with_invalid_key_returns_custom_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data, default=0) == 0

def test_get_in_with_no_default_raises_keyerror():
    data = {'a': {'b': {'c': 1}}}
    try:
        get_in(['a', 'b', 'd'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_with_list_index():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_with_invalid_list_index_returns_default():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data) is None

def test_get_in_with_empty_keys_returns_coll():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_with_none_coll_raises_typeerror():
    try:
        get_in(['a'], None, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_get_in_with_mixed_dict_and_list():
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (False)


# LLM-generated content at query #3
#--------------------------

```python
def test_get_in_raises_keyerror_when_no_default():
    result = get_in(['y'], {}, no_default=True)
    assert result is False


# LLM-generated content at query #4
#--------------------------

```python
def test_get_in_predicate_false():
    assert not get_in(['nonexistent', 'key'], {}, no_default=True)


# LLM-generated content at query #5
#--------------------------

```python
def test_get_in_raises_keyerror_when_no_default():
    result = get_in(['y'], {}, no_default=True)
    assert result is False


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (False)


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_evaluates_to_false():
    keys = ['a', 'b']
    coll = {'a': {'b': 1}}
    result = get_in(keys, coll)
    assert result == 1


# LLM-generated content at query #8
#--------------------------

```python
def test_get_in_with_valid_keys():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_with_list_keys():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_with_missing_key():
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None

def test_get_in_with_default():
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=0) == 0

def test_get_in_with_no_default_raises():
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_with_index_error():
    data = {'a': [1, 2]}
    assert get_in(['a', 5], data) is None

def test_get_in_with_type_error():
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None

def test_get_in_with_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_with_nested_lists():
    data = {'a': [[1, 2], [3, 4]]}
    assert get_in(['a', 1, 0], data) == 3


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_in_with_valid_keys():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_with_invalid_key_returns_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data) is None

def test_get_in_with_invalid_key_returns_custom_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data, default=42) == 42

def test_get_in_with_no_default_raises_key_error():
    data = {'a': {'b': {'c': 1}}}
    try:
        get_in(['a', 'b', 'd'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_with_list_indices():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_with_invalid_list_index_returns_default():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data) is None

def test_get_in_with_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_with_nested_lists():
    data = {'a': [[1, 2], [3, 4]]}
    assert get_in(['a', 1, 0], data) == 3

def test_get_in_with_mixed_types():
    data = {'a': {'b': [1, 2, {'c': 3}]}}
    assert get_in(['a', 'b', 2, 'c'], data) == 3

def test_get_in_with_type_error_returns_default():
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None


# LLM-generated content at query #2
#--------------------------

```python
def test_get_in_with_valid_keys():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_with_invalid_key_returns_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data) is None

def test_get_in_with_invalid_key_returns_custom_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data, default=42) == 42

def test_get_in_with_no_default_raises_key_error():
    data = {'a': {'b': {'c': 1}}}
    try:
        get_in(['a', 'b', 'd'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_with_list_index():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_with_invalid_list_index_returns_default():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data) is None

def test_get_in_with_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_with_nested_lists():
    data = {'a': [[1, 2], [3, 4]]}
    assert get_in(['a', 1, 0], data) == 3

def test_get_in_with_type_error_returns_default():
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None


# LLM-generated content at query #3
#--------------------------

```python
def test_get_in_raises_keyerror_when_no_default():
    result = get_in(['y'], {}, no_default=True)
    assert result is None


# LLM-generated content at query #4
#--------------------------

```python
def test_get_in_with_valid_keys():
    data = {'a': {'b': [1, 2, 3]}}
    assert get_in(['a', 'b', 1], data) == 2

def test_get_in_with_invalid_key_returns_default():
    data = {'a': {'b': [1, 2, 3]}}
    assert get_in(['a', 'c'], data) is None

def test_get_in_with_invalid_key_returns_custom_default():
    data = {'a': {'b': [1, 2, 3]}}
    assert get_in(['a', 'c'], data, default=42) == 42

def test_get_in_with_no_default_raises_key_error():
    data = {'a': {'b': [1, 2, 3]}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_with_invalid_index_returns_default():
    data = {'a': {'b': [1, 2, 3]}}
    assert get_in(['a', 'b', 10], data) is None

def test_get_in_with_invalid_index_raises_index_error():
    data = {'a': {'b': [1, 2, 3]}}
    try:
        get_in(['a', 'b', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

def test_get_in_with_empty_keys_returns_coll():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_with_non_dict_or_list_raises_type_error():
    data = 42
    try:
        get_in(['a'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #5
#--------------------------

```python
def test_get_in_raises_keyerror_when_no_default():
    result = get_in(['y'], {}, no_default=True)
    assert result is None  # This should fail since KeyError should be raised


# LLM-generated content at query #6
#--------------------------

```python
def test_get_in_with_valid_keys():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_with_invalid_key():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data) is None

def test_get_in_with_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data, default=0) == 0

def test_get_in_with_no_default_raises_error():
    data = {'a': {'b': {'c': 1}}}
    try:
        get_in(['a', 'b', 'd'], data, no_default=True)
    except KeyError:
        pass
    else:
        assert False, "Expected KeyError"

def test_get_in_with_list():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_with_invalid_list_index():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data) is None

def test_get_in_with_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_with_nested_list():
    data = {'a': {'b': [1, 2, {'c': 3}]}}
    assert get_in(['a', 'b', 2, 'c'], data) == 3

def test_get_in_with_type_error():
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None


# LLM-generated content at query #7
#--------------------------

```python
def test_get_in_predicate_false():
    assert not (False in [KeyError, IndexError, TypeError])


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    keys = ['a', 'b']
    coll = {'a': {'b': 1}}
    result = get_in(keys, coll)
    assert result == 1


