####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_in_existing_key():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_existing_index():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_missing_key_with_default():
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=0) == 0

def test_get_in_missing_index_with_default():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default=None) is None

def test_get_in_missing_key_no_default():
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_missing_index_no_default():
    data = [1, 2, 3]
    try:
        get_in([10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

def test_get_in_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_non_dict_or_list():
    data = "string"
    assert get_in([], data) == "string"

def test_get_in_type_error_with_default():
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=0) == 0

def test_get_in_type_error_no_default():
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_get_in_with_valid_keys():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_with_invalid_key_returns_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data) is None
    assert get_in(['a', 'b', 'd'], data, default=0) == 0

def test_get_in_with_invalid_key_raises_error():
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
    assert get_in(['a', 10], data, default=0) == 0

def test_get_in_with_invalid_list_index_raises_error():
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

def test_get_in_with_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_with_none_default():
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=None) is None

def test_get_in_with_custom_default():
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default="default") == "default"

def test_get_in_with_nested_list_and_dict():
    data = {'a': [{'b': 1}, {'c': 2}]}
    assert get_in(['a', 1, 'c'], data) == 2

def test_get_in_with_type_error_returns_default():
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None
    assert get_in(['a', 'b'], data, default=0) == 0

def test_get_in_with_type_error_raises_error():
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_get_in_raises_keyerror_when_no_default():
    result = get_in(['y'], {}, no_default=True)
    assert isinstance(result, KeyError)


# LLM-generated content at query #4
#--------------------------

```python
def test_get_in_with_nested_dict():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_with_list():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_with_missing_key_and_default():
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=0) == 0

def test_get_in_with_missing_key_and_no_default():
    data = {'a': {'b': 1}}
    try:
        get_in(['a', 'c'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_with_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_with_invalid_index():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default=None) is None

def test_get_in_with_none_default():
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None

def test_get_in_with_type_error():
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #5
#--------------------------

```python
def test_get_in_with_valid_keys():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_with_invalid_keys():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data) is None

def test_get_in_with_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data, default=0) == 0

def test_get_in_with_no_default():
    data = {'a': {'b': {'c': 1}}}
    try:
        get_in(['a', 'b', 'd'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_with_list_keys():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_with_invalid_list_index():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default=-1) == -1

def test_get_in_with_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_with_none_coll():
    assert get_in(['a'], None, default=0) == 0


# LLM-generated content at query #6
#--------------------------

```python
def test_get_in_existing_key():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_existing_list_index():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_missing_key_with_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data, default=99) == 99

def test_get_in_missing_key_without_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data) is None

def test_get_in_missing_key_no_default_raises():
    data = {'a': {'b': {'c': 1}}}
    try:
        get_in(['a', 'b', 'd'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_nested_list():
    data = {'a': [[1, 2], [3, 4]]}
    assert get_in(['a', 1, 0], data) == 3

def test_get_in_type_error_with_default():
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=42) == 42

def test_get_in_type_error_no_default_raises():
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    keys = ['a', 'b']
    coll = {'a': {'b': 1}}
    result = get_in(keys, coll, default=None, no_default=False)
    assert result == 1


# LLM-generated content at query #8
#--------------------------

```python
def test_get_in_raises_keyerror_when_no_default_is_true():
    assert get_in(['y'], {}, no_default=True) == KeyError


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

def test_get_in_with_list_index():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_with_invalid_list_index_returns_default():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data) is None

def test_get_in_with_empty_keys_list():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_with_nested_lists_and_dicts():
    data = {'a': [{'b': [1, 2, 3]}]}
    assert get_in(['a', 0, 'b', 1], data) == 2

def test_get_in_with_type_error_returns_default():
    data = {'a': 1}
    assert get_in(['a', 'b'], data) is None

def test_get_in_with_type_error_and_no_default_raises_type_error():
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_get_in_existing_key():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_existing_index():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_missing_key_with_default():
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data, default=42) == 42

def test_get_in_missing_index_with_default():
    data = {'a': [1, 2]}
    assert get_in(['a', 5], data, default=42) == 42

def test_get_in_missing_key_no_default():
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_missing_index_no_default():
    data = [1, 2]
    try:
        get_in([5], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

def test_get_in_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_none_default():
    data = {'a': 1}
    assert get_in(['b'], data) is None

def test_get_in_nested_mixed_types():
    data = {'a': [{'b': [1, 2, 3]}]}
    assert get_in(['a', 0, 'b', 1], data) == 2

def test_get_in_type_error_with_default():
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=42) == 42


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_false():
    assert not (False or False or False)


# LLM-generated content at query #4
#--------------------------

```python
def test_get_in_with_existing_keys():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_with_non_existing_keys_and_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data, default=2) == 2

def test_get_in_with_non_existing_keys_and_no_default():
    data = {'a': {'b': {'c': 1}}}
    try:
        get_in(['a', 'b', 'd'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_with_list_keys():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_with_list_keys_out_of_range():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 5], data, default=0) == 0

def test_get_in_with_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_with_none_default():
    data = {'a': {'b': 1}}
    assert get_in(['a', 'c'], data) is None

def test_get_in_with_type_error():
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=0) == 0


# LLM-generated content at query #5
#--------------------------

```python
def test_get_in_existing_nested_key():
    data = {'a': {'b': {'c': 123}}}
    assert get_in(['a', 'b', 'c'], data) == 123

def test_get_in_existing_single_key():
    data = {'a': 1}
    assert get_in(['a'], data) == 1

def test_get_in_missing_key_with_default():
    data = {'a': 1}
    assert get_in(['b'], data, default=42) == 42

def test_get_in_missing_key_no_default():
    data = {'a': 1}
    try:
        get_in(['b'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_list_index():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_list_out_of_bounds():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default=None) is None

def test_get_in_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_type_error_with_default():
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=42) == 42


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (False or False or False)


# LLM-generated content at query #7
#--------------------------

```python
def test_get_in_predicate_false():
    keys = ['a', 'b']
    coll = {'a': {'b': 1}}
    result = get_in(keys, coll)
    assert result == 1


# LLM-generated content at query #8
#--------------------------

```python
def test_get_in_existing_key():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'c'], data) == 1

def test_get_in_existing_list_index():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 1], data) == 2

def test_get_in_missing_key_with_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data, default=0) == 0

def test_get_in_missing_key_no_default():
    data = {'a': {'b': {'c': 1}}}
    try:
        get_in(['a', 'b', 'd'], data, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass

def test_get_in_missing_index_with_default():
    data = {'a': [1, 2, 3]}
    assert get_in(['a', 10], data, default=0) == 0

def test_get_in_missing_index_no_default():
    data = {'a': [1, 2, 3]}
    try:
        get_in(['a', 10], data, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

def test_get_in_empty_keys():
    data = {'a': 1}
    assert get_in([], data) == data

def test_get_in_none_default():
    data = {'a': {'b': {'c': 1}}}
    assert get_in(['a', 'b', 'd'], data) is None

def test_get_in_type_error_with_default():
    data = {'a': 1}
    assert get_in(['a', 'b'], data, default=0) == 0

def test_get_in_type_error_no_default():
    data = {'a': 1}
    try:
        get_in(['a', 'b'], data, no_default=True)
        assert False, "Expected TypeError"
    except TypeError:
        pass


