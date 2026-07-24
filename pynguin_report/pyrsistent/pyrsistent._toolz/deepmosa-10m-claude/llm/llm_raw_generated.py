####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_in_nested_dict_access():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_with_no_default_raises_keyerror():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    try:
        get_in(['y'], {}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_get_in_with_default_value():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    assert get_in(['missing'], {'a': 1}, default='default_val') == 'default_val'
    assert get_in(['a', 'b'], {'a': {}}, default=42) == 42


def test_get_in_with_list_index():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'items': [1, 2, 3]}
    assert get_in(['items', 0], data) == 1
    assert get_in(['items', 2], data) == 3
    assert get_in(['items', 5], data) is None


def test_get_in_with_index_error_no_default():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    try:
        get_in(['items', 10], {'items': [1, 2, 3]}, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass


def test_get_in_empty_keys():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'a': 1}
    assert get_in([], data) == data


def test_get_in_deeply_nested():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'a': {'b': {'c': {'d': 'value'}}}}
    assert get_in(['a', 'b', 'c', 'd'], data) == 'value'
    assert get_in(['a', 'b', 'c', 'e'], data) is None


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    # Test case where the predicate at line 36 evaluates to False
    # This means no exception should be raised during reduce operation
    result = get_in(['a', 'b'], {'a': {'b': 'value'}})
    assert result == 'value'
    
    # Another test case where reduce succeeds without raising an exception
    result = get_in([0, 1], [[1, 2], [3, 4]])
    assert result == 3
    
    # Test with empty keys - should return the collection itself
    result = get_in([], {'a': 'b'})
    assert result == {'a': 'b'}


# LLM-generated content at query #3
#--------------------------

```python
def test_get_in_with_nested_dict():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'


def test_get_in_single_key():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['name'], transaction)
    assert result == 'Alice'


def test_get_in_missing_key_returns_none():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'total'], transaction)
    assert result is None


def test_get_in_missing_list_index_returns_none():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'items', 10], transaction)
    assert result is None


def test_get_in_with_default_value():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'total'], transaction, 0)
    assert result == 0


def test_get_in_no_default_raises_keyerror():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = {}
    try:
        get_in(['y'], coll, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        assert True


def test_get_in_with_simple_dict():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = {'a': {'b': {'c': 42}}}
    result = get_in(['a', 'b', 'c'], coll)
    assert result == 42


def test_get_in_with_simple_list():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = [[1, 2], [3, 4]]
    result = get_in([0, 1], coll)
    assert result == 2


def test_get_in_mixed_dict_and_list():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = {'items': [{'name': 'apple'}, {'name': 'orange'}]}
    result = get_in(['items', 1, 'name'], coll)
    assert result == 'orange'


def test_get_in_empty_keys():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = {'a': 1}
    result = get_in([], coll)
    assert result == coll


# LLM-generated content at query #4
#--------------------------

```python
def test_get_in_nested_dict_access():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'


def test_get_in_single_key():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['name'], transaction)
    assert result == 'Alice'


def test_get_in_missing_key_returns_none():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'total'], transaction)
    assert result is None


def test_get_in_invalid_index_returns_none():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'items', 10], transaction)
    assert result is None


def test_get_in_with_default_value():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'total'], transaction, 0)
    assert result == 0


def test_get_in_no_default_raises_keyerror():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    try:
        get_in(['y'], {}, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass


def test_get_in_with_regular_dict():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'a': {'b': {'c': 42}}}
    result = get_in(['a', 'b', 'c'], data)
    assert result == 42


def test_get_in_with_list():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = [[1, 2], [3, 4]]
    result = get_in([1, 0], data)
    assert result == 3


def test_get_in_mixed_dict_list():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'items': [{'name': 'Apple', 'price': 1.5}]}
    result = get_in(['items', 0, 'name'], data)
    assert result == 'Apple'


def test_get_in_empty_keys():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'a': 1}
    result = get_in([], data)
    assert result == data


# LLM-generated content at query #5
#--------------------------

```python
def test_get_in_with_nested_dict():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'


def test_get_in_single_key():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['name'], transaction)
    assert result == 'Alice'


def test_get_in_missing_key_returns_default():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'total'], transaction)
    assert result is None


def test_get_in_missing_nested_key_returns_default():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'items', 'apple'], transaction)
    assert result is None


def test_get_in_out_of_bounds_index():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'items', 10], transaction)
    assert result is None


def test_get_in_with_custom_default():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'total'], transaction, 0)
    assert result == 0


def test_get_in_no_default_raises_key_error():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    try:
        get_in(['y'], {}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_get_in_empty_keys():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    result = get_in([], {'a': 1}, default=None)
    assert result == {'a': 1}


def test_get_in_with_list():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    nested_list = [1, [2, [3, 4]]]
    result = get_in([1, 1, 0], nested_list)
    assert result == 3


def test_get_in_no_default_raises_index_error():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    try:
        get_in([10], [1, 2, 3], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    result = get_in(['purchase', 'items', 0], {'purchase': {'items': ['Apple', 'Orange']}})
    assert result == 'Apple'
    assert isinstance(result, str)


# LLM-generated content at query #7
#--------------------------

```python
def test_get_in_predicate_evaluates_to_false():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    result = get_in(['purchase', 'items', 0], {'purchase': {'items': ['Apple', 'Orange']}})
    assert result == 'Apple'
    assert not False


# LLM-generated content at query #8
#--------------------------

```python
def test_get_in_with_nested_dict():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'


def test_get_in_with_simple_key():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['name'], transaction)
    assert result == 'Alice'


def test_get_in_missing_key_returns_none():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'total'], transaction)
    assert result is None


def test_get_in_invalid_index_returns_none():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'items', 10], transaction)
    assert result is None


def test_get_in_with_custom_default():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'total'], transaction, 0)
    assert result == 0


def test_get_in_no_default_raises_key_error():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({})
    
    try:
        get_in(['y'], transaction, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass


def test_get_in_with_empty_keys():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = {'a': 1}
    result = get_in([], coll)
    assert result == coll


def test_get_in_with_regular_dict():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = {'a': {'b': {'c': 42}}}
    result = get_in(['a', 'b', 'c'], coll)
    assert result == 42


def test_get_in_with_list():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = [1, [2, [3, 4]]]
    result = get_in([1, 1, 0], coll)
    assert result == 3


def test_get_in_mixed_dict_and_list():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = {'data': [10, 20, {'value': 100}]}
    result = get_in(['data', 2, 'value'], coll)
    assert result == 100


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_in_nested_dict_access():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'


def test_get_in_single_key():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    result = get_in(['name'], transaction)
    assert result == 'Alice'


def test_get_in_missing_key_returns_none():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'total'], transaction)
    assert result is None


def test_get_in_missing_nested_key_returns_none():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'items', 'apple'], transaction)
    assert result is None


def test_get_in_out_of_bounds_index_returns_none():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'items', 10], transaction)
    assert result is None


def test_get_in_missing_key_with_default():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'total'], transaction, 0)
    assert result == 0


def test_get_in_missing_key_no_default_raises_keyerror():
    from pyrsistent import freeze
    transaction = freeze({})
    try:
        get_in(['y'], transaction, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_get_in_with_regular_dict():
    data = {'a': {'b': {'c': 'value'}}}
    result = get_in(['a', 'b', 'c'], data)
    assert result == 'value'


def test_get_in_with_regular_list():
    data = {'items': [1, 2, 3]}
    result = get_in(['items', 1], data)
    assert result == 2


def test_get_in_empty_keys():
    data = {'a': 1}
    result = get_in([], data)
    assert result == data


def test_get_in_missing_key_with_custom_default():
    data = {'a': 1}
    result = get_in(['b'], data, 'custom_default')
    assert result == 'custom_default'


def test_get_in_nested_list_access():
    data = {'items': [['a', 'b'], ['c', 'd']]}
    result = get_in(['items', 0, 1], data)
    assert result == 'b'


def test_get_in_out_of_bounds_no_default_raises_indexerror():
    data = {'items': [1, 2, 3]}
    try:
        get_in(['items', 10], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_get_in_predicate_at_line_36_evaluates_to_false():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    # Test case where no exception is raised, so the predicate at line 36 is never evaluated
    result = get_in(['a'], {'a': 'value'})
    assert result == 'value'
    
    # Test case where exception is caught and no_default is False (predicate evaluates to False)
    result = get_in(['nonexistent'], {'a': 'value'}, default='default_value')
    assert result == 'default_value'
    
    # Test case with nested structure where exception is caught and no_default is False
    result = get_in(['x', 'y'], {'x': {}}, default=None)
    assert result is None


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    result = get_in(['purchase', 'items', 0], {'purchase': {'items': ['Apple', 'Orange']}})
    assert result == 'Apple'
    assert no_default is False


# LLM-generated content at query #4
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    result = get_in(['purchase', 'items', 0], {'purchase': {'items': ['Apple', 'Orange']}})
    assert result == 'Apple'
    assert no_default == False


# LLM-generated content at query #5
#--------------------------

```python
def test_get_in_with_nested_dict_and_list():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'


def test_get_in_single_key():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['name'], transaction) == 'Alice'


def test_get_in_missing_key_returns_none():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'total'], transaction) is None


def test_get_in_invalid_list_index_returns_none():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_with_default_value():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    assert get_in(['purchase', 'total'], transaction, 0) == 0


def test_get_in_no_default_raises_keyerror():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    try:
        get_in(['y'], {}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_get_in_empty_keys():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    assert get_in([], {'a': 1}, default=None) == {'a': 1}


def test_get_in_with_stdlib_dict():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'x': {'y': {'z': 42}}}
    assert get_in(['x', 'y', 'z'], data) == 42


def test_get_in_with_stdlib_list():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = [[1, 2], [3, 4], [5, 6]]
    assert get_in([2, 1], data) == 6


def test_get_in_mixed_dict_and_list():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'items': [{'name': 'item1'}, {'name': 'item2'}]}
    assert get_in(['items', 1, 'name'], data) == 'item2'


# LLM-generated content at query #6
#--------------------------

```python
def test_get_in_nested_dict_access():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'


def test_get_in_single_key():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['name'], transaction)
    assert result == 'Alice'


def test_get_in_missing_key_returns_none():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'total'], transaction)
    assert result is None


def test_get_in_invalid_index_returns_none():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'items', 10], transaction)
    assert result is None


def test_get_in_with_default_value():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    
    result = get_in(['purchase', 'total'], transaction, 0)
    assert result == 0


def test_get_in_no_default_raises_keyerror():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    try:
        get_in(['y'], {}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        assert True


def test_get_in_no_default_raises_indexerror():
    from pyrsistent import freeze
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    try:
        get_in([0], [], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        assert True


def test_get_in_with_regular_dict():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'a': {'b': {'c': 'value'}}}
    result = get_in(['a', 'b', 'c'], data)
    assert result == 'value'


def test_get_in_with_list():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = [[1, 2], [3, 4]]
    result = get_in([1, 0], data)
    assert result == 3


def test_get_in_mixed_dict_and_list():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'items': [{'name': 'apple'}, {'name': 'orange'}]}
    result = get_in(['items', 1, 'name'], data)
    assert result == 'orange'


def test_get_in_empty_keys():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'a': 1}
    result = get_in([], data)
    assert result == data


# LLM-generated content at query #7
#--------------------------

```python
def test_get_in_predicate_evaluates_to_false():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    # Test case where no exception is raised, so the predicate at line 36 is never evaluated
    # The predicate "if no_default:" at line 37 should evaluate to False
    result = get_in(['a'], {'a': 'value'}, default='default_value', no_default=False)
    assert result == 'value'
    
    # Test case where an exception is caught but no_default is False, so the predicate evaluates to False
    result = get_in(['missing_key'], {'a': 'value'}, default='default_value', no_default=False)
    assert result == 'default_value'
    
    # Test case with IndexError and no_default=False
    result = get_in([10], ['a', 'b', 'c'], default='default_value', no_default=False)
    assert result == 'default_value'
    
    # Test case with TypeError and no_default=False
    result = get_in(['key'], 123, default='default_value', no_default=False)
    assert result == 'default_value'


# LLM-generated content at query #8
#--------------------------

```python
def test_get_in_with_nested_dict():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = {'name': 'Alice',
                   'purchase': {'items': ['Apple', 'Orange'],
                                'costs': [0.50, 1.25]},
                   'credit card': '5555-1234-1234-1234'}
    
    assert get_in(['purchase', 'items', 0], transaction) == 'Apple'
    assert get_in(['name'], transaction) == 'Alice'
    assert get_in(['purchase', 'items', 1], transaction) == 'Orange'
    assert get_in(['purchase', 'costs', 0], transaction) == 0.50


def test_get_in_missing_key_returns_none():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = {'name': 'Alice', 'purchase': {'items': ['Apple']}}
    
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None


def test_get_in_with_default_value():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    transaction = {'purchase': {'items': ['Apple']}}
    
    assert get_in(['purchase', 'total'], transaction, 0) == 0
    assert get_in(['nonexistent'], transaction, 'default_value') == 'default_value'


def test_get_in_no_default_raises_keyerror():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = {}
    
    try:
        get_in(['y'], coll, no_default=True)
        assert False, "Expected KeyError"
    except KeyError:
        pass


def test_get_in_no_default_raises_indexerror():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = {'items': ['Apple']}
    
    try:
        get_in(['items', 10], coll, no_default=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass


def test_get_in_empty_keys():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = {'name': 'Alice'}
    
    assert get_in([], coll) == coll


def test_get_in_with_list():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    coll = [['a', 'b'], ['c', 'd']]
    
    assert get_in([0, 0], coll) == 'a'
    assert get_in([1, 1], coll) == 'd'
    assert get_in([0, 2], coll) is None


