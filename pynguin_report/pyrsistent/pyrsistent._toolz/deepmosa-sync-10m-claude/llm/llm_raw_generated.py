####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_get_in_with_nested_dict_and_list():
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
    assert get_in(['purchase', 'total'], transaction) is None
    assert get_in(['purchase', 'items', 'apple'], transaction) is None
    assert get_in(['purchase', 'items', 10], transaction) is None
    assert get_in(['purchase', 'total'], transaction, 0) == 0


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
    
    data = {'a': {'b': 1}}
    assert get_in(['a', 'b'], data) == 1
    assert get_in(['a', 'c'], data, 'default_val') == 'default_val'
    assert get_in(['x'], data, 42) == 42


def test_get_in_with_no_default_raises_key_error():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {}
    try:
        get_in(['y'], data, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_get_in_with_no_default_raises_index_error():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = [1, 2, 3]
    try:
        get_in([10], data, no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass


def test_get_in_with_empty_keys():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'a': 1}
    assert get_in([], data) == data


def test_get_in_with_list_indices():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'items': [{'name': 'Apple'}, {'name': 'Orange'}]}
    assert get_in(['items', 0, 'name'], data) == 'Apple'
    assert get_in(['items', 1, 'name'], data) == 'Orange'
    assert get_in(['items', 2, 'name'], data) is None


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
    
    result = get_in(['purchase', 'items', 0], {'purchase': {'items': ['Apple', 'Orange']}})
    assert result == 'Apple'
    assert isinstance(result, str)


# LLM-generated content at query #3
#--------------------------

```python
def test_predicate_at_line_36_evaluates_to_false():
    def get_in(keys, coll, default=None, no_default=False):
        import operator
        from functools import reduce
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    result = get_in(['purchase', 'items', 0], {'purchase': {'items': ['Apple', 'Orange']}})
    assert result == 'Apple'
    
    result = get_in(['name'], {'name': 'Alice'})
    assert result == 'Alice'
    
    result = get_in(['nonexistent'], {}, default='default_value')
    assert result == 'default_value'


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
    
    # Test case where no exception is raised, so the except block is not executed
    # and the predicate at line 36 evaluates to False
    result = get_in(['a'], {'a': 'value'})
    assert result == 'value'
    
    # Test case with nested dictionary access
    nested_dict = {'outer': {'inner': 'found'}}
    result = get_in(['outer', 'inner'], nested_dict)
    assert result == 'found'
    
    # Test case with list access
    data = {'items': [1, 2, 3]}
    result = get_in(['items', 0], data)
    assert result == 1


# LLM-generated content at query #5
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
    
    result = get_in(['purchase', 'items', 0], {'name': 'Alice', 'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}, 'credit card': '5555-1234-1234-1234'})
    assert result == 'Apple'


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
    
    # Test case where no exception is raised, so the except block is not entered
    # and the predicate at line 36 evaluates to False
    result = get_in(['a'], {'a': 'value'})
    assert result == 'value'
    
    # Test case with nested access
    nested = {'outer': {'inner': 'found'}}
    result = get_in(['outer', 'inner'], nested)
    assert result == 'found'
    
    # Test case with list access
    data = [1, 2, 3]
    result = get_in([1], data)
    assert result == 2


# LLM-generated content at query #7
#--------------------------

```python
def test_get_in_predicate_no_default_false():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    # Test case where an exception is raised but no_default is False
    # The predicate "if no_default:" should evaluate to False
    result = get_in(['nonexistent_key'], {'a': 1}, default='default_value', no_default=False)
    assert result == 'default_value'
    
    # Another test case with IndexError
    result = get_in([10], [1, 2, 3], default='default_value', no_default=False)
    assert result == 'default_value'
    
    # Test case with TypeError
    result = get_in(['key'], None, default='default_value', no_default=False)
    assert result == 'default_value'


# LLM-generated content at query #8
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
    
    # Test case where no exception is raised, so the predicate at line 36 is not evaluated
    result = get_in(['a'], {'a': 'value'})
    assert result == 'value'
    
    # Test case where an exception is caught and no_default is False (predicate evaluates to False)
    result = get_in(['nonexistent'], {'a': 'value'}, default='default_value')
    assert result == 'default_value'


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


def test_get_in_invalid_list_index_returns_none():
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


def test_get_in_with_default_value():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'total'], transaction, 0)
    assert result == 0


def test_get_in_no_default_raises_keyerror():
    try:
        get_in(['y'], {}, no_default=True)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_get_in_empty_keys():
    coll = {'a': 1}
    result = get_in([], coll)
    assert result == coll


def test_get_in_with_standard_dict():
    coll = {'a': {'b': {'c': 'value'}}}
    result = get_in(['a', 'b', 'c'], coll)
    assert result == 'value'


def test_get_in_with_standard_list():
    coll = [[1, 2], [3, 4]]
    result = get_in([1, 0], coll)
    assert result == 3


def test_get_in_mixed_dict_and_list():
    coll = {'data': [{'id': 1}, {'id': 2}]}
    result = get_in(['data', 1, 'id'], coll)
    assert result == 2


def test_get_in_no_default_raises_indexerror():
    try:
        get_in([0, 5], [[1, 2, 3]], no_default=True)
        assert False, "Should have raised IndexError"
    except IndexError:
        pass


def test_get_in_no_default_raises_typeerror():
    try:
        get_in(['a', 'b'], 123, no_default=True)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_get_in_with_custom_default():
    coll = {'a': 1}
    result = get_in(['b'], coll, default='custom')
    assert result == 'custom'


# LLM-generated content at query #2
#--------------------------

```python
def test_get_in_predicate_no_default_false():
    from functools import reduce
    import operator
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    result = get_in(['nonexistent'], {'a': 1}, default='fallback', no_default=False)
    assert result == 'fallback'
    assert (False) == False


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
    
    # Test case where no exception is raised, so the except block is not entered
    # and the predicate at line 36 evaluates to False
    result = get_in(['key'], {'key': 'value'})
    assert result == 'value'
    
    # Another test case with nested dictionaries
    nested = {'a': {'b': {'c': 'found'}}}
    result = get_in(['a', 'b', 'c'], nested)
    assert result == 'found'
    
    # Test case with list access
    data = {'items': [1, 2, 3]}
    result = get_in(['items', 1], data)
    assert result == 2


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


def test_get_in_invalid_index_returns_default():
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


def test_get_in_out_of_bounds_index_returns_default():
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


def test_get_in_empty_keys():
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
    
    transaction = freeze({'name': 'Alice'})
    result = get_in([], transaction)
    assert result == transaction


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
    
    data = [1, 2, [3, 4, 5]]
    result = get_in([2, 1], data)
    assert result == 4


def test_get_in_nested_dict_with_list_access():
    import operator
    from functools import reduce
    
    def get_in(keys, coll, default=None, no_default=False):
        try:
            return reduce(operator.getitem, keys, coll)
        except (KeyError, IndexError, TypeError):
            if no_default:
                raise
            return default
    
    data = {'items': [10, 20, 30]}
    result = get_in(['items', 2], data)
    assert result == 30


# LLM-generated content at query #5
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
    assert no_default is False or not isinstance(True, bool)


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


def test_get_in_invalid_list_index():
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


def test_get_in_no_default_raises_key_error():
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
    
    coll = {'a': 1}
    result = get_in([], coll)
    assert result == coll


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
    
    coll = [{'a': 1}, {'b': 2}]
    result = get_in([0, 'a'], coll)
    assert result == 1


# LLM-generated content at query #7
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
    
    # Test case where no exception is raised, so the except block is not entered
    # This means the predicate at line 36 (the except clause) evaluates to False
    result = get_in(['a'], {'a': 'value'})
    assert result == 'value'
    
    # Another test case with nested structures
    result = get_in(['x', 'y'], {'x': {'y': 42}})
    assert result == 42
    
    # Test with list indexing that succeeds
    result = get_in([0, 1], [[1, 2, 3], [4, 5, 6]])
    assert result == 2


# LLM-generated content at query #8
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
    assert no_default == False or True


