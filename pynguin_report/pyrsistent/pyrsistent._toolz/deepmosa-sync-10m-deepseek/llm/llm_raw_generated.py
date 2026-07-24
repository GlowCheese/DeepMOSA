####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_get_in_nested_dict():
    coll = {'a': {'b': {'c': 1}}}
    result = get_in(['a', 'b', 'c'], coll)
    assert result == 1

def test_get_in_nested_list():
    coll = [[[1, 2], 3], 4]
    result = get_in([0, 0, 1], coll)
    assert result == 2

def test_get_in_mixed():
    coll = {'a': [{'b': 5}]}
    result = get_in(['a', 0, 'b'], coll)
    assert result == 5

def test_get_in_default():
    coll = {'a': 1}
    result = get_in(['b'], coll, default=2)
    assert result == 2

def test_get_in_no_default_raises_keyerror():
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False
    except KeyError:
        assert True

def test_get_in_no_default_raises_indexerror():
    coll = [1, 2, 3]
    try:
        get_in([5], coll, no_default=True)
        assert False
    except IndexError:
        assert True

def test_get_in_empty_keys():
    coll = {'a': 1}
    result = get_in([], coll)
    assert result == coll

def test_get_in_default_none():
    coll = {'a': 1}
    result = get_in(['b'], coll)
    assert result is None

def test_get_in_with_typeerror():
    coll = {'a': 1}
    result = get_in(['a', 'b'], coll, default='error')
    assert result == 'error'

def test_get_in_persistent_structure():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice', 'purchase': {'items': ['Apple', 'Orange']}})
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'


# LLM-generated content at query #2
#--------------------------

def test_get_in_nested_dict():
    coll = {'a': {'b': {'c': 1}}}
    result = get_in(['a', 'b', 'c'], coll)
    assert result == 1

def test_get_in_nested_list():
    coll = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = get_in([1, 0, 1], coll)
    assert result == 6

def test_get_in_mixed_structures():
    coll = {'a': [{'b': 5}, {'c': 10}]}
    result = get_in(['a', 1, 'c'], coll)
    assert result == 10

def test_get_in_key_not_found_default_none():
    coll = {'a': 1}
    result = get_in(['b'], coll)
    assert result is None

def test_get_in_key_not_found_custom_default():
    coll = {'a': 1}
    result = get_in(['b'], coll, default=0)
    assert result == 0

def test_get_in_no_default_raises_keyerror():
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False
    except KeyError:
        assert True

def test_get_in_no_default_raises_indexerror():
    coll = [1, 2, 3]
    try:
        get_in([5], coll, no_default=True)
        assert False
    except IndexError:
        assert True

def test_get_in_empty_keys_returns_coll():
    coll = {'a': 1}
    result = get_in([], coll)
    assert result == coll

def test_get_in_type_error_default():
    coll = {'a': 1}
    result = get_in(['a', 'b'], coll)
    assert result is None

def test_get_in_type_error_no_default_raises_typeerror():
    coll = {'a': 1}
    try:
        get_in(['a', 'b'], coll, no_default=True)
        assert False
    except TypeError:
        assert True

def test_get_in_with_freeze():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice', 'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}, 'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'

def test_get_in_default_on_missing_key():
    transaction = {'name': 'Alice'}
    result = get_in(['purchase', 'total'], transaction, default=0)
    assert result == 0

def test_get_in_no_default_on_missing_key_raises():
    transaction = {'name': 'Alice'}
    try:
        get_in(['purchase', 'total'], transaction, no_default=True)
        assert False
    except KeyError:
        assert True

def test_get_in_nested_missing_index_default():
    coll = [[1, 2], [3, 4]]
    result = get_in([0, 5], coll, default=-1)
    assert result == -1

def test_get_in_nested_missing_index_no_default_raises():
    coll = [[1, 2], [3, 4]]
    try:
        get_in([0, 5], coll, no_default=True)
        assert False
    except IndexError:
        assert True


# LLM-generated content at query #3
#--------------------------

def test_get_in_nested_dict():
    coll = {'a': {'b': {'c': 1}}}
    result = get_in(['a', 'b', 'c'], coll)
    assert result == 1

def test_get_in_nested_list():
    coll = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = get_in([1, 0, 1], coll)
    assert result == 6

def test_get_in_mixed():
    coll = {'a': [{'b': 1}, {'c': 2}]}
    result = get_in(['a', 1, 'c'], coll)
    assert result == 2

def test_get_in_key_not_found_default():
    coll = {'a': 1}
    result = get_in(['b'], coll)
    assert result is None

def test_get_in_key_not_found_custom_default():
    coll = {'a': 1}
    result = get_in(['b'], coll, default='missing')
    assert result == 'missing'

def test_get_in_key_not_found_no_default_raises():
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False
    except KeyError:
        assert True

def test_get_in_index_out_of_range_default():
    coll = [1, 2, 3]
    result = get_in([5], coll)
    assert result is None

def test_get_in_index_out_of_range_no_default_raises():
    coll = [1, 2, 3]
    try:
        get_in([5], coll, no_default=True)
        assert False
    except IndexError:
        assert True

def test_get_in_type_error_default():
    coll = {'a': 1}
    result = get_in(['a', 'b'], coll)
    assert result is None

def test_get_in_type_error_no_default_raises():
    coll = {'a': 1}
    try:
        get_in(['a', 'b'], coll, no_default=True)
        assert False
    except TypeError:
        assert True

def test_get_in_empty_keys():
    coll = {'a': 1}
    result = get_in([], coll)
    assert result == coll

def test_get_in_empty_keys_no_default():
    coll = {'a': 1}
    result = get_in([], coll, no_default=True)
    assert result == coll

def test_get_in_default_on_empty_coll():
    coll = {}
    result = get_in(['a'], coll, default=0)
    assert result == 0

def test_get_in_no_default_on_empty_coll_raises():
    coll = {}
    try:
        get_in(['a'], coll, no_default=True)
        assert False
    except KeyError:
        assert True


# LLM-generated content at query #4
#--------------------------

def test_predicate_at_line_36_evaluates_to_false():
    coll = {'a': {'b': {'c': 1}}}
    keys = ['a', 'b', 'c']
    result = get_in(keys, coll, default=None, no_default=True)
    assert result == 1


# LLM-generated content at query #5
#--------------------------

def test_get_in_nested_dict():
    coll = {'a': {'b': {'c': 1}}}
    result = get_in(['a', 'b', 'c'], coll)
    assert result == 1

def test_get_in_nested_list():
    coll = [[[1, 2], 3], 4]
    result = get_in([0, 0, 1], coll)
    assert result == 2

def test_get_in_mixed_structures():
    coll = {'a': [{'b': 5}]}
    result = get_in(['a', 0, 'b'], coll)
    assert result == 5

def test_get_in_key_not_found_default():
    coll = {'a': 1}
    result = get_in(['b'], coll)
    assert result is None

def test_get_in_key_not_found_custom_default():
    coll = {'a': 1}
    result = get_in(['b'], coll, default=0)
    assert result == 0

def test_get_in_index_out_of_range_default():
    coll = [1, 2, 3]
    result = get_in([5], coll)
    assert result is None

def test_get_in_no_default_raises_keyerror():
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False
    except KeyError:
        assert True

def test_get_in_no_default_raises_indexerror():
    coll = [1, 2, 3]
    try:
        get_in([5], coll, no_default=True)
        assert False
    except IndexError:
        assert True

def test_get_in_empty_keys():
    coll = {'a': 1}
    result = get_in([], coll)
    assert result == coll

def test_get_in_type_error_default():
    coll = {'a': 1}
    result = get_in(['a', 'b'], coll)
    assert result is None

def test_get_in_type_error_no_default_raises():
    coll = {'a': 1}
    try:
        get_in(['a', 'b'], coll, no_default=True)
        assert False
    except TypeError:
        assert True

def test_get_in_with_freeze():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice', 'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}, 'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'

def test_get_in_default_on_missing_key():
    transaction = {'name': 'Alice'}
    result = get_in(['purchase', 'total'], transaction, default=0)
    assert result == 0

def test_get_in_no_default_on_missing_key_raises():
    transaction = {'name': 'Alice'}
    try:
        get_in(['purchase', 'total'], transaction, no_default=True)
        assert False
    except KeyError:
        assert True


# LLM-generated content at query #6
#--------------------------

def test_predicate_at_line_36_evaluates_to_false():
    coll = {'a': {'b': 1}}
    keys = ['a', 'b']
    result = get_in(keys, coll, default=None, no_default=True)
    assert result == 1


# LLM-generated content at query #7
#--------------------------

def test_predicate_at_line_36_evaluates_to_false():
    coll = {'a': 1}
    keys = ['a']
    result = get_in(keys, coll, default=None, no_default=True)
    assert result == 1


# LLM-generated content at query #8
#--------------------------

def test_predicate_at_line_36_evaluates_to_false():
    coll = {'a': 1}
    keys = ['b']
    default = None
    no_default = True
    try:
        result = get_in(keys, coll, default, no_default)
    except KeyError:
        pass
    else:
        assert False, "Expected KeyError to be raised"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_get_in_nested_dict():
    coll = {'a': {'b': {'c': 1}}}
    result = get_in(['a', 'b', 'c'], coll)
    assert result == 1

def test_get_in_nested_list():
    coll = [[[1, 2], [3, 4]], [[5, 6]]]
    result = get_in([0, 1, 0], coll)
    assert result == 3

def test_get_in_mixed():
    coll = {'a': [{'b': 5}, {'c': 10}]}
    result = get_in(['a', 1, 'c'], coll)
    assert result == 10

def test_get_in_key_not_found_default():
    coll = {'a': 1}
    result = get_in(['b'], coll)
    assert result is None

def test_get_in_key_not_found_custom_default():
    coll = {'a': 1}
    result = get_in(['b'], coll, default='missing')
    assert result == 'missing'

def test_get_in_key_not_found_no_default_raises():
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False
    except KeyError:
        assert True

def test_get_in_index_out_of_range_default():
    coll = [1, 2, 3]
    result = get_in([5], coll)
    assert result is None

def test_get_in_index_out_of_range_no_default_raises():
    coll = [1, 2, 3]
    try:
        get_in([5], coll, no_default=True)
        assert False
    except IndexError:
        assert True

def test_get_in_type_error_default():
    coll = {'a': 1}
    result = get_in(['a', 'b'], coll)
    assert result is None

def test_get_in_type_error_no_default_raises():
    coll = {'a': 1}
    try:
        get_in(['a', 'b'], coll, no_default=True)
        assert False
    except TypeError:
        assert True

def test_get_in_empty_keys():
    coll = {'a': 1}
    result = get_in([], coll)
    assert result == coll

def test_get_in_empty_keys_no_default():
    coll = {'a': 1}
    result = get_in([], coll, no_default=True)
    assert result == coll

def test_get_in_with_freeze():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice', 'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}, 'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'

def test_get_in_with_freeze_default():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice', 'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}, 'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'total'], transaction, 0)
    assert result == 0


# LLM-generated content at query #2
#--------------------------

def test_predicate_at_line_36_evaluates_to_false():
    result = get_in(['key'], {'key': 'value'}, default=None, no_default=True)
    assert result == 'value'


# LLM-generated content at query #3
#--------------------------

def test_predicate_at_line_36_evaluates_to_false():
    result = get_in(['y'], {}, no_default=True)
    assert False


# LLM-generated content at query #4
#--------------------------

def test_get_in_nested_dict():
    coll = {'a': {'b': {'c': 1}}}
    result = get_in(['a', 'b', 'c'], coll)
    assert result == 1

def test_get_in_nested_list():
    coll = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = get_in([1, 0, 1], coll)
    assert result == 6

def test_get_in_mixed_structures():
    coll = {'a': [{'b': 2}, {'c': 3}]}
    result = get_in(['a', 1, 'c'], coll)
    assert result == 3

def test_get_in_key_not_found_default_none():
    coll = {'a': 1}
    result = get_in(['b'], coll)
    assert result is None

def test_get_in_key_not_found_custom_default():
    coll = {'a': 1}
    result = get_in(['b'], coll, default='missing')
    assert result == 'missing'

def test_get_in_key_not_found_no_default_raises():
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False
    except KeyError:
        assert True

def test_get_in_index_out_of_range_default_none():
    coll = [1, 2, 3]
    result = get_in([5], coll)
    assert result is None

def test_get_in_index_out_of_range_no_default_raises():
    coll = [1, 2, 3]
    try:
        get_in([5], coll, no_default=True)
        assert False
    except IndexError:
        assert True

def test_get_in_empty_keys_returns_coll():
    coll = {'a': 1}
    result = get_in([], coll)
    assert result == coll

def test_get_in_type_error_default_none():
    coll = {'a': 1}
    result = get_in(['a', 'b'], coll)
    assert result is None

def test_get_in_type_error_no_default_raises():
    coll = {'a': 1}
    try:
        get_in(['a', 'b'], coll, no_default=True)
        assert False
    except TypeError:
        assert True

def test_get_in_with_freeze():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice', 'purchase': {'items': ['Apple', 'Orange'], 'costs': [0.50, 1.25]}, 'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'

def test_get_in_default_on_missing_key():
    coll = {'x': 10}
    result = get_in(['y'], coll, default=100)
    assert result == 100

def test_get_in_no_default_on_missing_key_in_nested():
    coll = {'a': {'b': {}}}
    try:
        get_in(['a', 'b', 'c'], coll, no_default=True)
        assert False
    except KeyError:
        assert True


# LLM-generated content at query #5
#--------------------------

def test_predicate_at_line_36_evaluates_to_false():
    coll = {'a': 1}
    keys = ['a']
    result = get_in(keys, coll, default=None, no_default=True)
    assert result == 1


# LLM-generated content at query #6
#--------------------------

def test_predicate_at_line_36_evaluates_to_false():
    result = get_in(['a'], {'a': 1}, no_default=True)
    assert result == 1


# LLM-generated content at query #7
#--------------------------

def test_predicate_at_line_36_evaluates_to_false():
    coll = {'a': 1}
    keys = ['a']
    result = get_in(keys, coll, default=None, no_default=True)
    assert result == 1


# LLM-generated content at query #8
#--------------------------

def test_get_in_nested_dict():
    coll = {'a': {'b': {'c': 1}}}
    result = get_in(['a', 'b', 'c'], coll)
    assert result == 1

def test_get_in_nested_list():
    coll = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    result = get_in([1, 0, 1], coll)
    assert result == 6

def test_get_in_mixed_structures():
    coll = {'a': [{'b': 2}, {'c': 3}]}
    result = get_in(['a', 1, 'c'], coll)
    assert result == 3

def test_get_in_key_not_found_default_none():
    coll = {'a': 1}
    result = get_in(['b'], coll)
    assert result is None

def test_get_in_key_not_found_custom_default():
    coll = {'a': 1}
    result = get_in(['b'], coll, default=0)
    assert result == 0

def test_get_in_no_default_raises_keyerror():
    coll = {'a': 1}
    try:
        get_in(['b'], coll, no_default=True)
        assert False
    except KeyError:
        assert True

def test_get_in_no_default_raises_indexerror():
    coll = [1, 2, 3]
    try:
        get_in([5], coll, no_default=True)
        assert False
    except IndexError:
        assert True

def test_get_in_empty_keys_returns_coll():
    coll = {'a': 1}
    result = get_in([], coll)
    assert result == coll

def test_get_in_default_with_no_default_false():
    coll = {'a': 1}
    result = get_in(['b'], coll, default=10, no_default=False)
    assert result == 10

def test_get_in_type_error_handled():
    coll = {'a': 1}
    result = get_in(['a', 'b'], coll)
    assert result is None

def test_get_in_frozen_dict_example():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice',
                          'purchase': {'items': ['Apple', 'Orange'],
                                       'costs': [0.50, 1.25]},
                          'credit card': '5555-1234-1234-1234'})
    result = get_in(['purchase', 'items', 0], transaction)
    assert result == 'Apple'

def test_get_in_frozen_dict_missing_key_default():
    from pyrsistent import freeze
    transaction = freeze({'name': 'Alice'})
    result = get_in(['purchase', 'total'], transaction, 0)
    assert result == 0


