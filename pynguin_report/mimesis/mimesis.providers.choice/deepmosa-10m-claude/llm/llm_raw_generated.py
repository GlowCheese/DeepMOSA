####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_call_returns_single_element_when_length_is_zero():
    from mimesis import Choice
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=0)
    assert result in items


def test_call_returns_list_when_items_is_list():
    from mimesis import Choice
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=2)
    assert isinstance(result, list)
    assert len(result) == 2


def test_call_returns_tuple_when_items_is_tuple():
    from mimesis import Choice
    choice = Choice()
    items = ('a', 'b', 'c')
    result = choice(items=items, length=2)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_call_returns_string_when_items_is_string():
    from mimesis import Choice
    choice = Choice()
    items = 'abc'
    result = choice(items=items, length=2)
    assert isinstance(result, str)
    assert len(result) == 2


def test_call_with_unique_elements():
    from mimesis import Choice
    choice = Choice()
    items = ['a', 'b', 'c', 'd', 'e']
    result = choice(items=items, length=3, unique=True)
    assert isinstance(result, list)
    assert len(result) == 3
    assert len(set(result)) == 3


def test_call_with_unique_string():
    from mimesis import Choice
    choice = Choice()
    items = 'abcde'
    result = choice(items=items, length=3, unique=True)
    assert isinstance(result, str)
    assert len(result) == 3
    assert len(set(result)) == 3


def test_call_raises_type_error_for_non_sequence():
    from mimesis import Choice
    choice = Choice()
    try:
        choice(items=123, length=1)
        assert False, "Expected TypeError"
    except TypeError:
        assert True


def test_call_raises_value_error_for_empty_sequence():
    from mimesis import Choice
    choice = Choice()
    try:
        choice(items=[], length=1)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_call_raises_value_error_for_negative_length():
    from mimesis import Choice
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_call_raises_value_error_for_insufficient_unique_elements():
    from mimesis import Choice
    choice = Choice()
    try:
        choice(items=['a', 'b'], length=5, unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_call_with_length_greater_than_items_without_unique():
    from mimesis import Choice
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=5)
    assert isinstance(result, list)
    assert len(result) == 5


def test_call_with_duplicate_items_in_sequence():
    from mimesis import Choice
    choice = Choice()
    items = ['a', 'a', 'b', 'b', 'c']
    result = choice(items=items, length=3)
    assert isinstance(result, list)
    assert len(result) == 3


def test_call_returns_all_elements_from_items_when_length_equals_items_length():
    from mimesis import Choice
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=3, unique=True)
    assert isinstance(result, list)
    assert len(result) == 3
    assert set(result) == set(items)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_call_with_list_no_length():
    from mimesis import Choice
    choice = Choice()
    result = choice(items=['a', 'b', 'c'])
    assert result in ['a', 'b', 'c']


def test_call_with_list_and_length():
    from mimesis import Choice
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(item in ['a', 'b', 'c'] for item in result)


def test_call_with_tuple_and_length():
    from mimesis import Choice
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=3)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(item in ('a', 'b', 'c') for item in result)


def test_call_with_string_and_length():
    from mimesis import Choice
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(char in 'abc' for char in result)


def test_call_with_unique_true():
    from mimesis import Choice
    choice = Choice()
    result = choice(items=['a', 'b', 'c', 'd', 'e'], length=3, unique=True)
    assert isinstance(result, list)
    assert len(result) == 3
    assert len(set(result)) == 3


def test_call_with_string_unique():
    from mimesis import Choice
    choice = Choice()
    result = choice(items='aabbbccccddddd', length=4, unique=True)
    assert isinstance(result, str)
    assert len(result) == 4
    assert len(set(result)) == 4


def test_call_raises_type_error_for_non_sequence():
    from mimesis import Choice
    choice = Choice()
    try:
        choice(items=123)
        assert False, "Should raise TypeError"
    except TypeError:
        pass


def test_call_raises_value_error_for_empty_sequence():
    from mimesis import Choice
    choice = Choice()
    try:
        choice(items=[])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_call_raises_value_error_for_negative_length():
    from mimesis import Choice
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_call_raises_value_error_for_insufficient_unique_elements():
    from mimesis import Choice
    choice = Choice()
    try:
        choice(items=['a', 'b'], length=5, unique=True)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_call_with_length_zero():
    from mimesis import Choice
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=0)
    assert result in ['a', 'b', 'c']


def test_call_returns_list_for_list_input():
    from mimesis import Choice
    choice = Choice()
    result = choice(items=['x', 'y', 'z'], length=2)
    assert isinstance(result, list)


def test_call_returns_tuple_for_tuple_input():
    from mimesis import Choice
    choice = Choice()
    result = choice(items=('x', 'y', 'z'), length=2)
    assert isinstance(result, tuple)


def test_call_returns_string_for_string_input():
    from mimesis import Choice
    choice = Choice()
    result = choice(items='xyz', length=2)
    assert isinstance(result, str)


