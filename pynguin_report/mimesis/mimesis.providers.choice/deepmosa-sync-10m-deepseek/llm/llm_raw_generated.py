####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_choice_with_list_and_no_length():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items)
    assert result in items


def test_choice_with_list_and_length_one():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=1)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in items


def test_choice_with_string_and_length():
    choice = Choice()
    items = 'abc'
    result = choice(items=items, length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(char in items for char in result)


def test_choice_with_tuple_and_length():
    choice = Choice()
    items = ('a', 'b', 'c')
    result = choice(items=items, length=5)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert all(element in items for element in result)


def test_choice_with_unique_true():
    choice = Choice()
    items = 'aabbbccccddddd'
    result = choice(items=items, length=4, unique=True)
    assert isinstance(result, str)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert all(char in items for char in result)


def test_choice_raises_type_error_for_non_sequence():
    choice = Choice()
    try:
        choice(items=123)
        assert False
    except TypeError:
        assert True


def test_choice_raises_value_error_for_empty_sequence():
    choice = Choice()
    try:
        choice(items=[])
        assert False
    except ValueError:
        assert True


def test_choice_raises_value_error_for_negative_length():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
        assert False
    except ValueError:
        assert True


def test_choice_raises_value_error_for_insufficient_unique_elements():
    choice = Choice()
    try:
        choice(items='aa', length=3, unique=True)
        assert False
    except ValueError:
        assert True


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_choice_with_list_and_no_length():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items)
    assert result in items


def test_choice_with_list_and_length_one():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=1)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in items


def test_choice_with_string_and_length():
    choice = Choice()
    items = 'abc'
    length = 2
    result = choice(items=items, length=length)
    assert isinstance(result, str)
    assert len(result) == length
    assert all(char in items for char in result)


def test_choice_with_tuple_and_length():
    choice = Choice()
    items = ('a', 'b', 'c')
    length = 5
    result = choice(items=items, length=length)
    assert isinstance(result, tuple)
    assert len(result) == length
    assert all(element in items for element in result)


def test_choice_with_unique_elements():
    choice = Choice()
    items = 'aabbbccccddddd'
    length = 4
    result = choice(items=items, length=length, unique=True)
    assert isinstance(result, str)
    assert len(result) == length
    assert len(set(result)) == length
    assert all(char in items for char in result)


def test_choice_raises_type_error_for_non_sequence():
    choice = Choice()
    try:
        choice(items=123)
        assert False
    except TypeError:
        assert True


def test_choice_raises_value_error_for_empty_sequence():
    choice = Choice()
    try:
        choice(items=[])
        assert False
    except ValueError:
        assert True


def test_choice_raises_value_error_for_negative_length():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
        assert False
    except ValueError:
        assert True


def test_choice_raises_value_error_for_insufficient_unique_elements():
    choice = Choice()
    items = ['a', 'a', 'b']
    length = 3
    unique = True
    try:
        choice(items=items, length=length, unique=unique)
        assert False
    except ValueError:
        assert True


