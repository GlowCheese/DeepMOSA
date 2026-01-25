####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_choice_single_element():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'])
    assert result in ['a', 'b', 'c']

def test_choice_list_with_length():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=1)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in ['a', 'b', 'c']

def test_choice_string_with_length():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(c in 'abc' for c in result)

def test_choice_tuple_with_length():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=5)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert all(c in ('a', 'b', 'c') for c in result)

def test_choice_unique_elements():
    choice = Choice()
    result = choice(items='aabbbccccddddd', length=4, unique=True)
    assert isinstance(result, str)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert all(c in 'aabbbccccddddd' for c in result)

def test_choice_non_sequence_raises_type_error():
    choice = Choice()
    try:
        choice(items=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_choice_empty_sequence_raises_value_error():
    choice = Choice()
    try:
        choice(items=[])
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_negative_length_raises_value_error():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_insufficient_unique_elements_raises_value_error():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=5, unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        pass


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_choice_call_with_empty_sequence():
    choice = Choice()
    with pytest.raises(ValueError, match="**items** must be a non-empty sequence."):
        choice(items=[])

def test_choice_call_with_non_sequence():
    choice = Choice()
    with pytest.raises(TypeError, match="**items** must be non-empty sequence."):
        choice(items=123)

def test_choice_call_with_negative_length():
    choice = Choice()
    with pytest.raises(ValueError, match="**length** should be a positive integer."):
        choice(items=['a', 'b', 'c'], length=-1)

def test_choice_call_with_insufficient_unique_elements():
    choice = Choice()
    with pytest.raises(ValueError, match="There are not enough unique elements in **items** to provide the specified **number**."):
        choice(items=['a', 'b', 'c'], length=5, unique=True)

def test_choice_call_with_zero_length():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=0)
    assert result in ['a', 'b', 'c']

def test_choice_call_with_list_and_length():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(item in ['a', 'b', 'c'] for item in result)

def test_choice_call_with_tuple_and_length():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=2)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(item in ('a', 'b', 'c') for item in result)

def test_choice_call_with_string_and_length():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(char in 'abc' for char in result)

def test_choice_call_with_unique_elements():
    choice = Choice()
    result = choice(items=['a', 'b', 'c', 'd', 'e'], length=3, unique=True)
    assert isinstance(result, list)
    assert len(result) == 3
    assert len(set(result)) == 3
    assert all(item in ['a', 'b', 'c', 'd', 'e'] for item in result)

def test_choice_call_with_unique_string_elements():
    choice = Choice()
    result = choice(items='abcde', length=3, unique=True)
    assert isinstance(result, str)
    assert len(result) == 3
    assert len(set(result)) == 3
    assert all(char in 'abcde' for char in result)


