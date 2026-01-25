####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_choice_call_with_empty_sequence():
    choice = Choice()
    with pytest.raises(ValueError):
        choice(items=[])

def test_choice_call_with_non_sequence():
    choice = Choice()
    with pytest.raises(TypeError):
        choice(items=123)

def test_choice_call_with_negative_length():
    choice = Choice()
    with pytest.raises(ValueError):
        choice(items=['a', 'b', 'c'], length=-1)

def test_choice_call_with_insufficient_unique_elements():
    choice = Choice()
    with pytest.raises(ValueError):
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
    assert all(item in ['a', 'b', 'c'] for item in result)

def test_choice_call_with_string_and_length():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(char in 'abc' for char in result)

def test_choice_call_with_unique_elements():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=2, unique=True)
    assert isinstance(result, list)
    assert len(result) == 2
    assert len(set(result)) == 2
    assert all(item in ['a', 'b', 'c'] for item in result)

def test_choice_call_with_unique_string_elements():
    choice = Choice()
    result = choice(items='aabbbccccddddd', length=4, unique=True)
    assert isinstance(result, str)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert all(char in 'aabbbccccddddd' for char in result)


# LLM-generated content at query #2
#--------------------------

```python
def test_isinstance_items_list():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=2)
    assert isinstance(result, list)


# LLM-generated content at query #3
#--------------------------

```python
def test_choice_single_element():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'])
    assert result in ['a', 'b', 'c']

def test_choice_single_element_from_string():
    choice = Choice()
    result = choice(items='abc')
    assert result in ['a', 'b', 'c']

def test_choice_single_element_from_tuple():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'))
    assert result in ['a', 'b', 'c']

def test_choice_list_of_length_1():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=1)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in ['a', 'b', 'c']

def test_choice_string_of_length_2():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(c in 'abc' for c in result)

def test_choice_tuple_of_length_5():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=5)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert all(c in ['a', 'b', 'c'] for c in result)

def test_choice_unique_elements():
    choice = Choice()
    result = choice(items='aabbbccccddddd', length=4, unique=True)
    assert isinstance(result, str)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert all(c in 'abcd' for c in result)

def test_choice_raises_type_error_for_non_sequence():
    choice = Choice()
    try:
        choice(items=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_choice_raises_value_error_for_empty_sequence():
    choice = Choice()
    try:
        choice(items=[])
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_raises_value_error_for_negative_length():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_raises_value_error_for_insufficient_unique_elements():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=5, unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_isinstance_tuple():
    choice = Choice()
    items = ('a', 'b', 'c')
    result = choice(items=items, length=3)
    assert isinstance(result, tuple)


# LLM-generated content at query #5
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

def test_choice_call_with_list():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(item in ['a', 'b', 'c'] for item in result)

def test_choice_call_with_tuple():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=2)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(item in ['a', 'b', 'c'] for item in result)

def test_choice_call_with_string():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(char in 'abc' for char in result)

def test_choice_call_with_unique_list():
    choice = Choice()
    result = choice(items=['a', 'b', 'c', 'd'], length=3, unique=True)
    assert isinstance(result, list)
    assert len(result) == 3
    assert len(set(result)) == 3
    assert all(item in ['a', 'b', 'c', 'd'] for item in result)

def test_choice_call_with_unique_string():
    choice = Choice()
    result = choice(items='aabbbccccddddd', length=4, unique=True)
    assert isinstance(result, str)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert all(char in 'aabbbccccddddd' for char in result)


# LLM-generated content at query #6
#--------------------------

```python
def test_choice_call_with_single_item():
    choice = Choice()
    result = choice(items=['a'])
    assert result in ['a']

def test_choice_call_with_multiple_items():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'])
    assert result in ['a', 'b', 'c']

def test_choice_call_with_length_one():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=1)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in ['a', 'b', 'c']

def test_choice_call_with_string_items():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(c in 'abc' for c in result)

def test_choice_call_with_tuple_items():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=5)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert all(item in ('a', 'b', 'c') for item in result)

def test_choice_call_with_unique_elements():
    choice = Choice()
    result = choice(items='aabbbccccddddd', length=4, unique=True)
    assert isinstance(result, str)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert all(c in 'abcdd' for c in result)

def test_choice_call_with_non_sequence_items():
    choice = Choice()
    try:
        choice(items=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_choice_call_with_empty_sequence():
    choice = Choice()
    try:
        choice(items=[])
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_call_with_negative_length():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_call_with_insufficient_unique_elements():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=5, unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_choice_returns_tuple_when_input_is_tuple():
    choice = Choice()
    items = ('a', 'b', 'c')
    result = choice(items=items, length=2)
    assert isinstance(result, tuple)


# LLM-generated content at query #8
#--------------------------

```python
def test_choice_call_with_empty_items():
    choice = Choice()
    try:
        choice(items=[])
        assert False, "Expected ValueError for empty items"
    except ValueError:
        assert True

def test_choice_call_with_non_sequence_items():
    choice = Choice()
    try:
        choice(items=123)
        assert False, "Expected TypeError for non-sequence items"
    except TypeError:
        assert True

def test_choice_call_with_negative_length():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
        assert False, "Expected ValueError for negative length"
    except ValueError:
        assert True

def test_choice_call_with_unique_and_insufficient_elements():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=5, unique=True)
        assert False, "Expected ValueError for insufficient unique elements"
    except ValueError:
        assert True

def test_choice_call_with_zero_length():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=0)
    assert isinstance(result, str) and result in ['a', 'b', 'c']

def test_choice_call_with_list_and_length():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=2)
    assert isinstance(result, list) and len(result) == 2 and all(item in ['a', 'b', 'c'] for item in result)

def test_choice_call_with_tuple_and_length():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=2)
    assert isinstance(result, tuple) and len(result) == 2 and all(item in ['a', 'b', 'c'] for item in result)

def test_choice_call_with_string_and_length():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str) and len(result) == 2 and all(c in 'abc' for c in result)

def test_choice_call_with_unique_list():
    choice = Choice()
    result = choice(items=['a', 'b', 'c', 'd'], length=3, unique=True)
    assert isinstance(result, list) and len(result) == 3 and len(set(result)) == 3 and all(item in ['a', 'b', 'c', 'd'] for item in result)

def test_choice_call_with_unique_string():
    choice = Choice()
    result = choice(items='aabbbccccddddd', length=4, unique=True)
    assert isinstance(result, str) and len(result) == 4 and len(set(result)) == 4 and all(c in 'aabbbccccddddd' for c in result)


# LLM-generated content at query #9
#--------------------------

```python
def test_choice_returns_list_when_input_is_list():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=2)
    assert isinstance(result, list)


# LLM-generated content at query #10
#--------------------------

```python
def test_choice_returns_tuple_when_input_is_tuple():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=5)
    assert isinstance(result, tuple)


# LLM-generated content at query #11
#--------------------------

```python
def test_choice_call_with_empty_sequence():
    choice = Choice()
    try:
        choice(items=[])
    except ValueError as e:
        assert str(e) == "**items** must be a non-empty sequence."

def test_choice_call_with_non_sequence():
    choice = Choice()
    try:
        choice(items=123)
    except TypeError as e:
        assert str(e) == "**items** must be non-empty sequence."

def test_choice_call_with_negative_length():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
    except ValueError as e:
        assert str(e) == "**length** should be a positive integer."

def test_choice_call_with_insufficient_unique_elements():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=5, unique=True)
    except ValueError as e:
        assert str(e) == "There are not enough unique elements in **items** to provide the specified **number**."

def test_choice_call_with_zero_length():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=0)
    assert result in ['a', 'b', 'c']

def test_choice_call_with_list():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(item in ['a', 'b', 'c'] for item in result)

def test_choice_call_with_tuple():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=2)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(item in ['a', 'b', 'c'] for item in result)

def test_choice_call_with_string():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(char in 'abc' for char in result)

def test_choice_call_with_unique_list():
    choice = Choice()
    result = choice(items=['a', 'b', 'c', 'd'], length=3, unique=True)
    assert isinstance(result, list)
    assert len(result) == 3
    assert len(set(result)) == 3
    assert all(item in ['a', 'b', 'c', 'd'] for item in result)

def test_choice_call_with_unique_string():
    choice = Choice()
    result = choice(items='aabbbccccddddd', length=4, unique=True)
    assert isinstance(result, str)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert all(char in 'aabbbccccddddd' for char in result)


# LLM-generated content at query #12
#--------------------------

```python
def test_choice_returns_list_when_items_is_list():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=2)
    assert isinstance(result, list)


# LLM-generated content at query #13
#--------------------------

```python
def test_unique_elements_with_insufficient_unique_elements():
    choice = Choice()
    items = 'aabbbccccddddd'
    length = 10
    unique = True
    try:
        choice(items=items, length=length, unique=unique)
        assert False, "Expected ValueError not raised"
    except ValueError as e:
        assert str(e) == "There are not enough unique elements in **items** to provide the specified **number**."


# LLM-generated content at query #14
#--------------------------

```python
def test_isinstance_items_list():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=1)
    assert isinstance(result, list)


# LLM-generated content at query #15
#--------------------------

```python
def test_choice_returns_tuple_when_input_is_tuple():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=5)
    assert isinstance(result, tuple)


# LLM-generated content at query #16
#--------------------------

```python
def test_choice_returns_list_when_input_is_list():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=2)
    assert isinstance(result, list)


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
    result = choice(items=['a', 'b', 'c'], length=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(item in ['a', 'b', 'c'] for item in result)

def test_choice_string_with_length():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(c in 'abc' for c in result)

def test_choice_tuple_with_length():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=3)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(item in ('a', 'b', 'c') for item in result)

def test_choice_unique_elements():
    choice = Choice()
    result = choice(items='aabbbccccddddd', length=4, unique=True)
    assert isinstance(result, str)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert all(c in 'abcde' for c in result)

def test_choice_raises_type_error_for_non_sequence():
    choice = Choice()
    try:
        choice(items=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_choice_raises_value_error_for_empty_sequence():
    choice = Choice()
    try:
        choice(items=[])
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_raises_value_error_for_negative_length():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_raises_value_error_for_insufficient_unique_elements():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=5, unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_choice_call_with_single_item():
    choice = Choice()
    result = choice(items=['a'])
    assert result == 'a'

def test_choice_call_with_empty_sequence():
    choice = Choice()
    try:
        choice(items=[])
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_call_with_non_sequence():
    choice = Choice()
    try:
        choice(items=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_choice_call_with_negative_length():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_call_with_length_zero():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=0)
    assert result in ['a', 'b', 'c']

def test_choice_call_with_unique_true():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=2, unique=True)
    assert len(result) == 2
    assert len(set(result)) == 2

def test_choice_call_with_unique_true_insufficient_elements():
    choice = Choice()
    try:
        choice(items=['a', 'b'], length=3, unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_call_with_list():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=2)
    assert isinstance(result, list)
    assert len(result) == 2

def test_choice_call_with_tuple():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=2)
    assert isinstance(result, tuple)
    assert len(result) == 2

def test_choice_call_with_string():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2


# LLM-generated content at query #3
#--------------------------

```python
def test_choice_returns_list_when_items_is_list():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=3)
    assert isinstance(result, list)


# LLM-generated content at query #4
#--------------------------

```python
def test_choice_returns_list_when_items_is_list():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=2)
    assert isinstance(result, list)


# LLM-generated content at query #5
#--------------------------

```python
def test_choice_returns_tuple_when_input_is_tuple():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=3)
    assert isinstance(result, tuple)


# LLM-generated content at query #6
#--------------------------

```python
def test_choice_with_single_element():
    choice = Choice()
    result = choice(items=['a'])
    assert result == 'a'

def test_choice_with_list_and_length():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(item in ['a', 'b', 'c'] for item in result)

def test_choice_with_string_and_length():
    choice = Choice()
    result = choice(items='abc', length=3)
    assert isinstance(result, str)
    assert len(result) == 3
    assert all(c in 'abc' for c in result)

def test_choice_with_tuple_and_length():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=4)
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert all(item in ('a', 'b', 'c') for item in result)

def test_choice_with_unique_elements():
    choice = Choice()
    result = choice(items='aabbcc', length=3, unique=True)
    assert isinstance(result, str)
    assert len(result) == 3
    assert len(set(result)) == 3

def test_choice_raises_type_error_for_non_sequence():
    choice = Choice()
    try:
        choice(items=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_choice_raises_value_error_for_empty_sequence():
    choice = Choice()
    try:
        choice(items=[])
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_raises_value_error_for_negative_length():
    choice = Choice()
    try:
        choice(items=['a', 'b'], length=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_raises_value_error_for_insufficient_unique_elements():
    choice = Choice()
    try:
        choice(items=['a', 'b'], length=3, unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #7
#--------------------------

```python
def test_choice_call_with_single_element():
    choice = Choice()
    result = choice(items=['a'])
    assert result == 'a'

def test_choice_call_with_length_one():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=1)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in ['a', 'b', 'c']

def test_choice_call_with_string_items():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(c in 'abc' for c in result)

def test_choice_call_with_tuple_items():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=5)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert all(item in ('a', 'b', 'c') for item in result)

def test_choice_call_with_unique_elements():
    choice = Choice()
    result = choice(items='aabbbccccddddd', length=4, unique=True)
    assert isinstance(result, str)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert all(c in 'abc' for c in result)

def test_choice_call_with_non_sequence_items():
    choice = Choice()
    try:
        choice(items=123)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_choice_call_with_empty_items():
    choice = Choice()
    try:
        choice(items=[])
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_call_with_negative_length():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=-1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_choice_call_with_insufficient_unique_elements():
    choice = Choice()
    try:
        choice(items=['a', 'b', 'c'], length=5, unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #8
#--------------------------

```python
def test_tuple_input_returns_tuple():
    choice = Choice()
    items = ('a', 'b', 'c')
    result = choice(items=items, length=5)
    assert isinstance(result, tuple)


# LLM-generated content at query #9
#--------------------------

```python
def test_isinstance_items_list():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=1)
    assert isinstance(result, list)


# LLM-generated content at query #10
#--------------------------

```python
def test_choice_returns_tuple_when_items_is_tuple():
    choice = Choice()
    items = ('a', 'b', 'c')
    result = choice(items=items, length=3)
    assert isinstance(result, tuple)


# LLM-generated content at query #11
#--------------------------

```python
def test_choice_returns_tuple_when_input_is_tuple():
    choice = Choice()
    items = ('a', 'b', 'c')
    result = choice(items=items, length=5)
    assert isinstance(result, tuple)


# LLM-generated content at query #12
#--------------------------

```python
def test_choice_returns_list_when_items_is_list():
    choice = Choice()
    items = ['a', 'b', 'c']
    result = choice(items=items, length=2)
    assert isinstance(result, list)


# LLM-generated content at query #13
#--------------------------

```python
def test_choice_with_empty_sequence():
    choice = Choice()
    with pytest.raises(ValueError):
        choice(items=[])

def test_choice_with_non_sequence():
    choice = Choice()
    with pytest.raises(TypeError):
        choice(items=123)

def test_choice_with_negative_length():
    choice = Choice()
    with pytest.raises(ValueError):
        choice(items=['a', 'b', 'c'], length=-1)

def test_choice_with_insufficient_unique_elements():
    choice = Choice()
    with pytest.raises(ValueError):
        choice(items=['a', 'b', 'c'], length=5, unique=True)

def test_choice_single_element():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'])
    assert result in ['a', 'b', 'c']

def test_choice_with_length_one():
    choice = Choice()
    result = choice(items=['a', 'b', 'c'], length=1)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in ['a', 'b', 'c']

def test_choice_with_string():
    choice = Choice()
    result = choice(items='abc', length=2)
    assert isinstance(result, str)
    assert len(result) == 2
    assert all(c in 'abc' for c in result)

def test_choice_with_tuple():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=5)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert all(item in ('a', 'b', 'c') for item in result)

def test_choice_with_unique_elements():
    choice = Choice()
    result = choice(items='aabbbccccddddd', length=4, unique=True)
    assert isinstance(result, str)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert all(c in 'aabbbccccddddd' for c in result)


# LLM-generated content at query #14
#--------------------------

```python
def test_choice_returns_tuple_when_input_is_tuple():
    choice = Choice()
    result = choice(items=('a', 'b', 'c'), length=5)
    assert isinstance(result, tuple)


