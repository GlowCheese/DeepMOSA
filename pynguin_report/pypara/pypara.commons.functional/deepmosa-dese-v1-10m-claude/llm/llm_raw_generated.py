####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_chunk_empty_list_size_1():
    result = list(chunk([], 1))
    assert result == []


def test_chunk_empty_list_size_2():
    result = list(chunk([], 2))
    assert result == []


def test_chunk_four_items_size_1():
    result = list(chunk([1, 2, 3, 4], 1))
    assert result == [[1], [2], [3], [4]]


def test_chunk_four_items_size_2():
    result = list(chunk([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]


def test_chunk_five_items_size_2():
    result = list(chunk([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]


def test_chunk_single_item():
    result = list(chunk([1], 1))
    assert result == [[1]]


def test_chunk_size_larger_than_list():
    result = list(chunk([1, 2, 3], 5))
    assert result == [[1, 2, 3]]


def test_chunk_three_items_size_3():
    result = list(chunk([1, 2, 3], 3))
    assert result == [[1, 2, 3]]


def test_chunk_string_list():
    result = list(chunk(['a', 'b', 'c', 'd'], 2))
    assert result == [['a', 'b'], ['c', 'd']]


def test_chunk_six_items_size_3():
    result = list(chunk([1, 2, 3, 4, 5, 6], 3))
    assert result == [[1, 2, 3], [4, 5, 6]]


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_chunk_empty_list_size_1():
    result = list(chunk([], 1))
    assert result == []

def test_chunk_empty_list_size_2():
    result = list(chunk([], 2))
    assert result == []

def test_chunk_four_elements_size_1():
    result = list(chunk([1, 2, 3, 4], 1))
    assert result == [[1], [2], [3], [4]]

def test_chunk_four_elements_size_2():
    result = list(chunk([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]

def test_chunk_five_elements_size_2():
    result = list(chunk([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

def test_chunk_single_element():
    result = list(chunk([1], 1))
    assert result == [[1]]

def test_chunk_size_larger_than_list():
    result = list(chunk([1, 2, 3], 5))
    assert result == [[1, 2, 3]]

def test_chunk_with_strings():
    result = list(chunk(['a', 'b', 'c', 'd'], 2))
    assert result == [['a', 'b'], ['c', 'd']]

def test_chunk_size_equals_list_length():
    result = list(chunk([1, 2, 3, 4], 4))
    assert result == [[1, 2, 3, 4]]

def test_chunk_many_small_chunks():
    result = list(chunk([1, 2, 3, 4, 5, 6], 1))
    assert result == [[1], [2], [3], [4], [5], [6]]


