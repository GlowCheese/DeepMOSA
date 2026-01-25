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


def test_chunk_single_size_chunks():
    result = list(chunk([1, 2, 3, 4], 1))
    assert result == [[1], [2], [3], [4]]


def test_chunk_equal_division():
    result = list(chunk([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]


def test_chunk_unequal_division():
    result = list(chunk([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]


def test_chunk_large_chunk_size():
    result = list(chunk([1, 2, 3], 10))
    assert result == [[1, 2, 3]]


def test_chunk_single_element():
    result = list(chunk([1], 1))
    assert result == [[1]]


def test_chunk_strings():
    result = list(chunk(['a', 'b', 'c', 'd'], 2))
    assert result == [['a', 'b'], ['c', 'd']]


def test_chunk_mixed_types():
    result = list(chunk([1, 'a', 2.5, None, True], 2))
    assert result == [[1, 'a'], [2.5, None], [True]]


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

def test_chunk_list_size_1():
    result = list(chunk([1, 2, 3, 4], 1))
    assert result == [[1], [2], [3], [4]]

def test_chunk_list_size_2_even():
    result = list(chunk([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]

def test_chunk_list_size_2_uneven():
    result = list(chunk([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

def test_chunk_list_size_3():
    result = list(chunk([1, 2, 3, 4, 5, 6, 7], 3))
    assert result == [[1, 2, 3], [4, 5, 6], [7]]

def test_chunk_list_size_larger_than_list():
    result = list(chunk([1, 2, 3], 5))
    assert result == [[1, 2, 3]]

def test_chunk_single_element():
    result = list(chunk([1], 1))
    assert result == [[1]]

def test_chunk_string_list():
    result = list(chunk(['a', 'b', 'c', 'd'], 2))
    assert result == [['a', 'b'], ['c', 'd']]


