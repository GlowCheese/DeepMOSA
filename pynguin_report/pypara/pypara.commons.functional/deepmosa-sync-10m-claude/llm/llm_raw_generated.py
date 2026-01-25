####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_chunk_empty_list_chunk_size_1():
    result = list(chunk([], 1))
    assert result == []

def test_chunk_empty_list_chunk_size_2():
    result = list(chunk([], 2))
    assert result == []

def test_chunk_list_of_4_chunk_size_1():
    result = list(chunk([1, 2, 3, 4], 1))
    assert result == [[1], [2], [3], [4]]

def test_chunk_list_of_4_chunk_size_2():
    result = list(chunk([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]

def test_chunk_list_of_5_chunk_size_2():
    result = list(chunk([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

def test_chunk_single_element():
    result = list(chunk([1], 1))
    assert result == [[1]]

def test_chunk_single_element_larger_chunk_size():
    result = list(chunk([1], 5))
    assert result == [[1]]

def test_chunk_list_of_strings():
    result = list(chunk(['a', 'b', 'c', 'd'], 2))
    assert result == [['a', 'b'], ['c', 'd']]

def test_chunk_chunk_size_larger_than_list():
    result = list(chunk([1, 2, 3], 10))
    assert result == [[1, 2, 3]]

def test_chunk_chunk_size_equals_list_length():
    result = list(chunk([1, 2, 3, 4], 4))
    assert result == [[1, 2, 3, 4]]


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_chunk_empty_list():
    result = list(chunk([], 1))
    assert result == []

def test_chunk_empty_list_size_two():
    result = list(chunk([], 2))
    assert result == []

def test_chunk_single_element_chunks():
    result = list(chunk([1, 2, 3, 4], 1))
    assert result == [[1], [2], [3], [4]]

def test_chunk_two_element_chunks():
    result = list(chunk([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]

def test_chunk_uneven_division():
    result = list(chunk([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

def test_chunk_size_larger_than_list():
    result = list(chunk([1, 2, 3], 5))
    assert result == [[1, 2, 3]]

def test_chunk_single_element_list():
    result = list(chunk([1], 1))
    assert result == [[1]]

def test_chunk_size_equals_list_length():
    result = list(chunk([1, 2, 3], 3))
    assert result == [[1, 2, 3]]

def test_chunk_with_strings():
    result = list(chunk(['a', 'b', 'c', 'd'], 2))
    assert result == [['a', 'b'], ['c', 'd']]

def test_chunk_with_mixed_types():
    result = list(chunk([1, 'a', 2, 'b', 3], 2))
    assert result == [[1, 'a'], [2, 'b'], [3]]


