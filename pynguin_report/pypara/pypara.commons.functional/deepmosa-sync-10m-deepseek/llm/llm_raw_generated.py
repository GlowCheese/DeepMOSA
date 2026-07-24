####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_chunk_empty_list():
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []

def test_chunk_exact_division():
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    assert list(chunk([1, 2, 3, 4], 4)) == [[1, 2, 3, 4]]

def test_chunk_uneven_division():
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunk([1, 2, 3, 4, 5, 6, 7], 3)) == [[1, 2, 3], [4, 5, 6], [7]]

def test_chunk_larger_than_list():
    assert list(chunk([1, 2, 3], 5)) == [[1, 2, 3]]


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_empty_list_with_chunk_size_1():
    result = list(chunk([], 1))
    assert result == []

def test_empty_list_with_chunk_size_2():
    result = list(chunk([], 2))
    assert result == []

def test_list_with_even_length_and_chunk_size_1():
    result = list(chunk([1, 2, 3, 4], 1))
    assert result == [[1], [2], [3], [4]]

def test_list_with_even_length_and_chunk_size_2():
    result = list(chunk([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]

def test_list_with_odd_length_and_chunk_size_2():
    result = list(chunk([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

def test_list_with_length_less_than_chunk_size():
    result = list(chunk([1, 2], 3))
    assert result == [[1, 2]]

def test_list_with_single_element_and_chunk_size_1():
    result = list(chunk([1], 1))
    assert result == [[1]]

def test_list_with_multiple_elements_and_large_chunk_size():
    result = list(chunk([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4))
    assert result == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]


