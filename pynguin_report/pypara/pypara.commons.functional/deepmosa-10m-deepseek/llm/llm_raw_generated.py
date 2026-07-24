####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_empty_list():
    result = list(chunk([], 1))
    assert result == []

def test_empty_list_with_larger_chunk_size():
    result = list(chunk([], 2))
    assert result == []

def test_single_element_chunks():
    result = list(chunk([1, 2, 3, 4], 1))
    assert result == [[1], [2], [3], [4]]

def test_even_chunks():
    result = list(chunk([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]

def test_uneven_chunks():
    result = list(chunk([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

def test_larger_chunk_size_than_list():
    result = list(chunk([1, 2, 3], 5))
    assert result == [[1, 2, 3]]

def test_exact_chunk_size():
    result = list(chunk([1, 2, 3, 4], 4))
    assert result == [[1, 2, 3, 4]]

def test_chunk_size_zero():
    result = list(chunk([1, 2, 3], 0))
    assert result == []

def test_chunk_size_negative():
    result = list(chunk([1, 2, 3], -1))
    assert result == []


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_chunk_empty_list():
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []

def test_chunk_single_element_chunks():
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]

def test_chunk_even_chunks():
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

def test_chunk_uneven_chunks():
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

def test_chunk_larger_chunk_size():
    assert list(chunk([1, 2, 3, 4, 5], 10)) == [[1, 2, 3, 4, 5]]

def test_chunk_zero_chunk_size():
    assert list(chunk([1, 2, 3, 4], 0)) == []

def test_chunk_negative_chunk_size():
    assert list(chunk([1, 2, 3, 4], -1)) == []


