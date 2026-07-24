####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_chunk_empty_list():
    result = list(chunk([], 1))
    assert result == []


def test_chunk_empty_list_larger_n():
    result = list(chunk([], 2))
    assert result == []


def test_chunk_exact_division():
    result = list(chunk([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]


def test_chunk_single_element_chunks():
    result = list(chunk([1, 2, 3, 4], 1))
    assert result == [[1], [2], [3], [4]]


def test_chunk_remainder():
    result = list(chunk([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]


def test_chunk_n_larger_than_list():
    result = list(chunk([1, 2, 3], 5))
    assert result == [[1, 2, 3]]


def test_chunk_n_equals_list_length():
    result = list(chunk([1, 2, 3], 3))
    assert result == [[1, 2, 3]]


def test_chunk_string_list():
    result = list(chunk(['a', 'b', 'c', 'd'], 2))
    assert result == [['a', 'b'], ['c', 'd']]


def test_chunk_mixed_types():
    result = list(chunk([1, 'a', 2.5, True], 2))
    assert result == [[1, 'a'], [2.5, True]]


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_chunk_empty_list():
    result = list(chunk([], 1))
    assert result == []

def test_chunk_empty_list_with_size_two():
    result = list(chunk([], 2))
    assert result == []

def test_chunk_list_size_one():
    result = list(chunk([1, 2, 3, 4], 1))
    assert result == [[1], [2], [3], [4]]

def test_chunk_list_even_chunks():
    result = list(chunk([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]

def test_chunk_list_uneven_chunks():
    result = list(chunk([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]

def test_chunk_list_larger_chunk_size():
    result = list(chunk([1, 2, 3, 4, 5], 10))
    assert result == [[1, 2, 3, 4, 5]]

def test_chunk_list_exact_chunk_size():
    result = list(chunk([1, 2, 3, 4, 5, 6], 3))
    assert result == [[1, 2, 3], [4, 5, 6]]

def test_chunk_list_single_element():
    result = list(chunk([42], 1))
    assert result == [[42]]

def test_chunk_list_single_element_larger_chunk():
    result = list(chunk([42], 5))
    assert result == [[42]]


