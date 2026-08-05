####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_chunk_empty_list():
    assert list(chunk([], 1)) == []

def test_chunk_empty_list_different_size():
    assert list(chunk([], 2)) == []

def test_chunk_single_element_chunks():
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]

def test_chunk_even_split():
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

def test_chunk_uneven_split():
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

def test_chunk_large_n():
    assert list(chunk([1, 2, 3], 10)) == [[1, 2, 3]]

def test_chunk_strings():
    assert list(chunk(['a', 'b', 'c'], 2)) == [['a', 'b'], ['c']]
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_chunk_empty_list_size_one():
    assert list(chunk([], 1)) == []

def test_chunk_empty_list_size_two():
    assert list(chunk([], 2)) == []

def test_chunk_single_elements():
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]

def test_chunk_even_split():
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

def test_chunk_uneven_split():
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

def test_chunk_n_larger_than_list():
    assert list(chunk([1, 2, 3], 10)) == [[1, 2, 3]]

def test_chunk_n_equals_list_length():
    assert list(chunk([1, 2, 3], 3)) == [[1, 2, 3]]
```


