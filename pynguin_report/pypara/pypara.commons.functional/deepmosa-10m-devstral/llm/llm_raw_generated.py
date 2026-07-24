####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_chunk_empty_list():
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []

def test_chunk_single_element_chunks():
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]

def test_chunk_even_size():
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

def test_chunk_odd_size():
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_chunk_empty_list():
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []

def test_chunk_single_element_chunks():
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]

def test_chunk_even_size():
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

def test_chunk_odd_size():
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


