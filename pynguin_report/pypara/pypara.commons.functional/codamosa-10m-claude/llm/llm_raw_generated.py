####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_chunk():
    # Test with empty list
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []
    assert list(chunk([], 5)) == []
    
    # Test with single element
    assert list(chunk([1], 1)) == [[1]]
    assert list(chunk([1], 2)) == [[1]]
    
    # Test with exact division
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    assert list(chunk([1, 2, 3, 4, 5, 6], 3)) == [[1, 2, 3], [4, 5, 6]]
    
    # Test with remainder
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunk([1, 2, 3, 4, 5], 3)) == [[1, 2, 3], [4, 5]]
    assert list(chunk([1, 2, 3], 4)) == [[1, 2, 3]]
    
    # Test with different data types
    assert list(chunk(['a', 'b', 'c', 'd'], 2)) == [['a', 'b'], ['c', 'd']]
    assert list(chunk([1.5, 2.5, 3.5, 4.5], 2)) == [[1.5, 2.5], [3.5, 4.5]]
    
    # Test with large chunk size
    assert list(chunk([1, 2, 3], 100)) == [[1, 2, 3]]
    
    # Test with chunk size of 1
    assert list(chunk([1, 2, 3], 1)) == [[1], [2], [3]]
    
    # Test that chunks are lists
    chunks = list(chunk([1, 2, 3, 4], 2))
    assert all(isinstance(chunk_item, list) for chunk_item in chunks)


# LLM-generated content at query #2
#--------------------------

def test_chunk():
    # Test empty list
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []
    assert list(chunk([], 5)) == []
    
    # Test chunk size 1
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]
    assert list(chunk([1], 1)) == [[1]]
    assert list(chunk(['a', 'b', 'c'], 1)) == [['a'], ['b'], ['c']]
    
    # Test chunk size 2
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    assert list(chunk([1, 2], 2)) == [[1, 2]]
    assert list(chunk([1], 2)) == [[1]]
    
    # Test uneven chunks
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunk([1, 2, 3, 4, 5], 3)) == [[1, 2, 3], [4, 5]]
    assert list(chunk([1, 2, 3], 2)) == [[1, 2], [3]]
    
    # Test chunk size larger than list
    assert list(chunk([1, 2, 3], 5)) == [[1, 2, 3]]
    assert list(chunk([1], 10)) == [[1]]
    
    # Test with different types
    assert list(chunk(['a', 'b', 'c', 'd'], 2)) == [['a', 'b'], ['c', 'd']]
    assert list(chunk([1.5, 2.5, 3.5], 2)) == [[1.5, 2.5], [3.5]]
    
    # Test single element
    assert list(chunk([42], 1)) == [[42]]
    assert list(chunk([42], 5)) == [[42]]


# LLM-generated content at query #3
#--------------------------

def test_chunk():
    # Test empty list
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []
    
    # Test single chunk size
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]
    
    # Test even division
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    
    # Test uneven division
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    
    # Test chunk size larger than list
    assert list(chunk([1, 2, 3], 5)) == [[1, 2, 3]]
    
    # Test chunk size equal to list length
    assert list(chunk([1, 2, 3], 3)) == [[1, 2, 3]]
    
    # Test single element list
    assert list(chunk([1], 1)) == [[1]]
    
    # Test with strings
    assert list(chunk(['a', 'b', 'c', 'd'], 2)) == [['a', 'b'], ['c', 'd']]
    
    # Test with larger chunk size
    assert list(chunk([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    # Test with larger chunk size with remainder
    assert list(chunk([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)) == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]


# LLM-generated content at query #4
#--------------------------

def test_chunk():
    # Test empty list
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []
    assert list(chunk([], 5)) == []
    
    # Test chunk size of 1
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]
    
    # Test chunk size of 2
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    
    # Test uneven chunks
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunk([1, 2, 3, 4, 5], 3)) == [[1, 2, 3], [4, 5]]
    
    # Test chunk size larger than list
    assert list(chunk([1, 2, 3], 5)) == [[1, 2, 3]]
    
    # Test single element
    assert list(chunk([1], 1)) == [[1]]
    assert list(chunk([1], 2)) == [[1]]
    
    # Test with strings
    assert list(chunk(['a', 'b', 'c', 'd'], 2)) == [['a', 'b'], ['c', 'd']]
    
    # Test with different types
    assert list(chunk([1, 'a', 2.5, None], 2)) == [[1, 'a'], [2.5, None]]
    
    # Test chunk size equal to list length
    assert list(chunk([1, 2, 3], 3)) == [[1, 2, 3]]


# LLM-generated content at query #5
#--------------------------

def test_chunk():
    # Test empty list
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []
    
    # Test chunk size 1
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]
    
    # Test chunk size 2 with even length list
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    
    # Test chunk size 2 with odd length list
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    
    # Test chunk size larger than list
    assert list(chunk([1, 2, 3], 5)) == [[1, 2, 3]]
    
    # Test chunk size equals list length
    assert list(chunk([1, 2, 3], 3)) == [[1, 2, 3]]
    
    # Test with single element
    assert list(chunk([1], 1)) == [[1]]
    assert list(chunk([1], 2)) == [[1]]
    
    # Test with strings
    assert list(chunk(['a', 'b', 'c', 'd'], 2)) == [['a', 'b'], ['c', 'd']]
    
    # Test with larger chunk size
    assert list(chunk([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert list(chunk([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)) == [[1, 2, 3, 4], [5, 6, 7, 8], [9]]


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_chunk():
    # Test empty list
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []
    
    # Test chunk size of 1
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]
    
    # Test chunk size of 2 with even length list
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    
    # Test chunk size of 2 with odd length list
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    
    # Test chunk size larger than list
    assert list(chunk([1, 2, 3], 5)) == [[1, 2, 3]]
    
    # Test chunk size equal to list length
    assert list(chunk([1, 2, 3], 3)) == [[1, 2, 3]]
    
    # Test chunk size of 3
    assert list(chunk([1, 2, 3, 4, 5, 6, 7], 3)) == [[1, 2, 3], [4, 5, 6], [7]]
    
    # Test with string list
    assert list(chunk(['a', 'b', 'c', 'd'], 2)) == [['a', 'b'], ['c', 'd']]
    
    # Test single element list
    assert list(chunk([1], 1)) == [[1]]
    assert list(chunk([1], 2)) == [[1]]


# LLM-generated content at query #2
#--------------------------

```python
def test_chunk():
    # Test empty list
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []
    
    # Test single element chunks
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]
    
    # Test even division
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    
    # Test uneven division
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    
    # Test chunk size larger than list
    assert list(chunk([1, 2, 3], 5)) == [[1, 2, 3]]
    
    # Test chunk size of 1
    assert list(chunk([1], 1)) == [[1]]
    
    # Test larger list with various chunk sizes
    assert list(chunk([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert list(chunk([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)) == [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
    
    # Test with string elements
    assert list(chunk(['a', 'b', 'c', 'd'], 2)) == [['a', 'b'], ['c', 'd']]
    
    # Test that result is iterable
    result = chunk([1, 2, 3, 4], 2)
    assert hasattr(result, '__iter__')


# LLM-generated content at query #3
#--------------------------

def test_chunk():
    # Test empty list
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []
    
    # Test chunk size of 1
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]
    
    # Test chunk size equal to list length
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    
    # Test chunk size larger than remaining elements
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    
    # Test single element list
    assert list(chunk([1], 1)) == [[1]]
    assert list(chunk([1], 2)) == [[1]]
    
    # Test chunk size larger than list
    assert list(chunk([1, 2, 3], 5)) == [[1, 2, 3]]
    
    # Test with string elements
    assert list(chunk(['a', 'b', 'c', 'd'], 2)) == [['a', 'b'], ['c', 'd']]
    
    # Test with mixed types
    assert list(chunk([1, 'a', 2, 'b', 3], 2)) == [[1, 'a'], [2, 'b'], [3]]
    
    # Test chunk size of 3
    assert list(chunk([1, 2, 3, 4, 5, 6, 7], 3)) == [[1, 2, 3], [4, 5, 6], [7]]


# LLM-generated content at query #4
#--------------------------

def test_chunk():
    # Test empty list
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []
    assert list(chunk([], 5)) == []
    
    # Test chunk size of 1
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]
    assert list(chunk(['a', 'b', 'c'], 1)) == [['a'], ['b'], ['c']]
    
    # Test chunk size of 2
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    
    # Test chunk size larger than list
    assert list(chunk([1, 2, 3], 5)) == [[1, 2, 3]]
    assert list(chunk([1], 10)) == [[1]]
    
    # Test chunk size equal to list length
    assert list(chunk([1, 2, 3, 4], 4)) == [[1, 2, 3, 4]]
    
    # Test with different data types
    assert list(chunk(['a', 'b', 'c', 'd', 'e'], 2)) == [['a', 'b'], ['c', 'd'], ['e']]
    assert list(chunk([1.5, 2.5, 3.5, 4.5], 2)) == [[1.5, 2.5], [3.5, 4.5]]
    
    # Test with single element
    assert list(chunk([42], 1)) == [[42]]
    assert list(chunk([42], 2)) == [[42]]
    
    # Test larger list
    assert list(chunk([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert list(chunk([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)) == [[1, 2, 3, 4], [5, 6, 7, 8], [9]]


# LLM-generated content at query #5
#--------------------------

def test_chunk():
    # Test empty list
    assert list(chunk([], 1)) == []
    assert list(chunk([], 2)) == []
    
    # Test chunk size of 1
    assert list(chunk([1, 2, 3, 4], 1)) == [[1], [2], [3], [4]]
    
    # Test chunk size of 2 with even length list
    assert list(chunk([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]
    
    # Test chunk size of 2 with odd length list
    assert list(chunk([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    
    # Test chunk size larger than list
    assert list(chunk([1, 2, 3], 5)) == [[1, 2, 3]]
    
    # Test chunk size of 3
    assert list(chunk([1, 2, 3, 4, 5, 6, 7], 3)) == [[1, 2, 3], [4, 5, 6], [7]]
    
    # Test with single element
    assert list(chunk([1], 1)) == [[1]]
    assert list(chunk([1], 2)) == [[1]]
    
    # Test with strings
    assert list(chunk(['a', 'b', 'c', 'd'], 2)) == [['a', 'b'], ['c', 'd']]
    
    # Test with larger chunk size
    assert list(chunk([1, 2, 3, 4, 5], 10)) == [[1, 2, 3, 4, 5]]


