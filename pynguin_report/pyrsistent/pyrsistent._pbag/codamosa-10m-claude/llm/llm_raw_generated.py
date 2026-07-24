####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_PBag___sub__():
    # Test basic subtraction
    bag1 = pbag([1, 2, 2, 2, 3])
    bag2 = pbag([2, 3, 3, 4])
    result = bag1 - bag2
    assert result == pbag([1, 2, 2])
    
    # Test subtraction with empty bag
    bag3 = pbag([1, 2, 3])
    empty_bag = pbag([])
    assert bag3 - empty_bag == bag3
    
    # Test empty bag minus non-empty bag
    assert empty_bag - bag3 == empty_bag
    
    # Test subtracting more elements than exist
    bag4 = pbag([1, 2])
    bag5 = pbag([1, 1, 1, 2, 2, 2])
    result2 = bag4 - bag5
    assert result2 == pbag([])
    
    # Test subtracting non-existent element
    bag6 = pbag([1, 2, 3])
    bag7 = pbag([4, 5])
    result3 = bag6 - bag7
    assert result3 == bag6
    
    # Test single element bags
    single1 = pbag([1])
    single2 = pbag([1])
    assert single1 - single2 == pbag([])
    
    # Test with duplicate elements
    bag8 = pbag([1, 1, 1, 2, 2])
    bag9 = pbag([1, 1, 2])
    result4 = bag8 - bag9
    assert result4 == pbag([1, 2])
    
    # Test type error with non-PBag
    bag10 = pbag([1, 2, 3])
    assert (bag10 - [1, 2]) == NotImplemented
    assert (bag10 - {1, 2}) == NotImplemented
    assert (bag10 - "123") == NotImplemented
    
    # Test that original bags are unchanged
    original1 = pbag([1, 2, 3])
    original2 = pbag([1, 2])
    original1 - original2
    assert original1 == pbag([1, 2, 3])
    assert original2 == pbag([1, 2])


# LLM-generated content at query #2
#--------------------------

def test_PBag___sub__():
    # Test basic subtraction
    bag1 = pbag([1, 2, 2, 2, 3])
    bag2 = pbag([2, 3, 3, 4])
    result = bag1 - bag2
    assert result == pbag([1, 2, 2])
    
    # Test subtraction with no common elements
    bag1 = pbag([1, 2, 3])
    bag2 = pbag([4, 5, 6])
    result = bag1 - bag2
    assert result == pbag([1, 2, 3])
    
    # Test subtraction with all elements removed
    bag1 = pbag([1, 2, 3])
    bag2 = pbag([1, 2, 3, 4, 5])
    result = bag1 - bag2
    assert result == pbag([])
    
    # Test subtraction with empty bag
    bag1 = pbag([1, 2, 3])
    bag2 = pbag([])
    result = bag1 - bag2
    assert result == pbag([1, 2, 3])
    
    # Test subtraction from empty bag
    bag1 = pbag([])
    bag2 = pbag([1, 2, 3])
    result = bag1 - bag2
    assert result == pbag([])
    
    # Test subtraction with equal bags
    bag1 = pbag([1, 1, 2])
    bag2 = pbag([1, 1, 2])
    result = bag1 - bag2
    assert result == pbag([])
    
    # Test subtraction with partial overlaps
    bag1 = pbag([1, 1, 1, 2, 2, 3])
    bag2 = pbag([1, 2, 2, 2])
    result = bag1 - bag2
    assert result == pbag([1, 1, 3])
    
    # Test that original bags are not modified
    bag1 = pbag([1, 2, 3])
    bag2 = pbag([2])
    original_bag1 = bag1
    result = bag1 - bag2
    assert bag1 == original_bag1
    assert result == pbag([1, 3])
    
    # Test subtraction with non-PBag raises NotImplemented
    bag1 = pbag([1, 2, 3])
    assert (bag1 - [1, 2]) == NotImplemented or isinstance(bag1 - [1, 2], type(NotImplemented))


# LLM-generated content at query #3
#--------------------------

def test_PBag___sub__():
    # Test basic subtraction
    bag1 = pbag([1, 2, 2, 2, 3])
    bag2 = pbag([2, 3, 3, 4])
    result = bag1 - bag2
    assert result == pbag([1, 2, 2])
    
    # Test subtraction with no common elements
    bag1 = pbag([1, 2, 3])
    bag2 = pbag([4, 5, 6])
    result = bag1 - bag2
    assert result == pbag([1, 2, 3])
    
    # Test subtraction resulting in empty bag
    bag1 = pbag([1, 2])
    bag2 = pbag([1, 2, 3])
    result = bag1 - bag2
    assert result == pbag([])
    
    # Test subtraction with identical bags
    bag1 = pbag([1, 1, 2])
    bag2 = pbag([1, 1, 2])
    result = bag1 - bag2
    assert result == pbag([])
    
    # Test subtraction with empty bag
    bag1 = pbag([1, 2, 3])
    bag2 = pbag([])
    result = bag1 - bag2
    assert result == pbag([1, 2, 3])
    
    # Test subtraction from empty bag
    bag1 = pbag([])
    bag2 = pbag([1, 2])
    result = bag1 - bag2
    assert result == pbag([])
    
    # Test subtraction with duplicates
    bag1 = pbag([1, 1, 1, 2, 2])
    bag2 = pbag([1, 2, 2, 2])
    result = bag1 - bag2
    assert result == pbag([1, 1])
    
    # Test that subtraction with non-PBag raises NotImplemented
    bag1 = pbag([1, 2, 3])
    result = bag1.__sub__([1, 2])
    assert result is NotImplemented
    
    # Test that original bags are not modified
    bag1 = pbag([1, 2, 2, 3])
    bag2 = pbag([2, 3])
    result = bag1 - bag2
    assert bag1 == pbag([1, 2, 2, 3])
    assert bag2 == pbag([2, 3])


# LLM-generated content at query #4
#--------------------------

```python
def test_PBag___sub__():
    # Test basic subtraction
    bag1 = pbag([1, 2, 2, 2, 3])
    bag2 = pbag([2, 3, 3, 4])
    result = bag1 - bag2
    assert result == pbag([1, 2, 2])
    
    # Test subtraction with empty bag
    bag1 = pbag([1, 2, 3])
    bag2 = pbag([])
    result = bag1 - bag2
    assert result == bag1
    
    # Test subtraction from empty bag
    bag1 = pbag([])
    bag2 = pbag([1, 2, 3])
    result = bag1 - bag2
    assert result == pbag([])
    
    # Test subtraction of non-existent elements
    bag1 = pbag([1, 2, 3])
    bag2 = pbag([4, 5, 6])
    result = bag1 - bag2
    assert result == bag1
    
    # Test complete removal of all elements
    bag1 = pbag([1, 1, 2])
    bag2 = pbag([1, 1, 2])
    result = bag1 - bag2
    assert result == pbag([])
    
    # Test partial removal of duplicates
    bag1 = pbag([1, 1, 1, 1, 2, 2])
    bag2 = pbag([1, 1, 2])
    result = bag1 - bag2
    assert result == pbag([1, 1, 2])
    
    # Test subtraction returns new instance
    bag1 = pbag([1, 2, 3])
    bag2 = pbag([1])
    result = bag1 - bag2
    assert result is not bag1
    
    # Test TypeError when subtracting non-PBag
    bag1 = pbag([1, 2, 3])
    assert (bag1 - [1, 2]) == NotImplemented
    assert (bag1 - {1, 2}) == NotImplemented
    assert (bag1 - "123") == NotImplemented
    
    # Test subtraction with single element bags
    bag1 = pbag([5])
    bag2 = pbag([5])
    result = bag1 - bag2
    assert result == pbag([])
    
    # Test complex scenario with multiple duplicates
    bag1 = pbag([1, 1, 1, 2, 2, 3, 3, 3, 3])
    bag2 = pbag([1, 2, 2, 3, 3])
    result = bag1 - bag2
    assert result == pbag([1, 1, 3, 3])


# LLM-generated content at query #5
#--------------------------

def test_PBag___sub__():
    # Test basic subtraction
    bag1 = pbag([1, 2, 2, 2, 3])
    bag2 = pbag([2, 3, 3, 4])
    result = bag1 - bag2
    assert result == pbag([1, 2, 2])
    
    # Test subtraction with no common elements
    bag3 = pbag([1, 2, 3])
    bag4 = pbag([4, 5, 6])
    result2 = bag3 - bag4
    assert result2 == bag3
    
    # Test subtraction resulting in empty bag
    bag5 = pbag([1, 2])
    bag6 = pbag([1, 2, 3])
    result3 = bag5 - bag6
    assert result3 == pbag([])
    assert len(result3) == 0
    
    # Test subtraction with duplicate removal
    bag7 = pbag([1, 1, 1, 2, 2])
    bag8 = pbag([1, 1, 2])
    result4 = bag7 - bag8
    assert result4 == pbag([1, 2])
    assert result4.count(1) == 1
    assert result4.count(2) == 1
    
    # Test subtraction from empty bag
    bag9 = pbag([])
    bag10 = pbag([1, 2])
    result5 = bag9 - bag10
    assert result5 == pbag([])
    
    # Test subtraction with single elements
    bag11 = pbag([1])
    bag12 = pbag([1])
    result6 = bag11 - bag12
    assert result6 == pbag([])
    
    # Test that original bags are not modified
    bag13 = pbag([1, 2, 2, 3])
    bag14 = pbag([2, 3])
    original_bag13 = pbag([1, 2, 2, 3])
    original_bag14 = pbag([2, 3])
    bag13 - bag14
    assert bag13 == original_bag13
    assert bag14 == original_bag14
    
    # Test subtraction with non-PBag raises NotImplemented
    bag15 = pbag([1, 2])
    result7 = bag15 - [1]
    assert result7 is NotImplemented
    
    # Test subtraction with all counts going to zero
    bag16 = pbag([1, 2, 2])
    bag17 = pbag([1, 2, 2, 3])
    result8 = bag16 - bag17
    assert result8 == pbag([])


