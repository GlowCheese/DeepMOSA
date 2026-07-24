####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_remove_single_occurrence():
    s = pbag([1, 2, 3])
    s2 = s.remove(1)
    assert s2 == pbag([2, 3])

def test_remove_multiple_occurrences():
    s = pbag([1, 1, 2])
    s2 = s.remove(1)
    assert s2 == pbag([1, 2])

def test_remove_non_existent_element_raises_keyerror():
    s = pbag([1, 2, 3])
    try:
        s.remove(4)
    except KeyError:
        pass
    else:
        assert False, "Expected KeyError"

def test_remove_last_occurrence():
    s = pbag([1])
    s2 = s.remove(1)
    assert s2 == pbag([])

def test_remove_does_not_modify_original():
    s = pbag([1, 2, 3])
    s2 = s.remove(1)
    assert s == pbag([1, 2, 3])
    assert s2 == pbag([2, 3])


# LLM-generated content at query #2
#--------------------------

```python
def test_remove_raises_keyerror():
    s = pbag([1, 1, 2])
    try:
        s.remove(3)
        assert False, "Expected KeyError"
    except KeyError:
        pass


# LLM-generated content at query #3
#--------------------------

```python
def test_remove_raises_keyerror_for_nonexistent_element():
    s = pbag([1, 1, 2])
    try:
        s.remove(3)
        assert False, "Expected KeyError"
    except KeyError:
        pass


