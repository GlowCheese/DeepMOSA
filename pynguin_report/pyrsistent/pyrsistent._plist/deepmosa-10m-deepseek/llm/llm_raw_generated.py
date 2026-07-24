####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test___getitem___with_integer_index():
    pl = plist([1, 2, 3, 4])
    assert pl[0] == 1
    assert pl[1] == 2
    assert pl[3] == 4

def test___getitem___with_negative_index():
    pl = plist([1, 2, 3, 4])
    assert pl[-1] == 4
    assert pl[-2] == 3
    assert pl[-4] == 1

def test___getitem___with_out_of_range_index():
    pl = plist([1, 2, 3])
    try:
        pl[3]
        assert False
    except IndexError:
        pass

def test___getitem___with_slice():
    pl = plist([1, 2, 3, 4, 5])
    assert pl[1:3] == plist([2, 3])
    assert pl[1:] == plist([2, 3, 4, 5])
    assert pl[:3] == plist([1, 2, 3])
    assert pl[::2] == plist([1, 3, 5])

def test___getitem___with_non_integer_index():
    pl = plist([1, 2, 3])
    try:
        pl["invalid"]
        assert False
    except TypeError:
        pass


# LLM-generated content at query #2
#--------------------------

```
def test___getitem___with_integer_index():
    lst = plist([1, 2, 3, 4, 5])
    assert lst[0] == 1
    assert lst[2] == 3
    assert lst[-1] == 5
    assert lst[-3] == 3

def test___getitem___raises_index_error_for_out_of_range_index():
    lst = plist([1, 2, 3])
    try:
        lst[3]
        assert False, "Expected IndexError"
    except IndexError:
        pass
    try:
        lst[-4]
        assert False, "Expected IndexError"
    except IndexError:
        pass

def test___getitem___with_slice():
    lst = plist([1, 2, 3, 4, 5])
    assert lst[1:4] == plist([2, 3, 4])
    assert lst[2:] == plist([3, 4, 5])
    assert lst[:3] == plist([1, 2, 3])
    assert lst[::2] == plist([1, 3, 5])

def test___getitem___raises_type_error_for_non_integer_index():
    lst = plist([1, 2, 3])
    try:
        lst["invalid"]
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #3
#--------------------------

```
def test_getitem_slice_with_stop_not_none():
    lst = plist([1, 2, 3])
    result = lst[1:2]
    assert result == plist([2])

def test_getitem_slice_with_step_not_one_or_none():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[0:4:2]
    assert result == plist([1, 3])

def test_getitem_slice_with_start_none():
    lst = plist([1, 2, 3])
    result = lst[:2]
    assert result == plist([1, 2])

def test_getitem_slice_with_start_and_stop_and_step():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[1:4:2]
    assert result == plist([2, 4])


# LLM-generated content at query #4
#--------------------------

```
def test_getitem_slice_condition_false_when_stop_not_none():
    lst = plist([1, 2, 3, 4])
    slice_obj = slice(1, 2, None)
    result = lst[slice_obj]
    assert result == plist([2])

def test_getitem_slice_condition_false_when_start_none():
    lst = plist([1, 2, 3, 4])
    slice_obj = slice(None, None, 1)
    result = lst[slice_obj]
    assert result == plist([1, 2, 3, 4])

def test_getitem_slice_condition_false_when_step_not_one_or_none():
    lst = plist([1, 2, 3, 4])
    slice_obj = slice(1, None, 2)
    result = lst[slice_obj]
    assert result == plist([2, 4])


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_plist_empty_iterable():
    assert plist() == _EMPTY_PLIST
    assert plist([]) == _EMPTY_PLIST
    assert plist([], reverse=True) == _EMPTY_PLIST

def test_plist_single_element():
    assert plist([1]) == _EMPTY_PLIST.cons(1)
    assert plist([1], reverse=True) == _EMPTY_PLIST.cons(1)

def test_plist_multiple_elements():
    assert plist([1, 2, 3]) == _EMPTY_PLIST.cons(1).cons(2).cons(3)
    assert plist([1, 2, 3], reverse=True) == _EMPTY_PLIST.cons(3).cons(2).cons(1)

def test_plist_string_iterable():
    assert plist("abc") == _EMPTY_PLIST.cons('a').cons('b').cons('c')
    assert plist("abc", reverse=True) == _EMPTY_PLIST.cons('c').cons('b').cons('a')

def test_plist_tuple_iterable():
    assert plist((1, 2, 3)) == _EMPTY_PLIST.cons(1).cons(2).cons(3)
    assert plist((1, 2, 3), reverse=True) == _EMPTY_PLIST.cons(3).cons(2).cons(1)


# LLM-generated content at query #2
#--------------------------

```
def test___getitem___with_valid_index():
    lst = plist([1, 2, 3, 4])
    assert lst[0] == 1
    assert lst[1] == 2
    assert lst[2] == 3
    assert lst[3] == 4

def test___getitem___with_negative_index():
    lst = plist([1, 2, 3, 4])
    assert lst[-1] == 4
    assert lst[-2] == 3
    assert lst[-3] == 2
    assert lst[-4] == 1

def test___getitem___with_out_of_range_index():
    lst = plist([1, 2, 3])
    try:
        lst[3]
        assert False
    except IndexError:
        pass

    try:
        lst[-4]
        assert False
    except IndexError:
        pass

def test___getitem___with_slice():
    lst = plist([1, 2, 3, 4, 5])
    assert lst[1:4] == plist([2, 3, 4])
    assert lst[1:] == plist([2, 3, 4, 5])
    assert lst[:3] == plist([1, 2, 3])
    assert lst[::2] == plist([1, 3, 5])

def test___getitem___with_non_integer_index():
    lst = plist([1, 2, 3])
    try:
        lst["invalid"]
        assert False
    except TypeError:
        pass


# LLM-generated content at query #3
#--------------------------

```
def test_split_empty_list():
    empty_list = _EMPTY_PLIST
    left, right = empty_list.split(0)
    assert left == _EMPTY_PLIST
    assert right == _EMPTY_PLIST

def test_split_single_element_list_at_0():
    single_list = plist([1])
    left, right = single_list.split(0)
    assert left == _EMPTY_PLIST
    assert right == single_list

def test_split_single_element_list_at_1():
    single_list = plist([1])
    left, right = single_list.split(1)
    assert left == single_list
    assert right == _EMPTY_PLIST

def test_split_two_elements_list_at_1():
    two_list = plist([1, 2])
    left, right = two_list.split(1)
    assert left == plist([1])
    assert right == plist([2])

def test_split_three_elements_list_at_2():
    three_list = plist([1, 2, 3])
    left, right = three_list.split(2)
    assert left == plist([1, 2])
    assert right == plist([3])

def test_split_three_elements_list_at_0():
    three_list = plist([1, 2, 3])
    left, right = three_list.split(0)
    assert left == _EMPTY_PLIST
    assert right == three_list

def test_split_three_elements_list_at_3():
    three_list = plist([1, 2, 3])
    left, right = three_list.split(3)
    assert left == three_list
    assert right == _EMPTY_PLIST

def test_split_four_elements_list_at_2():
    four_list = plist([1, 2, 3, 4])
    left, right = four_list.split(2)
    assert left == plist([1, 2])
    assert right == plist([3, 4])


# LLM-generated content at query #4
#--------------------------

```python
def test_getitem_slice_start_not_none_stop_not_none_step_1():
    lst = _PListBase()
    slice_obj = slice(1, 2, 1)
    assert not (slice_obj.start is not None and slice_obj.stop is None and (slice_obj.step is None or slice_obj.step == 1))

def test_getitem_slice_start_none_stop_none_step_none():
    lst = _PListBase()
    slice_obj = slice(None, None, None)
    assert not (slice_obj.start is not None and slice_obj.stop is None and (slice_obj.step is None or slice_obj.step == 1))

def test_getitem_slice_start_not_none_stop_none_step_not_1():
    lst = _PListBase()
    slice_obj = slice(1, None, 2)
    assert not (slice_obj.start is not None and slice_obj.stop is None and (slice_obj.step is None or slice_obj.step == 1))

def test_getitem_slice_start_not_none_stop_not_none_step_not_1():
    lst = _PListBase()
    slice_obj = slice(1, 2, 2)
    assert not (slice_obj.start is not None and slice_obj.stop is None and (slice_obj.step is None or slice_obj.step == 1))


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_6_evaluates_to_false():
    pl = plist([1, 2, 3, 4])
    sl = slice(1, 3, 1)
    assert not (sl.start is not None and sl.stop is None and (sl.step is None or sl.step == 1))


# LLM-generated content at query #6
#--------------------------

```
def test_getitem_slice_with_stop_not_none():
    lst = plist([1, 2, 3])
    lst[1:2]

def test_getitem_slice_with_step_not_none_or_one():
    lst = plist([1, 2, 3])
    lst[1::2]

def test_getitem_slice_with_start_none():
    lst = plist([1, 2, 3])
    lst[:2]


# LLM-generated content at query #7
#--------------------------

```python
def test_getitem_slice_start_not_none_stop_none_step_not_one():
    lst = plist([1, 2, 3, 4])
    slice_obj = slice(1, None, 2)
    result = lst[slice_obj]
    expected = plist([2, 4])
    assert result == expected

def test_getitem_slice_start_not_none_stop_not_none_step_none():
    lst = plist([1, 2, 3, 4])
    slice_obj = slice(1, 3, None)
    result = lst[slice_obj]
    expected = plist([2, 3])
    assert result == expected

def test_getitem_slice_start_not_none_stop_not_none_step_not_one():
    lst = plist([1, 2, 3, 4])
    slice_obj = slice(1, 3, 2)
    result = lst[slice_obj]
    expected = plist([2])
    assert result == expected

def test_getitem_slice_start_none_stop_not_none_step_none():
    lst = plist([1, 2, 3, 4])
    slice_obj = slice(None, 3, None)
    result = lst[slice_obj]
    expected = plist([1, 2, 3])
    assert result == expected

def test_getitem_slice_start_none_stop_none_step_not_one():
    lst = plist([1, 2, 3, 4])
    slice_obj = slice(None, None, 2)
    result = lst[slice_obj]
    expected = plist([1, 3])
    assert result == expected


