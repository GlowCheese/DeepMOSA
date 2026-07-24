####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test___getitem___with_valid_index():
    lst = plist([1, 2, 3])
    result = lst[0]
    expected = 1
    assert result == expected

def test___getitem___with_negative_index():
    lst = plist([1, 2, 3])
    result = lst[-1]
    expected = 3
    assert result == expected

def test___getitem___with_out_of_range_index():
    lst = plist([1, 2, 3])
    try:
        lst[5]
        assert False
    except IndexError:
        assert True

def test___getitem___with_slice_start_only():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[2:]
    expected = plist([3, 4, 5])
    assert result == expected

def test___getitem___with_slice_start_and_stop():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[1:4]
    expected = plist([2, 3, 4])
    assert result == expected

def test___getitem___with_slice_step():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[::2]
    expected = plist([1, 3, 5])
    assert result == expected

def test___getitem___with_non_integer_index():
    lst = plist([1, 2, 3])
    try:
        lst["invalid"]
        assert False
    except TypeError:
        assert True

def test___getitem___with_empty_list():
    lst = plist()
    try:
        lst[0]
        assert False
    except IndexError:
        assert True

def test___getitem___with_single_element_list():
    lst = plist([42])
    result = lst[0]
    expected = 42
    assert result == expected

def test___getitem___with_negative_index_out_of_range():
    lst = plist([1, 2, 3])
    try:
        lst[-5]
        assert False
    except IndexError:
        assert True


# LLM-generated content at query #2
#--------------------------

def test_getitem_slice_start_not_none_stop_none_step_none_or_one():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[2:]
    expected = plist([3, 4, 5])
    assert result == expected

def test_getitem_slice_start_not_none_stop_none_step_one():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[2::1]
    expected = plist([3, 4, 5])
    assert result == expected

def test_getitem_slice_start_none_stop_none_step_none():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[:]
    expected = plist([1, 2, 3, 4, 5])
    assert result == expected

def test_getitem_slice_start_not_none_stop_not_none_step_none():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[1:3]
    expected = plist([2, 3])
    assert result == expected

def test_getitem_slice_start_not_none_stop_none_step_not_one():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[1::2]
    expected = plist([2, 4])
    assert result == expected

def test_getitem_slice_start_none_stop_not_none_step_none():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[:3]
    expected = plist([1, 2, 3])
    assert result == expected

def test_getitem_slice_start_none_stop_none_step_not_one():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[::2]
    expected = plist([1, 3, 5])
    assert result == expected

def test_getitem_slice_start_not_none_stop_not_none_step_not_one():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[1:4:2]
    expected = plist([2, 4])
    assert result == expected


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_plist_empty_iterable():
    result = plist()
    expected = _EMPTY_PLIST
    assert result == expected


def test_plist_empty_list():
    result = plist([])
    expected = _EMPTY_PLIST
    assert result == expected


def test_plist_with_elements():
    result = plist([1, 2, 3])
    expected = _EMPTY_PLIST.cons(1).cons(2).cons(3)
    assert result == expected


def test_plist_reverse_false():
    result = plist([1, 2, 3], reverse=False)
    expected = _EMPTY_PLIST.cons(1).cons(2).cons(3)
    assert result == expected


def test_plist_reverse_true():
    result = plist([1, 2, 3], reverse=True)
    expected = _EMPTY_PLIST.cons(3).cons(2).cons(1)
    assert result == expected


def test_plist_with_strings():
    result = plist(['a', 'b', 'c'])
    expected = _EMPTY_PLIST.cons('a').cons('b').cons('c')
    assert result == expected


def test_plist_with_mixed_types():
    result = plist([1, 'a', 3.14])
    expected = _EMPTY_PLIST.cons(1).cons('a').cons(3.14)
    assert result == expected


def test_plist_reverse_with_strings():
    result = plist(['a', 'b', 'c'], reverse=True)
    expected = _EMPTY_PLIST.cons('c').cons('b').cons('a')
    assert result == expected


def test_plist_from_tuple():
    result = plist((1, 2, 3))
    expected = _EMPTY_PLIST.cons(1).cons(2).cons(3)
    assert result == expected


def test_plist_from_range():
    result = plist(range(3))
    expected = _EMPTY_PLIST.cons(0).cons(1).cons(2)
    assert result == expected


# LLM-generated content at query #2
#--------------------------

def test_constructor_empty():
    pl = plist()
    assert len(pl) == 0
    assert list(pl) == []

def test_constructor_single_element():
    pl = plist([1])
    assert len(pl) == 1
    assert list(pl) == [1]

def test_constructor_multiple_elements():
    pl = plist([1, 2, 3])
    assert len(pl) == 3
    assert list(pl) == [1, 2, 3]

def test_constructor_nested_iterable():
    pl = plist((1, 2, 3))
    assert len(pl) == 3
    assert list(pl) == [1, 2, 3]

def test_constructor_from_plist():
    pl1 = plist([1, 2, 3])
    pl2 = plist(pl1)
    assert pl1 == pl2
    assert pl1 is not pl2

def test_constructor_empty_iterable():
    pl = plist([])
    assert len(pl) == 0
    assert list(pl) == []

def test_constructor_string_iterable():
    pl = plist("abc")
    assert len(pl) == 3
    assert list(pl) == ['a', 'b', 'c']

def test_constructor_generator():
    gen = (x for x in range(3))
    pl = plist(gen)
    assert len(pl) == 3
    assert list(pl) == [0, 1, 2]

def test_constructor_range():
    pl = plist(range(3))
    assert len(pl) == 3
    assert list(pl) == [0, 1, 2]

def test_constructor_preserves_order():
    pl = plist([3, 1, 2])
    assert list(pl) == [3, 1, 2]

def test_constructor_with_none():
    pl = plist([None, 1, None])
    assert len(pl) == 3
    assert list(pl) == [None, 1, None]


# LLM-generated content at query #3
#--------------------------

def test_getitem_with_valid_index():
    lst = plist([1, 2, 3])
    result = lst[0]
    expected = 1
    assert result == expected

def test_getitem_with_middle_index():
    lst = plist([10, 20, 30])
    result = lst[1]
    expected = 20
    assert result == expected

def test_getitem_with_last_index():
    lst = plist([5, 6, 7])
    result = lst[2]
    expected = 7
    assert result == expected

def test_getitem_with_negative_index():
    lst = plist([1, 2, 3, 4])
    result = lst[-1]
    expected = 4
    assert result == expected

def test_getitem_with_negative_index_middle():
    lst = plist([1, 2, 3, 4])
    result = lst[-2]
    expected = 3
    assert result == expected

def test_getitem_raises_index_error_for_out_of_range_positive():
    lst = plist([1, 2, 3])
    try:
        lst[5]
        assert False
    except IndexError:
        assert True

def test_getitem_raises_index_error_for_out_of_range_negative():
    lst = plist([1, 2, 3])
    try:
        lst[-5]
        assert False
    except IndexError:
        assert True

def test_getitem_with_slice_start_only():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[2:]
    expected = plist([3, 4, 5])
    assert result == expected

def test_getitem_with_slice_start_and_stop():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[1:4]
    expected = plist([2, 3, 4])
    assert result == expected

def test_getitem_with_slice_step():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[::2]
    expected = plist([1, 3, 5])
    assert result == expected

def test_getitem_with_full_slice():
    lst = plist([1, 2, 3])
    result = lst[:]
    expected = plist([1, 2, 3])
    assert result == expected

def test_getitem_with_negative_slice():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[-3:-1]
    expected = plist([3, 4])
    assert result == expected

def test_getitem_raises_type_error_for_non_integer_index():
    lst = plist([1, 2, 3])
    try:
        lst["invalid"]
        assert False
    except TypeError:
        assert True

def test_getitem_on_empty_list_raises_index_error():
    lst = plist()
    try:
        lst[0]
        assert False
    except IndexError:
        assert True

def test_getitem_on_empty_list_with_negative_index():
    lst = plist()
    try:
        lst[-1]
        assert False
    except IndexError:
        assert True

def test_getitem_with_slice_on_empty_list():
    lst = plist()
    result = lst[:]
    expected = plist()
    assert result == expected

def test_getitem_with_large_index_on_single_element_list():
    lst = plist([42])
    result = lst[0]
    expected = 42
    assert result == expected

def test_getitem_with_negative_index_on_single_element_list():
    lst = plist([99])
    result = lst[-1]
    expected = 99
    assert result == expected


# LLM-generated content at query #4
#--------------------------

def test_reversed_returns_reversed_list():
    pl = plist([1, 2, 3])
    result = pl.__reversed__()
    expected = plist([3, 2, 1])
    assert result == expected
def test_reversed_on_empty_list():
    pl = plist()
    result = pl.__reversed__()
    expected = plist()
    assert result == expected
def test_reversed_on_single_element():
    pl = plist([5])
    result = pl.__reversed__()
    expected = plist([5])
    assert result == expected
def test_reversed_using_builtin_reversed():
    pl = plist([1, 2, 3])
    result = reversed(pl)
    expected = plist([3, 2, 1])
    assert result == expected
def test_reversed_preserves_original():
    original = plist([1, 2, 3])
    reversed_list = original.__reversed__()
    assert original == plist([1, 2, 3])
    assert reversed_list == plist([3, 2, 1])
def test_reversed_on_large_list():
    pl = plist(range(100))
    result = pl.__reversed__()
    expected = plist(list(range(99, -1, -1)))
    assert result == expected


# LLM-generated content at query #5
#--------------------------

def test_split_empty_list():
    lst = plist()
    left, right = lst.split(0)
    assert left == plist()
    assert right == plist()

def test_split_single_element_list_at_zero():
    lst = plist([1])
    left, right = lst.split(0)
    assert left == plist()
    assert right == plist([1])

def test_split_single_element_list_at_one():
    lst = plist([1])
    left, right = lst.split(1)
    assert left == plist([1])
    assert right == plist()

def test_split_multi_element_list_at_zero():
    lst = plist([1, 2, 3])
    left, right = lst.split(0)
    assert left == plist()
    assert right == plist([1, 2, 3])

def test_split_multi_element_list_at_middle():
    lst = plist([1, 2, 3, 4])
    left, right = lst.split(2)
    assert left == plist([1, 2])
    assert right == plist([3, 4])

def test_split_multi_element_list_at_end():
    lst = plist([1, 2, 3])
    left, right = lst.split(3)
    assert left == plist([1, 2, 3])
    assert right == plist()

def test_split_multi_element_list_beyond_end():
    lst = plist([1, 2, 3])
    left, right = lst.split(5)
    assert left == plist([1, 2, 3])
    assert right == plist()

def test_split_multi_element_list_at_first():
    lst = plist([1, 2, 3])
    left, right = lst.split(1)
    assert left == plist([1])
    assert right == plist([2, 3])

def test_split_multi_element_list_at_last():
    lst = plist([1, 2, 3])
    left, right = lst.split(2)
    assert left == plist([1, 2])
    assert right == plist([3])

def test_split_large_list():
    lst = plist(range(10))
    left, right = lst.split(5)
    assert left == plist([0, 1, 2, 3, 4])
    assert right == plist([5, 6, 7, 8, 9])


# LLM-generated content at query #6
#--------------------------

def test_getitem_slice_start_not_none_stop_not_none_step_none():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[1:3]
    expected = plist([2, 3])
    assert result == expected

def test_getitem_slice_start_none_stop_none_step_none():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[:]
    expected = plist([1, 2, 3, 4, 5])
    assert result == expected

def test_getitem_slice_start_not_none_stop_not_none_step_not_one():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[0:5:2]
    expected = plist([1, 3, 5])
    assert result == expected

def test_getitem_slice_start_none_stop_not_none_step_none():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[:3]
    expected = plist([1, 2, 3])
    assert result == expected

def test_getitem_slice_start_not_none_stop_none_step_not_one():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[1::2]
    expected = plist([2, 4])
    assert result == expected

def test_getitem_slice_start_none_stop_none_step_not_one():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[::2]
    expected = plist([1, 3, 5])
    assert result == expected

def test_getitem_slice_start_not_none_stop_not_none_step_negative():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[4:1:-1]
    expected = plist([5, 4, 3])
    assert result == expected

def test_getitem_slice_start_not_none_stop_none_step_negative():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[4::-1]
    expected = plist([5, 4, 3, 2, 1])
    assert result == expected

def test_getitem_slice_start_none_stop_not_none_step_negative():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[:1:-1]
    expected = plist([5, 4, 3])
    assert result == expected

def test_getitem_slice_start_none_stop_none_step_negative():
    lst = plist([1, 2, 3, 4, 5])
    result = lst[::-1]
    expected = plist([5, 4, 3, 2, 1])
    assert result == expected


