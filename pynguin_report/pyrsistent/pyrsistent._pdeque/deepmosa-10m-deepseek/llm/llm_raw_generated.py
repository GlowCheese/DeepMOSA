####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_with_valid_inputs():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_maxlen_zero():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_raises_type_error_for_non_integer_maxlen():
    left_list = plist([1])
    right_list = plist()
    length = 1
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_raises_value_error_for_negative_maxlen():
    left_list = plist([1])
    right_list = plist()
    length = 1
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True

def test_constructor_with_empty_lists():
    left_list = plist()
    right_list = plist()
    length = 0
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_only_left_list():
    left_list = plist([1, 2, 3])
    right_list = plist()
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length

def test_constructor_with_only_right_list():
    left_list = plist()
    right_list = plist([1, 2, 3])
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length


# LLM-generated content at query #2
#--------------------------

def test_rotate_positive_steps():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.rotate(2)
    expected = pdeque([4, 5, 1, 2, 3])
    assert result == expected

def test_rotate_negative_steps():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.rotate(-2)
    expected = pdeque([3, 4, 5, 1, 2])
    assert result == expected

def test_rotate_zero_steps():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.rotate(0)
    assert result == dq

def test_rotate_steps_greater_than_length():
    dq = pdeque([1, 2, 3])
    result = dq.rotate(5)
    expected = pdeque([2, 3, 1])
    assert result == expected

def test_rotate_negative_steps_greater_than_length():
    dq = pdeque([1, 2, 3])
    result = dq.rotate(-5)
    expected = pdeque([3, 1, 2])
    assert result == expected

def test_rotate_single_element():
    dq = pdeque([42])
    result = dq.rotate(3)
    assert result == dq

def test_rotate_empty():
    dq = pdeque([])
    result = dq.rotate(10)
    assert result == dq

def test_rotate_with_maxlen():
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.rotate(1)
    expected = pdeque([3, 1, 2], maxlen=3)
    assert result == expected

def test_rotate_negative_with_maxlen():
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.rotate(-1)
    expected = pdeque([2, 3, 1], maxlen=3)
    assert result == expected

def test_rotate_steps_equal_length():
    dq = pdeque([1, 2, 3])
    result = dq.rotate(3)
    assert result == dq


# LLM-generated content at query #3
#--------------------------

def test_popleft_empty_deque():
    dq = pdeque([])
    result = dq.popleft()
    assert result == pdeque([])
    assert len(result) == 0


def test_popleft_single_element():
    dq = pdeque([1])
    result = dq.popleft()
    assert result == pdeque([])
    assert len(result) == 0


def test_popleft_multiple_elements():
    dq = pdeque([1, 2, 3])
    result = dq.popleft()
    assert result == pdeque([2, 3])
    assert len(result) == 2


def test_popleft_with_count():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.popleft(3)
    assert result == pdeque([4, 5])
    assert len(result) == 2


def test_popleft_count_exceeds_length():
    dq = pdeque([1, 2, 3])
    result = dq.popleft(5)
    assert result == pdeque([])
    assert len(result) == 0


def test_popleft_negative_count():
    dq = pdeque([1, 2, 3])
    result = dq.popleft(-2)
    assert result == pdeque([3])
    assert len(result) == 1


def test_popleft_on_bounded_deque():
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.popleft()
    assert result == pdeque([2, 3], maxlen=3)
    assert len(result) == 2


def test_popleft_on_bounded_deque_with_count():
    dq = pdeque([1, 2, 3, 4], maxlen=4)
    result = dq.popleft(2)
    assert result == pdeque([3, 4], maxlen=4)
    assert len(result) == 2


def test_popleft_preserves_maxlen():
    dq = pdeque([1, 2, 3], maxlen=5)
    result = dq.popleft(2)
    assert result.maxlen == 5
    assert result == pdeque([3], maxlen=5)


def test_popleft_identity_when_count_zero():
    dq = pdeque([1, 2, 3])
    result = dq.popleft(0)
    assert result == dq
    assert len(result) == 3


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_with_valid_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen == maxlen

def test_constructor_without_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen_zero():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen == maxlen

def test_constructor_with_non_integer_maxlen_raises_typeerror():
    left_list = plist([1])
    right_list = plist([2])
    length = 2
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_with_negative_maxlen_raises_valueerror():
    left_list = plist([1])
    right_list = plist([2])
    length = 2
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #2
#--------------------------

def test_constructor_with_valid_arguments():
    left = plist([1, 2])
    right = plist([3, 4])
    dq = PDeque(left, right, 4, maxlen=5)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 4
    assert dq._maxlen == 5

def test_constructor_without_maxlen():
    left = plist([1])
    right = plist([2])
    dq = PDeque(left, right, 2)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 2
    assert dq._maxlen is None

def test_constructor_with_maxlen_zero():
    left = plist()
    right = plist()
    dq = PDeque(left, right, 0, maxlen=0)
    assert dq._maxlen == 0

def test_constructor_with_negative_maxlen_raises_value_error():
    left = plist()
    right = plist()
    try:
        PDeque(left, right, 0, maxlen=-1)
        assert False
    except ValueError:
        assert True

def test_constructor_with_non_integer_maxlen_raises_type_error():
    left = plist()
    right = plist()
    try:
        PDeque(left, right, 0, maxlen="invalid")
        assert False
    except TypeError:
        assert True

def test_constructor_with_empty_lists():
    left = plist()
    right = plist()
    dq = PDeque(left, right, 0)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 0
    assert dq._maxlen is None

def test_constructor_with_only_left_list():
    left = plist([1, 2, 3])
    right = plist()
    dq = PDeque(left, right, 3)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 3

def test_constructor_with_only_right_list():
    left = plist()
    right = plist([1, 2, 3])
    dq = PDeque(left, right, 3)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 3

def test_constructor_with_maxlen_equal_to_length():
    left = plist([1, 2])
    right = plist([3, 4])
    dq = PDeque(left, right, 4, maxlen=4)
    assert dq._maxlen == 4

def test_constructor_with_maxlen_greater_than_length():
    left = plist([1])
    right = plist([2])
    dq = PDeque(left, right, 2, maxlen=10)
    assert dq._maxlen == 10


# LLM-generated content at query #3
#--------------------------

def test_constructor_with_valid_inputs():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_maxlen_zero():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_invalid_maxlen_type():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_with_negative_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True

def test_constructor_with_empty_lists():
    left_list = plist()
    right_list = plist()
    length = 0
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length

def test_constructor_with_only_left_list():
    left_list = plist([1, 2, 3])
    right_list = plist()
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length

def test_constructor_with_only_right_list():
    left_list = plist()
    right_list = plist([1, 2, 3])
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length

def test_constructor_length_mismatch():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 10
    dq = PDeque(left_list, right_list, length)
    assert dq._length == length


# LLM-generated content at query #4
#--------------------------

def test_rotate_positive_steps():
    dq = pdeque([1, 2, 3])
    result = dq.rotate(1)
    expected = pdeque([3, 1, 2])
    assert result == expected

def test_rotate_negative_steps():
    dq = pdeque([1, 2, 3])
    result = dq.rotate(-2)
    expected = pdeque([3, 1, 2])
    assert result == expected

def test_rotate_zero_steps():
    dq = pdeque([1, 2, 3])
    result = dq.rotate(0)
    assert result == dq

def test_rotate_more_than_length():
    dq = pdeque([1, 2, 3])
    result = dq.rotate(5)
    expected = pdeque([2, 3, 1])
    assert result == expected

def test_rotate_negative_more_than_length():
    dq = pdeque([1, 2, 3])
    result = dq.rotate(-4)
    expected = pdeque([2, 3, 1])
    assert result == expected

def test_rotate_single_element():
    dq = pdeque([42])
    result = dq.rotate(10)
    assert result == dq

def test_rotate_empty():
    dq = pdeque([])
    result = dq.rotate(3)
    assert result == dq

def test_rotate_with_maxlen():
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.rotate(1)
    expected = pdeque([3, 1, 2], maxlen=3)
    assert result == expected

def test_rotate_negative_with_maxlen():
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.rotate(-1)
    expected = pdeque([2, 3, 1], maxlen=3)
    assert result == expected


# LLM-generated content at query #5
#--------------------------

def test_getitem_single_positive_index():
    dq = pdeque([10, 20, 30])
    result = dq[0]
    expected = 10
    assert result == expected

def test_getitem_single_negative_index():
    dq = pdeque([10, 20, 30])
    result = dq[-1]
    expected = 30
    assert result == expected

def test_getitem_index_out_of_range_positive():
    dq = pdeque([10, 20, 30])
    try:
        dq[5]
        assert False
    except IndexError:
        assert True

def test_getitem_index_out_of_range_negative():
    dq = pdeque([10, 20, 30])
    try:
        dq[-5]
        assert False
    except IndexError:
        assert True

def test_getitem_slice_full():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq[:]
    expected = pdeque([1, 2, 3, 4, 5])
    assert result == expected

def test_getitem_slice_start_only():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq[2:]
    expected = pdeque([3, 4, 5])
    assert result == expected

def test_getitem_slice_stop_only():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq[:3]
    expected = pdeque([1, 2, 3])
    assert result == expected

def test_getitem_slice_start_and_stop():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq[1:4]
    expected = pdeque([2, 3, 4])
    assert result == expected

def test_getitem_slice_negative_start():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq[-3:]
    expected = pdeque([3, 4, 5])
    assert result == expected

def test_getitem_slice_negative_stop():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq[:-2]
    expected = pdeque([1, 2, 3])
    assert result == expected

def test_getitem_slice_with_step_not_one():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq[::2]
    expected = pdeque([1, 3, 5])
    assert result == expected

def test_getitem_slice_with_negative_step():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq[::-1]
    expected = pdeque([5, 4, 3, 2, 1])
    assert result == expected

def test_getitem_slice_start_stop_step():
    dq = pdeque([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = dq[1:8:3]
    expected = pdeque([2, 5, 8])
    assert result == expected

def test_getitem_slice_empty_result():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq[3:2]
    expected = pdeque([])
    assert result == expected

def test_getitem_slice_with_maxlen():
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq[:]
    expected = pdeque([3, 4, 5], maxlen=3)
    assert result == expected

def test_getitem_slice_start_beyond_length():
    dq = pdeque([1, 2, 3])
    result = dq[10:]
    expected = pdeque([])
    assert result == expected

def test_getitem_slice_stop_beyond_length():
    dq = pdeque([1, 2, 3])
    result = dq[:10]
    expected = pdeque([1, 2, 3])
    assert result == expected

def test_getitem_slice_negative_start_beyond_length():
    dq = pdeque([1, 2, 3])
    result = dq[-10:]
    expected = pdeque([1, 2, 3])
    assert result == expected

def test_getitem_slice_negative_stop_beyond_length():
    dq = pdeque([1, 2, 3])
    result = dq[:-10]
    expected = pdeque([])
    assert result == expected

def test_getitem_with_non_integer_index():
    dq = pdeque([1, 2, 3])
    try:
        dq["invalid"]
        assert False
    except TypeError:
        assert True

def test_getitem_slice_with_step_none():
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq[1:4:None]
    expected = pdeque([2, 3, 4])
    assert result == expected


# LLM-generated content at query #6
#--------------------------

def test_remove_existing_element_from_left():
    dq = pdeque([2, 1, 2])
    result = dq.remove(2)
    expected = pdeque([1, 2])
    assert result == expected

def test_remove_existing_element_from_right():
    dq = pdeque([1, 2, 3])
    result = dq.remove(3)
    expected = pdeque([1, 2])
    assert result == expected

def test_remove_only_element():
    dq = pdeque([5])
    result = dq.remove(5)
    expected = pdeque([])
    assert result == expected

def test_remove_first_occurrence():
    dq = pdeque([1, 2, 1, 3])
    result = dq.remove(1)
    expected = pdeque([2, 1, 3])
    assert result == expected

def test_remove_element_not_present_raises_value_error():
    dq = pdeque([1, 2, 3])
    try:
        dq.remove(4)
        assert False
    except ValueError:
        assert True

def test_remove_from_empty_deque_raises_value_error():
    dq = pdeque([])
    try:
        dq.remove(1)
        assert False
    except ValueError:
        assert True

def test_remove_maintains_maxlen():
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3], maxlen=3)
    assert result == expected

def test_remove_element_with_multiple_occurrences():
    dq = pdeque([4, 5, 4, 6, 4])
    result = dq.remove(4)
    expected = pdeque([5, 4, 6, 4])
    assert result == expected

def test_remove_after_append():
    dq = pdeque([1, 2]).append(3)
    result = dq.remove(2)
    expected = pdeque([1, 3])
    assert result == expected

def test_remove_after_appendleft():
    dq = pdeque([1, 2]).appendleft(0)
    result = dq.remove(1)
    expected = pdeque([0, 2])
    assert result == expected


# LLM-generated content at query #7
#--------------------------

def test_popleft_empty_deque():
    d = pdeque([])
    result = d.popleft()
    assert result == pdeque([])
    assert len(result) == 0

def test_popleft_single_element():
    d = pdeque([1])
    result = d.popleft()
    assert result == pdeque([])
    assert len(result) == 0

def test_popleft_multiple_elements():
    d = pdeque([1, 2, 3])
    result = d.popleft()
    assert result == pdeque([2, 3])
    assert len(result) == 2

def test_popleft_with_count():
    d = pdeque([1, 2, 3, 4])
    result = d.popleft(2)
    assert result == pdeque([3, 4])
    assert len(result) == 2

def test_popleft_all_elements():
    d = pdeque([1, 2, 3])
    result = d.popleft(3)
    assert result == pdeque([])
    assert len(result) == 0

def test_popleft_more_than_length():
    d = pdeque([1, 2])
    result = d.popleft(5)
    assert result == pdeque([])
    assert len(result) == 0

def test_popleft_negative_count():
    d = pdeque([1, 2, 3])
    result = d.popleft(-2)
    assert result == pdeque([1])
    assert len(result) == 1

def test_popleft_with_maxlen():
    d = pdeque([1, 2, 3], maxlen=3)
    result = d.popleft()
    assert result == pdeque([2, 3], maxlen=3)
    assert len(result) == 2
    assert result.maxlen == 3

def test_popleft_preserves_maxlen():
    d = pdeque([1, 2], maxlen=5)
    result = d.popleft()
    assert result.maxlen == 5

def test_popleft_on_deque_with_single_list_side():
    d = pdeque([1, 2, 3])
    d = d.popleft(2)
    result = d.popleft()
    assert result == pdeque([3])
    assert len(result) == 1


# LLM-generated content at query #8
#--------------------------

def test___new___creates_instance_with_valid_arguments():
    from pyrsistent import plist
    left = plist([1, 2])
    right = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left, right, length, maxlen)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == length
    assert dq._maxlen == maxlen

def test___new___creates_instance_without_maxlen():
    from pyrsistent import plist
    left = plist([1])
    right = plist([2])
    length = 2
    dq = PDeque(left, right, length)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == length
    assert dq._maxlen is None

def test___new___raises_type_error_for_non_integral_maxlen():
    from pyrsistent import plist
    left = plist()
    right = plist()
    length = 0
    maxlen = "invalid"
    try:
        PDeque(left, right, length, maxlen)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert str(e) == 'An integer is required as maxlen'

def test___new___raises_value_error_for_negative_maxlen():
    from pyrsistent import plist
    left = plist()
    right = plist()
    length = 0
    maxlen = -1
    try:
        PDeque(left, right, length, maxlen)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "maxlen must be non-negative"

def test___new___accepts_zero_maxlen():
    from pyrsistent import plist
    left = plist()
    right = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left, right, length, maxlen)
    assert dq._maxlen == 0

def test___new___accepts_positive_maxlen():
    from pyrsistent import plist
    left = plist([1])
    right = plist([2])
    length = 2
    maxlen = 10
    dq = PDeque(left, right, length, maxlen)
    assert dq._maxlen == 10


# LLM-generated content at query #9
#--------------------------

def test_constructor_creates_pdeque_with_correct_attributes():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen == maxlen

def test_constructor_without_maxlen_sets_maxlen_to_none():
    left_list = plist([1])
    right_list = plist([2])
    length = 2
    dq = PDeque(left_list, right_list, length)
    assert dq._maxlen is None

def test_constructor_with_maxlen_as_zero():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == 0

def test_constructor_with_negative_maxlen_raises_value_error():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError as e:
        assert str(e) == "maxlen must be non-negative"

def test_constructor_with_non_integer_maxlen_raises_type_error():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError as e:
        assert str(e) == "An integer is required as maxlen"

def test_constructor_with_maxlen_as_positive_integer():
    left_list = plist([1])
    right_list = plist([2])
    length = 2
    maxlen = 3
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == 3


# LLM-generated content at query #10
#--------------------------

def test_constructor_creates_pdeque_with_correct_attributes():
    left = plist([1, 2])
    right = plist([3, 4])
    dq = PDeque(left, right, 4, maxlen=5)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 4
    assert dq._maxlen == 5

def test_constructor_without_maxlen():
    left = plist([1])
    right = plist([2])
    dq = PDeque(left, right, 2)
    assert dq._maxlen is None

def test_constructor_with_maxlen_zero():
    dq = PDeque(plist(), plist(), 0, maxlen=0)
    assert dq._maxlen == 0

def test_constructor_raises_type_error_for_non_integer_maxlen():
    try:
        PDeque(plist(), plist(), 0, maxlen="invalid")
        assert False
    except TypeError:
        assert True

def test_constructor_raises_value_error_for_negative_maxlen():
    try:
        PDeque(plist(), plist(), 0, maxlen=-1)
        assert False
    except ValueError:
        assert True

def test_constructor_with_maxlen_positive():
    dq = PDeque(plist([1]), plist([2]), 2, maxlen=10)
    assert dq._maxlen == 10

def test_constructor_with_empty_lists():
    dq = PDeque(plist(), plist(), 0)
    assert dq._length == 0
    assert dq._left_list == plist()
    assert dq._right_list == plist()

def test_constructor_length_zero_with_nonempty_lists():
    dq = PDeque(plist([1]), plist([2]), 0)
    assert dq._length == 0

def test_constructor_assigns_weakref_slot():
    left = plist()
    right = plist()
    dq = PDeque(left, right, 0)
    assert hasattr(dq, '__weakref__')


# LLM-generated content at query #11
#--------------------------

def test_constructor_creates_valid_pdeque():
    left = plist([1, 2])
    right = plist([3, 4])
    dq = PDeque(left, right, 4, None)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 4
    assert dq._maxlen is None

def test_constructor_with_maxlen():
    left = plist([1])
    right = plist([2])
    dq = PDeque(left, right, 2, 5)
    assert dq._maxlen == 5

def test_constructor_maxlen_non_negative():
    left = plist()
    right = plist()
    dq = PDeque(left, right, 0, 0)
    assert dq._maxlen == 0

def test_constructor_raises_type_error_for_invalid_maxlen():
    left = plist()
    right = plist()
    try:
        PDeque(left, right, 0, "invalid")
        assert False
    except TypeError:
        assert True

def test_constructor_raises_value_error_for_negative_maxlen():
    left = plist()
    right = plist()
    try:
        PDeque(left, right, 0, -1)
        assert False
    except ValueError:
        assert True

def test_constructor_with_empty_lists():
    left = plist()
    right = plist()
    dq = PDeque(left, right, 0, None)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 0
    assert dq._maxlen is None

def test_constructor_length_zero_with_nonempty_lists():
    left = plist([1])
    right = plist([2])
    dq = PDeque(left, right, 0, None)
    assert dq._length == 0

def test_constructor_preserves_weakref_slot():
    left = plist()
    right = plist()
    dq = PDeque(left, right, 0, None)
    assert hasattr(dq, '__weakref__')

def test_constructor_maxlen_none_implies_unbounded():
    left = plist([1, 2])
    right = plist([3, 4])
    dq = PDeque(left, right, 4, None)
    assert dq.maxlen is None

def test_constructor_sets_correct_length():
    left = plist([1, 2, 3])
    right = plist([4, 5])
    dq = PDeque(left, right, 5, None)
    assert len(dq) == 5


# LLM-generated content at query #12
#--------------------------

def test_constructor_with_valid_arguments():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_maxlen_zero():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_non_integral_maxlen_raises_typeerror():
    left_list = plist([1])
    right_list = plist()
    length = 1
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_with_negative_maxlen_raises_valueerror():
    left_list = plist([1])
    right_list = plist()
    length = 1
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True

def test_constructor_with_empty_lists():
    left_list = plist()
    right_list = plist()
    length = 0
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length

def test_constructor_with_only_left_list():
    left_list = plist([1, 2, 3])
    right_list = plist()
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list

def test_constructor_with_only_right_list():
    left_list = plist()
    right_list = plist([1, 2, 3])
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list

def test_constructor_length_mismatch_but_still_constructs():
    left_list = plist([1, 2])
    right_list = plist([3])
    length = 10
    dq = PDeque(left_list, right_list, length)
    assert dq._length == length


# LLM-generated content at query #13
#--------------------------

```python
def test_maxlen_is_integer():
    try:
        PDeque(plist(), plist(), 0, maxlen="not an integer")
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError for non-integer maxlen"


# LLM-generated content at query #14
#--------------------------

```python
def test_remove_elem_in_left_list():
    dq = pdeque([2, 1, 2])
    result = dq.remove(2)
    expected = pdeque([1, 2])
    assert result == expected


# LLM-generated content at query #15
#--------------------------

def test_constructor_with_valid_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen == maxlen

def test_constructor_without_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen_zero():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == 0

def test_constructor_raises_type_error_for_non_integer_maxlen():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_raises_value_error_for_negative_maxlen():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #16
#--------------------------

def test_eq_same_elements_same_order():
    dq1 = pdeque([1, 2, 3])
    dq2 = pdeque([1, 2, 3])
    result = dq1 == dq2
    assert result is True

def test_eq_different_elements():
    dq1 = pdeque([1, 2, 3])
    dq2 = pdeque([1, 2, 4])
    result = dq1 == dq2
    assert result is False

def test_eq_different_lengths():
    dq1 = pdeque([1, 2, 3])
    dq2 = pdeque([1, 2])
    result = dq1 == dq2
    assert result is False

def test_eq_same_elements_different_maxlen():
    dq1 = pdeque([1, 2, 3], maxlen=5)
    dq2 = pdeque([1, 2, 3], maxlen=10)
    result = dq1 == dq2
    assert result is True

def test_eq_empty_deques():
    dq1 = pdeque([])
    dq2 = pdeque([])
    result = dq1 == dq2
    assert result is True

def test_eq_with_non_pdeque():
    dq = pdeque([1, 2, 3])
    lst = [1, 2, 3]
    result = dq == lst
    assert result is NotImplemented

def test_eq_same_elements_different_internal_structure():
    dq1 = pdeque([1, 2, 3])
    dq2 = dq1.append(4).popleft()
    result = dq1 == dq2
    assert result is True

def test_eq_hash_consistency():
    dq1 = pdeque([1, 2, 3])
    dq2 = pdeque([1, 2, 3])
    eq_result = dq1 == dq2
    hash_result = hash(dq1) == hash(dq2)
    assert eq_result == hash_result


# LLM-generated content at query #17
#--------------------------

def test_remove_elem_not_found_raises_value_error():
    dq = pdeque([1, 2, 3])
    try:
        dq.remove(4)
        assert False
    except ValueError as e:
        assert str(e) == "4 not found in PDeque"


# LLM-generated content at query #18
#--------------------------

def test_constructor_with_valid_inputs():
    left = plist([1, 2])
    right = plist([3, 4])
    dq = PDeque(left, right, 4, maxlen=5)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 4
    assert dq._maxlen == 5

def test_constructor_without_maxlen():
    left = plist([1])
    right = plist([2])
    dq = PDeque(left, right, 2)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 2
    assert dq._maxlen is None

def test_constructor_with_maxlen_zero():
    left = plist()
    right = plist()
    dq = PDeque(left, right, 0, maxlen=0)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 0
    assert dq._maxlen == 0

def test_constructor_with_negative_maxlen_raises_value_error():
    left = plist()
    right = plist()
    try:
        PDeque(left, right, 0, maxlen=-1)
        assert False
    except ValueError:
        assert True

def test_constructor_with_non_integer_maxlen_raises_type_error():
    left = plist()
    right = plist()
    try:
        PDeque(left, right, 0, maxlen="invalid")
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #19
#--------------------------

```python
def test_remove_elem_in_left_list():
    dq = pdeque([1, 2, 3])
    result = dq.remove(2)
    expected = pdeque([1, 3])
    assert result == expected

def test_remove_elem_in_right_list():
    dq = pdeque([1, 2, 3])
    dq = dq.appendleft(4)
    result = dq.remove(4)
    expected = pdeque([1, 2, 3])
    assert result == expected

def test_remove_first_occurrence_from_left():
    dq = pdeque([2, 1, 2])
    result = dq.remove(2)
    expected = pdeque([1, 2])
    assert result == expected

def test_remove_with_multiple_operations():
    dq = pdeque([1, 2, 3])
    dq = dq.append(4)
    dq = dq.appendleft(0)
    result = dq.remove(3)
    expected = pdeque([0, 1, 2, 4])
    assert result == expected


# LLM-generated content at query #20
#--------------------------

def test_eq_assertion_fails_when_lengths_differ_but_tuples_equal():
    from pyrsistent import pdeque
    from unittest.mock import Mock
    mock_other = Mock(spec=pdeque)
    mock_other.__len__ = Mock(return_value=5)
    mock_other.__iter__ = Mock(return_value=iter([1, 2, 3]))
    test_instance = pdeque([1, 2, 3])
    test_instance._length = 3
    test_instance._left_list = Mock()
    test_instance._right_list = Mock()
    test_instance._left_list.__iter__ = Mock(return_value=iter([1, 2, 3]))
    test_instance._right_list.__iter__ = Mock(return_value=iter([]))
    test_instance._right_list.reverse = Mock(return_value=Mock(__iter__=Mock(return_value=iter([]))))
    result = test_instance.__eq__(mock_other)
    assert result is True


# LLM-generated content at query #21
#--------------------------

def test_remove_elem_not_in_left_list_but_in_right_list():
    from pyrsistent import pdeque
    dq = pdeque([1, 2, 3])
    dq._left_list = pdeque([])._left_list
    dq._right_list = pdeque([1, 2, 3])._right_list
    dq._length = 3
    result = dq.remove(2)
    expected = pdeque([1, 3])
    assert result == expected


# LLM-generated content at query #22
#--------------------------

def test_constructor_creates_valid_pdeque():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_maxlen_zero():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_raises_type_error_for_non_integer_maxlen():
    left_list = plist([1])
    right_list = plist()
    length = 1
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_raises_value_error_for_negative_maxlen():
    left_list = plist([1])
    right_list = plist()
    length = 1
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True

def test_constructor_with_empty_lists():
    left_list = plist()
    right_list = plist()
    length = 0
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == 0
    assert dq._maxlen is None

def test_constructor_with_only_left_list():
    left_list = plist([1, 2, 3])
    right_list = plist()
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length

def test_constructor_with_only_right_list():
    left_list = plist()
    right_list = plist([1, 2, 3])
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length

def test_constructor_length_mismatch_handled():
    left_list = plist([1, 2])
    right_list = plist([3])
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._length == length

def test_constructor_maxlen_none_is_default():
    left_list = plist([1])
    right_list = plist([2])
    length = 2
    dq = PDeque(left_list, right_list, length)
    assert dq._maxlen is None


# LLM-generated content at query #23
#--------------------------

def test___new___creates_instance_with_correct_attributes():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen == maxlen

def test___new___creates_instance_without_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test___new___raises_type_error_for_non_integer_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError as e:
        assert str(e) == 'An integer is required as maxlen'

def test___new___raises_value_error_for_negative_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError as e:
        assert str(e) == "maxlen must be non-negative"

def test___new___accepts_zero_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == 0

def test___new___accepts_positive_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 10
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == 10


# LLM-generated content at query #24
#--------------------------

def test_constructor_with_valid_inputs():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_maxlen_zero():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_non_integer_maxlen_raises_typeerror():
    left_list = plist([1])
    right_list = plist()
    length = 1
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_with_negative_maxlen_raises_valueerror():
    left_list = plist([1])
    right_list = plist()
    length = 1
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True

def test_constructor_with_empty_lists():
    left_list = plist()
    right_list = plist()
    length = 0
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == 0

def test_constructor_with_only_left_list():
    left_list = plist([1, 2, 3])
    right_list = plist()
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list

def test_constructor_with_only_right_list():
    left_list = plist()
    right_list = plist([1, 2, 3])
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list

def test_constructor_length_mismatch_but_still_creates():
    left_list = plist([1, 2])
    right_list = plist([3])
    length = 10
    dq = PDeque(left_list, right_list, length)
    assert dq._length == length


# LLM-generated content at query #25
#--------------------------

def test_constructor_creates_pdeque_with_correct_attributes():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen == maxlen

def test_constructor_without_maxlen():
    left_list = plist([1])
    right_list = plist([2])
    length = 2
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen_zero():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == 0

def test_constructor_raises_type_error_for_non_integer_maxlen():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_raises_value_error_for_negative_maxlen():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #26
#--------------------------

```python
def test_maxlen_is_integer():
    try:
        PDeque(plist(), plist(), 0, maxlen="not an integer")
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert str(e) == 'An integer is required as maxlen'


# LLM-generated content at query #27
#--------------------------

def test_remove_elem_not_found_in_left_list_but_found_in_right_list():
    from pyrsistent import pdeque
    dq = pdeque([1, 2, 3])
    dq._left_list = pdeque([])._left_list
    dq._right_list = pdeque([1, 2, 3])._right_list
    dq._length = 3
    result = dq.remove(2)
    expected = pdeque([1, 3])
    assert result == expected


# LLM-generated content at query #28
#--------------------------

def test_constructor_creates_pdeque_with_correct_attributes():
    left = plist([1, 2])
    right = plist([3, 4])
    dq = PDeque(left, right, 4, maxlen=5)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 4
    assert dq._maxlen == 5

def test_constructor_without_maxlen():
    left = plist([1])
    right = plist([2])
    dq = PDeque(left, right, 2)
    assert dq._left_list == left
    assert dq._right_list == right
    assert dq._length == 2
    assert dq._maxlen is None

def test_constructor_with_maxlen_zero():
    left = plist()
    right = plist()
    dq = PDeque(left, right, 0, maxlen=0)
    assert dq._maxlen == 0

def test_constructor_raises_type_error_for_non_integer_maxlen():
    left = plist()
    right = plist()
    try:
        PDeque(left, right, 0, maxlen="invalid")
        assert False
    except TypeError:
        assert True

def test_constructor_raises_value_error_for_negative_maxlen():
    left = plist()
    right = plist()
    try:
        PDeque(left, right, 0, maxlen=-1)
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #29
#--------------------------

def test_remove_existing_element_from_left():
    dq = pdeque([2, 1, 2])
    result = dq.remove(2)
    expected = pdeque([1, 2])
    assert result == expected

def test_remove_existing_element_from_right():
    dq = pdeque([1, 2, 3])
    result = dq.remove(3)
    expected = pdeque([1, 2])
    assert result == expected

def test_remove_only_element():
    dq = pdeque([5])
    result = dq.remove(5)
    expected = pdeque([])
    assert result == expected

def test_remove_element_not_present():
    dq = pdeque([1, 2, 3])
    try:
        dq.remove(4)
        assert False
    except ValueError as e:
        assert str(e) == "4 not found in PDeque"

def test_remove_from_empty_deque():
    dq = pdeque([])
    try:
        dq.remove(1)
        assert False
    except ValueError as e:
        assert str(e) == "1 not found in PDeque"

def test_remove_first_occurrence():
    dq = pdeque([1, 2, 1, 3, 1])
    result = dq.remove(1)
    expected = pdeque([2, 1, 3, 1])
    assert result == expected

def test_remove_with_maxlen():
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3], maxlen=3)
    assert result == expected

def test_remove_preserves_maxlen():
    dq = pdeque([1, 2, 3, 4], maxlen=4)
    result = dq.remove(3)
    expected = pdeque([1, 2, 4], maxlen=4)
    assert result == expected

def test_remove_element_at_right_end():
    dq = pdeque([1, 2, 3, 4])
    result = dq.remove(4)
    expected = pdeque([1, 2, 3])
    assert result == expected

def test_remove_element_at_left_end():
    dq = pdeque([1, 2, 3, 4])
    result = dq.remove(1)
    expected = pdeque([2, 3, 4])
    assert result == expected


# LLM-generated content at query #30
#--------------------------

def test_constructor_with_valid_inputs():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen == maxlen

def test_constructor_without_maxlen():
    left_list = plist([1])
    right_list = plist([2])
    length = 2
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen_none():
    left_list = plist([1])
    right_list = plist([2])
    length = 2
    dq = PDeque(left_list, right_list, length, None)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen_zero():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen == maxlen

def test_constructor_with_non_integral_maxlen_raises_typeerror():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_with_negative_maxlen_raises_valueerror():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True

def test_constructor_with_empty_lists():
    left_list = plist()
    right_list = plist()
    length = 0
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_only_left_list():
    left_list = plist([1, 2, 3])
    right_list = plist()
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length

def test_constructor_with_only_right_list():
    left_list = plist()
    right_list = plist([1, 2, 3])
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length

def test_constructor_with_maxlen_equal_to_length():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 4
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_maxlen_greater_than_length():
    left_list = plist([1])
    right_list = plist([2])
    length = 2
    maxlen = 10
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_maxlen_less_than_length():
    left_list = plist([1, 2, 3])
    right_list = plist([4, 5])
    length = 5
    maxlen = 3
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen


# LLM-generated content at query #31
#--------------------------

def test_assertion_failure_on_unequal_lengths():
    dq1 = pdeque([1, 2, 3])
    dq2 = pdeque([1, 2, 3])
    dq2._length = 2
    result = dq1 == dq2


# LLM-generated content at query #32
#--------------------------

def test_eq_sanity_check_fails():
    left_list = plist([1])
    right_list = plist([2])
    length = 2
    deque1 = PDeque(left_list, right_list, length)
    deque2 = PDeque(left_list, right_list, length + 1)
    result = deque1 == deque2
    assert result is False


# LLM-generated content at query #33
#--------------------------

def test_constructor_with_valid_arguments():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    deque = PDeque(left_list, right_list, length)
    assert deque._left_list == left_list
    assert deque._right_list == right_list
    assert deque._length == length
    assert deque._maxlen is None

def test_constructor_with_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    deque = PDeque(left_list, right_list, length, maxlen)
    assert deque._maxlen == maxlen

def test_constructor_with_maxlen_zero():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 0
    deque = PDeque(left_list, right_list, length, maxlen)
    assert deque._maxlen == maxlen

def test_constructor_with_invalid_maxlen_type():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_with_negative_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True

def test_constructor_with_empty_lists():
    left_list = plist()
    right_list = plist()
    length = 0
    deque = PDeque(left_list, right_list, length)
    assert deque._left_list == left_list
    assert deque._right_list == right_list
    assert deque._length == length

def test_constructor_with_only_left_list():
    left_list = plist([1, 2, 3])
    right_list = plist()
    length = 3
    deque = PDeque(left_list, right_list, length)
    assert deque._left_list == left_list
    assert deque._right_list == right_list
    assert deque._length == length

def test_constructor_with_only_right_list():
    left_list = plist()
    right_list = plist([1, 2, 3])
    length = 3
    deque = PDeque(left_list, right_list, length)
    assert deque._left_list == left_list
    assert deque._right_list == right_list
    assert deque._length == length

def test_constructor_length_mismatch():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 10
    deque = PDeque(left_list, right_list, length)
    assert deque._length == length


# LLM-generated content at query #34
#--------------------------

def test_remove_existing_element_from_left():
    dq = pdeque([2, 1, 2])
    result = dq.remove(2)
    expected = pdeque([1, 2])
    assert result == expected

def test_remove_existing_element_from_right():
    dq = pdeque([1, 2, 3])
    result = dq.remove(3)
    expected = pdeque([1, 2])
    assert result == expected

def test_remove_only_element():
    dq = pdeque([5])
    result = dq.remove(5)
    expected = pdeque([])
    assert result == expected

def test_remove_element_from_empty_deque_raises_error():
    dq = pdeque([])
    try:
        dq.remove(1)
        assert False
    except ValueError as e:
        assert str(e) == "1 not found in PDeque"

def test_remove_non_existing_element_raises_error():
    dq = pdeque([1, 2, 3])
    try:
        dq.remove(4)
        assert False
    except ValueError as e:
        assert str(e) == "4 not found in PDeque"

def test_remove_first_occurrence_from_left():
    dq = pdeque([1, 2, 1, 3])
    result = dq.remove(1)
    expected = pdeque([2, 1, 3])
    assert result == expected

def test_remove_with_duplicates_in_right_list():
    dq = pdeque([1, 2, 3, 2, 4])
    result = dq.remove(2)
    expected = pdeque([1, 3, 2, 4])
    assert result == expected

def test_remove_maintains_maxlen():
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3], maxlen=3)
    assert result == expected
    assert result.maxlen == 3

def test_remove_element_from_single_element_bounded_queue():
    dq = pdeque([7], maxlen=1)
    result = dq.remove(7)
    expected = pdeque([], maxlen=1)
    assert result == expected

def test_remove_preserves_hash():
    dq = pdeque([1, 2, 3])
    result = dq.remove(2)
    assert hash(result) == hash(pdeque([1, 3]))


# LLM-generated content at query #35
#--------------------------

def test_constructor_with_valid_inputs():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_maxlen_zero():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_invalid_maxlen_type():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_with_negative_maxlen():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True

def test_constructor_with_empty_lists():
    left_list = plist()
    right_list = plist()
    length = 0
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == 0

def test_constructor_with_only_left_list():
    left_list = plist([1, 2, 3])
    right_list = plist()
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list

def test_constructor_with_only_right_list():
    left_list = plist()
    right_list = plist([1, 2, 3])
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list

def test_constructor_length_mismatch():
    left_list = plist([1, 2])
    right_list = plist([3])
    length = 10
    dq = PDeque(left_list, right_list, length)
    assert dq._length == length


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_constructor_with_valid_inputs():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen is None

def test_constructor_with_maxlen():
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_maxlen_zero():
    left_list = plist()
    right_list = plist()
    length = 0
    maxlen = 0
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._maxlen == maxlen

def test_constructor_with_invalid_maxlen_type():
    left_list = plist([1])
    right_list = plist()
    length = 1
    maxlen = "invalid"
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except TypeError:
        assert True

def test_constructor_with_negative_maxlen():
    left_list = plist([1])
    right_list = plist()
    length = 1
    maxlen = -1
    try:
        PDeque(left_list, right_list, length, maxlen)
        assert False
    except ValueError:
        assert True

def test_constructor_with_empty_lists():
    left_list = plist()
    right_list = plist()
    length = 0
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == 0

def test_constructor_with_only_left_list():
    left_list = plist([1, 2, 3])
    right_list = plist()
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list

def test_constructor_with_only_right_list():
    left_list = plist()
    right_list = plist([1, 2, 3])
    length = 3
    dq = PDeque(left_list, right_list, length)
    assert dq._left_list == left_list
    assert dq._right_list == right_list

def test_constructor_length_mismatch():
    left_list = plist([1, 2])
    right_list = plist([3])
    length = 10
    dq = PDeque(left_list, right_list, length)
    assert dq._length == length


# LLM-generated content at query #2
#--------------------------

def test_pop_empty_deque():
    dq = pdeque([])
    result = dq.pop()
    assert result == pdeque([])
    assert len(result) == 0

def test_pop_single_element():
    dq = pdeque([1])
    result = dq.pop()
    assert result == pdeque([])
    assert len(result) == 0

def test_pop_multiple_elements():
    dq = pdeque([1, 2, 3])
    result = dq.pop()
    assert result == pdeque([1, 2])
    assert len(result) == 2

def test_pop_with_count():
    dq = pdeque([1, 2, 3, 4])
    result = dq.pop(2)
    assert result == pdeque([1, 2])
    assert len(result) == 2

def test_pop_all_elements():
    dq = pdeque([1, 2, 3])
    result = dq.pop(3)
    assert result == pdeque([])
    assert len(result) == 0

def test_pop_more_than_length():
    dq = pdeque([1, 2])
    result = dq.pop(5)
    assert result == pdeque([])
    assert len(result) == 0

def test_pop_negative_count():
    dq = pdeque([1, 2, 3])
    result = dq.pop(-2)
    assert result == pdeque([3])
    assert len(result) == 1

def test_pop_on_bounded_deque():
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.pop()
    assert result == pdeque([1, 2], maxlen=3)
    assert len(result) == 2

def test_pop_on_bounded_deque_with_count():
    dq = pdeque([1, 2, 3, 4], maxlen=4)
    result = dq.pop(2)
    assert result == pdeque([1, 2], maxlen=4)
    assert len(result) == 2

def test_pop_maintains_maxlen():
    dq = pdeque([1, 2, 3], maxlen=5)
    result = dq.pop(2)
    assert result.maxlen == 5
    assert result == pdeque([1], maxlen=5)

def test_pop_zero_count():
    dq = pdeque([1, 2, 3])
    result = dq.pop(0)
    assert result == pdeque([1, 2, 3])
    assert len(result) == 3


# LLM-generated content at query #3
#--------------------------

def test_eq_same_elements_same_order():
    dq1 = pdeque([1, 2, 3])
    dq2 = pdeque([1, 2, 3])
    result = dq1 == dq2
    assert result == True

def test_eq_different_elements():
    dq1 = pdeque([1, 2, 3])
    dq2 = pdeque([1, 2, 4])
    result = dq1 == dq2
    assert result == False

def test_eq_different_lengths():
    dq1 = pdeque([1, 2, 3])
    dq2 = pdeque([1, 2])
    result = dq1 == dq2
    assert result == False

def test_eq_same_elements_different_maxlen():
    dq1 = pdeque([1, 2, 3], maxlen=5)
    dq2 = pdeque([1, 2, 3], maxlen=10)
    result = dq1 == dq2
    assert result == True

def test_eq_empty_deques():
    dq1 = pdeque([])
    dq2 = pdeque([])
    result = dq1 == dq2
    assert result == True

def test_eq_with_non_pdeque():
    dq1 = pdeque([1, 2, 3])
    other = [1, 2, 3]
    result = dq1 == other
    assert result == NotImplemented

def test_eq_single_element():
    dq1 = pdeque([42])
    dq2 = pdeque([42])
    result = dq1 == dq2
    assert result == True

def test_eq_large_identical_deques():
    dq1 = pdeque(range(1000))
    dq2 = pdeque(range(1000))
    result = dq1 == dq2
    assert result == True

def test_eq_after_operations():
    dq1 = pdeque([1, 2]).append(3).appendleft(0)
    dq2 = pdeque([0, 1, 2, 3])
    result = dq1 == dq2
    assert result == True

def test_eq_identical_objects():
    dq1 = pdeque([1, 2, 3])
    result = dq1 == dq1
    assert result == True


