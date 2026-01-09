####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method __new__ of class PDeque
def test_PDeque___new__():  
    # Test with valid inputs
    left_list = plist([1, 2])
    right_list = plist([3, 4])
    length = 4
    maxlen = 5
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen == maxlen

    # Test with maxlen as None
    dq = PDeque(left_list, right_list, length, None)
    assert dq._maxlen is None

    # Test with maxlen as negative integer (should raise ValueError)
    try:
        PDeque(left_list, right_list, length, -1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as non-integer (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, "invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as zero
    dq = PDeque(left_list, right_list, length, 0)
    assert dq._maxlen == 0

    # Test with maxlen equal to length
    dq = PDeque(left_list, right_list, length, length)
    assert dq._maxlen == length

    # Test with maxlen greater than length
    dq = PDeque(left_list, right_list, length, length + 1)
    assert dq._maxlen == length + 1

    # Test with maxlen less than length
    dq = PDeque(left_list, right_list, length, length - 1)
    assert dq._maxlen == length - 1

    # Test with empty left and right lists
    empty_left = plist()
    empty_right = plist()
    dq = PDeque(empty_left, empty_right, 0, None)
    assert dq._left_list == empty_left
    assert dq._right_list == empty_right
    assert dq._length == 0
    assert dq._maxlen is None

    # Test with maxlen as float (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, 3.14)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as boolean (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as string (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, "5")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as list (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, [5])
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as tuple (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, (5,))
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as dict (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, {"maxlen": 5})
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as set (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, {5})
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as complex number (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, complex(5, 0))
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as negative zero (should raise ValueError)
    try:
        PDeque(left_list, right_list, length, -0)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as large integer
    large_int = 10**6
    dq = PDeque(left_list, right_list, length, large_int)
    assert dq._maxlen == large_int

    # Test with maxlen as zero and empty lists
    dq = PDeque(empty_left, empty_right, 0, 0)
    assert dq._maxlen == 0

    # Test with maxlen as negative integer and empty lists (should raise ValueError)
    try:
        PDeque(empty_left, empty_right, 0, -1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as non-integer and empty lists (should raise TypeError)
    try:
        PDeque(empty_left, empty_right, 0, "invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as None and empty lists
    dq = PDeque(empty_left, empty_right, 0, None)
    assert dq._maxlen is None

    # Test with maxlen as zero and non-empty lists
    dq = PDeque(left_list, right_list, length, 0)
    assert dq._maxlen == 0

    # Test with maxlen as negative integer and non-empty lists (should raise ValueError)
    try:
        PDeque(left_list, right_list, length, -1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as non-integer and non-empty lists (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, "invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as None and non-empty lists
    dq = PDeque(left_list, right_list, length, None)
    assert dq._maxlen is None

    # Test with maxlen as zero and length zero
    dq = PDeque(empty_left, empty_right, 0, 0)
    assert dq._maxlen == 0

    # Test with maxlen as negative integer and length zero (should raise ValueError)
    try:
        PDeque(empty_left, empty_right, 0, -1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as non-integer and length zero (should raise TypeError)
    try:
        PDeque(empty_left, empty_right, 0, "invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as None and length zero
    dq = PDeque(empty_left, empty_right, 0, None)
    assert dq._maxlen is None

    # Test with maxlen as zero and length non-zero
    dq = PDeque(left_list, right_list, length, 0)
    assert dq._maxlen == 0

    # Test with maxlen as negative integer and length non-zero (should raise ValueError)
    try:
        PDeque(left_list, right_list, length, -1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as non-integer and length non-zero (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, "invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as None and length non-zero
    dq = PDeque(left_list, right_list, length, None)
    assert dq._maxlen is None

    # Test with maxlen as zero and left_list empty, right_list non-empty
    dq = PDeque(empty_left, right_list, length, 0)
    assert dq._maxlen == 0

    # Test with maxlen as negative integer and left_list empty, right_list non-empty (should raise ValueError)
    try:
        PDeque(empty_left, right_list, length, -1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as non-integer and left_list empty, right_list non-empty (should raise TypeError)
    try:
        PDeque(empty_left, right_list, length, "invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as None and left_list empty, right_list non-empty
    dq = PDeque(empty_left, right_list, length, None)
    assert dq._maxlen is None

    # Test with maxlen as zero and left_list non-empty, right_list empty
    dq = PDeque(left_list, empty_right, length, 0)
    assert dq._maxlen == 0

    # Test with maxlen as negative integer and left_list non-empty, right_list empty (should raise ValueError)
    try:
        PDeque(left_list, empty_right, length, -1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as non-integer and left_list non-empty, right_list empty (should raise TypeError)
    try:
        PDeque(left


# LLM-generated content at query #2
#--------------------------

# Unit test for method __repr__ of class PDeque
def test_PDeque___repr__(): 
    d = pdeque([1, 2, 3])
    assert repr(d) == "pdeque([1, 2, 3])"
    d = pdeque([1, 2, 3], maxlen=3)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=3)"
    d = pdeque([], maxlen=0)
    assert repr(d) == "pdeque([], maxlen=0)"
    d = pdeque([], maxlen=1)
    assert repr(d) == "pdeque([], maxlen=1)"
    d = pdeque([1], maxlen=1)
    assert repr(d) == "pdeque([1], maxlen=1)"
    d = pdeque([1, 2], maxlen=1)
    assert repr(d) == "pdeque([2], maxlen=1)"
    d = pdeque([1, 2], maxlen=2)
    assert repr(d) == "pdeque([1, 2], maxlen=2)"
    d = pdeque([1, 2, 3], maxlen=2)
    assert repr(d) == "pdeque([2, 3], maxlen=2)"
    d = pdeque([1, 2, 3], maxlen=3)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=3)"
    d = pdeque([1, 2, 3], maxlen=4)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=4)"
    d = pdeque([1, 2, 3], maxlen=5)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=5)"
    d = pdeque([1, 2, 3], maxlen=6)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=6)"
    d = pdeque([1, 2, 3], maxlen=7)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=7)"
    d = pdeque([1, 2, 3], maxlen=8)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=8)"
    d = pdeque([1, 2, 3], maxlen=9)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=9)"
    d = pdeque([1, 2, 3], maxlen=10)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=10)"
    d = pdeque([1, 2, 3], maxlen=11)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=11)"
    d = pdeque([1, 2, 3], maxlen=12)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=12)"
    d = pdeque([1, 2, 3], maxlen=13)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=13)"
    d = pdeque([1, 2, 3], maxlen=14)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=14)"
    d = pdeque([1, 2, 3], maxlen=15)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=15)"
    d = pdeque([1, 2, 3], maxlen=16)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=16)"
    d = pdeque([1, 2, 3], maxlen=17)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=17)"
    d = pdeque([1, 2, 3], maxlen=18)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=18)"
    d = pdeque([1, 2, 3], maxlen=19)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=19)"
    d = pdeque([1, 2, 3], maxlen=20)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=20)"
    d = pdeque([1, 2, 3], maxlen=21)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=21)"
    d = pdeque([1, 2, 3], maxlen=22)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=22)"
    d = pdeque([1, 2, 3], maxlen=23)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=23)"
    d = pdeque([1, 2, 3], maxlen=24)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=24)"
    d = pdeque([1, 2, 3], maxlen=25)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=25)"
    d = pdeque([1, 2, 3], maxlen=26)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=26)"
    d = pdeque([1, 2, 3], maxlen=27)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=27)"
    d = pdeque([1, 2, 3], maxlen=28)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=28)"
    d = pdeque([1, 2, 3], maxlen=29)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=29)"
    d = pdeque([1, 2, 3], maxlen=30)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=30)"
    d = pdeque([1, 2, 3], maxlen=31)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=31)"
    d = pdeque([1, 2, 3], maxlen=32)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=32)"
    d = pdeque([1, 2, 3], maxlen=33)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=33)"
    d = pdeque([1, 2, 3], maxlen=34)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=34)"
    d = pdeque([1, 2, 3], maxlen=35)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=35)"
    d = pdeque([1, 2, 3], maxlen=36)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=36)"
    d = pdeque([1, 2, 3], maxlen=37)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=37)"
    d = pdeque([1, 2, 3], maxlen=38)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=38)"
    d = pdeque([1, 2, 3], maxlen=39)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=39)"
    d = pdeque([1, 2, 3], maxlen=40)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=40)"
    d = pdeque([1, 2, 3], maxlen=41)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=41)"
    d = pdeque([1, 2, 3], maxlen=42)
    assert repr(d) == "pdeque([1, 2, 3], maxlen=42)"
    d


# LLM-generated content at query #3
#--------------------------

# Unit test for method remove of class PDeque
def test_PDeque_remove(): 
    # Test case 1: Remove element from left list
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.remove(3)
    expected = pdeque([1, 2, 4, 5])
    assert result == expected

    # Test case 2: Remove element from right list
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.remove(5)
    expected = pdeque([1, 2, 3, 4])
    assert result == expected

    # Test case 3: Remove element that appears multiple times
    dq = pdeque([1, 2, 3, 2, 1])
    result = dq.remove(2)
    expected = pdeque([1, 3, 2, 1])
    assert result == expected

    # Test case 4: Remove element that is not in the deque
    dq = pdeque([1, 2, 3])
    try:
        dq.remove(4)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "4 not found in PDeque"

    # Test case 5: Remove element from empty deque
    dq = pdeque([])
    try:
        dq.remove(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "1 not found in PDeque"

    # Test case 6: Remove element from deque with maxlen
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq.remove(3)
    expected = pdeque([1, 2, 4, 5], maxlen=3)
    assert result == expected

    # Test case 7: Remove element from deque with maxlen, causing elements to be discarded
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4, 5], maxlen=3)
    assert result == expected

    # Test case 8: Remove element from deque with maxlen, causing elements to be discarded from left
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq.remove(1)
    expected = pdeque([2, 3, 4, 5], maxlen=3)
    assert result == expected

    # Test case 9: Remove element from deque with maxlen, causing elements to be discarded from right
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq.remove(5)
    expected = pdeque([1, 2, 3, 4], maxlen=3)
    assert result == expected

    # Test case 10: Remove element from deque with maxlen, causing elements to be discarded from both sides
    dq = pdeque([1, 2, 3, 4, 5], maxlen=2)
    result = dq.remove(3)
    expected = pdeque([2, 4, 5], maxlen=2)
    assert result == expected

    print("All test cases passed")

test_PDeque_remove()


# LLM-generated content at query #4
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    # Test with integer index
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq[0] == 1
    assert dq[2] == 3
    assert dq[-1] == 5
    assert dq[-3] == 3

    # Test with slice
    assert dq[1:3] == pdeque([2, 3])
    assert dq[:3] == pdeque([1, 2, 3])
    assert dq[2:] == pdeque([3, 4, 5])
    assert dq[::2] == pdeque([1, 3, 5])

    # Test with negative slice
    assert dq[-3:-1] == pdeque([3, 4])
    assert dq[-3:] == pdeque([3, 4, 5])
    assert dq[:-2] == pdeque([1, 2, 3])

    # Test with step in slice
    assert dq[1:5:2] == pdeque([2, 4])
    assert dq[::3] == pdeque([1, 4])

    # Test with empty slice
    assert dq[5:] == pdeque([])
    assert dq[:0] == pdeque([])

    # Test with out of range index
    try:
        dq[10]
        assert False, "IndexError should be raised"
    except IndexError:
        pass

    # Test with invalid index type
    try:
        dq["invalid"]
        assert False, "TypeError should be raised"
    except TypeError:
        pass

    # Test with maxlen
    dq_maxlen = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq_maxlen[0] == 3
    assert dq_maxlen[-1] == 5
    assert dq_maxlen[:2] == pdeque([3, 4], maxlen=3)

    # Test with empty deque
    dq_empty = pdeque([])
    try:
        dq_empty[0]
        assert False, "IndexError should be raised"
    except IndexError:
        pass

    # Test with single element deque
    dq_single = pdeque([42])
    assert dq_single[0] == 42
    assert dq_single[-1] == 42
    assert dq_single[:] == pdeque([42])

    # Test with large deque
    dq_large = pdeque(range(1000))
    assert dq_large[500] == 500
    assert dq_large[-500] == 500
    assert dq_large[200:300] == pdeque(range(200, 300))

    # Test with negative step in slice (should return reversed deque)
    assert dq[::-1] == pdeque([5, 4, 3, 2, 1])
    assert dq[3:0:-1] == pdeque([4, 3, 2])
    assert dq[-1:-4:-1] == pdeque([5, 4, 3])

    # Test with step and negative indices
    assert dq[-1:-6:-2] == pdeque([5, 3, 1])
    assert dq[-2:-6:-2] == pdeque([4, 2])

    # Test with slice that goes beyond boundaries
    assert dq[-10:10] == pdeque([1, 2, 3, 4, 5])
    assert dq[2:100] == pdeque([3, 4, 5])
    assert dq[-100:-2] == pdeque([1, 2, 3])

    # Test with step that's not 1 and maxlen
    dq_maxlen_step = pdeque([1, 2, 3, 4, 5], maxlen=4)
    assert dq_maxlen_step[::2] == pdeque([2, 4], maxlen=4)

    # Test with slice that results in empty deque
    assert dq[3:3] == pdeque([])
    assert dq[10:15] == pdeque([])
    assert dq[-10:-20] == pdeque([])

    # Test with slice step that results in single element
    assert dq[2:3] == pdeque([3])
    assert dq[-2:-1] == pdeque([4])

    # Test with slice that includes all elements
    assert dq[:] == pdeque([1, 2, 3, 4, 5])
    assert dq[0:5] == pdeque([1, 2, 3, 4, 5])
    assert dq[-5:] == pdeque([1, 2, 3, 4, 5])

    # Test with slice that starts from negative index and goes to positive
    assert dq[-3:4] == pdeque([3, 4])
    assert dq[-5:3] == pdeque([1, 2, 3])

    # Test with slice that uses None as start/stop
    assert dq[None:3] == pdeque([1, 2, 3])
    assert dq[2:None] == pdeque([3, 4, 5])
    assert dq[None:None] == pdeque([1, 2, 3, 4, 5])

    # Test with complex slice (start > stop with positive step)
    assert dq[3:1] == pdeque([])
    assert dq[4:2] == pdeque([])

    # Test with slice that has step > 1 and negative indices
    assert dq[-1:-6:-2] == pdeque([5, 3, 1])
    assert dq[-2:-6:-2] == pdeque([4, 2])

    # Test with slice that has step < -1
    assert dq[4:0:-2] == pdeque([5, 3])
    assert dq[3:0:-2] == pdeque([4, 2])

    # Test with slice that has step = -1 and includes all elements
    assert dq[::-1] == pdeque([5, 4, 3, 2, 1])
    assert dq[4:-6:-1] == pdeque([5, 4, 3, 2, 1])

    # Test with slice that has step = -1 and partial range
    assert dq[3:0:-1] == pdeque([4, 3, 2])
    assert dq[-2:-5:-1] == pdeque([4, 3, 2])

    # Test with slice that has step = -2
    assert dq[4:0:-2] == pdeque([5, 3])
    assert dq[-1:-6:-2] == pdeque([5, 3, 1])

    # Test with slice that has step = -3
    assert dq[4:0:-3] == pdeque([5, 2])
    assert dq[-1:-6:-3] == pdeque([5, 2])

    # Test with empty deque and slice
    dq_empty = pdeque([])
    assert dq_empty[:] == pdeque([])
    assert dq_empty[0:5] == pdeque([])
    assert dq_empty[-5:-1] == pdeque([])

    # Test with single element deque and slice
    dq_single = pdeque([42])
    assert dq_single[:] == pdeque([42])
    assert dq_single[0:1] == pdeque([42])
    assert dq_single[-1:] == pdeque([42])
    assert dq_single[:-1] == pdeque([])
    assert dq_single[1:] == pdeque([])

    # Test with deque of two elements
    dq_two = pdeque([10, 20])
    assert dq_two[:] == pdeque([10, 20])
    assert dq_two[0:1] == pdeque([10])
    assert dq_two[1:2] == pdeque([20])
    assert dq_two[-1:] == pdeque([20])
    assert dq_two[:-1] == pdeque([10])
    assert dq_two[::-1] == pdeque([20, 10])

    # Test with slice that has step = 0 (should raise ValueError)
    try:
        dq[::0]
        assert False, "ValueError should be raised for step=0"
    except ValueError:
        pass

    # Test with slice that has large step
    assert dq[::10] == pdeque([1])
    assert dq[1::10] == pdeque([2])
    assert dq


# LLM-generated content at query #5
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    # Test with integer index
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq[0] == 1
    assert dq[2] == 3
    assert dq[-1] == 5
    assert dq[-3] == 3

    # Test with slice
    assert dq[1:4] == pdeque([2, 3, 4])
    assert dq[:3] == pdeque([1, 2, 3])
    assert dq[2:] == pdeque([3, 4, 5])
    assert dq[::2] == pdeque([1, 3, 5])

    # Test with negative slice
    assert dq[-4:-1] == pdeque([2, 3, 4])
    assert dq[-3:] == pdeque([3, 4, 5])
    assert dq[:-2] == pdeque([1, 2, 3])

    # Test with step in slice
    assert dq[1:5:2] == pdeque([2, 4])
    assert dq[::3] == pdeque([1, 4])
    assert dq[2::2] == pdeque([3, 5])

    # Test with out of range index
    try:
        dq[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with empty deque
    empty_dq = pdeque()
    try:
        empty_dq[0]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with maxlen
    dq_maxlen = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq_maxlen[0] == 3
    assert dq_maxlen[-1] == 5
    assert dq_maxlen[:2] == pdeque([3, 4], maxlen=3)

    print("All tests passed!")

# Run the unit test
test_PDeque___getitem__()


# LLM-generated content at query #6
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    # Test with integer index
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq[0] == 1
    assert dq[2] == 3
    assert dq[-1] == 5
    assert dq[-3] == 3

    # Test with slice
    assert dq[1:4] == pdeque([2, 3, 4])
    assert dq[:3] == pdeque([1, 2, 3])
    assert dq[2:] == pdeque([3, 4, 5])
    assert dq[::2] == pdeque([1, 3, 5])
    assert dq[::-1] == pdeque([5, 4, 3, 2, 1])

    # Test with maxlen
    dq_maxlen = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq_maxlen[0] == 3
    assert dq_maxlen[-1] == 5
    assert dq_maxlen[:2] == pdeque([3, 4], maxlen=3)

    # Test with empty deque
    dq_empty = pdeque()
    try:
        dq_empty[0]
        assert False, "Should raise IndexError"
    except IndexError:
        pass

    # Test with out of range index
    try:
        dq[10]
        assert False, "Should raise IndexError"
    except IndexError:
        pass

    # Test with negative out of range index
    try:
        dq[-10]
        assert False, "Should raise IndexError"
    except IndexError:
        pass

    # Test with non-integer index
    try:
        dq["invalid"]
        assert False, "Should raise TypeError"
    except TypeError:
        pass

    print("All tests passed!")

# Run the unit test
test_PDeque___getitem__()


# LLM-generated content at query #7
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    # Test with integer index
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq[0] == 1
    assert dq[2] == 3
    assert dq[-1] == 5
    assert dq[-3] == 3

    # Test with slice
    assert dq[1:4] == pdeque([2, 3, 4])
    assert dq[:3] == pdeque([1, 2, 3])
    assert dq[2:] == pdeque([3, 4, 5])
    assert dq[::2] == pdeque([1, 3, 5])

    # Test with negative slice
    assert dq[-4:-1] == pdeque([2, 3, 4])
    assert dq[-3:] == pdeque([3, 4, 5])
    assert dq[:-2] == pdeque([1, 2, 3])

    # Test with step in slice
    assert dq[1:5:2] == pdeque([2, 4])
    assert dq[::3] == pdeque([1, 4])
    assert dq[2::2] == pdeque([3, 5])

    # Test with empty slice
    assert dq[5:] == pdeque([])
    assert dq[:0] == pdeque([])

    # Test with out of range index
    try:
        dq[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    try:
        dq[-10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with non-integer index
    try:
        dq["invalid"]
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen
    dq_maxlen = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq_maxlen[0] == 3
    assert dq_maxlen[-1] == 5
    assert dq_maxlen[:2] == pdeque([3, 4], maxlen=3)

    # Test with single element
    dq_single = pdeque([42])
    assert dq_single[0] == 42
    assert dq_single[-1] == 42

    # Test with empty deque
    dq_empty = pdeque([])
    try:
        dq_empty[0]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    print("All tests passed!")

# Run the unit test
test_PDeque___getitem__()


# LLM-generated content at query #8
#--------------------------

# Unit test for method remove of class PDeque
def test_PDeque_remove(): 
    # Test case 1: remove existing element from left list
    dq = pdeque([2, 1, 2])
    result = dq.remove(2)
    expected = pdeque([1, 2])
    assert result == expected

    # Test case 2: remove existing element from right list
    dq = pdeque([1, 2, 3])
    result = dq.remove(3)
    expected = pdeque([1, 2])
    assert result == expected

    # Test case 3: remove non-existing element
    dq = pdeque([1, 2, 3])
    try:
        dq.remove(4)
        assert False, "ValueError should be raised"
    except ValueError as e:
        assert str(e) == "4 not found in PDeque"

    # Test case 4: remove element from empty deque
    dq = pdeque([])
    try:
        dq.remove(1)
        assert False, "ValueError should be raised"
    except ValueError as e:
        assert str(e) == "1 not found in PDeque"

    # Test case 5: remove element from deque with single element
    dq = pdeque([1])
    result = dq.remove(1)
    expected = pdeque([])
    assert result == expected

    # Test case 6: remove element from deque with duplicate elements
    dq = pdeque([1, 2, 1, 3, 1])
    result = dq.remove(1)
    expected = pdeque([2, 1, 3, 1])
    assert result == expected

    # Test case 7: remove element from deque with maxlen
    dq = pdeque([1, 2, 3], maxlen=2)
    result = dq.remove(2)
    expected = pdeque([3], maxlen=2)
    assert result == expected

    # Test case 8: remove element from deque with maxlen and left list
    dq = pdeque([1, 2, 3], maxlen=2)
    result = dq.remove(1)
    expected = pdeque([3], maxlen=2)
    assert result == expected

    # Test case 9: remove element from deque with maxlen and right list
    dq = pdeque([1, 2, 3], maxlen=2)
    result = dq.remove(3)
    expected = pdeque([2], maxlen=2)
    assert result == expected

    # Test case 10: remove element from deque with maxlen and both lists
    dq = pdeque([1, 2, 3, 4], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4], maxlen=3)
    assert result == expected

    print("All test cases passed")

test_PDeque_remove()


# LLM-generated content at query #9
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    # Test with integer index
    d = pdeque([1, 2, 3, 4, 5])
    assert d[0] == 1
    assert d[2] == 3
    assert d[-1] == 5
    assert d[-3] == 3

    # Test with slice
    assert d[1:3] == pdeque([2, 3])
    assert d[:3] == pdeque([1, 2, 3])
    assert d[2:] == pdeque([3, 4, 5])
    assert d[::2] == pdeque([1, 3, 5])
    assert d[::-1] == pdeque([5, 4, 3, 2, 1])

    # Test with maxlen
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[0] == 3
    assert d[1] == 4
    assert d[2] == 5
    assert d[-1] == 5
    assert d[-2] == 4
    assert d[-3] == 3

    # Test with empty deque
    d = pdeque()
    try:
        d[0]
        assert False, "Should raise IndexError"
    except IndexError:
        pass

    # Test with out of range index
    d = pdeque([1, 2, 3])
    try:
        d[5]
        assert False, "Should raise IndexError"
    except IndexError:
        pass

    # Test with negative out of range index
    try:
        d[-5]
        assert False, "Should raise IndexError"
    except IndexError:
        pass

    # Test with non-integer index
    try:
        d["a"]
        assert False, "Should raise TypeError"
    except TypeError:
        pass

    # Test with slice step
    d = pdeque([1, 2, 3, 4, 5])
    assert d[1:4:2] == pdeque([2, 4])
    assert d[::3] == pdeque([1, 4])
    assert d[::-2] == pdeque([5, 3, 1])

    # Test with maxlen and slice
    d = pdeque([1, 2, 3, 4, 5], maxlen=4)
    assert d[1:3] == pdeque([2, 3])
    assert d[:2] == pdeque([2, 3])
    assert d[2:] == pdeque([4, 5])
    assert d[::2] == pdeque([2, 4])
    assert d[::-1] == pdeque([5, 4, 3, 2])

    # Test with maxlen and negative slice
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[-2:] == pdeque([4, 5])
    assert d[:-1] == pdeque([3, 4])
    assert d[-3:-1] == pdeque([3, 4])

    # Test with maxlen and slice step
    d = pdeque([1, 2, 3, 4, 5], maxlen=4)
    assert d[::2] == pdeque([2, 4])
    assert d[::-2] == pdeque([5, 3])

    # Test with maxlen and slice step negative
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[::-1] == pdeque([5, 4, 3])
    assert d[::-2] == pdeque([5, 3])

    # Test with maxlen and slice step out of range
    d = pdeque([1, 2, 3, 4, 5], maxlen=2)
    assert d[::3] == pdeque([4])
    assert d[::-3] == pdeque([5])

    # Test with maxlen and slice step zero
    try:
        d[::0]
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # Test with maxlen and slice step negative zero
    try:
        d[::-0]
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # Test with maxlen and slice step negative one
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[::-1] == pdeque([5, 4, 3])

    # Test with maxlen and slice step negative two
    assert d[::-2] == pdeque([5, 3])

    # Test with maxlen and slice step negative three
    assert d[::-3] == pdeque([5])

    # Test with maxlen and slice step negative four
    assert d[::-4] == pdeque([5])

    # Test with maxlen and slice step negative five
    assert d[::-5] == pdeque([5])

    # Test with maxlen and slice step negative six
    assert d[::-6] == pdeque([5])

    # Test with maxlen and slice step negative seven
    assert d[::-7] == pdeque([5])

    # Test with maxlen and slice step negative eight
    assert d[::-8] == pdeque([5])

    # Test with maxlen and slice step negative nine
    assert d[::-9] == pdeque([5])

    # Test with maxlen and slice step negative ten
    assert d[::-10] == pdeque([5])

    # Test with maxlen and slice step negative eleven
    assert d[::-11] == pdeque([5])

    # Test with maxlen and slice step negative twelve
    assert d[::-12] == pdeque([5])

    # Test with maxlen and slice step negative thirteen
    assert d[::-13] == pdeque([5])

    # Test with maxlen and slice step negative fourteen
    assert d[::-14] == pdeque([5])

    # Test with maxlen and slice step negative fifteen
    assert d[::-15] == pdeque([5])

    # Test with maxlen and slice step negative sixteen
    assert d[::-16] == pdeque([5])

    # Test with maxlen and slice step negative seventeen
    assert d[::-17] == pdeque([5])

    # Test with maxlen and slice step negative eighteen
    assert d[::-18] == pdeque([5])

    # Test with maxlen and slice step negative nineteen
    assert d[::-19] == pdeque([5])

    # Test with maxlen and slice step negative twenty
    assert d[::-20] == pdeque([5])

    # Test with maxlen and slice step negative twenty-one
    assert d[::-21] == pdeque([5])

    # Test with maxlen and slice step negative twenty-two
    assert d[::-22] == pdeque([5])

    # Test with maxlen and slice step negative twenty-three
    assert d[::-23] == pdeque([5])

    # Test with maxlen and slice step negative twenty-four
    assert d[::-24] == pdeque([5])

    # Test with maxlen and slice step negative twenty-five
    assert d[::-25] == pdeque([5])

    # Test with maxlen and slice step negative twenty-six
    assert d[::-26] == pdeque([5])

    # Test with maxlen and slice step negative twenty-seven
    assert d[::-27] == pdeque([5])

    # Test with maxlen and slice step negative twenty-eight
    assert d[::-28] == pdeque([5])

    # Test with maxlen and slice step negative twenty-nine
    assert d[::-29] == pdeque([5])

    # Test with maxlen and slice step negative thirty
    assert d[::-30] == pdeque([5])

    # Test with maxlen and slice step negative thirty-one
    assert d[::-31] == pdeque([5])

    # Test with maxlen and slice step negative thirty-two
    assert d[::-32] == pdeque([5])

    # Test with maxlen and slice step negative thirty-three
    assert d[::-33] == pdeque([5])

    # Test with maxlen and slice step negative thirty-four
    assert d[::-34] == pdeque([5])

    # Test with maxlen and slice step negative thirty-five
    assert d[::-35] == pdeque([5])

    # Test with maxlen and slice step negative thirty-six
    assert d[::-36] == pdeque([5])

    # Test with maxlen and slice step negative thirty-seven
    assert d[::-37] == pdeque([5])

    # Test with maxlen and slice step negative thirty-eight
    assert d[::-38] == pdeque([5])

    # Test with maxlen and slice step negative thirty-nine
    assert d[::-39] == pdeque([5])

    # Test with maxlen and slice step negative forty
    assert d[::-40] ==


# LLM-generated content at query #10
#--------------------------

# Unit test for method remove of class PDeque
def test_PDeque_remove(): 
    # Test case 1: remove existing element from left list
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.remove(3)
    expected = pdeque([1, 2, 4, 5])
    assert result == expected

    # Test case 2: remove existing element from right list
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.remove(5)
    expected = pdeque([1, 2, 3, 4])
    assert result == expected

    # Test case 3: remove non-existing element
    dq = pdeque([1, 2, 3, 4, 5])
    try:
        dq.remove(6)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 4: remove element from empty deque
    dq = pdeque([])
    try:
        dq.remove(1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 5: remove element from deque with duplicate values
    dq = pdeque([1, 2, 3, 2, 1])
    result = dq.remove(2)
    expected = pdeque([1, 3, 2, 1])
    assert result == expected

    # Test case 6: remove element from deque with maxlen
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    result = dq.remove(3)
    expected = pdeque([1, 2, 4, 5], maxlen=4)
    assert result == expected

    # Test case 7: remove element from deque with maxlen and left list empty
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq.remove(3)
    expected = pdeque([1, 2, 4, 5], maxlen=3)
    assert result == expected

    # Test case 8: remove element from deque with maxlen and right list empty
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq.remove(5)
    expected = pdeque([1, 2, 3, 4], maxlen=3)
    assert result == expected

    # Test case 9: remove element from deque with maxlen and both lists non-empty
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4, 5], maxlen=4)
    assert result == expected

    # Test case 10: remove element from deque with maxlen and both lists non-empty, element in right list
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    result = dq.remove(4)
    expected = pdeque([1, 2, 3, 5], maxlen=4)
    assert result == expected

    print("All test cases passed")

test_PDeque_remove()


# LLM-generated content at query #11
#--------------------------

# Unit test for method remove of class PDeque
def test_PDeque_remove(): 
    # Test case 1: remove existing element from left list
    dq = pdeque([2, 1, 2])
    result = dq.remove(2)
    expected = pdeque([1, 2])
    assert result == expected

    # Test case 2: remove existing element from right list
    dq = pdeque([1, 2, 3])
    result = dq.remove(3)
    expected = pdeque([1, 2])
    assert result == expected

    # Test case 3: remove non-existing element
    dq = pdeque([1, 2, 3])
    try:
        dq.remove(4)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "4 not found in PDeque"

    # Test case 4: remove element from empty deque
    dq = pdeque([])
    try:
        dq.remove(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "1 not found in PDeque"

    # Test case 5: remove element from deque with single element
    dq = pdeque([1])
    result = dq.remove(1)
    expected = pdeque([])
    assert result == expected

    # Test case 6: remove element from deque with duplicate elements
    dq = pdeque([1, 2, 1, 3, 1])
    result = dq.remove(1)
    expected = pdeque([2, 1, 3, 1])
    assert result == expected

    # Test case 7: remove element from deque with maxlen
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3], maxlen=3)
    assert result == expected

    # Test case 8: remove element from deque with maxlen and left list
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(1)
    expected = pdeque([2, 3], maxlen=3)
    assert result == expected

    # Test case 9: remove element from deque with maxlen and right list
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(3)
    expected = pdeque([1, 2], maxlen=3)
    assert result == expected

    # Test case 10: remove element from deque with maxlen and both lists
    dq = pdeque([1, 2, 3, 4], maxlen=4)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4], maxlen=4)
    assert result == expected

    print("All test cases passed")

test_PDeque_remove()


# LLM-generated content at query #12
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    # Test with integer index
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq[0] == 1
    assert dq[2] == 3
    assert dq[-1] == 5
    assert dq[-3] == 3

    # Test with slice
    assert dq[1:4] == pdeque([2, 3, 4])
    assert dq[:3] == pdeque([1, 2, 3])
    assert dq[2:] == pdeque([3, 4, 5])
    assert dq[::2] == pdeque([1, 3, 5])
    assert dq[::-1] == pdeque([5, 4, 3, 2, 1])

    # Test with negative slice indices
    assert dq[-4:-1] == pdeque([2, 3, 4])
    assert dq[-3:] == pdeque([3, 4, 5])
    assert dq[:-2] == pdeque([1, 2, 3])

    # Test with step in slice
    assert dq[1:5:2] == pdeque([2, 4])
    assert dq[::3] == pdeque([1, 4])
    assert dq[2::2] == pdeque([3, 5])

    # Test with empty slice
    assert dq[5:] == pdeque([])
    assert dq[:0] == pdeque([])

    # Test with out of range index
    try:
        dq[10]
        assert False, "IndexError should have been raised"
    except IndexError:
        pass

    try:
        dq[-10]
        assert False, "IndexError should have been raised"
    except IndexError:
        pass

    # Test with non-integer index
    try:
        dq["invalid"]
        assert False, "TypeError should have been raised"
    except TypeError:
        pass

    # Test with maxlen
    dq_maxlen = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq_maxlen[0] == 3
    assert dq_maxlen[-1] == 5
    assert dq_maxlen[:2] == pdeque([3, 4], maxlen=3)

    # Test with single element deque
    dq_single = pdeque([42])
    assert dq_single[0] == 42
    assert dq_single[-1] == 42
    assert dq_single[:] == pdeque([42])

    # Test with empty deque
    dq_empty = pdeque([])
    try:
        dq_empty[0]
        assert False, "IndexError should have been raised"
    except IndexError:
        pass

    assert dq_empty[:] == pdeque([])

    print("All tests passed!")

# Run the unit test
test_PDeque___getitem__()


# LLM-generated content at query #13
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    # Test with integer index
    d = pdeque([1, 2, 3, 4, 5])
    assert d[0] == 1
    assert d[2] == 3
    assert d[-1] == 5
    assert d[-3] == 3

    # Test with slice
    assert d[1:3] == pdeque([2, 3])
    assert d[:3] == pdeque([1, 2, 3])
    assert d[2:] == pdeque([3, 4, 5])
    assert d[::2] == pdeque([1, 3, 5])
    assert d[::-1] == pdeque([5, 4, 3, 2, 1])

    # Test with maxlen
    d2 = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d2[0] == 3
    assert d2[-1] == 5
    assert d2[:2] == pdeque([3, 4], maxlen=3)

    # Test with empty deque
    d3 = pdeque()
    try:
        d3[0]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test with out of range index
    try:
        d[10]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test with negative out of range index
    try:
        d[-10]
        assert False, "Should have raised IndexError"
    except IndexError:
        pass

    # Test with non-integer index
    try:
        d["invalid"]
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    print("All tests passed for __getitem__")

# Run the test
test_PDeque___getitem__()


# LLM-generated content at query #14
#--------------------------

# Unit test for method __eq__ of class PDeque
def test_PDeque___eq__(): 
    # Test case 1: Compare two empty deques
    d1 = pdeque()
    d2 = pdeque()
    assert d1 == d2

    # Test case 2: Compare two deques with same elements
    d1 = pdeque([1, 2, 3])
    d2 = pdeque([1, 2, 3])
    assert d1 == d2

    # Test case 3: Compare two deques with different elements
    d1 = pdeque([1, 2, 3])
    d2 = pdeque([4, 5, 6])
    assert not (d1 == d2)

    # Test case 4: Compare deque with non-deque object
    d1 = pdeque([1, 2, 3])
    assert not (d1 == [1, 2, 3])

    # Test case 5: Compare deque with different length
    d1 = pdeque([1, 2, 3])
    d2 = pdeque([1, 2])
    assert not (d1 == d2)

    # Test case 6: Compare deque with same elements but different order
    d1 = pdeque([1, 2, 3])
    d2 = pdeque([3, 2, 1])
    assert not (d1 == d2)

    # Test case 7: Compare deque with same elements and same order but different maxlen
    d1 = pdeque([1, 2, 3], maxlen=5)
    d2 = pdeque([1, 2, 3], maxlen=10)
    assert d1 == d2

    # Test case 8: Compare deque with same elements and same order but different maxlen (one bounded, one unbounded)
    d1 = pdeque([1, 2, 3], maxlen=5)
    d2 = pdeque([1, 2, 3])
    assert d1 == d2

    # Test case 9: Compare deque with same elements and same order but different maxlen (both bounded)
    d1 = pdeque([1, 2, 3], maxlen=5)
    d2 = pdeque([1, 2, 3], maxlen=3)
    assert d1 == d2

    # Test case 10: Compare deque with same elements and same order but different maxlen (both bounded, one full)
    d1 = pdeque([1, 2, 3], maxlen=3)
    d2 = pdeque([1, 2, 3], maxlen=5)
    assert d1 == d2

    # Test case 11: Compare deque with same elements and same order but different maxlen (both bounded, both full)
    d1 = pdeque([1, 2, 3], maxlen=3)
    d2 = pdeque([1, 2, 3], maxlen=3)
    assert d1 == d2

    # Test case 12: Compare deque with same elements and same order but different maxlen (both bounded, one empty)
    d1 = pdeque([], maxlen=3)
    d2 = pdeque([], maxlen=5)
    assert d1 == d2

    # Test case 13: Compare deque with same elements and same order but different maxlen (both bounded, both empty)
    d1 = pdeque([], maxlen=3)
    d2 = pdeque([], maxlen=3)
    assert d1 == d2

    # Test case 14: Compare deque with same elements and same order but different maxlen (one bounded, one empty)
    d1 = pdeque([], maxlen=3)
    d2 = pdeque([])
    assert d1 == d2

    # Test case 15: Compare deque with same elements and same order but different maxlen (both bounded, one with extra capacity)
    d1 = pdeque([1, 2, 3], maxlen=5)
    d2 = pdeque([1, 2, 3], maxlen=3)
    assert d1 == d2

    # Test case 16: Compare deque with same elements and same order but different maxlen (both bounded, one with extra capacity, one full)
    d1 = pdeque([1, 2, 3], maxlen=5)
    d2 = pdeque([1, 2, 3], maxlen=3)
    assert d1 == d2

    # Test case 17: Compare deque with same elements and same order but different maxlen (both bounded, both with extra capacity)
    d1 = pdeque([1, 2, 3], maxlen=5)
    d2 = pdeque([1, 2, 3], maxlen=10)
    assert d1 == d2

    # Test case 18: Compare deque with same elements and same order but different maxlen (both bounded, both with extra capacity, one empty)
    d1 = pdeque([], maxlen=5)
    d2 = pdeque([], maxlen=10)
    assert d1 == d2

    # Test case 19: Compare deque with same elements and same order but different maxlen (both bounded, both with extra capacity, both empty)
    d1 = pdeque([], maxlen=5)
    d2 = pdeque([], maxlen=5)
    assert d1 == d2

    # Test case 20: Compare deque with same elements and same order but different maxlen (one bounded, one with extra capacity)
    d1 = pdeque([1, 2, 3], maxlen=5)
    d2 = pdeque([1, 2, 3])
    assert d1 == d2

    # Test case 21: Compare deque with same elements and same order but different maxlen (one bounded, one with extra capacity, one empty)
    d1 = pdeque([], maxlen=5)
    d2 = pdeque([])
    assert d1 == d2

    # Test case 22: Compare deque with same elements and same order but different maxlen (one bounded, one with extra capacity, both empty)
    d1 = pdeque([], maxlen=5)
    d2 = pdeque([], maxlen=5)
    assert d1 == d2

    # Test case 23: Compare deque with same elements and same order but different maxlen (both bounded, both with extra capacity, one full)
    d1 = pdeque([1, 2, 3], maxlen=5)
    d2 = pdeque([1, 2, 3], maxlen=3)
    assert d1 == d2

    # Test case 24: Compare deque with same elements and same order but different maxlen (both bounded, both with extra capacity, both full)
    d1 = pdeque([1, 2, 3], maxlen=5)
    d2 = pdeque([1, 2, 3], maxlen=3)
    assert d1 == d2

    # Test case 25: Compare deque with same elements and same order but different maxlen (both bounded, both with extra capacity, one empty, one full)
    d1 = pdeque([], maxlen=5)
    d2 = pdeque([1, 2, 3], maxlen=3)
    assert not (d1 == d2)

    # Test case 26: Compare deque with same elements and same order but different maxlen (both bounded, both with extra capacity, both empty, one full)
    d1 = pdeque([], maxlen=5)
    d2 = pdeque([1, 2, 3], maxlen=3)
    assert not (d1 == d2)

    # Test case 27: Compare deque with same elements and same order but different maxlen (both bounded, both with extra capacity, both empty, both full)
    d1 = pdeque([], maxlen=5)
    d2 = pdeque([], maxlen=3)
    assert d1 == d2

    # Test case 28: Compare deque with same elements and same order but different maxlen (both bounded, both with extra capacity, both empty, both full, one with extra capacity)
    d1 = pdeque([], maxlen=5)
    d2 = pdeque([], maxlen=10)
    assert d1 == d2

    # Test case 29: Compare deque with same elements and same order but different maxlen (both bounded, both with extra capacity, both empty, both full, both with extra capacity)
    d1 = pdeque([], maxlen=5)
    d2 = pdeque([], maxlen=5)
    assert d1 == d2

    # Test case 30: Compare deque with same elements and same order but different maxlen (both bounded, both with extra capacity, both empty, both full, both with extra capacity, one empty)
    d1 = pdeque([], maxlen=5)
    d2 = pdeque([], maxlen=10)
    assert d1 == d2

    #


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method rotate of class PDeque
def test_PDeque_rotate(): 
    # Test case 1: Rotate by positive steps
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq.rotate(2) == pdeque([4, 5, 1, 2, 3])
    
    # Test case 2: Rotate by negative steps
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq.rotate(-2) == pdeque([3, 4, 5, 1, 2])
    
    # Test case 3: Rotate by zero steps
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq.rotate(0) == pdeque([1, 2, 3, 4, 5])
    
    # Test case 4: Rotate by steps greater than length
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq.rotate(7) == pdeque([4, 5, 1, 2, 3])
    
    # Test case 5: Rotate by negative steps greater than length
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq.rotate(-7) == pdeque([3, 4, 5, 1, 2])
    
    # Test case 6: Rotate empty deque
    dq = pdeque([])
    assert dq.rotate(3) == pdeque([])
    
    # Test case 7: Rotate deque with one element
    dq = pdeque([1])
    assert dq.rotate(5) == pdeque([1])
    
    # Test case 8: Rotate deque with two elements
    dq = pdeque([1, 2])
    assert dq.rotate(1) == pdeque([2, 1])
    
    # Test case 9: Rotate deque with two elements by negative steps
    dq = pdeque([1, 2])
    assert dq.rotate(-1) == pdeque([2, 1])
    
    # Test case 10: Rotate deque with maxlen
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    assert dq.rotate(2) == pdeque([4, 5, 1, 2], maxlen=4)
    
    # Test case 11: Rotate deque with maxlen by negative steps
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    assert dq.rotate(-2) == pdeque([3, 4, 5, 1], maxlen=4)
    
    # Test case 12: Rotate deque with maxlen and steps greater than length
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    assert dq.rotate(7) == pdeque([4, 5, 1, 2], maxlen=4)
    
    # Test case 13: Rotate deque with maxlen and negative steps greater than length
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    assert dq.rotate(-7) == pdeque([3, 4, 5, 1], maxlen=4)
    
    # Test case 14: Rotate deque with maxlen and zero steps
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    assert dq.rotate(0) == pdeque([2, 3, 4, 5], maxlen=4)
    
    # Test case 15: Rotate deque with maxlen and one element
    dq = pdeque([1], maxlen=1)
    assert dq.rotate(5) == pdeque([1], maxlen=1)
    
    # Test case 16: Rotate deque with maxlen and two elements
    dq = pdeque([1, 2], maxlen=2)
    assert dq.rotate(1) == pdeque([2, 1], maxlen=2)
    
    # Test case 17: Rotate deque with maxlen and two elements by negative steps
    dq = pdeque([1, 2], maxlen=2)
    assert dq.rotate(-1) == pdeque([2, 1], maxlen=2)
    
    # Test case 18: Rotate deque with maxlen and empty deque
    dq = pdeque([], maxlen=3)
    assert dq.rotate(3) == pdeque([], maxlen=3)
    
    # Test case 19: Rotate deque with maxlen and steps equal to length
    dq = pdeque([1, 2, 3, 4, 5], maxlen=5)
    assert dq.rotate(5) == pdeque([1, 2, 3, 4, 5], maxlen=5)
    
    # Test case 20: Rotate deque with maxlen and negative steps equal to length
    dq = pdeque([1, 2, 3, 4, 5], maxlen=5)
    assert dq.rotate(-5) == pdeque([1, 2, 3, 4, 5], maxlen=5)
    
    # Test case 21: Rotate deque with maxlen and steps equal to maxlen
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.rotate(3) == pdeque([3, 4, 5], maxlen=3)
    
    # Test case 22: Rotate deque with maxlen and negative steps equal to maxlen
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.rotate(-3) == pdeque([3, 4, 5], maxlen=3)
    
    # Test case 23: Rotate deque with maxlen and steps greater than maxlen
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.rotate(7) == pdeque([3, 4, 5], maxlen=3)
    
    # Test case 24: Rotate deque with maxlen and negative steps greater than maxlen
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.rotate(-7) == pdeque([3, 4, 5], maxlen=3)
    
    # Test case 25: Rotate deque with maxlen and steps equal to zero
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.rotate(0) == pdeque([3, 4, 5], maxlen=3)
    
    # Test case 26: Rotate deque with maxlen and negative steps equal to zero
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.rotate(-0) == pdeque([3, 4, 5], maxlen=3)
    
    # Test case 27: Rotate deque with maxlen and steps equal to one
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.rotate(1) == pdeque([5, 3, 4], maxlen=3)
    
    # Test case 28: Rotate deque with maxlen and negative steps equal to one
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.rotate(-1) == pdeque([4, 5, 3], maxlen=3)
    
    # Test case 29: Rotate deque with maxlen and steps equal to two
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.rotate(2) == pdeque([4, 5, 3], maxlen=3)
    
    # Test case 30: Rotate deque with maxlen and negative steps equal to two
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.rotate(-2) == pdeque([5, 3, 4], maxlen=3)
    
    # Test case 31: Rotate deque with maxlen and steps equal to three
    dq = pdeque([1, 2


# LLM-generated content at query #2
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    # Test with integer index
    d = pdeque([1, 2, 3, 4, 5])
    assert d[0] == 1
    assert d[2] == 3
    assert d[-1] == 5
    assert d[-3] == 3

    # Test with slice
    assert d[1:3] == pdeque([2, 3])
    assert d[:3] == pdeque([1, 2, 3])
    assert d[2:] == pdeque([3, 4, 5])
    assert d[::2] == pdeque([1, 3, 5])
    assert d[::-1] == pdeque([5, 4, 3, 2, 1])

    # Test with maxlen
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[0] == 3
    assert d[1] == 4
    assert d[2] == 5
    assert d[-1] == 5
    assert d[-2] == 4
    assert d[-3] == 3

    # Test with empty deque
    d = pdeque()
    with pytest.raises(IndexError):
        d[0]
    with pytest.raises(IndexError):
        d[-1]

    # Test with single element deque
    d = pdeque([42])
    assert d[0] == 42
    assert d[-1] == 42

    # Test with large deque
    d = pdeque(range(1000))
    assert d[500] == 500
    assert d[-500] == 500

    # Test with slice and maxlen
    d = pdeque([1, 2, 3, 4, 5], maxlen=4)
    assert d[1:3] == pdeque([2, 3], maxlen=4)
    assert d[:3] == pdeque([1, 2, 3], maxlen=4)
    assert d[2:] == pdeque([3, 4, 5], maxlen=4)

    # Test with step in slice
    d = pdeque([1, 2, 3, 4, 5])
    assert d[1:4:2] == pdeque([2, 4])
    assert d[::3] == pdeque([1, 4])
    assert d[4:1:-2] == pdeque([5, 3])

    # Test with invalid index type
    d = pdeque([1, 2, 3])
    with pytest.raises(TypeError):
        d['invalid']
    with pytest.raises(TypeError):
        d[2.5]

    # Test with out of range index
    d = pdeque([1, 2, 3])
    with pytest.raises(IndexError):
        d[10]
    with pytest.raises(IndexError):
        d[-10]

    # Test that slice returns new PDeque with same maxlen
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    sliced = d[1:3]
    assert sliced.maxlen == 3
    assert sliced == pdeque([2, 3], maxlen=3)

    # Test negative slice indices
    d = pdeque([1, 2, 3, 4, 5])
    assert d[-3:-1] == pdeque([3, 4])
    assert d[-4:] == pdeque([2, 3, 4, 5])
    assert d[:-2] == pdeque([1, 2, 3])

    # Test slice with step and maxlen
    d = pdeque([1, 2, 3, 4, 5], maxlen=4)
    assert d[::2] == pdeque([1, 3, 5], maxlen=4)  # Note: maxlen is preserved but may not be meaningful

    # Test that original deque is unchanged
    d = pdeque([1, 2, 3, 4, 5])
    _ = d[1:4]
    assert d == pdeque([1, 2, 3, 4, 5])

    # Test with single element slice
    d = pdeque([1, 2, 3, 4, 5])
    assert d[2:3] == pdeque([3])
    assert d[-2:-1] == pdeque([4])

    # Test empty slice
    d = pdeque([1, 2, 3, 4, 5])
    assert d[3:3] == pdeque([])
    assert d[10:15] == pdeque([])
    assert d[-10:-5] == pdeque([])

    # Test slice with None indices
    d = pdeque([1, 2, 3, 4, 5])
    assert d[None:3] == pdeque([1, 2, 3])
    assert d[2:None] == pdeque([3, 4, 5])
    assert d[None:None] == pdeque([1, 2, 3, 4, 5])

    # Test that slice step of 1 uses structural sharing
    d = pdeque([1, 2, 3, 4, 5])
    sliced = d[1:4]
    # Since step is 1, should use structural sharing (pop/popleft)
    # We can't directly test this, but we can verify the result is correct
    assert sliced == pdeque([2, 3, 4])

    # Test that slice with step != 1 creates new deque
    d = pdeque([1, 2, 3, 4, 5])
    sliced = d[::2]
    # This should create a new deque from tuple
    assert sliced == pdeque([1, 3, 5])

    # Test with very large step
    d = pdeque(range(100))
    assert d[::10] == pdeque(range(0, 100, 10))

    # Test negative step
    d = pdeque([1, 2, 3, 4, 5])
    assert d[::-1] == pdeque([5, 4, 3, 2, 1])
    assert d[::-2] == pdeque([5, 3, 1])
    assert d[4:1:-1] == pdeque([5, 4, 3])
    assert d[4:1:-2] == pdeque([5, 3])

    # Test complex slice with negative start and stop
    d = pdeque([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert d[-3:2:-1] == pdeque([8, 7, 6, 5, 4])
    assert d[2:-3:1] == pdeque([3, 4, 5, 6, 7])
    assert d[-2:-8:-2] == pdeque([9, 7, 5])

    # Test that maxlen is preserved in all slice cases
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[:].maxlen == 3
    assert d[::2].maxlen == 3
    assert d[::-1].maxlen == 3
    assert d[1:4].maxlen == 3

    # Test with bounded deque and slice that would exceed bounds
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    # d is actually [3, 4, 5] due to maxlen
    assert d[:] == pdeque([3, 4, 5], maxlen=3)
    assert d[1:] == pdeque([4, 5], maxlen=3)
    assert d[:2] == pdeque([3, 4], maxlen=3)

    # Test index with bounded deque
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[0] == 3
    assert d[1] == 4
    assert d[2] == 5
    with pytest.raises(IndexError):
        d[3]

    # Test negative index with bounded deque
    assert d[-1] == 5
    assert d[-2] == 4
    assert d[-3] == 3
    with pytest.raises(IndexError):
        d[-4]

    # Test that __getitem__ works with subclass
    class SubclassDeque(PDeque):
        pass

    d = SubclassDeque(plist([1, 2]), plist([


# LLM-generated content at query #3
#--------------------------

# Unit test for method remove of class PDeque
def test_PDeque_remove(): 
    # Test case 1: remove element from left list
    dq = pdeque([2, 1, 2])
    result = dq.remove(2)
    expected = pdeque([1, 2])
    assert result == expected

    # Test case 2: remove element from right list
    dq = pdeque([1, 2, 3])
    result = dq.remove(3)
    expected = pdeque([1, 2])
    assert result == expected

    # Test case 3: remove element not present in deque
    dq = pdeque([1, 2, 3])
    try:
        dq.remove(4)
        assert False, "ValueError should be raised"
    except ValueError as e:
        assert str(e) == "4 not found in PDeque"

    # Test case 4: remove element from empty deque
    dq = pdeque([])
    try:
        dq.remove(1)
        assert False, "ValueError should be raised"
    except ValueError as e:
        assert str(e) == "1 not found in PDeque"

    # Test case 5: remove element from deque with single element
    dq = pdeque([1])
    result = dq.remove(1)
    expected = pdeque([])
    assert result == expected

    # Test case 6: remove element from deque with duplicate elements
    dq = pdeque([1, 2, 1, 3, 1])
    result = dq.remove(1)
    expected = pdeque([2, 1, 3, 1])
    assert result == expected

    # Test case 7: remove element from deque with maxlen
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3], maxlen=3)
    assert result == expected

    # Test case 8: remove element from deque with maxlen and element not present
    dq = pdeque([1, 2, 3], maxlen=3)
    try:
        dq.remove(4)
        assert False, "ValueError should be raised"
    except ValueError as e:
        assert str(e) == "4 not found in PDeque"

    # Test case 9: remove element from deque with maxlen and element at the leftmost position
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(1)
    expected = pdeque([2, 3], maxlen=3)
    assert result == expected

    # Test case 10: remove element from deque with maxlen and element at the rightmost position
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(3)
    expected = pdeque([1, 2], maxlen=3)
    assert result == expected

    print("All test cases passed")

test_PDeque_remove()


# LLM-generated content at query #4
#--------------------------

# Unit test for method pop of class PDeque
def test_PDeque_pop(): 
    # Test popping from an empty deque
    dq = pdeque([])
    assert dq.pop() == pdeque([])
    
    # Test popping from a deque with one element
    dq = pdeque([1])
    assert dq.pop() == pdeque([])
    
    # Test popping from a deque with multiple elements
    dq = pdeque([1, 2, 3])
    assert dq.pop() == pdeque([1, 2])
    assert dq.pop(2) == pdeque([1])
    
    # Test popping with negative count (should popleft)
    dq = pdeque([1, 2, 3])
    assert dq.pop(-1) == pdeque([2, 3])
    
    # Test popping more elements than exist
    dq = pdeque([1, 2])
    assert dq.pop(5) == pdeque([])
    
    # Test popping from a bounded deque
    dq = pdeque([1, 2, 3], maxlen=3)
    assert dq.pop() == pdeque([1, 2], maxlen=3)
    
    # Test popping from a bounded deque with maxlen 0
    dq = pdeque([], maxlen=0)
    assert dq.pop() == pdeque([], maxlen=0)


# LLM-generated content at query #5
#--------------------------

# Unit test for method popleft of class PDeque
def test_PDeque_popleft(): 
    # Test popleft with positive count
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.popleft(2)
    assert result == pdeque([3, 4, 5])
    
    # Test popleft with negative count (should call pop)
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.popleft(-2)
    assert result == pdeque([1, 2, 3])
    
    # Test popleft with count larger than length
    dq = pdeque([1, 2, 3])
    result = dq.popleft(5)
    assert result == pdeque([])
    
    # Test popleft on empty deque
    dq = pdeque([])
    result = dq.popleft(1)
    assert result == pdeque([])
    
    # Test popleft with maxlen
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.popleft(1)
    assert result == pdeque([2, 3], maxlen=3)
    
    # Test popleft with maxlen and count larger than length
    dq = pdeque([1, 2], maxlen=2)
    result = dq.popleft(3)
    assert result == pdeque([], maxlen=2)
    
    print("All tests passed!")

test_PDeque_popleft()


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method rotate of class PDeque
def test_PDeque_rotate(): 
    # Test case 1: rotate with positive steps
    x = pdeque([1, 2, 3])
    assert x.rotate(1) == pdeque([3, 1, 2])
    
    # Test case 2: rotate with negative steps
    x = pdeque([1, 2, 3])
    assert x.rotate(-2) == pdeque([3, 1, 2])
    
    # Test case 3: rotate with steps equal to length of deque
    x = pdeque([1, 2, 3])
    assert x.rotate(3) == pdeque([1, 2, 3])
    
    # Test case 4: rotate with steps greater than length of deque
    x = pdeque([1, 2, 3])
    assert x.rotate(4) == pdeque([3, 1, 2])
    
    # Test case 5: rotate with steps less than negative length of deque
    x = pdeque([1, 2, 3])
    assert x.rotate(-4) == pdeque([2, 3, 1])
    
    # Test case 6: rotate empty deque
    x = pdeque([])
    assert x.rotate(1) == pdeque([])
    
    # Test case 7: rotate deque with one element
    x = pdeque([1])
    assert x.rotate(1) == pdeque([1])
    
    # Test case 8: rotate deque with two elements
    x = pdeque([1, 2])
    assert x.rotate(1) == pdeque([2, 1])
    
    # Test case 9: rotate deque with two elements and negative steps
    x = pdeque([1, 2])
    assert x.rotate(-1) == pdeque([2, 1])
    
    # Test case 10: rotate deque with three elements and steps equal to half length
    x = pdeque([1, 2, 3])
    assert x.rotate(2) == pdeque([2, 3, 1])


# LLM-generated content at query #2
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    # Test with integer index
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq[0] == 1
    assert dq[2] == 3
    assert dq[-1] == 5
    assert dq[-3] == 3

    # Test with slice
    assert dq[1:4] == pdeque([2, 3, 4])
    assert dq[:3] == pdeque([1, 2, 3])
    assert dq[2:] == pdeque([3, 4, 5])
    assert dq[::2] == pdeque([1, 3, 5])

    # Test with negative slice
    assert dq[-4:-1] == pdeque([2, 3, 4])
    assert dq[-3:] == pdeque([3, 4, 5])
    assert dq[:-2] == pdeque([1, 2, 3])

    # Test with step in slice
    assert dq[1:5:2] == pdeque([2, 4])
    assert dq[::3] == pdeque([1, 4])
    assert dq[2::2] == pdeque([3, 5])

    # Test with out of range index
    try:
        dq[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with empty deque
    empty_dq = pdeque()
    try:
        empty_dq[0]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with maxlen
    bounded_dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert bounded_dq[0] == 3
    assert bounded_dq[2] == 5
    assert bounded_dq[-1] == 5
    assert bounded_dq[-3] == 3

    # Test with slice and maxlen
    assert bounded_dq[1:3] == pdeque([4, 5], maxlen=3)
    assert bounded_dq[:2] == pdeque([3, 4], maxlen=3)
    assert bounded_dq[1:] == pdeque([4, 5], maxlen=3)

    # Test with step in slice and maxlen
    assert bounded_dq[::2] == pdeque([3, 5], maxlen=3)

    # Test with negative slice and maxlen
    assert bounded_dq[-3:-1] == pdeque([3, 4], maxlen=3)
    assert bounded_dq[-2:] == pdeque([4, 5], maxlen=3)
    assert bounded_dq[:-1] == pdeque([3, 4], maxlen=3)

    print("All tests passed!")

# Run the unit test
test_PDeque___getitem__()


# LLM-generated content at query #3
#--------------------------

# Unit test for method remove of class PDeque
def test_PDeque_remove(): 
    # Test case 1: Remove element from left list
    dq = pdeque([2, 1, 2])
    result = dq.remove(2)
    expected = pdeque([1, 2])
    assert result == expected

    # Test case 2: Remove element from right list
    dq = pdeque([1, 2, 3])
    result = dq.remove(3)
    expected = pdeque([1, 2])
    assert result == expected

    # Test case 3: Remove element not present in deque
    dq = pdeque([1, 2, 3])
    try:
        dq.remove(4)
        assert False, "ValueError not raised"
    except ValueError as e:
        assert str(e) == "4 not found in PDeque"

    # Test case 4: Remove element from empty deque
    dq = pdeque([])
    try:
        dq.remove(1)
        assert False, "ValueError not raised"
    except ValueError as e:
        assert str(e) == "1 not found in PDeque"

    # Test case 5: Remove element from deque with single element
    dq = pdeque([1])
    result = dq.remove(1)
    expected = pdeque([])
    assert result == expected

    # Test case 6: Remove element from deque with duplicate elements
    dq = pdeque([1, 2, 1, 3, 1])
    result = dq.remove(1)
    expected = pdeque([2, 1, 3, 1])
    assert result == expected

    # Test case 7: Remove element from deque with maxlen
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3], maxlen=3)
    assert result == expected

    # Test case 8: Remove element from deque with maxlen and element not present
    dq = pdeque([1, 2, 3], maxlen=3)
    try:
        dq.remove(4)
        assert False, "ValueError not raised"
    except ValueError as e:
        assert str(e) == "4 not found in PDeque"

    # Test case 9: Remove element from deque with maxlen and element at the end
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(3)
    expected = pdeque([1, 2], maxlen=3)
    assert result == expected

    # Test case 10: Remove element from deque with maxlen and element at the beginning
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(1)
    expected = pdeque([2, 3], maxlen=3)
    assert result == expected

    # Test case 11: Remove element from deque with maxlen and element in the middle
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3], maxlen=3)
    assert result == expected

    # Test case 12: Remove element from deque with maxlen and element not present, but deque is full
    dq = pdeque([1, 2, 3], maxlen=3)
    try:
        dq.remove(4)
        assert False, "ValueError not raised"
    except ValueError as e:
        assert str(e) == "4 not found in PDeque"

    # Test case 13: Remove element from deque with maxlen and element present, but deque is full
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3], maxlen=3)
    assert result == expected

    # Test case 14: Remove element from deque with maxlen and element present, but deque is not full
    dq = pdeque([1, 2], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1], maxlen=3)
    assert result == expected

    # Test case 15: Remove element from deque with maxlen and element not present, but deque is not full
    dq = pdeque([1, 2], maxlen=3)
    try:
        dq.remove(3)
        assert False, "ValueError not raised"
    except ValueError as e:
        assert str(e) == "3 not found in PDeque"

    # Test case 16: Remove element from deque with maxlen and element present, but deque is empty
    dq = pdeque([], maxlen=3)
    try:
        dq.remove(1)
        assert False, "ValueError not raised"
    except ValueError as e:
        assert str(e) == "1 not found in PDeque"

    # Test case 17: Remove element from deque with maxlen and element present, but deque has one element
    dq = pdeque([1], maxlen=3)
    result = dq.remove(1)
    expected = pdeque([], maxlen=3)
    assert result == expected

    # Test case 18: Remove element from deque with maxlen and element present, but deque has two elements
    dq = pdeque([1, 2], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1], maxlen=3)
    assert result == expected

    # Test case 19: Remove element from deque with maxlen and element present, but deque has three elements
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3], maxlen=3)
    assert result == expected

    # Test case 20: Remove element from deque with maxlen and element present, but deque has four elements
    dq = pdeque([1, 2, 3, 4], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4], maxlen=3)
    assert result == expected

    # Test case 21: Remove element from deque with maxlen and element present, but deque has five elements
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4, 5], maxlen=3)
    assert result == expected

    # Test case 22: Remove element from deque with maxlen and element present, but deque has six elements
    dq = pdeque([1, 2, 3, 4, 5, 6], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4, 5, 6], maxlen=3)
    assert result == expected

    # Test case 23: Remove element from deque with maxlen and element present, but deque has seven elements
    dq = pdeque([1, 2, 3, 4, 5, 6, 7], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4, 5, 6, 7], maxlen=3)
    assert result == expected

    # Test case 24: Remove element from deque with maxlen and element present, but deque has eight elements
    dq = pdeque([1, 2, 3, 4, 5, 6, 7, 8], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4, 5, 6, 7, 8], maxlen=3)
    assert result == expected

    # Test case 25: Remove element from deque with maxlen and element present, but deque has nine elements
    dq = pdeque([1, 2, 3, 4, 5, 6, 7, 8, 9], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4, 5, 6, 7, 8, 9], maxlen=3)
    assert result == expected

    # Test case 26: Remove element from deque with maxlen and element present, but deque has ten elements
    dq = pdeque([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], maxlen=3)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4, 5, 6, 7, 8, 9, 10], maxlen=3)
    assert result ==


# LLM-generated content at query #4
#--------------------------

# Unit test for method pop of class PDeque
def test_PDeque_pop(): 
    # Test popping from an empty deque
    dq = pdeque()
    assert dq.pop() == pdeque()
    assert dq.pop(2) == pdeque()
    assert dq.pop(-1) == pdeque()

    # Test popping from a deque with one element
    dq = pdeque([1])
    assert dq.pop() == pdeque()
    assert dq.pop(2) == pdeque()
    assert dq.pop(-1) == pdeque([1])

    # Test popping from a deque with multiple elements
    dq = pdeque([1, 2, 3])
    assert dq.pop() == pdeque([1, 2])
    assert dq.pop(2) == pdeque([1])
    assert dq.pop(-1) == pdeque([2, 3])

    # Test popping with a negative count
    dq = pdeque([1, 2, 3])
    assert dq.pop(-2) == pdeque([3])

    # Test popping with a count larger than the deque length
    dq = pdeque([1, 2, 3])
    assert dq.pop(5) == pdeque()

    # Test popping from a bounded deque
    dq = pdeque([1, 2, 3], maxlen=3)
    assert dq.pop() == pdeque([1, 2], maxlen=3)
    assert dq.pop(2) == pdeque([1], maxlen=3)
    assert dq.pop(-1) == pdeque([2, 3], maxlen=3)

    # Test popping from a bounded deque with maxlen 0
    dq = pdeque([], maxlen=0)
    assert dq.pop() == pdeque([], maxlen=0)
    assert dq.pop(2) == pdeque([], maxlen=0)
    assert dq.pop(-1) == pdeque([], maxlen=0)

    # Test popping from a bounded deque with maxlen 1
    dq = pdeque([1], maxlen=1)
    assert dq.pop() == pdeque([], maxlen=1)
    assert dq.pop(2) == pdeque([], maxlen=1)
    assert dq.pop(-1) == pdeque([1], maxlen=1)

    # Test popping from a bounded deque with maxlen larger than the deque length
    dq = pdeque([1, 2, 3], maxlen=5)
    assert dq.pop() == pdeque([1, 2], maxlen=5)
    assert dq.pop(2) == pdeque([1], maxlen=5)
    assert dq.pop(-1) == pdeque([2, 3], maxlen=5)

    # Test popping from a bounded deque with maxlen equal to the deque length
    dq = pdeque([1, 2, 3], maxlen=3)
    assert dq.pop() == pdeque([1, 2], maxlen=3)
    assert dq.pop(2) == pdeque([1], maxlen=3)
    assert dq.pop(-1) == pdeque([2, 3], maxlen=3)

    # Test popping from a bounded deque with maxlen smaller than the deque length
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.pop() == pdeque([2, 3, 4], maxlen=3)
    assert dq.pop(2) == pdeque([2, 3], maxlen=3)
    assert dq.pop(-1) == pdeque([3, 4, 5], maxlen=3)

    # Test popping from a bounded deque with maxlen 0 and popping with a negative count
    dq = pdeque([], maxlen=0)
    assert dq.pop(-1) == pdeque([], maxlen=0)

    # Test popping from a bounded deque with maxlen 1 and popping with a negative count
    dq = pdeque([1], maxlen=1)
    assert dq.pop(-1) == pdeque([1], maxlen=1)

    # Test popping from a bounded deque with maxlen larger than the deque length and popping with a negative count
    dq = pdeque([1, 2, 3], maxlen=5)
    assert dq.pop(-1) == pdeque([2, 3], maxlen=5)

    # Test popping from a bounded deque with maxlen equal to the deque length and popping with a negative count
    dq = pdeque([1, 2, 3], maxlen=3)
    assert dq.pop(-1) == pdeque([2, 3], maxlen=3)

    # Test popping from a bounded deque with maxlen smaller than the deque length and popping with a negative count
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq.pop(-1) == pdeque([3, 4, 5], maxlen=3)

    # Test popping from a deque with a large number of elements
    dq = pdeque(range(1000))
    assert dq.pop() == pdeque(range(999))
    assert dq.pop(500) == pdeque(range(500))
    assert dq.pop(-500) == pdeque(range(500, 1000))

    # Test popping from a deque with a large number of elements and a bounded maxlen
    dq = pdeque(range(1000), maxlen=500)
    assert dq.pop() == pdeque(range(500, 999), maxlen=500)
    assert dq.pop(250) == pdeque(range(500, 749), maxlen=500)
    assert dq.pop(-250) == pdeque(range(750, 1000), maxlen=500)

    # Test popping from a deque with a large number of elements and a maxlen of 0
    dq = pdeque(range(1000), maxlen=0)
    assert dq.pop() == pdeque([], maxlen=0)
    assert dq.pop(500) == pdeque([], maxlen=0)
    assert dq.pop(-500) == pdeque([], maxlen=0)

    # Test popping from a deque with a large number of elements and a maxlen of 1
    dq = pdeque(range(1000), maxlen=1)
    assert dq.pop() == pdeque([], maxlen=1)
    assert dq.pop(500) == pdeque([], maxlen=1)
    assert dq.pop(-500) == pdeque([999], maxlen=1)

    # Test popping from a deque with a large number of elements and a maxlen larger than the deque length
    dq = pdeque(range(1000), maxlen=1500)
    assert dq.pop() == pdeque(range(999), maxlen=1500)
    assert dq.pop(500) == pdeque(range(500), maxlen=1500)
    assert dq.pop(-500) == pdeque(range(500, 1000), maxlen=1500)

    # Test popping from a deque with a large number of elements and a maxlen equal to the deque length
    dq = pdeque(range(1000), maxlen=1000)
    assert dq.pop() == pdeque(range(999), maxlen=1000)
    assert dq.pop(500) == pdeque(range(500), maxlen=1000)
    assert dq.pop(-500) == pdeque(range(500, 1000), maxlen=1000)

    # Test popping from a deque with a large number of elements and a maxlen smaller than the deque length
    dq = pdeque(range(1000), maxlen=500)
    assert dq.pop() == pdeque(range(500, 999), maxlen=500)
    assert dq.pop(250) == pdeque(range(500, 749), maxlen=500)
    assert dq.pop(-250) == pdeque(range(750, 1000), maxlen=500)

    # Test popping from a deque with a large number of elements and a maxlen of 0 and popping with a negative count
    dq = pdeque(range(1000), maxlen=0)
    assert dq.pop(-1) == pdeque([], maxlen=0)

    # Test popping from a deque with a large number of elements and a maxlen of 1 and popping with a negative count
    dq = pdeque(range(1000), maxlen=1)
    assert dq.pop(-1) == pdeque([999], maxlen=1)

    # Test popping from a deque with a large number of elements and a maxlen larger than the deque length and popping with a negative count
    dq = pdeque(range(1000), maxlen=1500)
   


# LLM-generated content at query #5
#--------------------------

# Unit test for method popleft of class PDeque
def test_PDeque_popleft(): 
    # Test case 1: popleft from an empty deque
    dq = pdeque()
    result = dq.popleft()
    assert result == pdeque()
    assert len(result) == 0

    # Test case 2: popleft from a deque with one element
    dq = pdeque([1])
    result = dq.popleft()
    assert result == pdeque()
    assert len(result) == 0

    # Test case 3: popleft from a deque with multiple elements
    dq = pdeque([1, 2, 3])
    result = dq.popleft()
    assert result == pdeque([2, 3])
    assert len(result) == 2

    # Test case 4: popleft with count > length of deque
    dq = pdeque([1, 2, 3])
    result = dq.popleft(5)
    assert result == pdeque()
    assert len(result) == 0

    # Test case 5: popleft with negative count (should call pop with positive count)
    dq = pdeque([1, 2, 3])
    result = dq.popleft(-2)
    assert result == pdeque([1])
    assert len(result) == 1

    # Test case 6: popleft with count = 0 (should return the same deque)
    dq = pdeque([1, 2, 3])
    result = dq.popleft(0)
    assert result == dq
    assert len(result) == 3

    # Test case 7: popleft from a deque with maxlen
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.popleft()
    assert result == pdeque([2, 3], maxlen=3)
    assert len(result) == 2

    # Test case 8: popleft from a deque with maxlen and count > length
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.popleft(5)
    assert result == pdeque([], maxlen=3)
    assert len(result) == 0

    # Test case 9: popleft from a deque with maxlen and negative count
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.popleft(-2)
    assert result == pdeque([1], maxlen=3)
    assert len(result) == 1

    # Test case 10: popleft from a deque with maxlen and count = 0
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.popleft(0)
    assert result == dq
    assert len(result) == 3

    print("All test cases passed!")

# Run the unit test
test_PDeque_popleft()


# LLM-generated content at query #6
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    d = pdeque([1, 2, 3, 4, 5])
    assert d[0] == 1
    assert d[-1] == 5
    assert d[2] == 3
    assert d[-3] == 3
    assert d[1:3] == pdeque([2, 3])
    assert d[-4:-1] == pdeque([2, 3, 4])
    assert d[::2] == pdeque([1, 3, 5])
    assert d[::-1] == pdeque([5, 4, 3, 2, 1])
    assert d[10] is None  # Index out of range
    assert d[-10] is None  # Index out of range
    assert d[2:10] == pdeque([3, 4, 5])
    assert d[-10:2] == pdeque([1, 2])
    assert d[10:20] == pdeque([])
    assert d[-20:-10] == pdeque([])
    assert d[0:0] == pdeque([])
    assert d[5:5] == pdeque([])
    assert d[-5:-5] == pdeque([])
    assert d[0:5:2] == pdeque([1, 3, 5])
    assert d[4:0:-2] == pdeque([5, 3])
    assert d[4:0:-1] == pdeque([5, 4, 3, 2])
    assert d[0:5:-1] == pdeque([])
    assert d[5:0:1] == pdeque([])
    assert d[0:5:1] == pdeque([1, 2, 3, 4, 5])
    assert d[5:0:-1] == pdeque([5, 4, 3, 2])
    assert d[0:5:3] == pdeque([1, 4])
    assert d[5:0:-3] == pdeque([5, 2])
    assert d[0:5:4] == pdeque([1, 5])
    assert d[5:0:-4] == pdeque([5, 1])
    assert d[0:5:5] == pdeque([1])
    assert d[5:0:-5] == pdeque([5])
    assert d[0:5:6] == pdeque([1])
    assert d[5:0:-6] == pdeque([5])
    assert d[0:5:7] == pdeque([1])
    assert d[5:0:-7] == pdeque([5])
    assert d[0:5:8] == pdeque([1])
    assert d[5:0:-8] == pdeque([5])
    assert d[0:5:9] == pdeque([1])
    assert d[5:0:-9] == pdeque([5])
    assert d[0:5:10] == pdeque([1])
    assert d[5:0:-10] == pdeque([5])
    assert d[0:5:11] == pdeque([1])
    assert d[5:0:-11] == pdeque([5])
    assert d[0:5:12] == pdeque([1])
    assert d[5:0:-12] == pdeque([5])
    assert d[0:5:13] == pdeque([1])
    assert d[5:0:-13] == pdeque([5])
    assert d[0:5:14] == pdeque([1])
    assert d[5:0:-14] == pdeque([5])
    assert d[0:5:15] == pdeque([1])
    assert d[5:0:-15] == pdeque([5])
    assert d[0:5:16] == pdeque([1])
    assert d[5:0:-16] == pdeque([5])
    assert d[0:5:17] == pdeque([1])
    assert d[5:0:-17] == pdeque([5])
    assert d[0:5:18] == pdeque([1])
    assert d[5:0:-18] == pdeque([5])
    assert d[0:5:19] == pdeque([1])
    assert d[5:0:-19] == pdeque([5])
    assert d[0:5:20] == pdeque([1])
    assert d[5:0:-20] == pdeque([5])
    assert d[0:5:21] == pdeque([1])
    assert d[5:0:-21] == pdeque([5])
    assert d[0:5:22] == pdeque([1])
    assert d[5:0:-22] == pdeque([5])
    assert d[0:5:23] == pdeque([1])
    assert d[5:0:-23] == pdeque([5])
    assert d[0:5:24] == pdeque([1])
    assert d[5:0:-24] == pdeque([5])
    assert d[0:5:25] == pdeque([1])
    assert d[5:0:-25] == pdeque([5])
    assert d[0:5:26] == pdeque([1])
    assert d[5:0:-26] == pdeque([5])
    assert d[0:5:27] == pdeque([1])
    assert d[5:0:-27] == pdeque([5])
    assert d[0:5:28] == pdeque([1])
    assert d[5:0:-28] == pdeque([5])
    assert d[0:5:29] == pdeque([1])
    assert d[5:0:-29] == pdeque([5])
    assert d[0:5:30] == pdeque([1])
    assert d[5:0:-30] == pdeque([5])
    assert d[0:5:31] == pdeque([1])
    assert d[5:0:-31] == pdeque([5])
    assert d[0:5:32] == pdeque([1])
    assert d[5:0:-32] == pdeque([5])
    assert d[0:5:33] == pdeque([1])
    assert d[5:0:-33] == pdeque([5])
    assert d[0:5:34] == pdeque([1])
    assert d[5:0:-34] == pdeque([5])
    assert d[0:5:35] == pdeque([1])
    assert d[5:0:-35] == pdeque([5])
    assert d[0:5:36] == pdeque([1])
    assert d[5:0:-36] == pdeque([5])
    assert d[0:5:37] == pdeque([1])
    assert d[5:0:-37] == pdeque([5])
    assert d[0:5:38] == pdeque([1])
    assert d[5:0:-38] == pdeque([5])
    assert d[0:5:39] == pdeque([1])
    assert d[5:0:-39] == pdeque([5])
    assert d[0:5:40] == pdeque([1])
    assert d[5:0:-40] == pdeque([5])
    assert d[0:5:41] == pdeque([1])
    assert d[5:0:-41] == pdeque([5])
    assert d[0:5:42] == pdeque([1])
    assert d[5:0:-42] == pdeque([5])
    assert d[0:5:43] == pdeque([1])
    assert d[5:0:-43] == pdeque([5])
    assert d[0:5:44] == pdeque([1])
    assert d[5:0:-44] == pdeque([5])
    assert d[0:5:45] == pdeque([1])
    assert d[5:0:-45] == pdeque([5])
    assert d[0:5:46] == pdeque([1])
    assert d[5:0:-46] == pdeque([5])
    assert d[0:5:47] == pdeque([1])
    assert d[5:0:-47] == pdeque([5])
    assert d[0:5:48] == pdeque([1])
    assert d[5:0:-48] == pdeque([5])
    assert d[0:5:49] == pdeque([1])
    assert d[5:0:-49] == pdeque


# LLM-generated content at query #7
#--------------------------

# Unit test for method popleft of class PDeque
def test_PDeque_popleft(): 
    # Test popleft with default count
    dq = pdeque([1, 2, 3])
    result = dq.popleft()
    assert result == pdeque([2, 3])
    assert len(result) == 2

    # Test popleft with count > 1
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.popleft(3)
    assert result == pdeque([4, 5])
    assert len(result) == 2

    # Test popleft with count > len(dq)
    dq = pdeque([1, 2])
    result = dq.popleft(5)
    assert result == pdeque([])
    assert len(result) == 0

    # Test popleft with negative count (should call pop)
    dq = pdeque([1, 2, 3])
    result = dq.popleft(-2)
    assert result == pdeque([1])
    assert len(result) == 1

    # Test popleft on empty deque
    dq = pdeque([])
    result = dq.popleft()
    assert result == pdeque([])
    assert len(result) == 0

    # Test popleft with maxlen
    dq = pdeque([1, 2, 3], maxlen=3)
    result = dq.popleft()
    assert result == pdeque([2, 3], maxlen=3)
    assert len(result) == 2

    # Test popleft with count and maxlen
    dq = pdeque([1, 2, 3, 4], maxlen=3)
    result = dq.popleft(2)
    assert result == pdeque([3, 4], maxlen=3)
    assert len(result) == 2

    # Test popleft with count 0
    dq = pdeque([1, 2, 3])
    result = dq.popleft(0)
    assert result == dq
    assert len(result) == 3

    # Test popleft with large count on large deque
    dq = pdeque(range(1000))
    result = dq.popleft(500)
    assert len(result) == 500
    assert result.left == 500
    assert result.right == 999

    # Test popleft maintains hashability
    dq = pdeque([1, 2, 3])
    result = dq.popleft()
    try:
        hash(result)
    except TypeError:
        assert False, "popleft result should be hashable"

    # Test popleft with mixed types
    dq = pdeque(['a', 1, 3.14, None])
    result = dq.popleft(2)
    assert result == pdeque([3.14, None])
    assert len(result) == 2

    # Test popleft then append
    dq = pdeque([1, 2, 3])
    result = dq.popleft().append(4)
    assert result == pdeque([2, 3, 4])
    assert len(result) == 3

    # Test popleft then appendleft
    dq = pdeque([1, 2, 3])
    result = dq.popleft().appendleft(0)
    assert result == pdeque([0, 2, 3])
    assert len(result) == 3

    # Test popleft with single element deque
    dq = pdeque([42])
    result = dq.popleft()
    assert result == pdeque([])
    assert len(result) == 0

    # Test popleft preserves maxlen when removing elements
    dq = pdeque([1, 2, 3, 4], maxlen=3)
    result = dq.popleft(1)
    assert result.maxlen == 3
    assert len(result) == 3

    # Test popleft with count equal to length
    dq = pdeque([1, 2, 3])
    result = dq.popleft(3)
    assert result == pdeque([])
    assert len(result) == 0

    # Test popleft with count greater than length (should empty deque)
    dq = pdeque([1, 2])
    result = dq.popleft(10)
    assert result == pdeque([])
    assert len(result) == 0

    # Test popleft maintains correct internal structure
    dq = pdeque(range(10))
    result = dq.popleft(5)
    # Check that iteration order is preserved
    assert list(result) == list(range(5, 10))

    # Test popleft with negative count on empty deque
    dq = pdeque([])
    result = dq.popleft(-1)
    assert result == pdeque([])
    assert len(result) == 0

    # Test popleft then index access
    dq = pdeque([10, 20, 30, 40])
    result = dq.popleft(2)
    assert result[0] == 30
    assert result[-1] == 40

    # Test popleft with maxlen=0
    dq = pdeque([], maxlen=0)
    result = dq.popleft()
    assert result == pdeque([], maxlen=0)
    assert len(result) == 0

    # Test popleft with maxlen and count exceeding maxlen
    dq = pdeque([1, 2, 3], maxlen=5)
    result = dq.popleft(10)
    assert result == pdeque([], maxlen=5)
    assert len(result) == 0

    # Test popleft maintains sequence protocol
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.popleft(2)
    assert isinstance(result, Sequence)
    assert list(result) == [3, 4, 5]

    # Test popleft with very large deque (stress test)
    dq = pdeque(range(10000))
    result = dq.popleft(5000)
    assert len(result) == 5000
    assert result.left == 5000
    assert result.right == 9999

    # Test popleft then extend
    dq = pdeque([1, 2, 3])
    result = dq.popleft().extend([4, 5])
    assert result == pdeque([2, 3, 4, 5])
    assert len(result) == 4

    # Test popleft then extendleft
    dq = pdeque([1, 2, 3])
    result = dq.popleft().extendleft([0, -1])
    assert result == pdeque([-1, 0, 2, 3])
    assert len(result) == 4

    # Test popleft with count=1 is same as default
    dq = pdeque([1, 2, 3])
    result1 = dq.popleft()
    result2 = dq.popleft(1)
    assert result1 == result2

    # Test popleft on deque created from empty iterable
    dq = pdeque([])
    result = dq.popleft()
    assert result == pdeque([])
    assert len(result) == 0

    # Test popleft maintains correct length after multiple operations
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.popleft(2).popleft(1).append(6).appendleft(0)
    assert result == pdeque([0, 4, 5, 6])
    assert len(result) == 4

    # Test popleft with count that causes rebalancing
    dq = pdeque(range(100))
    # Pop enough elements to force rebalancing
    result = dq.popleft(60)
    assert len(result) == 40
    assert list(result) == list(range(60, 100))

    # Test popleft then reverse
    dq = pdeque([1, 2, 3, 4])
    result = dq.popleft(2).reverse()
    assert result == pdeque([4, 3])
    assert len(result) == 2

    # Test popleft with count and then count elements
    dq = pdeque([1, 2, 2, 3, 2, 4])
    result = dq.popleft(3)
    assert result.count(2) == 2
    assert len(result) == 3

    # Test popleft with remove after
    dq = pdeque([1, 2, 3, 2, 4])
    result = dq.popleft(2).remove(2)
    assert result == pdeque


# LLM-generated content at query #8
#--------------------------

# Unit test for method remove of class PDeque
def test_PDeque_remove(): 
    # Test case 1: remove existing element from left list
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.remove(3)
    expected = pdeque([1, 2, 4, 5])
    assert result == expected

    # Test case 2: remove existing element from right list
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.remove(5)
    expected = pdeque([1, 2, 3, 4])
    assert result == expected

    # Test case 3: remove non-existing element
    dq = pdeque([1, 2, 3, 4, 5])
    try:
        dq.remove(6)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 4: remove element from empty deque
    dq = pdeque([])
    try:
        dq.remove(1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 5: remove element from deque with duplicate values
    dq = pdeque([1, 2, 3, 2, 1])
    result = dq.remove(2)
    expected = pdeque([1, 3, 2, 1])
    assert result == expected

    # Test case 6: remove element from deque with maxlen
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    result = dq.remove(3)
    expected = pdeque([1, 2, 4, 5], maxlen=4)
    assert result == expected

    # Test case 7: remove element from deque with maxlen and left list empty
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq.remove(3)
    expected = pdeque([2, 4, 5], maxlen=3)
    assert result == expected

    # Test case 8: remove element from deque with maxlen and right list empty
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq.remove(5)
    expected = pdeque([1, 2, 4], maxlen=3)
    assert result == expected

    # Test case 9: remove element from deque with maxlen and both lists non-empty
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    result = dq.remove(2)
    expected = pdeque([1, 3, 4, 5], maxlen=4)
    assert result == expected

    # Test case 10: remove element from deque with maxlen and both lists non-empty, element in right list
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    result = dq.remove(4)
    expected = pdeque([1, 2, 3, 5], maxlen=4)
    assert result == expected

    # Test case 11: remove element from deque with maxlen and both lists non-empty, element in left list
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    result = dq.remove(1)
    expected = pdeque([2, 3, 4, 5], maxlen=4)
    assert result == expected

    # Test case 12: remove element from deque with maxlen and both lists non-empty, element in both lists
    dq = pdeque([1, 2, 3, 2, 1], maxlen=4)
    result = dq.remove(2)
    expected = pdeque([1, 3, 2, 1], maxlen=4)
    assert result == expected

    # Test case 13: remove element from deque with maxlen and both lists non-empty, element not present
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    try:
        dq.remove(6)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 14: remove element from deque with maxlen and both lists non-empty, element in left list, left list becomes empty
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq.remove(1)
    expected = pdeque([2, 3, 5], maxlen=3)
    assert result == expected

    # Test case 15: remove element from deque with maxlen and both lists non-empty, element in right list, right list becomes empty
    dq = pdeque([1, 2, 3, 4, 5], maxlen=3)
    result = dq.remove(5)
    expected = pdeque([1, 2, 4], maxlen=3)
    assert result == expected

    # Test case 16: remove element from deque with maxlen and both lists non-empty, element in left list, left list becomes empty, right list becomes empty
    dq = pdeque([1, 2, 3], maxlen=2)
    result = dq.remove(1)
    expected = pdeque([2, 3], maxlen=2)
    assert result == expected

    # Test case 17: remove element from deque with maxlen and both lists non-empty, element in right list, left list becomes empty, right list becomes empty
    dq = pdeque([1, 2, 3], maxlen=2)
    result = dq.remove(3)
    expected = pdeque([1, 2], maxlen=2)
    assert result == expected

    # Test case 18: remove element from deque with maxlen and both lists non-empty, element in left list, left list becomes empty, right list becomes empty, maxlen is 0
    dq = pdeque([], maxlen=0)
    try:
        dq.remove(1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 19: remove element from deque with maxlen and both lists non-empty, element in right list, left list becomes empty, right list becomes empty, maxlen is 0
    dq = pdeque([], maxlen=0)
    try:
        dq.remove(1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 20: remove element from deque with maxlen and both lists non-empty, element in left list, left list becomes empty, right list becomes empty, maxlen is negative
    dq = pdeque([1, 2, 3], maxlen=-1)
    try:
        dq.remove(1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 21: remove element from deque with maxlen and both lists non-empty, element in right list, left list becomes empty, right list becomes empty, maxlen is negative
    dq = pdeque([1, 2, 3], maxlen=-1)
    try:
        dq.remove(3)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 22: remove element from deque with maxlen and both lists non-empty, element in left list, left list becomes empty, right list becomes empty, maxlen is float
    dq = pdeque([1, 2, 3], maxlen=2.5)
    try:
        dq.remove(1)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test case 23: remove element from deque with maxlen and both lists non-empty, element in right list, left list becomes empty, right list becomes empty, maxlen is float
    dq = pdeque([1, 2, 3], maxlen=2.5)
    try:
        dq.remove(3)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test case 24: remove element from deque with maxlen and both lists non-empty, element in left list, left list becomes empty, right list becomes empty, maxlen is string
    dq = pdeque([1, 2, 3], maxlen="2")
    try:
        dq.remove(1)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test case 25: remove element from deque with maxlen and both lists non-empty, element in right list, left list becomes empty, right list becomes empty, maxlen is string
    dq = pdeque([1, 2, 3], maxlen="2")
    try:
        dq.remove(3)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test case 26: remove element from deque with maxlen and both lists non-empty,


# LLM-generated content at query #9
#--------------------------

# Unit test for method __eq__ of class PDeque
def test_PDeque___eq__(): 
    # Test case 1: Compare two empty deques
    deque1 = pdeque()
    deque2 = pdeque()
    assert deque1 == deque2

    # Test case 2: Compare two deques with same elements
    deque1 = pdeque([1, 2, 3])
    deque2 = pdeque([1, 2, 3])
    assert deque1 == deque2

    # Test case 3: Compare two deques with different elements
    deque1 = pdeque([1, 2, 3])
    deque2 = pdeque([4, 5, 6])
    assert not deque1 == deque2

    # Test case 4: Compare deque with non-deque object
    deque1 = pdeque([1, 2, 3])
    other = [1, 2, 3]
    assert not deque1 == other

    # Test case 5: Compare deque with different maxlen
    deque1 = pdeque([1, 2, 3], maxlen=5)
    deque2 = pdeque([1, 2, 3], maxlen=10)
    assert not deque1 == deque2

    # Test case 6: Compare deque with same elements but different order
    deque1 = pdeque([1, 2, 3])
    deque2 = pdeque([3, 2, 1])
    assert not deque1 == deque2

    # Test case 7: Compare deque with same elements but different lengths
    deque1 = pdeque([1, 2, 3])
    deque2 = pdeque([1, 2, 3, 4])
    assert not deque1 == deque2

    # Test case 8: Compare deque with same elements but different maxlen
    deque1 = pdeque([1, 2, 3], maxlen=5)
    deque2 = pdeque([1, 2, 3], maxlen=3)
    assert not deque1 == deque2

    # Test case 9: Compare deque with same elements but different internal representation
    deque1 = pdeque([1, 2, 3])
    deque2 = pdeque([1, 2, 3])
    deque2._left_list = plist([1, 2])
    deque2._right_list = plist([3], reverse=True)
    assert deque1 == deque2

    # Test case 10: Compare deque with same elements but different internal representation and maxlen
    deque1 = pdeque([1, 2, 3], maxlen=5)
    deque2 = pdeque([1, 2, 3], maxlen=5)
    deque2._left_list = plist([1, 2])
    deque2._right_list = plist([3], reverse=True)
    assert deque1 == deque2


# LLM-generated content at query #10
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    # Test case 1: Indexing with a positive integer
    d = pdeque([1, 2, 3, 4, 5])
    assert d[0] == 1
    assert d[2] == 3
    assert d[4] == 5

    # Test case 2: Indexing with a negative integer
    d = pdeque([1, 2, 3, 4, 5])
    assert d[-1] == 5
    assert d[-3] == 3
    assert d[-5] == 1

    # Test case 3: Indexing with a slice
    d = pdeque([1, 2, 3, 4, 5])
    assert d[1:4] == pdeque([2, 3, 4])
    assert d[:3] == pdeque([1, 2, 3])
    assert d[2:] == pdeque([3, 4, 5])
    assert d[::2] == pdeque([1, 3, 5])

    # Test case 4: Indexing with a slice and negative step
    d = pdeque([1, 2, 3, 4, 5])
    assert d[::-1] == pdeque([5, 4, 3, 2, 1])
    assert d[4:1:-1] == pdeque([5, 4, 3])
    assert d[2::-1] == pdeque([3, 2, 1])

    # Test case 5: Indexing with a slice and step not equal to 1
    d = pdeque([1, 2, 3, 4, 5])
    assert d[1:5:2] == pdeque([2, 4])
    assert d[::3] == pdeque([1, 4])
    assert d[2:5:2] == pdeque([3, 5])

    # Test case 6: Indexing with a slice and start/stop out of range
    d = pdeque([1, 2, 3, 4, 5])
    assert d[2:10] == pdeque([3, 4, 5])
    assert d[-10:3] == pdeque([1, 2, 3])
    assert d[10:20] == pdeque([])

    # Test case 7: Indexing with a slice and negative start/stop
    d = pdeque([1, 2, 3, 4, 5])
    assert d[-4:-1] == pdeque([2, 3, 4])
    assert d[-2:] == pdeque([4, 5])
    assert d[:-3] == pdeque([1, 2])

    # Test case 8: Indexing with a slice and step negative
    d = pdeque([1, 2, 3, 4, 5])
    assert d[4:1:-2] == pdeque([5, 3])
    assert d[::-2] == pdeque([5, 3, 1])
    assert d[3::-2] == pdeque([4, 2])

    # Test case 9: Indexing with a slice and step 0 (should raise ValueError)
    d = pdeque([1, 2, 3, 4, 5])
    try:
        d[::0]
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 10: Indexing with a non-integer index (should raise TypeError)
    d = pdeque([1, 2, 3, 4, 5])
    try:
        d["invalid"]
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test case 11: Indexing with an out-of-range index (should raise IndexError)
    d = pdeque([1, 2, 3, 4, 5])
    try:
        d[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test case 12: Indexing with an empty deque (should raise IndexError)
    d = pdeque()
    try:
        d[0]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test case 13: Indexing with a slice and maxlen specified
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[:] == pdeque([3, 4, 5], maxlen=3)
    assert d[1:] == pdeque([4, 5], maxlen=3)
    assert d[:2] == pdeque([3, 4], maxlen=3)

    # Test case 14: Indexing with a slice and maxlen specified, step not equal to 1
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[::2] == pdeque([3, 5], maxlen=3)
    assert d[1::2] == pdeque([4], maxlen=3)

    # Test case 15: Indexing with a slice and maxlen specified, negative step
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[::-1] == pdeque([5, 4, 3], maxlen=3)
    assert d[::-2] == pdeque([5, 3], maxlen=3)

    # Test case 16: Indexing with a slice and maxlen specified, start/stop out of range
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[2:10] == pdeque([5], maxlen=3)
    assert d[-10:2] == pdeque([3, 4], maxlen=3)
    assert d[10:20] == pdeque([], maxlen=3)

    # Test case 17: Indexing with a slice and maxlen specified, negative start/stop
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[-3:-1] == pdeque([3, 4], maxlen=3)
    assert d[-2:] == pdeque([4, 5], maxlen=3)
    assert d[:-2] == pdeque([3], maxlen=3)

    # Test case 18: Indexing with a slice and maxlen specified, step negative
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert d[2:0:-1] == pdeque([5, 4], maxlen=3)
    assert d[::-2] == pdeque([5, 3], maxlen=3)
    assert d[1::-1] == pdeque([4, 3], maxlen=3)

    # Test case 19: Indexing with a slice and maxlen specified, step 0 (should raise ValueError)
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    try:
        d[::0]
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test case 20: Indexing with a non-integer index and maxlen specified (should raise TypeError)
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    try:
        d["invalid"]
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test case 21: Indexing with an out-of-range index and maxlen specified (should raise IndexError)
    d = pdeque([1, 2, 3, 4, 5], maxlen=3)
    try:
        d[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test case 22: Indexing with an empty deque and maxlen specified (should raise IndexError)
    d = pdeque(maxlen=3)
    try:
        d[0]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    print("All test cases passed!")

test_PDeque___getitem__()


# LLM-generated content at query #11
#--------------------------

# Unit test for method remove of class PDeque
def test_PDeque_remove(): 
    # Test case 1: Remove element from left list
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.remove(3)
    expected = pdeque([1, 2, 4, 5])
    assert result == expected

    # Test case 2: Remove element from right list
    dq = pdeque([1, 2, 3, 4, 5])
    result = dq.remove(5)
    expected = pdeque([1, 2, 3, 4])
    assert result == expected

    # Test case 3: Remove element that appears multiple times
    dq = pdeque([1, 2, 3, 2, 4])
    result = dq.remove(2)
    expected = pdeque([1, 3, 2, 4])
    assert result == expected

    # Test case 4: Remove element that is not in the deque
    dq = pdeque([1, 2, 3, 4, 5])
    try:
        dq.remove(6)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "6 not found in PDeque"

    # Test case 5: Remove element from an empty deque
    dq = pdeque([])
    try:
        dq.remove(1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "1 not found in PDeque"

    # Test case 6: Remove element from a deque with a single element
    dq = pdeque([1])
    result = dq.remove(1)
    expected = pdeque([])
    assert result == expected

    # Test case 7: Remove element from a deque with duplicate elements
    dq = pdeque([1, 1, 1])
    result = dq.remove(1)
    expected = pdeque([1, 1])
    assert result == expected

    # Test case 8: Remove element from a deque with maxlen
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    result = dq.remove(3)
    expected = pdeque([1, 2, 4, 5], maxlen=4)
    assert result == expected

    # Test case 9: Remove element from a deque with maxlen and element not in deque
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    try:
        dq.remove(6)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "6 not found in PDeque"

    # Test case 10: Remove element from a deque with maxlen and element appears multiple times
    dq = pdeque([1, 2, 3, 2, 4], maxlen=4)
    result = dq.remove(2)
    expected = pdeque([1, 3, 2, 4], maxlen=4)
    assert result == expected

    print("All test cases passed")

test_PDeque_remove()


# LLM-generated content at query #12
#--------------------------

# Unit test for method rotate of class PDeque
def test_PDeque_rotate(): 
    # Test case 1: Rotate positive steps
    dq = pdeque([1, 2, 3, 4, 5])
    rotated = dq.rotate(2)
    assert rotated == pdeque([4, 5, 1, 2, 3])

    # Test case 2: Rotate negative steps
    dq = pdeque([1, 2, 3, 4, 5])
    rotated = dq.rotate(-2)
    assert rotated == pdeque([3, 4, 5, 1, 2])

    # Test case 3: Rotate zero steps
    dq = pdeque([1, 2, 3, 4, 5])
    rotated = dq.rotate(0)
    assert rotated == dq

    # Test case 4: Rotate steps greater than length
    dq = pdeque([1, 2, 3, 4, 5])
    rotated = dq.rotate(7)
    assert rotated == pdeque([4, 5, 1, 2, 3])

    # Test case 5: Rotate steps negative greater than length
    dq = pdeque([1, 2, 3, 4, 5])
    rotated = dq.rotate(-7)
    assert rotated == pdeque([3, 4, 5, 1, 2])

    # Test case 6: Rotate empty deque
    dq = pdeque([])
    rotated = dq.rotate(3)
    assert rotated == pdeque([])

    # Test case 7: Rotate single element deque
    dq = pdeque([1])
    rotated = dq.rotate(5)
    assert rotated == pdeque([1])

    # Test case 8: Rotate with maxlen
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    rotated = dq.rotate(2)
    assert rotated == pdeque([4, 5, 1, 2], maxlen=4)

    # Test case 9: Rotate negative with maxlen
    dq = pdeque([1, 2, 3, 4, 5], maxlen=4)
    rotated = dq.rotate(-2)
    assert rotated == pdeque([3, 4, 5, 1], maxlen=4)

    print("All test cases passed!")

# Run the unit tests
test_PDeque_rotate()


# LLM-generated content at query #13
#--------------------------

# Unit test for method __new__ of class PDeque
def test_PDeque___new__():  
    # Test with valid inputs
    left_list = plist([1, 2, 3])
    right_list = plist([4, 5, 6])
    length = 6
    maxlen = 10
    dq = PDeque(left_list, right_list, length, maxlen)
    assert dq._left_list == left_list
    assert dq._right_list == right_list
    assert dq._length == length
    assert dq._maxlen == maxlen

    # Test with maxlen as None
    dq2 = PDeque(left_list, right_list, length, None)
    assert dq2._maxlen is None

    # Test with maxlen as negative integer (should raise ValueError)
    try:
        PDeque(left_list, right_list, length, -1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as non-integer (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, "invalid")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as zero
    dq3 = PDeque(left_list, right_list, length, 0)
    assert dq3._maxlen == 0

    # Test with maxlen as positive integer
    dq4 = PDeque(left_list, right_list, length, 5)
    assert dq4._maxlen == 5

    # Test with empty lists and zero length
    empty_left = plist()
    empty_right = plist()
    dq5 = PDeque(empty_left, empty_right, 0, None)
    assert dq5._left_list == empty_left
    assert dq5._right_list == empty_right
    assert dq5._length == 0
    assert dq5._maxlen is None

    # Test with maxlen equal to length
    dq6 = PDeque(left_list, right_list, length, length)
    assert dq6._maxlen == length

    # Test with maxlen greater than length
    dq7 = PDeque(left_list, right_list, length, length + 5)
    assert dq7._maxlen == length + 5

    # Test with maxlen less than length (should be allowed, but may affect appends)
    dq8 = PDeque(left_list, right_list, length, length - 2)
    assert dq8._maxlen == length - 2

    # Test with maxlen as float (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, 3.14)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as boolean (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, True)
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as large integer
    dq9 = PDeque(left_list, right_list, length, 10**6)
    assert dq9._maxlen == 10**6

    # Test with maxlen as zero and empty lists
    dq10 = PDeque(empty_left, empty_right, 0, 0)
    assert dq10._maxlen == 0

    # Test with maxlen as negative zero (should be treated as zero)
    dq11 = PDeque(left_list, right_list, length, -0)
    assert dq11._maxlen == 0

    # Test with maxlen as string integer (should raise TypeError)
    try:
        PDeque(left_list, right_list, length, "10")
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen as None and empty lists
    dq12 = PDeque(empty_left, empty_right, 0, None)
    assert dq12._maxlen is None

    # Test with maxlen as negative integer (should raise ValueError)
    try:
        PDeque(left_list, right_list, length, -5)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as negative integer and empty lists
    try:
        PDeque(empty_left, empty_right, 0, -1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as negative integer and zero length
    try:
        PDeque(empty_left, empty_right, 0, -10)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as negative integer and non-empty lists
    try:
        PDeque(left_list, right_list, length, -3)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as negative integer and large length
    try:
        PDeque(left_list, right_list, 1000, -100)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test with maxlen as negative integer and maxlen as negative zero
    dq13 = PDeque(left_list, right_list, length, -0)
    assert dq13._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and empty lists
    dq14 = PDeque(empty_left, empty_right, 0, -0)
    assert dq14._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and non-empty lists
    dq15 = PDeque(left_list, right_list, length, -0)
    assert dq15._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and large length
    dq16 = PDeque(left_list, right_list, 1000, -0)
    assert dq16._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and zero length
    dq17 = PDeque(empty_left, empty_right, 0, -0)
    assert dq17._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as negative zero
    dq18 = PDeque(left_list, right_list, length, -0)
    assert dq18._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as negative zero and empty lists
    dq19 = PDeque(empty_left, empty_right, 0, -0)
    assert dq19._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as negative zero and non-empty lists
    dq20 = PDeque(left_list, right_list, length, -0)
    assert dq20._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as negative zero and large length
    dq21 = PDeque(left_list, right_list, 1000, -0)
    assert dq21._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as negative zero and zero length
    dq22 = PDeque(empty_left, empty_right, 0, -0)
    assert dq22._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as negative zero and maxlen as negative zero
    dq23 = PDeque(left_list, right_list, length, -0)
    assert dq23._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as negative zero and maxlen as negative zero and empty lists
    dq24 = PDeque(empty_left, empty_right, 0, -0)
    assert dq24._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as negative zero and maxlen as negative zero and non-empty lists
    dq25 = PDeque(left_list, right_list, length, -0)
    assert dq25._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as negative zero and maxlen as negative zero and large length
    dq26 = PDeque(left_list, right_list, 1000, -0)
    assert dq26._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as negative zero and maxlen as negative zero and zero length
    dq27 = PDeque(empty_left, empty_right, 0, -0)
    assert dq27._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as negative zero and maxlen as negative zero and maxlen as negative zero
    dq28 = PDeque(left_list, right_list, length, -0)
    assert dq28._maxlen == 0

    # Test with maxlen as negative integer and maxlen as negative zero and maxlen as


# LLM-generated content at query #14
#--------------------------

# Unit test for method __getitem__ of class PDeque
def test_PDeque___getitem__(): 
    # Test with integer index
    dq = pdeque([1, 2, 3, 4, 5])
    assert dq[0] == 1
    assert dq[2] == 3
    assert dq[-1] == 5
    assert dq[-3] == 3

    # Test with slice
    assert dq[1:3] == pdeque([2, 3])
    assert dq[:3] == pdeque([1, 2, 3])
    assert dq[2:] == pdeque([3, 4, 5])
    assert dq[::2] == pdeque([1, 3, 5])

    # Test with negative slice
    assert dq[-3:-1] == pdeque([3, 4])
    assert dq[-4:] == pdeque([2, 3, 4, 5])

    # Test with step in slice
    assert dq[1:5:2] == pdeque([2, 4])
    assert dq[::3] == pdeque([1, 4])

    # Test with empty slice
    assert dq[3:3] == pdeque([])

    # Test with out of range index
    try:
        dq[10]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with invalid index type
    try:
        dq["invalid"]
        assert False, "Expected TypeError"
    except TypeError:
        pass

    # Test with maxlen
    dq_maxlen = pdeque([1, 2, 3, 4, 5], maxlen=3)
    assert dq_maxlen[0] == 3
    assert dq_maxlen[-1] == 5
    assert dq_maxlen[:2] == pdeque([3, 4], maxlen=3)

    # Test with single element
    dq_single = pdeque([42])
    assert dq_single[0] == 42
    assert dq_single[-1] == 42

    # Test with empty deque
    dq_empty = pdeque([])
    try:
        dq_empty[0]
        assert False, "Expected IndexError"
    except IndexError:
        pass

    print("All tests passed!")

# Run the unit test
test_PDeque___getitem__()


