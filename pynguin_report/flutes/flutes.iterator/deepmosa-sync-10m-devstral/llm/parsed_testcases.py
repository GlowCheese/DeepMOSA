####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = iter(var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [0, 1, 2, 3, 4])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 6
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = iter(var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [2, 3, 4, 5])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = iter(var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 3, 5, 7, 9])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = iter(var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = -1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = iter(var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [5, 4, 3, 2, 1])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = iter(var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [0, 1, 2])
    assert var_5 is True
    var_6 = iter(var_2)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [0, 1, 2])
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[0]
    assert var_5 == 1
    var_6 = var_4[1]
    assert var_6 == 3
    var_7 = var_4[4]
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[-1]
    assert var_5 == 9
    var_6 = var_4[-2]
    assert var_6 == 7
    var_7 = var_4[-5]
    assert var_7 == 1

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[1:4]
    var_6 = bool(var_4[1:4] == [3, 5, 7])
    assert var_6 is True
    var_7 = var_4[:3]
    var_8 = bool(var_4[:3] == [1, 3, 5])
    assert var_8 is True
    var_9 = var_4[2:]
    var_10 = bool(var_4[2:] == [5, 7, 9])
    assert var_10 is True
    var_11 = var_4[:]
    var_12 = bool(var_4[:] == [1, 3, 5, 7, 9])
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[10:20]
    var_6 = bool(var_4[10:20] == [])
    assert var_6 is True
    var_7 = var_4[-10:-20]
    var_8 = bool(var_4[-10:-20] == [])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[::2]
    var_6 = bool(var_4[::2] == [1, 5, 9])
    assert var_6 is True
    var_7 = var_4[1::2]
    var_8 = bool(var_4[1::2] == [3, 7])
    assert var_8 is True
    var_9 = var_4[::-1]
    var_10 = bool(var_4[::-1] == [9, 7, 5, 3, 1])
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #4
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.exhausted
    assert var_5 is False
    var_6 = var_4.list
    var_7 = bool(var_4.list == [])
    assert var_7 is True
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_lazy_list_constructor_with_generator. Retrieved 1/7 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = var_1.exhausted
    assert var_4 is False
    var_5 = 'iter'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True

def test_case_0():
    var_0 = 'iter'



# Parsed testcases at query #6
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_iterable_is_converted_to_iterator. Retrieved 6/7 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter



# Parsed testcases at query #8
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #9
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 0
    var_6 = [var_5]
    var_7 = module_0.scanl(var_0, var_4, *var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [0, 1, 3, 6])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = module_0.scanl(var_0, var_4, *var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 3, 6])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x, y: y + x
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = []
    var_6 = module_0.scanl(var_0, var_4, *var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['a', 'ba', 'cba'])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = [var_5]
    var_7 = module_0.scanl(var_0, var_4, *var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [1, 2, 6, 24])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = 5
    var_3 = [var_2]
    var_4 = module_0.scanl(var_0, var_1, *var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [5])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 10
    var_2 = [var_1]
    var_3 = 5
    var_4 = [var_3]
    var_5 = module_0.scanl(var_0, var_2, *var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [5, 15])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 10
    var_2 = [var_1]
    var_3 = []
    var_4 = module_0.scanl(var_0, var_2, *var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [10])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 0
    var_5 = [var_4, var_1]
    var_6 = module_0.scanl(var_0, var_3, *var_5)
    var_7 = list(var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #10
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.val
    assert var_6 == 0
    var_7 = var_2.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.val
    assert var_7 == 1
    var_8 = var_3.length
    assert var_8 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.val
    assert var_8 == 1
    var_9 = var_4.length
    assert var_9 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 3
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1, 2], [4, 5], [7, 8]])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 3
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = True
    var_6 = module_0.split_by(var_1, var_5, criterion=var_4)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [[1, 2], [4, 5], [7, 8], []])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'Split by:'
    var_1 = ' '
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [['S', 'p', 'l', 'i', 't'], ['b', 'y:']])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = ' Split by: '
    var_1 = True
    var_2 = ' '
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':'], []])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = 3
    var_2 = 0
    var_3 = lambda x: x % var_1 == var_2
    var_4 = module_0.split_by(var_0, criterion=var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [[]])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = 3
    var_2 = 0
    var_3 = lambda x: x % var_1 == var_2
    var_4 = True
    var_5 = module_0.split_by(var_0, var_4, criterion=var_3)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[]])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 6
    var_2 = 9
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda x: x % var_0 == var_4
    var_6 = module_0.split_by(var_3, criterion=var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 6
    var_2 = 9
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda x: x % var_0 == var_4
    var_6 = True
    var_7 = module_0.split_by(var_3, var_6, criterion=var_5)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [[], [], []])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 4
    var_3 = [var_0, var_1, var_2]
    var_4 = 3
    var_5 = 0
    var_6 = lambda x: x % var_4 == var_5
    var_7 = module_0.split_by(var_3, criterion=var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [[1, 2, 4]])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda x: x % var_2 == var_4
    var_6 = ' '
    var_7 = module_0.split_by(var_3, criterion=var_5, separator=var_6)
    var_8 = list(var_7)
    var_9 = bool(False)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.split_by(var_3)
    var_5 = list(var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_drop_with_generator. Retrieved 3/6 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [4, 5])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = module_0.drop(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 7

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'hello'
    var_2 = module_0.drop(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['l', 'l', 'o'])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6.list
    var_8 = bool(var_6.list == [])
    assert var_8 is True
    var_9 = var_6.exhausted
    assert var_9 is False
    var_10 = var_6.iter
    var_11 = iter(var_5)
    var_12 = type(var_11)
    var_13 = isinstance(var_10, var_12)
    var_14 = bool(var_13)
    assert var_14 is True



# Parsed testcases at query #14
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 1000000
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [0, 1, 2, 3, 4])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [0, 1, 2, 3, 4])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(False)
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = module_0.take(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 'hello'
    var_2 = module_0.take(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['h', 'e', 'l'])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = 4
    var_4 = [var_1, var_0, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 2])
    assert var_7 is True



# Parsed testcases at query #15
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.val
    assert var_6 == 0
    var_7 = var_2.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.val
    assert var_7 == 1
    var_8 = var_3.length
    assert var_8 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.val
    assert var_8 == 1
    var_9 = var_4.length
    assert var_9 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #16
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #17
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = []
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = [var_1, var_2, var_0, var_3, var_4, var_5]
    var_7 = module_0.chunk(var_0, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [[1, 2, 3], [4, 5, 6]])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = 7
    var_7 = [var_1, var_2, var_0, var_3, var_4, var_5, var_6]
    var_8 = module_0.chunk(var_0, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [[1, 2, 3], [4, 5, 6], [7]])
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.chunk(var_0, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [[1], [2], [3]])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1, 2, 3]])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6[0]
    assert var_7 == 1
    var_8 = var_6[2]
    assert var_8 == 3
    var_9 = var_6[4]
    assert var_9 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = 5
    var_6 = var_4[var_5]
    var_7 = bool(False)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6[-1]
    assert var_7 == 5
    var_8 = var_6[-3]
    assert var_8 == 3

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6[:]
    var_8 = bool(var_6[:] == [1, 2, 3, 4, 5])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6[2:]
    var_8 = bool(var_6[2:] == [3, 4, 5])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6[:3]
    var_8 = bool(var_6[:3] == [1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6[1:4]
    var_8 = bool(var_6[1:4] == [2, 3, 4])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6[::2]
    var_8 = bool(var_6[::2] == [1, 3, 5])
    assert var_8 is True
    var_9 = var_6[1::2]
    var_10 = bool(var_6[1::2] == [2, 4])
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 100
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2[0]
    assert var_3 == 0
    var_4 = var_2[50]
    assert var_4 == 50
    var_5 = var_2.list
    var_6 = len(var_5)
    assert var_6 == 51
    var_7 = var_2[99]
    assert var_7 == 99
    var_8 = var_2.exhausted
    assert var_8 is True



# Parsed testcases at query #19
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10
    var_7 = var_2.val
    assert var_7 == 0

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 10
    var_8 = var_3.val
    assert var_8 == 1

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5
    var_9 = var_4.val
    assert var_9 == 1

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_2, var_0, var_3, var_4, var_5]
    var_7 = module_0.MapList(var_1, var_6)
    var_8 = var_7[0]
    assert var_8 == 2
    var_9 = var_7[1]
    assert var_9 == 4
    var_10 = var_7[2]
    assert var_10 == 6
    var_11 = var_7[3]
    assert var_11 == 8
    var_12 = var_7[4]
    assert var_12 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_2, var_0, var_3, var_4, var_5]
    var_7 = module_0.MapList(var_1, var_6)
    var_8 = var_7[1:3]
    var_9 = bool(var_7[1:3] == [4, 6])
    assert var_9 is True
    var_10 = var_7[:2]
    var_11 = bool(var_7[:2] == [2, 4])
    assert var_11 is True
    var_12 = var_7[2:]
    var_13 = bool(var_7[2:] == [6, 8, 10])
    assert var_13 is True
    var_14 = var_7[:]
    var_15 = bool(var_7[:] == [2, 4, 6, 8, 10])
    assert var_15 is True
    var_16 = var_7[::2]
    var_17 = bool(var_7[::2] == [2, 6, 10])
    assert var_17 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_2, var_0, var_3, var_4, var_5]
    var_7 = module_0.MapList(var_1, var_6)
    var_8 = var_7[-1]
    assert var_8 == 10
    var_9 = var_7[-2]
    assert var_9 == 8
    var_10 = var_7[-3]
    assert var_10 == 6



# Parsed testcases at query #21
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.exhausted
    assert var_5 is False
    var_6 = var_4.list
    var_7 = bool(var_4.list == [])
    assert var_7 is True
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #22
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5[var_2]
    assert var_6 == 4



# Parsed testcases at query #23
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #24
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5[0]
    assert var_6 == 2
    var_7 = var_5[1]
    assert var_7 == 4
    var_8 = var_5[2]
    assert var_8 == 6



# Parsed testcases at query #25
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.val
    assert var_6 == 0
    var_7 = var_2.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.val
    assert var_7 == 1
    var_8 = var_3.length
    assert var_8 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.val
    assert var_8 == 1
    var_9 = var_4.length
    assert var_9 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #26
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #27
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10
    var_7 = var_2.val
    assert var_7 == 0

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 10
    var_8 = var_3.val
    assert var_8 == 1

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5
    var_9 = var_4.val
    assert var_9 == 1

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #28
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #29
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #30
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #31
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #32
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.exhausted
    assert var_5 is False
    var_6 = var_4.list
    var_7 = bool(var_4.list == [])
    assert var_7 is True
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #33
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #34
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == var_2)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.MapList(var_0, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_0)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = None
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == var_2)
    assert var_7 is True



# Parsed testcases at query #35
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #36
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #37
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #38
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #39
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = bool(not var_1.exhausted)
    assert var_4 is True
    var_5 = 'iter'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #40
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #41
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #42
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #43
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = 6
    var_7 = [var_2, var_3, var_4, var_5, var_0, var_6]
    var_8 = module_0.drop_until(var_1, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [6])
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'c'
    var_1 = lambda x: x == var_0
    var_2 = 'abcdef'
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == ['c', 'd', 'e', 'f'])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 4
    var_5 = 5
    var_6 = (var_2, var_3, var_0, var_4, var_5)
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [4, 5])
    assert var_9 is True



# Parsed testcases at query #44
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True
    var_8 = iter(var_3)
    var_9 = var_4.iter
    var_10 = bool(var_4.iter is var_8)
    assert var_10 is True



# Parsed testcases at query #45
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #46
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #47
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.val
    assert var_6 == 0
    var_7 = var_2.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.val
    assert var_7 == 1
    var_8 = var_3.length
    assert var_8 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.val
    assert var_8 == 1
    var_9 = var_4.length
    assert var_9 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #48
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #49
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [1, 2, 3, 4])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x == var_0
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3, 4])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 4
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4, var_0]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [4])
    assert var_8 is True



# Parsed testcases at query #50
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x >= var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [0, 1, 2, 3, 4])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [0, 1, 2, 3, 4])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 4
    var_1 = lambda x: x == var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [4])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'a'
    var_3 = 'bb'
    var_4 = 'ccc'
    var_5 = 'dddd'
    var_6 = 'eeee'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.drop_until(var_1, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['dddd', 'eeee'])
    assert var_10 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_drop_until_predicate_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(var_0)
    assert var_5 is True



# Parsed testcases at query #52
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #53
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #54
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #55
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.drop_until(var_1, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [3])
    assert var_7 is True



# Parsed testcases at query #56
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5, var_0]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [1, 2, 3, 4, 5])
    assert var_9 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_drop_until_predicate_false. Retrieved 5/11 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(var_0)
    assert var_5 is True



# Parsed testcases at query #58
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True



# Parsed testcases at query #59
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x >= var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [0, 1, 2, 3, 4])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x > var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [3, 4])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x > var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [4])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x >= var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [0, 1, 2, 3, 4])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'c'
    var_1 = lambda x: x == var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'd'
    var_5 = [var_2, var_3, var_0, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['c', 'd'])
    assert var_8 is True



# Parsed testcases at query #60
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6.list
    var_8 = bool(var_6.list == [])
    assert var_8 is True
    var_9 = var_6.exhausted
    assert var_9 is False
    var_10 = var_6.iter
    var_11 = iter(var_5)
    var_12 = type(var_11)
    var_13 = isinstance(var_10, var_12)
    var_14 = bool(var_13)
    assert var_14 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [0, 1, 2, 3, 4])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 5
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [0, 1, 2, 3, 4])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(False)
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = module_0.take(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 'hello'
    var_2 = module_0.take(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['h', 'e', 'l'])
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_drop_with_generator. Retrieved 3/6 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [4, 5])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = []
    var_2 = module_0.drop(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.drop(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 2
    var_1 = 5
    var_2 = range(var_1)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 'hello'
    var_2 = module_0.drop(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['l', 'o'])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.drop(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [5, 6, 7, 8, 9])
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6[0]
    assert var_7 == 1
    var_8 = var_6[2]
    assert var_8 == 3
    var_9 = var_6[4]
    assert var_9 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6[-1]
    assert var_7 == 5
    var_8 = var_6[-3]
    assert var_8 == 3

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6[1:4]
    var_8 = bool(var_6[1:4] == [2, 3, 4])
    assert var_8 is True
    var_9 = var_6[:3]
    var_10 = bool(var_6[:3] == [1, 2, 3])
    assert var_10 is True
    var_11 = var_6[2:]
    var_12 = bool(var_6[2:] == [3, 4, 5])
    assert var_12 is True
    var_13 = var_6[:]
    var_14 = bool(var_6[:] == [1, 2, 3, 4, 5])
    assert var_14 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4[2:2]
    var_6 = bool(var_4[2:2] == [])
    assert var_6 is True
    var_7 = var_4[5:10]
    var_8 = bool(var_4[5:10] == [])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4[var_1]
    var_6 = var_4[0]
    assert var_6 == 1
    var_7 = var_4[1]
    assert var_7 == 2
    var_8 = var_4[2]
    assert var_8 == 3



# Parsed testcases at query #4
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = next(var_3)
    assert var_4 == 1
    var_5 = next(var_3)
    assert var_5 == 2
    var_6 = next(var_3)
    assert var_6 == 3
    var_7 = next(var_3)
    assert var_7 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = next(var_4)
    assert var_5 == 0
    var_6 = next(var_4)
    assert var_6 == 2
    var_7 = next(var_4)
    assert var_7 == 4
    var_8 = next(var_4)
    assert var_8 == 6
    var_9 = next(var_4)
    assert var_9 == 8

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = next(var_3)
    assert var_4 == 1
    var_5 = next(var_3)
    assert var_5 == 2
    var_6 = next(var_3)
    var_7 = bool(False)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -3
    var_1 = 1
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = next(var_3)
    assert var_4 == -3
    var_5 = next(var_3)
    assert var_5 == -2
    var_6 = next(var_3)
    assert var_6 == -1
    var_7 = next(var_3)
    assert var_7 == 0

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 6
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = next(var_3)
    assert var_4 == 5
    var_5 = next(var_3)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = 3
    var_3 = 0
    var_4 = lambda x: x % var_2 == var_3
    var_5 = module_0.split_by(var_1, criterion=var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1, 2], [4, 5], [7, 8]])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'Split by: '
    var_1 = ' '
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [['S', 'p', 'l', 'i', 't'], ['b', 'y', ':', '']])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = ' Split by: '
    var_1 = True
    var_2 = ' '
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':', '']])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = True
    var_7 = 0
    var_8 = lambda x: x % var_1 == var_7
    var_9 = module_0.split_by(var_5, var_6, criterion=var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [[1], [], [3], [], [5]])
    assert var_11 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = ' '
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [[]])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = ' '
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[]])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.split_by(var_3)
    var_5 = list(var_4)
    var_6 = bool(False)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x == var_0
    var_5 = module_0.split_by(var_3, criterion=var_4, separator=var_1)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = []
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.chunk(var_0, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [[1], [2], [3]])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1, 2, 3]])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_0, var_2, var_3, var_4]
    var_6 = module_0.chunk(var_0, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [[1, 2], [3, 4], [5]])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = [var_1, var_2, var_0, var_3, var_4, var_5]
    var_7 = module_0.chunk(var_0, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [[1, 2, 3], [4, 5, 6]])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = 6
    var_6 = 7
    var_7 = [var_1, var_2, var_0, var_3, var_4, var_5, var_6]
    var_8 = module_0.chunk(var_0, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [[1, 2, 3], [4, 5, 6], [7]])
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)



# Parsed testcases at query #7
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = 6
    var_7 = 7
    var_8 = 8
    var_9 = 9
    var_10 = [var_2, var_3, var_4, var_5, var_0, var_6, var_7, var_8, var_9]
    var_11 = module_0.drop_until(var_1, var_10)
    var_12 = list(var_11)
    var_13 = bool(var_12 == [6, 7, 8, 9])
    assert var_13 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = 6
    var_7 = [var_2, var_3, var_4, var_5, var_0, var_6]
    var_8 = module_0.drop_until(var_1, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [6])
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'c'
    var_1 = lambda x: x == var_0
    var_2 = 'abcdef'
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == ['c', 'd', 'e', 'f'])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 4
    var_5 = 5
    var_6 = (var_2, var_3, var_0, var_4, var_5)
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [4, 5])
    assert var_9 is True



# Parsed testcases at query #9
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[0]
    assert var_5 == 1
    var_6 = var_4[2]
    assert var_6 == 5
    var_7 = var_4[4]
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[-1]
    assert var_5 == 9
    var_6 = var_4[-3]
    assert var_6 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[1:4]
    var_6 = bool(var_4[1:4] == [3, 5, 7])
    assert var_6 is True
    var_7 = var_4[:3]
    var_8 = bool(var_4[:3] == [1, 3, 5])
    assert var_8 is True
    var_9 = var_4[2:]
    var_10 = bool(var_4[2:] == [5, 7, 9])
    assert var_10 is True
    var_11 = var_4[::2]
    var_12 = bool(var_4[::2] == [1, 5, 9])
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[5:2]
    var_6 = bool(var_4[5:2] == [])
    assert var_6 is True
    var_7 = var_4[10:20]
    var_8 = bool(var_4[10:20] == [])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3[1:8:2]
    var_5 = bool(var_3[1:8:2] == [2, 4, 6, 8])
    assert var_5 is True
    var_6 = var_3[::3]
    var_7 = bool(var_3[::3] == [1, 4, 7, 10])
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_drop_until_with_custom_objects. Retrieved 5/16 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x >= var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [0, 1, 2, 3, 4])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'a'
    var_3 = 'ab'
    var_4 = 'abc'
    var_5 = 'abcd'
    var_6 = 'abcde'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.drop_until(var_1, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['abcd', 'abcde'])
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = lambda item: item.value > var_1



# Parsed testcases at query #11
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = bool(not var_1.exhausted)
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x * x
    var_7 = module_0.MapList(var_6, var_5)
    var_8 = var_7[0]
    assert var_8 == 1
    var_9 = var_7[1]
    assert var_9 == 4
    var_10 = var_7[2]
    assert var_10 == 9
    var_11 = var_7[3]
    assert var_11 == 16
    var_12 = var_7[4]
    assert var_12 == 25

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x * x
    var_7 = module_0.MapList(var_6, var_5)
    var_8 = var_7[-1]
    assert var_8 == 25
    var_9 = var_7[-2]
    assert var_9 == 16
    var_10 = var_7[-3]
    assert var_10 == 9
    var_11 = var_7[-4]
    assert var_11 == 4
    var_12 = var_7[-5]
    assert var_12 == 1

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = lambda x: x * x
    var_7 = module_0.MapList(var_6, var_5)
    var_8 = var_7[1:4]
    var_9 = bool(var_7[1:4] == [4, 9, 16])
    assert var_9 is True
    var_10 = var_7[:3]
    var_11 = bool(var_7[:3] == [1, 4, 9])
    assert var_11 is True
    var_12 = var_7[2:]
    var_13 = bool(var_7[2:] == [9, 16, 25])
    assert var_13 is True
    var_14 = var_7[::2]
    var_15 = bool(var_7[::2] == [1, 9, 25])
    assert var_15 is True
    var_16 = var_7[1:4:2]
    var_17 = bool(var_7[1:4:2] == [4, 16])
    assert var_17 is True



# Parsed testcases at query #13
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #14
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_2, var_0, var_3, var_4, var_5]
    var_7 = module_0.MapList(var_1, var_6)
    var_8 = var_7.func
    var_9 = bool(var_7.func == var_1)
    assert var_9 is True
    var_10 = var_7.list
    var_11 = bool(var_7.list == var_6)
    assert var_11 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_constructor_creates_maplist_instance. Retrieved 6/7 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #16
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #17
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_lazy_list_constructor_with_empty_iterable. Retrieved 3/4 statements.
# Partially parsed test_lazy_list_constructor_with_non_empty_iterable. Retrieved 6/7 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = bool(not var_1.exhausted)
    assert var_4 is True
    var_5 = var_1.iter

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True
    var_8 = var_4.iter



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_lazy_list_constructor_with_empty_iterable. Retrieved 3/4 statements.
# Partially parsed test_lazy_list_constructor_with_non_empty_iterable. Retrieved 6/7 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = bool(not var_1.exhausted)
    assert var_4 is True
    var_5 = var_1.iter

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True
    var_8 = var_4.iter



# Parsed testcases at query #20
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_lazy_list_constructor_with_generator. Retrieved 3/6 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = bool(not var_1.exhausted)
    assert var_4 is True
    var_5 = 'iter'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = 'iter'



# Parsed testcases at query #22
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #23
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #24
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #25
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #26
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #27
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_drop_until_with_custom_objects. Retrieved 5/15 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [1, 2, 3, 4])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x == var_0
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3, 4])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 4
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4, var_0]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [4])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'a'
    var_3 = 'ab'
    var_4 = 'abc'
    var_5 = 'abcd'
    var_6 = 'abcde'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.drop_until(var_1, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['abcd', 'abcde'])
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = lambda item: item.value > var_1



# Parsed testcases at query #29
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #30
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == var_2)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = [var_1, var_2]
    var_4 = module_0.MapList(var_0, var_3)
    var_5 = var_4.func
    var_6 = bool(var_4.func == var_0)
    assert var_6 is True
    var_7 = var_4.list
    var_8 = bool(var_4.list == var_3)
    assert var_8 is True



# Parsed testcases at query #31
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_drop_until_predicate_evaluates_to_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(var_0)
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #34
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.val
    assert var_6 == 0
    var_7 = var_2.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.val
    assert var_7 == 1
    var_8 = var_3.length
    assert var_8 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.val
    assert var_8 == 1
    var_9 = var_4.length
    assert var_9 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #35
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #36
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #37
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_drop_until_predicate_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = False
    assert var_0 is True
    var_1 = 6
    var_2 = 7
    var_3 = 8
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #39
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = 'iter'
    var_9 = hasattr(var_4, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #40
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #41
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6.list
    var_8 = bool(var_6.list == [])
    assert var_8 is True
    var_9 = var_6.exhausted
    assert var_9 is False
    var_10 = 'iter'
    var_11 = hasattr(var_6, var_10)
    var_12 = bool(var_11)
    assert var_12 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_lazy_list_constructor_with_empty_iterable. Retrieved 3/4 statements.
# Partially parsed test_lazy_list_constructor_with_non_empty_iterable. Retrieved 6/7 statements.
# Failed to parse test_lazy_list_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = var_1.list
    var_3 = bool(var_1.list == [])
    assert var_3 is True
    var_4 = bool(not var_1.exhausted)
    assert var_4 is True
    var_5 = var_1.iter

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.list
    var_6 = bool(var_4.list == [])
    assert var_6 is True
    var_7 = bool(not var_4.exhausted)
    assert var_7 is True
    var_8 = var_4.iter



# Parsed testcases at query #43
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [1, 2, 3, 4])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [1, 2, 3, 4])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 4
    var_5 = [var_2, var_3, var_0, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [4])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'a'
    var_3 = 'bb'
    var_4 = 'ccc'
    var_5 = 'dddd'
    var_6 = 'eeee'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.drop_until(var_1, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['dddd', 'eeee'])
    assert var_10 is True



# Parsed testcases at query #44
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_drop_until_with_custom_objects. Retrieved 5/16 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [6, 7, 8, 9])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x >= var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [0, 1, 2, 3, 4])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x > var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 5
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [0, 1, 2, 3, 4])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda s: len(s) > var_0
    var_2 = 'a'
    var_3 = 'ab'
    var_4 = 'abc'
    var_5 = 'abcd'
    var_6 = 'abcde'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.drop_until(var_1, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['abcd', 'abcde'])
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = lambda item: item.value > var_1



# Parsed testcases at query #46
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #47
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 10
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #48
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.val
    assert var_6 == 0
    var_7 = var_2.length
    assert var_7 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 11
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.val
    assert var_7 == 1
    var_8 = var_3.length
    assert var_8 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.val
    assert var_8 == 1
    var_9 = var_4.length
    assert var_9 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #49
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5.func
    var_7 = bool(var_5.func == var_1)
    assert var_7 is True
    var_8 = var_5.list
    var_9 = bool(var_5.list == var_4)
    assert var_9 is True



# Parsed testcases at query #50
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2.l
    assert var_3 == 0
    var_4 = var_2.r
    assert var_4 == 10
    var_5 = var_2.step
    assert var_5 == 1
    var_6 = var_2.length
    assert var_6 == 10

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 1
    var_5 = var_3.r
    assert var_5 == 10
    var_6 = var_3.step
    assert var_6 == 1
    var_7 = var_3.length
    assert var_7 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 1
    var_6 = var_4.r
    assert var_6 == 11
    var_7 = var_4.step
    assert var_7 == 2
    var_8 = var_4.length
    assert var_8 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6]
    var_8 = module_0.Range(*var_7)
    var_9 = bool(False)
    assert var_9 is True



