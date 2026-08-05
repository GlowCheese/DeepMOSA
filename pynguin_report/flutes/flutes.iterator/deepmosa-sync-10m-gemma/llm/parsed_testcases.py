####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_drop_with_generator. Retrieved 3/6 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 0
    var_2 = 1
    var_3 = 2
    var_4 = 4
    var_5 = 5
    var_6 = [var_1, var_2, var_3, var_0, var_4, var_5]
    var_7 = module_0.drop(var_0, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [3, 4, 5])
    assert var_9 is True

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
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2, var_0]
    var_4 = module_0.drop(var_0, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.drop(var_0, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

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
    var_0 = 5
    var_1 = range(var_0)
    var_2 = 2



# Parsed testcases at query #2
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
    var_0 = 'a.b.c'
    var_1 = '.'
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [['a', 'b', 'c']])
    assert var_4 is True
    var_5 = module_0.split_by(var_0, separator=var_1)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [['a'], ['b'], ['c']])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = ' Split by: '
    var_1 = True
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[], ['S', 'p', 'l', 'i', 't'], ['b', 'y', ':'], []])
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
    var_7 = bool(True)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = lambda x: x > var_0
    var_5 = '.'
    var_6 = module_0.split_by(var_3, criterion=var_4, separator=var_5)
    var_7 = list(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True
    var_6 = []
    var_7 = True
    var_8 = module_0.split_by(var_6, var_7, separator=var_2)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [[]])
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a..b'
    var_1 = '.'
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [['a'], ['b']])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a..b'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [['a'], [], ['b']])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = '.a.'
    var_1 = '.'
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [['a']])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = '.a.'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[], ['a'], []])
    assert var_5 is True



# Parsed testcases at query #3
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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2[0]
    assert var_3 == 0
    var_4 = var_2[5]
    assert var_4 == 5
    var_5 = var_2[9]
    assert var_5 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3[0]
    assert var_4 == 1
    var_5 = var_3[1]
    assert var_5 == 2
    var_6 = var_3[8]
    assert var_6 == 9

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
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2[-1]
    assert var_3 == 9
    var_4 = var_2[-10]
    assert var_4 == 0

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[0:3]
    var_6 = bool(var_4[0:3] == [1, 3, 5])
    assert var_6 is True
    var_7 = var_4[1:4]
    var_8 = bool(var_4[1:4] == [3, 5, 7])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[::2]
    var_6 = bool(var_4[::2] == [0, 2, 4, 6, 8])
    assert var_6 is True
    var_7 = var_4[1::2]
    var_8 = bool(var_4[1::2] == [1, 3, 5, 7, 9])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4[-4:-1]
    var_6 = bool(var_4[-4:-1] == [6, 7, 8])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2[0]
    assert var_3 == 0



# Parsed testcases at query #5
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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #7
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_2, var_0, var_3, var_4]
    var_6 = module_0.take(var_0, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
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
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

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
    var_0 = -1
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 5
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [0, 1])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 'hello'
    var_2 = module_0.take(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['h', 'e', 'l'])
    assert var_4 is True



# Parsed testcases at query #8
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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_lazy_list_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2.iter
    var_4 = bool(var_2.iter is not None)
    assert var_4 is True
    var_5 = var_2.exhausted
    assert var_5 is False
    var_6 = var_2.list
    var_7 = bool(var_2.list == [])
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_maplist_constructor_works_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



# Parsed testcases at query #12
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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #13
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = [var_2, var_0, var_3, var_4]
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = var_6[0]
    assert var_7 == 2
    var_8 = var_6[2]
    assert var_8 == 6

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = 5
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.MapList(var_1, var_7)
    var_9 = var_8[1:4]
    var_10 = bool(var_8[1:4] == [11, 12, 13])
    assert var_10 is True
    var_11 = var_8[:2]
    var_12 = bool(var_8[:2] == [10, 11])
    assert var_12 is True
    var_13 = var_8[3:]
    var_14 = bool(var_8[3:] == [13, 14, 15])
    assert var_14 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = var_2[0:0]
    var_4 = bool(var_2[0:0] == [])
    assert var_4 is True



# Parsed testcases at query #14
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
    var_0 = 20
    var_1 = lambda x: x > var_0
    var_2 = 10
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
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = lambda s: s == var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_0, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['target', 'c'])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x < var_0
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
    var_0 = None
    var_1 = lambda x: x is var_0
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.drop_until(var_1, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [None, 2])
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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #16
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2[0]
    assert var_3 == 0
    var_4 = var_2.list
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_2[2]
    assert var_6 == 2
    var_7 = var_2.list
    var_8 = len(var_7)
    assert var_8 == 3
    var_9 = var_2[1]
    assert var_9 == 1

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = 50
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = var_6[1:4]
    var_8 = bool(var_6[1:4] == [20, 30, 40])
    assert var_8 is True
    var_9 = var_6.list
    var_10 = len(var_9)
    assert var_10 == 4
    var_11 = var_6[:2]
    var_12 = bool(var_6[:2] == [10, 20])
    assert var_12 is True
    var_13 = var_6.list
    var_14 = len(var_13)
    assert var_14 == 2

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.LazyList(var_2)
    var_4 = 5
    var_5 = var_3[var_4]
    var_6 = 'IndexError not raised'
    var_7 = AssertionError(var_6)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = iter(var_2)
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4[0]
    assert var_5 == 1
    var_6 = var_4[1]
    assert var_6 == 2
    var_7 = var_4.exhausted
    assert var_7 is True
    var_8 = var_4[0]
    assert var_8 == 1



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
    assert var_8 == 4

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #18
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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #19
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_lazy_list_constructor_with_generator. Retrieved 2/4 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = iter(var_3)
    var_6 = var_4.iter
    var_7 = bool(var_4.iter == var_5)
    assert var_7 is True
    var_8 = var_4.exhausted
    assert var_8 is False
    var_9 = var_4.list
    var_10 = bool(var_4.list == [])
    assert var_10 is True

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)



# Parsed testcases at query #22
#--------------------------




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
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = 4
    var_4 = [var_1, var_0, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1, 2], [3, 4]])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = [var_1]
    var_3 = module_0.chunk(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[1]])
    assert var_5 is True

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
    var_0 = 100
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
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.chunk(var_0, var_3)
    var_5 = list(var_4)

import flutes.iterator as module_0

def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.chunk(var_0, var_3)
    var_5 = list(var_4)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



# Parsed testcases at query #24
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
    var_0 = 20
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [0, 1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x < var_0
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
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = iter(var_5)
    var_7 = lambda x: x == var_2
    var_8 = module_0.drop_until(var_7, var_6)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [3, 4, 5])
    assert var_10 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_drop_until_all_false. Retrieved 6/8 statements.


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
    var_0 = 20
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [0, 1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = lambda s: s == var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_2, var_3, var_0, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['target', 'c', 'd'])
    assert var_9 is True

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #28
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 4
    var_5 = 5
    var_6 = [var_2, var_3, var_0, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [3, 4, 5])
    assert var_9 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_lazy_list_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_lazy_list_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True



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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #34
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_lazy_list_constructor_handles_generator. Retrieved 2/4 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)



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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #37
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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #39
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 4
    var_5 = 5
    var_6 = [var_2, var_3, var_0, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [3, 4, 5])
    assert var_9 is True



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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



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
    var_0 = 5
    var_1 = 15
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 5
    var_5 = var_3.r
    assert var_5 == 15
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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #43
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True



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
    var_2 = 'Should raise ValueError for zero arguments'
    var_3 = AssertionError(var_2)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = 'Should raise ValueError for more than three arguments'
    var_7 = AssertionError(var_6)



# Parsed testcases at query #45
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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



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
    var_0 = 5
    var_1 = 15
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 5
    var_5 = var_3.r
    assert var_5 == 15
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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



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
    var_2 = 'Should have raised ValueError'
    var_3 = AssertionError(var_2)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = 'Should have raised ValueError'
    var_7 = AssertionError(var_6)



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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



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
    var_0 = 20
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [0, 1, 2, 3])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = lambda s: s == var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_2, var_3, var_0, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['target', 'c', 'd'])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x < var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = -1
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [-1])
    assert var_9 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_lazy_list_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2.iter
    var_4 = bool(var_2.iter is not None)
    assert var_4 is True
    var_5 = var_2.exhausted
    assert var_5 is False
    var_6 = var_2.list
    var_7 = bool(var_2.list == [])
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = len(var_4)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4[var_1]
    var_6 = len(var_4)
    assert var_6 == 3

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = len(var_4)
    assert var_5 == 3

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.LazyList(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.LazyList(var_5)
    var_7 = 0
    var_8 = slice(var_7, var_1)
    var_9 = var_6[var_8]
    var_10 = len(var_6)
    var_11 = var_6[var_3]
    var_12 = len(var_6)
    assert var_12 == 5



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_lazy_list_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True



# Parsed testcases at query #5
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
    var_0 = 20
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_0, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_2, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [2, 3, 4, 5])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = lambda s: s == var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_2, var_3, var_0, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['target', 'c', 'd'])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x < var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True



# Parsed testcases at query #6
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2[0]
    assert var_3 == 0
    var_4 = var_2[5]
    assert var_4 == 5
    var_5 = var_2[9]
    assert var_5 == 9

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 11
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3[0]
    assert var_4 == 1
    var_5 = var_3[5]
    assert var_5 == 6
    var_6 = var_3[9]
    assert var_6 == 10

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
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2[-1]
    assert var_3 == 9
    var_4 = var_2[-5]
    assert var_4 == 5
    var_5 = var_2[-10]
    assert var_5 == 0

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3[1:4]
    var_5 = bool(var_3[1:4] == [1, 2, 3])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3[::2]
    var_5 = bool(var_3[::2] == [0, 2, 4, 6, 8])
    assert var_5 is True
    var_6 = var_3[1:8:3]
    var_7 = bool(var_3[1:8:3] == [1, 4, 7])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3[-5:-2]
    var_5 = bool(var_3[-5:-2] == [5, 6, 7])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 5
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3[5]
    assert var_4 == 5

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3[5:2]
    var_5 = bool(var_3[5:2] == [])
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------




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
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = 4
    var_4 = [var_1, var_0, var_2, var_3]
    var_5 = module_0.chunk(var_0, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [[1, 2], [3, 4]])
    assert var_7 is True

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
    var_1 = []
    var_2 = module_0.chunk(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

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



# Parsed testcases at query #8
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
    var_0 = 'a,b,c'
    var_1 = ','
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [['a'], ['b'], ['c']])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a,,b'
    var_1 = True
    var_2 = ','
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [['a'], [], ['b']])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a,,b'
    var_1 = False
    var_2 = ','
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [['a'], ['b']])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = ',a,b,'
    var_1 = ','
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = []
    var_5 = 'a'
    var_6 = [var_5]
    var_7 = 'b'
    var_8 = [var_7]
    var_9 = []
    var_10 = [var_4, var_6, var_8, var_9]
    var_11 = var_3 == var_10
    var_12 = bool(var_11 if False else [['a'], ['b']])
    assert var_12 is True
    var_13 = True
    var_14 = module_0.split_by(var_0, var_13, separator=var_1)
    var_15 = list(var_14)
    var_16 = bool(var_15 == [[], ['a'], ['b'], []])
    assert var_16 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = ','
    var_6 = module_0.split_by(var_2, criterion=var_4, separator=var_5)
    var_7 = list(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.split_by(var_2)
    var_4 = list(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.split_by(var_3, separator=var_1)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [[1], [2]])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'abc def'
    var_1 = ' '
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [['a', 'b', 'c'], ['d', 'e', 'f']])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = module_0.split_by(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True
    var_5 = []
    var_6 = True
    var_7 = module_0.split_by(var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [[]])
    assert var_9 is True



# Parsed testcases at query #9
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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #14
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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_lazy_list_constructor_works_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = iter(var_1)
    var_3 = module_0.LazyList(var_2)
    var_4 = var_3.iter
    var_5 = bool(var_3.iter is var_2)
    assert var_5 is True



# Parsed testcases at query #16
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = 20
    var_3 = 30
    var_4 = 40
    var_5 = 50
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.take(var_0, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [10, 20, 30])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.take(var_0, var_4)
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
    var_5 = module_0.take(var_0, var_4)
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
    var_5 = module_0.take(var_0, var_4)
    var_6 = list(var_5)

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
    var_0 = 2
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [0, 1])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = [var_1, var_0, var_2]
    var_4 = module_0.take(var_0, var_3)
    var_5 = '__next__'
    var_6 = hasattr(var_4, var_5)
    var_7 = bool(var_6)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = [var_2, var_0, var_3, var_4]
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = var_6[0]
    assert var_7 == 2
    var_8 = var_6[2]
    assert var_8 == 6

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = 5
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.MapList(var_1, var_7)
    var_9 = var_8[1:4]
    var_10 = bool(var_8[1:4] == [12, 13, 14])
    assert var_10 is True
    var_11 = var_8[:2]
    var_12 = bool(var_8[:2] == [11, 12])
    assert var_12 is True
    var_13 = var_8[3:]
    var_14 = bool(var_8[3:] == [14, 15])
    assert var_14 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = 0
    var_4 = var_2[var_3]
    var_5 = var_2[0:5]
    var_6 = bool(var_2[0:5] == [])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.MapList(var_1, var_4)
    var_6 = var_5[-1]
    assert var_6 == 9
    var_7 = var_5[-3:-1]
    var_8 = bool(var_5[-3:-1] == [1, 4])
    assert var_8 is True



# Parsed testcases at query #18
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_1, var_0, var_2, var_3, var_4]
    var_6 = module_0.drop(var_0, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [3, 4, 5])
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
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2, var_0]
    var_4 = module_0.drop(var_0, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

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
    var_0 = 2
    var_1 = []
    var_2 = module_0.drop(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.drop(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [5, 6, 7, 8, 9])
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True



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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_lazy_list_constructor_handles_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = iter(var_3)
    var_6 = var_4.iter
    var_7 = bool(var_4.iter == var_5)
    assert var_7 is True
    var_8 = var_4.exhausted
    assert var_8 is False
    var_9 = var_4.list
    var_10 = bool(var_4.list == [])
    assert var_10 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = iter(var_1)
    var_4 = var_2.iter
    var_5 = bool(var_2.iter == var_3)
    assert var_5 is True
    var_6 = var_2.exhausted
    assert var_6 is False
    var_7 = var_2.list
    var_8 = bool(var_2.list == [])
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



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



# Parsed testcases at query #27
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
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.drop_until(var_1, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [0, 1, 2])
    assert var_7 is True

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
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = lambda s: s == var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_2, var_3, var_0, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['target', 'c', 'd'])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #30
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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #31
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = 6
    var_7 = 7
    var_8 = [var_2, var_3, var_4, var_5, var_0, var_6, var_7]
    var_9 = module_0.drop_until(var_1, var_8)
    var_10 = list(var_9)
    var_11 = var_10[0]
    assert var_11 == 5



# Parsed testcases at query #32
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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #33
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True



# Parsed testcases at query #34
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
    var_0 = 2
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_0, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_2, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [2, 3, 4, 5])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'c'
    var_1 = lambda x: x == var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'd'
    var_5 = 'e'
    var_6 = [var_2, var_3, var_0, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['c', 'd', 'e'])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_maplist_constructor_with_different_types. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x + var_0
    var_2 = []
    var_3 = module_0.MapList(var_1, var_2)
    var_4 = var_3.func
    var_5 = bool(var_3.func == var_1)
    assert var_5 is True
    var_6 = var_3.list
    var_7 = bool(var_3.list == [])
    assert var_7 is True



# Parsed testcases at query #37
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
    var_0 = 2
    var_1 = 0
    var_2 = lambda x: x % var_0 == var_1
    var_3 = 3
    var_4 = 4
    var_5 = [var_0, var_3, var_4]
    var_6 = module_0.drop_until(var_2, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [2, 3, 4])
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
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = lambda x: x == var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.drop_until(var_1, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #39
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #40
#--------------------------

# Failed to parse test_lazy_list_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.exhausted
    assert var_7 is False
    var_8 = var_4.list
    var_9 = bool(var_4.list == [])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = var_2.iter
    var_4 = bool(var_2.iter is not None)
    assert var_4 is True
    var_5 = var_2.exhausted
    assert var_5 is False
    var_6 = var_2.list
    var_7 = bool(var_2.list == [])
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_lazy_list_constructor_works_with_generator. Retrieved 2/6 statements.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = iter(var_3)
    var_6 = var_4.iter
    var_7 = bool(var_4.iter == var_5)
    assert var_7 is True
    var_8 = var_4.exhausted
    assert var_8 is False
    var_9 = var_4.list
    var_10 = bool(var_4.list == [])
    assert var_10 is True

def test_case_0():
    var_0 = 5
    var_1 = range(var_0)



# Parsed testcases at query #42
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.drop_until(var_1, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #43
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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



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
    var_3 = bool(True)
    assert var_3 is True

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
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #45
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
    var_0 = 5
    var_1 = 15
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3.l
    assert var_4 == 5
    var_5 = var_3.r
    assert var_5 == 15
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
    var_2 = 'Should have raised ValueError'
    var_3 = AssertionError(var_2)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)
    var_6 = 'Should have raised ValueError'
    var_7 = AssertionError(var_6)



