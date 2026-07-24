####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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
    var_0 = 5
    var_1 = 10
    var_2 = range(var_1)
    var_3 = module_0.take(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [0, 1, 2, 3, 4])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = [var_1, var_0, var_2]
    var_4 = module_0.take(var_0, var_3)
    var_5 = iter(var_4)
    var_6 = bool(var_5 is var_4)
    assert var_6 is True



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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #4
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = next(var_2)
    assert var_3 == 0
    var_4 = next(var_2)
    assert var_4 == 1
    var_5 = next(var_2)
    assert var_5 == 2
    var_6 = next(var_2)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = 8
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = next(var_3)
    assert var_4 == 5
    var_5 = next(var_3)
    assert var_5 == 6
    var_6 = next(var_3)
    assert var_6 == 7
    var_7 = next(var_3)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = next(var_4)
    assert var_5 == 1
    var_6 = next(var_4)
    assert var_6 == 4
    var_7 = next(var_4)
    assert var_7 == 7
    var_8 = next(var_4)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = [var_0, var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = next(var_2)



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_lazy_list_constructor_with_generator.


import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = iter(var_5)
    var_7 = var_4.iter
    var_8 = bool(var_4.iter == var_6)
    assert var_8 is True
    var_9 = var_4.exhausted
    assert var_9 is False
    var_10 = var_4.list
    var_11 = bool(var_4.list == [])
    assert var_11 is True



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



# Parsed testcases at query #7
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
    var_4 = bool(var_3 == [['a'], ['b']])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = ',a,b,'
    var_1 = True
    var_2 = ','
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[], ['a'], ['b'], []])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = module_0.split_by(var_2, criterion=var_4, separator=var_3)
    var_6 = list(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

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
    var_0 = []
    var_1 = False
    var_2 = lambda x: var_1
    var_3 = module_0.split_by(var_0, criterion=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[]])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = True
    var_7 = module_0.split_by(var_3, var_6, criterion=var_5)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [[], [], [], []])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = False
    var_7 = module_0.split_by(var_3, var_6, criterion=var_5)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True



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
    var_0 = 5
    var_1 = 15
    var_2 = [var_0, var_1]
    var_3 = module_0.Range(*var_2)
    var_4 = var_3[0]
    assert var_4 == 5
    var_5 = var_3[5]
    assert var_5 == 10
    var_6 = var_3[9]
    assert var_6 == 14

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
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = 5
    var_6 = slice(var_0, var_5)
    var_7 = var_4[var_6]
    var_8 = bool(var_7 == [0, 2, 4, 6, 8])
    assert var_8 is True
    var_9 = 1
    var_10 = 4
    var_11 = slice(var_9, var_10)
    var_12 = var_4[var_11]
    var_13 = bool(var_12 == [2, 4, 6])
    assert var_13 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 1
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = 2
    var_6 = slice(var_0, var_1, var_5)
    var_7 = var_4[var_6]
    var_8 = bool(var_7 == [0, 2, 4, 6, 8])
    assert var_8 is True
    var_9 = None
    var_10 = -1
    var_11 = slice(var_9, var_9, var_10)
    var_12 = var_4[var_11]
    var_13 = bool(var_12 == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    assert var_13 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = 5
    var_4 = var_2[var_3]
    var_5 = 'Should have raised IndexError'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #10
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
    var_1 = lambda x: x < var_0
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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = [var_2, var_0, var_2, var_0]
    var_4 = module_0.drop_until(var_1, var_3)
    var_5 = list(var_4)
    var_6 = bool(var_5 == [2, 1, 2])
    assert var_6 is True



# Parsed testcases at query #11
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



# Parsed testcases at query #13
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.LazyList(var_4)
    var_6 = var_5[0]
    assert var_6 == 10
    var_7 = var_5[2]
    assert var_7 == 30
    var_8 = var_5.list
    var_9 = bool(var_5.list == [10, 20, 30])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = range(var_0)
    var_2 = module_0.LazyList(var_1)
    var_3 = 1
    var_4 = 4
    var_5 = var_2[var_3:var_4]
    var_6 = bool(var_5 == [1, 2, 3])
    assert var_6 is True
    var_7 = var_2.list
    var_8 = bool(var_2.list == [1, 2, 3, 4])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = None
    var_6 = var_4[var_0:var_5]
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True
    var_8 = var_4.exhausted
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.LazyList(var_2)
    var_4 = 5
    var_5 = var_3[var_4]
    var_6 = bool(True)
    assert var_6 is True
    var_7 = 'IndexError not raised'
    var_8 = AssertionError(var_7)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = -1
    var_6 = var_4[var_5]
    var_7 = var_4.exhausted
    assert var_7 is True
    var_8 = var_4.list
    var_9 = bool(var_4.list == [1, 2, 3])
    assert var_9 is True



# Parsed testcases at query #14
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



# Parsed testcases at query #17
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



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------




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
    var_0 = 2
    var_1 = 5
    var_2 = range(var_1)
    var_3 = module_0.drop(var_0, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [2, 3, 4])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'hello'
    var_2 = module_0.drop(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['l', 'l', 'o'])
    assert var_4 is True



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

# Failed to parse test_lazy_list_constructor_with_generator.


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



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_lazy_list_constructor_with_generator.


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



# Parsed testcases at query #24
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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Range(*var_0)



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_drop_until_basic_functionality. Retrieved 4/6 statements.
# Partially parsed test_drop_until_all_elements_dropped. Retrieved 4/6 statements.
# Partially parsed test_drop_until_first_element_matches. Retrieved 6/8 statements.
# Partially parsed test_drop_until_empty_iterable. Retrieved 3/5 statements.
# Partially parsed test_drop_until_strings. Retrieved 7/9 statements.
# Partially parsed test_drop_until_none_match_single_element. Retrieved 4/6 statements.
# Partially parsed test_drop_until_preserves_types. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)

def test_case_0():
    var_0 = 20
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = range(var_2)

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x == var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_2, var_3, var_4]

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = []

def test_case_0():
    var_0 = 'target'
    var_1 = lambda x: x == var_0
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_2, var_3, var_0, var_4, var_5]

def test_case_0():
    var_0 = False
    var_1 = lambda x: var_0
    var_2 = 1
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = 'hello'
    var_3 = True
    var_4 = [var_0, var_1, var_2, var_3]



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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #34
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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_lazy_list_constructor_with_generator. Retrieved 2/6 statements.


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



# Parsed testcases at query #40
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



# Parsed testcases at query #41
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
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.MapList(var_0, var_3)
    var_5 = 5
    var_6 = var_4[var_5]
    var_7 = bool(True)
    assert var_7 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = []
    var_2 = module_0.MapList(var_0, var_1)
    var_3 = 0
    var_4 = var_2[var_3:var_3]
    var_5 = len(var_4)
    assert var_5 == 0



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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_lazy_list_constructor_with_generator. Retrieved 2/6 statements.


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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_scanl_with_initial_value. Retrieved 6/10 statements.
# Partially parsed test_scanl_without_initial_value. Retrieved 5/9 statements.
# Partially parsed test_scanl_empty_iterable_with_initial. Retrieved 2/6 statements.
# Partially parsed test_scanl_multiple_arguments_error. Retrieved 3/8 statements.
# Partially parsed test_scanl_single_element_with_initial. Retrieved 3/7 statements.
# Partially parsed test_scanl_single_element_no_initial. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda s, x: x + s
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 'd'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = []
    var_7 = module_0.scanl(var_0, var_5, *var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['a', 'ba', 'cba', 'dcba'])
    assert var_9 is True

def test_case_0():
    var_0 = []
    var_1 = 5

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 0
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 2

def test_case_0():
    var_0 = 5
    var_1 = [var_0]



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_lazy_list_constructor_handles_generator. Retrieved 2/4 statements.


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



# Parsed testcases at query #6
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



# Parsed testcases at query #7
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
    var_4 = bool(var_3 == [['a'], ['b'], ['c']])
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
    var_0 = 'a..b'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [['a'], ['b']])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = ' '
    var_6 = module_0.split_by(var_2, criterion=var_4, separator=var_5)
    var_7 = list(var_6)
    var_8 = bool(False)
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.split_by(var_2)
    var_4 = list(var_3)
    var_5 = bool(False)
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = '.a.b'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[], ['a'], ['b']])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a.b.'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [['a'], ['b'], []])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = module_0.split_by(var_0, criterion=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = False
    var_7 = module_0.split_by(var_3, var_6, criterion=var_5)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = True
    var_7 = module_0.split_by(var_3, var_6, criterion=var_5)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [[], [], [], []])
    assert var_9 is True



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
    var_0 = 'a.b.c'
    var_1 = '.'
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [['a'], ['b'], ['c']])
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
    var_0 = 'a..b'
    var_1 = False
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [['a'], ['b']])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = '.a.b'
    var_1 = '.'
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [['a'], ['b']])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = '.a.b'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [[], ['a'], ['b']])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a.b.'
    var_1 = '.'
    var_2 = module_0.split_by(var_0, separator=var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [['a'], ['b']])
    assert var_4 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 'a.b.'
    var_1 = True
    var_2 = '.'
    var_3 = module_0.split_by(var_0, var_1, separator=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [['a'], ['b'], []])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = '.'
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
    var_0 = []
    var_1 = False
    var_2 = lambda x: var_1
    var_3 = module_0.split_by(var_0, criterion=var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True



# Parsed testcases at query #9
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
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2[1:4]
    var_4 = bool(var_2[1:4] == [1, 2, 3])
    assert var_4 is True
    var_5 = var_2[:3]
    var_6 = bool(var_2[:3] == [0, 1, 2])
    assert var_6 is True
    var_7 = var_2[7:]
    var_8 = bool(var_2[7:] == [7, 8, 9])
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2[::2]
    var_4 = bool(var_2[::2] == [0, 2, 4, 6, 8])
    assert var_4 is True
    var_5 = var_2[1:8:3]
    var_6 = bool(var_2[1:8:3] == [1, 4, 7])
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = module_0.Range(*var_1)
    var_3 = var_2[-5]
    assert var_3 == 0



# Parsed testcases at query #10
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



# Parsed testcases at query #11
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
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.chunk(var_0, var_3)
    var_5 = list(var_4)
    var_6 = bool(False)
    assert var_6 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = -5
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.chunk(var_0, var_3)
    var_5 = list(var_4)
    var_6 = bool(False)
    assert var_6 is True



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



# Parsed testcases at query #14
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
    var_8 = var_4.iter
    var_9 = list(var_8)
    var_10 = bool(var_9 == [1, 2, 3])
    assert var_10 is True



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
    var_7 = var_2.val
    assert var_7 == 0

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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #18
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = iter(var_5)
    var_7 = var_4.iter
    var_8 = bool(var_4.iter == var_6)
    assert var_8 is True
    var_9 = var_4.exhausted
    assert var_9 is False
    var_10 = var_4.list
    var_11 = bool(var_4.list == [])
    assert var_11 is True



# Parsed testcases at query #19
#--------------------------




import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = 40
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.LazyList(var_4)
    var_6 = var_5[1]
    assert var_6 == 20
    var_7 = var_5[0]
    assert var_7 == 10
    var_8 = var_5[3]
    assert var_8 == 40

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
    var_9 = var_6[:2]
    var_10 = bool(var_6[:2] == [1, 2])
    assert var_10 is True
    var_11 = var_6[3:]
    var_12 = bool(var_6[3:] == [4, 5])
    assert var_12 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.LazyList(var_2)
    var_4 = 5
    var_5 = var_3[var_4]
    var_6 = bool(True)
    assert var_6 is True
    var_7 = 'IndexError not raised'
    var_8 = AssertionError(var_7)

import flutes.iterator as module_0

def test_case_0():
    var_0 = 3
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
    var_9 = var_2.exhausted
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.LazyList(var_3)
    var_5 = 0
    var_6 = None
    var_7 = var_4[var_5:var_6]
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True
    var_9 = var_4.exhausted
    assert var_9 is True



# Parsed testcases at query #20
#--------------------------




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
    var_0 = 5
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_3, var_4, var_0]
    var_6 = module_0.drop(var_0, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'hello'
    var_2 = module_0.drop(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == ['l', 'l', 'o'])
    assert var_4 is True



# Parsed testcases at query #21
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



# Parsed testcases at query #22
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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_constructor_stores_correct_values. Retrieved 5/11 statements.


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
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]
    var_3 = 0
    var_4 = 1



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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.Range(*var_4)



# Parsed testcases at query #25
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
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.drop_until(var_1, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
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



# Parsed testcases at query #27
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
    var_9 = var_8[0]
    assert var_9 == 3



# Parsed testcases at query #29
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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = module_0.MapList(var_1, var_6)
    var_8 = var_7[0:2]
    var_9 = bool(var_7[0:2] == [11, 12])
    assert var_9 is True
    var_10 = var_7[1:4]
    var_11 = bool(var_7[1:4] == [13, 14, 15])
    assert var_11 is True
    var_12 = var_7[:]
    var_13 = bool(var_7[:] == [11, 12, 13, 14])
    assert var_13 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.MapList(var_1, var_5)
    var_7 = var_6[-1]
    assert var_7 == 30
    var_8 = var_6[-3:-1]
    var_9 = bool(var_6[-3:-1] == [10, 20])
    assert var_9 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.MapList(var_0, var_3)
    var_5 = 5
    var_6 = var_4[var_5]
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #30
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



# Parsed testcases at query #31
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
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True



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

# Partially parsed test_lazy_list_constructor_handles_generator. Retrieved 2/6 statements.


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



# Parsed testcases at query #35
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
    var_11 = var_8[0:1]
    var_12 = bool(var_8[0:1] == [11])
    assert var_12 is True
    var_13 = var_8[:]
    var_14 = bool(var_8[:] == [11, 12, 13, 14, 15])
    assert var_14 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = module_0.MapList(var_0, var_3)
    var_5 = 5
    var_6 = var_4[var_5]
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = lambda x: str(x).upper()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.MapList(var_0, var_4)
    var_6 = var_5[1]
    assert var_6 == 'B'
    var_7 = var_5[0:2]
    var_8 = bool(var_5[0:2] == ['A', 'B'])
    assert var_8 is True



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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = [var_0, var_1]
    var_3 = iter(var_2)
    var_4 = module_0.LazyList(var_3)
    var_5 = var_4.iter
    var_6 = bool(var_4.iter is not None)
    assert var_6 is True
    var_7 = var_4.iter
    var_8 = next(var_7)
    assert var_8 == 10



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



# Parsed testcases at query #39
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
    var_0 = 0
    var_1 = 10
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Range(*var_3)
    var_5 = var_4.l
    assert var_5 == 0
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



# Parsed testcases at query #42
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



# Parsed testcases at query #44
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
    var_9 = var_8[0]
    assert var_9 == 3



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

# Failed to parse test_lazy_list_constructor_with_generator.


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



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_lazy_list_constructor_with_generator. Retrieved 2/6 statements.


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



# Parsed testcases at query #52
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
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = []
    var_3 = module_0.drop_until(var_1, var_2)
    var_4 = list(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.iterator as module_0

def test_case_0():
    var_0 = 1
    var_1 = lambda x: x == var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.drop_until(var_1, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

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

import flutes.iterator as module_0

def test_case_0():
    var_0 = 0
    var_1 = 2
    var_2 = lambda x: x[var_0] == var_1
    var_3 = 1
    var_4 = 'a'
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = (var_1, var_6)
    var_8 = 3
    var_9 = 'c'
    var_10 = (var_8, var_9)
    var_11 = [var_5, var_7, var_10]
    var_12 = module_0.drop_until(var_2, var_11)
    var_13 = list(var_12)
    var_14 = bool(var_13 == [(2, 'b'), (3, 'c')])
    assert var_14 is True



# Parsed testcases at query #53
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
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True



# Parsed testcases at query #54
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



# Parsed testcases at query #55
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



# Parsed testcases at query #56
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



