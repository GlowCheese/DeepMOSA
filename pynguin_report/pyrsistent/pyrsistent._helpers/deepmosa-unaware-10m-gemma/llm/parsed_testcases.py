####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = [var_0, var_1]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = [var_0, var_1]
    var_10 = [var_0, var_1]
    var_11 = [var_0, var_1]
    var_12 = 3
    var_13 = [var_1, var_12]
    var_14 = {var_3: var_13}
    var_15 = 4
    var_16 = 'b'
    var_17 = 5
    var_18 = {var_16: var_17}
    var_19 = (var_15, var_18)
    var_20 = [var_0, var_14, var_19]
    var_21 = 'data'
    var_22 = [var_0, var_1]
    var_23 = {var_21: var_22}
    var_24 = var_23[var_21]



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = 'a'
    var_8 = 'c'
    var_9 = 'b'
    var_10 = {var_9: var_1}
    var_11 = [var_0, var_10]
    var_12 = 4
    var_13 = (var_2, var_12)
    var_14 = {var_7: var_11, var_8: var_13}
    var_15 = [var_0, var_1, var_2]
    var_16 = [var_0, var_1, var_2]
    var_17 = 'key'
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = [var_19]
    var_21 = 0



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = [var_0, var_1]
    var_6 = {var_3: var_0}
    var_7 = [var_0, var_1]
    var_8 = 3
    var_9 = [var_1, var_8]
    var_10 = {var_3: var_9}
    var_11 = 4
    var_12 = 5
    var_13 = [var_12]
    var_14 = (var_11, var_13)
    var_15 = [var_0, var_10, var_14]
    var_16 = [var_0, var_1, var_8]
    var_17 = [var_1]
    var_18 = (var_0, var_17)
    var_19 = {var_0, var_1, var_8}



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = [var_0, var_1]
    var_7 = {var_4: var_0}
    var_8 = [var_0, var_1]
    var_9 = {var_4: var_0}
    var_10 = module_0.pmap(var_9)
    var_11 = 'inner'
    var_12 = 'original'
    var_13 = {var_11: var_12}
    var_14 = [var_13]
    var_15 = {var_11: var_12}
    var_16 = module_0.pmap(var_15)
    var_17 = [var_16]
    var_18 = [var_0]
    var_19 = 'c'
    var_20 = {var_19: var_1}
    var_21 = 'b'
    var_22 = [var_0]
    var_23 = {var_19: var_1}
    var_24 = module_0.pmap(var_23)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = [var_0, var_1]
    var_6 = {var_3: var_0}
    var_7 = [var_0, var_1]
    var_8 = 'inner'
    var_9 = [var_0]
    var_10 = {var_8: var_9}
    var_11 = [var_0]
    var_12 = 3
    var_13 = [var_1, var_12]
    var_14 = {var_3: var_13}
    var_15 = 4
    var_16 = 5
    var_17 = {var_16}
    var_18 = (var_15, var_17)
    var_19 = [var_0, var_14, var_18]
    var_20 = [var_1, var_12]
    var_21 = 'x'
    var_22 = {var_21: var_0}
    var_23 = module_0.pmap(var_22)
    var_24 = [var_23]
    var_25 = 0



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'inner'
    var_5 = [var_0, var_1]
    var_6 = {var_4: var_5}
    var_7 = 10
    var_8 = [var_0]
    var_9 = 5
    var_10 = 'string'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_2]
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_2]
    var_6 = 'key'
    var_7 = [var_0, var_1]
    var_8 = {var_6: var_7}
    var_9 = 'inner'
    var_10 = {var_9: var_1}
    var_11 = [var_0, var_10]
    var_12 = 10
    var_13 = [var_12]
    var_14 = [var_0, var_1, var_2]
    var_15 = [var_0, var_1, var_2]
    var_16 = 'a'
    var_17 = 'c'
    var_18 = 'b'
    var_19 = {var_18: var_1}
    var_20 = [var_0, var_19]
    var_21 = 4
    var_22 = [var_21]
    var_23 = (var_2, var_22)
    var_24 = {var_16: var_20, var_17: var_23}
    var_25 = [var_0, var_1]
    var_26 = 'z'
    var_27 = {var_26: var_2}
    var_28 = 'x'
    var_29 = 'y'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = [var_0, var_9]
    var_11 = 'existing'
    var_12 = True
    var_13 = {var_11: var_12}
    var_14 = False



# Parsed testcases at query #9
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = [var_0, var_1]
    var_7 = [var_0, var_1]
    var_8 = 'key'
    var_9 = 'val'
    var_10 = {var_8: var_9}
    var_11 = 'inner'
    var_12 = [var_0, var_1]
    var_13 = {var_8: var_9}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_0]
    var_16 = {var_4: var_1}
    var_17 = [var_0]
    var_18 = {var_4: var_1}
    var_19 = module_0.pmap(var_18)
    var_20 = 'hello'
    var_21 = ' world'
    var_22 = [var_1]
    var_23 = (var_0, var_22)
    var_24 = [var_1]



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = 'x'
    var_10 = 10
    var_11 = {var_9: var_10}
    var_12 = [var_0, var_11]
    var_13 = 'y'
    var_14 = [var_0, var_1]
    var_15 = {var_13: var_14}
    var_16 = [var_0, var_1]
    var_17 = {var_13: var_16}
    var_18 = [var_0, var_1]
    var_19 = [var_0, var_1]
    var_20 = 5
    var_21 = 'string'



# Parsed testcases at query #11
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]
    var_6 = 'key'
    var_7 = [var_0, var_1]
    var_8 = {var_6: var_7}
    var_9 = 'b'
    var_10 = 'nested'
    var_11 = {var_10: var_2}
    var_12 = {var_9: var_11}
    var_13 = var_12[var_9]
    var_14 = [var_0, var_1]
    var_15 = {var_10: var_2}
    var_16 = module_0.pmap(var_15)
    var_17 = [var_0]
    var_18 = {var_1: var_2}
    var_19 = 4
    var_20 = (var_19,)
    var_21 = [var_0]
    var_22 = {var_1: var_2}
    var_23 = module_0.pmap(var_22)
    var_24 = (var_19,)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = 20
    var_3 = [var_1, var_2]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'z'
    var_9 = 0
    var_10 = 1
    var_11 = [var_10, var_0]
    var_12 = 'c'
    var_13 = 3
    var_14 = {var_12: var_13}
    var_15 = 'a'
    var_16 = 'b'
    var_17 = len(var_11)
    assert var_17 == 2



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'val'
    var_7 = [var_0, var_1]
    var_8 = {var_6: var_7}
    var_9 = [var_2, var_4]
    var_10 = [var_0, var_1]
    var_11 = [var_2, var_4]
    var_12 = 10
    var_13 = 20
    var_14 = [var_12, var_13]
    var_15 = [var_12, var_13]
    var_16 = 5
    var_17 = 'hello'
    var_18 = 'a'
    var_19 = 'c'
    var_20 = 'b'
    var_21 = {var_20: var_1}
    var_22 = [var_0, var_21]
    var_23 = (var_2, var_4)
    var_24 = {var_18: var_22, var_19: var_23}



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_2]
    var_4 = [var_0, var_1, var_3]
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_1]
    var_8 = {var_5: var_0, var_6: var_7}
    var_9 = 99
    var_10 = 'inner_list'
    var_11 = 'data'
    var_12 = [var_0, var_1]
    var_13 = {var_11: var_12}
    var_14 = 5
    var_15 = 'string'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]
    var_6 = 4
    var_7 = [var_0, var_1, var_2, var_6]
    var_8 = 'a'
    var_9 = {var_8: var_0}
    var_10 = [var_9]
    var_11 = 10
    var_12 = [var_11]
    var_13 = 0
    var_14 = 'added'
    var_15 = [var_11, var_14]
    var_16 = 'key'
    var_17 = [var_0, var_1]
    var_18 = {var_16: var_17}
    var_19 = [var_0, var_1]
    var_20 = {var_8: var_0}
    var_21 = 'b'
    var_22 = {var_21: var_1}
    var_23 = [var_20, var_22]



# Parsed testcases at query #16
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 10
    var_9 = 20
    var_10 = [var_8, var_9]
    var_11 = [var_0]
    var_12 = 'x'
    var_13 = {var_12: var_1}
    var_14 = [var_0]
    var_15 = {var_12: var_1}
    var_16 = module_0.pmap(var_15)
    var_17 = [var_0, var_1]
    var_18 = 'inner'
    var_19 = [var_1, var_2]
    var_20 = {var_18: var_19}
    var_21 = 5
    var_22 = (var_4, var_21)
    var_23 = [var_0, var_20, var_22]



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = 3
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_4]
    var_7 = 'a'
    var_8 = 'c'
    var_9 = 'd'
    var_10 = 'b'
    var_11 = {var_10: var_4}
    var_12 = [var_0, var_1, var_11]
    var_13 = 4
    var_14 = 5
    var_15 = (var_13, var_14)
    var_16 = 6
    var_17 = 7
    var_18 = {var_16, var_17}
    var_19 = {var_7: var_12, var_8: var_15, var_9: var_18}
    var_20 = 0
    var_21 = 'key'
    var_22 = [var_0, var_1]
    var_23 = {var_21: var_22}
    var_24 = 10
    var_25 = 'string'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'inner'
    var_7 = 'list'
    var_8 = 'val'
    var_9 = {var_8: var_0}
    var_10 = [var_0, var_1]
    var_11 = {var_6: var_9, var_7: var_10}
    var_12 = 'existing'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = 10
    var_16 = 'string'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = [var_0, var_1]
    var_9 = [var_0, var_1]
    var_10 = 'list'
    var_11 = 'tuple'
    var_12 = 'inner'
    var_13 = {var_12: var_1}
    var_14 = [var_0, var_13]
    var_15 = 4
    var_16 = (var_2, var_15)
    var_17 = {var_10: var_14, var_11: var_16}
    var_18 = 10
    var_19 = 20
    var_20 = [var_18, var_19]



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 'a'
    var_6 = {var_5: var_1}
    var_7 = [var_0, var_6]
    var_8 = 'key'
    var_9 = [var_0, var_1]
    var_10 = {var_8: var_9}
    var_11 = [var_0, var_1]
    var_12 = [var_0, var_1, var_2]
    var_13 = [var_1, var_2]
    var_14 = (var_0, var_13)
    var_15 = [var_2]
    var_16 = frozenset(var_15)
    var_17 = {var_0, var_1, var_16}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 3
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = [var_0, var_1, var_3]
    var_7 = {var_2: var_0}
    var_8 = 'c'
    var_9 = 'b'
    var_10 = {var_9: var_3}
    var_11 = [var_0, var_1, var_10]
    var_12 = 4
    var_13 = 5
    var_14 = 6
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)
    var_17 = {var_2: var_11, var_8: var_16}
    var_18 = [var_0, var_1]
    var_19 = 'x'
    var_20 = 10
    var_21 = {var_19: var_20}
    var_22 = [var_0, var_1]
    var_23 = [var_0]
    var_24 = {var_2: var_23}
    var_25 = [var_1]
    var_26 = (var_0, var_25)
    var_27 = {var_0, var_1}



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 10
    var_3 = 2
    var_4 = [var_0, var_3]
    var_5 = 'a'
    var_6 = 'c'
    var_7 = 'b'
    var_8 = {var_7: var_3}
    var_9 = [var_0, var_8]
    var_10 = 3
    var_11 = 4
    var_12 = (var_10, var_11)
    var_13 = {var_5: var_9, var_6: var_12}
    var_14 = 0
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = [var_0]
    var_19 = {var_5: var_18}
    var_20 = [var_0]



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 'key'
    var_7 = 'val'
    var_8 = {var_6: var_7}
    var_9 = [var_8]
    var_10 = 'old_key'
    var_11 = 'old_val'
    var_12 = {var_10: var_11}
    var_13 = 'inner'
    var_14 = {var_13: var_0}



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = [var_0, var_1]
    var_6 = [var_0, var_1]
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 0
    var_11 = 3
    var_12 = [var_0, var_1, var_11]
    var_13 = [var_0, var_1, var_11]
    var_14 = [var_1, var_11]
    var_15 = (var_0, var_14)
    var_16 = {var_0, var_1, var_11}
    var_17 = {var_0, var_1, var_11}
    var_18 = module_0.pset(var_17)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = [var_0, var_1]
    var_6 = {var_4: var_5}
    var_7 = 'nested'
    var_8 = 10
    var_9 = {var_7: var_8}
    var_10 = [var_9]
    var_11 = 'data'
    var_12 = 'items'
    var_13 = 0
    var_14 = [var_0, var_1]
    var_15 = 'a'
    var_16 = 'b'
    var_17 = [var_0, var_1]
    var_18 = 'c'
    var_19 = {var_18: var_2}
    var_20 = {var_15: var_17, var_16: var_19}
    var_21 = [var_0, var_1, var_2]
    var_22 = [var_0, var_1, var_2]
    var_23 = 5
    var_24 = 'string'
    var_25 = None



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = [var_0, var_1]
    var_7 = [var_0, var_1]
    var_8 = [var_0, var_1]
    var_9 = {var_4: var_8}
    var_10 = [var_0, var_1]
    var_11 = 10
    var_12 = [var_11]
    var_13 = [var_11]
    var_14 = 'c'
    var_15 = 'd'
    var_16 = 'b'
    var_17 = {var_16: var_1}
    var_18 = [var_0, var_17]
    var_19 = 3
    var_20 = 4
    var_21 = (var_19, var_20)
    var_22 = 5
    var_23 = 6
    var_24 = {var_22, var_23}
    var_25 = {var_4: var_18, var_14: var_21, var_15: var_24}
    var_26 = {var_16: var_1}
    var_27 = module_0.pmap(var_26)
    var_28 = [var_0, var_27]
    var_29 = (var_19, var_20)
    var_30 = {var_22, var_23}
    var_31 = module_1.pset(var_30)
    var_32 = [var_0, var_1]
    var_33 = [var_0, var_1]



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 'mutated'
    var_6 = False
    var_7 = {var_5: var_6}
    var_8 = [var_0, var_1, var_2]
    var_9 = 'a'
    var_10 = {var_9: var_0}
    var_11 = 'c'
    var_12 = 'b'
    var_13 = {var_12: var_1}
    var_14 = [var_0, var_13]
    var_15 = 4
    var_16 = (var_2, var_15)
    var_17 = {var_9: var_14, var_11: var_16}
    var_18 = 'x'
    var_19 = 10
    var_20 = [var_19]
    var_21 = {var_18: var_20}
    var_22 = {var_0, var_1, var_2}
    var_23 = {var_0, var_1, var_2}
    var_24 = module_0.pset(var_23)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0, var_1]
    var_5 = 3
    var_6 = [var_0, var_1, var_5]
    var_7 = 'a'
    var_8 = {var_7: var_0}
    var_9 = 4
    var_10 = [var_0, var_1, var_5, var_9]
    var_11 = 'new'
    var_12 = 5
    var_13 = {var_7: var_0, var_11: var_12}
    var_14 = module_0.pmap(var_13)
    var_15 = 10
    var_16 = [var_15]
    var_17 = [var_15]
    var_18 = 'c'
    var_19 = 'b'
    var_20 = {var_19: var_1}
    var_21 = [var_0, var_20]
    var_22 = (var_5, var_9)
    var_23 = {var_7: var_21, var_18: var_22}
    var_24 = [var_0, var_1]
    var_25 = 'x'
    var_26 = {var_25: var_0}
    var_27 = module_0.pmap(var_26)
    var_28 = [var_0, var_1]
    var_29 = module_1.pset(var_28)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = [var_0, var_1]
    var_6 = 3
    var_7 = {var_3: var_6}
    var_8 = [var_0, var_1]
    var_9 = [var_0, var_1]
    var_10 = [var_0, var_1]
    var_11 = 'inner'
    var_12 = {var_3: var_0}
    var_13 = {var_11: var_12}
    var_14 = {var_3: var_0}
    var_15 = module_0.pmap(var_14)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = [var_0, var_1]
    var_6 = 3
    var_7 = {var_3: var_6}
    var_8 = [var_0, var_1, var_6]
    var_9 = [var_0, var_1, var_6]
    var_10 = [var_0]
    var_11 = 'x'
    var_12 = {var_11: var_1}
    var_13 = 'inner'
    var_14 = 10
    var_15 = [var_14]
    var_16 = {var_13: var_15}
    var_17 = [var_16]
    var_18 = [var_14]



# Parsed testcases at query #8
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'default'
    var_5 = None
    var_6 = {var_2: var_0, var_3: var_1, var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = 'key'
    var_9 = 'val'
    var_10 = {var_8: var_9}
    var_11 = [var_0, var_10]
    var_12 = 'item'
    var_13 = 10
    var_14 = 20
    var_15 = [var_13, var_14]
    var_16 = {var_12: var_15}
    var_17 = 'data'
    var_18 = 'extra'
    var_19 = {var_8: var_9}
    var_20 = module_0.pmap(var_19)
    var_21 = [var_0, var_20]
    var_22 = [var_13, var_14]
    var_23 = [var_0, var_1]
    var_24 = {var_2: var_23}
    var_25 = [var_0, var_1]
    var_26 = 5
    var_27 = 'result'
    var_28 = 'list'
    var_29 = 'set'
    var_30 = 'tuple'
    var_31 = 3
    var_32 = (var_1, var_31)
    var_33 = 4
    var_34 = {var_2: var_33}
    var_35 = [var_0, var_32, var_34]
    var_36 = 6
    var_37 = {var_26, var_36}
    var_38 = 7
    var_39 = 8
    var_40 = [var_39]
    var_41 = (var_38, var_40)
    var_42 = {var_28: var_35, var_29: var_37, var_30: var_41}
    var_43 = [var_39]



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 10
    var_6 = [var_5]
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 20
    var_11 = 'inner_list'
    var_12 = 'a'
    var_13 = {var_12: var_1}
    var_14 = [var_0, var_13]
    var_15 = {var_11: var_14}
    var_16 = 4
    var_17 = (var_2, var_16)
    var_18 = [var_15, var_17]
    var_19 = 0
    var_20 = [var_0, var_1]
    var_21 = 99
    var_22 = [var_0, var_1, var_21]



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 4
    var_7 = [var_0, var_1, var_2, var_6]
    var_8 = 'inner'
    var_9 = [var_1, var_2]
    var_10 = {var_8: var_9}
    var_11 = [var_0, var_10]
    var_12 = 5
    var_13 = 'string'
    var_14 = 3



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 3
    var_6 = [var_0, var_1, var_5]
    var_7 = 4
    var_8 = [var_0, var_1, var_5, var_7]
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 'inner'
    var_13 = 'original'
    var_14 = {var_12: var_13}
    var_15 = [var_14]
    var_16 = [var_1]
    var_17 = (var_0, var_16)
    var_18 = 'existing'
    var_19 = True
    var_20 = {var_18: var_19}



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1]
    var_6 = 'a'
    var_7 = {var_6: var_2}
    var_8 = 4
    var_9 = 5
    var_10 = [var_8, var_9]
    var_11 = [var_0, var_1]
    var_12 = {var_6: var_11}
    var_13 = 'b'
    var_14 = (var_2, var_8)
    var_15 = {var_13: var_14}
    var_16 = [var_12, var_15]
    var_17 = 0
    var_18 = [var_0, var_1, var_2]
    var_19 = 'inner'
    var_20 = [var_0]
    var_21 = {var_19: var_20}
    var_22 = {var_0, var_1, var_2}



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'existing'
    var_9 = 123
    var_10 = {var_8: var_9}
    var_11 = 'inner'
    var_12 = [var_0]
    var_13 = {var_11: var_12}
    var_14 = [var_13]
    var_15 = [var_0, var_1]



# Parsed testcases at query #14
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = 3
    var_10 = [var_4, var_8, var_9]
    var_11 = module_0.freeze(var_10)
    var_12 = [var_8, var_9]
    var_13 = [var_4, var_12]
    var_14 = module_0.freeze(var_13)
    var_15 = [var_8, var_9]
    var_16 = []
    var_17 = module_0.freeze(var_16)
    var_18 = []
    var_19 = 'a'
    var_20 = {var_19: var_4}
    var_21 = module_0.freeze(var_20)
    var_22 = 'b'
    var_23 = [var_8, var_9]
    var_24 = {var_19: var_4, var_22: var_23}
    var_25 = module_0.freeze(var_24)
    var_26 = [var_8, var_9]
    var_27 = {}
    var_28 = module_0.freeze(var_27)
    var_29 = {}
    var_30 = module_1.pmap(var_29)
    var_31 = [var_8]
    var_32 = (var_4, var_31)
    var_33 = module_0.freeze(var_32)
    var_34 = [var_8]
    var_35 = {var_19: var_8}
    var_36 = (var_4, var_35)
    var_37 = module_0.freeze(var_36)
    var_38 = {var_19: var_8}
    var_39 = module_1.pmap(var_38)
    var_40 = (var_4, var_39)
    var_41 = {var_4, var_8}
    var_42 = module_0.freeze(var_41)
    var_43 = {var_4, var_8}
    var_44 = module_0.freeze(var_43)
    var_45 = [var_4, var_8]
    var_46 = module_2.pset(var_45)
    var_47 = [var_4, var_8]
    var_48 = {var_19: var_47}
    var_49 = [var_4, var_8]
    var_50 = [var_8]
    var_51 = [var_4, var_50]
    var_52 = False
    var_53 = module_0.freeze(var_51, var_52)
    var_54 = [var_8]
    var_55 = 'd'
    var_56 = 'c'
    var_57 = {var_56: var_9}
    var_58 = [var_4, var_8, var_57]
    var_59 = 4
    var_60 = 5
    var_61 = (var_59, var_60)
    var_62 = 6
    var_63 = 7
    var_64 = {var_62, var_63}
    var_65 = {var_19: var_58, var_22: var_61, var_55: var_64}
    var_66 = {var_56: var_9}
    var_67 = module_1.pmap(var_66)
    var_68 = [var_4, var_8, var_67]
    var_69 = (var_59, var_60)
    var_70 = [var_62, var_63]
    var_71 = module_2.pset(var_70)
    var_72 = module_0.freeze(var_65)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0]
    var_3 = [var_1]
    var_4 = [var_0, var_1]



# Parsed testcases at query #15
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]
    var_6 = 'a'
    var_7 = [var_0, var_1]
    var_8 = {var_6: var_7}
    var_9 = 'x'
    var_10 = 10
    var_11 = {var_9: var_10}
    var_12 = 11
    var_13 = {var_9: var_12}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_0]
    var_16 = {var_1: var_2}
    var_17 = 4
    var_18 = (var_17,)
    var_19 = [var_0]
    var_20 = {var_1: var_2}
    var_21 = module_0.pmap(var_20)
    var_22 = (var_17,)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_2]
    var_4 = [var_0, var_1, var_3]
    var_5 = [var_2]
    var_6 = [var_0, var_1]
    var_7 = 'a'
    var_8 = {var_7: var_0}
    var_9 = [var_0, var_1]
    var_10 = 'c'
    var_11 = 'b'
    var_12 = {var_11: var_1}
    var_13 = [var_0, var_12]
    var_14 = 4
    var_15 = (var_2, var_14)
    var_16 = {var_7: var_13, var_10: var_15}
    var_17 = 10
    var_18 = 'string'
    var_19 = [var_0, var_1, var_2]
    var_20 = [var_0, var_1, var_2]



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = [var_0, var_1, var_2]
    var_7 = 'b'
    var_8 = [var_0, var_1]
    var_9 = 'c'
    var_10 = {var_9: var_2}
    var_11 = {var_4: var_8, var_7: var_10}
    var_12 = [var_0]
    var_13 = 'z'
    var_14 = {var_13: var_1}
    var_15 = 'x'
    var_16 = 'y'
    var_17 = [var_1, var_2]
    var_18 = (var_0, var_17)



# Parsed testcases at query #18
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'new_key'
    var_8 = 'new_value'
    var_9 = {var_5: var_0, var_7: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = 'data'
    var_12 = 10
    var_13 = 20
    var_14 = [var_12, var_13]
    var_15 = {var_11: var_14}
    var_16 = [var_12, var_13]
    var_17 = [var_12, var_13]
    var_18 = [var_1, var_2]
    var_19 = {var_5: var_18}
    var_20 = 4
    var_21 = 5
    var_22 = (var_20, var_21)
    var_23 = [var_0, var_19, var_22]
    var_24 = [var_1, var_2]
    var_25 = [var_21]
    var_26 = 0



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = {var_4: var_0}
    var_7 = [var_1, var_2]
    var_8 = {var_4: var_7}
    var_9 = 4
    var_10 = 5
    var_11 = (var_9, var_10)
    var_12 = [var_0, var_8, var_11]



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = [var_0, var_1]
    var_7 = 3
    var_8 = [var_0, var_1, var_7]
    var_9 = [var_0]
    var_10 = {var_4: var_9}
    var_11 = [var_1, var_7]
    var_12 = 'new'
    var_13 = [var_0]
    var_14 = [var_1, var_7]
    var_15 = 10
    var_16 = [var_15]
    var_17 = [var_15]
    var_18 = [var_0, var_1]
    var_19 = [var_0, var_1]



# Parsed testcases at query #21
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = [var_0, var_1]
    var_7 = {var_4: var_0}
    var_8 = [var_0, var_1]
    var_9 = 3
    var_10 = [var_1, var_9]
    var_11 = {var_4: var_10}
    var_12 = 4
    var_13 = 5
    var_14 = (var_12, var_13)
    var_15 = [var_0, var_11, var_14]
    var_16 = [var_1, var_9]
    var_17 = (var_12, var_13)
    var_18 = 10
    var_19 = 20
    var_20 = 'res'
    var_21 = 30
    var_22 = {var_20: var_21}
    var_23 = module_0.pmap(var_22)
    var_24 = 'val'
    var_25 = {var_24: var_0}



# Parsed testcases at query #22
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.freeze(var_0)
    assert var_1 == 1
    var_2 = 'string'
    var_3 = module_0.freeze(var_2)
    assert var_3 == 'string'
    var_4 = True
    var_5 = module_0.freeze(var_4)
    assert var_5 is True
    var_6 = None
    var_7 = module_0.freeze(var_6)
    assert var_7 is None
    var_8 = 2
    var_9 = [var_4, var_8]
    var_10 = module_0.freeze(var_9)
    var_11 = 3
    var_12 = [var_8, var_11]
    var_13 = [var_4, var_12]
    var_14 = module_0.freeze(var_13)
    var_15 = [var_8, var_11]
    var_16 = []
    var_17 = module_0.freeze(var_16)
    var_18 = []
    var_19 = 'a'
    var_20 = {var_19: var_4}
    var_21 = module_0.freeze(var_20)
    var_22 = 'b'
    var_23 = [var_4, var_8]
    var_24 = 'c'
    var_25 = {var_24: var_11}
    var_26 = {var_19: var_23, var_22: var_25}
    var_27 = module_0.freeze(var_26)
    var_28 = [var_4, var_8]
    var_29 = {var_24: var_11}
    var_30 = module_1.pmap(var_29)
    var_31 = {}
    var_32 = module_0.freeze(var_31)
    var_33 = {}
    var_34 = module_1.pmap(var_33)
    var_35 = [var_4]
    var_36 = [var_8]
    var_37 = {var_19: var_35, var_22: var_36}
    var_38 = [var_4]
    var_39 = [var_8]
    var_40 = {var_4, var_8}
    var_41 = module_0.freeze(var_40)
    var_42 = {var_4, var_8}
    var_43 = module_0.freeze(var_42)
    var_44 = [var_4, var_8]
    var_45 = module_2.pset(var_44)
    var_46 = [var_8]
    var_47 = (var_4, var_46)
    var_48 = module_0.freeze(var_47)
    var_49 = [var_8]
    var_50 = {var_19: var_11}
    var_51 = (var_4, var_49, var_50)
    var_52 = module_0.freeze(var_51)
    var_53 = [var_8]
    var_54 = {var_19: var_11}
    var_55 = module_1.pmap(var_54)
    var_56 = [var_8]
    var_57 = [var_4, var_56]
    var_58 = False
    var_59 = [var_4]
    var_60 = {var_19: var_59}
    var_61 = module_1.pmap(var_60)
    var_62 = module_0.freeze(var_61, var_58)

import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_1]
    var_3 = [var_0, var_2]
    var_4 = [var_1]
    var_5 = [var_0, var_4]
    var_6 = module_0.thaw(var_5)
    var_7 = 'a'
    var_8 = 'b'
    var_9 = [var_1]
    var_10 = {var_7: var_0, var_8: var_9}
    var_11 = module_1.pmap(var_10)
    var_12 = module_0.thaw(var_11)
    var_13 = [var_1]
    var_14 = {var_7: var_0, var_8: var_13}
    var_15 = module_0.thaw(var_14)
    var_16 = [var_0, var_1]
    var_17 = module_2.pset(var_16)
    var_18 = module_0.thaw(var_17)
    var_19 = [var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = [var_0, var_1]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = [var_0]
    var_10 = 'nested'
    var_11 = [var_1]
    var_12 = {var_10: var_11}
    var_13 = 3
    var_14 = [var_0, var_1, var_13]
    var_15 = [var_0, var_1, var_13]
    var_16 = 'c'
    var_17 = 'b'
    var_18 = {var_17: var_1}
    var_19 = [var_0, var_18]
    var_20 = 4
    var_21 = (var_13, var_20)
    var_22 = {var_3: var_19, var_16: var_21}
    var_23 = [var_1, var_13]
    var_24 = (var_0, var_23)



