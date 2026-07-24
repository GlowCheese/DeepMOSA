####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_1.pset(var_5)
    var_7 = 'a'
    var_8 = 3
    var_9 = {var_7: var_8}
    var_10 = [var_0, var_9]
    var_11 = module_0.freeze(var_10)
    var_12 = {var_7: var_8}
    var_13 = module_2.pmap(var_12)
    var_14 = [var_0, var_13]
    var_15 = []
    var_16 = (var_0, var_15)
    var_17 = module_0.freeze(var_16)
    var_18 = []



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4



# Parsed testcases at query #3
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_10: var_0, var_11: var_1}
    var_15 = module_1.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = 'c'
    var_18 = {var_17: var_2}
    var_19 = {var_10: var_16, var_11: var_18}
    var_20 = module_0.freeze(var_19)
    var_21 = [var_0, var_1]
    var_22 = {var_17: var_2}
    var_23 = module_1.pmap(var_22)
    var_24 = {var_0, var_1, var_2}
    var_25 = module_0.freeze(var_24)
    var_26 = {var_0, var_1, var_2}
    var_27 = module_2.pset(var_26)
    var_28 = [var_1, var_2]
    var_29 = (var_0, var_28)
    var_30 = module_0.freeze(var_29)
    var_31 = [var_1, var_2]
    var_32 = 42
    var_33 = module_0.freeze(var_32)
    assert var_33 == 42
    var_34 = 'hello'
    var_35 = module_0.freeze(var_34)
    assert var_35 == 'hello'



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1
import pyrsistent._helpers as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = 4
    var_5 = 5
    var_6 = (var_4, var_5)
    var_7 = 6
    var_8 = 7
    var_9 = [var_7, var_8]
    var_10 = set(var_9)
    var_11 = [var_0, var_3, var_6, var_10]
    var_12 = {var_1: var_2}
    var_13 = module_0.pmap(var_12)
    var_14 = (var_4, var_5)
    var_15 = [var_7, var_8]
    var_16 = module_1.pset(var_15)
    var_17 = [var_0, var_13, var_14, var_16]
    var_18 = module_2.freeze(var_11)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = {}
    var_3 = 1
    var_4 = [var_3]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = 2
    var_10 = 'All tests passed for mutant'
    var_11 = print(var_10)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = {}
    var_3 = 1
    var_4 = [var_3]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = 2
    var_10 = 'All tests passed for mutant'
    var_11 = print(var_10)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = 'All tests passed.'
    var_6 = print(var_5)



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2, var_0]
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = 'new_key'
    var_9 = 'new_value'
    var_10 = {var_5: var_6, var_8: var_9}
    var_11 = module_0.pmap(var_10)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_0, var_1, var_2}
    var_8 = (var_0, var_1, var_2)
    var_9 = 'All test cases passed!'
    var_10 = print(var_9)



# Parsed testcases at query #11
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_0, var_7: var_1}
    var_9 = module_0.freeze(var_8)
    var_10 = {var_6: var_0, var_7: var_1}
    var_11 = module_1.pmap(var_10)
    var_12 = {var_0, var_1, var_2}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_0, var_1, var_2}
    var_15 = module_2.pset(var_14)
    var_16 = (var_0, var_1, var_2)
    var_17 = module_0.freeze(var_16)
    var_18 = {var_6: var_0}
    var_19 = {var_1, var_2}
    var_20 = [var_18, var_19]
    var_21 = module_0.freeze(var_20)
    var_22 = {var_6: var_0}
    var_23 = module_1.pmap(var_22)
    var_24 = {var_1, var_2}
    var_25 = module_2.pset(var_24)
    var_26 = [var_23, var_25]



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #14
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.freeze(var_2)
    var_4 = [var_0, var_1]
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = module_0.freeze(var_7)
    var_9 = {var_5: var_0, var_6: var_1}
    var_10 = module_1.pmap(var_9)
    var_11 = (var_0, var_1)
    var_12 = module_0.freeze(var_11)
    var_13 = [var_0, var_1]
    var_14 = set(var_13)
    var_15 = module_0.freeze(var_14)
    var_16 = [var_0, var_1]
    var_17 = module_2.pset(var_16)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = [var_0]



# Parsed testcases at query #17
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.freeze(var_2)
    var_4 = [var_0, var_1]
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = module_0.freeze(var_7)
    var_9 = {var_5: var_0, var_6: var_1}
    var_10 = module_1.pmap(var_9)
    var_11 = [var_0, var_1]
    var_12 = set(var_11)
    var_13 = module_0.freeze(var_12)
    var_14 = [var_0, var_1]
    var_15 = module_2.pset(var_14)
    var_16 = [var_1]
    var_17 = (var_0, var_16)
    var_18 = module_0.freeze(var_17)
    var_19 = [var_1]
    var_20 = {var_5: var_0}
    var_21 = {var_5: var_0}
    var_22 = module_1.pmap(var_21)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #19
#--------------------------


import pyrsistent._helpers as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = module_0.freeze(var_5)



# Parsed testcases at query #20
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_7, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = {var_0, var_1, var_2}
    var_14 = {var_0, var_1, var_2, var_4}
    var_15 = module_1.pset(var_14)
    var_16 = 'All tests passed.'
    var_17 = print(var_16)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4



# Parsed testcases at query #22
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_10: var_0, var_11: var_1}
    var_15 = module_1.pmap(var_14)
    var_16 = {var_11: var_0}
    var_17 = {var_10: var_16}
    var_18 = module_0.freeze(var_17)
    var_19 = {var_11: var_0}
    var_20 = module_1.pmap(var_19)
    var_21 = {var_10: var_20}
    var_22 = module_1.pmap(var_21)
    var_23 = {var_0, var_1, var_2}
    var_24 = module_0.freeze(var_23)
    var_25 = {var_0, var_1, var_2}
    var_26 = module_2.pset(var_25)
    var_27 = [var_1, var_2]
    var_28 = (var_0, var_27)
    var_29 = module_0.freeze(var_28)
    var_30 = [var_1, var_2]
    var_31 = 42
    var_32 = module_0.freeze(var_31)
    assert var_32 == 42



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_0, var_1, var_2}
    var_9 = (var_0, var_1, var_2)
    var_10 = 99
    var_11 = 'All tests passed!'
    var_12 = print(var_11)



# Parsed testcases at query #24
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'initial'
    var_6 = 'data'
    var_7 = {var_5: var_6}
    var_8 = 'Returned list should be frozen'
    var_9 = 'Returned dict should be frozen'
    var_10 = 100
    var_11 = [var_1, var_2, var_3, var_10]
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_5: var_6, var_12: var_13}
    var_15 = module_0.pmap(var_14)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 100
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'initial'
    var_6 = 'data'
    var_7 = {var_5: var_6}
    var_8 = 'Returned list should be frozen'
    var_9 = 'Returned dict should be frozen'
    var_10 = 100
    var_11 = [var_1, var_2, var_3, var_10]
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_5: var_6, var_12: var_13}
    var_15 = module_0.pmap(var_14)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 'result'
    var_2 = 'modified'
    var_3 = []
    var_4 = {}
    var_5 = 'result'
    var_6 = 'modified'

def test_case_0():
    var_0 = 1
    var_1 = 'result'
    var_2 = 'modified'
    var_3 = []
    var_4 = {}
    var_5 = 'result'
    var_6 = 'modified'



# Parsed testcases at query #26
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = set(var_2)
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1]
    var_6 = module_1.pset(var_5)
    var_7 = 'a'
    var_8 = 3
    var_9 = {var_7: var_8}
    var_10 = [var_0, var_9]
    var_11 = module_0.freeze(var_10)
    var_12 = {var_7: var_8}
    var_13 = module_2.pmap(var_12)
    var_14 = [var_0, var_13]
    var_15 = []
    var_16 = (var_0, var_15)
    var_17 = module_0.freeze(var_16)
    var_18 = []



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #28
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_10: var_0, var_11: var_1}
    var_15 = module_1.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = {var_10: var_16}
    var_18 = module_0.freeze(var_17)
    var_19 = [var_0, var_1]
    var_20 = {var_0, var_1, var_2}
    var_21 = module_0.freeze(var_20)
    var_22 = {var_0, var_1, var_2}
    var_23 = module_2.pset(var_22)
    var_24 = [var_1, var_2]
    var_25 = (var_0, var_24)
    var_26 = module_0.freeze(var_25)
    var_27 = [var_1, var_2]
    var_28 = 42
    var_29 = module_0.freeze(var_28)
    assert var_29 == 42
    var_30 = 'hello'
    var_31 = module_0.freeze(var_30)
    assert var_31 == 'hello'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Result should be a PVector'
    var_5 = len(var_3)
    assert var_5 == 3



# Parsed testcases at query #30
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_7]
    var_9 = []
    var_10 = (var_0, var_9)
    var_11 = module_0.freeze(var_10)
    var_12 = []
    var_13 = 2
    var_14 = [var_0, var_13]
    var_15 = set(var_14)
    var_16 = module_0.freeze(var_15)
    var_17 = [var_0, var_13]
    var_18 = module_2.pset(var_17)
    var_19 = 'b'
    var_20 = 'c'
    var_21 = {var_20: var_13}
    var_22 = {var_1: var_0, var_19: var_21}
    var_23 = module_0.freeze(var_22)
    var_24 = {var_20: var_13}
    var_25 = module_1.pmap(var_24)
    var_26 = {var_1: var_0, var_19: var_25}
    var_27 = module_1.pmap(var_26)



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = {}

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = {}



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_0, var_1, var_2}
    var_9 = 'All tests passed!'
    var_10 = print(var_9)



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 4
    var_1 = 'd'
    var_2 = {var_1: var_0}
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_3, var_8: var_4, var_9: var_5}
    var_11 = {var_3, var_4, var_5}

def test_case_0():
    var_0 = 4
    var_1 = 'd'
    var_2 = {var_1: var_0}
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = {var_7: var_3, var_8: var_4, var_9: var_5}
    var_11 = {var_3, var_4, var_5}



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'original'
    var_2 = {var_0: var_1}



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_0, var_1, var_2}



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = [var_2, var_3, var_5]



# Parsed testcases at query #39
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_7, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = {var_0, var_1, var_2}
    var_14 = {var_0, var_1, var_2, var_4}
    var_15 = module_1.pset(var_14)
    var_16 = [var_0, var_1, var_2]
    var_17 = [var_0, var_1, var_2]



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_0, var_1, var_2}
    var_8 = (var_0, var_1, var_2)
    var_9 = 'All tests passed.'
    var_10 = print(var_9)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_10: var_0, var_11: var_1}
    var_15 = module_1.pmap(var_14)
    var_16 = {var_11: var_1}
    var_17 = {var_10: var_16}
    var_18 = module_0.freeze(var_17)
    var_19 = {var_11: var_1}
    var_20 = module_1.pmap(var_19)
    var_21 = {var_10: var_20}
    var_22 = module_1.pmap(var_21)
    var_23 = {var_0, var_1, var_2}
    var_24 = module_0.freeze(var_23)
    var_25 = {var_0, var_1, var_2}
    var_26 = module_2.pset(var_25)
    var_27 = [var_1, var_2]
    var_28 = (var_0, var_27)
    var_29 = module_0.freeze(var_28)
    var_30 = [var_1, var_2]
    var_31 = 42
    var_32 = module_0.freeze(var_31)
    assert var_32 == 42



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pvector as module_1
import pyrsistent._helpers as module_2
import pyrsistent._pset as module_3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.m()
    var_4 = module_1.v()
    var_5 = (var_0, var_4)
    var_6 = module_2.thaw(var_5)
    var_7 = [var_1, var_2]
    var_8 = 4
    var_9 = 5
    var_10 = module_0.m()
    var_11 = False
    var_12 = 42
    var_13 = module_2.thaw(var_12)
    assert var_13 == 42
    var_14 = 'hello'
    var_15 = module_2.thaw(var_14)
    assert var_15 == 'hello'
    var_16 = module_1.v()
    var_17 = module_2.thaw(var_16)
    var_18 = module_0.m()
    var_19 = module_2.thaw(var_18)
    var_20 = module_3.s()
    var_21 = module_2.thaw(var_20)
    var_22 = set()



# Parsed testcases at query #3
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.m()
    var_4 = module_0.m()
    var_5 = module_1.thaw(var_4)
    var_6 = module_1.thaw(var_0)
    assert var_6 == 1
    var_7 = 'hello'
    var_8 = module_1.thaw(var_7)
    assert var_8 == 'hello'
    var_9 = False
    var_10 = 'a'



# Parsed testcases at query #4
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_7, var_9: var_10}
    var_12 = module_0.pmap(var_11)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._helpers as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_4: var_0, var_5: var_1}
    var_7 = module_0.pmap(var_6)
    var_8 = module_1.thaw(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_2.pset(var_9)
    var_11 = module_1.thaw(var_10)
    var_12 = [var_1, var_2]
    var_13 = {var_4: var_0}
    var_14 = module_0.pmap(var_13)
    var_15 = [var_1, var_2]
    var_16 = module_2.pset(var_15)
    var_17 = [var_14, var_16]



# Parsed testcases at query #6
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'The result should be a PVector'
    var_5 = 4
    var_6 = [var_0, var_1, var_2, var_5]
    var_7 = [var_0, var_1, var_2]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = {var_8: var_0, var_9: var_1}
    var_11 = module_0.pmap(var_10)
    var_12 = 'The result should be a PMap'
    var_13 = 'c'
    var_14 = {var_8: var_0, var_9: var_1, var_13: var_2}
    var_15 = module_0.pmap(var_14)
    var_16 = {var_8: var_0, var_9: var_1}
    var_17 = module_0.pmap(var_16)



# Parsed testcases at query #7
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 'y'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_3]
    var_6 = 'x'
    var_7 = 2
    var_8 = {var_2: var_3, var_6: var_7}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = [var_0]



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]



# Parsed testcases at query #10
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_0, var_1, var_3]
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = 'c'
    var_9 = {var_5: var_0, var_6: var_1, var_8: var_3}
    var_10 = module_0.pmap(var_9)
    var_11 = {var_0, var_1}
    var_12 = {var_0, var_1, var_3}
    var_13 = module_1.pset(var_12)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'b'
    var_1 = 'All mutant tests passed'
    var_2 = print(var_1)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #13
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_0}
    var_8 = 'b'
    var_9 = {var_6: var_0, var_8: var_1}
    var_10 = module_0.pmap(var_9)
    var_11 = {var_0, var_1}
    var_12 = {var_0, var_1, var_2}
    var_13 = module_1.pset(var_12)
    var_14 = 'All tests passed!'
    var_15 = print(var_14)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_0, var_1, var_2}



# Parsed testcases at query #15
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = module_0.freeze(var_4)
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)
    var_8 = [var_0, var_7]
    var_9 = []
    var_10 = (var_0, var_9)
    var_11 = module_0.freeze(var_10)
    var_12 = []
    var_13 = 2
    var_14 = [var_0, var_13]
    var_15 = set(var_14)
    var_16 = module_0.freeze(var_15)
    var_17 = [var_0, var_13]
    var_18 = module_2.pset(var_17)
    var_19 = 'b'
    var_20 = 4
    var_21 = 5
    var_22 = [var_20, var_21]
    var_23 = {var_19: var_22}
    var_24 = module_0.freeze(var_23)
    var_25 = [var_20, var_21]
    var_26 = module_0.freeze(var_0)
    assert var_26 == 1
    var_27 = 'test'
    var_28 = module_0.freeze(var_27)
    assert var_28 == 'test'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_0, var_1, var_2}
    var_9 = 'All tests passed!'
    var_10 = print(var_9)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = 'All tests passed!'
    var_6 = print(var_5)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = 'All tests passed!'
    var_6 = print(var_5)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_0, var_1, var_2}
    var_8 = [var_0, var_1]
    var_9 = 4
    var_10 = [var_2, var_9]
    var_11 = 'a'
    var_12 = {var_11: var_0}
    var_13 = 'b'
    var_14 = {var_13: var_1}



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = 'x'
    var_9 = 10
    var_10 = {var_8: var_9}
    var_11 = 20



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = 'All mutant tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #23
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_10: var_0, var_11: var_1}
    var_15 = module_1.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = 'c'
    var_18 = {var_17: var_2}
    var_19 = {var_10: var_16, var_11: var_18}
    var_20 = module_0.freeze(var_19)
    var_21 = [var_0, var_1]
    var_22 = {var_17: var_2}
    var_23 = module_1.pmap(var_22)
    var_24 = {var_0, var_1, var_2}
    var_25 = module_0.freeze(var_24)
    var_26 = {var_0, var_1, var_2}
    var_27 = module_2.pset(var_26)
    var_28 = [var_1, var_2]
    var_29 = (var_0, var_28)
    var_30 = module_0.freeze(var_29)
    var_31 = [var_1, var_2]
    var_32 = 42
    var_33 = module_0.freeze(var_32)
    assert var_33 == 42
    var_34 = 'hello'
    var_35 = module_0.freeze(var_34)
    assert var_35 == 'hello'



# Parsed testcases at query #24
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_7, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = {var_0, var_1, var_2}
    var_14 = {var_0, var_1, var_2, var_4}
    var_15 = module_1.pset(var_14)



# Parsed testcases at query #25
#--------------------------


import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_7, var_9: var_10}
    var_12 = module_0.pmap(var_11)
    var_13 = {var_0, var_1, var_2}
    var_14 = {var_0, var_1, var_2, var_4}
    var_15 = module_1.pset(var_14)



# Parsed testcases at query #26
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key1'
    var_5 = 'value1'
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = len(var_3)
    assert var_8 == 3
    var_9 = len(var_6)
    assert var_9 == 1



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_0, var_1, var_2}
    var_9 = 'All tests passed!'
    var_10 = print(var_9)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #31
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_6: var_7, var_9: var_10}
    var_12 = module_0.pmap(var_11)



# Parsed testcases at query #32
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 'b'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = 3
    var_7 = [var_0, var_1, var_6]
    var_8 = 'c'
    var_9 = {var_3: var_0, var_4: var_1, var_8: var_6}
    var_10 = module_0.pmap(var_9)



# Parsed testcases at query #33
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_10: var_0, var_11: var_1}
    var_15 = module_1.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = 'c'
    var_18 = {var_17: var_2}
    var_19 = {var_10: var_16, var_11: var_18}
    var_20 = module_0.freeze(var_19)
    var_21 = [var_0, var_1]
    var_22 = {var_17: var_2}
    var_23 = module_1.pmap(var_22)
    var_24 = {var_0, var_1, var_2}
    var_25 = module_0.freeze(var_24)
    var_26 = {var_0, var_1, var_2}
    var_27 = module_2.pset(var_26)
    var_28 = [var_1, var_2]
    var_29 = (var_0, var_28)
    var_30 = module_0.freeze(var_29)
    var_31 = [var_1, var_2]
    var_32 = 42
    var_33 = module_0.freeze(var_32)
    assert var_33 == 42
    var_34 = 'hello'
    var_35 = module_0.freeze(var_34)
    assert var_35 == 'hello'



# Parsed testcases at query #34
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 0
    var_8 = [var_0, var_1, var_2, var_0]
    var_9 = 'new_key'
    var_10 = 'new_value'
    var_11 = {var_4: var_5, var_9: var_10}
    var_12 = module_0.pmap(var_11)



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key'
    var_5 = 'old'
    var_6 = {var_4: var_5}
    var_7 = {var_0, var_1, var_2}
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'Modified list should be a PVector'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_0}



# Parsed testcases at query #38
#--------------------------


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = 0
    var_7 = [var_0, var_1, var_2, var_0]
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_4: var_0, var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = [var_0, var_1]
    var_13 = [var_0, var_1]
    var_14 = {var_4: var_0}
    var_15 = {var_4: var_0}
    var_16 = module_0.pmap(var_15)



# Parsed testcases at query #39
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pset as module_1
import pyrsistent._pmap as module_2

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.freeze(var_4)
    var_6 = [var_1, var_2]
    var_7 = [var_1, var_2]
    var_8 = set(var_7)
    var_9 = module_0.freeze(var_8)
    var_10 = [var_1, var_2]
    var_11 = module_1.pset(var_10)
    var_12 = 3
    var_13 = {var_0: var_12}
    var_14 = [var_1, var_13]
    var_15 = module_0.freeze(var_14)
    var_16 = {var_0: var_12}
    var_17 = module_2.pmap(var_16)
    var_18 = [var_1, var_17]
    var_19 = []
    var_20 = (var_1, var_19)
    var_21 = module_0.freeze(var_20)
    var_22 = []



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'b'
    var_8 = {var_0, var_1, var_2}
    var_9 = 'All mutant tests passed!'
    var_10 = print(var_9)



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 'a'
    var_6 = 'b'
    var_7 = {var_5: var_0, var_6: var_1}
    var_8 = 'c'



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_1, var_2, var_4]



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = 'All tests passed.'
    var_6 = print(var_5)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = 'All tests passed.'
    var_6 = print(var_5)



# Parsed testcases at query #44
#--------------------------


import pyrsistent._helpers as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.freeze(var_3)
    var_5 = [var_0, var_1, var_2]
    var_6 = [var_1, var_2]
    var_7 = [var_0, var_6]
    var_8 = module_0.freeze(var_7)
    var_9 = [var_1, var_2]
    var_10 = 'a'
    var_11 = 'b'
    var_12 = {var_10: var_0, var_11: var_1}
    var_13 = module_0.freeze(var_12)
    var_14 = {var_10: var_0, var_11: var_1}
    var_15 = module_1.pmap(var_14)
    var_16 = [var_0, var_1]
    var_17 = {var_10: var_16}
    var_18 = module_0.freeze(var_17)
    var_19 = [var_0, var_1]
    var_20 = {var_0, var_1, var_2}
    var_21 = module_0.freeze(var_20)
    var_22 = {var_0, var_1, var_2}
    var_23 = module_2.pset(var_22)
    var_24 = [var_1, var_2]
    var_25 = (var_0, var_24)
    var_26 = module_0.freeze(var_25)
    var_27 = [var_1, var_2]
    var_28 = 42
    var_29 = module_0.freeze(var_28)
    assert var_29 == 42



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4



