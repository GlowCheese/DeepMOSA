####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'c'
    var_4 = 3
    var_5 = 4
    var_6 = 'All tests for discard passed successfully!'
    var_7 = print(var_6)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'c'
    var_4 = 3
    var_5 = 5
    var_6 = 4



# Parsed testcases at query #4
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_1: var_2}
    var_6 = {var_0: var_5}
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = 'a1'
    var_9 = 'a2'
    var_10 = 3
    var_11 = {var_8: var_2, var_9: var_3, var_1: var_10}
    var_12 = '^a'
    var_13 = module_0.rex(var_12)
    var_14 = {var_0: var_2, var_1: var_3}
    var_15 = 'All transform tests passed.'
    var_16 = print(var_15)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 4



# Parsed testcases at query #6
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = '123abc'
    var_5 = ''



# Parsed testcases at query #7
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = '123'



# Parsed testcases at query #8
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5



# Parsed testcases at query #9
#--------------------------


import pyrsistent._transformations as module_0
import re as module_1

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_1.match(var_2)
    var_4 = module_0.rex(var_0)
    var_5 = 'bac'
    var_6 = module_1.match(var_5)
    var_7 = module_0.rex(var_0)
    var_8 = 'a'
    var_9 = module_1.match(var_8)
    var_10 = module_0.rex(var_0)
    var_11 = ''
    var_12 = module_1.match(var_11)



# Parsed testcases at query #10
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #11
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #12
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'def'
    var_4 = 123



# Parsed testcases at query #13
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.rex(var_0)
    var_2 = 'b'
    var_3 = module_0.rex(var_0)
    var_4 = module_0.rex(var_0)
    var_5 = 1
    var_6 = module_0.rex(var_0)
    var_7 = 'ab'
    var_8 = 'a.*'
    var_9 = module_0.rex(var_8)
    var_10 = module_0.rex(var_0)
    var_11 = module_0.rex(var_0)



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_0.rex(var_0)
    var_4 = 'bac'
    var_5 = module_0.rex(var_0)
    var_6 = 1



# Parsed testcases at query #15
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5



# Parsed testcases at query #16
#--------------------------


import pyrsistent._transformations as module_0
import re as module_1

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_1.match(var_2)
    var_4 = module_0.rex(var_0)
    var_5 = 'bac'
    var_6 = module_1.match(var_5)
    assert var_6 is None
    var_7 = module_0.rex(var_0)
    var_8 = 123
    var_9 = module_1.match(var_8)
    assert var_9 is None



# Parsed testcases at query #17
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d{3}$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = module_0.rex(var_0)
    var_4 = '1234'
    var_5 = module_0.rex(var_0)
    var_6 = 123



# Parsed testcases at query #18
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'



# Parsed testcases at query #19
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #20
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^[a-z]*$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_0.rex(var_0)
    var_4 = '123'
    var_5 = module_0.rex(var_0)
    var_6 = 123



# Parsed testcases at query #21
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = '^\\d+$'
    var_8 = module_0.rex(var_7)
    var_9 = '123'
    var_10 = module_0.rex(var_7)
    var_11 = 'a123'



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0
import re as module_1

def test_case_0():
    var_0 = ''
    var_1 = module_0.rex(var_0)
    var_2 = None
    var_3 = module_0.rex(var_0)
    var_4 = 'foo'
    var_5 = module_0.rex(var_4)
    var_6 = 'bar'
    var_7 = module_0.rex(var_4)
    var_8 = module_0.rex(var_4)
    var_9 = 'foobar'
    var_10 = module_0.rex(var_4)
    var_11 = 'barfoo'
    var_12 = '^foo$'
    var_13 = module_0.rex(var_12)
    var_14 = module_0.rex(var_12)
    var_15 = module_0.rex(var_12)
    var_16 = 'föö'
    var_17 = module_0.rex(var_16)
    var_18 = module_0.rex(var_16)
    var_19 = module_1.compile(var_4)
    var_20 = module_0.rex(var_19)



# Parsed testcases at query #23
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.rex(var_0)
    var_2 = 'b'
    var_3 = 1



# Parsed testcases at query #24
#--------------------------


import pyrsistent._transformations as module_0
import re as module_1

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = module_1.match(var_2)
    var_4 = module_0.rex(var_0)
    var_5 = 'banana'
    var_6 = module_1.match(var_5)
    var_7 = '\\d+'
    var_8 = module_0.rex(var_7)
    var_9 = '123'
    var_10 = module_1.match(var_9)
    var_11 = module_0.rex(var_7)
    var_12 = 'abc'
    var_13 = module_1.match(var_12)



# Parsed testcases at query #25
#--------------------------


import pyrsistent._transformations as module_0
import re as module_1

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = module_1.match(var_2)
    var_4 = module_0.rex(var_0)
    var_5 = 'b'
    var_6 = module_1.match(var_5)



# Parsed testcases at query #26
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_0.rex(var_0)
    var_4 = 'bac'
    var_5 = module_0.rex(var_0)
    var_6 = 'a'
    var_7 = module_0.rex(var_0)
    var_8 = 'b'



# Parsed testcases at query #27
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = module_0.rex(var_0)
    var_4 = 'banana'
    var_5 = '\\d'
    var_6 = module_0.rex(var_5)
    var_7 = '5'
    var_8 = module_0.rex(var_5)
    var_9 = 'a'
    var_10 = '\\d+'
    var_11 = module_0.rex(var_10)
    var_12 = '123'
    var_13 = module_0.rex(var_10)
    var_14 = 'abc'



# Parsed testcases at query #28
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 'a'
    var_9 = module_0.rex(var_0)
    var_10 = 'b'
    var_11 = module_0.rex(var_0)
    var_12 = 'apple'
    var_13 = module_0.rex(var_0)
    var_14 = 'banana'
    var_15 = module_0.rex(var_0)
    var_16 = 'application'
    var_17 = module_0.rex(var_0)
    var_18 = 'basket'
    var_19 = module_0.rex(var_0)
    var_20 = 'aardvark'
    var_21 = module_0.rex(var_0)
    var_22 = 'zebra'



# Parsed testcases at query #29
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5



# Parsed testcases at query #30
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #31
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d{4}$'
    var_1 = module_0.rex(var_0)
    var_2 = '1234'
    var_3 = module_0.rex(var_0)
    var_4 = '123'
    var_5 = module_0.rex(var_0)
    var_6 = 123



# Parsed testcases at query #32
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = module_0.rex(var_0)
    var_4 = 'banana'
    var_5 = '\\d+'
    var_6 = module_0.rex(var_5)
    var_7 = '123'
    var_8 = module_0.rex(var_5)
    var_9 = 'abc'



# Parsed testcases at query #33
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = module_0.rex(var_0)
    var_4 = 'b'
    var_5 = module_0.rex(var_0)
    var_6 = 'ab'
    var_7 = module_0.rex(var_0)
    var_8 = 'ba'
    var_9 = module_0.rex(var_0)
    var_10 = module_0.rex(var_0)
    var_11 = module_0.rex(var_0)
    var_12 = module_0.rex(var_0)



# Parsed testcases at query #34
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = module_0.rex(var_0)
    var_9 = module_0.rex(var_0)
    var_10 = 5



# Parsed testcases at query #35
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = 'world'
    var_4 = None
    var_5 = 123



# Parsed testcases at query #36
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #37
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 'a'
    var_9 = module_0.rex(var_0)
    var_10 = 'b'
    var_11 = module_0.rex(var_0)
    var_12 = 1



# Parsed testcases at query #38
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^foo'
    var_1 = module_0.rex(var_0)
    var_2 = 'foo'
    var_3 = module_0.rex(var_0)
    var_4 = 'bar'
    var_5 = module_0.rex(var_0)
    var_6 = 'foobar'
    var_7 = module_0.rex(var_0)
    var_8 = 'barfoo'



# Parsed testcases at query #39
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'def'



# Parsed testcases at query #40
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'c'
    var_4 = 3
    var_5 = 5
    var_6 = 'b'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'c'
    var_4 = 3
    var_5 = 5
    var_6 = 4



# Parsed testcases at query #3
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 123



# Parsed testcases at query #4
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = 'def'
    var_4 = 123



# Parsed testcases at query #5
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_8 = module_0.discard(var_7, var_0)
    var_9 = 'd'
    var_10 = module_0.discard(var_7, var_9)



# Parsed testcases at query #6
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5



# Parsed testcases at query #7
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #8
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = module_0.rex(var_0)
    var_4 = 'banana'
    var_5 = module_0.rex(var_0)
    var_6 = 123



# Parsed testcases at query #9
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5



# Parsed testcases at query #10
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_0.rex(var_0)
    var_4 = 'bac'
    var_5 = module_0.rex(var_0)
    var_6 = 5



# Parsed testcases at query #11
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123
    var_5 = 'avocado'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #13
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_0.rex(var_0)
    var_4 = 'xyz'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._transformations as module_0
import re as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.rex(var_0)
    var_2 = module_0.rex(var_0)
    var_3 = 'b'
    var_4 = module_0.rex(var_0)
    var_5 = 'ab'
    var_6 = module_0.rex(var_0)
    var_7 = 'ba'
    var_8 = module_0.rex(var_0)
    var_9 = module_1.match(var_0)
    var_10 = module_0.rex(var_0)
    var_11 = module_1.match(var_3)
    assert var_11 is None
    var_12 = module_0.rex(var_0)
    var_13 = module_1.match(var_5)
    var_14 = module_0.rex(var_0)
    var_15 = module_1.match(var_7)
    assert var_15 is None



# Parsed testcases at query #15
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = '^[a-z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = module_0.rex(var_5)



# Parsed testcases at query #16
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_0.rex(var_0)
    var_4 = 'bac'
    var_5 = module_0.rex(var_0)
    var_6 = 5



# Parsed testcases at query #17
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #18
#--------------------------


import re as module_0
import pyrsistent._transformations as module_1

def test_case_0():
    var_0 = '^([A-Z][0-9]+)+$'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.rex(var_0)
    var_3 = 'A1'
    var_4 = module_0.match(var_3)
    var_5 = module_1.rex(var_0)
    var_6 = 'a1'
    var_7 = module_1.rex(var_0)
    var_8 = 'A1B2'
    var_9 = module_0.match(var_8)
    var_10 = module_1.rex(var_0)
    var_11 = 'A1b2'
    var_12 = module_1.rex(var_0)
    var_13 = 1



# Parsed testcases at query #19
#--------------------------


import pyrsistent._transformations as module_0
import re as module_1

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = module_1.match(var_2)
    var_4 = module_0.rex(var_0)
    var_5 = 'banana'
    var_6 = module_1.match(var_5)



# Parsed testcases at query #20
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 'a'
    var_9 = module_0.rex(var_0)
    var_10 = 'b'
    var_11 = module_0.rex(var_0)
    var_12 = ''
    var_13 = module_0.rex(var_0)
    var_14 = 1



# Parsed testcases at query #21
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = ' Test regex matcher '
    var_1 = '^a'
    var_2 = module_0.rex(var_1)
    var_3 = 'apple'
    var_4 = 'banana'
    var_5 = '^[0-9]'
    var_6 = module_0.rex(var_5)
    var_7 = '123'
    var_8 = 'abc'
    var_9 = 123



# Parsed testcases at query #22
#--------------------------


import pyrsistent._transformations as module_0
import re as module_1

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = module_1.match(var_2)
    var_4 = module_0.rex(var_0)
    var_5 = 'b'
    var_6 = module_1.match(var_5)



# Parsed testcases at query #23
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5



# Parsed testcases at query #24
#--------------------------


import pyrsistent._transformations as module_0
import re as module_1

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = module_1.match(var_2)
    var_4 = module_0.rex(var_0)
    var_5 = 'b'
    var_6 = module_1.match(var_5)
    assert var_6 is None
    var_7 = module_0.rex(var_0)
    var_8 = 1
    var_9 = module_1.match(var_8)
    assert var_9 is None



# Parsed testcases at query #25
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^\\\\d+$'
    var_1 = module_0.rex(var_0)
    var_2 = '123'
    var_3 = 'abc'
    var_4 = '123abc'



# Parsed testcases at query #26
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #27
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #28
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_0.rex(var_0)
    var_4 = 'bac'
    var_5 = '^a$'
    var_6 = module_0.rex(var_5)
    var_7 = 'a'
    var_8 = module_0.rex(var_5)
    var_9 = 'ab'
    var_10 = module_0.rex(var_5)
    var_11 = 'ba'
    var_12 = module_0.rex(var_5)
    var_13 = ''



# Parsed testcases at query #29
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5



# Parsed testcases at query #30
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = '^a$'
    var_8 = module_0.rex(var_7)
    var_9 = 'a'
    var_10 = module_0.rex(var_7)
    var_11 = 'ab'



# Parsed testcases at query #31
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5



# Parsed testcases at query #32
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5



# Parsed testcases at query #33
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = module_0.rex(var_0)
    var_4 = 'banana'



# Parsed testcases at query #34
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_0.rex(var_0)
    var_4 = 'bc'
    var_5 = module_0.rex(var_0)
    var_6 = 123



# Parsed testcases at query #35
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.rex(var_0)
    var_2 = module_0.rex(var_0)
    var_3 = 'b'
    var_4 = module_0.rex(var_0)
    var_5 = 'aa'
    var_6 = '^a$'
    var_7 = module_0.rex(var_6)
    var_8 = module_0.rex(var_0)
    var_9 = 1



# Parsed testcases at query #36
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^Hello$'
    var_1 = module_0.rex(var_0)
    var_2 = 'Hello'
    var_3 = module_0.rex(var_0)
    var_4 = 'NotHello'
    var_5 = '^[a-z]+$'
    var_6 = module_0.rex(var_5)
    var_7 = 'abc'
    var_8 = module_0.rex(var_5)
    var_9 = '123'
    var_10 = '^\\d+$'
    var_11 = module_0.rex(var_10)
    var_12 = module_0.rex(var_10)



# Parsed testcases at query #37
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = '^b'
    var_4 = module_0.rex(var_3)
    var_5 = module_0.rex(var_0)
    var_6 = 123



# Parsed testcases at query #38
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = module_0.rex(var_0)
    var_8 = 5



# Parsed testcases at query #39
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_0.rex(var_0)
    var_4 = 'bcd'
    var_5 = module_0.rex(var_0)
    var_6 = 1



# Parsed testcases at query #40
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_0.rex(var_0)
    var_4 = 'bac'
    var_5 = module_0.rex(var_0)
    var_6 = ''
    var_7 = module_0.rex(var_0)
    var_8 = 'a'
    var_9 = module_0.rex(var_0)
    var_10 = 100
    var_11 = var_8 * var_10
    var_12 = module_0.rex(var_0)
    var_13 = 'b'
    var_14 = var_13 * var_10
    var_15 = module_0.rex(var_0)
    var_16 = var_8 * var_10
    var_17 = var_16 + var_13
    var_18 = module_0.rex(var_0)
    var_19 = var_13 * var_10
    var_20 = var_19 + var_8
    var_21 = module_0.rex(var_0)
    var_22 = var_8 * var_10
    var_23 = var_13 * var_10
    var_24 = var_22 + var_23
    var_25 = module_0.rex(var_0)
    var_26 = var_13 * var_10
    var_27 = var_8 * var_10
    var_28 = var_26 + var_27
    var_29 = module_0.rex(var_0)
    var_30 = var_8 * var_10
    var_31 = var_13 * var_10
    var_32 = var_30 + var_31
    var_33 = var_32 + var_8
    var_34 = module_0.rex(var_0)
    var_35 = var_13 * var_10
    var_36 = var_8 * var_10
    var_37 = var_35 + var_36
    var_38 = var_37 + var_13
    var_39 = module_0.rex(var_0)
    var_40 = var_8 * var_10
    var_41 = var_13 * var_10
    var_42 = var_40 + var_41
    var_43 = var_8 * var_10
    var_44 = var_42 + var_43
    var_45 = module_0.rex(var_0)
    var_46 = var_13 * var_10
    var_47 = var_8 * var_10
    var_48 = var_46 + var_47
    var_49 = var_13 * var_10
    var_50 = var_48 + var_49
    var_51 = module_0.rex(var_0)
    var_52 = var_8 * var_10
    var_53 = var_13 * var_10
    var_54 = var_52 + var_53
    var_55 = var_8 * var_10
    var_56 = var_54 + var_55
    var_57 = var_56 + var_13
    var_58 = module_0.rex(var_0)
    var_59 = var_13 * var_10
    var_60 = var_8 * var_10
    var_61 = var_59 + var_60
    var_62 = var_13 * var_10
    var_63 = var_61 + var_62
    var_64 = var_63 + var_8
    var_65 = module_0.rex(var_0)
    var_66 = var_8 * var_10
    var_67 = var_13 * var_10
    var_68 = var_66 + var_67
    var_69 = var_8 * var_10
    var_70 = var_68 + var_69
    var_71 = var_13 * var_10
    var_72 = var_70 + var_71
    var_73 = module_0.rex(var_0)
    var_74 = var_13 * var_10
    var_75 = var_8 * var_10
    var_76 = var_74 + var_75
    var_77 = var_13 * var_10
    var_78 = var_76 + var_77
    var_79 = var_8 * var_10
    var_80 = var_78 + var_79
    var_81 = module_0.rex(var_0)
    var_82 = var_8 * var_10
    var_83 = var_13 * var_10
    var_84 = var_82 + var_83
    var_85 = var_8 * var_10
    var_86 = var_84 + var_85
    var_87 = var_13 * var_10
    var_88 = var_86 + var_87
    var_89 = var_88 + var_8
    var_90 = module_0.rex(var_0)
    var_91 = var_13 * var_10
    var_92 = var_8 * var_10
    var_93 = var_91 + var_92
    var_94 = var_13 * var_10
    var_95 = var_93 + var_94
    var_96 = var_8 * var_10
    var_97 = var_95 + var_96
    var_98 = var_97 + var_13
    var_99 = module_0.rex(var_0)
    var_100 = var_8 * var_10
    var_101 = var_13 * var_10
    var_102 = var_100 + var_101
    var_103 = var_8 * var_10
    var_104 = var_102 + var_103
    var_105 = var_13 * var_10
    var_106 = var_104 + var_105
    var_107 = var_8 * var_10
    var_108 = var_106 + var_107
    var_109 = module_0.rex(var_0)
    var_110 = var_13 * var_10
    var_111 = var_8 * var_10
    var_112 = var_110 + var_111
    var_113 = var_13 * var_10
    var_114 = var_112 + var_113
    var_115 = var_8 * var_10
    var_116 = var_114 + var_115
    var_117 = var_13 * var_10
    var_118 = var_116 + var_117
    var_119 = module_0.rex(var_0)
    var_120 = var_8 * var_10
    var_121 = var_13 * var_10
    var_122 = var_120 + var_121
    var_123 = var_8 * var_10
    var_124 = var_122 + var_123
    var_125 = var_13 * var_10
    var_126 = var_124 + var_125
    var_127 = var_8 * var_10
    var_128 = var_126 + var_127
    var_129 = var_128 + var_13
    var_130 = module_0.rex(var_0)
    var_131 = var_13 * var_10
    var_132 = var_8 * var_10
    var_133 = var_131 + var_132
    var_134 = var_13 * var_10
    var_135 = var_133 + var_134
    var_136 = var_8 * var_10
    var_137 = var_135 + var_136
    var_138 = var_13 * var_10
    var_139 = var_137 + var_138
    var_140 = var_139 + var_8
    var_141 = module_0.rex(var_0)
    var_142 = var_8 * var_10
    var_143 = var_13 * var_10
    var_144 = var_142 + var_143
    var_145 = var_8 * var_10
    var_146 = var_144 + var_145
    var_147 = var_13 * var_10
    var_148 = var_146 + var_147
    var_149 = var_8 * var_10
    var_150 = var_148 + var_149
    var_151 = var_13 * var_10
    var_152 = var_150 + var_151
    var_153 = module_0.rex(var_0)
    var_154 = var_13 * var_10
    var_155 = var_8 * var_10
    var_156 = var_154 + var_155
    var_157 = var_13 * var_10
    var_158 = var_156 + var_157
    var_159 = var_8 * var_10
    var_160 = var_158 + var_159
    var_161 = var_13 * var_10
    var_162 = var_160 + var_161
    var_163 = var_8 * var_10
    var_164 = var_162 + var_163
    var_165 = module_0.rex(var_0)
    var_166 = var_8 * var_10
    var_167 = var_13 * var_10
    var_168 = var_166 + var_167
    var_169 = var_8 * var_10
    var_170 = var_168 + var_169
    var_171 = var_13 * var_10
    var_172 = var_170 + var_171
    var_173 = var_8 * var_10
    var_174 = var_172 + var_173
    var_175 = var_13 * var_10
    var_176 = var_174 + var_175
    var_177 = var_176 + var_8
    var_178 = module_0.rex(var_0)
    var_179 = var_13 * var_10
    var_180 = var_8 * var_10
    var_181 = var_179 + var_180
    var_182 = var_13 * var_10
    var_183 = var_181 + var_182
    var_184 = var_8 * var_10
    var_185 = var_183 + var_184
    var_186 = var_13 * var_10
    var_187 = var_185 + var_186
    var_188 = var_8 * var_10
    var_189 = var_187 + var_188
    var_190 = var_189 + var_13
    var_191 = module_0.rex(var_0)
    var_192 = var_8 * var_10
    var_193 = var_13 * var_10
    var_194 = var_192 + var_193
    var_195 = var_8 * var_10
    var_196 = var_194 + var_195
    var_197 = var_13 * var_10
    var_198 = var_196 + var_197
    var_199 = var_8 * var_10
    var_200 = var_198 + var_199
    var_201 = var_13 * var_10
    var_202 = var_200 + var_201
    var_203 = var_8 * var_10
    var_204 = var_202 + var_203
    var_205 = module_0.rex(var_0)
    var_206 = var_13 * var_10
    var_207 = var_8 * var_10
    var_208 = var_206 + var_207
    var_209 = var_13 * var_10
    var_210 = var_208 + var_209
    var_211 = var_8 * var_10
    var_212 = var_210 + var_211
    var_213 = var_13 * var_10
    var_214 = var_212 + var_213
    var_215 = var_8 * var_10
    var_216 = var_214 + var_215
    var_217 = var_13 * var_10
    var_218 = var_216 + var_217



# Parsed testcases at query #41
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = var_1.__name__
    assert var_2 == '<lambda>'
    var_3 = module_0.rex(var_0)
    var_4 = 'abc'
    var_5 = module_0.rex(var_0)
    var_6 = 'bac'
    var_7 = '^[ab]'
    var_8 = module_0.rex(var_7)
    var_9 = 'a'
    var_10 = module_0.rex(var_7)
    var_11 = 'b'
    var_12 = module_0.rex(var_7)
    var_13 = 'c'



# Parsed testcases at query #42
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a$'
    var_1 = module_0.rex(var_0)
    var_2 = 'a'
    var_3 = module_0.rex(var_0)
    var_4 = 'b'
    var_5 = '^[a-z]$'
    var_6 = module_0.rex(var_5)
    var_7 = 'c'
    var_8 = module_0.rex(var_5)
    var_9 = '1'



# Parsed testcases at query #43
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.rex(var_0)
    var_2 = 'apple'
    var_3 = 'banana'
    var_4 = 123



# Parsed testcases at query #44
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^h'
    var_1 = module_0.rex(var_0)
    var_2 = 'hello'
    var_3 = module_0.rex(var_0)
    var_4 = 'world'
    var_5 = '^\\d+'
    var_6 = module_0.rex(var_5)
    var_7 = '123'
    var_8 = module_0.rex(var_5)
    var_9 = 'abc'



# Parsed testcases at query #45
#--------------------------


import pyrsistent._transformations as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc'
    var_3 = module_0.rex(var_0)
    var_4 = 'bac'
    var_5 = module_0.rex(var_0)
    var_6 = 'a'
    var_7 = module_0.rex(var_0)
    var_8 = 'b'
    var_9 = module_0.rex(var_0)
    var_10 = ''
    var_11 = module_0.rex(var_0)
    var_12 = 'a '
    var_13 = module_0.rex(var_0)
    var_14 = ' a'
    var_15 = module_0.rex(var_0)
    var_16 = 'a\n'
    var_17 = module_0.rex(var_0)
    var_18 = '\na'
    var_19 = module_0.rex(var_0)
    var_20 = 'a\t'
    var_21 = module_0.rex(var_0)
    var_22 = '\ta'
    var_23 = module_0.rex(var_0)
    var_24 = 'a\r'
    var_25 = module_0.rex(var_0)
    var_26 = '\ra'
    var_27 = module_0.rex(var_0)
    var_28 = 'a\x0c'
    var_29 = module_0.rex(var_0)
    var_30 = '\x0ca'
    var_31 = module_0.rex(var_0)
    var_32 = 'a\x0b'
    var_33 = module_0.rex(var_0)
    var_34 = '\x0ba'
    var_35 = '\\d+'
    var_36 = module_0.rex(var_35)
    var_37 = '123'
    var_38 = module_0.rex(var_35)
    var_39 = module_0.rex(var_35)
    var_40 = '123abc'
    var_41 = module_0.rex(var_35)
    var_42 = 'abc123'
    var_43 = module_0.rex(var_35)
    var_44 = '123abc123'
    var_45 = module_0.rex(var_35)
    var_46 = 'abc123abc'
    var_47 = module_0.rex(var_35)
    var_48 = '123abc123abc'
    var_49 = module_0.rex(var_35)
    var_50 = 'abc123abc123'
    var_51 = module_0.rex(var_35)
    var_52 = '123abc123abc123'
    var_53 = module_0.rex(var_35)
    var_54 = 'abc123abc123abc'
    var_55 = module_0.rex(var_35)
    var_56 = '123abc123abc123abc'
    var_57 = module_0.rex(var_35)
    var_58 = 'abc123abc123abc123'
    var_59 = module_0.rex(var_35)
    var_60 = '123abc123abc123abc123'
    var_61 = module_0.rex(var_35)
    var_62 = 'abc123abc123abc123abc'
    var_63 = module_0.rex(var_35)
    var_64 = '123abc123abc123abc123abc'
    var_65 = module_0.rex(var_35)
    var_66 = 'abc123abc123abc123abc123'
    var_67 = module_0.rex(var_35)
    var_68 = '123abc123abc123abc123abc123'
    var_69 = module_0.rex(var_35)
    var_70 = 'abc123abc123abc123abc123abc'
    var_71 = module_0.rex(var_35)
    var_72 = '123abc123abc123abc123abc123abc'
    var_73 = module_0.rex(var_35)
    var_74 = 'abc123abc123abc123abc123abc123'
    var_75 = module_0.rex(var_35)
    var_76 = '123abc123abc123abc123abc123abc123'
    var_77 = module_0.rex(var_35)
    var_78 = 'abc123abc123abc123abc123abc123abc'
    var_79 = module_0.rex(var_35)
    var_80 = '123abc123abc123abc123abc123abc123abc'
    var_81 = module_0.rex(var_35)
    var_82 = 'abc123abc123abc123abc123abc123abc123'
    var_83 = module_0.rex(var_35)
    var_84 = '123abc123abc123abc123abc123abc123abc123'
    var_85 = module_0.rex(var_35)
    var_86 = 'abc123abc123abc123abc123abc123abc123abc'
    var_87 = module_0.rex(var_35)
    var_88 = '123abc123abc123abc123abc123abc123abc123abc'
    var_89 = module_0.rex(var_35)
    var_90 = 'abc123abc123abc123abc123abc123abc123abc123'
    var_91 = module_0.rex(var_35)
    var_92 = '123abc123abc123abc123abc123abc123abc123abc123'
    var_93 = module_0.rex(var_35)
    var_94 = 'abc123abc123abc123abc123abc123abc123abc123abc'
    var_95 = module_0.rex(var_35)
    var_96 = '123abc123abc123abc123abc123abc123abc123abc123abc'
    var_97 = module_0.rex(var_35)
    var_98 = 'abc123abc123abc123abc123abc123abc123abc123abc123'
    var_99 = module_0.rex(var_35)
    var_100 = '123abc123abc123abc123abc123abc123abc123abc123abc123'
    var_101 = module_0.rex(var_35)
    var_102 = 'abc123abc123abc123abc123abc123abc123abc123abc123abc'
    var_103 = module_0.rex(var_35)
    var_104 = '123abc123abc123abc123abc123abc123abc123abc123abc123abc'
    var_105 = module_0.rex(var_35)
    var_106 = 'abc123abc123abc123abc123abc123abc123abc123abc123abc123'
    var_107 = module_0.rex(var_35)
    var_108 = '123abc123abc123abc123abc123abc123abc123abc123abc123abc123'
    var_109 = module_0.rex(var_35)
    var_110 = 'abc123abc123abc123abc123abc123abc123abc123abc123abc123abc'
    var_111 = module_0.rex(var_35)
    var_112 = '123abc123abc123abc123abc123abc123abc123abc123abc123abc123abc'
    var_113 = module_0.rex(var_35)
    var_114 = 'abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123'
    var_115 = module_0.rex(var_35)
    var_116 = '123abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123'
    var_117 = module_0.rex(var_35)
    var_118 = 'abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abc'



