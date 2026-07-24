####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.sign(var_0)
    assert var_1 == 1

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.sign(var_0)
    assert var_1 == 0

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = -1
    var_1 = module_0.sign(var_0)
    assert var_1 == -1

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 5.5
    var_1 = module_0.sign(var_0)
    assert var_1 == 1

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = -3.14
    var_1 = module_0.sign(var_0)
    assert var_1 == -1

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = -0.0
    var_1 = module_0.sign(var_0)
    assert var_1 == 0

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.sign(var_3)
    assert var_4 == 1

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '-1.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.sign(var_3)
    assert var_4 == -1

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.sign(var_3)
    assert var_4 == 0

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '-0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.sign(var_3)
    assert var_4 == 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_natural_number_valid_zero. Retrieved 2/3 statements.
# Partially parsed test_natural_number_valid_positive. Retrieved 2/3 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.NaturalNumber(*var_1, **var_2)
    assert var_3 == 0

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.NaturalNumber(*var_1, **var_2)
    assert var_3 == 10

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.NaturalNumber(*var_1, **var_2)

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.NaturalNumber(*var_1, **var_2)
    var_4 = type(var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_positive_integer_valid_input. Retrieved 2/3 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.PositiveInteger(*var_1, **var_2)
    assert var_3 == 1

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 10
    var_1 = 18
    var_2 = var_0 ** var_1
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.PositiveInteger(*var_3, **var_4)
    var_6 = bool(var_5 == 10 ** 18)
    assert var_6 is True

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.PositiveInteger(*var_1, **var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = -5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.PositiveInteger(*var_1, **var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------




import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '0'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '123.000'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '123'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.500'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '1.234500'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '1.2345'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '1.2E+2'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '120'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '1.23'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = [var_0]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = bool(var_4 == var_7)
    assert var_8 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '-5.00'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '-5'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '100.0000001'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = [var_0]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = bool(var_4 == var_7)
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 4/7 statements.
# Partially parsed test_weirdiv_dividend_negative_one_divisor_none. Retrieved 6/10 statements.
# Partially parsed test_weirdiv_divisor_zero_returns_max_float. Retrieved 5/8 statements.


import pypara.commons.numbers as module_0
import decimal as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.weirdiv(var_0, var_0)
    var_2 = '0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.Decimal(*var_3, **var_4)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)
    var_5 = module_1.weirdiv(var_0, var_4)
    var_6 = '0'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)
    var_5 = module_1.weirdiv(var_0, var_4)
    var_6 = '0'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = None
    var_5 = module_1.weirdiv(var_3, var_4)
    var_6 = '0'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = None
    var_5 = module_1.weirdiv(var_3, var_4)

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = None
    var_5 = module_1.weirdiv(var_3, var_4)
    var_6 = -1
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 3
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = module_1.weirdiv(var_3, var_7)
    var_9 = '3'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = bool(var_8 == var_12)
    assert var_13 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '10.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = '2'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = module_1.weirdiv(var_3, var_7)
    var_9 = '5.25'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = bool(var_8 == var_12)
    assert var_13 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 0
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = module_1.weirdiv(var_3, var_7)

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 5
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = module_1.weirdiv(var_3, var_7)
    var_9 = '0'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = bool(var_8 == var_12)
    assert var_13 is True



# Parsed testcases at query #6
#--------------------------




import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '1.2345'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '1.23'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_4(var_8)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.0001'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '1.2'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '1.2000'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_4(var_8)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '12.7'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '13'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_4(var_8)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_positive_integer_valid_value. Retrieved 2/3 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.PositiveInteger(*var_1, **var_2)
    assert var_3 == 5

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.PositiveInteger(*var_1, **var_2)
    assert var_3 == 1

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.PositiveInteger(*var_1, **var_2)
    var_4 = bool(True)
    assert var_4 is True
    var_5 = 'Expected AssertionError for value 0'
    var_6 = AssertionError(var_5)

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = -10
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.PositiveInteger(*var_1, **var_2)
    var_4 = bool(True)
    assert var_4 is True
    var_5 = 'Expected AssertionError for negative value'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #2
#--------------------------




import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '0'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '5.00'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '5'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '1.2300'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '1.23'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.100'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '0.1'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '-1.50'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '-1.5'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '123.45600'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '123.456'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------




import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '10.567'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '10.57'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_4(var_8)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '10.5'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '11'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_4(var_8)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.001'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '0.0000'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '0.000'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_4(var_8)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.0001'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '1.23456'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '1.2346'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_4(var_8)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 4/6 statements.
# Partially parsed test_weirdiv_dividend_negative_one_divisor_none. Retrieved 6/9 statements.
# Partially parsed test_weirdiv_divisor_zero_returns_max_float_positive. Retrieved 6/12 statements.
# Partially parsed test_weirdiv_divisor_zero_returns_max_float_negative. Retrieved 7/10 statements.


import pypara.commons.numbers as module_0
import decimal as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.weirdiv(var_0, var_0)
    var_2 = '0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.Decimal(*var_3, **var_4)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)
    var_5 = module_1.weirdiv(var_0, var_4)
    var_6 = '0'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)
    var_5 = module_1.weirdiv(var_0, var_4)
    var_6 = '0'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = None
    var_5 = module_1.weirdiv(var_3, var_4)
    var_6 = '0'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = None
    var_5 = module_1.weirdiv(var_3, var_4)

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = -1
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = None
    var_9 = module_1.weirdiv(var_7, var_8)

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 3
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = module_1.weirdiv(var_3, var_7)
    var_9 = '3'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = bool(var_8 == var_12)
    assert var_13 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '10.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = '2'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = module_1.weirdiv(var_3, var_7)
    var_9 = '5.25'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = bool(var_8 == var_12)
    assert var_13 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = 'float_info'
    var_1 = 5
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)
    var_5 = 0
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = module_1.weirdiv(var_4, var_8)

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = -5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 0
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = module_1.weirdiv(var_3, var_7)
    var_9 = -1
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_natural_number_new_valid_zero. Retrieved 2/3 statements.
# Partially parsed test_natural_number_new_valid_positive. Retrieved 2/3 statements.
# Partially parsed test_natural_number_new_type_is_int_subclass. Retrieved 2/3 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.NaturalNumber(*var_1, **var_2)
    assert var_3 == 0

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.NaturalNumber(*var_1, **var_2)
    assert var_3 == 10

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.NaturalNumber(*var_1, **var_2)

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.NaturalNumber(*var_1, **var_2)



