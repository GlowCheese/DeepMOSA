####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_0 = '10.00'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '10'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '1.2345'
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
    var_0 = '1.200'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = '1.2'
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
    var_0 = '0.0001'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.normalize(var_3)
    var_5 = [var_0]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = bool(var_4 == var_7)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_sign_negative_zero_decimal. Retrieved 2/4 statements.


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
    var_0 = -0.01
    var_1 = module_0.sign(var_0)
    assert var_1 == -1

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 0.0
    var_1 = module_0.sign(var_0)
    assert var_1 == 0

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
    var_0 = '0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.sign(var_3)
    assert var_4 == 0

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '-1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.sign(var_3)
    assert var_4 == -1

import decimal as module_0

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_positive_integer_valid_input. Retrieved 2/3 statements.
# Partially parsed test_positive_integer_type_check. Retrieved 2/3 statements.


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

def test_case_0():
    var_0 = 0

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.PositiveInteger(*var_1, **var_2)
    assert var_3 == 5

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.PositiveInteger(*var_1, **var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_new_valid_zero. Retrieved 2/3 statements.
# Partially parsed test_new_valid_positive. Retrieved 2/3 statements.


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
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.NaturalNumber(*var_1, **var_2)
    var_4 = type(var_3)



# Parsed testcases at query #5
#--------------------------




import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '12.345'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '12'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_4(var_8)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '12.345'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '12.35'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_4(var_8)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.000'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '12.345'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = [var_5]
    var_10 = {}
    var_11 = module_0.Decimal(*var_9, **var_10)
    var_12 = var_4(var_8)
    var_13 = bool(var_12 == var_11)
    assert var_13 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '-12.345'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '-12.3'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_4(var_8)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True



# Parsed testcases at query #6
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



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_4 = 'Failed to raise AssertionError'
    var_5 = Exception(var_4)

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = -5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.PositiveInteger(*var_1, **var_2)
    var_4 = 'Failed to raise AssertionError'
    var_5 = Exception(var_4)



# Parsed testcases at query #2
#--------------------------




import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '12.56'
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

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '12.567'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '12.57'
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
    var_5 = '12.5678'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = '12.568'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_4(var_8)
    var_14 = bool(var_13 == var_12)
    assert var_14 is True

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = module_1.make_quantize_func(var_3)
    var_5 = '12.56'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = [var_5]
    var_10 = {}
    var_11 = module_0.Decimal(*var_9, **var_10)
    var_12 = var_4(var_8)
    var_13 = bool(var_12 == var_11)
    assert var_13 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 4/7 statements.
# Partially parsed test_weirdiv_dividend_negative_one_divisor_none. Retrieved 6/10 statements.


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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_weirdiv_divisor_is_none. Retrieved 10/14 statements.
# Partially parsed test_weirdiv_divisor_is_zero. Retrieved 12/16 statements.


import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '10'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = None
    var_5 = module_1.weirdiv(var_3, var_4)
    var_6 = 'inf'
    var_7 = float(var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = var_5 == var_10
    var_12 = [var_0]
    var_13 = {}
    var_14 = module_0.Decimal(*var_12, **var_13)
    var_15 = module_1.weirdiv(var_14, var_4)

import decimal as module_0
import pypara.commons.numbers as module_1

def test_case_0():
    var_0 = '5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = '0'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = module_1.weirdiv(var_3, var_7)
    var_9 = 'inf'
    var_10 = float(var_9)
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_0.Decimal(*var_11, **var_12)
    var_14 = var_8 == var_13
    var_15 = [var_0]
    var_16 = {}
    var_17 = module_0.Decimal(*var_15, **var_16)
    var_18 = [var_4]
    var_19 = {}
    var_20 = module_0.Decimal(*var_18, **var_19)
    var_21 = module_1.weirdiv(var_17, var_20)



# Parsed testcases at query #5
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
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.NaturalNumber(*var_1, **var_2)
    assert var_3 == 5



