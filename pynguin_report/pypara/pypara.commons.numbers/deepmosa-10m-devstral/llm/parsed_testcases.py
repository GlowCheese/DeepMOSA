####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_sign_positive_decimal. Retrieved 1/3 statements.
# Partially parsed test_sign_zero_decimal. Retrieved 1/3 statements.
# Partially parsed test_sign_negative_zero_decimal. Retrieved 1/4 statements.
# Partially parsed test_sign_negative_decimal. Retrieved 1/3 statements.


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
    var_0 = 0
    var_1 = module_0.sign(var_0)
    assert var_1 == 0

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = -1
    var_1 = module_0.sign(var_0)
    assert var_1 == -1

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]

def test_case_0():
    var_0 = '-1'
    var_1 = [var_0]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_positive_integer_creation_with_valid_value. Retrieved 1/3 statements.
# Partially parsed test_positive_integer_creation_with_zero. Retrieved 1/3 statements.
# Partially parsed test_positive_integer_creation_with_negative_value. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_weirdiv_both_none. Retrieved 3/4 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_zero. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_one. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 2/4 statements.
# Partially parsed test_weirdiv_dividend_nine_divisor_three. Retrieved 3/7 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.weirdiv(var_0, var_0)
    var_2 = '0'
    var_3 = [var_2]

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = None
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = 3
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_divisor_none_or_zero_returns_max_decimal. Retrieved 5/25 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None
    var_3 = -1
    var_4 = [var_3]
    var_5 = [var_0]
    var_6 = 0
    var_7 = [var_6]
    var_8 = -1
    var_9 = [var_8]
    var_10 = [var_6]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_natural_number_creation_with_valid_value. Retrieved 1/4 statements.
# Partially parsed test_natural_number_creation_with_zero. Retrieved 1/4 statements.
# Partially parsed test_natural_number_creation_with_negative_value. Retrieved 1/3 statements.
# Partially parsed test_natural_number_creation_with_large_value. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 2
    var_1 = 63
    var_2 = var_0 ** var_1
    var_3 = 1
    var_4 = var_2 - var_3
    var_5 = [var_4]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_weirdiv_predicate_at_line_30. Retrieved 3/11 statements.


def test_case_0():
    var_0 = None
    var_1 = var_0 is var_0
    var_2 = 0
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_normalize_integral_value. Retrieved 6/15 statements.
# Partially parsed test_normalize_non_integral_value. Retrieved 3/12 statements.


def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]
    var_4 = '5.00'
    var_5 = [var_4]
    var_6 = '5'
    var_7 = [var_6]
    var_8 = '10.000'
    var_9 = [var_8]
    var_10 = '10'
    var_11 = [var_10]

def test_case_0():
    var_0 = '0.123'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = '5.678'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = '10.001'
    var_7 = [var_6]
    var_8 = [var_6]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_rounds_up. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_rounds_down. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.145'
    var_3 = [var_2]
    var_4 = '3.15'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.144'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_divisor_none_or_zero. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = None
    var_4 = [var_0]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_weirdiv_predicate_at_line_30. Retrieved 3/11 statements.


def test_case_0():
    var_0 = None
    var_1 = var_0 is var_0
    var_2 = 0
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_normalize_integral_value. Retrieved 6/15 statements.
# Partially parsed test_normalize_non_integral_value. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]
    var_4 = '1.00'
    var_5 = [var_4]
    var_6 = '1'
    var_7 = [var_6]
    var_8 = '10.00'
    var_9 = [var_8]
    var_10 = '10'
    var_11 = [var_10]

def test_case_0():
    var_0 = '0.12300'
    var_1 = [var_0]
    var_2 = '0.123'
    var_3 = [var_2]
    var_4 = '1.2300'
    var_5 = [var_4]
    var_6 = '1.23'
    var_7 = [var_6]
    var_8 = '10.00100'
    var_9 = [var_8]
    var_10 = '10.001'
    var_11 = [var_10]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_divisor_none_or_zero. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None
    var_3 = [var_0]
    var_4 = 0
    var_5 = [var_4]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_normalize_when_value_is_not_equal_to_integral. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '0.001'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 7/18 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]
    var_6 = '2.71828'
    var_7 = [var_6]
    var_8 = '2.72'
    var_9 = [var_8]
    var_10 = '1.00000'
    var_11 = [var_10]
    var_12 = '1.00'
    var_13 = [var_12]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_normalize_with_non_integral_value. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '1.23'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_zero. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]
    var_4 = '0.00'
    var_5 = [var_4]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_zero. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_negative_number. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]
    var_4 = '0.00'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '-3.14159'
    var_3 = [var_2]
    var_4 = '-3.14'
    var_5 = [var_4]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_negative_value. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '-3.14159'
    var_3 = [var_2]
    var_4 = '-3.14'
    var_5 = [var_4]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_positive_integer_new_with_valid_value. Retrieved 1/3 statements.
# Partially parsed test_positive_integer_new_with_zero. Retrieved 1/3 statements.
# Partially parsed test_positive_integer_new_with_negative_value. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_positive_integer_creation_with_valid_value. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_normalize_zero. Retrieved 2/5 statements.
# Partially parsed test_normalize_integer. Retrieved 2/5 statements.
# Partially parsed test_normalize_non_integer. Retrieved 1/4 statements.
# Partially parsed test_normalize_negative_integer. Retrieved 2/5 statements.
# Partially parsed test_normalize_negative_non_integer. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]

def test_case_0():
    var_0 = '5.00'
    var_1 = [var_0]
    var_2 = '5'
    var_3 = [var_2]

def test_case_0():
    var_0 = '5.123'
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = '-3.00'
    var_1 = [var_0]
    var_2 = '-3'
    var_3 = [var_2]

def test_case_0():
    var_0 = '-3.456'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_natural_number_creation_with_valid_value. Retrieved 1/3 statements.
# Partially parsed test_natural_number_creation_with_zero. Retrieved 1/3 statements.
# Partially parsed test_natural_number_creation_with_negative_value_raises_assertion_error. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_negative_value. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '-3.14159'
    var_3 = [var_2]
    var_4 = '-3.14'
    var_5 = [var_4]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_normalize_with_non_integral_value. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '1.234'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_weirdiv_none_none. Retrieved 3/4 statements.
# Partially parsed test_weirdiv_none_zero. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_none_one. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_zero_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_one_none_large. Retrieved 2/4 statements.
# Partially parsed test_weirdiv_nine_three. Retrieved 3/7 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.weirdiv(var_0, var_0)
    var_2 = '0'
    var_3 = [var_2]

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = None
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = 3
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_weirdiv_predicate_false. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None
    var_3 = [var_0]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None
    var_3 = [var_0]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_weirdiv_both_none. Retrieved 3/4 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_zero. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_one. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 2/4 statements.
# Partially parsed test_weirdiv_dividend_nine_divisor_three. Retrieved 3/7 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.weirdiv(var_0, var_0)
    var_2 = '0'
    var_3 = [var_2]

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = None
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = 3
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_zero. Retrieved 2/7 statements.
# Partially parsed test_make_quantize_func_with_negative_value. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]
    var_4 = [var_2]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '-3.14159'
    var_3 = [var_2]
    var_4 = '-3.14'
    var_5 = [var_4]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_normalize_zero. Retrieved 2/5 statements.
# Partially parsed test_normalize_integer. Retrieved 2/5 statements.
# Partially parsed test_normalize_non_integer. Retrieved 1/4 statements.
# Partially parsed test_normalize_negative_integer. Retrieved 2/5 statements.
# Partially parsed test_normalize_negative_non_integer. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]

def test_case_0():
    var_0 = '123.00'
    var_1 = [var_0]
    var_2 = '123'
    var_3 = [var_2]

def test_case_0():
    var_0 = '123.456'
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = '-123.00'
    var_1 = [var_0]
    var_2 = '-123'
    var_3 = [var_2]

def test_case_0():
    var_0 = '-123.456'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_normalize_returns_false_for_non_integral_value. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '1.23'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_negative_value. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '-3.14159'
    var_3 = [var_2]
    var_4 = '-3.14'
    var_5 = [var_4]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_false. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None
    var_3 = [var_0]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_zero. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_negative. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]
    var_4 = '0.00'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '-3.14159'
    var_3 = [var_2]
    var_4 = '-3.14'
    var_5 = [var_4]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_normalize_predicate_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '1.23'
    var_1 = [var_0]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_normalize_returns_false_for_predicate. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '1.00'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_weirdiv_both_none. Retrieved 3/4 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_zero. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_one. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 2/4 statements.
# Partially parsed test_weirdiv_dividend_nine_divisor_three. Retrieved 3/7 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.weirdiv(var_0, var_0)
    var_2 = '0'
    var_3 = [var_2]

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = None
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = 3
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_preserves_precision. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.001'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.142'
    var_5 = [var_4]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_normalize_predicate_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '1.23'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_make_quantize_func_returns_callable. Retrieved 1/4 statements.
# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1'
    var_5 = [var_4]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_normalize_integral_value. Retrieved 6/15 statements.
# Partially parsed test_normalize_non_integral_value. Retrieved 3/12 statements.


def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]
    var_4 = '5.00'
    var_5 = [var_4]
    var_6 = '5'
    var_7 = [var_6]
    var_8 = '10.000'
    var_9 = [var_8]
    var_10 = '10'
    var_11 = [var_10]

def test_case_0():
    var_0 = '0.123'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = '5.678'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = '10.001'
    var_7 = [var_6]
    var_8 = [var_6]



