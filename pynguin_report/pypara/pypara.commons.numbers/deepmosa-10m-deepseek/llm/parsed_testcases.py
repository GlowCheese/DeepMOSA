####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
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

# Partially parsed test_weirdiv_both_none. Retrieved 3/4 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_zero. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_one. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_normal_division. Retrieved 3/7 statements.


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
    var_3 = '1e10'
    var_4 = [var_3]

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = 3
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_PositiveInteger_valid_value. Retrieved 2/4 statements.
# Partially parsed test_PositiveInteger_zero_value. Retrieved 1/3 statements.
# Partially parsed test_PositiveInteger_negative_value. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 42
    var_3 = [var_2]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = -5
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_zero_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_large_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_negative_number. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_exact_value. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.456'
    var_3 = [var_2]
    var_4 = '3.46'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = '3.456'
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]

def test_case_0():
    var_0 = '1.0'
    var_1 = [var_0]
    var_2 = '3.456'
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '-3.456'
    var_3 = [var_2]
    var_4 = '-3.5'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.25'
    var_1 = [var_0]
    var_2 = '1.25'
    var_3 = [var_2]
    var_4 = [var_2]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test___new___creates_instance_with_non_negative_integer. Retrieved 1/3 statements.
# Partially parsed test___new___creates_instance_with_positive_integer. Retrieved 1/3 statements.
# Partially parsed test___new___raises_assertion_error_with_negative_integer. Retrieved 1/3 statements.
# Partially parsed test___new___creates_instance_with_zero. Retrieved 1/3 statements.
# Partially parsed test___new___creates_instance_with_large_integer. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 0
    var_1 = [var_0]

def test_case_0():
    var_0 = 42
    var_1 = [var_0]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 0
    var_1 = [var_0]

def test_case_0():
    var_0 = 999999
    var_1 = [var_0]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_normalize_integer_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_non_integer_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_zero_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_negative_integer_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_negative_non_integer_value. Retrieved 1/4 statements.
# Partially parsed test_normalize_large_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_small_value. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '5.00'
    var_1 = [var_0]
    var_2 = '5'
    var_3 = [var_2]

def test_case_0():
    var_0 = '5.50'
    var_1 = [var_0]
    var_2 = '5.5'
    var_3 = [var_2]

def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]

def test_case_0():
    var_0 = '-3.00'
    var_1 = [var_0]
    var_2 = '-3'
    var_3 = [var_2]

def test_case_0():
    var_0 = '-3.75'
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = '123456789.00000'
    var_1 = [var_0]
    var_2 = '123456789'
    var_3 = [var_2]

def test_case_0():
    var_0 = '0.000001'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '1.23'
    var_1 = [var_0]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_weirdiv_divisor_none_or_zero_returns_max_value. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None
    var_3 = 0
    var_4 = [var_3]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_30_evaluates_to_true. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None
    var_3 = [var_0]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_11_evaluates_to_false. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = '0.01'
    var_3 = [var_2]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_normalize_zero. Retrieved 2/5 statements.
# Partially parsed test_normalize_integer. Retrieved 2/5 statements.
# Partially parsed test_normalize_decimal. Retrieved 2/5 statements.
# Partially parsed test_normalize_large_decimal. Retrieved 1/4 statements.
# Partially parsed test_normalize_negative. Retrieved 2/5 statements.
# Partially parsed test_normalize_small_decimal. Retrieved 1/4 statements.


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
    var_0 = '5.50'
    var_1 = [var_0]
    var_2 = '5.5'
    var_3 = [var_2]

def test_case_0():
    var_0 = '123456789.987654321'
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = '-3.00'
    var_1 = [var_0]
    var_2 = '-3'
    var_3 = [var_2]

def test_case_0():
    var_0 = '0.000001'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_weirdiv_both_none. Retrieved 3/4 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_zero. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_one. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 4/7 statements.
# Partially parsed test_weirdiv_normal_division. Retrieved 3/7 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_none. Retrieved 4/8 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_zero. Retrieved 2/6 statements.
# Partially parsed test_weirdiv_dividend_positive_divisor_zero. Retrieved 2/7 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_zero. Retrieved 2/8 statements.


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
    var_3 = 10
    var_4 = var_3 ** var_3
    var_5 = [var_4]

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = 3
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = None
    var_3 = 10
    var_4 = var_3 ** var_3

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_30_evaluates_to_true. Retrieved 7/31 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None
    var_3 = -1
    var_4 = [var_3]
    var_5 = -1
    var_6 = [var_5]
    var_7 = [var_0]
    var_8 = 0
    var_9 = [var_8]
    var_10 = -1
    var_11 = [var_10]
    var_12 = [var_8]
    var_13 = -1
    var_14 = [var_13]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_weirdiv_both_none. Retrieved 3/4 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_zero. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_one. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 4/7 statements.
# Partially parsed test_weirdiv_normal_division. Retrieved 3/7 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_one. Retrieved 3/7 statements.
# Partially parsed test_weirdiv_dividend_one_divisor_zero. Retrieved 2/7 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_zero. Retrieved 2/8 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_one. Retrieved 3/7 statements.


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
    var_3 = 10
    var_4 = var_3 ** var_3
    var_5 = [var_4]

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = 3
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 1
    var_3 = [var_2]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]

def test_case_0():
    var_0 = -9
    var_1 = [var_0]
    var_2 = 3
    var_3 = [var_2]
    var_4 = '-3'
    var_5 = [var_4]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test___new___creates_instance_with_non_negative_integer. Retrieved 1/3 statements.
# Partially parsed test___new___creates_instance_with_positive_integer. Retrieved 1/3 statements.
# Partially parsed test___new___raises_assertion_error_with_negative_integer. Retrieved 1/3 statements.
# Partially parsed test___new___raises_assertion_error_with_non_integer. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 0
    var_1 = [var_0]

def test_case_0():
    var_0 = 42
    var_1 = [var_0]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'not an integer'
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_weirdiv_both_none. Retrieved 3/4 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_zero. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_one. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 4/7 statements.
# Partially parsed test_weirdiv_normal_division. Retrieved 3/7 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_zero. Retrieved 2/6 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_none. Retrieved 4/8 statements.
# Partially parsed test_weirdiv_dividend_positive_divisor_zero. Retrieved 4/8 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_zero. Retrieved 4/9 statements.


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
    var_3 = 10
    var_4 = var_3 ** var_3
    var_5 = [var_4]

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = 3
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = None
    var_3 = 10
    var_4 = var_3 ** var_3

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]
    var_4 = 10
    var_5 = var_4 ** var_4
    var_6 = [var_5]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]
    var_4 = 10
    var_5 = var_4 ** var_4



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_constructor_raises_assertion_error_for_negative_value. Retrieved 1/3 statements.


def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_natural_number_creation_with_negative_value. Retrieved 1/3 statements.


def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/10 statements.
# Partially parsed test_make_quantize_func_with_zero. Retrieved 3/10 statements.
# Partially parsed test_make_quantize_func_with_negative_number. Retrieved 3/10 statements.
# Partially parsed test_make_quantize_func_with_large_quantizer. Retrieved 2/9 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '1.234'
    var_3 = [var_2]
    var_4 = '1.23'
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
    var_2 = '-1.234'
    var_3 = [var_2]
    var_4 = '-1.23'
    var_5 = [var_4]

def test_case_0():
    var_0 = '1.00'
    var_1 = [var_0]
    var_2 = '1.234'
    var_3 = [var_2]
    var_4 = [var_0]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_quantize_func_with_integer_quantizer. Retrieved 4/12 statements.
# Partially parsed test_make_quantize_func_with_decimal_quantizer. Retrieved 5/13 statements.
# Partially parsed test_make_quantize_func_with_zero. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_negative_numbers. Retrieved 5/13 statements.


def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]
    var_6 = '2.71828'
    var_7 = [var_6]
    var_8 = [var_4]

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

def test_case_0():
    var_0 = '0.0001'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]
    var_4 = '0.0000'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '-1.234'
    var_3 = [var_2]
    var_4 = '-1.2'
    var_5 = [var_4]
    var_6 = '-5.678'
    var_7 = [var_6]
    var_8 = '-5.7'
    var_9 = [var_8]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_make_quantize_func_with_integer_quantizer. Retrieved 5/16 statements.
# Partially parsed test_make_quantize_func_with_decimal_quantizer. Retrieved 7/18 statements.
# Partially parsed test_make_quantize_func_with_small_quantizer. Retrieved 7/18 statements.
# Partially parsed test_make_quantize_func_with_zero. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_negative_numbers. Retrieved 7/18 statements.


def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]
    var_6 = '2.71828'
    var_7 = [var_6]
    var_8 = [var_4]
    var_9 = '0.99999'
    var_10 = [var_9]
    var_11 = [var_0]

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
    var_10 = '0.99999'
    var_11 = [var_10]
    var_12 = '1.00'
    var_13 = [var_12]

def test_case_0():
    var_0 = '0.0001'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.1416'
    var_5 = [var_4]
    var_6 = '2.71828'
    var_7 = [var_6]
    var_8 = '2.7183'
    var_9 = [var_8]
    var_10 = '0.99999'
    var_11 = [var_10]
    var_12 = '1.0000'
    var_13 = [var_12]

def test_case_0():
    var_0 = '0.0001'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]
    var_4 = '0.0000'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '-3.14159'
    var_3 = [var_2]
    var_4 = '-3.1'
    var_5 = [var_4]
    var_6 = '-2.71828'
    var_7 = [var_6]
    var_8 = '-2.7'
    var_9 = [var_8]
    var_10 = '-0.99999'
    var_11 = [var_10]
    var_12 = '-1.0'
    var_13 = [var_12]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_weirdiv_divisor_none. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_normalize_integer_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_non_integer_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_zero_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_negative_integer_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_negative_non_integer_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_large_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_small_value. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '5.00'
    var_1 = [var_0]
    var_2 = '5'
    var_3 = [var_2]

def test_case_0():
    var_0 = '5.50'
    var_1 = [var_0]
    var_2 = '5.5'
    var_3 = [var_2]

def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]

def test_case_0():
    var_0 = '-3.00'
    var_1 = [var_0]
    var_2 = '-3'
    var_3 = [var_2]

def test_case_0():
    var_0 = '-3.50'
    var_1 = [var_0]
    var_2 = '-3.5'
    var_3 = [var_2]

def test_case_0():
    var_0 = '123456789.00'
    var_1 = [var_0]
    var_2 = '123456789'
    var_3 = [var_2]

def test_case_0():
    var_0 = '0.000001'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_natural_number_negative_value_raises_assertion_error. Retrieved 1/2 statements.


def test_case_0():
    var_0 = -1
    var_1 = [var_0]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_make_quantize_func. Retrieved 15/44 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '1.234'
    var_3 = [var_2]
    var_4 = '1.23'
    var_5 = [var_4]
    var_6 = '1.235'
    var_7 = [var_6]
    var_8 = '1.24'
    var_9 = [var_8]
    var_10 = '1.000'
    var_11 = [var_10]
    var_12 = '1.00'
    var_13 = [var_12]
    var_14 = '0.999'
    var_15 = [var_14]
    var_16 = [var_12]
    var_17 = '0.0001'
    var_18 = [var_17]
    var_19 = '1.2345'
    var_20 = [var_19]
    var_21 = [var_19]
    var_22 = '1.23456'
    var_23 = [var_22]
    var_24 = '1.2346'
    var_25 = [var_24]
    var_26 = '1.23454'
    var_27 = [var_26]
    var_28 = [var_19]
    var_29 = '0.00001'
    var_30 = [var_29]
    var_31 = '0.0000'
    var_32 = [var_31]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_normalize_zero. Retrieved 2/5 statements.
# Partially parsed test_normalize_whole_number. Retrieved 2/5 statements.
# Partially parsed test_normalize_decimal. Retrieved 1/4 statements.
# Partially parsed test_normalize_negative_whole_number. Retrieved 2/5 statements.
# Partially parsed test_normalize_negative_decimal. Retrieved 1/4 statements.
# Partially parsed test_normalize_large_number. Retrieved 2/5 statements.
# Partially parsed test_normalize_small_number. Retrieved 1/4 statements.
# Partially parsed test_normalize_very_small_number. Retrieved 1/4 statements.


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
    var_0 = '3.14'
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = '-7.00'
    var_1 = [var_0]
    var_2 = '-7'
    var_3 = [var_2]

def test_case_0():
    var_0 = '-2.71'
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = '123456789.00'
    var_1 = [var_0]
    var_2 = '123456789'
    var_3 = [var_2]

def test_case_0():
    var_0 = '0.000001'
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = '0.000000000000001'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_30_evaluates_to_true_when_divisor_is_none. Retrieved 2/4 statements.
# Partially parsed test_predicate_at_line_30_evaluates_to_true_when_divisor_is_zero. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = None

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_make_quantize_func_basic. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_zero. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_negative. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_large_number. Retrieved 2/7 statements.
# Partially parsed test_make_quantize_func_small_quantizer. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '1.234'
    var_3 = [var_2]
    var_4 = '1.23'
    var_5 = [var_4]

def test_case_0():
    var_0 = '1.00'
    var_1 = [var_0]
    var_2 = '0.99'
    var_3 = [var_2]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '-1.23'
    var_3 = [var_2]
    var_4 = '-1.2'
    var_5 = [var_4]

def test_case_0():
    var_0 = '1000'
    var_1 = [var_0]
    var_2 = '1234'
    var_3 = [var_2]
    var_4 = [var_0]

def test_case_0():
    var_0 = '0.0001'
    var_1 = [var_0]
    var_2 = '1.23456789'
    var_3 = [var_2]
    var_4 = '1.2346'
    var_5 = [var_4]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_assertion_error_raised_for_negative_value. Retrieved 1/3 statements.


def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_negative_value_raises_assertion_error. Retrieved 1/3 statements.


def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_normalize_zero. Retrieved 2/5 statements.
# Partially parsed test_normalize_integer. Retrieved 2/5 statements.
# Partially parsed test_normalize_decimal. Retrieved 1/4 statements.
# Partially parsed test_normalize_negative. Retrieved 2/5 statements.
# Partially parsed test_normalize_large_number. Retrieved 2/5 statements.
# Partially parsed test_normalize_small_number. Retrieved 1/4 statements.


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
    var_0 = '3.14'
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = '-2.50'
    var_1 = [var_0]
    var_2 = '-2.5'
    var_3 = [var_2]

def test_case_0():
    var_0 = '1234567890.00'
    var_1 = [var_0]
    var_2 = '1234567890'
    var_3 = [var_2]

def test_case_0():
    var_0 = '0.000001'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/9 statements.
# Partially parsed test_make_quantize_func_with_different_quantizer. Retrieved 3/9 statements.
# Partially parsed test_make_quantize_func_with_integer_quantizer. Retrieved 3/9 statements.
# Partially parsed test_make_quantize_func_with_zero_input. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.000'
    var_1 = [var_0]
    var_2 = '2.71828'
    var_3 = [var_2]
    var_4 = '2.718'
    var_5 = [var_4]

def test_case_0():
    var_0 = '1.'
    var_1 = [var_0]
    var_2 = '5.678'
    var_3 = [var_2]
    var_4 = '6'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '0.00'
    var_3 = [var_2]
    var_4 = [var_2]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_negative_value_raises_assertion_error. Retrieved 1/3 statements.


def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = '0.01'
    var_3 = [var_2]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_make_quantize_func_rounds_to_two_decimal_places. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_rounds_to_three_decimal_places. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_rounds_to_integer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_rounds_negative_number. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_rounds_zero. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = '3.14159'
    var_3 = [var_2]
    var_4 = '3.14'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.000'
    var_1 = [var_0]
    var_2 = '2.71828'
    var_3 = [var_2]
    var_4 = '2.718'
    var_5 = [var_4]

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = '7.89'
    var_3 = [var_2]
    var_4 = '8'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '-5.678'
    var_3 = [var_2]
    var_4 = '-5.68'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.0001'
    var_1 = [var_0]
    var_2 = '0.00005'
    var_3 = [var_2]
    var_4 = [var_0]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test___new___creates_instance_with_positive_integer. Retrieved 1/3 statements.
# Partially parsed test___new___raises_assertion_error_with_zero. Retrieved 1/3 statements.
# Partially parsed test___new___raises_assertion_error_with_negative_integer. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_NaturalNumber_creates_instance_with_non_negative_integer. Retrieved 1/3 statements.
# Partially parsed test_NaturalNumber_raises_assertion_error_with_negative_integer. Retrieved 1/3 statements.
# Partially parsed test_NaturalNumber_creates_instance_with_zero. Retrieved 1/3 statements.
# Partially parsed test_NaturalNumber_creates_instance_with_large_positive_integer. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 5
    var_1 = [var_0]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 0
    var_1 = [var_0]

def test_case_0():
    var_0 = 1000000
    var_1 = [var_0]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test___new___with_positive_integer. Retrieved 1/2 statements.
# Partially parsed test___new___with_zero. Retrieved 1/2 statements.
# Partially parsed test___new___with_negative_integer_raises_assertion_error. Retrieved 1/3 statements.
# Partially parsed test___new___with_non_integer_raises_type_error. Retrieved 1/3 statements.


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
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = 'not an integer'
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_quantize_func_with_zero. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_small_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_large_quantizer. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_negative_number. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = '123.456'
    var_3 = [var_2]
    var_4 = '123.46'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.001'
    var_1 = [var_0]
    var_2 = '123.4567'
    var_3 = [var_2]
    var_4 = '123.457'
    var_5 = [var_4]

def test_case_0():
    var_0 = '10'
    var_1 = [var_0]
    var_2 = '123.456'
    var_3 = [var_2]
    var_4 = '120'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '-123.456'
    var_3 = [var_2]
    var_4 = '-123.46'
    var_5 = [var_4]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_positive_integer_creation_with_valid_value. Retrieved 1/2 statements.
# Partially parsed test_positive_integer_creation_with_large_valid_value. Retrieved 1/2 statements.
# Partially parsed test_positive_integer_creation_with_minimum_valid_value. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 1000
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_make_quantize_func_with_default_context. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_zero_decimal. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_large_quantizer. Retrieved 2/7 statements.
# Partially parsed test_make_quantize_func_with_small_quantizer. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '1.234'
    var_3 = [var_2]
    var_4 = '1.23'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.000'
    var_1 = [var_0]
    var_2 = '1.234'
    var_3 = [var_2]
    var_4 = '1'
    var_5 = [var_4]

def test_case_0():
    var_0 = '1.00'
    var_1 = [var_0]
    var_2 = '1.234'
    var_3 = [var_2]
    var_4 = [var_0]

def test_case_0():
    var_0 = '0.0001'
    var_1 = [var_0]
    var_2 = '1.2345'
    var_3 = [var_2]
    var_4 = [var_2]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_normalize_zero. Retrieved 2/5 statements.
# Partially parsed test_normalize_integer. Retrieved 2/5 statements.
# Partially parsed test_normalize_decimal. Retrieved 2/5 statements.
# Partially parsed test_normalize_negative_integer. Retrieved 2/5 statements.
# Partially parsed test_normalize_negative_decimal. Retrieved 2/5 statements.
# Partially parsed test_normalize_large_number. Retrieved 2/5 statements.
# Partially parsed test_normalize_small_decimal. Retrieved 2/5 statements.


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
    var_0 = '3.1400'
    var_1 = [var_0]
    var_2 = '3.14'
    var_3 = [var_2]

def test_case_0():
    var_0 = '-2.00'
    var_1 = [var_0]
    var_2 = '-2'
    var_3 = [var_2]

def test_case_0():
    var_0 = '-1.2300'
    var_1 = [var_0]
    var_2 = '-1.23'
    var_3 = [var_2]

def test_case_0():
    var_0 = '123456789.0000'
    var_1 = [var_0]
    var_2 = '123456789'
    var_3 = [var_2]

def test_case_0():
    var_0 = '0.0000123400'
    var_1 = [var_0]
    var_2 = '0.00001234'
    var_3 = [var_2]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/9 statements.
# Partially parsed test_make_quantize_func_with_zero_quantizer. Retrieved 3/9 statements.
# Partially parsed test_make_quantize_func_with_large_quantizer. Retrieved 3/9 statements.
# Partially parsed test_make_quantize_func_with_negative_number. Retrieved 3/9 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.456'
    var_3 = [var_2]
    var_4 = '3.46'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = '3.456'
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]

def test_case_0():
    var_0 = '10'
    var_1 = [var_0]
    var_2 = '3.456'
    var_3 = [var_2]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '-3.456'
    var_3 = [var_2]
    var_4 = '-3.5'
    var_5 = [var_4]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_positive_integer_creation_with_positive_value. Retrieved 1/3 statements.
# Partially parsed test_positive_integer_creation_with_large_positive_value. Retrieved 1/3 statements.
# Partially parsed test_positive_integer_creation_with_zero_raises_assertion_error. Retrieved 1/3 statements.
# Partially parsed test_positive_integer_creation_with_negative_value_raises_assertion_error. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 999999
    var_1 = [var_0]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_normalize_non_integral_value. Retrieved 1/4 statements.
# Partially parsed test_normalize_integral_value. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '1.23'
    var_1 = [var_0]

def test_case_0():
    var_0 = '2.00'
    var_1 = [var_0]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_normalize_zero. Retrieved 2/5 statements.
# Partially parsed test_normalize_integer. Retrieved 2/5 statements.
# Partially parsed test_normalize_decimal. Retrieved 1/4 statements.
# Partially parsed test_normalize_large_number. Retrieved 2/5 statements.
# Partially parsed test_normalize_small_number. Retrieved 1/4 statements.


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
    var_0 = '3.14'
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = '123456789.00000'
    var_1 = [var_0]
    var_2 = '123456789'
    var_3 = [var_2]

def test_case_0():
    var_0 = '0.00000123'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_positive_integer_creation. Retrieved 1/2 statements.
# Partially parsed test_positive_integer_creation_with_large_value. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 1000000
    var_1 = [var_0]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_quantize_func_basic. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_zero. Retrieved 2/7 statements.
# Partially parsed test_make_quantize_func_large_number. Retrieved 2/7 statements.
# Partially parsed test_make_quantize_func_small_precision. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_negative_numbers. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '1.234'
    var_3 = [var_2]
    var_4 = '1.23'
    var_5 = [var_4]

def test_case_0():
    var_0 = '1.00'
    var_1 = [var_0]
    var_2 = '0.99'
    var_3 = [var_2]
    var_4 = [var_0]

def test_case_0():
    var_0 = '1000'
    var_1 = [var_0]
    var_2 = '1234'
    var_3 = [var_2]
    var_4 = [var_0]

def test_case_0():
    var_0 = '0.0001'
    var_1 = [var_0]
    var_2 = '1.23456789'
    var_3 = [var_2]
    var_4 = '1.2346'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '-1.23'
    var_3 = [var_2]
    var_4 = '-1.2'
    var_5 = [var_4]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_weirdiv_both_none. Retrieved 3/4 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_zero. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_one. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 2/4 statements.
# Partially parsed test_weirdiv_normal_division. Retrieved 3/7 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_one. Retrieved 3/7 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_none. Retrieved 2/4 statements.
# Partially parsed test_weirdiv_dividend_positive_divisor_zero. Retrieved 2/7 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_zero. Retrieved 3/10 statements.


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

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 1
    var_3 = [var_2]
    var_4 = '0'
    var_5 = [var_4]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = None

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]
    var_4 = -1
    var_5 = [var_4]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_weirdiv_predicate_at_line_26_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = [var_0]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_normalize_integer_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_non_integer_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_zero_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_negative_integer_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_negative_non_integer_value. Retrieved 1/4 statements.
# Partially parsed test_normalize_large_value. Retrieved 2/5 statements.
# Partially parsed test_normalize_small_value. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '5.00'
    var_1 = [var_0]
    var_2 = '5'
    var_3 = [var_2]

def test_case_0():
    var_0 = '5.50'
    var_1 = [var_0]
    var_2 = '5.5'
    var_3 = [var_2]

def test_case_0():
    var_0 = '0.00'
    var_1 = [var_0]
    var_2 = '0'
    var_3 = [var_2]

def test_case_0():
    var_0 = '-3.00'
    var_1 = [var_0]
    var_2 = '-3'
    var_3 = [var_2]

def test_case_0():
    var_0 = '-3.75'
    var_1 = [var_0]
    var_2 = [var_0]

def test_case_0():
    var_0 = '123456789.00'
    var_1 = [var_0]
    var_2 = '123456789'
    var_3 = [var_2]

def test_case_0():
    var_0 = '0.000001'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_weirdiv_both_none. Retrieved 3/4 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_zero. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_one. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 4/7 statements.
# Partially parsed test_weirdiv_normal_division. Retrieved 3/7 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_zero. Retrieved 2/6 statements.
# Partially parsed test_weirdiv_dividend_positive_divisor_zero. Retrieved 4/8 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_zero. Retrieved 4/9 statements.
# Partially parsed test_weirdiv_dividend_positive_divisor_positive. Retrieved 3/7 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_positive. Retrieved 3/7 statements.
# Partially parsed test_weirdiv_dividend_positive_divisor_negative. Retrieved 3/7 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_negative. Retrieved 3/7 statements.


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
    var_3 = 10
    var_4 = var_3 ** var_3
    var_5 = [var_4]

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = 3
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 5
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]
    var_4 = 10
    var_5 = var_4 ** var_4
    var_6 = [var_5]

def test_case_0():
    var_0 = -5
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]
    var_4 = 10
    var_5 = var_4 ** var_4

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = '5'
    var_5 = [var_4]

def test_case_0():
    var_0 = -10
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = '-5'
    var_5 = [var_4]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = -2
    var_3 = [var_2]
    var_4 = '-5'
    var_5 = [var_4]

def test_case_0():
    var_0 = -10
    var_1 = [var_0]
    var_2 = -2
    var_3 = [var_2]
    var_4 = '5'
    var_5 = [var_4]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_weirdiv_dividend_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_zero. Retrieved 3/7 statements.


def test_case_0():
    var_0 = None
    var_1 = 2
    var_2 = [var_1]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = '0'
    var_5 = [var_4]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_weirdiv_predicate_at_line_26_evaluates_to_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = '0.5'
    var_5 = [var_4]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_normalize_returns_normalized_value_when_not_integral. Retrieved 1/4 statements.
# Partially parsed test_normalize_returns_quantized_value_when_integral. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '1.23'
    var_1 = [var_0]

def test_case_0():
    var_0 = '1.00'
    var_1 = [var_0]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_make_quantize_func_with_rounding. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_whole_number. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_zero. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_with_negative_number. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.456'
    var_3 = [var_2]
    var_4 = '3.46'
    var_5 = [var_4]

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = '5.678'
    var_3 = [var_2]
    var_4 = '6'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = '7.89'
    var_3 = [var_2]
    var_4 = '8'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.1'
    var_1 = [var_0]
    var_2 = '-2.34'
    var_3 = [var_2]
    var_4 = '-2.3'
    var_5 = [var_4]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_positive_integer_creation_with_positive_value. Retrieved 1/3 statements.
# Partially parsed test_positive_integer_creation_with_large_positive_value. Retrieved 1/3 statements.
# Partially parsed test_positive_integer_creation_with_minimal_positive_value. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 999999
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_normalize_non_integral_value. Retrieved 1/4 statements.
# Partially parsed test_normalize_integral_value. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '1.23'
    var_1 = [var_0]

def test_case_0():
    var_0 = '2.00'
    var_1 = [var_0]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_make_quantize_func. Retrieved 13/37 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '1.234'
    var_3 = [var_2]
    var_4 = '1.23'
    var_5 = [var_4]
    var_6 = '1.235'
    var_7 = [var_6]
    var_8 = '1.24'
    var_9 = [var_8]
    var_10 = '1.236'
    var_11 = [var_10]
    var_12 = [var_8]
    var_13 = '0.000'
    var_14 = [var_13]
    var_15 = '0.00'
    var_16 = [var_15]
    var_17 = '-1.234'
    var_18 = [var_17]
    var_19 = '-1.23'
    var_20 = [var_19]
    var_21 = '-1.235'
    var_22 = [var_21]
    var_23 = '-1.24'
    var_24 = [var_23]
    var_25 = '-1.236'
    var_26 = [var_25]
    var_27 = [var_23]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_weirdiv_predicate_at_line_26_evaluates_to_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_positive_integer_creation_with_positive_value. Retrieved 1/3 statements.
# Partially parsed test_positive_integer_creation_with_large_positive_value. Retrieved 1/3 statements.
# Partially parsed test_positive_integer_creation_with_min_positive_value. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 999999
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_positive_integer_creation_with_valid_value. Retrieved 1/2 statements.
# Partially parsed test_positive_integer_creation_with_large_value. Retrieved 1/2 statements.
# Partially parsed test_positive_integer_creation_with_min_valid_value. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]

def test_case_0():
    var_0 = 1000000
    var_1 = [var_0]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_normalize_non_integral_value. Retrieved 1/4 statements.
# Partially parsed test_normalize_integral_value. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '1.23'
    var_1 = [var_0]

def test_case_0():
    var_0 = '1.00'
    var_1 = [var_0]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_weirdiv_both_none. Retrieved 3/4 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_zero. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_none_divisor_one. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_none. Retrieved 3/6 statements.
# Partially parsed test_weirdiv_dividend_one_divisor_none. Retrieved 4/7 statements.
# Partially parsed test_weirdiv_normal_division. Retrieved 3/7 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_none. Retrieved 4/8 statements.
# Partially parsed test_weirdiv_dividend_zero_divisor_zero. Retrieved 2/6 statements.
# Partially parsed test_weirdiv_dividend_positive_divisor_zero. Retrieved 4/8 statements.
# Partially parsed test_weirdiv_dividend_negative_divisor_zero. Retrieved 4/9 statements.


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
    var_3 = 10
    var_4 = var_3 ** var_3
    var_5 = [var_4]

def test_case_0():
    var_0 = 9
    var_1 = [var_0]
    var_2 = 3
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = None
    var_3 = 10
    var_4 = var_3 ** var_3

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]
    var_4 = 10
    var_5 = var_4 ** var_4
    var_6 = [var_5]

def test_case_0():
    var_0 = -1
    var_1 = [var_0]
    var_2 = 0
    var_3 = [var_2]
    var_4 = 10
    var_5 = var_4 ** var_4



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_make_quantize_func_rounds_to_two_decimals. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_rounds_to_zero_decimals. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_rounds_to_five_decimals. Retrieved 3/8 statements.
# Partially parsed test_make_quantize_func_rounds_negative_numbers. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '3.456'
    var_3 = [var_2]
    var_4 = '3.46'
    var_5 = [var_4]

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = '3.456'
    var_3 = [var_2]
    var_4 = '3'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.00001'
    var_1 = [var_0]
    var_2 = '3.456789'
    var_3 = [var_2]
    var_4 = '3.45679'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '-3.456'
    var_3 = [var_2]
    var_4 = '-3.46'
    var_5 = [var_4]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '1.23'
    var_1 = [var_0]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_make_quantize_func_quantizes_correctly. Retrieved 3/9 statements.
# Partially parsed test_make_quantize_func_rounds_up_correctly. Retrieved 3/9 statements.
# Partially parsed test_make_quantize_func_handles_zero. Retrieved 3/9 statements.
# Partially parsed test_make_quantize_func_handles_negative_numbers. Retrieved 3/9 statements.


def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '1.234'
    var_3 = [var_2]
    var_4 = '1.23'
    var_5 = [var_4]

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = '1.235'
    var_3 = [var_2]
    var_4 = '1.24'
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
    var_2 = '-1.234'
    var_3 = [var_2]
    var_4 = '-1.23'
    var_5 = [var_4]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_weirdiv_dividend_is_zero. Retrieved 2/5 statements.
# Partially parsed test_weirdiv_dividend_is_none. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = 1
    var_3 = [var_2]

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = [var_1]



