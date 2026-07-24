####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_qty_or_none_defined_price. Retrieved 4/13 statements.
# Failed to parse test_qty_or_none_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_someprice_abs. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '-100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '100.50'
    var_6 = [var_5]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_with_ccy_defined_price. Retrieved 5/16 statements.
# Partially parsed test_with_ccy_undefined_price. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = [var_1]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'EUR'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sub_defined_money. Retrieved 5/19 statements.
# Partially parsed test_sub_undefined_money. Retrieved 4/13 statements.
# Partially parsed test_sub_incompatible_currency. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]
    var_9 = [var_6]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '5'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_ccy_or_none_defined_price. Retrieved 4/11 statements.
# Partially parsed test_ccy_or_none_undefined_price. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_mul_defined_money. Retrieved 6/16 statements.
# Partially parsed test_mul_undefined_money. Retrieved 1/3 statements.
# Partially parsed test_mul_by_zero. Retrieved 6/16 statements.
# Partially parsed test_mul_by_negative. Retrieved 6/16 statements.
# Partially parsed test_mul_by_float. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2
    var_7 = '21.00'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 0
    var_7 = '0.00'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = -1
    var_7 = '-10.50'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 1.5
    var_7 = '15.75'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_someprice_neg. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '-10.5'
    var_6 = [var_5]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dov_or_none. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_truediv_valid_division. Retrieved 6/15 statements.
# Partially parsed test_truediv_division_by_zero. Retrieved 5/10 statements.
# Partially parsed test_truediv_invalid_operation. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = '5.0'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 'NaN'
    var_6 = [var_5]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_ccy_or_returns_currency. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '10.5'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = 'EUR'
    var_8 = 'Euro'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_abs_defined_price. Retrieved 5/20 statements.
# Failed to parse test_abs_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '-1.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '1.5'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_floor_divide_defined_money. Retrieved 5/16 statements.
# Partially parsed test_floor_divide_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_floor_divide_by_zero. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '3'
    var_7 = [var_6]
    var_8 = [var_6]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '3'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '0'
    var_7 = [var_6]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_gt_undefined_vs_defined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_vs_undefined. Retrieved 4/10 statements.
# Partially parsed test_gt_same_currency_and_quantity. Retrieved 5/15 statements.
# Partially parsed test_gt_same_currency_different_quantity. Retrieved 6/16 statements.
# Partially parsed test_gt_different_currency. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]
    var_7 = 2
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_lte_undefined_less_than_or_equal_to_defined. Retrieved 4/10 statements.
# Failed to parse test_lte_undefined_less_than_or_equal_to_undefined.
# Partially parsed test_lte_defined_less_than_or_equal_to_undefined. Retrieved 4/10 statements.
# Partially parsed test_lte_same_currency_less_than_or_equal. Retrieved 7/34 statements.
# Partially parsed test_lte_different_currency_raises_error. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = 2
    var_8 = [var_5]
    var_9 = [var_5]
    var_10 = '3'
    var_11 = [var_10]
    var_12 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pos_defined_price. Retrieved 4/15 statements.
# Failed to parse test_pos_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = [var_3, var_4, var_4]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_multiply_defined_money. Retrieved 6/16 statements.
# Partially parsed test_multiply_undefined_money. Retrieved 1/3 statements.
# Partially parsed test_multiply_by_zero. Retrieved 6/16 statements.
# Partially parsed test_multiply_by_negative. Retrieved 6/16 statements.
# Partially parsed test_multiply_by_float. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2
    var_7 = '20'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 0
    var_7 = '0'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = -1
    var_7 = '-10'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 1.5
    var_7 = '15.00'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_lt_undefined_vs_defined. Retrieved 4/10 statements.
# Partially parsed test_lt_defined_vs_undefined. Retrieved 4/10 statements.
# Partially parsed test_lt_defined_vs_defined_same_currency. Retrieved 5/23 statements.
# Partially parsed test_lt_defined_vs_defined_different_currency. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '2'
    var_7 = [var_6]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_scalar_add_defined_price. Retrieved 6/17 statements.
# Partially parsed test_scalar_add_undefined_price. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5.5'
    var_7 = [var_6]
    var_8 = '16.0'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '5.5'
    var_1 = [var_0]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_neg_defined_price. Retrieved 5/15 statements.
# Failed to parse test_neg_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '-10.50'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dov_or_defined_price. Retrieved 5/15 statements.
# Partially parsed test_dov_or_undefined_price. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2001
    var_7 = [var_6, var_4, var_4]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = None
    var_1 = 2019
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 2001
    var_5 = [var_4, var_2, var_2]
    var_6 = [var_4, var_2, var_2]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_as_integer_defined_price. Retrieved 4/12 statements.
# Failed to parse test_as_integer_undefined_price_raises_exception.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_someprice_gt_true. Retrieved 5/12 statements.
# Partially parsed test_someprice_gt_false. Retrieved 5/12 statements.
# Partially parsed test_someprice_gt_equal. Retrieved 4/11 statements.
# Partially parsed test_someprice_gt_different_currency. Retrieved 6/15 statements.
# Partially parsed test_someprice_gt_non_someprice. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '5.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '10.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '10.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '5.00'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_price_defined_positive_quantity. Retrieved 4/10 statements.
# Partially parsed test_price_defined_zero_quantity. Retrieved 4/10 statements.
# Failed to parse test_price_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dov_or_defined_price. Retrieved 5/15 statements.
# Partially parsed test_dov_or_undefined_price. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2001
    var_7 = [var_6, var_4, var_4]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 2001
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_convert_same_currency. Retrieved 6/15 statements.
# Partially parsed test_convert_different_currency_with_rate. Retrieved 10/26 statements.
# Partially parsed test_convert_different_currency_no_rate_non_strict. Retrieved 10/19 statements.
# Partially parsed test_convert_different_currency_no_rate_strict. Retrieved 10/20 statements.
# Partially parsed test_convert_with_custom_asof_date. Retrieved 10/27 statements.
# Partially parsed test_convert_no_fx_service_set. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = '0.85'
    var_10 = [var_9]
    var_11 = '85.00'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = {}
    var_10 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = {}
    var_10 = True
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = '0.90'
    var_10 = [var_9]
    var_11 = '90.00'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_equality_of_two_defined_money_objects_with_same_attributes. Retrieved 4/15 statements.
# Partially parsed test_equality_of_two_defined_money_objects_with_different_currencies. Retrieved 5/16 statements.
# Partially parsed test_equality_of_two_defined_money_objects_with_different_quantities. Retrieved 5/16 statements.
# Partially parsed test_equality_of_two_defined_money_objects_with_different_dates. Retrieved 5/16 statements.
# Partially parsed test_equality_of_defined_and_undefined_money_objects. Retrieved 4/12 statements.
# Failed to parse test_equality_of_two_undefined_money_objects.
# Partially parsed test_equality_of_money_object_with_non_money_object. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = [var_1]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '200.00'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = 2
    var_8 = [var_3, var_4, var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_none_price_constructor.




# Parsed testcases at query #28
#--------------------------

# Partially parsed test_some_money_ge_with_same_currency_and_greater_quantity. Retrieved 6/13 statements.
# Partially parsed test_some_money_ge_with_same_currency_and_equal_quantity. Retrieved 5/12 statements.
# Partially parsed test_some_money_ge_with_same_currency_and_less_quantity. Retrieved 6/13 statements.
# Partially parsed test_some_money_ge_with_different_currency. Retrieved 7/17 statements.
# Partially parsed test_some_money_ge_with_non_money_object. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '50.00'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_2]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '50.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '100.00'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = '50.00'
    var_8 = [var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_neg_returns_negative_quantity. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '-100.00'
    var_7 = [var_6]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_someprice_truediv_valid_division. Retrieved 6/16 statements.
# Partially parsed test_someprice_truediv_by_zero. Retrieved 5/11 statements.
# Partially parsed test_someprice_truediv_invalid_operation. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '2.0'
    var_6 = [var_5]
    var_7 = '5.0'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '0.0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 'invalid'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_with_ccy_defined_price. Retrieved 5/14 statements.
# Partially parsed test_with_ccy_undefined_price. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]

def test_case_0():
    var_0 = 'EUR'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_some_money_ge_with_same_currency_and_greater_quantity. Retrieved 6/13 statements.
# Partially parsed test_some_money_ge_with_same_currency_and_equal_quantity. Retrieved 5/12 statements.
# Partially parsed test_some_money_ge_with_same_currency_and_lesser_quantity. Retrieved 6/13 statements.
# Partially parsed test_some_money_ge_with_different_currency. Retrieved 6/15 statements.
# Partially parsed test_some_money_ge_with_non_money_object. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10.50'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '9.50'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_2]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '9.50'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '10.50'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '10.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_some_money_equality. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = '200.00'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 1



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_none_price_constructor.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_scalar_add_defined_money. Retrieved 6/17 statements.
# Partially parsed test_scalar_add_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = '15.75'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '5.25'
    var_1 = [var_0]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_some_money_int_conversion. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '123.45'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_le_with_undefined_money. Retrieved 4/9 statements.
# Partially parsed test_le_with_same_currency. Retrieved 6/14 statements.
# Partially parsed test_le_with_different_currency. Retrieved 5/15 statements.
# Partially parsed test_le_with_equal_money. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = 2



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_with_ccy_defined_money. Retrieved 5/14 statements.
# Partially parsed test_with_ccy_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]

def test_case_0():
    var_0 = 'EUR'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_someprice_constructor. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_price_ge_defined_vs_defined. Retrieved 6/15 statements.
# Partially parsed test_price_ge_defined_vs_undefined. Retrieved 4/10 statements.
# Partially parsed test_price_ge_undefined_vs_defined. Retrieved 5/10 statements.
# Failed to parse test_price_ge_undefined_vs_undefined.
# Partially parsed test_price_ge_incompatible_currency. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '5'
    var_7 = [var_6]
    var_8 = 2



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_multiply_defined_price. Retrieved 6/16 statements.
# Partially parsed test_multiply_undefined_price. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '2.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2
    var_7 = '5.0'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 5



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_floor_divide_defined_money. Retrieved 5/16 statements.
# Partially parsed test_floor_divide_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_floor_divide_by_zero. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '3'
    var_7 = [var_6]
    var_8 = [var_6]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '3'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_lt_with_same_currency. Retrieved 6/13 statements.
# Partially parsed test_lt_with_different_currency. Retrieved 7/17 statements.
# Partially parsed test_lt_with_non_somemoney_object. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '200.00'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = '200.00'
    var_8 = [var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_subtract_same_currency. Retrieved 6/17 statements.
# Partially parsed test_subtract_different_currency_raises_error. Retrieved 7/17 statements.
# Partially parsed test_subtract_undefined_money. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.50'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '50.25'
    var_7 = [var_6]
    var_8 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '100.50'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = '50.25'
    var_8 = [var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.50'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_qty_or_zero_defined_price. Retrieved 4/13 statements.
# Partially parsed test_qty_or_zero_undefined_price. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_mul_defined_price_with_scalar. Retrieved 6/16 statements.
# Partially parsed test_mul_undefined_price_with_scalar. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2
    var_7 = '21.0'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 5



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_abs_defined_price. Retrieved 5/18 statements.
# Failed to parse test_abs_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '-1.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1.5'
    var_6 = [var_5]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_scalar_add_defined_money. Retrieved 6/17 statements.
# Partially parsed test_scalar_add_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = '15.75'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '5.25'
    var_1 = [var_0]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_qty_or_zero_defined_money. Retrieved 4/13 statements.
# Partially parsed test_qty_or_zero_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.00'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_qty_or_defined_money. Retrieved 6/16 statements.
# Partially parsed test_qty_or_undefined_money. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 0
    var_7 = [var_6]
    var_8 = '1.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 0
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_as_integer_defined_money. Retrieved 4/10 statements.
# Failed to parse test_as_integer_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_gt_undefined_vs_defined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_vs_undefined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_same_currency. Retrieved 5/14 statements.
# Partially parsed test_gt_defined_different_currency. Retrieved 5/15 statements.
# Failed to parse test_gt_undefined_vs_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_bool_defined_nonzero_money. Retrieved 4/12 statements.
# Partially parsed test_bool_defined_zero_money. Retrieved 4/12 statements.
# Failed to parse test_bool_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_floor_divide_defined_price. Retrieved 5/15 statements.
# Partially parsed test_floor_divide_undefined_price. Retrieved 1/4 statements.
# Partially parsed test_floor_divide_by_zero. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '3'
    var_7 = [var_6]
    var_8 = [var_6]

def test_case_0():
    var_0 = '3'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '0'
    var_7 = [var_6]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_price_le_defined_vs_undefined. Retrieved 4/12 statements.
# Partially parsed test_price_le_same_currency. Retrieved 5/20 statements.
# Partially parsed test_price_le_different_currency. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]
    var_9 = [var_1]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = [var_1]
    var_8 = [var_3, var_4, var_4]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_int_defined_money. Retrieved 4/10 statements.
# Failed to parse test_int_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '123.45'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_lt_undefined_money_is_less_than_defined_money. Retrieved 4/11 statements.
# Partially parsed test_lt_defined_money_is_not_less_than_undefined_money. Retrieved 4/11 statements.
# Partially parsed test_lt_defined_money_with_same_currency_and_smaller_quantity. Retrieved 5/15 statements.
# Partially parsed test_lt_defined_money_with_same_currency_and_larger_quantity. Retrieved 5/15 statements.
# Partially parsed test_lt_defined_money_with_same_currency_and_equal_quantity. Retrieved 4/14 statements.
# Partially parsed test_lt_defined_money_with_different_currency_raises_error. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_mul_defined_money_with_scalar. Retrieved 6/16 statements.
# Partially parsed test_mul_undefined_money_with_scalar. Retrieved 1/3 statements.
# Partially parsed test_mul_defined_money_with_zero. Retrieved 6/16 statements.
# Partially parsed test_mul_defined_money_with_negative_scalar. Retrieved 6/16 statements.
# Partially parsed test_mul_defined_money_with_float. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2
    var_7 = '21.00'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 0
    var_7 = '0.00'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = -1
    var_7 = '-10.50'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 1.5
    var_7 = '15.75'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dov_or_defined_price. Retrieved 5/15 statements.
# Partially parsed test_dov_or_undefined_price. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2001
    var_7 = [var_6, var_4, var_4]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 2001
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = [var_0, var_1, var_1]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_or_else_defined_money_returns_itself. Retrieved 7/20 statements.
# Partially parsed test_or_else_undefined_money_returns_fallback. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '2'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_3, var_4, var_9]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_gte_with_defined_prices_same_currency. Retrieved 6/16 statements.
# Partially parsed test_gte_with_defined_prices_different_currency. Retrieved 7/18 statements.
# Partially parsed test_gte_with_undefined_price_and_defined_price. Retrieved 5/11 statements.
# Partially parsed test_gte_with_defined_price_and_undefined_price. Retrieved 4/11 statements.
# Failed to parse test_gte_with_both_undefined_prices.
# Partially parsed test_gte_with_equal_defined_prices. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '5'
    var_7 = [var_6]
    var_8 = 2
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = 2



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dov_or_defined_money. Retrieved 5/14 statements.
# Partially parsed test_dov_or_undefined_money. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2001
    var_7 = [var_6, var_4, var_4]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = None
    var_1 = 2019
    var_2 = 1
    var_3 = [var_1, var_2, var_2]
    var_4 = 2001
    var_5 = [var_4, var_2, var_2]
    var_6 = [var_4, var_2, var_2]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pos_defined_price. Retrieved 4/13 statements.
# Failed to parse test_pos_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_or_else_returns_itself_when_defined. Retrieved 7/20 statements.
# Partially parsed test_or_else_returns_fallback_when_undefined. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '2'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_3, var_4, var_9]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = None
    var_7 = [var_1]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_qty_or_none_defined_money. Retrieved 4/13 statements.
# Failed to parse test_qty_or_none_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_price_le_undefined_vs_defined. Retrieved 4/9 statements.
# Partially parsed test_price_le_defined_vs_undefined. Retrieved 4/9 statements.
# Partially parsed test_price_le_same_currency. Retrieved 5/13 statements.
# Partially parsed test_price_le_equal_prices. Retrieved 4/12 statements.
# Partially parsed test_price_le_different_currency. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]
    var_7 = bool(False)
    assert var_7 is True



