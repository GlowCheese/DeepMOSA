####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_qty_or_none. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = None
    var_7 = [var_1]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_scalar_subtract. Retrieved 14/54 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '7'
    var_8 = [var_7]
    var_9 = 'EUR'
    var_10 = '5'
    var_11 = [var_10]
    var_12 = '-2'
    var_13 = [var_12]
    var_14 = [var_7]
    var_15 = [var_10]
    var_16 = 'GBP'
    var_17 = [var_10]
    var_18 = [var_10]
    var_19 = '0'
    var_20 = [var_19]
    var_21 = 'JPY'
    var_22 = '2'
    var_23 = [var_22]
    var_24 = [var_7]
    var_25 = '-5'
    var_26 = [var_25]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dov_or. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 15
    var_6 = [var_3, var_4, var_5]
    var_7 = 2020
    var_8 = [var_7, var_4, var_4]
    var_9 = [var_3, var_4, var_5]
    var_10 = 2025
    var_11 = 12
    var_12 = 31
    var_13 = [var_10, var_11, var_12]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_fmap_with_defined_price. Retrieved 7/21 statements.
# Partially parsed test_fmap_with_undefined_price. Retrieved 2/9 statements.
# Partially parsed test_fmap_function_receives_correct_price. Retrieved 6/21 statements.
# Partially parsed test_fmap_returns_result_of_function. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = 10
    var_8 = '2'
    var_9 = [var_8]
    var_10 = 11
    var_11 = [var_3, var_4, var_10]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 6
    var_5 = 15
    var_6 = [var_3, var_4, var_5]
    var_7 = None
    var_8 = var_7.ccy.code
    assert var_8 == 'USD'
    var_9 = [var_1]
    var_10 = var_7.qty
    var_11 = [var_3, var_4, var_5]
    var_12 = var_7.dov

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '20'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_round_defined_price. Retrieved 6/21 statements.
# Partially parsed test_round_undefined_price. Retrieved 1/5 statements.
# Partially parsed test_round_zero_digits. Retrieved 6/17 statements.
# Partially parsed test_round_negative_quantity. Retrieved 6/17 statements.
# Partially parsed test_round_half_even. Retrieved 6/17 statements.
# Partially parsed test_round_large_ndigits. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.456'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = '1.46'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.567'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 0
    var_6 = '2'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-1.456'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = '-1.46'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.225'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = '1.22'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.123456789'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 5
    var_6 = '1.12346'
    var_7 = [var_6]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_scalar_add_with_defined_money. Retrieved 6/17 statements.
# Partially parsed test_scalar_add_with_undefined_money. Retrieved 1/5 statements.
# Partially parsed test_scalar_add_with_negative_scalar. Retrieved 6/16 statements.
# Partially parsed test_scalar_add_with_zero_scalar. Retrieved 5/15 statements.
# Partially parsed test_scalar_add_with_large_scalar. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5.00'
    var_6 = [var_5]
    var_7 = '15.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = '5.00'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3.00'
    var_6 = [var_5]
    var_7 = '7.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0.00'
    var_6 = [var_5]
    var_7 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '999999.99'
    var_6 = [var_5]
    var_7 = '1000100.49'
    var_8 = [var_7]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_is_equal_with_same_defined_money. Retrieved 4/16 statements.
# Partially parsed test_is_equal_with_different_quantity. Retrieved 5/17 statements.
# Partially parsed test_is_equal_with_different_currency. Retrieved 5/17 statements.
# Partially parsed test_is_equal_with_different_date. Retrieved 5/17 statements.
# Failed to parse test_is_equal_with_undefined_money.
# Partially parsed test_is_equal_with_defined_and_undefined. Retrieved 4/13 statements.
# Partially parsed test_is_equal_with_non_money_object. Retrieved 5/13 statements.
# Partially parsed test_is_equal_with_none. Retrieved 5/13 statements.


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
    var_5 = '2'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]

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

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'not a money object'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_someprice_sub. Retrieved 12/40 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = 'EUR'
    var_4 = [var_1]
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = '30.00'
    var_11 = [var_10]
    var_12 = [var_7, var_8, var_8]
    var_13 = '50.00'
    var_14 = [var_13]
    var_15 = [var_7, var_8, var_8]
    var_16 = '70.00'
    var_17 = [var_16]
    var_18 = [var_7, var_8, var_8]
    var_19 = bool(False)
    assert var_19 is True
    var_20 = '20.00'
    var_21 = [var_20]
    var_22 = 15
    var_23 = [var_7, var_8, var_22]
    var_24 = '80.00'
    var_25 = [var_24]
    var_26 = [var_7, var_8, var_22]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_money_floordiv. Retrieved 13/53 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '3.00'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]
    var_11 = '5'
    var_12 = [var_11]
    var_13 = '0.00'
    var_14 = [var_13]
    var_15 = [var_1]
    var_16 = '-3'
    var_17 = [var_16]
    var_18 = '-4.00'
    var_19 = [var_18]
    var_20 = [var_11]
    var_21 = [var_1]
    var_22 = '0'
    var_23 = [var_22]
    var_24 = '-10'
    var_25 = [var_24]
    var_26 = [var_5]
    var_27 = [var_18]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_abs_defined_positive_money. Retrieved 4/15 statements.
# Partially parsed test_abs_defined_negative_money. Retrieved 5/16 statements.
# Partially parsed test_abs_defined_zero_money. Retrieved 6/16 statements.
# Failed to parse test_abs_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-10.50'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '10.50'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 6
    var_5 = 15
    var_6 = '0.00'
    var_7 = [var_6]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_money_truediv_with_defined_money. Retrieved 6/20 statements.
# Partially parsed test_money_truediv_with_undefined_money. Retrieved 1/6 statements.
# Partially parsed test_money_truediv_by_zero. Retrieved 5/15 statements.
# Partially parsed test_money_truediv_preserves_currency. Retrieved 6/20 statements.
# Partially parsed test_money_truediv_with_decimal_result. Retrieved 6/18 statements.
# Partially parsed test_money_truediv_with_one. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '5.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '20'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '4'
    var_6 = [var_5]
    var_7 = '5.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '3.33'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '15'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '15.00'
    var_8 = [var_7]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_divide_defined_price_by_positive_number. Retrieved 6/20 statements.
# Partially parsed test_divide_defined_price_by_negative_number. Retrieved 6/18 statements.
# Partially parsed test_divide_defined_price_by_zero. Retrieved 5/15 statements.
# Partially parsed test_divide_undefined_price. Retrieved 1/6 statements.
# Partially parsed test_divide_decimal_result. Retrieved 5/19 statements.
# Partially parsed test_divide_preserves_currency. Retrieved 6/20 statements.
# Partially parsed test_divide_by_one. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '5'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-2'
    var_6 = [var_5]
    var_7 = '-5'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = [var_5]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = '20'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '42'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]
    var_7 = [var_1]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_qty_or_else. Retrieved 22/63 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '42'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = True
    var_9 = lambda : var_8
    var_10 = [var_1]
    var_11 = None
    var_12 = [var_1]
    var_13 = [var_5]
    var_14 = [var_5]
    var_15 = False
    var_16 = lambda : var_15
    var_17 = 'EUR'
    var_18 = '0'
    var_19 = [var_18]
    var_20 = 2020
    var_21 = 6
    var_22 = 15
    var_23 = '100'
    var_24 = [var_23]
    var_25 = [var_18]
    var_26 = [var_18]
    var_27 = [var_18]
    var_28 = 'GBP'
    var_29 = '-5'
    var_30 = [var_29]
    var_31 = 2021
    var_32 = 3
    var_33 = 10
    var_34 = '10'
    var_35 = [var_34]
    var_36 = [var_29]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_add_with_same_currency. Retrieved 8/21 statements.
# Partially parsed test_add_with_different_currency_raises_error. Retrieved 8/22 statements.
# Partially parsed test_add_with_undefined_price. Retrieved 5/13 statements.
# Partially parsed test_add_takes_later_date. Retrieved 8/20 statements.
# Partially parsed test_add_with_negative_quantities. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '100.50'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = '50.25'
    var_9 = [var_8]
    var_10 = 2
    var_11 = [var_5, var_6, var_10]
    var_12 = '150.75'
    var_13 = [var_12]
    var_14 = [var_5, var_6, var_10]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = 'EUR'
    var_4 = [var_1]
    var_5 = '100.50'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = '50.25'
    var_11 = [var_10]
    var_12 = 2
    var_13 = [var_7, var_8, var_12]
    var_14 = bool(False)
    assert var_14 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '100.50'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = 5
    var_8 = [var_5, var_6, var_7]
    var_9 = '50.00'
    var_10 = [var_9]
    var_11 = 3
    var_12 = [var_5, var_6, var_11]
    var_13 = [var_5, var_6, var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = '-30.00'
    var_9 = [var_8]
    var_10 = [var_5, var_6, var_6]
    var_11 = '70.00'
    var_12 = [var_11]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_somemoney_sub. Retrieved 10/56 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = '30.00'
    var_9 = [var_8]
    var_10 = [var_5, var_6, var_6]
    var_11 = '70.00'
    var_12 = [var_11]
    var_13 = [var_5, var_6, var_6]
    var_14 = [var_3]
    var_15 = [var_5, var_6, var_6]
    var_16 = [var_8]
    var_17 = 15
    var_18 = [var_5, var_6, var_17]
    var_19 = [var_5, var_6, var_17]
    var_20 = [var_11]
    var_21 = [var_3]
    var_22 = [var_5, var_6, var_6]
    var_23 = [var_3]
    var_24 = [var_5, var_6, var_6]
    var_25 = '50.00'
    var_26 = [var_25]
    var_27 = [var_5, var_6, var_6]
    var_28 = bool(False)
    assert var_28 is True
    var_29 = 'IncompatibleCurrencyError'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_add_returns_someprice_with_sum_of_quantities_and_max_date. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 15
    var_3 = [var_0, var_1, var_2]
    var_4 = 20
    var_5 = [var_0, var_1, var_4]
    var_6 = '100.50'
    var_7 = [var_6]
    var_8 = '50.25'
    var_9 = [var_8]
    var_10 = '150.75'
    var_11 = [var_10]



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_nonmoney_constructor.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_money_le_operator. Retrieved 6/46 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = [var_5]
    var_9 = '3'
    var_10 = [var_9]
    var_11 = [var_5]
    var_12 = [var_1]
    var_13 = [var_1]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_with_dov. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2020
    var_6 = 6
    var_7 = 15
    var_8 = '100.00'
    var_9 = [var_8]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_someprice_convert_with_valid_rate. Retrieved 12/38 statements.
# Partially parsed test_someprice_convert_uses_dov_as_default_asof. Retrieved 10/36 statements.
# Partially parsed test_someprice_convert_with_no_rate_strict_true. Retrieved 11/33 statements.
# Partially parsed test_someprice_convert_with_no_rate_strict_false. Retrieved 11/32 statements.
# Partially parsed test_someprice_convert_with_no_fx_rate_service. Retrieved 10/30 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = [var_7, var_8, var_8]
    var_11 = '0.92'
    var_12 = [var_11]
    var_13 = 2023
    var_14 = 1
    var_15 = [var_13, var_14, var_14]
    var_16 = '92'
    var_17 = [var_16]
    var_18 = [var_13, var_14, var_14]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 6
    var_7 = 15
    var_8 = [var_5, var_6, var_7]
    var_9 = '100'
    var_10 = [var_9]
    var_11 = '0.92'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 2023
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = True
    var_14 = bool(False)
    assert var_14 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 2023
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 2023
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Did you implement and set the default FX rate service?'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_as_boolean. Retrieved 7/33 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '-1'
    var_8 = [var_7]
    var_9 = '0.01'
    var_10 = [var_9]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_divide_with_defined_money. Retrieved 6/17 statements.
# Partially parsed test_divide_with_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_divide_by_zero_returns_undefined. Retrieved 5/14 statements.
# Partially parsed test_divide_with_decimal_result. Retrieved 6/17 statements.
# Partially parsed test_divide_with_negative_divisor. Retrieved 6/17 statements.
# Partially parsed test_divide_preserves_currency. Retrieved 5/16 statements.
# Partially parsed test_divide_preserves_date. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '50'
    var_8 = [var_7]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '33.33'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-2'
    var_6 = [var_5]
    var_7 = '-50'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '4'
    var_6 = [var_5]

def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = 'USD'
    var_3 = '100'
    var_4 = [var_3]
    var_5 = '5'
    var_6 = [var_5]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_price_gt. Retrieved 5/31 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = [var_1]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_qty_or_none. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = None
    var_8 = [var_1]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_dov_or. Retrieved 15/42 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2001
    var_6 = None
    var_7 = 2005
    var_8 = 6
    var_9 = 15
    var_10 = 'EUR'
    var_11 = '100'
    var_12 = [var_11]
    var_13 = 2020
    var_14 = 12
    var_15 = 25
    var_16 = 2000



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_price_abs. Retrieved 6/37 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '-10'
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = '0'
    var_10 = [var_9]
    var_11 = [var_9]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_round_defined_money. Retrieved 6/17 statements.
# Partially parsed test_round_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_round_zero_digits. Retrieved 6/16 statements.
# Partially parsed test_round_negative_quantity. Retrieved 6/16 statements.
# Partially parsed test_round_half_even_method. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.456'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = '1.46'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.6'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 0
    var_6 = '2'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-1.456'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = '-1.46'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '2.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 0
    var_6 = '2'
    var_7 = [var_6]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_someprice_gt. Retrieved 7/37 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '50'
    var_8 = [var_7]
    var_9 = [var_4, var_5, var_5]
    var_10 = [var_2]
    var_11 = [var_4, var_5, var_5]
    var_12 = [var_2]
    var_13 = [var_4, var_5, var_5]
    var_14 = 'not a price'
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'IncompatibleCurrencyError'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_convert_predicate_line_11_evaluates_to_false. Retrieved 11/25 statements.


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
    var_9 = [var_7, var_8, var_8]
    var_10 = 2023
    var_11 = 1
    var_12 = [var_10, var_11, var_11]
    var_13 = False
    var_14 = bool(True)
    assert var_14 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_price_floordiv. Retrieved 10/49 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = '7'
    var_9 = [var_8]
    var_10 = '2'
    var_11 = [var_10]
    var_12 = [var_5]
    var_13 = [var_10]
    var_14 = [var_1]
    var_15 = '0'
    var_16 = [var_15]
    var_17 = '-10'
    var_18 = [var_17]
    var_19 = [var_5]
    var_20 = '-4'
    var_21 = [var_20]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_price_sub. Retrieved 9/55 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '7'
    var_8 = [var_7]
    var_9 = '5'
    var_10 = [var_9]
    var_11 = [var_5]
    var_12 = [var_1]
    var_13 = '-7'
    var_14 = [var_13]
    var_15 = [var_9]
    var_16 = '0'
    var_17 = [var_16]
    var_18 = [var_9]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_ccy_or_with_defined_money. Retrieved 5/15 statements.
# Partially parsed test_ccy_or_with_undefined_money. Retrieved 3/11 statements.
# Partially parsed test_ccy_or_with_none_currency_returns_default. Retrieved 3/11 statements.
# Partially parsed test_ccy_or_with_none_date_returns_default. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = None
    var_2 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = None
    var_4 = 'EUR'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_qty_or_else. Retrieved 17/58 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '42'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = [var_1]
    var_9 = True
    var_10 = lambda : var_9
    var_11 = [var_1]
    var_12 = None
    var_13 = [var_1]
    var_14 = [var_5]
    var_15 = [var_5]
    var_16 = [var_1]
    var_17 = False
    var_18 = lambda : var_17
    var_19 = '0'
    var_20 = [var_19]
    var_21 = '100'
    var_22 = [var_21]
    var_23 = [var_21]
    var_24 = 'EUR'
    var_25 = [var_19]
    var_26 = 2020
    var_27 = 5
    var_28 = 15
    var_29 = '50'
    var_30 = [var_29]
    var_31 = [var_19]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_subtract_two_defined_prices_same_currency. Retrieved 7/24 statements.
# Partially parsed test_subtract_two_defined_prices_different_currency_raises_error. Retrieved 7/21 statements.
# Partially parsed test_subtract_defined_from_undefined_returns_defined. Retrieved 4/16 statements.
# Partially parsed test_subtract_undefined_from_defined_returns_defined. Retrieved 4/16 statements.
# Failed to parse test_subtract_two_undefined_prices_returns_undefined.
# Partially parsed test_subtract_negative_quantity. Retrieved 7/22 statements.
# Partially parsed test_subtract_zero_quantity. Retrieved 6/21 statements.
# Partially parsed test_subtract_carries_forward_date. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '7'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '3'
    var_7 = [var_6]
    var_8 = 2
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'IncompatibleCurrencyError'

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '10'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '-7'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = 2
    var_7 = '0'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 5
    var_6 = '3'
    var_7 = [var_6]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_positive_returns_same_money_when_defined. Retrieved 5/16 statements.
# Failed to parse test_positive_returns_itself_when_undefined.
# Partially parsed test_positive_with_negative_quantity. Retrieved 5/16 statements.
# Partially parsed test_positive_with_zero_quantity. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '100.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-50'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-50.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0.00'
    var_6 = [var_5]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dov_or_none. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = None



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_scalar_subtract. Retrieved 22/65 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '7.00'
    var_8 = [var_7]
    var_9 = [var_1]
    var_10 = '-5'
    var_11 = [var_10]
    var_12 = '15.00'
    var_13 = [var_12]
    var_14 = '5'
    var_15 = [var_14]
    var_16 = None
    var_17 = 'EUR'
    var_18 = '100'
    var_19 = [var_18]
    var_20 = 2020
    var_21 = 6
    var_22 = 15
    var_23 = '0'
    var_24 = [var_23]
    var_25 = '100.00'
    var_26 = [var_25]
    var_27 = 'GBP'
    var_28 = [var_14]
    var_29 = 2021
    var_30 = 3
    var_31 = 10
    var_32 = [var_1]
    var_33 = '-5.00'
    var_34 = [var_33]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_qty_map_defined_price. Retrieved 6/19 statements.
# Partially parsed test_qty_map_undefined_price. Retrieved 2/10 statements.
# Partially parsed test_qty_map_defined_price_with_different_function. Retrieved 7/20 statements.
# Partially parsed test_qty_map_undefined_price_calls_else_function. Retrieved 2/10 statements.
# Partially parsed test_qty_map_defined_price_with_string_function. Retrieved 8/16 statements.
# Partially parsed test_qty_map_undefined_price_with_string_else. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '42'
    var_7 = [var_6]
    var_8 = '2'
    var_9 = [var_8]

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = '42'
    var_3 = [var_2]
    var_4 = [var_2]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '0'
    var_8 = [var_7]
    var_9 = '10'
    var_10 = [var_9]

def test_case_0():
    var_0 = '100'
    var_1 = [var_0]
    var_2 = '99'
    var_3 = [var_2]
    var_4 = [var_2]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '3'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 5
    var_5 = 15
    var_6 = lambda x: str(x)
    var_7 = 'error'
    var_8 = lambda : var_7

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 'fallback'
    var_2 = lambda : var_1



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_price_lte. Retrieved 6/51 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = [var_5]
    var_9 = '3'
    var_10 = [var_9]
    var_11 = [var_5]
    var_12 = [var_1]
    var_13 = [var_1]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_lt_with_same_currency_less_than. Retrieved 6/17 statements.
# Partially parsed test_lt_with_same_currency_not_less_than. Retrieved 6/17 statements.
# Partially parsed test_lt_with_same_currency_equal. Retrieved 5/16 statements.
# Partially parsed test_lt_with_different_currency_raises_error. Retrieved 6/20 statements.
# Partially parsed test_lt_with_non_some_price_returns_false. Retrieved 6/14 statements.
# Partially parsed test_lt_with_no_price_returns_false. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '10.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = '20.00'
    var_9 = [var_8]
    var_10 = [var_5, var_6, var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '20.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = '10.00'
    var_9 = [var_8]
    var_10 = [var_5, var_6, var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '10.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = [var_3]
    var_9 = [var_5, var_6, var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = 'EUR'
    var_4 = [var_1]
    var_5 = '10.00'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = [var_5]
    var_11 = [var_7, var_8, var_8]
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '10.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = 'not a price'

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '10.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_fmap_with_defined_money. Retrieved 5/17 statements.
# Partially parsed test_fmap_with_undefined_money. Retrieved 2/10 statements.
# Partially parsed test_fmap_transforms_quantity. Retrieved 7/18 statements.
# Partially parsed test_fmap_transforms_date. Retrieved 7/19 statements.
# Partially parsed test_fmap_with_currency_change. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '2.00'
    var_7 = [var_6]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = [var_1]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 6
    var_5 = 15
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '10.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'GBP'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 30
    var_6 = 31
    var_7 = '100.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '50'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 6
    var_5 = 1
    var_6 = 'JPY'
    var_7 = '50.00'
    var_8 = [var_7]



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_price_add_two_defined_prices_same_currency. Retrieved 7/23 statements.
# Partially parsed test_price_add_defined_with_undefined. Retrieved 4/13 statements.
# Partially parsed test_price_add_undefined_with_defined. Retrieved 4/13 statements.
# Failed to parse test_price_add_two_undefined_prices.
# Partially parsed test_price_add_different_currencies_raises_error. Retrieved 6/19 statements.
# Partially parsed test_price_add_negative_quantities. Retrieved 7/21 statements.
# Partially parsed test_price_add_zero_quantities. Retrieved 5/19 statements.
# Partially parsed test_price_add_carries_forward_date. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '15'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '5'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'IncompatibleCurrencyError'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '7'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = 2
    var_7 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = 15



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_money_abs. Retrieved 8/31 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '10.00'
    var_6 = [var_5]
    var_7 = '-10'
    var_8 = [var_7]
    var_9 = [var_5]
    var_10 = '0'
    var_11 = [var_10]
    var_12 = '0.00'
    var_13 = [var_12]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_add_returns_someprice_with_correct_values. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = 2
    var_4 = [var_0, var_1, var_3]
    var_5 = '10.00'
    var_6 = [var_5]
    var_7 = '5.00'
    var_8 = [var_7]
    var_9 = '15.00'
    var_10 = [var_9]



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_price_negative. Retrieved 16/42 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-100'
    var_6 = [var_5]
    var_7 = 'EUR'
    var_8 = '-50'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 6
    var_12 = 15
    var_13 = '50'
    var_14 = [var_13]
    var_15 = 'GBP'
    var_16 = '0'
    var_17 = [var_16]
    var_18 = 2021
    var_19 = 3
    var_20 = 10
    var_21 = [var_16]



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_money_lte. Retrieved 9/50 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '50'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = 2
    var_9 = '30'
    var_10 = [var_9]
    var_11 = '70'
    var_12 = [var_11]
    var_13 = [var_5]
    var_14 = 'EUR'
    var_15 = [var_5]
    var_16 = bool(False)
    assert var_16 is True
    var_17 = 'IncompatibleCurrencyError'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_somemoney_sub. Retrieved 12/45 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_3, var_4, var_1]
    var_7 = '100.00'
    var_8 = [var_7]
    var_9 = '30.00'
    var_10 = [var_9]
    var_11 = '70.00'
    var_12 = [var_11]
    var_13 = '20.00'
    var_14 = [var_13]
    var_15 = '50.00'
    var_16 = [var_15]
    var_17 = '-30.00'
    var_18 = [var_17]
    var_19 = [var_7]
    var_20 = [var_7]
    var_21 = [var_15]
    var_22 = bool(False)
    assert var_22 is True
    var_23 = [var_7]
    var_24 = '0.00'
    var_25 = [var_24]
    var_26 = [var_7]



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_as_boolean. Retrieved 8/33 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '-1'
    var_8 = [var_7]
    var_9 = 'EUR'
    var_10 = '0.01'
    var_11 = [var_10]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_money_sub_method. Retrieved 11/62 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '7.00'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]
    var_11 = '5'
    var_12 = [var_11]
    var_13 = '-3.00'
    var_14 = [var_13]
    var_15 = [var_1]
    var_16 = [var_1]
    var_17 = '10.00'
    var_18 = [var_17]
    var_19 = [var_1]
    var_20 = 'EUR'
    var_21 = [var_1]
    var_22 = bool(False)
    assert var_22 is True
    var_23 = 'IncompatibleCurrencyError'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_money_neg. Retrieved 9/34 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-100.00'
    var_6 = [var_5]
    var_7 = '-50'
    var_8 = [var_7]
    var_9 = '50.00'
    var_10 = [var_9]
    var_11 = '0'
    var_12 = [var_11]
    var_13 = '0.00'
    var_14 = [var_13]



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_qty_or_none. Retrieved 10/31 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = None
    var_7 = [var_1]
    var_8 = 'EUR'
    var_9 = '42.50'
    var_10 = [var_9]
    var_11 = 2020
    var_12 = 6
    var_13 = 15
    var_14 = [var_9]



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_as_float. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '123.45'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'MonetaryOperationException'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_someprice_constructor. Retrieved 5/11 statements.
# Partially parsed test_someprice_constructor_with_different_currencies. Retrieved 5/11 statements.
# Partially parsed test_someprice_constructor_with_zero_quantity. Retrieved 5/11 statements.
# Partially parsed test_someprice_constructor_with_negative_quantity. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2024
    var_4 = 1
    var_5 = 15
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '250.75'
    var_2 = [var_1]
    var_3 = 2024
    var_4 = 6
    var_5 = 30
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'GBP'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2024
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'JPY'
    var_1 = '-500.25'
    var_2 = [var_1]
    var_3 = 2024
    var_4 = 3
    var_5 = 15
    var_6 = [var_3, var_4, var_5]



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_multiply_defined_money_positive_scalar. Retrieved 6/19 statements.
# Partially parsed test_multiply_defined_money_negative_scalar. Retrieved 6/17 statements.
# Partially parsed test_multiply_defined_money_zero_scalar. Retrieved 6/17 statements.
# Partially parsed test_multiply_defined_money_fractional_scalar. Retrieved 6/17 statements.
# Partially parsed test_multiply_undefined_money_returns_as_is. Retrieved 1/7 statements.
# Partially parsed test_multiply_defined_money_with_integer_scalar. Retrieved 7/19 statements.
# Partially parsed test_multiply_preserves_currency_and_date. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '20.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = '-30.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '0.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = '50.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = '5'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '25'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 6
    var_5 = 15
    var_6 = 4
    var_7 = '100.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2019
    var_1 = 3
    var_2 = 10
    var_3 = 'GBP'
    var_4 = '50'
    var_5 = [var_4]
    var_6 = '2'
    var_7 = [var_6]



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_subtract_defined_money_objects. Retrieved 6/22 statements.
# Partially parsed test_subtract_with_undefined_left_operand. Retrieved 4/13 statements.
# Partially parsed test_subtract_with_undefined_right_operand. Retrieved 4/13 statements.
# Failed to parse test_subtract_both_undefined.
# Partially parsed test_subtract_negative_result. Retrieved 6/20 statements.
# Partially parsed test_subtract_same_currency_different_dates. Retrieved 7/23 statements.
# Partially parsed test_subtract_zero_result. Retrieved 5/19 statements.
# Partially parsed test_subtract_incompatible_currencies_raises_error. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '7.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '10'
    var_6 = [var_5]
    var_7 = '-7.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = 15
    var_8 = '7.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '0.00'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '3'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'IncompatibleCurrencyError'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_dov_or_none. Retrieved 5/28 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = None
    var_6 = [var_1]



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_multiply_defined_money_by_positive_scalar. Retrieved 6/19 statements.
# Partially parsed test_multiply_defined_money_by_negative_scalar. Retrieved 6/17 statements.
# Partially parsed test_multiply_defined_money_by_zero. Retrieved 6/17 statements.
# Partially parsed test_multiply_defined_money_by_fractional_scalar. Retrieved 6/17 statements.
# Partially parsed test_multiply_undefined_money_returns_undefined. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '20.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = '-30.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '0.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = '5.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = '5'
    var_1 = [var_0]



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_convert_defined_price. Retrieved 5/16 statements.
# Partially parsed test_convert_undefined_price. Retrieved 1/7 statements.
# Partially parsed test_convert_with_asof_date. Retrieved 6/17 statements.
# Partially parsed test_convert_same_currency. Retrieved 4/15 statements.
# Partially parsed test_convert_strict_mode. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = 2019
    var_5 = 1

def test_case_0():
    var_0 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'GBP'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = 2019
    var_5 = 1
    var_6 = 6

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'JPY'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = 2019
    var_5 = 1
    var_6 = True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_ccy_or. Retrieved 13/36 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = None
    var_7 = [var_1]
    var_8 = 'GBP'
    var_9 = '100'
    var_10 = [var_9]
    var_11 = 2020
    var_12 = 5
    var_13 = 15
    var_14 = 'JPY'
    var_15 = 'CHF'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_dov_or. Retrieved 8/27 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2001
    var_6 = None
    var_7 = 2005
    var_8 = 5



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_price_add. Retrieved 11/64 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '15'
    var_9 = [var_8]
    var_10 = [var_1]
    var_11 = 'EUR'
    var_12 = '20'
    var_13 = [var_12]
    var_14 = [var_1]
    var_15 = '-3'
    var_16 = [var_15]
    var_17 = '7'
    var_18 = [var_17]
    var_19 = [var_1]
    var_20 = [var_5]
    var_21 = bool(False)
    assert var_21 is True
    var_22 = 'IncompatibleCurrencyError'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_floor_divide_with_defined_price. Retrieved 5/19 statements.
# Partially parsed test_floor_divide_with_undefined_price. Retrieved 1/6 statements.
# Partially parsed test_floor_divide_by_zero. Retrieved 5/15 statements.
# Partially parsed test_floor_divide_preserves_currency. Retrieved 7/21 statements.
# Partially parsed test_floor_divide_with_negative_divisor. Retrieved 6/18 statements.
# Partially parsed test_floor_divide_with_decimal_divisor. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = '3'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '20'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 6
    var_5 = 15
    var_6 = '4'
    var_7 = [var_6]
    var_8 = '5'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = '-4'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2.5'
    var_6 = [var_5]
    var_7 = '4'
    var_8 = [var_7]



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_with_qty. Retrieved 6/24 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = '10'
    var_9 = [var_8]



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_lt_defined_prices_same_currency. Retrieved 5/19 statements.
# Partially parsed test_lt_defined_prices_equal. Retrieved 4/17 statements.
# Partially parsed test_lt_undefined_vs_defined. Retrieved 4/14 statements.
# Partially parsed test_lt_defined_vs_undefined. Retrieved 4/14 statements.
# Failed to parse test_lt_undefined_vs_undefined.
# Partially parsed test_lt_incompatible_currencies. Retrieved 6/20 statements.


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
    var_5 = 'EUR'
    var_6 = '2'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'IncompatibleCurrencyError'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_dov_or_none. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = None



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_price_int_conversion. Retrieved 7/31 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '42.7'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-42.7'
    var_6 = [var_5]
    var_7 = '0'
    var_8 = [var_7]
    var_9 = '100'
    var_10 = [var_9]



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_money_eq. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = 'EUR'
    var_9 = [var_1]
    var_10 = [var_1]
    var_11 = 2



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_divide_defined_price_by_positive_number. Retrieved 6/20 statements.
# Partially parsed test_divide_defined_price_by_negative_number. Retrieved 6/18 statements.
# Partially parsed test_divide_defined_price_by_zero. Retrieved 5/15 statements.
# Partially parsed test_divide_undefined_price. Retrieved 1/5 statements.
# Partially parsed test_divide_defined_price_by_fraction. Retrieved 6/18 statements.
# Partially parsed test_divide_preserves_currency. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '5'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-2'
    var_6 = [var_5]
    var_7 = '-5'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = '20'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '4'
    var_6 = [var_5]
    var_7 = '25'
    var_8 = [var_7]



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_convert_with_valid_rate. Retrieved 12/35 statements.
# Partially parsed test_convert_uses_dov_when_asof_not_provided. Retrieved 11/32 statements.
# Partially parsed test_convert_with_no_rate_strict_mode. Retrieved 15/37 statements.
# Partially parsed test_convert_with_no_rate_non_strict_mode. Retrieved 12/33 statements.
# Partially parsed test_convert_no_service_raises_error. Retrieved 12/31 statements.
# Partially parsed test_convert_quantizes_result. Retrieved 12/34 statements.


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
    var_9 = [var_7, var_8, var_8]
    var_10 = None
    var_11 = 2023
    var_12 = 1
    var_13 = [var_11, var_12, var_12]
    var_14 = '85.00'
    var_15 = [var_14]
    var_16 = [var_11, var_12, var_12]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pound'
    var_5 = 2023
    var_6 = 6
    var_7 = 15
    var_8 = [var_5, var_6, var_7]
    var_9 = '100.00'
    var_10 = [var_9]
    var_11 = None
    var_12 = None

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 1
    var_10 = [var_8, var_9, var_9]
    var_11 = None
    var_12 = False
    var_13 = 2023
    var_14 = 1
    var_15 = [var_13, var_14, var_14]
    var_16 = True
    var_17 = True
    var_18 = bool(var_17)
    assert var_18 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'CHF'
    var_4 = 'Swiss Franc'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = None
    var_11 = 2023
    var_12 = 1
    var_13 = [var_11, var_12, var_12]
    var_14 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'CAD'
    var_4 = 'Canadian Dollar'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = False
    var_11 = 2023
    var_12 = 1
    var_13 = [var_11, var_12, var_12]
    var_14 = True
    var_15 = bool(var_14)
    assert var_15 is True

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
    var_9 = [var_7, var_8, var_8]
    var_10 = None
    var_11 = 2023
    var_12 = 1
    var_13 = [var_11, var_12, var_12]
    var_14 = '85.67'
    var_15 = [var_14]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_somemoney_neg. Retrieved 5/23 statements.
# Partially parsed test_somemoney_neg_negative_value. Retrieved 6/24 statements.
# Partially parsed test_somemoney_neg_zero. Retrieved 5/23 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '-100.50'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '-50.25'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 6
    var_5 = 15
    var_6 = [var_3, var_4, var_5]
    var_7 = '50.25'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'GBP'
    var_1 = '0.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 12
    var_5 = 31
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_1]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_scalar_add_with_defined_money. Retrieved 6/19 statements.
# Partially parsed test_scalar_add_with_undefined_money. Retrieved 1/5 statements.
# Partially parsed test_scalar_add_with_negative_scalar. Retrieved 6/17 statements.
# Partially parsed test_scalar_add_with_zero. Retrieved 6/17 statements.
# Partially parsed test_scalar_add_with_large_number. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = '15.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = '5'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = '7.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '10.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '999.50'
    var_6 = [var_5]
    var_7 = '1100.00'
    var_8 = [var_7]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_convert_same_currency. Retrieved 5/15 statements.
# Partially parsed test_convert_with_asof_date. Retrieved 6/17 statements.
# Partially parsed test_convert_undefined_money. Retrieved 1/6 statements.
# Partially parsed test_convert_strict_mode. Retrieved 6/15 statements.
# Partially parsed test_convert_different_currency. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '100.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = 6

def test_case_0():
    var_0 = 'USD'

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'GBP'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_price_gte. Retrieved 6/28 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '50'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = 2



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_qty_or_zero. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = None
    var_8 = [var_1]
    var_9 = '0'
    var_10 = [var_9]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_money_is_equal. Retrieved 10/44 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = 'EUR'
    var_7 = [var_1]
    var_8 = '200'
    var_9 = [var_8]
    var_10 = [var_1]
    var_11 = 2
    var_12 = 'not a money'
    var_13 = 100
    var_14 = None



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_with_dov. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2020
    var_6 = 6
    var_7 = 15
    var_8 = '100.00'
    var_9 = [var_8]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_money_add_two_defined_money_objects_with_same_currency. Retrieved 6/23 statements.
# Partially parsed test_money_add_defined_with_undefined_money. Retrieved 5/17 statements.
# Partially parsed test_money_add_undefined_with_defined_money. Retrieved 5/17 statements.
# Failed to parse test_money_add_two_undefined_money_objects.
# Partially parsed test_money_add_with_different_currencies_raises_error. Retrieved 6/21 statements.
# Partially parsed test_money_add_carries_forward_date. Retrieved 6/21 statements.
# Partially parsed test_money_add_negative_quantities. Retrieved 6/21 statements.
# Partially parsed test_money_add_zero_quantities. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '20'
    var_6 = [var_5]
    var_7 = '30.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '10.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '10.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '20'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '20'
    var_6 = [var_5]
    var_7 = 5

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-5'
    var_6 = [var_5]
    var_7 = '5.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '10'
    var_6 = [var_5]
    var_7 = '10.00'
    var_8 = [var_7]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_price_truediv. Retrieved 16/52 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '5'
    var_8 = [var_7]
    var_9 = [var_1]
    var_10 = '0'
    var_11 = [var_10]
    var_12 = [var_5]
    var_13 = 'EUR'
    var_14 = '20'
    var_15 = [var_14]
    var_16 = 6
    var_17 = 15
    var_18 = 4.0
    var_19 = [var_7]
    var_20 = 'GBP'
    var_21 = '15'
    var_22 = [var_21]
    var_23 = 3
    var_24 = 20
    var_25 = [var_7]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dov_or. Retrieved 9/28 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2001
    var_6 = None
    var_7 = 2020
    var_8 = 6
    var_9 = 15



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_someprice_le. Retrieved 8/31 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = 'EUR'
    var_4 = [var_1]
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = [var_5]
    var_11 = [var_7, var_8, var_8]
    var_12 = '150.00'
    var_13 = [var_12]
    var_14 = [var_7, var_8, var_8]
    var_15 = '50.00'
    var_16 = [var_15]
    var_17 = [var_7, var_8, var_8]
    var_18 = [var_5]
    var_19 = [var_7, var_8, var_8]
    var_20 = 'not a price'
    assert var_20 is False
    var_21 = bool(False)
    assert var_21 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_qty_map_defined_price. Retrieved 6/20 statements.
# Partially parsed test_qty_map_undefined_price. Retrieved 3/13 statements.
# Partially parsed test_qty_map_defined_price_with_different_function. Retrieved 7/21 statements.
# Partially parsed test_qty_map_undefined_price_calls_else_function. Retrieved 4/14 statements.
# Partially parsed test_qty_map_defined_price_with_string_return. Retrieved 7/16 statements.
# Partially parsed test_qty_map_undefined_price_with_string_return. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '42'
    var_7 = [var_6]
    var_8 = '2'
    var_9 = [var_8]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = [var_1]
    var_4 = '42'
    var_5 = [var_4]
    var_6 = [var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '0'
    var_8 = [var_7]
    var_9 = '20'
    var_10 = [var_9]

def test_case_0():
    var_0 = None
    var_1 = '5'
    var_2 = [var_1]
    var_3 = '10'
    var_4 = [var_3]
    var_5 = '99'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = lambda x: str(x)
    var_6 = 'fallback'
    var_7 = lambda : var_6

def test_case_0():
    var_0 = None
    var_1 = '5'
    var_2 = [var_1]
    var_3 = lambda x: str(x)
    var_4 = 'fallback'
    var_5 = lambda : var_4



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_money_truediv. Retrieved 14/45 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '50.00'
    var_8 = [var_7]
    var_9 = [var_1]
    var_10 = '0'
    var_11 = [var_10]
    var_12 = [var_5]
    var_13 = 'EUR'
    var_14 = '50'
    var_15 = [var_14]
    var_16 = 5
    var_17 = '10.00'
    var_18 = [var_17]
    var_19 = 'GBP'
    var_20 = [var_1]
    var_21 = 2.5
    var_22 = '40.00'
    var_23 = [var_22]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_lte_raises_incompatible_currency_error_when_currencies_differ. Retrieved 5/24 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = [var_2]
    var_8 = [var_4, var_5, var_5]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_somemoney_add. Retrieved 13/44 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = '50.00'
    var_9 = [var_8]
    var_10 = 15
    var_11 = [var_5, var_6, var_10]
    var_12 = '150.00'
    var_13 = [var_12]
    var_14 = [var_5, var_6, var_10]
    var_15 = '25.00'
    var_16 = [var_15]
    var_17 = [var_5, var_1, var_6]
    var_18 = '125.00'
    var_19 = [var_18]
    var_20 = [var_5, var_1, var_6]
    var_21 = '30.00'
    var_22 = [var_21]
    var_23 = [var_5, var_6, var_6]
    var_24 = '130.00'
    var_25 = [var_24]
    var_26 = [var_5, var_6, var_6]
    var_27 = [var_3]
    var_28 = [var_5, var_6, var_6]
    var_29 = bool(False)
    assert var_29 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_qty_or. Retrieved 17/45 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 0
    var_6 = [var_5]
    var_7 = '1.00'
    var_8 = [var_7]
    var_9 = None
    var_10 = [var_1]
    var_11 = [var_5]
    var_12 = '0'
    var_13 = [var_12]
    var_14 = 'EUR'
    var_15 = '42.50'
    var_16 = [var_15]
    var_17 = 2020
    var_18 = 6
    var_19 = 15
    var_20 = 100
    var_21 = [var_20]
    var_22 = [var_15]
    var_23 = '99'
    var_24 = [var_23]
    var_25 = 55
    var_26 = [var_25]
    var_27 = '55'
    var_28 = [var_27]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_price_pos. Retrieved 9/28 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = 'EUR'
    var_7 = '-50'
    var_8 = [var_7]
    var_9 = 2020
    var_10 = 6
    var_11 = 15
    var_12 = [var_7]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_as_integer. Retrieved 6/27 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '42.75'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-42.75'
    var_6 = [var_5]
    var_7 = '0'
    var_8 = [var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'MonetaryOperationException'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_as_boolean. Retrieved 11/37 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '-1'
    var_8 = [var_7]
    var_9 = 'EUR'
    var_10 = '0.01'
    var_11 = [var_10]
    var_12 = 2020
    var_13 = 6
    var_14 = 15



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_floordiv_with_valid_divisor. Retrieved 7/21 statements.
# Partially parsed test_floordiv_with_zero_divisor. Retrieved 6/18 statements.
# Partially parsed test_floordiv_with_decimal_divisor. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 3
    var_8 = '33.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '50.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = '33.00'
    var_10 = [var_9]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_price_float. Retrieved 14/36 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '123.45'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'MonetaryOperationException'
    var_7 = 'EUR'
    var_8 = '-50.75'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 6
    var_12 = 15
    var_13 = 'GBP'
    var_14 = '0'
    var_15 = [var_14]
    var_16 = 2021
    var_17 = 3
    var_18 = 10



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_qty_or_zero. Retrieved 12/34 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = None
    var_8 = [var_1]
    var_9 = '0'
    var_10 = [var_9]
    var_11 = 'EUR'
    var_12 = '42.50'
    var_13 = [var_12]
    var_14 = 2020
    var_15 = 6
    var_16 = 15
    var_17 = [var_12]
    var_18 = [var_9]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_lt_with_same_currency_less_than. Retrieved 6/17 statements.
# Partially parsed test_lt_with_same_currency_not_less_than. Retrieved 6/17 statements.
# Partially parsed test_lt_with_same_currency_equal_quantities. Retrieved 5/16 statements.
# Partially parsed test_lt_with_different_currency_raises_error. Retrieved 6/20 statements.
# Partially parsed test_lt_with_non_some_price_returns_false. Retrieved 6/14 statements.
# Partially parsed test_lt_with_no_price_returns_false. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = '200.00'
    var_9 = [var_8]
    var_10 = [var_5, var_6, var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '200.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = '100.00'
    var_9 = [var_8]
    var_10 = [var_5, var_6, var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = [var_3]
    var_9 = [var_5, var_6, var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = 'EUR'
    var_4 = [var_1]
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = [var_5]
    var_11 = [var_7, var_8, var_8]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = '< comparison'

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = 'not a price'

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_is_equal. Retrieved 10/46 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '200'
    var_7 = [var_6]
    var_8 = 'EUR'
    var_9 = [var_1]
    var_10 = [var_1]
    var_11 = 2
    var_12 = 'not a price'
    var_13 = None
    var_14 = 100



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_floor_divide_defined_price_positive_divisor. Retrieved 5/18 statements.
# Partially parsed test_floor_divide_defined_price_negative_divisor. Retrieved 6/17 statements.
# Partially parsed test_floor_divide_defined_price_by_one. Retrieved 5/16 statements.
# Partially parsed test_floor_divide_defined_price_by_zero. Retrieved 5/14 statements.
# Partially parsed test_floor_divide_undefined_price. Retrieved 1/4 statements.
# Partially parsed test_floor_divide_negative_quantity. Retrieved 6/17 statements.
# Partially parsed test_floor_divide_decimal_quantity. Retrieved 6/17 statements.
# Partially parsed test_floor_divide_preserves_currency. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = '-4'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]
    var_7 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = '5'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '-4'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '7.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '3'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '15'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '4'
    var_6 = [var_5]
    var_7 = '3'
    var_8 = [var_7]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_price_gte. Retrieved 6/28 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '50'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = 2



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_money_eq. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '200'
    var_7 = [var_6]
    var_8 = 'EUR'
    var_9 = [var_1]
    var_10 = [var_1]
    var_11 = 2



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_ccy_or_defined_money. Retrieved 5/15 statements.
# Partially parsed test_ccy_or_undefined_money. Retrieved 3/11 statements.
# Partially parsed test_ccy_or_with_none_currency. Retrieved 3/11 statements.
# Partially parsed test_ccy_or_with_none_quantity. Retrieved 5/14 statements.
# Partially parsed test_ccy_or_with_none_date. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = None
    var_2 = 'EUR'

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 'USD'

def test_case_0():
    var_0 = 'GBP'
    var_1 = None
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = 'JPY'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = None
    var_4 = 'CHF'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_as_boolean. Retrieved 11/37 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]
    var_7 = '-5'
    var_8 = [var_7]
    var_9 = 'EUR'
    var_10 = '0.01'
    var_11 = [var_10]
    var_12 = 2020
    var_13 = 6
    var_14 = 15



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_multiply_defined_price_by_positive_scalar. Retrieved 6/21 statements.
# Partially parsed test_multiply_defined_price_by_negative_scalar. Retrieved 6/19 statements.
# Partially parsed test_multiply_defined_price_by_zero. Retrieved 5/16 statements.
# Partially parsed test_multiply_defined_price_by_fractional_scalar. Retrieved 6/17 statements.
# Partially parsed test_multiply_undefined_price_returns_itself. Retrieved 1/5 statements.
# Partially parsed test_multiply_defined_price_by_integer. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '20'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = '-30'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0.5'
    var_6 = [var_5]
    var_7 = '5'
    var_8 = [var_7]

def test_case_0():
    var_0 = '5'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '25'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 6
    var_5 = 15
    var_6 = 4
    var_7 = '100'
    var_8 = [var_7]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_is_equal_defined_prices_same_values. Retrieved 4/17 statements.
# Partially parsed test_is_equal_defined_prices_different_quantities. Retrieved 5/18 statements.
# Partially parsed test_is_equal_defined_prices_different_currencies. Retrieved 5/18 statements.
# Partially parsed test_is_equal_defined_prices_different_dates. Retrieved 5/18 statements.
# Failed to parse test_is_equal_undefined_prices.
# Partially parsed test_is_equal_defined_and_undefined. Retrieved 4/14 statements.
# Partially parsed test_is_equal_with_non_price_object. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '200'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'not a price'
    var_6 = 100
    var_7 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_someprice_sub. Retrieved 24/52 statements.


def test_case_0():
    var_0 = 'Currency'
    var_1 = ()
    var_2 = '__eq__'
    var_3 = '__hash__'
    var_4 = True
    var_5 = lambda self, other: var_4
    var_6 = lambda self: var_4
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = ()
    var_10 = False
    var_11 = lambda self, other: var_10
    var_12 = 2
    var_13 = lambda self: var_12
    var_14 = {var_2: var_11, var_3: var_13}
    var_15 = [var_0, var_9, var_14]
    var_16 = '100.00'
    var_17 = [var_16]
    var_18 = 2023
    var_19 = [var_18, var_4, var_4]
    var_20 = '30.00'
    var_21 = [var_20]
    var_22 = [var_18, var_4, var_12]
    var_23 = '70.00'
    var_24 = [var_23]
    var_25 = [var_18, var_4, var_12]
    var_26 = 'NoPrice'
    var_27 = ()
    var_28 = 'undefined'
    var_29 = {var_28: var_4}
    var_30 = [var_26, var_27, var_29]
    var_31 = '40.00'
    var_32 = [var_31]
    var_33 = [var_18, var_4, var_4]
    var_34 = [var_18, var_4, var_4]
    var_35 = '50.00'
    var_36 = [var_35]
    var_37 = [var_18, var_4, var_4]
    var_38 = bool(False)
    assert var_38 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_price_or_else_with_defined_price. Retrieved 7/20 statements.
# Partially parsed test_price_or_else_with_undefined_price. Retrieved 5/16 statements.
# Partially parsed test_price_or_else_returns_self_when_defined. Retrieved 7/20 statements.
# Partially parsed test_price_or_else_calls_combinator_when_undefined. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '2'
    var_7 = [var_6]
    var_8 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = None
    var_6 = [var_1]

def test_case_0():
    var_0 = 'GBP'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 5
    var_5 = 15
    var_6 = 'USD'
    var_7 = '5'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'JPY'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2021
    var_4 = 3
    var_5 = 10



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_price_gt. Retrieved 6/29 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '50'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = 2



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_someprice_sub. Retrieved 12/42 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = 2
    var_6 = [var_2, var_3, var_5]
    var_7 = '100'
    var_8 = [var_7]
    var_9 = '30'
    var_10 = [var_9]
    var_11 = '50'
    var_12 = [var_11]
    var_13 = '20'
    var_14 = [var_13]
    var_15 = '70'
    var_16 = [var_15]
    var_17 = '80'
    var_18 = [var_17]
    var_19 = True
    var_20 = bool(False)
    assert var_20 is True
    var_21 = 'IncompatibleCurrencyError'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_price_gt. Retrieved 6/39 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '20'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = 2
    var_9 = [var_1]
    var_10 = [var_1]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_price_ge. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '50'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = 2



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_price_floordiv. Retrieved 20/58 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = '2'
    var_9 = [var_8]
    var_10 = 'EUR'
    var_11 = '5'
    var_12 = [var_11]
    var_13 = 2
    var_14 = '0'
    var_15 = [var_14]
    var_16 = 'GBP'
    var_17 = '7'
    var_18 = [var_17]
    var_19 = 3
    var_20 = '-2'
    var_21 = [var_20]
    var_22 = '-4'
    var_23 = [var_22]
    var_24 = 'JPY'
    var_25 = '100'
    var_26 = [var_25]
    var_27 = 4
    var_28 = '2.5'
    var_29 = [var_28]
    var_30 = '40'
    var_31 = [var_30]



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_money_truediv_defined_money_by_positive_numeric. Retrieved 6/19 statements.
# Partially parsed test_money_truediv_defined_money_by_negative_numeric. Retrieved 6/17 statements.
# Partially parsed test_money_truediv_defined_money_by_zero. Retrieved 5/14 statements.
# Partially parsed test_money_truediv_undefined_money. Retrieved 1/5 statements.
# Partially parsed test_money_truediv_defined_money_by_integer. Retrieved 6/16 statements.
# Partially parsed test_money_truediv_defined_money_by_float. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '50.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-2'
    var_6 = [var_5]
    var_7 = '-50.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 4
    var_6 = '25.00'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2.5
    var_6 = '40.00'
    var_7 = [var_6]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_negative. Retrieved 13/39 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-100.00'
    var_6 = [var_5]
    var_7 = 'EUR'
    var_8 = '-50'
    var_9 = [var_8]
    var_10 = 15
    var_11 = '50.00'
    var_12 = [var_11]
    var_13 = 'GBP'
    var_14 = '0'
    var_15 = [var_14]
    var_16 = 2
    var_17 = '0.00'
    var_18 = [var_17]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_money_float_conversion. Retrieved 14/36 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '123.45'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'MonetaryOperationException'
    var_7 = 'EUR'
    var_8 = '-50.25'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 6
    var_12 = 15
    var_13 = 'GBP'
    var_14 = '0'
    var_15 = [var_14]
    var_16 = 2021
    var_17 = 12
    var_18 = 31



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_as_integer. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '42.7'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'MonetaryOperationException'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_somemoney_ge. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = '50.00'
    var_9 = [var_8]
    var_10 = [var_6]
    var_11 = [var_6]
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_qty_map. Retrieved 27/75 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '42'
    var_7 = [var_6]
    var_8 = '2'
    var_9 = [var_8]
    var_10 = None
    var_11 = [var_1]
    var_12 = [var_1]
    var_13 = [var_6]
    var_14 = [var_6]
    var_15 = 'EUR'
    var_16 = '5'
    var_17 = [var_16]
    var_18 = 2020
    var_19 = 6
    var_20 = 15
    var_21 = [var_8]
    var_22 = '0'
    var_23 = [var_22]
    var_24 = '10'
    var_25 = [var_24]
    var_26 = '100'
    var_27 = [var_26]
    var_28 = [var_8]
    var_29 = '999'
    var_30 = [var_29]
    var_31 = [var_29]
    var_32 = 'GBP'
    var_33 = '7'
    var_34 = [var_33]
    var_35 = 2021
    var_36 = 3
    var_37 = 10
    var_38 = lambda x: str(x)
    var_39 = 'fallback'
    var_40 = lambda : var_39
    var_41 = '50'
    var_42 = [var_41]
    var_43 = lambda x: str(x)
    var_44 = lambda : var_39



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_someprice_ge_with_same_currency_and_greater_quantity. Retrieved 5/15 statements.
# Partially parsed test_someprice_ge_with_same_currency_and_equal_quantity. Retrieved 4/14 statements.
# Partially parsed test_someprice_ge_with_same_currency_and_lesser_quantity. Retrieved 5/15 statements.
# Partially parsed test_someprice_ge_with_different_currency_raises_error. Retrieved 6/18 statements.
# Partially parsed test_someprice_ge_with_non_someprice_returns_true. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '50'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '50'
    var_2 = [var_1]
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '100'
    var_7 = [var_6]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '50'
    var_8 = [var_7]
    var_9 = [var_4, var_5, var_5]
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'not a price'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_dimap_with_defined_price. Retrieved 7/15 statements.
# Partially parsed test_dimap_with_undefined_price. Retrieved 5/10 statements.
# Partially parsed test_dimap_applies_function_to_defined_price. Retrieved 7/20 statements.
# Partially parsed test_dimap_calls_else_combinator_for_undefined_price. Retrieved 4/14 statements.
# Partially parsed test_dimap_with_date_extraction_from_defined_price. Retrieved 8/19 statements.
# Partially parsed test_dimap_with_date_extraction_from_undefined_price. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = lambda x: x.ccy.code
    var_6 = 'EUR'
    var_7 = lambda : var_6

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = lambda x: x.ccy.code
    var_4 = 'EUR'
    var_5 = lambda : var_4

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '0'
    var_8 = [var_7]
    var_9 = '200'
    var_10 = [var_9]

def test_case_0():
    var_0 = None
    var_1 = '50'
    var_2 = [var_1]
    var_3 = '10'
    var_4 = [var_3]
    var_5 = '999'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 6
    var_5 = 15
    var_6 = lambda x: x.dov
    var_7 = 2000
    var_8 = 1

def test_case_0():
    var_0 = None
    var_1 = '5'
    var_2 = [var_1]
    var_3 = lambda x: x.dov
    var_4 = 2000
    var_5 = 1



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_somemoney_constructor. Retrieved 10/29 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = 2024
    var_4 = 1
    var_5 = 15
    var_6 = [var_3, var_4, var_5]
    var_7 = '100.50'
    var_8 = [var_7]
    var_9 = [var_7]
    var_10 = '50.00'
    var_11 = [var_10]
    var_12 = [var_10]
    var_13 = '0'
    var_14 = [var_13]
    var_15 = [var_13]
    var_16 = '-25.75'
    var_17 = [var_16]
    var_18 = [var_16]
    var_19 = [var_7]



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_price_negative. Retrieved 16/48 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-100'
    var_6 = [var_5]
    var_7 = 'EUR'
    var_8 = '-50'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 6
    var_12 = 15
    var_13 = '50'
    var_14 = [var_13]
    var_15 = 'GBP'
    var_16 = '0'
    var_17 = [var_16]
    var_18 = 2021
    var_19 = 3
    var_20 = 10
    var_21 = [var_16]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_money_floordiv. Retrieved 15/56 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '3.00'
    var_8 = [var_7]
    var_9 = [var_1]
    var_10 = '-3'
    var_11 = [var_10]
    var_12 = '-4.00'
    var_13 = [var_12]
    var_14 = '7.5'
    var_15 = [var_14]
    var_16 = '2.5'
    var_17 = [var_16]
    var_18 = [var_7]
    var_19 = '5'
    var_20 = [var_19]
    var_21 = [var_1]
    var_22 = '0'
    var_23 = [var_22]
    var_24 = 'EUR'
    var_25 = '20'
    var_26 = [var_25]
    var_27 = '6'
    var_28 = [var_27]
    var_29 = [var_7]



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_lte_raises_incompatible_currency_error_when_currencies_differ. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = 'EUR'
    var_4 = [var_1]
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = '100.00'
    var_9 = [var_8]
    var_10 = [var_8]
    var_11 = bool(False)
    assert var_11 is True
    var_12 = '<= comparison'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_someprice_lt. Retrieved 8/52 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '200'
    var_8 = [var_7]
    var_9 = [var_4, var_5, var_5]
    var_10 = '300'
    var_11 = [var_10]
    var_12 = [var_4, var_5, var_5]
    var_13 = [var_7]
    var_14 = [var_4, var_5, var_5]
    var_15 = [var_7]
    var_16 = [var_4, var_5, var_5]
    var_17 = [var_7]
    var_18 = [var_4, var_5, var_5]
    var_19 = [var_2]
    var_20 = [var_4, var_5, var_5]
    var_21 = 'not a price'
    var_22 = [var_2]
    var_23 = [var_4, var_5, var_5]
    var_24 = [var_2]
    var_25 = [var_4, var_5, var_5]
    var_26 = bool(False)
    assert var_26 is True
    var_27 = 'IncompatibleCurrencyError'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_price_bool. Retrieved 15/41 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = 'EUR'
    var_8 = '42.5'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 6
    var_12 = 15
    var_13 = 'GBP'
    var_14 = '-10.25'
    var_15 = [var_14]
    var_16 = 2021
    var_17 = 3
    var_18 = 20



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_truediv_valid_division. Retrieved 9/22 statements.
# Partially parsed test_truediv_division_by_zero. Retrieved 8/19 statements.
# Partially parsed test_truediv_with_integer. Retrieved 9/21 statements.
# Partially parsed test_truediv_with_float. Retrieved 9/23 statements.
# Partially parsed test_truediv_preserves_date. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'Currency'
    var_1 = ()
    var_2 = 'quantizer'
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = '100'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = '2'
    var_11 = [var_10]
    var_12 = '50'
    var_13 = [var_12]
    var_14 = [var_7, var_8, var_8]

def test_case_0():
    var_0 = 'Currency'
    var_1 = ()
    var_2 = 'quantizer'
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = '100'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = '0'
    var_11 = [var_10]

def test_case_0():
    var_0 = 'Currency'
    var_1 = ()
    var_2 = 'quantizer'
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = '100'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 4
    var_11 = '25'
    var_12 = [var_11]
    var_13 = [var_7, var_8, var_8]

def test_case_0():
    var_0 = 'Currency'
    var_1 = ()
    var_2 = 'quantizer'
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = '100'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = 2.5
    var_11 = [var_5]
    var_12 = '2.5'
    var_13 = [var_12]
    var_14 = [var_7, var_8, var_8]

def test_case_0():
    var_0 = 'Currency'
    var_1 = ()
    var_2 = 'quantizer'
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 6
    var_7 = 15
    var_8 = [var_5, var_6, var_7]
    var_9 = '150'
    var_10 = [var_9]
    var_11 = '3'
    var_12 = [var_11]
    var_13 = '50'
    var_14 = [var_13]



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_somemoney_constructor. Retrieved 6/21 statements.
# Partially parsed test_somemoney_constructor_with_different_currencies. Retrieved 6/21 statements.
# Partially parsed test_somemoney_constructor_with_zero_quantity. Retrieved 5/21 statements.
# Partially parsed test_somemoney_constructor_with_negative_quantity. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.50'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = 15
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 'EUR'
    var_1 = 2
    var_2 = '50.25'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 6
    var_6 = 30
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 'GBP'
    var_1 = 2
    var_2 = '0.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = [var_2]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 0
    var_2 = '-1000'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 3
    var_6 = 15
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_2]



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_money_eq. Retrieved 6/27 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = 'EUR'
    var_9 = [var_1]



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_convert_with_valid_rate. Retrieved 13/38 statements.
# Partially parsed test_convert_with_undefined_rate_non_strict. Retrieved 12/33 statements.
# Partially parsed test_convert_with_undefined_rate_strict. Retrieved 14/35 statements.
# Partially parsed test_convert_without_explicit_asof_uses_dov. Retrieved 13/36 statements.
# Partially parsed test_convert_no_default_fx_rate_service. Retrieved 10/28 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = None
    var_11 = 2023
    var_12 = 1
    var_13 = [var_11, var_12, var_12]
    var_14 = False
    var_15 = '92.00'
    var_16 = [var_15]
    var_17 = [var_11, var_12, var_12]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pounds'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = None
    var_11 = 2023
    var_12 = 1
    var_13 = [var_11, var_12, var_12]
    var_14 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pounds'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = None
    var_11 = False
    var_12 = 2023
    var_13 = 1
    var_14 = [var_12, var_13, var_13]
    var_15 = True
    var_16 = True
    var_17 = bool(var_16)
    assert var_17 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 6
    var_8 = 15
    var_9 = [var_6, var_7, var_8]
    var_10 = '100.00'
    var_11 = [var_10]
    var_12 = None
    var_13 = False
    var_14 = '13050'
    var_15 = [var_14]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2023
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = False
    var_11 = True
    var_12 = 'Did you implement and set the default FX rate service?'
    var_13 = bool(var_11)
    assert var_13 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_lt_defined_money_with_smaller_quantity. Retrieved 5/18 statements.
# Partially parsed test_lt_defined_money_with_larger_quantity. Retrieved 5/18 statements.
# Partially parsed test_lt_defined_money_with_equal_quantity. Retrieved 4/17 statements.
# Partially parsed test_lt_undefined_money_with_defined_money. Retrieved 4/14 statements.
# Partially parsed test_lt_defined_money_with_undefined_money. Retrieved 4/14 statements.
# Failed to parse test_lt_undefined_money_with_undefined_money.
# Partially parsed test_lt_different_currencies_raises_error. Retrieved 5/19 statements.


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
    var_5 = 'EUR'
    var_6 = [var_1]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'IncompatibleCurrencyError'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_someprice_add. Retrieved 11/36 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = 'EUR'
    var_4 = [var_1]
    var_5 = '100.00'
    var_6 = [var_5]
    var_7 = 2024
    var_8 = 1
    var_9 = [var_7, var_8, var_8]
    var_10 = '50.00'
    var_11 = [var_10]
    var_12 = 15
    var_13 = [var_7, var_8, var_12]
    var_14 = '75.00'
    var_15 = [var_14]
    var_16 = 10
    var_17 = [var_7, var_8, var_16]
    var_18 = '150.00'
    var_19 = [var_18]
    var_20 = [var_7, var_8, var_12]
    var_21 = [var_18]
    var_22 = [var_7, var_8, var_12]
    var_23 = bool(False)
    assert var_23 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_price_abs. Retrieved 7/40 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '-10'
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = 'EUR'
    var_10 = '0'
    var_11 = [var_10]
    var_12 = [var_10]



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_price_ge. Retrieved 6/29 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '50'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = 2



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_qty_or_else_with_defined_price. Retrieved 5/16 statements.
# Partially parsed test_qty_or_else_with_defined_price_returns_qty_not_combinator. Retrieved 6/15 statements.
# Partially parsed test_qty_or_else_with_undefined_price_returns_combinator_result. Retrieved 3/11 statements.
# Partially parsed test_qty_or_else_with_undefined_price_returns_non_decimal_combinator. Retrieved 4/9 statements.
# Partially parsed test_qty_or_else_with_undefined_price_calls_combinator. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '42'
    var_6 = [var_5]
    var_7 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = True
    var_6 = lambda : var_5
    var_7 = [var_1]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = '42'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = False
    var_4 = lambda : var_3

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = []
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = '100'
    var_6 = [var_5]



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_ccy_or_with_defined_price. Retrieved 5/15 statements.
# Partially parsed test_ccy_or_with_undefined_price. Retrieved 3/11 statements.
# Partially parsed test_ccy_or_returns_default_when_price_undefined. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 'EUR'

def test_case_0():
    var_0 = 'GBP'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_somemoney_gt. Retrieved 7/25 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '100.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = '50.00'
    var_9 = [var_8]
    var_10 = [var_5, var_6, var_6]
    var_11 = [var_3]
    var_12 = [var_5, var_6, var_6]
    var_13 = [var_3]
    var_14 = [var_5, var_6, var_6]
    var_15 = 'not_money'
    assert var_15 is True
    var_16 = 100
    assert var_16 is True
    var_17 = None
    assert var_17 is True
    var_18 = bool(False)
    assert var_18 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_as_boolean. Retrieved 19/45 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '100.50'
    var_7 = [var_6]
    var_8 = 2020
    var_9 = 6
    var_10 = 15
    var_11 = 'GBP'
    var_12 = '-50.25'
    var_13 = [var_12]
    var_14 = 2021
    var_15 = 3
    var_16 = 10
    var_17 = 'JPY'
    var_18 = '0'
    var_19 = [var_18]
    var_20 = 2022
    var_21 = 12
    var_22 = 25



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_floordiv_with_valid_numeric. Retrieved 7/18 statements.
# Partially parsed test_floordiv_with_decimal. Retrieved 8/19 statements.
# Partially parsed test_floordiv_with_zero_returns_no_price. Retrieved 7/15 statements.
# Partially parsed test_floordiv_with_negative_divisor. Retrieved 8/19 statements.
# Partially parsed test_floordiv_with_float. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '10.5'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = 3
    var_9 = '3'
    var_10 = [var_9]
    var_11 = [var_5, var_6, var_6]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '20'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 6
    var_7 = 15
    var_8 = [var_5, var_6, var_7]
    var_9 = '7'
    var_10 = [var_9]
    var_11 = '2'
    var_12 = [var_11]

def test_case_0():
    var_0 = 'GBP'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '100'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 3
    var_7 = 20
    var_8 = [var_5, var_6, var_7]
    var_9 = 0

def test_case_0():
    var_0 = 'JPY'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = '50'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 12
    var_7 = 1
    var_8 = [var_5, var_6, var_7]
    var_9 = -2
    var_10 = '-25'
    var_11 = [var_10]
    var_12 = [var_5, var_6, var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = '15.7'
    var_4 = [var_3]
    var_5 = 2023
    var_6 = 5
    var_7 = 10
    var_8 = [var_5, var_6, var_7]
    var_9 = 2.5
    var_10 = '6'
    var_11 = [var_10]



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_positive_returns_same_price_when_defined. Retrieved 4/14 statements.
# Failed to parse test_positive_returns_itself_when_undefined.
# Partially parsed test_positive_returns_same_price_with_negative_quantity. Retrieved 5/15 statements.
# Partially parsed test_positive_returns_same_price_with_zero_quantity. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '-5'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 6
    var_5 = 15
    var_6 = [var_1]

def test_case_0():
    var_0 = 'GBP'
    var_1 = '0'
    var_2 = [var_1]
    var_3 = 2021
    var_4 = 3
    var_5 = 20
    var_6 = [var_1]



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_with_qty. Retrieved 10/31 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = '5.00'
    var_8 = [var_7]
    var_9 = '10'
    var_10 = [var_9]
    var_11 = '0'
    var_12 = [var_11]
    var_13 = '0.00'
    var_14 = [var_13]
    var_15 = '-3.50'
    var_16 = [var_15]
    var_17 = [var_15]



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_dimap_with_defined_money. Retrieved 7/16 statements.
# Partially parsed test_dimap_with_undefined_money. Retrieved 3/6 statements.
# Partially parsed test_dimap_with_defined_money_numeric_result. Retrieved 7/21 statements.
# Partially parsed test_dimap_with_undefined_money_numeric_fallback. Retrieved 2/9 statements.
# Partially parsed test_dimap_applies_function_to_defined_money. Retrieved 8/20 statements.
# Partially parsed test_dimap_calls_combinator_on_undefined_money. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = lambda x: x.ccy.code
    var_6 = 'EUR'
    var_7 = lambda : var_6

def test_case_0():
    var_0 = lambda x: x.ccy.code
    var_1 = 'EUR'
    var_2 = lambda : var_1

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '0'
    var_8 = [var_7]
    var_9 = '200'
    var_10 = [var_9]

def test_case_0():
    var_0 = lambda x: x.qty
    var_1 = '42'
    var_2 = [var_1]
    var_3 = [var_1]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '50'
    var_2 = [var_1]
    var_3 = 2020
    var_4 = 6
    var_5 = 15
    var_6 = lambda x: (x.ccy.code, x.qty, x.dov)
    var_7 = None
    var_8 = lambda : var_7
    var_9 = [var_1]

def test_case_0():
    var_0 = []
    var_1 = 'should_not_be_called'
    var_2 = lambda x: var_1
    var_3 = len(var_0)
    assert var_3 == 1



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_money_eq. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '200'
    var_7 = [var_6]
    var_8 = 'EUR'
    var_9 = [var_1]
    var_10 = [var_1]
    var_11 = 2



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_subtract. Retrieved 9/50 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '7.00'
    var_8 = [var_7]
    var_9 = [var_5]
    var_10 = [var_1]
    var_11 = '-7.00'
    var_12 = [var_11]
    var_13 = '5'
    var_14 = [var_13]
    var_15 = [var_1]
    var_16 = 'EUR'
    var_17 = [var_13]
    var_18 = bool(False)
    assert var_18 is True
    var_19 = 'IncompatibleCurrencyError'



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_subtract_defined_money_same_currency. Retrieved 7/23 statements.
# Partially parsed test_subtract_defined_money_different_currency. Retrieved 7/20 statements.
# Partially parsed test_subtract_undefined_left_operand. Retrieved 5/14 statements.
# Partially parsed test_subtract_undefined_right_operand. Retrieved 4/13 statements.
# Failed to parse test_subtract_both_undefined.
# Partially parsed test_subtract_negative_result. Retrieved 7/21 statements.
# Partially parsed test_subtract_zero_result. Retrieved 6/20 statements.
# Partially parsed test_subtract_carries_forward_date. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '7.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '3'
    var_7 = [var_6]
    var_8 = 2
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'IncompatibleCurrencyError'

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
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

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '10'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '-7.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = 2
    var_7 = '0.00'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = 15



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_dov_or_none. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = None



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_price_neg. Retrieved 10/38 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-100'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = 'EUR'
    var_9 = '0'
    var_10 = [var_9]
    var_11 = 2020
    var_12 = 6
    var_13 = 15
    var_14 = [var_9]



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_with_dov. Retrieved 9/26 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2020
    var_6 = 6
    var_7 = 15
    var_8 = '1.00'
    var_9 = [var_8]
    var_10 = None
    var_11 = [var_1]



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_truediv_divides_quantity_by_numeric. Retrieved 6/15 statements.
# Partially parsed test_truediv_quantizes_result. Retrieved 7/16 statements.
# Partially parsed test_truediv_by_zero_returns_no_money. Retrieved 6/13 statements.
# Partially parsed test_truediv_with_float. Retrieved 7/15 statements.
# Partially parsed test_truediv_with_string_numeric. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '50.00'
    var_8 = [var_7]
    var_9 = [var_4, var_5, var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '3'
    var_8 = [var_7]
    var_9 = '33.33'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 2.5
    var_8 = '40.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '100.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '4'
    var_8 = '25.00'
    var_9 = [var_8]



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_price_int. Retrieved 6/27 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '42'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '42.75'
    var_6 = [var_5]
    var_7 = '-42.5'
    var_8 = [var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'MonetaryOperationException'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_floor_divide_defined_price. Retrieved 5/19 statements.
# Partially parsed test_floor_divide_undefined_price. Retrieved 1/6 statements.
# Partially parsed test_floor_divide_by_zero. Retrieved 5/15 statements.
# Partially parsed test_floor_divide_with_negative_divisor. Retrieved 6/18 statements.
# Partially parsed test_floor_divide_with_fractional_divisor. Retrieved 6/18 statements.
# Partially parsed test_floor_divide_preserves_currency. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = '3'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-3'
    var_6 = [var_5]
    var_7 = '-4'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2.5'
    var_6 = [var_5]
    var_7 = '4'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '20'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 6
    var_5 = 15
    var_6 = '4'
    var_7 = [var_6]



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_gt_defined_money_greater_than_undefined. Retrieved 4/13 statements.
# Partially parsed test_gt_undefined_money_not_greater_than_defined. Retrieved 4/13 statements.
# Failed to parse test_gt_undefined_money_not_greater_than_undefined.
# Partially parsed test_gt_defined_money_greater_than_defined_same_currency. Retrieved 5/17 statements.
# Partially parsed test_gt_defined_money_not_greater_than_defined_same_currency. Retrieved 5/17 statements.
# Partially parsed test_gt_defined_money_equal_quantities_not_greater. Retrieved 4/16 statements.
# Partially parsed test_gt_defined_money_different_currency_raises_error. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '20'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '10'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '20'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '20'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '10'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_price_add_with_same_currency. Retrieved 6/23 statements.
# Partially parsed test_price_add_with_different_currency_raises_error. Retrieved 6/20 statements.
# Partially parsed test_price_add_with_undefined_operand. Retrieved 4/16 statements.
# Failed to parse test_price_add_both_undefined.
# Partially parsed test_price_add_carries_forward_date. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = '15'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '5'
    var_7 = [var_6]
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'IncompatibleCurrencyError'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '5'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '15'
    var_9 = [var_8]



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_scalar_subtract. Retrieved 13/55 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '3'
    var_6 = [var_5]
    var_7 = '7'
    var_8 = [var_7]
    var_9 = [var_1]
    var_10 = '0'
    var_11 = [var_10]
    var_12 = [var_1]
    var_13 = '5'
    var_14 = [var_13]
    var_15 = [var_1]
    var_16 = '-5'
    var_17 = [var_16]
    var_18 = [var_13]
    var_19 = 'EUR'
    var_20 = '10.75'
    var_21 = [var_20]
    var_22 = '2.25'
    var_23 = [var_22]
    var_24 = '8.50'
    var_25 = [var_24]



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_dov_or. Retrieved 16/44 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2001
    var_6 = None
    var_7 = 'EUR'
    var_8 = '100'
    var_9 = [var_8]
    var_10 = 2020
    var_11 = 6
    var_12 = 15
    var_13 = 2000
    var_14 = '50'
    var_15 = [var_14]
    var_16 = 2015
    var_17 = 12
    var_18 = 31



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_lt_with_same_currency_less_than. Retrieved 6/15 statements.
# Partially parsed test_lt_with_same_currency_not_less_than. Retrieved 6/15 statements.
# Partially parsed test_lt_with_same_currency_equal. Retrieved 5/14 statements.
# Partially parsed test_lt_with_non_somemoney_object. Retrieved 5/11 statements.
# Partially parsed test_lt_with_different_currencies_raises_error. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '20.00'
    var_8 = [var_7]
    var_9 = [var_4, var_5, var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '20.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = '10.00'
    var_8 = [var_7]
    var_9 = [var_4, var_5, var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = [var_2]
    var_8 = [var_4, var_5, var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10.00'
    var_3 = [var_2]
    var_4 = 2024
    var_5 = 1
    var_6 = [var_4, var_5, var_5]
    var_7 = 'not money'
    assert var_7 is False

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = '10.00'
    var_4 = [var_3]
    var_5 = 2024
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = '20.00'
    var_9 = [var_8]
    var_10 = [var_5, var_6, var_6]
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_money_lt. Retrieved 5/50 statements.


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
    var_9 = [var_1]
    var_10 = [var_1]
    var_11 = [var_1]
    var_12 = [var_1]



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_round_defined_money. Retrieved 6/17 statements.
# Partially parsed test_round_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_round_default_ndigits. Retrieved 5/15 statements.
# Partially parsed test_round_negative_quantity. Retrieved 6/16 statements.
# Partially parsed test_round_zero_quantity. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.456'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = '1.46'
    var_7 = [var_6]

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.567'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2.00'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-1.456'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = '-1.46'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.0'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 2
    var_6 = '0.00'
    var_7 = [var_6]



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_qty_or. Retrieved 11/45 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = 0
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = None
    var_9 = [var_1]
    var_10 = [var_5]
    var_11 = '0'
    var_12 = [var_11]
    var_13 = '42'
    var_14 = [var_13]
    var_15 = '100'
    var_16 = [var_15]
    var_17 = [var_13]
    var_18 = [var_1]
    var_19 = '99'
    var_20 = [var_19]
    var_21 = [var_19]
    var_22 = '-5'
    var_23 = [var_22]
    var_24 = [var_11]
    var_25 = [var_22]



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_somemoney_ge. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = 'EUR'
    var_3 = 2024
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = '50.00'
    var_9 = [var_8]
    var_10 = [var_6]
    var_11 = [var_6]
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'not a money object'
    assert var_13 is True



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_convert_with_valid_currencies. Retrieved 5/15 statements.
# Partially parsed test_convert_undefined_money. Retrieved 3/10 statements.
# Partially parsed test_convert_same_currency. Retrieved 4/14 statements.
# Partially parsed test_convert_with_asof_date. Retrieved 7/17 statements.
# Partially parsed test_convert_without_asof_date. Retrieved 5/14 statements.
# Partially parsed test_convert_with_strict_mode. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = 2019
    var_5 = 1

def test_case_0():
    var_0 = 'EUR'
    var_1 = 2019
    var_2 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = 2019
    var_5 = 1
    var_6 = 6
    var_7 = 15

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = 2019
    var_5 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '100'
    var_3 = [var_2]
    var_4 = 2019
    var_5 = 1
    var_6 = True



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_qty_map_with_defined_money. Retrieved 6/20 statements.
# Partially parsed test_qty_map_with_undefined_money. Retrieved 3/13 statements.
# Partially parsed test_qty_map_with_defined_money_different_function. Retrieved 7/21 statements.
# Partially parsed test_qty_map_with_undefined_money_string_fallback. Retrieved 4/11 statements.
# Partially parsed test_qty_map_with_defined_money_string_function. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = '42'
    var_7 = [var_6]
    var_8 = '2.00'
    var_9 = [var_8]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = [var_1]
    var_4 = '42'
    var_5 = [var_4]
    var_6 = [var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '0'
    var_8 = [var_7]
    var_9 = '10.00'
    var_10 = [var_9]

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = [var_1]
    var_4 = 'fallback'
    var_5 = lambda : var_4

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = lambda x: str(x)
    var_6 = 'error'
    var_7 = lambda : var_6



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_as_float. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '123.45'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'MonetaryOperationException'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_divide_defined_money_by_positive_number. Retrieved 6/17 statements.
# Partially parsed test_divide_defined_money_by_negative_number. Retrieved 6/17 statements.
# Partially parsed test_divide_defined_money_by_zero. Retrieved 5/15 statements.
# Partially parsed test_divide_undefined_money. Retrieved 1/6 statements.
# Partially parsed test_divide_defined_money_by_fractional_number. Retrieved 6/17 statements.
# Partially parsed test_divide_preserves_currency. Retrieved 5/15 statements.
# Partially parsed test_divide_by_one. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2'
    var_6 = [var_5]
    var_7 = '5'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '-2'
    var_6 = [var_5]
    var_7 = '-5'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '2.5'
    var_6 = [var_5]
    var_7 = '4'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'EUR'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '4'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '42'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = '1'
    var_6 = [var_5]
    var_7 = [var_1]



