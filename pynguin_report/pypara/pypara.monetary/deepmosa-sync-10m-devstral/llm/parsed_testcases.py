####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_equal_with_same_money_objects. Retrieved 4/16 statements.
# Partially parsed test_is_equal_with_different_money_objects. Retrieved 5/17 statements.
# Failed to parse test_is_equal_with_undefined_money.
# Partially parsed test_is_equal_with_undefined_and_defined_money. Retrieved 4/13 statements.
# Partially parsed test_is_equal_with_non_money_object. Retrieved 5/13 statements.


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

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'not a money object'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_ccy_or_defined_price. Retrieved 5/15 statements.
# Partially parsed test_ccy_or_undefined_price. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 'EUR'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_lte_undefined_price_is_less_than_or_equal_to_defined_price. Retrieved 4/10 statements.
# Partially parsed test_lte_defined_price_is_not_less_than_or_equal_to_undefined_price. Retrieved 4/10 statements.
# Partially parsed test_lte_same_defined_prices_are_equal. Retrieved 4/13 statements.
# Partially parsed test_lte_lesser_defined_price_is_less_than_greater_defined_price. Retrieved 5/14 statements.
# Partially parsed test_lte_greater_defined_price_is_not_less_than_or_equal_to_lesser_defined_price. Retrieved 5/14 statements.
# Partially parsed test_lte_incompatible_currency_error_raised_for_different_currencies. Retrieved 5/15 statements.


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
    var_5 = 'EUR'
    var_6 = [var_1]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_some_price_equality. Retrieved 8/29 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]
    var_6 = '200.00'
    var_7 = [var_6]
    var_8 = 'EUR'
    var_9 = [var_1]
    var_10 = None
    var_11 = 'not a price'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_divide_defined_price. Retrieved 6/17 statements.
# Partially parsed test_divide_undefined_price. Retrieved 1/4 statements.
# Partially parsed test_divide_by_zero. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '5'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '2'
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_fmap_defined_price. Retrieved 7/19 statements.
# Partially parsed test_fmap_undefined_price. Retrieved 2/8 statements.


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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_multiply_defined_price. Retrieved 6/16 statements.
# Partially parsed test_multiply_undefined_price. Retrieved 1/3 statements.


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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_divide_defined_money. Retrieved 6/17 statements.
# Partially parsed test_divide_undefined_money. Retrieved 1/4 statements.
# Partially parsed test_divide_by_zero. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '5.00'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '2'
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_price_sub_defined_undefined. Retrieved 4/14 statements.
# Partially parsed test_price_sub_undefined_defined. Retrieved 4/14 statements.
# Failed to parse test_price_sub_undefined_undefined.
# Partially parsed test_price_sub_defined_defined_same_currency. Retrieved 6/20 statements.


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

def test_case_0():
    var_0 = 'USD'
    var_1 = '200.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '100.00'
    var_7 = [var_6]
    var_8 = 2
    var_9 = [var_3, var_4, var_8]
    var_10 = [var_6]
    var_11 = [var_3, var_4, var_8]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_qty_or_else_defined_price. Retrieved 7/20 statements.
# Partially parsed test_qty_or_else_undefined_price. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '20.0'
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = 'fallback'
    var_10 = lambda : var_9
    var_11 = [var_1]

def test_case_0():
    var_0 = '20.0'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = 'fallback'
    var_4 = lambda : var_3



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_add_defined_money_with_same_currency. Retrieved 7/21 statements.
# Partially parsed test_add_defined_money_with_different_currency. Retrieved 7/20 statements.
# Partially parsed test_add_undefined_money_with_defined. Retrieved 5/14 statements.
# Partially parsed test_add_defined_money_with_undefined. Retrieved 4/13 statements.
# Failed to parse test_add_two_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = 2
    var_9 = [var_3, var_4, var_8]
    var_10 = '15.75'
    var_11 = [var_10]
    var_12 = [var_3, var_4, var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '5.25'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_3, var_4, var_9]
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '5.25'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_le_undefined_vs_defined. Retrieved 4/9 statements.
# Partially parsed test_le_defined_vs_undefined. Retrieved 4/10 statements.
# Partially parsed test_le_same_currency. Retrieved 5/13 statements.
# Partially parsed test_le_different_currency. Retrieved 5/15 statements.


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
    var_5 = 'EUR'
    var_6 = [var_1]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_or_else_defined_money_returns_itself. Retrieved 7/20 statements.
# Partially parsed test_or_else_undefined_money_returns_fallback. Retrieved 5/16 statements.


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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_equal_returns_true_for_same_price_objects. Retrieved 4/16 statements.
# Partially parsed test_is_equal_returns_false_for_different_price_objects. Retrieved 5/17 statements.
# Partially parsed test_is_equal_returns_false_for_undefined_price. Retrieved 4/13 statements.
# Failed to parse test_is_equal_returns_true_for_two_undefined_prices.
# Partially parsed test_is_equal_returns_false_for_non_price_object. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = [var_1]
    var_8 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'not a price'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_price_truediv_defined. Retrieved 6/17 statements.
# Partially parsed test_price_truediv_undefined. Retrieved 1/4 statements.
# Partially parsed test_price_truediv_by_zero. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '2'
    var_7 = [var_6]
    var_8 = '5'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '2'
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_convert_with_valid_fx_rate. Retrieved 10/27 statements.
# Partially parsed test_convert_with_no_fx_rate_non_strict. Retrieved 10/20 statements.
# Partially parsed test_convert_with_no_fx_rate_strict. Retrieved 12/23 statements.
# Partially parsed test_convert_with_no_fx_service. Retrieved 10/21 statements.
# Partially parsed test_convert_with_default_asof. Retrieved 10/26 statements.


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
    var_10 = 2023
    var_11 = 1
    var_12 = True
    var_13 = bool(False)
    assert var_13 is True

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
    var_9 = 2023
    var_10 = 1
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
    var_9 = '0.85'
    var_10 = [var_9]
    var_11 = '85.00'
    var_12 = [var_11]



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_none_money_constructor.




# Parsed testcases at query #18
#--------------------------

# Failed to parse test_none_money_constructor.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_subtract_defined_prices_same_currency. Retrieved 6/20 statements.
# Partially parsed test_subtract_defined_prices_different_currency. Retrieved 7/21 statements.
# Partially parsed test_subtract_undefined_price_with_defined. Retrieved 5/14 statements.
# Partially parsed test_subtract_defined_price_with_undefined. Retrieved 4/13 statements.
# Failed to parse test_subtract_two_undefined_prices.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '50'
    var_7 = [var_6]
    var_8 = 2
    var_9 = [var_3, var_4, var_8]
    var_10 = [var_6]
    var_11 = [var_3, var_4, var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '50'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_3, var_4, var_9]
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_abs_defined_price. Retrieved 5/15 statements.
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_convert_with_valid_fx_rate. Retrieved 10/27 statements.
# Partially parsed test_convert_with_no_fx_rate_non_strict. Retrieved 10/20 statements.
# Partially parsed test_convert_with_no_fx_rate_strict. Retrieved 12/23 statements.
# Partially parsed test_convert_with_no_fx_service. Retrieved 10/21 statements.
# Partially parsed test_convert_with_default_date. Retrieved 10/26 statements.


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
    var_9 = '0.90'
    var_10 = [var_9]
    var_11 = '90.00'
    var_12 = [var_11]

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
    var_9 = {}
    var_10 = False

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
    var_9 = {}
    var_10 = 2023
    var_11 = 1
    var_12 = True
    var_13 = bool(False)
    assert var_13 is True

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
    var_9 = 2023
    var_10 = 1
    var_11 = bool(False)
    assert var_11 is True

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
    var_9 = '0.90'
    var_10 = [var_9]
    var_11 = '90.00'
    var_12 = [var_11]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_subtract_defined_minus_defined. Retrieved 7/21 statements.
# Partially parsed test_subtract_defined_minus_undefined. Retrieved 4/14 statements.
# Partially parsed test_subtract_undefined_minus_defined. Retrieved 5/15 statements.
# Failed to parse test_subtract_undefined_minus_undefined.
# Partially parsed test_subtract_incompatible_currency. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '3.00'
    var_7 = [var_6]
    var_8 = 2
    var_9 = [var_3, var_4, var_8]
    var_10 = '7.00'
    var_11 = [var_10]
    var_12 = [var_3, var_4, var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '3.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '3.00'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_3, var_4, var_9]
    var_11 = bool(False)
    assert var_11 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_somemoney_constructor. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dov_or_defined_money. Retrieved 5/14 statements.
# Partially parsed test_dov_or_undefined_money. Retrieved 4/9 statements.


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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_qty_or. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '0'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = '999.99'
    var_9 = [var_8]
    var_10 = [var_1]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_scalar_add_with_defined_price. Retrieved 6/17 statements.
# Partially parsed test_scalar_add_with_undefined_price. Retrieved 1/4 statements.
# Partially parsed test_scalar_add_with_integer. Retrieved 6/16 statements.
# Partially parsed test_scalar_add_with_float. Retrieved 6/16 statements.


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

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 5
    var_7 = '15.5'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 5.5
    var_7 = '16.0'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_mul_defined_money_with_scalar. Retrieved 6/16 statements.
# Partially parsed test_mul_defined_money_with_zero. Retrieved 6/16 statements.
# Partially parsed test_mul_undefined_money. Retrieved 1/3 statements.
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
    var_0 = 5

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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_abs_defined_positive_price. Retrieved 4/14 statements.
# Partially parsed test_abs_defined_negative_price. Retrieved 5/15 statements.
# Failed to parse test_abs_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '-10.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '10.5'
    var_7 = [var_6]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_add_defined_prices_with_same_currency. Retrieved 7/21 statements.
# Partially parsed test_add_undefined_price_with_defined_price. Retrieved 5/16 statements.
# Partially parsed test_add_defined_price_with_undefined_price. Retrieved 4/15 statements.
# Failed to parse test_add_two_undefined_prices.
# Partially parsed test_add_defined_prices_with_different_currencies_raises_error. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = 2
    var_9 = [var_3, var_4, var_8]
    var_10 = '15.75'
    var_11 = [var_10]
    var_12 = [var_3, var_4, var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5.25'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]
    var_7 = [var_1]
    var_8 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = [var_1]
    var_7 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '5.25'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_3, var_4, var_9]
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_subtract_defined_money. Retrieved 6/20 statements.
# Partially parsed test_subtract_undefined_money. Retrieved 5/11 statements.
# Partially parsed test_subtract_with_incompatible_currency. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = 2
    var_9 = [var_3, var_4, var_8]
    var_10 = [var_6]
    var_11 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5.25'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '5.25'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_3, var_4, var_9]
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_truediv_valid_division. Retrieved 6/15 statements.
# Partially parsed test_truediv_division_by_zero. Retrieved 5/10 statements.
# Partially parsed test_truediv_division_by_invalid_operand. Retrieved 5/11 statements.


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

# Partially parsed test_price_add_defined_prices_same_currency. Retrieved 7/21 statements.
# Partially parsed test_price_add_undefined_and_defined. Retrieved 5/14 statements.
# Partially parsed test_price_add_defined_and_undefined. Retrieved 4/13 statements.
# Failed to parse test_price_add_both_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = 2
    var_9 = [var_3, var_4, var_8]
    var_10 = '15.75'
    var_11 = [var_10]
    var_12 = [var_3, var_4, var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '5.25'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_somemoney_constructor. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_or_else_returns_itself_when_defined. Retrieved 7/20 statements.
# Partially parsed test_or_else_returns_fallback_when_undefined. Retrieved 4/14 statements.


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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_somemoney_constructor. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_qty_map_defined_money. Retrieved 6/17 statements.
# Partially parsed test_qty_map_undefined_money. Retrieved 3/11 statements.


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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_fmap_defined_money. Retrieved 7/19 statements.
# Partially parsed test_fmap_undefined_money. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_1]
    var_6 = 10
    var_7 = '2.00'
    var_8 = [var_7]
    var_9 = 11

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = [var_1]
    var_3 = [var_1]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_qty_or_defined_price. Retrieved 5/15 statements.
# Partially parsed test_qty_or_undefined_price. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '0'
    var_7 = [var_6]
    var_8 = [var_1]

def test_case_0():
    var_0 = '0'
    var_1 = [var_0]
    var_2 = [var_0]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_float_defined_price. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.5'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_add_defined_money_objects. Retrieved 7/21 statements.
# Partially parsed test_add_undefined_and_defined_money_objects. Retrieved 4/14 statements.
# Failed to parse test_add_undefined_money_objects.
# Partially parsed test_add_incompatible_currencies. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = 2
    var_9 = [var_3, var_4, var_8]
    var_10 = '15.75'
    var_11 = [var_10]
    var_12 = [var_3, var_4, var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '5.25'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_3, var_4, var_9]
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_convert_with_valid_fx_rate. Retrieved 10/27 statements.
# Partially parsed test_convert_with_no_fx_rate_non_strict. Retrieved 10/20 statements.
# Partially parsed test_convert_with_no_fx_rate_strict. Retrieved 12/23 statements.
# Partially parsed test_convert_with_no_fx_service. Retrieved 10/21 statements.
# Partially parsed test_convert_with_different_decimals. Retrieved 11/28 statements.
# Partially parsed test_convert_with_default_date. Retrieved 10/26 statements.


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
    var_9 = '0.85'
    var_10 = [var_9]
    var_11 = '85.00'
    var_12 = [var_11]

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
    var_9 = {}
    var_10 = False

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
    var_9 = {}
    var_10 = 2023
    var_11 = 1
    var_12 = True
    var_13 = bool(False)
    assert var_13 is True

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
    var_9 = 2023
    var_10 = 1
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = '100.50'
    var_7 = [var_6]
    var_8 = 2023
    var_9 = 1
    var_10 = '110.25'
    var_11 = [var_10]
    var_12 = '11073'
    var_13 = [var_12]

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
    var_9 = '0.85'
    var_10 = [var_9]
    var_11 = '85.00'
    var_12 = [var_11]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_someprice_add_with_same_currency. Retrieved 7/20 statements.
# Partially parsed test_someprice_add_with_different_currency. Retrieved 7/17 statements.
# Partially parsed test_someprice_add_with_undefined_price. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '5.25'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '15.75'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = 2
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_defined_price_with_nonzero_quantity_is_truthy. Retrieved 4/10 statements.
# Partially parsed test_defined_price_with_zero_quantity_is_falsy. Retrieved 4/10 statements.
# Failed to parse test_undefined_price_is_falsy.


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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_with_dov_defined_money. Retrieved 6/16 statements.
# Partially parsed test_with_dov_undefined_money. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 12
    var_7 = 31
    var_8 = [var_3, var_6, var_7]
    var_9 = [var_1]

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_with_qty_defined_money. Retrieved 5/16 statements.
# Partially parsed test_with_qty_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '20.75'
    var_7 = [var_6]
    var_8 = [var_6]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = '20.75'
    var_1 = [var_0]



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_none_money_constructor.




# Parsed testcases at query #26
#--------------------------

# Partially parsed test_multiply_defined_money_by_numeric. Retrieved 6/16 statements.
# Partially parsed test_multiply_undefined_money_by_numeric. Retrieved 1/3 statements.
# Partially parsed test_multiply_defined_money_by_zero. Retrieved 6/16 statements.
# Partially parsed test_multiply_defined_money_by_negative_numeric. Retrieved 6/16 statements.
# Partially parsed test_multiply_defined_money_by_fractional_numeric. Retrieved 6/17 statements.


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
    var_1 = '100.00'
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
    var_6 = -3
    var_7 = '-31.50'
    var_8 = [var_7]
    var_9 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = '5.00'
    var_9 = [var_8]
    var_10 = [var_3, var_4, var_4]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_sub_same_currency_and_different_dov. Retrieved 6/19 statements.
# Partially parsed test_sub_same_currency_and_same_dov. Retrieved 5/18 statements.
# Partially parsed test_sub_different_currency_raises_error. Retrieved 7/17 statements.
# Partially parsed test_sub_undefined_price_returns_self. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '5.25'
    var_6 = [var_5]
    var_7 = 2
    var_8 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '5.25'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = 2
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_add_same_currency. Retrieved 7/18 statements.
# Partially parsed test_add_different_currency_raises_error. Retrieved 7/17 statements.
# Partially parsed test_add_with_undefined_money. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '5.25'
    var_6 = [var_5]
    var_7 = 2
    var_8 = '15.75'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '10.50'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = 2
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_some_money_le_with_same_currency_and_smaller_quantity. Retrieved 6/13 statements.
# Partially parsed test_some_money_le_with_same_currency_and_equal_quantity. Retrieved 5/12 statements.
# Partially parsed test_some_money_le_with_same_currency_and_larger_quantity. Retrieved 6/13 statements.
# Partially parsed test_some_money_le_with_different_currency. Retrieved 6/16 statements.
# Partially parsed test_some_money_le_with_non_money_object. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '20.00'
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
    var_2 = '20.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1
    var_6 = '10.00'
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
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = '10.00'
    var_3 = [var_2]
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_subtract_defined_prices_same_currency. Retrieved 6/20 statements.
# Partially parsed test_subtract_defined_prices_different_currency. Retrieved 7/21 statements.
# Partially parsed test_subtract_undefined_price_with_defined. Retrieved 5/14 statements.
# Partially parsed test_subtract_defined_price_with_undefined. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = '5.25'
    var_7 = [var_6]
    var_8 = 2
    var_9 = [var_3, var_4, var_8]
    var_10 = [var_6]
    var_11 = [var_3, var_4, var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 'EUR'
    var_7 = '5.25'
    var_8 = [var_7]
    var_9 = 2
    var_10 = [var_3, var_4, var_9]
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '5.25'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_convert_with_valid_fx_rate. Retrieved 10/27 statements.
# Partially parsed test_convert_with_no_fx_rate_non_strict. Retrieved 10/20 statements.
# Partially parsed test_convert_with_no_fx_rate_strict. Retrieved 12/23 statements.
# Partially parsed test_convert_with_no_fx_service. Retrieved 10/21 statements.
# Partially parsed test_convert_with_default_date. Retrieved 10/26 statements.


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
    var_10 = 2023
    var_11 = 1
    var_12 = True
    var_13 = bool(False)
    assert var_13 is True

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
    var_9 = 2023
    var_10 = 1
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_subtract_with_undefined_other_returns_self. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.0'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_add_with_undefined_price. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_someprice_ge_with_same_currency. Retrieved 5/13 statements.
# Partially parsed test_someprice_ge_with_different_currency. Retrieved 5/15 statements.
# Partially parsed test_someprice_ge_with_none_price. Retrieved 4/8 statements.
# Partially parsed test_someprice_ge_with_equal_price. Retrieved 4/12 statements.
# Partially parsed test_someprice_ge_with_less_price. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
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
    var_5 = 'EUR'
    var_6 = [var_1]
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = [var_1]

def test_case_0():
    var_0 = 'USD'
    var_1 = '9.00'
    var_2 = [var_1]
    var_3 = 2023
    var_4 = 1
    var_5 = '10.00'
    var_6 = [var_5]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_round_defined_money. Retrieved 6/16 statements.
# Partially parsed test_round_undefined_money. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '3.14159'
    var_2 = [var_1]
    var_3 = 2019
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = 2
    var_7 = '3.14'
    var_8 = [var_7]

def test_case_0():
    var_0 = 2



