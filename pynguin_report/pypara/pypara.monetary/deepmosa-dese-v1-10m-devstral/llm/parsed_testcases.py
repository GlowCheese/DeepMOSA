####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_qty_or_none_defined_price. Retrieved 4/13 statements.
# Failed to parse test_qty_or_none_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_round_positive_ndigits. Retrieved 8/15 statements.
# Partially parsed test_round_negative_ndigits. Retrieved 8/15 statements.
# Partially parsed test_round_zero_ndigits. Retrieved 7/14 statements.
# Partially parsed test_round_negative_price. Retrieved 8/15 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '123.456'
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Currency(var_0)
    var_7 = '123.46'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '123.456'
    var_3 = 2023
    var_4 = 1
    var_5 = -1
    var_6 = module_0.Currency(var_0)
    var_7 = '120'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '123.456'
    var_3 = 2023
    var_4 = 1
    var_5 = module_0.Currency(var_0)
    var_6 = '123'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '-123.456'
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Currency(var_0)
    var_7 = '-123.46'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_as_boolean_defined_nonzero. Retrieved 4/10 statements.
# Partially parsed test_as_boolean_defined_zero. Retrieved 4/10 statements.
# Failed to parse test_as_boolean_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_qty_or_none_returns_qty. Retrieved 5/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.50'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_gte_defined_greater_than_defined. Retrieved 6/16 statements.
# Partially parsed test_gte_defined_equal_to_defined. Retrieved 5/15 statements.
# Partially parsed test_gte_defined_less_than_defined. Retrieved 6/16 statements.
# Failed to parse test_gte_undefined_greater_than_undefined.
# Partially parsed test_gte_undefined_less_than_defined. Retrieved 4/11 statements.
# Partially parsed test_gte_defined_greater_than_undefined. Retrieved 4/11 statements.
# Partially parsed test_gte_incompatible_currency_error. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = 2019
    var_3 = 1
    var_4 = '1'
    var_5 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '2'
    var_5 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'
    var_5 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_round_defined_money. Retrieved 9/28 statements.
# Partially parsed test_round_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.234'
    var_2 = 2019
    var_3 = 1
    var_4 = 2
    var_5 = '1.23'
    var_6 = '1.2'
    var_7 = 0
    var_8 = '1'

def test_case_0():
    var_0 = 2



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_ge_with_same_currency_and_greater_quantity. Retrieved 7/13 statements.
# Partially parsed test_ge_with_same_currency_and_equal_quantity. Retrieved 6/12 statements.
# Partially parsed test_ge_with_same_currency_and_lesser_quantity. Retrieved 7/13 statements.
# Partially parsed test_ge_with_different_currency. Retrieved 9/17 statements.
# Partially parsed test_ge_with_non_money_object. Retrieved 6/9 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '100.50'
    var_4 = 2023
    var_5 = 1
    var_6 = '50.25'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '75.00'
    var_4 = 2023
    var_5 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '25.00'
    var_4 = 2023
    var_5 = 1
    var_6 = '50.25'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3, var_1)
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = '50.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '100.00'
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_round_defined_price. Retrieved 6/16 statements.
# Partially parsed test_round_undefined_price. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '3.14159'
    var_2 = 2019
    var_3 = 1
    var_4 = 2
    var_5 = '3.14'

def test_case_0():
    var_0 = 2



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_qty_or_zero_defined_money. Retrieved 4/13 statements.
# Partially parsed test_qty_or_zero_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.50'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = '0'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_none_money_constructor. Retrieved 1/2 statements.


import pypara.monetary as module_0

def test_case_0():
    var_0 = module_0.NoneMoney()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_price_is_equal_returns_true_for_same_price_objects. Retrieved 4/16 statements.
# Partially parsed test_price_is_equal_returns_false_for_different_price_objects. Retrieved 5/17 statements.
# Partially parsed test_price_is_equal_returns_false_for_undefined_price. Retrieved 2/6 statements.
# Failed to parse test_price_is_equal_returns_true_for_same_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = None
    var_1 = '1'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_subtract_defined_minus_defined. Retrieved 6/20 statements.
# Partially parsed test_subtract_defined_minus_undefined. Retrieved 4/15 statements.
# Partially parsed test_subtract_undefined_minus_defined. Retrieved 5/14 statements.
# Failed to parse test_subtract_undefined_minus_undefined.
# Partially parsed test_subtract_incompatible_currency. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = '5.00'
    var_5 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '5.00'
    var_2 = 2023
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '5.00'
    var_6 = 2



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_gt_undefined_vs_defined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_vs_undefined. Retrieved 4/10 statements.
# Partially parsed test_gt_same_currency_defined. Retrieved 5/14 statements.
# Partially parsed test_gt_same_currency_not_greater. Retrieved 5/14 statements.
# Partially parsed test_gt_different_currency_raises_error. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = 2019
    var_3 = 1
    var_4 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_equal_defined_money_same_values. Retrieved 4/16 statements.
# Partially parsed test_is_equal_defined_money_different_values. Retrieved 5/17 statements.
# Failed to parse test_is_equal_undefined_money.
# Partially parsed test_is_equal_defined_vs_undefined. Retrieved 4/13 statements.
# Partially parsed test_is_equal_non_money_object. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = 2023
    var_3 = 1
    var_4 = '200.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = 2023
    var_3 = 1
    var_4 = 'not a money object'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dov_or_none_returns_dov. Retrieved 5/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #16
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
    var_4 = 2023
    var_5 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = '0.85'
    var_9 = '85.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = {}
    var_9 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = {}
    var_9 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = '0.90'
    var_9 = '90.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_neg_defined_price. Retrieved 5/13 statements.
# Failed to parse test_neg_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '-1'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_or_else_returns_self. Retrieved 9/17 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = '20.0'
    var_8 = 2



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_qty_or_zero. Retrieved 5/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_qty_map_defined_money. Retrieved 6/17 statements.
# Partially parsed test_qty_map_undefined_money. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '42'
    var_5 = '2.00'

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = '42'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_add_same_currency. Retrieved 10/20 statements.
# Partially parsed test_add_different_currency_raises_error. Retrieved 9/17 statements.
# Partially parsed test_add_with_undefined_price. Retrieved 5/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.50'
    var_3 = 2023
    var_4 = 1
    var_5 = module_0.Currency(var_0)
    var_6 = '5.25'
    var_7 = 2
    var_8 = module_0.Currency(var_0)
    var_9 = '15.75'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.50'
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = '5.25'
    var_8 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.50'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_subtract_defined_minus_defined. Retrieved 7/21 statements.
# Partially parsed test_subtract_defined_minus_undefined. Retrieved 4/13 statements.
# Partially parsed test_subtract_undefined_minus_defined. Retrieved 4/13 statements.
# Failed to parse test_subtract_undefined_minus_undefined.
# Partially parsed test_subtract_incompatible_currency. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = '3.25'
    var_5 = 2
    var_6 = '7.25'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '3.25'
    var_6 = 2



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_neg_defined_money. Retrieved 5/15 statements.
# Failed to parse test_neg_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = '-10.50'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_some_money_le_with_same_currency_and_smaller_quantity. Retrieved 7/13 statements.
# Partially parsed test_some_money_le_with_same_currency_and_equal_quantity. Retrieved 6/12 statements.
# Partially parsed test_some_money_le_with_same_currency_and_larger_quantity. Retrieved 7/13 statements.
# Partially parsed test_some_money_le_with_different_currency. Retrieved 8/16 statements.
# Partially parsed test_some_money_le_with_non_some_money_object. Retrieved 6/9 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '10.00'
    var_4 = 2023
    var_5 = 1
    var_6 = '20.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '10.00'
    var_4 = 2023
    var_5 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '20.00'
    var_4 = 2023
    var_5 = 1
    var_6 = '10.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3, var_1)
    var_5 = '10.00'
    var_6 = 2023
    var_7 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '10.00'
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_none_money_constructor. Retrieved 1/2 statements.


import pypara.monetary as module_0

def test_case_0():
    var_0 = module_0.NoneMoney()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_subtract_same_currency. Retrieved 7/16 statements.
# Partially parsed test_subtract_different_currency_raises_error. Retrieved 9/16 statements.
# Partially parsed test_subtract_undefined_money. Retrieved 6/11 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '100.50'
    var_4 = '50.25'
    var_5 = 2023
    var_6 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3, var_1)
    var_5 = '100.50'
    var_6 = '50.25'
    var_7 = 2023
    var_8 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '100.50'
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_scalar_subtract_defined_money. Retrieved 6/17 statements.
# Partially parsed test_scalar_subtract_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2019
    var_3 = 1
    var_4 = '2.50'
    var_5 = '8.00'

def test_case_0():
    var_0 = '2.50'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_lt_undefined_less_than_defined. Retrieved 4/10 statements.
# Partially parsed test_lt_defined_not_less_than_undefined. Retrieved 4/10 statements.
# Partially parsed test_lt_defined_less_than_defined. Retrieved 5/14 statements.
# Partially parsed test_lt_defined_not_less_than_defined. Retrieved 5/14 statements.
# Partially parsed test_lt_defined_equal_not_less_than. Retrieved 4/13 statements.
# Partially parsed test_lt_incompatible_currency_raises_error. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = 2019
    var_3 = 1
    var_4 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_le_undefined_money_is_less_than_or_equal_to_defined_money. Retrieved 4/10 statements.
# Partially parsed test_le_defined_money_is_not_less_than_or_equal_to_undefined_money. Retrieved 4/10 statements.
# Partially parsed test_le_defined_money_with_same_currency_and_quantity. Retrieved 5/14 statements.
# Partially parsed test_le_defined_money_with_same_currency_and_less_quantity. Retrieved 6/15 statements.
# Partially parsed test_le_defined_money_with_same_currency_and_greater_quantity. Retrieved 6/15 statements.
# Partially parsed test_le_defined_money_with_different_currency_raises_error. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '2'
    var_5 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = 2019
    var_3 = 1
    var_4 = '1'
    var_5 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'
    var_5 = 2



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_add_same_currency. Retrieved 8/18 statements.
# Partially parsed test_add_different_currency_raises_error. Retrieved 9/17 statements.
# Partially parsed test_add_with_undefined_money. Retrieved 6/11 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '100.50'
    var_4 = 2023
    var_5 = 1
    var_6 = '50.25'
    var_7 = '150.75'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3, var_1)
    var_5 = '100.50'
    var_6 = 2023
    var_7 = 1
    var_8 = '50.25'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '100.50'
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_as_integer_defined_price. Retrieved 4/12 statements.
# Failed to parse test_as_integer_undefined_price_raises_exception.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_add_defined_money_objects. Retrieved 7/21 statements.
# Partially parsed test_add_undefined_money_objects. Retrieved 4/11 statements.
# Partially parsed test_add_incompatible_currency. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.00'
    var_2 = 2019
    var_3 = 1
    var_4 = '2.00'
    var_5 = 2
    var_6 = '3.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.00'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.00'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_int_defined_money. Retrieved 4/12 statements.
# Failed to parse test_int_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_scalar_subtract_defined_money. Retrieved 6/17 statements.
# Partially parsed test_scalar_subtract_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = '2.30'
    var_5 = '8.20'

def test_case_0():
    var_0 = '2.30'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_times_defined_price. Retrieved 6/17 statements.
# Partially parsed test_times_undefined_price. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2019
    var_3 = 1
    var_4 = '2'
    var_5 = '20'

def test_case_0():
    var_0 = '2'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_abs_defined_price. Retrieved 5/20 statements.
# Failed to parse test_abs_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '-1.5'
    var_2 = 2019
    var_3 = 1
    var_4 = '1.5'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_add_with_different_currencies_raises_error. Retrieved 8/16 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.00'
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = '50.00'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_qty_map_defined_price. Retrieved 6/19 statements.
# Partially parsed test_qty_map_undefined_price. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '42'
    var_5 = '2'

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = '42'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dov_or_defined_money. Retrieved 5/14 statements.
# Partially parsed test_dov_or_undefined_money. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 2001

def test_case_0():
    var_0 = None
    var_1 = 2019
    var_2 = 1
    var_3 = 2001



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_add_raises_incompatible_currency_error. Retrieved 8/16 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = '100.00'
    var_5 = 2023
    var_6 = 1
    var_7 = '50.00'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_convert_same_currency. Retrieved 6/15 statements.
# Partially parsed test_convert_different_currency_with_rate. Retrieved 10/26 statements.
# Partially parsed test_convert_different_currency_no_rate_non_strict. Retrieved 10/19 statements.
# Partially parsed test_convert_different_currency_no_rate_strict. Retrieved 10/20 statements.
# Partially parsed test_convert_with_asof_date. Retrieved 10/27 statements.
# Partially parsed test_convert_no_fx_service. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '100.00'
    var_4 = 2023
    var_5 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = '0.85'
    var_9 = '85.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = {}
    var_9 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = {}
    var_9 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = '0.85'
    var_9 = '85.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_lte_undefined_always_less_than_or_equal_to_defined. Retrieved 4/10 statements.
# Partially parsed test_lte_defined_always_less_than_or_equal_to_undefined. Retrieved 4/10 statements.
# Failed to parse test_lte_both_undefined.
# Partially parsed test_lte_same_currency_and_quantity. Retrieved 4/13 statements.
# Partially parsed test_lte_same_currency_less_quantity. Retrieved 5/14 statements.
# Partially parsed test_lte_same_currency_greater_quantity. Retrieved 5/14 statements.
# Partially parsed test_lte_different_currency_raises_error. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = 2019
    var_3 = 1
    var_4 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_positive_defined_price_returns_same. Retrieved 4/13 statements.
# Failed to parse test_positive_undefined_price_returns_itself.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.5'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_negative_defined_price. Retrieved 5/15 statements.
# Failed to parse test_negative_undefined_price.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = 2023
    var_3 = 1
    var_4 = '-100.50'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_qty_or_defined_price. Retrieved 5/15 statements.
# Partially parsed test_qty_or_undefined_price. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.0'
    var_2 = 2019
    var_3 = 1
    var_4 = '0.0'

def test_case_0():
    var_0 = '0.0'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_with_ccy_defined_price. Retrieved 5/16 statements.
# Partially parsed test_with_ccy_undefined_price. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = 2023
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = 'EUR'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_neg_returns_negative_quantity. Retrieved 6/11 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.50'
    var_3 = 2023
    var_4 = 1
    var_5 = '-100.50'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_sub_defined_money. Retrieved 7/20 statements.
# Partially parsed test_sub_undefined_money. Retrieved 4/13 statements.
# Partially parsed test_sub_with_undefined_result. Retrieved 5/14 statements.
# Partially parsed test_sub_incompatible_currency. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = '3.25'
    var_5 = 2
    var_6 = '7.25'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '3.25'
    var_2 = 2023
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '10.50'
    var_3 = 2023
    var_4 = 1
    var_5 = '3.25'
    var_6 = 2



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dov_or_returns_correct_date. Retrieved 6/11 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.00'
    var_3 = 2023
    var_4 = 1
    var_5 = 2



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_le_same_currency_true. Retrieved 7/13 statements.
# Partially parsed test_le_same_currency_false. Retrieved 7/13 statements.
# Partially parsed test_le_same_currency_equal. Retrieved 6/12 statements.
# Partially parsed test_le_different_currency_raises_error. Retrieved 8/16 statements.
# Partially parsed test_le_non_someprice_returns_false. Retrieved 5/8 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.0'
    var_3 = 2023
    var_4 = 1
    var_5 = module_0.Currency(var_0)
    var_6 = '15.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '20.0'
    var_3 = 2023
    var_4 = 1
    var_5 = module_0.Currency(var_0)
    var_6 = '15.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '15.0'
    var_3 = 2023
    var_4 = 1
    var_5 = module_0.Currency(var_0)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.0'
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = '15.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.0'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_somemoney_constructor. Retrieved 6/9 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '100.50'
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_or_else_returns_itself_when_defined. Retrieved 7/20 statements.
# Partially parsed test_or_else_returns_fallback_when_undefined. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '2'
    var_6 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_sub_same_currency. Retrieved 7/17 statements.
# Partially parsed test_sub_different_currency_raises_error. Retrieved 9/17 statements.
# Partially parsed test_sub_with_undefined_price. Retrieved 5/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1
    var_5 = '5.25'
    var_6 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = '10.5'
    var_5 = 2023
    var_6 = 1
    var_7 = '5.25'
    var_8 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_qty_or_defined_money. Retrieved 6/16 statements.
# Partially parsed test_qty_or_undefined_money. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 0
    var_5 = '1.00'

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = 0
    var_3 = '0'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_subtract_incompatible_currency_error. Retrieved 8/16 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = '100'
    var_5 = 2023
    var_6 = 1
    var_7 = '50'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_subtract_defined_prices_same_currency. Retrieved 6/20 statements.
# Partially parsed test_subtract_undefined_price_with_defined. Retrieved 5/14 statements.
# Partially parsed test_subtract_defined_price_with_undefined. Retrieved 4/13 statements.
# Failed to parse test_subtract_undefined_prices.
# Partially parsed test_subtract_defined_prices_different_currency. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = '5.25'
    var_5 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '5.25'
    var_2 = 2023
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '5.25'
    var_6 = 2



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_scalar_add_defined_money. Retrieved 6/17 statements.
# Partially parsed test_scalar_add_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = '5.25'
    var_5 = '15.75'

def test_case_0():
    var_0 = '5.25'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_qty_or_zero_defined_money. Retrieved 4/13 statements.
# Partially parsed test_qty_or_zero_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = '0'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_some_money_ge_with_same_currency. Retrieved 7/13 statements.
# Partially parsed test_some_money_ge_with_different_currency. Retrieved 9/17 statements.
# Partially parsed test_some_money_ge_with_non_some_money. Retrieved 6/9 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '100.00'
    var_4 = 2023
    var_5 = 1
    var_6 = '50.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3, var_1)
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = '50.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '100.00'
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_gt_undefined_vs_defined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_vs_undefined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_same_currency. Retrieved 5/14 statements.
# Partially parsed test_gt_defined_different_currency. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = 2019
    var_3 = 1
    var_4 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_with_dov_defined_money. Retrieved 5/16 statements.
# Partially parsed test_with_dov_undefined_money. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = 2023
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 2
    var_2 = 1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_price_equality. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '2'
    var_6 = 2



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_round_defined_price. Retrieved 10/26 statements.
# Partially parsed test_round_undefined_price. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '3.14159'
    var_2 = 2019
    var_3 = 1
    var_4 = 2
    var_5 = '3.14'
    var_6 = 0
    var_7 = '3'
    var_8 = 4
    var_9 = '3.1416'

def test_case_0():
    var_0 = 2



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_equal_with_same_money. Retrieved 4/16 statements.
# Partially parsed test_is_equal_with_different_currency. Retrieved 5/17 statements.
# Partially parsed test_is_equal_with_different_quantity. Retrieved 5/17 statements.
# Partially parsed test_is_equal_with_different_date. Retrieved 5/17 statements.
# Failed to parse test_is_equal_with_undefined_money.
# Partially parsed test_is_equal_with_undefined_and_defined_money. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_floor_divide_defined_price. Retrieved 5/16 statements.
# Partially parsed test_floor_divide_undefined_price. Retrieved 1/4 statements.
# Partially parsed test_floor_divide_by_zero. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2019
    var_3 = 1
    var_4 = '3'

def test_case_0():
    var_0 = '3'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2019
    var_3 = 1
    var_4 = '0'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_ccy_or_none_defined_money. Retrieved 4/11 statements.
# Partially parsed test_ccy_or_none_undefined_money. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = None
    var_1 = '1'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_positive_defined_money. Retrieved 4/14 statements.
# Failed to parse test_positive_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.5'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_some_money_truediv. Retrieved 7/14 statements.
# Partially parsed test_some_money_truediv_by_zero. Retrieved 7/11 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '10.50'
    var_4 = 2023
    var_5 = 1
    var_6 = '5.25'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '10.50'
    var_4 = 2023
    var_5 = 1
    var_6 = 0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_price_ge_defined_same_currency. Retrieved 5/16 statements.
# Partially parsed test_price_ge_defined_different_currency. Retrieved 6/19 statements.
# Partially parsed test_price_ge_undefined_left. Retrieved 4/12 statements.
# Partially parsed test_price_ge_undefined_right. Retrieved 4/12 statements.
# Failed to parse test_price_ge_both_undefined.
# Partially parsed test_price_ge_equal_quantities. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2019
    var_3 = 1
    var_4 = '5'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '5'

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_none_price_constructor. Retrieved 1/2 statements.


import pypara.monetary as module_0

def test_case_0():
    var_0 = module_0.NonePrice()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_as_integer_defined_price. Retrieved 4/12 statements.
# Failed to parse test_as_integer_undefined_price_raises_exception.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_fmap_defined_price. Retrieved 7/20 statements.
# Partially parsed test_fmap_undefined_price. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 10
    var_5 = '2'
    var_6 = 11

def test_case_0():
    var_0 = None
    var_1 = '1'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_scalar_add_defined_price. Retrieved 6/17 statements.
# Partially parsed test_scalar_add_undefined_price. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1
    var_4 = '5.5'
    var_5 = '16.0'

def test_case_0():
    var_0 = '5.5'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_mul_defined_money. Retrieved 6/16 statements.
# Partially parsed test_mul_undefined_money. Retrieved 1/3 statements.
# Partially parsed test_mul_by_zero. Retrieved 6/16 statements.
# Partially parsed test_mul_by_negative. Retrieved 6/16 statements.
# Partially parsed test_mul_by_float. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = 2
    var_5 = '21.00'

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = '0.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = -1
    var_5 = '-10.50'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = 1.5
    var_5 = '15.75'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_is_equal_returns_true_for_equal_prices. Retrieved 4/16 statements.
# Partially parsed test_is_equal_returns_false_for_different_prices. Retrieved 5/17 statements.
# Partially parsed test_is_equal_returns_false_for_undefined_price. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_lt_undefined_vs_defined. Retrieved 4/10 statements.
# Partially parsed test_lt_defined_vs_undefined. Retrieved 4/10 statements.
# Failed to parse test_lt_undefined_vs_undefined.
# Partially parsed test_lt_same_currency. Retrieved 5/23 statements.
# Partially parsed test_lt_different_currency. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_convert_defined_price. Retrieved 5/17 statements.
# Partially parsed test_convert_undefined_price. Retrieved 3/7 statements.
# Partially parsed test_convert_same_currency. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 2023
    var_2 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_some_money_le_with_same_currency. Retrieved 8/14 statements.
# Partially parsed test_some_money_le_with_different_currency. Retrieved 10/18 statements.
# Partially parsed test_some_money_le_with_non_money_object. Retrieved 7/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = module_0.Currency(var_0, var_0, var_1, var_2)
    var_4 = '10.00'
    var_5 = 2023
    var_6 = 1
    var_7 = '15.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = module_0.Currency(var_0, var_0, var_1, var_2)
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = module_0.Currency(var_4, var_4, var_5, var_2)
    var_7 = '10.00'
    var_8 = 2023
    var_9 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = module_0.Currency(var_0, var_0, var_1, var_2)
    var_4 = '10.00'
    var_5 = 2023
    var_6 = 1



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_gt_undefined_vs_defined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_vs_undefined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_vs_defined_same_currency. Retrieved 5/14 statements.
# Partially parsed test_gt_defined_vs_defined_different_currency. Retrieved 5/15 statements.
# Failed to parse test_gt_undefined_vs_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = 2019
    var_3 = 1
    var_4 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_mul_defined_price_with_scalar. Retrieved 6/16 statements.
# Partially parsed test_mul_undefined_price_with_scalar. Retrieved 1/3 statements.
# Partially parsed test_mul_defined_price_with_zero. Retrieved 6/16 statements.
# Partially parsed test_mul_defined_price_with_negative_scalar. Retrieved 6/16 statements.
# Partially parsed test_mul_defined_price_with_float. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1
    var_4 = 2
    var_5 = '21.0'

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = '0'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1
    var_4 = -1
    var_5 = '-10.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1
    var_4 = 1.5
    var_5 = '15.75'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_scalar_subtract_defined_money. Retrieved 5/16 statements.
# Partially parsed test_scalar_subtract_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2019
    var_3 = 1
    var_4 = '5.25'

def test_case_0():
    var_0 = '5.25'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_gt_undefined_vs_defined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_vs_undefined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_vs_defined_same_currency. Retrieved 5/32 statements.
# Partially parsed test_gt_defined_vs_defined_different_currency. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = 2019
    var_3 = 1
    var_4 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_SomePrice___ge__. Retrieved 10/25 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.00'
    var_3 = 2023
    var_4 = 1
    var_5 = module_0.Currency(var_0)
    var_6 = '5.00'
    var_7 = module_0.Currency(var_0)
    var_8 = 'EUR'
    var_9 = module_0.Currency(var_8)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_gt_undefined_vs_defined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_vs_undefined. Retrieved 4/10 statements.
# Partially parsed test_gt_defined_vs_defined_same_currency. Retrieved 5/32 statements.
# Partially parsed test_gt_defined_vs_defined_different_currency. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'
    var_2 = 2019
    var_3 = 1
    var_4 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dov_or_none_defined_price. Retrieved 4/12 statements.
# Partially parsed test_dov_or_none_undefined_price. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = None
    var_1 = 2019
    var_2 = 1



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_scalar_subtract_defined_money. Retrieved 6/17 statements.
# Partially parsed test_scalar_subtract_undefined_money. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2019
    var_3 = 1
    var_4 = '2.50'
    var_5 = '8.00'

def test_case_0():
    var_0 = '2.50'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_add_defined_money_objects. Retrieved 7/21 statements.
# Failed to parse test_add_undefined_money_objects.
# Partially parsed test_add_defined_and_undefined_money_objects. Retrieved 4/13 statements.
# Partially parsed test_add_incompatible_currencies. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = '5.25'
    var_5 = 2
    var_6 = '15.75'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '5.25'
    var_6 = 2



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_or_else_defined_price_returns_itself. Retrieved 7/18 statements.
# Partially parsed test_or_else_undefined_price_returns_fallback. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '2'
    var_6 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = None



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_round_defined_price. Retrieved 6/15 statements.
# Partially parsed test_round_undefined_price. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '3.14159'
    var_2 = 2019
    var_3 = 1
    var_4 = 2
    var_5 = '3.14'

def test_case_0():
    var_0 = 2



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_or_else_returns_itself_when_defined. Retrieved 7/20 statements.
# Partially parsed test_or_else_returns_fallback_when_undefined. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '2'
    var_6 = 2

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = 'USD'
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_equality_of_two_money_objects. Retrieved 4/13 statements.
# Partially parsed test_equality_of_money_with_different_currency. Retrieved 5/14 statements.
# Partially parsed test_equality_of_money_with_different_quantity. Retrieved 5/14 statements.
# Partially parsed test_equality_of_money_with_different_date. Retrieved 5/14 statements.
# Failed to parse test_equality_of_undefined_money.
# Partially parsed test_equality_of_defined_and_undefined_money. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1



