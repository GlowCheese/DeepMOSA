####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_qty_or_none_defined. Retrieved 4/13 statements.
# Failed to parse test_qty_or_none_undefined.
# Partially parsed test_qty_or_none_with_zero_quantity. Retrieved 4/13 statements.


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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_somemoney_ge_equal_values. Retrieved 4/20 statements.
# Partially parsed test_somemoney_ge_greater_values. Retrieved 5/21 statements.


def test_case_0():
    var_0 = 2
    var_1 = 2023
    var_2 = 1
    var_3 = '10.00'

def test_case_0():
    var_0 = 2
    var_1 = 2023
    var_2 = 1
    var_3 = '15.00'
    var_4 = '10.00'

def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_qty_or_none_defined. Retrieved 4/13 statements.
# Partially parsed test_qty_or_none_undefined. Retrieved 2/8 statements.
# Partially parsed test_qty_or_none_zero_quantity. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.00'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = None
    var_1 = '1.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '0.00'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_price_le_defined_equal. Retrieved 4/15 statements.
# Partially parsed test_price_le_defined_less. Retrieved 5/16 statements.
# Failed to parse test_price_le_undefined_is_true.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = 2023
    var_3 = 1
    var_4 = '10'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_money_floordiv_defined_value. Retrieved 6/17 statements.
# Partially parsed test_money_floordiv_undefined_value. Retrieved 1/4 statements.
# Partially parsed test_money_floordiv_by_zero_returns_na. Retrieved 5/13 statements.
# Partially parsed test_money_floordiv_float_input. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1
    var_4 = 3
    var_5 = '3'

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.9'
    var_2 = 2023
    var_3 = 1
    var_4 = 1.5
    var_5 = '7'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_or_else_defined_price. Retrieved 7/19 statements.
# Partially parsed test_or_else_undefined_price. Retrieved 5/15 statements.


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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_with_dov_defined. Retrieved 5/14 statements.
# Partially parsed test_with_dov_undefined. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.00'
    var_2 = 2023
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_add_defined_prices_same_currency. Retrieved 7/21 statements.
# Partially parsed test_add_undefined_price_returns_other. Retrieved 4/14 statements.
# Partially parsed test_add_incompatible_currencies_raises_error. Retrieved 8/23 statements.
# Partially parsed test_add_carries_forward_date. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '5'
    var_5 = 2
    var_6 = '15'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '5'
    var_6 = 'IncompatibleCurrencyError not raised'
    var_7 = AssertionError(var_6)

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '5'
    var_5 = 2



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_gt_returns_true_for_different_class. Retrieved 6/10 statements.
# Partially parsed test_gt_raises_error_for_different_currency. Retrieved 7/13 statements.
# Partially parsed test_gt_returns_true_if_qty_is_greater. Retrieved 6/12 statements.
# Partially parsed test_gt_returns_false_if_qty_is_less. Retrieved 6/12 statements.
# Partially parsed test_gt_returns_false_if_qty_is_equal. Retrieved 5/11 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.0'
    var_3 = 2023
    var_4 = 1
    var_5 = 'not a price'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = '10.0'
    var_5 = 2023
    var_6 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = '20.0'
    var_5 = '10.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = '10.0'
    var_5 = '20.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = '10.0'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_lt_defined_less_than. Retrieved 5/16 statements.
# Partially parsed test_lt_defined_not_less_than. Retrieved 5/16 statements.
# Partially parsed test_lt_defined_equal. Retrieved 4/15 statements.
# Partially parsed test_lt_undefined_is_less_than_defined. Retrieved 4/13 statements.
# Partially parsed test_lt_defined_is_not_less_than_undefined. Retrieved 4/13 statements.
# Failed to parse test_lt_undefined_is_less_than_undefined.
# Partially parsed test_lt_incompatible_currencies_raises_error. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '20'

def test_case_0():
    var_0 = 'USD'
    var_1 = '20'
    var_2 = 2023
    var_3 = 1
    var_4 = '10'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '10'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_with_dov_defined_price. Retrieved 6/14 statements.
# Partially parsed test_with_dov_undefined_price. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_subtraction_same_currency. Retrieved 8/16 statements.
# Partially parsed test_subtraction_with_undefined_money. Retrieved 6/11 statements.
# Partially parsed test_subtraction_incompatible_currency_raises_error. Retrieved 9/18 statements.
# Partially parsed test_subtraction_updates_dov_to_latest. Retrieved 8/15 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '100.00'
    var_6 = '40.00'
    var_7 = '60.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '100.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3, var_1)
    var_5 = 2023
    var_6 = 1
    var_7 = '100.00'
    var_8 = '40.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = 5
    var_6 = '100.00'
    var_7 = '40.00'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_with_qty. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = '100.50'
    var_3 = '200.75'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_price_pos_defined. Retrieved 4/12 statements.
# Failed to parse test_price_pos_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '5.5'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_scalar_subtract_defined_price. Retrieved 6/15 statements.
# Partially parsed test_scalar_subtract_undefined_price. Retrieved 1/6 statements.
# Partially parsed test_scalar_subtract_zero_value. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '10'
    var_4 = '4'
    var_5 = '6'

def test_case_0():
    var_0 = '4'

def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '5'
    var_4 = '0'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_money_abs_defined. Retrieved 5/18 statements.
# Partially parsed test_money_abs_positive. Retrieved 4/15 statements.
# Failed to parse test_money_abs_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '-10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = '10.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_add_success. Retrieved 8/16 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 5
    var_5 = '10.00'
    var_6 = '5.50'
    var_7 = '15.50'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_round_precision_up. Retrieved 7/12 statements.
# Partially parsed test_round_precision_down. Retrieved 7/12 statements.
# Partially parsed test_round_integer. Retrieved 7/12 statements.
# Partially parsed test_round_default_zero. Retrieved 6/11 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.556'
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = '10.56'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.554'
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = '10.55'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'GBP'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.556'
    var_3 = 2023
    var_4 = 1
    var_5 = 0
    var_6 = '11'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'JPY'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.556'
    var_3 = 2023
    var_4 = 1
    var_5 = '11'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_qty_or_none_defined. Retrieved 4/13 statements.
# Failed to parse test_qty_or_none_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_price_times_with_defined_price_and_scalar. Retrieved 6/16 statements.
# Partially parsed test_price_times_with_undefined_price. Retrieved 2/9 statements.
# Partially parsed test_price_times_with_zero_scalar. Retrieved 5/15 statements.
# Partially parsed test_price_times_with_negative_scalar. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '5'
    var_4 = '2'
    var_5 = '10'

def test_case_0():
    var_0 = 'USD'
    var_1 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '5'
    var_4 = '0'

def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '5'
    var_4 = '-2'
    var_5 = '-10'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_subtract_same_currency_and_date. Retrieved 7/14 statements.
# Partially parsed test_subtract_different_dates_returns_latest. Retrieved 7/15 statements.
# Partially parsed test_subtract_undefined_price_returns_self. Retrieved 5/13 statements.
# Partially parsed test_subtract_incompatible_currency_raises_error. Retrieved 8/16 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = '5.5'
    var_4 = 2023
    var_5 = 1
    var_6 = '5.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.0'
    var_3 = 2023
    var_4 = 1
    var_5 = '5.0'
    var_6 = 10

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.0'
    var_3 = 2023
    var_4 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = '10.0'
    var_5 = 2023
    var_6 = 1
    var_7 = '5.0'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_price_as_integer_defined. Retrieved 4/11 statements.
# Failed to parse test_price_as_integer_undefined_raises_exception.
# Partially parsed test_price_as_integer_zero. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = 2023
    var_3 = 1



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test___neg__. Retrieved 5/34 statements.


def test_case_0():
    var_0 = 2
    var_1 = '10.50'
    var_2 = 2023
    var_3 = 1
    var_4 = '-10.50'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dimap_defined_value. Retrieved 7/14 statements.
# Partially parsed test_dimap_undefined_value. Retrieved 3/7 statements.
# Partially parsed test_dimap_with_complex_mapping. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '10'
    var_4 = lambda x: x.ccy.code
    var_5 = 'ERROR'
    var_6 = lambda : var_5

def test_case_0():
    var_0 = lambda x: x.ccy.code
    var_1 = 'EUR'
    var_2 = lambda : var_1

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'
    var_5 = '0'
    var_6 = '10'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test___pos__. Retrieved 5/29 statements.


def test_case_0():
    var_0 = 2
    var_1 = 2023
    var_2 = 1
    var_3 = '-50.00'
    var_4 = '50.00'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_add_same_currency_different_dates. Retrieved 8/16 statements.
# Partially parsed test_add_same_currency_same_dates. Retrieved 7/14 statements.
# Partially parsed test_add_undefined_price_returns_self. Retrieved 5/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 5
    var_5 = '10.0'
    var_6 = '5.5'
    var_7 = '15.5'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = '10.0'
    var_5 = '5.5'
    var_6 = '15.5'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = '10.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'GBP'
    var_3 = module_0.Currency(var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_qty_map_defined. Retrieved 6/19 statements.
# Partially parsed test_qty_map_undefined. Retrieved 2/10 statements.
# Partially parsed test_qty_map_different_return_type. Retrieved 7/15 statements.
# Partially parsed test_qty_map_undefined_return_type. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '1.00'
    var_4 = '42'
    var_5 = '2.00'

def test_case_0():
    var_0 = '1.00'
    var_1 = '42'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.00'
    var_2 = 2019
    var_3 = 1
    var_4 = lambda x: str(x)
    var_5 = 'fallback'
    var_6 = lambda : var_5

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = 'fallback'
    var_2 = lambda : var_1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_divide_defined_money_normal_division. Retrieved 6/18 statements.
# Partially parsed test_divide_defined_money_by_zero_returns_na. Retrieved 5/14 statements.
# Partially parsed test_divide_undefined_money_returns_itself. Retrieved 1/6 statements.
# Partially parsed test_divide_defined_money_float_divisor. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'
    var_5 = '5.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = '0'

def test_case_0():
    var_0 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = 2.5
    var_5 = '4.00'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_convert_success. Retrieved 9/24 statements.
# Partially parsed test_convert_returns_noprice_when_rate_not_found. Retrieved 9/21 statements.
# Partially parsed test_convert_raises_error_when_strict_and_rate_not_found. Retrieved 9/22 statements.
# Partially parsed test_convert_raises_programming_error_if_service_not_set. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = '100.00'
    var_8 = '85.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = '100.00'
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = '100.00'
    var_8 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = '100.00'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_with_dov_defined_money. Retrieved 5/17 statements.
# Partially parsed test_with_dov_undefined_money. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = 2024

def test_case_0():
    var_0 = 2024
    var_1 = 1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test___floordiv___valid_division. Retrieved 8/13 statements.
# Partially parsed test___floordiv___division_by_zero. Retrieved 4/5 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '10.50'
    var_4 = 2023
    var_5 = 1
    var_6 = 2
    var_7 = '5.25'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '10.50'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_convert_returns_noprice_when_rate_is_none. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = '100.00'
    var_8 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_price_as_float_defined. Retrieved 4/12 statements.
# Failed to parse test_price_as_float_undefined_raises_exception.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1.5'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_lt_defined_vs_defined_same_currency. Retrieved 5/17 statements.
# Partially parsed test_lt_defined_vs_defined_different_currency_raises_error. Retrieved 5/20 statements.
# Partially parsed test_lt_undefined_vs_defined. Retrieved 4/13 statements.
# Partially parsed test_lt_defined_vs_undefined. Retrieved 4/13 statements.
# Failed to parse test_lt_undefined_vs_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '20'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '10'
    var_3 = 2023
    var_4 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_price_equality_identical_objects. Retrieved 4/13 statements.
# Partially parsed test_price_equality_different_quantity. Retrieved 5/15 statements.
# Partially parsed test_price_equality_different_currency. Retrieved 5/15 statements.
# Partially parsed test_price_equality_different_date. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 2023
    var_2 = 1
    var_3 = '10.5'
    var_4 = '20.0'

def test_case_0():
    var_0 = '10.5'
    var_1 = 2023
    var_2 = 1
    var_3 = 'USD'
    var_4 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_money_multiplication_defined_positive. Retrieved 6/19 statements.
# Partially parsed test_money_multiplication_defined_negative. Retrieved 6/17 statements.
# Partially parsed test_money_multiplication_undefined. Retrieved 1/7 statements.
# Partially parsed test_money_multiplication_by_zero. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = '2.5'
    var_5 = '25.00'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = '-1'
    var_5 = '-10.00'

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = 'GBP'
    var_1 = '100.00'
    var_2 = 2023
    var_3 = 5
    var_4 = '0'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_money_subtraction_defined_values. Retrieved 6/20 statements.
# Partially parsed test_money_subtraction_different_currencies_raises_error. Retrieved 6/19 statements.
# Partially parsed test_money_subtraction_with_undefined_returns_other. Retrieved 4/16 statements.
# Partially parsed test_money_subtraction_scalar_subtract. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = '4.00'
    var_5 = '6.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '10.00'
    var_3 = 2023
    var_4 = 1
    var_5 = '4.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = '4.00'
    var_5 = '6.00'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_with_dov_defined_price. Retrieved 5/14 statements.
# Partially parsed test_with_dov_undefined_price. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = 2020
    var_4 = '100'

def test_case_0():
    var_0 = 2020
    var_1 = 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_ccy_or_none_defined. Retrieved 4/13 statements.
# Failed to parse test_ccy_or_none_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_as_boolean_defined_and_non_zero. Retrieved 4/12 statements.
# Partially parsed test_as_boolean_defined_and_zero. Retrieved 4/12 statements.
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_qty_or_else_defined. Retrieved 6/17 statements.
# Partially parsed test_qty_or_else_undefined_decimal. Retrieved 1/7 statements.
# Partially parsed test_qty_or_else_undefined_bool. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '42'
    var_5 = '1.00'

def test_case_0():
    var_0 = '42'

def test_case_0():
    var_0 = False
    var_1 = lambda : var_0

def test_case_0():
    pass



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_somemoney_gt_true_same_currency. Retrieved 8/15 statements.
# Partially parsed test_somemoney_gt_false_same_currency. Retrieved 8/15 statements.
# Partially parsed test_somemoney_gt_false_same_currency_equal. Retrieved 7/14 statements.
# Partially parsed test_somemoney_gt_true_different_type. Retrieved 7/11 statements.
# Partially parsed test_somemoney_gt_raises_incompatible_currency. Retrieved 9/18 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = module_0.Currency(var_0, var_1)
    var_4 = 2023
    var_5 = 1
    var_6 = '10.00'
    var_7 = '5.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = module_0.Currency(var_0, var_1)
    var_4 = 2023
    var_5 = 1
    var_6 = '5.00'
    var_7 = '10.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = module_0.Currency(var_0, var_1)
    var_4 = 2023
    var_5 = 1
    var_6 = '10.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '10.00'
    var_6 = 'Not a money object'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3, var_1)
    var_5 = 2023
    var_6 = 1
    var_7 = '10.00'
    var_8 = '5.00'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_divide_defined_price_by_scalar. Retrieved 6/16 statements.
# Partially parsed test_divide_undefined_price_returns_itself. Retrieved 1/7 statements.
# Partially parsed test_divide_by_zero_returns_undefined_price. Retrieved 5/14 statements.
# Partially parsed test_divide_by_large_scalar. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'
    var_5 = '5'

def test_case_0():
    var_0 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '0'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2023
    var_3 = 1
    var_4 = '100'
    var_5 = '0.01'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_money_round_defined_integer. Retrieved 6/16 statements.
# Partially parsed test_money_round_defined_half_even. Retrieved 8/25 statements.
# Partially parsed test_money_round_undefined. Retrieved 1/4 statements.
# Partially parsed test_money_round_zero_digits. Retrieved 6/16 statements.
# Partially parsed test_money_round_maintains_attributes. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '123.456'
    var_2 = 2023
    var_3 = 1
    var_4 = 2
    var_5 = '123.46'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1.225'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.235'
    var_5 = 2
    var_6 = '1.22'
    var_7 = '1.24'

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '123.5'
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = '124'

def test_case_0():
    var_0 = 'USD'
    var_1 = 2023
    var_2 = 1
    var_3 = '10.555'
    var_4 = 2



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_sub_subtracts_matching_currencies_and_returns_new_price. Retrieved 8/16 statements.
# Partially parsed test_sub_returns_self_when_other_is_undefined. Retrieved 5/10 statements.
# Partially parsed test_sub_raises_incompatible_currency_error. Retrieved 8/15 statements.
# Partially parsed test_sub_handles_negative_result. Retrieved 7/14 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = 5
    var_5 = '100.00'
    var_6 = '40.00'
    var_7 = '60.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = '100.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '100.00'
    var_7 = '40.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = '10.00'
    var_5 = '40.00'
    var_6 = '-30.00'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dov_or_defined. Retrieved 5/14 statements.
# Partially parsed test_dov_or_undefined. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = 'USD'
    var_3 = '1'
    var_4 = 2001

def test_case_0():
    var_0 = 2001
    var_1 = 1



