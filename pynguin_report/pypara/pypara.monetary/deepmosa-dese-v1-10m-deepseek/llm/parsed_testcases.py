####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_qty_or_none_returns_qty_for_defined_price. Retrieved 4/13 statements.
# Partially parsed test_qty_or_none_returns_none_for_undefined_price. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = None
    var_1 = '1'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test___ge___with_same_currency_and_greater_qty. Retrieved 6/13 statements.
# Partially parsed test___ge___with_same_currency_and_equal_qty. Retrieved 5/12 statements.
# Partially parsed test___ge___with_same_currency_and_lesser_qty. Retrieved 6/13 statements.
# Partially parsed test___ge___with_different_currency_raises_error. Retrieved 7/15 statements.
# Partially parsed test___ge___with_non_somemoney_returns_true. Retrieved 6/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.0'
    var_3 = 2023
    var_4 = 1
    var_5 = '50.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.0'
    var_3 = 2023
    var_4 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '50.0'
    var_3 = 2023
    var_4 = 1
    var_5 = '100.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = '100.0'
    var_5 = 2023
    var_6 = 1

import pypara.currencies as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.0'
    var_3 = 2023
    var_4 = 1
    var_5 = module_1.object()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_gte_returns_true_when_other_is_not_somemoney. Retrieved 5/10 statements.
# Partially parsed test_gte_raises_incompatiblecurrencyerror_when_currencies_differ. Retrieved 8/15 statements.
# Partially parsed test_gte_returns_true_when_qty_greater. Retrieved 6/12 statements.
# Partially parsed test_gte_returns_true_when_qty_equal. Retrieved 5/10 statements.
# Partially parsed test_gte_returns_false_when_qty_less. Retrieved 6/12 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.00'
    var_3 = 2023
    var_4 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = '100.00'
    var_5 = '90.00'
    var_6 = 2023
    var_7 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.00'
    var_3 = '90.00'
    var_4 = 2023
    var_5 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.00'
    var_3 = 2023
    var_4 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '90.00'
    var_3 = '100.00'
    var_4 = 2023
    var_5 = 1



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_equal_with_same_defined_price. Retrieved 4/15 statements.
# Partially parsed test_is_equal_with_different_quantity. Retrieved 5/16 statements.
# Partially parsed test_is_equal_with_different_currency. Retrieved 5/16 statements.
# Partially parsed test_is_equal_with_different_date. Retrieved 5/16 statements.
# Failed to parse test_is_equal_with_undefined_price.
# Partially parsed test_is_equal_defined_vs_undefined. Retrieved 4/12 statements.
# Partially parsed test_is_equal_with_non_price_object. Retrieved 5/12 statements.
# Partially parsed test_is_equal_with_none. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1
    var_4 = '200'

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1
    var_4 = 'not a price'

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1
    var_4 = None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_or_else_defined_money_returns_itself. Retrieved 7/20 statements.
# Partially parsed test_or_else_undefined_money_returns_fallback. Retrieved 4/14 statements.
# Partially parsed test_or_else_undefined_money_with_none_components_returns_fallback. Retrieved 5/16 statements.
# Partially parsed test_or_else_fallback_lambda_called_only_for_undefined. Retrieved 5/19 statements.


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

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = 'USD'
    var_3 = 2019
    var_4 = 1

def test_case_0():
    var_0 = 0
    assert var_0 == 0
    assert var_0 == 1
    var_1 = 'USD'
    var_2 = '1'
    var_3 = 2019
    var_4 = 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_floordiv_with_defined_money_and_non_zero_scalar. Retrieved 5/17 statements.
# Partially parsed test_floordiv_with_defined_money_and_zero_scalar. Retrieved 5/15 statements.
# Partially parsed test_floordiv_with_undefined_money. Retrieved 1/6 statements.
# Partially parsed test_floordiv_with_defined_money_and_integer_scalar. Retrieved 6/17 statements.
# Partially parsed test_floordiv_with_defined_money_and_float_scalar. Retrieved 6/17 statements.
# Partially parsed test_floordiv_with_defined_money_negative_scalar. Retrieved 6/18 statements.
# Partially parsed test_floordiv_with_defined_money_negative_quantity. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '3'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '0'

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '7'
    var_2 = 2023
    var_3 = 1
    var_4 = 2
    var_5 = '3'

def test_case_0():
    var_0 = 'JPY'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1
    var_4 = 30.0
    var_5 = '3'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '-3'
    var_5 = '-4'

def test_case_0():
    var_0 = 'USD'
    var_1 = '-10'
    var_2 = 2023
    var_3 = 1
    var_4 = '3'
    var_5 = '-4'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_convert_same_currency_no_conversion_needed. Retrieved 7/17 statements.
# Partially parsed test_convert_with_valid_fx_rate. Retrieved 11/27 statements.
# Partially parsed test_convert_with_strict_true_and_rate_found. Retrieved 11/27 statements.
# Partially parsed test_convert_with_strict_true_and_rate_not_found. Retrieved 12/23 statements.
# Partially parsed test_convert_with_strict_false_and_rate_not_found. Retrieved 10/20 statements.
# Partially parsed test_convert_with_asof_date_different_from_dov. Retrieved 12/28 statements.
# Partially parsed test_convert_with_asof_date_none_uses_dov. Retrieved 12/27 statements.
# Partially parsed test_convert_with_fx_rate_service_not_set. Retrieved 11/22 statements.
# Partially parsed test_convert_quantizes_result_to_target_currency. Retrieved 12/28 statements.
# Partially parsed test_convert_with_crypto_currency_and_negative_decimals. Retrieved 11/31 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '100.00'
    var_4 = 2023
    var_5 = 1
    var_6 = False

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
    var_9 = False
    var_10 = '85.00'

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
    var_9 = True
    var_10 = '85.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = None
    var_9 = 2023
    var_10 = 1
    var_11 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = None
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
    var_8 = '0.90'
    var_9 = 6
    var_10 = False
    var_11 = '90.00'

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
    var_9 = None
    var_10 = False
    var_11 = '85.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = 2023
    var_9 = 1
    var_10 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = '100.555'
    var_7 = 2023
    var_8 = 1
    var_9 = '110.123'
    var_10 = False
    var_11 = '11079'

def test_case_0():
    var_0 = 'BTC'
    var_1 = 'Bitcoin'
    var_2 = -1
    var_3 = 'ETH'
    var_4 = 'Ethereum'
    var_5 = -1
    var_6 = '1.123456789012345'
    var_7 = 2023
    var_8 = 1
    var_9 = '15.123456789012345'
    var_10 = False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_mul_with_defined_money_and_positive_scalar. Retrieved 6/18 statements.
# Partially parsed test_mul_with_defined_money_and_negative_scalar. Retrieved 6/18 statements.
# Partially parsed test_mul_with_defined_money_and_zero_scalar. Retrieved 5/17 statements.
# Partially parsed test_mul_with_undefined_money. Retrieved 1/6 statements.
# Partially parsed test_mul_with_defined_money_and_integer_scalar. Retrieved 6/17 statements.
# Partially parsed test_mul_with_defined_money_and_float_scalar. Retrieved 6/17 statements.
# Partially parsed test_mul_commutative_property. Retrieved 5/16 statements.
# Partially parsed test_mul_with_defined_money_and_decimal_scalar. Retrieved 6/18 statements.
# Partially parsed test_mul_preserves_currency_and_date. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'
    var_5 = '20.00'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '5'
    var_2 = 2023
    var_3 = 1
    var_4 = '-3'
    var_5 = '-15.00'

def test_case_0():
    var_0 = 'JPY'
    var_1 = '7'
    var_2 = 2023
    var_3 = 1
    var_4 = '0'

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
    var_2 = 2023
    var_3 = 1
    var_4 = 4
    var_5 = '12.00'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '2'
    var_2 = 2023
    var_3 = 1
    var_4 = 2.5
    var_5 = '5.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
    var_2 = 2023
    var_3 = 1
    var_4 = '4'

def test_case_0():
    var_0 = 'GBP'
    var_1 = '8'
    var_2 = 2023
    var_3 = 1
    var_4 = '0.5'
    var_5 = '4.00'

def test_case_0():
    var_0 = 'CAD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 5
    var_4 = 15
    var_5 = '2'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_with_qty_returns_new_money_with_given_quantity_when_defined. Retrieved 5/15 statements.
# Partially parsed test_with_qty_returns_itself_when_undefined. Retrieved 1/6 statements.
# Partially parsed test_with_qty_quantity_is_quantized_to_currency. Retrieved 5/16 statements.
# Partially parsed test_with_qty_handles_zero_quantity. Retrieved 5/16 statements.
# Partially parsed test_with_qty_handles_negative_quantity. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '20'

def test_case_0():
    var_0 = '20'

def test_case_0():
    var_0 = 'JPY'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1
    var_4 = '123.456'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '5'
    var_2 = 2023
    var_3 = 1
    var_4 = '0'

def test_case_0():
    var_0 = 'GBP'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '-5'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_round_positive_quantity_round_down. Retrieved 7/13 statements.
# Partially parsed test_round_positive_quantity_round_up. Retrieved 7/13 statements.
# Partially parsed test_round_negative_quantity_round_down. Retrieved 7/13 statements.
# Partially parsed test_round_negative_quantity_round_up. Retrieved 7/13 statements.
# Partially parsed test_round_zero_ndigits. Retrieved 8/14 statements.
# Partially parsed test_round_ndigits_greater_than_decimals. Retrieved 8/14 statements.
# Partially parsed test_round_exact_half_up. Retrieved 7/13 statements.
# Partially parsed test_round_exact_half_down. Retrieved 7/13 statements.
# Partially parsed test_round_negative_ndigits. Retrieved 8/14 statements.
# Partially parsed test_round_negative_ndigits_greater_than_magnitude. Retrieved 8/14 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '123.456'
    var_4 = 2023
    var_5 = 1
    var_6 = '123.4'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '123.456'
    var_4 = 2023
    var_5 = 1
    var_6 = '123.46'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '-123.456'
    var_4 = 2023
    var_5 = 1
    var_6 = '-123.4'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '-123.456'
    var_4 = 2023
    var_5 = 1
    var_6 = '-123.46'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '123.456'
    var_4 = 2023
    var_5 = 1
    var_6 = 0
    var_7 = '123'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '123.456'
    var_4 = 2023
    var_5 = 1
    var_6 = 5
    var_7 = '123.46'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '123.455'
    var_4 = 2023
    var_5 = 1
    var_6 = '123.46'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'JPY'
    var_1 = 0
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '123.5'
    var_4 = 2023
    var_5 = 1
    var_6 = '124'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '123.456'
    var_4 = 2023
    var_5 = 1
    var_6 = -1
    var_7 = '120'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = '123.456'
    var_4 = 2023
    var_5 = 1
    var_6 = -3
    var_7 = '0'



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_as_boolean_returns_false_for_undefined_money.
# Partially parsed test_as_boolean_returns_false_for_zero_quantity. Retrieved 4/11 statements.
# Partially parsed test_as_boolean_returns_true_for_positive_quantity. Retrieved 4/11 statements.
# Partially parsed test_as_boolean_returns_true_for_negative_quantity. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '-1'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test___neg___with_defined_money. Retrieved 5/16 statements.
# Failed to parse test___neg___with_undefined_money.
# Partially parsed test___neg___with_zero_quantity. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1
    var_4 = '-10.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '0'
    var_2 = 2023
    var_3 = 1



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_subtract_defined_money_same_currency. Retrieved 6/20 statements.
# Partially parsed test_subtract_defined_money_different_currency_raises. Retrieved 6/21 statements.
# Partially parsed test_subtract_first_operand_undefined. Retrieved 4/14 statements.
# Partially parsed test_subtract_second_operand_undefined. Retrieved 4/14 statements.
# Failed to parse test_subtract_both_operands_undefined.
# Partially parsed test_subtract_date_carried_forward. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '3'
    var_5 = '7'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '3'

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
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
    var_4 = '3'
    var_5 = 2



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_as_boolean_returns_false_for_undefined_price.
# Partially parsed test_as_boolean_returns_false_for_defined_price_with_zero_quantity. Retrieved 4/11 statements.
# Partially parsed test_as_boolean_returns_true_for_defined_price_with_positive_quantity. Retrieved 4/11 statements.
# Partially parsed test_as_boolean_returns_true_for_defined_price_with_negative_quantity. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '-1'
    var_2 = 2019
    var_3 = 1



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_divide_defined_price_by_positive_number. Retrieved 6/14 statements.
# Partially parsed test_divide_defined_price_by_one. Retrieved 6/14 statements.
# Partially parsed test_divide_defined_price_by_negative_number. Retrieved 7/15 statements.
# Partially parsed test_divide_defined_price_by_zero_yields_undefined. Retrieved 6/12 statements.
# Partially parsed test_divide_undefined_price_returns_itself. Retrieved 1/4 statements.
# Partially parsed test_divide_defined_price_by_decimal_fraction. Retrieved 7/15 statements.
# Partially parsed test_divide_defined_price_by_integer. Retrieved 7/14 statements.
# Partially parsed test_divide_defined_price_by_float. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'
    var_5 = '5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '7.5'
    var_2 = 2023
    var_3 = 2
    var_4 = 1
    var_5 = '1'

def test_case_0():
    var_0 = 'JPY'
    var_1 = '100'
    var_2 = 2023
    var_3 = 3
    var_4 = 1
    var_5 = '-2'
    var_6 = '-50'

def test_case_0():
    var_0 = 'GBP'
    var_1 = '15'
    var_2 = 2023
    var_3 = 4
    var_4 = 1
    var_5 = '0'

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = 'CAD'
    var_1 = '3'
    var_2 = 2023
    var_3 = 5
    var_4 = 1
    var_5 = '0.5'
    var_6 = '6'

def test_case_0():
    var_0 = 'AUD'
    var_1 = '9'
    var_2 = 2023
    var_3 = 6
    var_4 = 1
    var_5 = 3
    var_6 = '3'

def test_case_0():
    var_0 = 'CHF'
    var_1 = '8'
    var_2 = 2023
    var_3 = 7
    var_4 = 1
    var_5 = 2.0
    var_6 = '4'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test___eq__. Retrieved 10/39 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2019
    var_3 = 1
    var_4 = 2
    var_5 = '100'
    var_6 = '200'
    var_7 = None
    var_8 = 'string'
    var_9 = 123



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_qty_or_none_returns_qty_for_defined_money. Retrieved 5/15 statements.
# Partially parsed test_qty_or_none_returns_none_for_undefined_money. Retrieved 2/8 statements.
# Failed to parse test_qty_or_none_returns_none_for_na_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = '1.00'

def test_case_0():
    var_0 = None
    var_1 = '1'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_divide_defined_money_by_positive_number. Retrieved 6/18 statements.
# Partially parsed test_divide_defined_money_by_one. Retrieved 5/17 statements.
# Partially parsed test_divide_defined_money_by_negative_number. Retrieved 6/18 statements.
# Partially parsed test_divide_defined_money_by_zero_yields_undefined. Retrieved 5/15 statements.
# Partially parsed test_divide_undefined_money_returns_itself. Retrieved 1/6 statements.
# Partially parsed test_divide_defined_money_by_decimal_fraction. Retrieved 6/18 statements.
# Partially parsed test_divide_defined_money_by_integer. Retrieved 6/17 statements.
# Partially parsed test_divide_defined_money_by_float. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'
    var_5 = '5.00'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '7.50'
    var_2 = 2023
    var_3 = 1
    var_4 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '-2'
    var_5 = '-5.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '0'

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2023
    var_3 = 1
    var_4 = '0.5'
    var_5 = '2.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '9'
    var_2 = 2023
    var_3 = 1
    var_4 = 3
    var_5 = '3.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = 2.5
    var_5 = '4.00'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_abs_defined_positive. Retrieved 4/15 statements.
# Partially parsed test_abs_defined_negative. Retrieved 6/17 statements.
# Partially parsed test_abs_defined_zero. Retrieved 5/16 statements.
# Failed to parse test_abs_undefined.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'EUR'
    var_1 = '-200.75'
    var_2 = 2023
    var_3 = 2
    var_4 = 1
    var_5 = '200.75'

def test_case_0():
    var_0 = 'JPY'
    var_1 = '0'
    var_2 = 2023
    var_3 = 3
    var_4 = 1



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_convert_same_currency. Retrieved 6/15 statements.
# Partially parsed test_convert_different_currency_with_rate. Retrieved 11/27 statements.
# Partially parsed test_convert_different_currency_no_rate_strict_false. Retrieved 10/21 statements.
# Partially parsed test_convert_different_currency_no_rate_strict_true. Retrieved 12/24 statements.
# Partially parsed test_convert_with_asof_date. Retrieved 12/26 statements.
# Partially parsed test_convert_without_asof_date_uses_dov. Retrieved 12/25 statements.
# Partially parsed test_convert_fx_rate_service_not_set. Retrieved 8/18 statements.
# Partially parsed test_convert_fx_rate_service_raises_attribute_error. Retrieved 12/25 statements.


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
    var_5 = None
    var_6 = '0.85'
    var_7 = 2023
    var_8 = 1
    var_9 = '100.00'
    var_10 = '85.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = None
    var_6 = '100.00'
    var_7 = 2023
    var_8 = 1
    var_9 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = None
    var_6 = '100.00'
    var_7 = 2023
    var_8 = 1
    var_9 = 2023
    var_10 = 1
    var_11 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = None
    var_6 = '0.90'
    var_7 = '100.00'
    var_8 = 2023
    var_9 = 1
    var_10 = 6
    var_11 = '90.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = None
    var_6 = '0.80'
    var_7 = '100.00'
    var_8 = 2023
    var_9 = 5
    var_10 = 1
    var_11 = '80.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = None
    var_6 = ()
    var_7 = 'test'
    var_8 = AttributeError(var_7)
    var_9 = '100.00'
    var_10 = 2023
    var_11 = 1



# Parsed testcases at query #21
#--------------------------

# Partially parsed test___neg___with_defined_price. Retrieved 5/16 statements.
# Failed to parse test___neg___with_undefined_price.
# Partially parsed test___neg___with_zero_quantity. Retrieved 4/15 statements.
# Partially parsed test___neg___with_negative_quantity. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1
    var_4 = '-10.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '0'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'JPY'
    var_1 = '-15.75'
    var_2 = 2023
    var_3 = 1
    var_4 = '15.75'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_neg_positive_quantity. Retrieved 5/15 statements.
# Partially parsed test_neg_negative_quantity. Retrieved 5/15 statements.
# Partially parsed test_neg_zero_quantity. Retrieved 4/14 statements.
# Partially parsed test_neg_currency_preserved. Retrieved 5/15 statements.
# Partially parsed test_neg_date_preserved. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = 2023
    var_3 = 1
    var_4 = '-100.50'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '-200.75'
    var_2 = 2023
    var_3 = 1
    var_4 = '200.75'

def test_case_0():
    var_0 = 'GBP'
    var_1 = '0.00'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'JPY'
    var_1 = '500'
    var_2 = 2023
    var_3 = 1
    var_4 = '-500'

def test_case_0():
    var_0 = 'CAD'
    var_1 = '123.45'
    var_2 = 2022
    var_3 = 12
    var_4 = 31
    var_5 = '-123.45'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_lte_undefined_price_less_than_or_equal_to_defined_price. Retrieved 4/12 statements.
# Failed to parse test_lte_undefined_price_less_than_or_equal_to_undefined_price.
# Partially parsed test_lte_defined_price_less_than_or_equal_to_undefined_price. Retrieved 4/12 statements.
# Partially parsed test_lte_defined_price_less_than_or_equal_to_same_currency_lower_qty. Retrieved 5/16 statements.
# Partially parsed test_lte_defined_price_less_than_or_equal_to_same_currency_equal_qty. Retrieved 4/15 statements.
# Partially parsed test_lte_defined_price_less_than_or_equal_to_same_currency_higher_qty. Retrieved 5/16 statements.
# Partially parsed test_lte_raises_incompatible_currency_error. Retrieved 5/18 statements.


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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dov_or_none_with_defined_money. Retrieved 4/13 statements.
# Partially parsed test_dov_or_none_with_undefined_money. Retrieved 3/8 statements.
# Failed to parse test_dov_or_none_with_undefined_money_all_none.
# Partially parsed test_dov_or_none_with_defined_money_other_date. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1

def test_case_0():
    var_0 = None
    var_1 = 2019
    var_2 = 1

def test_case_0():
    var_0 = 'EUR'
    var_1 = '2.5'
    var_2 = 2020
    var_3 = 12
    var_4 = 31



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_ccy_or_returns_ccy_when_money_is_defined. Retrieved 5/14 statements.
# Partially parsed test_ccy_or_returns_default_when_money_is_undefined. Retrieved 3/9 statements.
# Partially parsed test_ccy_or_returns_default_when_ccy_is_none. Retrieved 5/13 statements.
# Partially parsed test_ccy_or_returns_default_when_qty_is_none. Retrieved 5/13 statements.
# Partially parsed test_ccy_or_returns_default_when_dov_is_none. Retrieved 4/11 statements.
# Partially parsed test_ccy_or_returns_ccy_for_defined_money_with_different_default. Retrieved 6/15 statements.
# Partially parsed test_ccy_or_returns_default_for_na_money_instance. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = 'EUR'

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = None
    var_2 = 2019
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = None
    var_3 = 'EUR'

def test_case_0():
    var_0 = 'GBP'
    var_1 = '100'
    var_2 = 2020
    var_3 = 5
    var_4 = 15
    var_5 = 'JPY'

def test_case_0():
    var_0 = 'CAD'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_as_integer_defined_price. Retrieved 4/13 statements.
# Failed to parse test_as_integer_undefined_price.
# Partially parsed test_as_integer_zero_quantity. Retrieved 4/13 statements.
# Partially parsed test_as_integer_negative_quantity. Retrieved 4/13 statements.
# Partially parsed test_as_integer_large_quantity. Retrieved 4/13 statements.
# Partially parsed test_as_integer_fractional_quantity_rounds_down. Retrieved 4/13 statements.
# Partially parsed test_as_integer_fractional_quantity_rounds_up. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '-5'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '123456789'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '3.14'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '2.99'
    var_2 = 2023
    var_3 = 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_constructor_with_valid_arguments. Retrieved 5/8 statements.
# Partially parsed test_constructor_with_zero_quantity. Retrieved 5/8 statements.
# Partially parsed test_constructor_with_negative_quantity. Retrieved 5/8 statements.
# Partially parsed test_constructor_with_different_currency. Retrieved 5/8 statements.
# Partially parsed test_constructor_with_different_date. Retrieved 6/9 statements.
# Partially parsed test_constructor_creates_namedtuple_subclass. Retrieved 6/11 statements.
# Partially parsed test_constructor_slots_are_empty. Retrieved 5/8 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.50'
    var_3 = 2023
    var_4 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = '0'
    var_3 = 2023
    var_4 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'GBP'
    var_1 = module_0.Currency(var_0)
    var_2 = '-50.75'
    var_3 = 2023
    var_4 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'JPY'
    var_1 = module_0.Currency(var_0)
    var_2 = '1000'
    var_3 = 2023
    var_4 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '200.00'
    var_3 = 2024
    var_4 = 12
    var_5 = 31

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.50'
    var_3 = 2023
    var_4 = 1
    var_5 = '_fields'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '100.50'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_floordiv_with_valid_numeric. Retrieved 8/15 statements.
# Partially parsed test_floordiv_with_zero_division. Retrieved 6/10 statements.
# Partially parsed test_floordiv_with_negative_numeric. Retrieved 8/15 statements.
# Partially parsed test_floordiv_with_decimal_numeric. Retrieved 8/16 statements.
# Partially parsed test_floordiv_with_invalid_operation. Retrieved 6/11 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Currency(var_0)
    var_7 = '5'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1
    var_5 = 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1
    var_5 = -2
    var_6 = module_0.Currency(var_0)
    var_7 = '-6'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1
    var_5 = '2.5'
    var_6 = module_0.Currency(var_0)
    var_7 = '4'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1
    var_5 = 'NaN'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_floordiv_defined_price_with_positive_divisor. Retrieved 5/17 statements.
# Partially parsed test_floordiv_defined_price_with_negative_divisor. Retrieved 6/18 statements.
# Partially parsed test_floordiv_defined_price_with_zero_divisor. Retrieved 5/15 statements.
# Partially parsed test_floordiv_undefined_price. Retrieved 1/6 statements.
# Partially parsed test_floordiv_defined_price_with_integer_divisor. Retrieved 6/17 statements.
# Partially parsed test_floordiv_defined_price_with_float_divisor. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '3'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '-3'
    var_5 = '-4'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '0'

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = 'JPY'
    var_1 = '7'
    var_2 = 2023
    var_3 = 1
    var_4 = 2
    var_5 = '3'

def test_case_0():
    var_0 = 'GBP'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = 3.0
    var_5 = '3'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_with_dov_returns_new_money_with_given_dov_when_defined. Retrieved 6/19 statements.
# Partially parsed test_with_dov_returns_itself_when_undefined. Retrieved 3/7 statements.
# Partially parsed test_with_dov_preserves_other_attributes. Retrieved 7/21 statements.
# Partially parsed test_with_dov_with_same_dov_returns_equal_money. Retrieved 5/14 statements.
# Partially parsed test_with_dov_on_undefined_money_does_not_change_undefined_state. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '100.50'
    var_2 = 2023
    var_3 = 1
    var_4 = 12
    var_5 = 31

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31

def test_case_0():
    var_0 = 'EUR'
    var_1 = '200.75'
    var_2 = 2022
    var_3 = 5
    var_4 = 15
    var_5 = 6
    var_6 = 20

def test_case_0():
    var_0 = 2023
    var_1 = 7
    var_2 = 4
    var_3 = 'GBP'
    var_4 = '150.25'

def test_case_0():
    var_0 = 2024
    var_1 = 1



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_convert_same_currency. Retrieved 6/15 statements.
# Partially parsed test_convert_different_currency_with_rate. Retrieved 11/27 statements.
# Partially parsed test_convert_strict_mode_no_rate. Retrieved 12/24 statements.
# Partially parsed test_convert_non_strict_mode_no_rate. Retrieved 10/21 statements.
# Partially parsed test_convert_uses_asof_date. Retrieved 11/24 statements.
# Partially parsed test_convert_default_asof_is_dov. Retrieved 10/22 statements.
# Partially parsed test_convert_no_fx_service_raises_error. Retrieved 8/18 statements.
# Partially parsed test_convert_quantizes_result. Retrieved 12/28 statements.


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
    var_8 = None
    var_9 = '0.85'
    var_10 = '85.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = None
    var_9 = 2023
    var_10 = 1
    var_11 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = None
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
    var_8 = None
    var_9 = '0.85'
    var_10 = 6

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1
    var_8 = None
    var_9 = '0.85'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '100.00'
    var_6 = 2023
    var_7 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = '100.555'
    var_7 = 2023
    var_8 = 1
    var_9 = None
    var_10 = '110.123'
    var_11 = '11076'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_truediv_with_defined_money_and_non_zero_divisor. Retrieved 6/18 statements.
# Partially parsed test_truediv_with_defined_money_and_zero_divisor. Retrieved 5/15 statements.
# Partially parsed test_truediv_with_undefined_money. Retrieved 1/6 statements.
# Partially parsed test_truediv_with_defined_money_and_integer_divisor. Retrieved 6/17 statements.
# Partially parsed test_truediv_with_defined_money_and_float_divisor. Retrieved 6/17 statements.
# Partially parsed test_truediv_ensures_quantization. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'
    var_5 = '5.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '0'

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = 'USD'
    var_1 = '9'
    var_2 = 2023
    var_3 = 1
    var_4 = 3
    var_5 = '3.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = 2023
    var_3 = 1
    var_4 = 2.5
    var_5 = '2.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2023
    var_3 = 1
    var_4 = '3'
    var_5 = '0.33'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test___ge___returns_true_when_self_qty_greater_than_other_qty. Retrieved 6/12 statements.
# Partially parsed test___ge___returns_true_when_self_qty_equal_to_other_qty. Retrieved 5/10 statements.
# Partially parsed test___ge___returns_false_when_self_qty_less_than_other_qty. Retrieved 6/12 statements.
# Partially parsed test___ge___raises_IncompatibleCurrencyError_when_currencies_differ. Retrieved 7/13 statements.
# Partially parsed test___ge___returns_True_when_other_is_not_SomePrice. Retrieved 5/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = '5.5'
    var_4 = 2023
    var_5 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '5.5'
    var_3 = '10.5'
    var_4 = 2023
    var_5 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = '10.5'
    var_5 = 2023
    var_6 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #13
#--------------------------

# Partially parsed test___ge___with_defined_money_same_currency_greater. Retrieved 5/14 statements.
# Partially parsed test___ge___with_defined_money_same_currency_equal. Retrieved 4/13 statements.
# Partially parsed test___ge___with_defined_money_same_currency_less. Retrieved 5/14 statements.
# Partially parsed test___ge___with_defined_money_different_currency. Retrieved 6/16 statements.
# Partially parsed test___ge___with_undefined_money_and_defined_money. Retrieved 4/10 statements.
# Partially parsed test___ge___with_defined_money_and_undefined_money. Retrieved 4/10 statements.
# Failed to parse test___ge___with_both_undefined_money.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2020
    var_3 = 1
    var_4 = '5'

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = 2020
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
    var_2 = 2020
    var_3 = 1
    var_4 = '5'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2020
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '5'

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = 2020
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = 2020
    var_3 = 1



# Parsed testcases at query #14
#--------------------------

# Partially parsed test___add___same_currency_and_dates. Retrieved 7/15 statements.
# Partially parsed test___add___same_currency_different_dates_later_dov1. Retrieved 8/17 statements.
# Partially parsed test___add___same_currency_different_dates_later_dov2. Retrieved 8/17 statements.
# Partially parsed test___add___different_currency_raises_incompatible_currency_error. Retrieved 8/15 statements.
# Partially parsed test___add___with_undefined_price_returns_self. Retrieved 5/10 statements.
# Partially parsed test___add___commutative_property. Retrieved 6/13 statements.
# Partially parsed test___add___zero_quantity. Retrieved 6/14 statements.
# Partially parsed test___add___negative_quantity. Retrieved 7/15 statements.
# Partially parsed test___add___large_quantities. Retrieved 7/15 statements.
# Partially parsed test___add___using_operator_overload. Retrieved 7/15 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = '5.2'
    var_4 = 2023
    var_5 = 1
    var_6 = '15.7'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = '5.2'
    var_4 = 2023
    var_5 = 1
    var_6 = 2
    var_7 = '15.7'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = '5.2'
    var_4 = 2023
    var_5 = 1
    var_6 = 2
    var_7 = '15.7'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = '10.5'
    var_5 = '5.2'
    var_6 = 2023
    var_7 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = '5.2'
    var_4 = 2023
    var_5 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = '0'
    var_4 = 2023
    var_5 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = '-5.2'
    var_4 = 2023
    var_5 = 1
    var_6 = '5.3'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '999999.99'
    var_3 = '0.01'
    var_4 = 2023
    var_5 = 1
    var_6 = '1000000.00'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = '10.5'
    var_3 = '5.2'
    var_4 = 2023
    var_5 = 1
    var_6 = '15.7'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_le_with_defined_prices_same_currency. Retrieved 5/18 statements.
# Partially parsed test_le_with_defined_prices_same_currency_equal. Retrieved 4/17 statements.
# Partially parsed test_le_with_defined_prices_same_currency_greater. Retrieved 5/18 statements.
# Partially parsed test_le_with_undefined_price_left. Retrieved 4/14 statements.
# Partially parsed test_le_with_undefined_price_right. Retrieved 4/14 statements.
# Failed to parse test_le_with_both_undefined.
# Partially parsed test_le_raises_incompatible_currency_error. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '20'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1

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
    var_4 = 'EUR'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_divide_defined_price_by_positive_number. Retrieved 6/21 statements.
# Partially parsed test_divide_defined_price_by_one. Retrieved 5/20 statements.
# Partially parsed test_divide_defined_price_by_negative_number. Retrieved 6/21 statements.
# Partially parsed test_divide_defined_price_by_zero_yields_undefined. Retrieved 5/15 statements.
# Partially parsed test_divide_undefined_price_returns_itself. Retrieved 1/6 statements.
# Partially parsed test_divide_defined_price_by_decimal_fraction. Retrieved 6/21 statements.
# Partially parsed test_divide_defined_price_by_large_number. Retrieved 5/20 statements.
# Partially parsed test_divide_defined_price_with_negative_quantity. Retrieved 6/21 statements.
# Partially parsed test_divide_defined_price_by_float_like_decimal. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'
    var_5 = '5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '7.5'
    var_2 = 2023
    var_3 = 1
    var_4 = '1'

def test_case_0():
    var_0 = 'JPY'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1
    var_4 = '-2'
    var_5 = '-50'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '0'

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = 'GBP'
    var_1 = '1'
    var_2 = 2023
    var_3 = 1
    var_4 = '0.5'
    var_5 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1000'
    var_2 = 2023
    var_3 = 1
    var_4 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '-8'
    var_2 = 2023
    var_3 = 1
    var_4 = '4'
    var_5 = '-2'

def test_case_0():
    var_0 = 'USD'
    var_1 = '9'
    var_2 = 2023
    var_3 = 1
    var_4 = '2.5'
    var_5 = '3.6'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_subtract_defined_prices_same_currency. Retrieved 6/21 statements.
# Partially parsed test_subtract_defined_prices_different_currency_raises. Retrieved 6/20 statements.
# Partially parsed test_subtract_first_undefined_returns_second. Retrieved 4/14 statements.
# Partially parsed test_subtract_second_undefined_returns_first. Retrieved 4/14 statements.
# Failed to parse test_subtract_both_undefined_returns_undefined.
# Partially parsed test_subtract_negative_result. Retrieved 6/21 statements.
# Partially parsed test_subtract_zero_result. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = '3'
    var_5 = '7'

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1
    var_4 = 'EUR'
    var_5 = '3'

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '10'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = '3'
    var_2 = 2023
    var_3 = 1
    var_4 = '10'
    var_5 = '-7'

def test_case_0():
    var_0 = 'USD'
    var_1 = '5'
    var_2 = 2023
    var_3 = 1
    var_4 = '0'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_qty_map_defined_money. Retrieved 6/20 statements.
# Partially parsed test_qty_map_undefined_money. Retrieved 3/15 statements.
# Partially parsed test_qty_map_defined_money_with_different_return_type. Retrieved 7/16 statements.
# Partially parsed test_qty_map_undefined_money_with_different_return_type. Retrieved 5/12 statements.
# Partially parsed test_qty_map_defined_money_zero_quantity. Retrieved 7/21 statements.
# Partially parsed test_qty_map_undefined_money_with_none_quantity. Retrieved 6/19 statements.


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

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = 2019
    var_3 = 1
    var_4 = lambda x: str(x)
    var_5 = 'fallback'
    var_6 = lambda : var_5

def test_case_0():
    var_0 = None
    var_1 = '1'
    var_2 = lambda x: str(x)
    var_3 = 'fallback'
    var_4 = lambda : var_3

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = 2019
    var_3 = 1
    var_4 = '2'
    var_5 = '99'
    var_6 = '0.00'

def test_case_0():
    var_0 = 'USD'
    var_1 = None
    var_2 = 2019
    var_3 = 1
    var_4 = '1'
    var_5 = '100'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_as_float_defined_money. Retrieved 5/14 statements.
# Failed to parse test_as_float_undefined_money.
# Partially parsed test_as_float_zero_quantity. Retrieved 5/14 statements.
# Partially parsed test_as_float_negative_quantity. Retrieved 5/14 statements.
# Partially parsed test_as_float_large_quantity. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = '123.456'
    var_2 = 2023
    var_3 = 1
    var_4 = 123.456

def test_case_0():
    var_0 = 'USD'
    var_1 = '0'
    var_2 = 2023
    var_3 = 1
    var_4 = 0.0

def test_case_0():
    var_0 = 'USD'
    var_1 = '-123.456'
    var_2 = 2023
    var_3 = 1
    var_4 = -123.456

def test_case_0():
    var_0 = 'USD'
    var_1 = '999999.999'
    var_2 = 2023
    var_3 = 1
    var_4 = 999999.999



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_convert_defined_price_with_valid_fx_rate. Retrieved 8/26 statements.
# Partially parsed test_convert_undefined_price_returns_itself. Retrieved 1/6 statements.
# Partially parsed test_convert_same_currency_returns_same_price. Retrieved 4/14 statements.
# Partially parsed test_convert_without_asof_uses_price_dov. Retrieved 7/27 statements.
# Partially parsed test_convert_with_asof_overrides_dov. Retrieved 8/29 statements.
# Partially parsed test_convert_strict_true_raises_on_missing_fx_rate. Retrieved 9/25 statements.
# Partially parsed test_convert_strict_false_returns_undefined_on_missing_fx_rate. Retrieved 7/22 statements.
# Partially parsed test_convert_uses_default_fx_service_when_not_provided. Retrieved 5/16 statements.
# Partially parsed test_convert_zero_quantity_returns_zero_in_target_currency. Retrieved 6/22 statements.
# Partially parsed test_convert_negative_quantity_converts_correctly. Retrieved 7/23 statements.


def test_case_0():
    var_0 = '0.85'
    var_1 = 'USD'
    var_2 = '100'
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = True
    var_7 = '85'

def test_case_0():
    var_0 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = '0.85'
    var_1 = 'USD'
    var_2 = '100'
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '85'

def test_case_0():
    var_0 = '0.90'
    var_1 = 'USD'
    var_2 = '100'
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = 6
    var_7 = '90'

def test_case_0():
    var_0 = 'Rate not found'
    var_1 = 'USD'
    var_2 = '100'
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = 2023
    var_7 = 1
    var_8 = True

def test_case_0():
    var_0 = 'Rate not found'
    var_1 = 'USD'
    var_2 = '100'
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = '100'
    var_2 = 2023
    var_3 = 1
    var_4 = 'EUR'

def test_case_0():
    var_0 = '0.85'
    var_1 = 'USD'
    var_2 = '0'
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'

def test_case_0():
    var_0 = '0.85'
    var_1 = 'USD'
    var_2 = '-100'
    var_3 = 2023
    var_4 = 1
    var_5 = 'EUR'
    var_6 = '-85'



