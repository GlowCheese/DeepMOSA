####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_invert_swaps_currencies_and_inverts_value. Retrieved 6/17 statements.
# Partially parsed test_invert_twice_returns_original. Retrieved 5/15 statements.
# Partially parsed test_invert_with_fractional_value. Retrieved 6/18 statements.
# Partially parsed test_invert_with_one. Retrieved 5/15 statements.
# Partially parsed test_invert_uses_indexed_access. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'
    var_5 = '0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '0.75'
    var_5 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_multiple_queries. Retrieved 20/36 statements.
# Partially parsed test_queries_with_strict_true_raises_lookup_error_on_missing_rate. Retrieved 9/16 statements.
# Partially parsed test_queries_with_empty_iterable_returns_empty_iterable. Retrieved 2/6 statements.
# Partially parsed test_queries_calls_with_correct_parameters. Retrieved 8/16 statements.
# Partially parsed test_queries_returns_iterable_of_same_length_as_input. Retrieved 20/35 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.1'
    var_1 = '1.2'
    var_2 = None
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = 2023
    var_8 = 1
    var_9 = 'GBP'
    var_10 = module_0.Currency(var_9)
    var_11 = 'JPY'
    var_12 = module_0.Currency(var_11)
    var_13 = 2
    var_14 = 'AUD'
    var_15 = module_0.Currency(var_14)
    var_16 = 'CAD'
    var_17 = module_0.Currency(var_16)
    var_18 = 3
    var_19 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'FX rate not found'
    var_1 = 'USD'
    var_2 = module_0.Currency(var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = True
    var_8 = list(var_1)

def test_case_0():
    var_0 = []
    var_1 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = True
    var_7 = True

import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.0'
    var_1 = '2.0'
    var_2 = '3.0'
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = 2023
    var_8 = 1
    var_9 = 'GBP'
    var_10 = module_0.Currency(var_9)
    var_11 = 'JPY'
    var_12 = module_0.Currency(var_11)
    var_13 = 2
    var_14 = 'AUD'
    var_15 = module_0.Currency(var_14)
    var_16 = 'CAD'
    var_17 = module_0.Currency(var_16)
    var_18 = 3
    var_19 = False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_query_returns_fxrate_for_valid_currency_pair_and_date. Retrieved 8/16 statements.
# Partially parsed test_query_returns_none_for_nonexistent_pair_when_strict_false. Retrieved 8/15 statements.
# Partially parsed test_query_raises_error_for_nonexistent_pair_when_strict_true. Retrieved 8/16 statements.
# Partially parsed test_query_handles_same_currency_pair. Retrieved 7/15 statements.
# Partially parsed test_query_returns_correct_rate_for_inverse_pair. Retrieved 9/20 statements.
# Partially parsed test_query_handles_different_currency_types. Retrieved 9/17 statements.
# Partially parsed test_query_with_future_date_returns_none. Retrieved 8/15 statements.
# Partially parsed test_query_with_past_date_returns_rate. Retrieved 8/16 statements.
# Partially parsed test_query_handles_currency_with_negative_decimals. Retrieved 9/17 statements.
# Partially parsed test_query_handles_currency_with_zero_decimals. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'XYZ'
    var_4 = 'Unknown'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'XYZ'
    var_4 = 'Unknown'
    var_5 = 2023
    var_6 = 1
    var_7 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2023
    var_4 = 1
    var_5 = False
    var_6 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False
    var_8 = '1'

def test_case_0():
    var_0 = 'BTC'
    var_1 = 'Bitcoin'
    var_2 = 8
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2050
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2000
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird'
    var_2 = -1
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = 2023
    var_7 = 1
    var_8 = False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_multiple_queries. Retrieved 12/25 statements.
# Partially parsed test_queries_raises_lookup_error_in_strict_mode_when_rate_missing. Retrieved 8/14 statements.
# Partially parsed test_queries_returns_none_for_missing_rate_in_non_strict_mode. Retrieved 7/13 statements.
# Partially parsed test_queries_handles_empty_queries_iterable. Retrieved 2/6 statements.
# Partially parsed test_queries_preserves_order_of_input_queries. Retrieved 13/26 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = 'GBP'
    var_7 = module_0.Currency(var_6)
    var_8 = 'JPY'
    var_9 = module_0.Currency(var_8)
    var_10 = 2
    var_11 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'XYZ'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = True
    var_7 = list(var_1)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'XYZ'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = False

def test_case_0():
    var_0 = []
    var_1 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = module_0.Currency(var_2)
    var_7 = module_0.Currency(var_0)
    var_8 = False
    var_9 = module_0.Currency(var_0)
    var_10 = module_0.Currency(var_2)
    var_11 = module_0.Currency(var_2)
    var_12 = module_0.Currency(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_fxrate_constructor_creates_instance_with_correct_attributes. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_allows_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_accepts_same_currency_with_value_one. Retrieved 2/9 statements.
# Partially parsed test_fxrate_constructor_accepts_positive_decimal_value. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_does_not_validate_input. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_does_not_enforce_same_currency_rule. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '2'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_multiple_queries. Retrieved 13/35 statements.
# Partially parsed test_queries_returns_none_for_missing_rates_when_strict_false. Retrieved 7/18 statements.
# Partially parsed test_queries_raises_error_for_missing_rates_when_strict_true. Retrieved 8/25 statements.
# Partially parsed test_queries_handles_empty_queries_list. Retrieved 1/9 statements.
# Partially parsed test_queries_preserves_order_of_input_queries. Retrieved 19/42 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = 'GBP'
    var_7 = module_0.Currency(var_6)
    var_8 = 'JPY'
    var_9 = module_0.Currency(var_8)
    var_10 = 2
    var_11 = '0.85'
    var_12 = '150.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = True
    var_7 = list(var_1)

def test_case_0():
    var_0 = []

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'C'
    var_1 = module_0.Currency(var_0)
    var_2 = 'D'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = 2
    var_7 = 'A'
    var_8 = module_0.Currency(var_7)
    var_9 = 'B'
    var_10 = module_0.Currency(var_9)
    var_11 = 'E'
    var_12 = module_0.Currency(var_11)
    var_13 = 'F'
    var_14 = module_0.Currency(var_13)
    var_15 = 3
    var_16 = '2.2'
    var_17 = '1.1'
    var_18 = '3.3'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 12/21 statements.
# Partially parsed test_queries_with_strict_mode_raises_error. Retrieved 8/14 statements.
# Partially parsed test_queries_with_strict_mode_false_returns_none. Retrieved 7/14 statements.
# Partially parsed test_queries_empty_list. Retrieved 2/6 statements.
# Partially parsed test_queries_consistent_with_single_query. Retrieved 9/17 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = 'GBP'
    var_7 = module_0.Currency(var_6)
    var_8 = 'JPY'
    var_9 = module_0.Currency(var_8)
    var_10 = 2
    var_11 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'XYZ'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = True
    var_7 = list(var_1)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'XYZ'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = False

def test_case_0():
    var_0 = []
    var_1 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = False
    var_7 = module_0.Currency(var_0)
    var_8 = module_0.Currency(var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_query_returns_fxrate_for_valid_currency_pair_and_date. Retrieved 8/16 statements.
# Partially parsed test_query_raises_lookup_error_in_strict_mode_when_rate_not_found. Retrieved 8/16 statements.
# Partially parsed test_query_returns_none_in_non_strict_mode_when_rate_not_found. Retrieved 8/15 statements.
# Partially parsed test_query_handles_same_currency_pair. Retrieved 6/12 statements.
# Partially parsed test_query_uses_default_service_when_available. Retrieved 8/16 statements.
# Partially parsed test_query_handles_different_currency_types. Retrieved 9/17 statements.
# Partially parsed test_query_handles_historical_date. Retrieved 8/16 statements.
# Partially parsed test_query_handles_future_date. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2023
    var_4 = 1
    var_5 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'BTC'
    var_4 = 'Bitcoin'
    var_5 = 8
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2000
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2100
    var_6 = 1
    var_7 = False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_query_returns_fxrate_for_valid_currency_pair_and_date. Retrieved 8/16 statements.
# Partially parsed test_query_returns_none_for_nonexistent_pair_when_not_strict. Retrieved 9/16 statements.
# Partially parsed test_query_raises_error_for_nonexistent_pair_when_strict. Retrieved 9/17 statements.
# Partially parsed test_query_handles_same_currency_pair. Retrieved 7/13 statements.
# Partially parsed test_query_returns_consistent_fxrate_for_same_inputs. Retrieved 8/16 statements.
# Partially parsed test_query_handles_different_dates. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2023
    var_4 = 1
    var_5 = False
    var_6 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_query_returns_fx_rate_for_currency_pair_and_date. Retrieved 8/16 statements.
# Partially parsed test_query_returns_none_when_rate_not_found_and_strict_false. Retrieved 8/15 statements.
# Partially parsed test_query_raises_error_when_rate_not_found_and_strict_true. Retrieved 8/16 statements.
# Partially parsed test_query_handles_same_currency_pair. Retrieved 6/12 statements.
# Partially parsed test_query_handles_different_currency_types. Retrieved 9/17 statements.
# Partially parsed test_query_handles_currency_with_negative_decimals. Retrieved 9/17 statements.
# Partially parsed test_query_handles_currency_with_zero_decimals. Retrieved 9/17 statements.
# Partially parsed test_query_handles_different_dates. Retrieved 10/21 statements.
# Partially parsed test_query_handles_reverse_currency_pair. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2023
    var_4 = 1
    var_5 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'BTC'
    var_4 = 'Bitcoin'
    var_5 = 8
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Currency'
    var_2 = -1
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 31
    var_9 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_constructor_creates_valid_fxrate. Retrieved 3/8 statements.
# Partially parsed test_constructor_allows_indexed_access. Retrieved 3/8 statements.
# Partially parsed test_constructor_creates_fxrate_with_same_currency. Retrieved 2/7 statements.
# Partially parsed test_constructor_creates_fxrate_with_decimal_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_creates_fxrate_with_negative_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_creates_fxrate_with_zero_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_creates_fxrate_with_future_date. Retrieved 4/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_past_date. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.2345'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 1
    var_3 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 1
    var_3 = '2'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 3/11 statements.
# Partially parsed test_constructor_allows_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_constructor_creates_fxrate_with_same_currency_and_value_one. Retrieved 2/9 statements.
# Partially parsed test_constructor_creates_fxrate_with_positive_decimal_value. Retrieved 3/12 statements.
# Partially parsed test_constructor_creates_fxrate_with_large_decimal_value. Retrieved 3/12 statements.
# Partially parsed test_constructor_creates_fxrate_with_future_date. Retrieved 6/15 statements.
# Partially parsed test_constructor_creates_fxrate_with_past_date. Retrieved 5/14 statements.
# Partially parsed test_constructor_creates_fxrate_with_different_currency_pair. Retrieved 3/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_value_one_for_same_currency. Retrieved 2/10 statements.
# Partially parsed test_constructor_creates_fxrate_with_indexed_access. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '12345.6789'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2100
    var_3 = 12
    var_4 = 31
    var_5 = '1.2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2000
    var_3 = 1
    var_4 = '0.8'

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = '150.75'

def test_case_0():
    var_0 = 'CAD'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.1'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 3/8 statements.
# Partially parsed test_constructor_allows_unpacking. Retrieved 3/9 statements.
# Partially parsed test_constructor_accepts_same_currency_with_value_one. Retrieved 2/6 statements.
# Partially parsed test_constructor_accepts_same_currency_with_value_one_decimal. Retrieved 2/6 statements.
# Partially parsed test_constructor_accepts_positive_decimal_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_accepts_large_positive_decimal_value. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1.0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '123456.789'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_constructor_creates_valid_instance. Retrieved 3/13 statements.
# Partially parsed test_constructor_allows_indexed_access. Retrieved 3/14 statements.
# Partially parsed test_constructor_creates_same_currency_rate. Retrieved 2/11 statements.
# Partially parsed test_constructor_accepts_decimal_value. Retrieved 3/12 statements.
# Partially parsed test_constructor_accepts_future_date. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.2345'

def test_case_0():
    var_0 = 10
    var_1 = 'EUR'
    var_2 = 'USD'
    var_3 = '1.5'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_fxrate_constructor_creates_instance_with_correct_attributes. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_allows_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_creates_instance_with_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_accepts_same_currency_with_value_one. Retrieved 2/10 statements.
# Partially parsed test_fxrate_constructor_accepts_same_currency_with_value_not_one. Retrieved 2/10 statements.
# Partially parsed test_fxrate_constructor_accepts_zero_value. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_accepts_negative_value. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-1'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_constructor_creates_valid_fxrate. Retrieved 3/8 statements.
# Partially parsed test_constructor_allows_indexed_access. Retrieved 3/8 statements.
# Partially parsed test_constructor_accepts_same_currency_with_one. Retrieved 2/7 statements.
# Partially parsed test_constructor_accepts_same_currency_with_non_one. Retrieved 2/7 statements.
# Partially parsed test_constructor_accepts_zero_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_accepts_negative_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_accepts_non_currency_for_ccy1. Retrieved 3/7 statements.
# Partially parsed test_constructor_accepts_non_currency_for_ccy2. Retrieved 3/7 statements.
# Partially parsed test_constructor_accepts_non_date_for_date. Retrieved 4/8 statements.
# Partially parsed test_constructor_accepts_non_decimal_for_value. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2023-01-01'
    var_3 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2.0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 3/11 statements.
# Partially parsed test_constructor_allows_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_constructor_creates_fxrate_with_same_currency_and_value_one. Retrieved 2/9 statements.
# Partially parsed test_constructor_creates_fxrate_with_positive_decimal_value. Retrieved 3/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_different_currencies_and_value_not_one. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_query_returns_fxrate_for_valid_currency_pair_and_date. Retrieved 6/19 statements.
# Partially parsed test_query_returns_none_for_missing_fxrate_when_strict_false. Retrieved 6/18 statements.
# Partially parsed test_query_raises_error_for_missing_fxrate_when_strict_true. Retrieved 6/21 statements.
# Partially parsed test_query_handles_same_currency_pair. Retrieved 4/17 statements.
# Partially parsed test_query_handles_different_currency_types. Retrieved 7/20 statements.
# Partially parsed test_query_handles_historical_date. Retrieved 8/21 statements.
# Partially parsed test_query_handles_future_date. Retrieved 8/21 statements.
# Partially parsed test_query_handles_currency_with_zero_decimals. Retrieved 7/20 statements.
# Partially parsed test_query_handles_currency_with_negative_decimals. Retrieved 7/20 statements.
# Partially parsed test_query_handles_reverse_currency_pair. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '1.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = '1.0'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'BTC'
    var_4 = 'Bitcoin'
    var_5 = 8
    var_6 = '2.0'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2020
    var_6 = 1
    var_7 = '0.9'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2030
    var_6 = 1
    var_7 = '1.1'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = 'USD'
    var_4 = 'US Dollar'
    var_5 = 2
    var_6 = '100.0'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Crypto'
    var_2 = -1
    var_3 = 'USD'
    var_4 = 'US Dollar'
    var_5 = 2
    var_6 = '0.000001'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '0.85'
    var_6 = '1.176470588235294'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 26/50 statements.
# Partially parsed test_queries_with_strict_raises_error. Retrieved 20/36 statements.
# Partially parsed test_queries_empty_iterable. Retrieved 1/6 statements.
# Partially parsed test_queries_handles_single_query. Retrieved 11/23 statements.
# Partially parsed test_queries_strict_false_returns_none_for_missing. Retrieved 26/50 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.1'
    var_1 = '1.2'
    var_2 = None
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = 2023
    var_8 = 1
    var_9 = 'GBP'
    var_10 = module_0.Currency(var_9)
    var_11 = 'JPY'
    var_12 = module_0.Currency(var_11)
    var_13 = 2
    var_14 = 'AUD'
    var_15 = module_0.Currency(var_14)
    var_16 = 'CAD'
    var_17 = module_0.Currency(var_16)
    var_18 = 3
    var_19 = module_0.Currency(var_3)
    var_20 = module_0.Currency(var_5)
    var_21 = False
    var_22 = module_0.Currency(var_9)
    var_23 = module_0.Currency(var_11)
    var_24 = module_0.Currency(var_14)
    var_25 = module_0.Currency(var_16)

import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.1'
    var_1 = 'Rate not found'
    var_2 = '1.3'
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = 2023
    var_8 = 1
    var_9 = 'GBP'
    var_10 = module_0.Currency(var_9)
    var_11 = 'JPY'
    var_12 = module_0.Currency(var_11)
    var_13 = 2
    var_14 = 'AUD'
    var_15 = module_0.Currency(var_14)
    var_16 = 'CAD'
    var_17 = module_0.Currency(var_16)
    var_18 = 3
    var_19 = True

def test_case_0():
    var_0 = []

import pypara.currencies as module_0

def test_case_0():
    var_0 = '0.85'
    var_1 = 'EUR'
    var_2 = module_0.Currency(var_1)
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 2023
    var_6 = 5
    var_7 = 10
    var_8 = module_0.Currency(var_1)
    var_9 = module_0.Currency(var_3)
    var_10 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.5'
    var_1 = None
    var_2 = '2.0'
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 'GBP'
    var_6 = module_0.Currency(var_5)
    var_7 = 2023
    var_8 = 2
    var_9 = 1
    var_10 = 'JPY'
    var_11 = module_0.Currency(var_10)
    var_12 = 'EUR'
    var_13 = module_0.Currency(var_12)
    var_14 = 'CAD'
    var_15 = module_0.Currency(var_14)
    var_16 = 'AUD'
    var_17 = module_0.Currency(var_16)
    var_18 = 3
    var_19 = False
    var_20 = module_0.Currency(var_3)
    var_21 = module_0.Currency(var_5)
    var_22 = module_0.Currency(var_10)
    var_23 = module_0.Currency(var_12)
    var_24 = module_0.Currency(var_14)
    var_25 = module_0.Currency(var_16)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_fxrate_constructor_creates_instance_with_correct_attributes. Retrieved 5/13 statements.
# Partially parsed test_fxrate_constructor_allows_unpacking. Retrieved 5/14 statements.
# Partially parsed test_fxrate_constructor_accepts_same_currency_with_value_one. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_accepts_positive_decimal_value. Retrieved 5/13 statements.
# Partially parsed test_fxrate_constructor_accepts_large_positive_decimal_value. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 2023
    var_2 = 1
    var_3 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1000000.123456'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 3/11 statements.
# Partially parsed test_constructor_allows_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_same_currency_and_value_one. Retrieved 2/9 statements.
# Partially parsed test_constructor_creates_fxrate_with_positive_decimal_value. Retrieved 3/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_large_decimal_value. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '12345.6789'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_query_returns_fxrate_for_valid_currency_pair_and_date. Retrieved 8/16 statements.
# Partially parsed test_query_returns_none_for_nonexistent_pair_when_strict_false. Retrieved 9/16 statements.
# Partially parsed test_query_raises_error_for_nonexistent_pair_when_strict_true. Retrieved 9/17 statements.
# Partially parsed test_query_returns_same_rate_for_inverse_currency_pair. Retrieved 8/17 statements.
# Partially parsed test_query_returns_none_for_same_currency_pair. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2023
    var_4 = 1
    var_5 = False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_constructor_creates_valid_fxrate. Retrieved 3/8 statements.
# Partially parsed test_constructor_allows_indexed_access. Retrieved 3/8 statements.
# Partially parsed test_constructor_creates_invertible_fxrate. Retrieved 4/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_same_currency. Retrieved 2/7 statements.
# Partially parsed test_constructor_creates_fxrate_with_decimal_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_creates_fxrate_with_future_date. Retrieved 4/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_past_date. Retrieved 4/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_high_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_creates_fxrate_with_small_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_creates_fxrate_with_negative_value. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = '0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.2345'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 1
    var_3 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 1
    var_3 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1000000'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0.000001'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-2'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_query_returns_fx_rate_for_valid_currency_pair_and_date. Retrieved 6/20 statements.
# Partially parsed test_query_returns_none_for_missing_rate_when_strict_false. Retrieved 6/18 statements.
# Partially parsed test_query_raises_error_for_missing_rate_when_strict_true. Retrieved 6/21 statements.
# Partially parsed test_query_handles_same_currency_pair. Retrieved 4/18 statements.
# Partially parsed test_query_uses_correct_currency_order. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '1.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '0.5'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_multiple_queries. Retrieved 18/34 statements.
# Partially parsed test_queries_raises_error_in_strict_mode_when_rate_missing. Retrieved 13/25 statements.
# Partially parsed test_queries_handles_empty_queries_list. Retrieved 2/5 statements.
# Partially parsed test_queries_calls_query_with_correct_parameters. Retrieved 11/20 statements.
# Partially parsed test_queries_passes_strict_flag_to_query_calls. Retrieved 11/20 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.1'
    var_1 = '1.2'
    var_2 = None
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = 2023
    var_8 = 1
    var_9 = 'GBP'
    var_10 = module_0.Currency(var_9)
    var_11 = module_0.Currency(var_3)
    var_12 = 2
    var_13 = 'JPY'
    var_14 = module_0.Currency(var_13)
    var_15 = module_0.Currency(var_5)
    var_16 = 3
    var_17 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.1'
    var_1 = 'Rate not found'
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 'EUR'
    var_5 = module_0.Currency(var_4)
    var_6 = 2023
    var_7 = 1
    var_8 = 'GBP'
    var_9 = module_0.Currency(var_8)
    var_10 = module_0.Currency(var_2)
    var_11 = 2
    var_12 = True

def test_case_0():
    var_0 = []
    var_1 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.5'
    var_1 = 'AUD'
    var_2 = module_0.Currency(var_1)
    var_3 = 'CAD'
    var_4 = module_0.Currency(var_3)
    var_5 = 2023
    var_6 = 5
    var_7 = 10
    var_8 = False
    var_9 = module_0.Currency(var_1)
    var_10 = module_0.Currency(var_3)

import pypara.currencies as module_0

def test_case_0():
    var_0 = '0.9'
    var_1 = 'CHF'
    var_2 = module_0.Currency(var_1)
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 2023
    var_6 = 7
    var_7 = 20
    var_8 = True
    var_9 = module_0.Currency(var_1)
    var_10 = module_0.Currency(var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_query_returns_fxrate_for_valid_currency_pair_and_date. Retrieved 8/16 statements.
# Partially parsed test_query_returns_none_for_nonexistent_pair_when_strict_false. Retrieved 9/16 statements.
# Partially parsed test_query_raises_error_for_nonexistent_pair_when_strict_true. Retrieved 9/17 statements.
# Partially parsed test_query_returns_same_rate_for_inverse_currency_pair. Retrieved 9/19 statements.
# Partially parsed test_query_handles_same_currency_pair. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False
    var_8 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2023
    var_4 = 1
    var_5 = False
    var_6 = '1'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 12/25 statements.
# Partially parsed test_queries_strict_mode_raises_error. Retrieved 8/14 statements.
# Partially parsed test_queries_empty_input. Retrieved 2/6 statements.
# Partially parsed test_queries_single_query. Retrieved 9/18 statements.
# Partially parsed test_queries_handles_none_rates. Retrieved 7/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = 'GBP'
    var_7 = module_0.Currency(var_6)
    var_8 = 'JPY'
    var_9 = module_0.Currency(var_8)
    var_10 = 2
    var_11 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'XYZ'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = True
    var_7 = list(var_1)

def test_case_0():
    var_0 = []
    var_1 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = False
    var_7 = module_0.Currency(var_0)
    var_8 = module_0.Currency(var_2)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'AAA'
    var_1 = module_0.Currency(var_0)
    var_2 = 'BBB'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_query_returns_fxrate_for_valid_currency_pair_and_date. Retrieved 6/19 statements.
# Partially parsed test_query_returns_none_for_missing_rate_when_strict_false. Retrieved 6/18 statements.
# Partially parsed test_query_raises_error_for_missing_rate_when_strict_true. Retrieved 6/21 statements.
# Partially parsed test_query_handles_same_currency_pair. Retrieved 4/17 statements.
# Partially parsed test_query_uses_correct_date. Retrieved 7/19 statements.
# Partially parsed test_query_handles_different_currency_types. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '1.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'BTC'
    var_4 = 'Bitcoin'
    var_5 = 8



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 16/32 statements.
# Partially parsed test_queries_strict_mode_raises_error. Retrieved 16/31 statements.
# Partially parsed test_queries_empty_input. Retrieved 2/5 statements.
# Partially parsed test_queries_single_query. Retrieved 8/17 statements.
# Partially parsed test_queries_all_none_when_strict_false. Retrieved 10/19 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.2'
    var_1 = '0.85'
    var_2 = None
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = 2023
    var_8 = 1
    var_9 = 'GBP'
    var_10 = module_0.Currency(var_9)
    var_11 = module_0.Currency(var_3)
    var_12 = 'JPY'
    var_13 = module_0.Currency(var_12)
    var_14 = module_0.Currency(var_5)
    var_15 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.2'
    var_1 = 'Rate not found'
    var_2 = '0.85'
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = 2023
    var_8 = 1
    var_9 = 'GBP'
    var_10 = module_0.Currency(var_9)
    var_11 = module_0.Currency(var_3)
    var_12 = 'JPY'
    var_13 = module_0.Currency(var_12)
    var_14 = module_0.Currency(var_5)
    var_15 = True

def test_case_0():
    var_0 = []
    var_1 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.1'
    var_1 = 'USD'
    var_2 = module_0.Currency(var_1)
    var_3 = 'CAD'
    var_4 = module_0.Currency(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = 'GBP'
    var_7 = module_0.Currency(var_6)
    var_8 = module_0.Currency(var_0)
    var_9 = False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 14/30 statements.
# Partially parsed test_queries_with_strict_raises_error. Retrieved 9/16 statements.
# Partially parsed test_queries_with_strict_false_returns_none. Retrieved 7/14 statements.
# Partially parsed test_queries_empty_iterable. Retrieved 2/5 statements.
# Partially parsed test_queries_calls_query_for_each_input. Retrieved 15/28 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '1.5'
    var_7 = '0.8'
    var_8 = module_0.Currency(var_0)
    var_9 = module_0.Currency(var_2)
    var_10 = 'GBP'
    var_11 = module_0.Currency(var_10)
    var_12 = module_0.Currency(var_0)
    var_13 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Rate not found'
    var_1 = 'USD'
    var_2 = module_0.Currency(var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = True
    var_8 = list(var_1)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = False

def test_case_0():
    var_0 = []
    var_1 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.0'
    var_1 = 'USD'
    var_2 = module_0.Currency(var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 'GBP'
    var_8 = module_0.Currency(var_7)
    var_9 = module_0.Currency(var_1)
    var_10 = False
    var_11 = module_0.Currency(var_1)
    var_12 = module_0.Currency(var_3)
    var_13 = module_0.Currency(var_7)
    var_14 = module_0.Currency(var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 3/11 statements.
# Partially parsed test_constructor_allows_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_constructor_creates_fxrate_with_same_currency_and_value_one. Retrieved 2/9 statements.
# Partially parsed test_constructor_creates_fxrate_with_positive_decimal_value. Retrieved 3/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_large_decimal_value. Retrieved 3/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_future_date. Retrieved 6/14 statements.
# Partially parsed test_constructor_creates_fxrate_with_past_date. Retrieved 5/13 statements.
# Partially parsed test_constructor_creates_fxrate_with_different_currency_objects. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '12345.6789'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2100
    var_3 = 12
    var_4 = 31
    var_5 = '1.2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2000
    var_3 = 1
    var_4 = '0.9'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'GBP'
    var_2 = '0.007'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 3/11 statements.
# Partially parsed test_constructor_supports_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_constructor_allows_same_currency_with_value_one. Retrieved 2/9 statements.
# Partially parsed test_constructor_allows_same_currency_with_value_not_one. Retrieved 2/9 statements.
# Partially parsed test_constructor_allows_negative_value. Retrieved 3/11 statements.
# Partially parsed test_constructor_allows_zero_value. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 7/10 statements.
# Partially parsed test_constructor_allows_tuple_unpacking. Retrieved 7/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_same_currency_and_one_value. Retrieved 5/8 statements.
# Partially parsed test_constructor_creates_fxrate_with_positive_decimal_value. Retrieved 7/10 statements.
# Partially parsed test_constructor_creates_fxrate_with_large_decimal_value. Retrieved 7/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '1.2'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '1.2'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = '1'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '0.5'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '1000.123456'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_fxrate_constructor_creates_instance_with_correct_attributes. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_allows_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_accepts_same_currency_with_value_one. Retrieved 2/9 statements.
# Partially parsed test_fxrate_constructor_accepts_same_currency_with_value_not_one. Retrieved 2/9 statements.
# Partially parsed test_fxrate_constructor_accepts_value_less_than_zero. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_accepts_value_equal_to_zero. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 3/11 statements.
# Partially parsed test_constructor_supports_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_constructor_allows_same_currency_with_value_one. Retrieved 2/9 statements.
# Partially parsed test_constructor_allows_same_currency_with_value_not_one. Retrieved 2/9 statements.
# Partially parsed test_constructor_allows_zero_value. Retrieved 3/11 statements.
# Partially parsed test_constructor_allows_negative_value. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-1'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 7/10 statements.
# Partially parsed test_constructor_allows_tuple_unpacking. Retrieved 7/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_same_currency_and_value_one. Retrieved 5/8 statements.
# Partially parsed test_constructor_creates_fxrate_with_positive_decimal_value. Retrieved 7/10 statements.
# Partially parsed test_constructor_creates_fxrate_with_large_decimal_value. Retrieved 7/10 statements.
# Partially parsed test_constructor_creates_fxrate_with_future_date. Retrieved 8/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_past_date. Retrieved 7/10 statements.
# Partially parsed test_constructor_creates_fxrate_with_different_currency_objects. Retrieved 9/12 statements.
# Partially parsed test_constructor_creates_fxrate_where_indexed_access_matches_properties. Retrieved 7/10 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '1.2'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '1.2'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 2023
    var_3 = 1
    var_4 = '1'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '0.5'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '1000000.123456'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 2050
    var_5 = 12
    var_6 = 31
    var_7 = '1.2'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 2000
    var_5 = 1
    var_6 = '1.2'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 978
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 'USD'
    var_4 = 840
    var_5 = module_0.Currency(var_3, var_4)
    var_6 = 2023
    var_7 = 1
    var_8 = '1.2'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '1.2'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 5/15 statements.
# Partially parsed test_constructor_allows_indexed_access. Retrieved 5/15 statements.
# Partially parsed test_constructor_allows_unpacking. Retrieved 5/15 statements.
# Partially parsed test_constructor_creates_fxrate_with_same_currency_and_one_value. Retrieved 4/13 statements.
# Partially parsed test_constructor_creates_fxrate_with_positive_value. Retrieved 6/15 statements.
# Partially parsed test_constructor_creates_fxrate_with_currency_objects. Retrieved 5/13 statements.
# Partially parsed test_constructor_creates_fxrate_with_date_object. Retrieved 5/13 statements.
# Partially parsed test_constructor_creates_fxrate_with_decimal_value. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 2023
    var_2 = 1
    var_3 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '0.0001'
    var_5 = '0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.1'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 'EUR'
    var_3 = 'USD'
    var_4 = '1.1'

def test_case_0():
    var_0 = '1.1'
    var_1 = 'EUR'
    var_2 = 'USD'
    var_3 = 2023
    var_4 = 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_query_returns_fxrate_for_valid_currency_pair_and_date. Retrieved 8/22 statements.
# Partially parsed test_query_returns_none_for_missing_fxrate_when_strict_false. Retrieved 8/20 statements.
# Partially parsed test_query_raises_error_for_missing_fxrate_when_strict_true. Retrieved 8/23 statements.
# Partially parsed test_query_handles_same_currency_pair. Retrieved 6/20 statements.
# Partially parsed test_query_uses_correct_asof_date. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = '1.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 2023
    var_4 = 1
    var_5 = '1'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 12
    var_7 = 31



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_constructor_creates_valid_instance. Retrieved 3/11 statements.
# Partially parsed test_constructor_allows_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_constructor_creates_invertible_instance. Retrieved 4/14 statements.
# Partially parsed test_constructor_creates_instance_with_same_currency. Retrieved 2/10 statements.
# Partially parsed test_constructor_creates_instance_with_positive_decimal. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = '0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0.0001'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 3/8 statements.
# Partially parsed test_constructor_allows_unpacking. Retrieved 3/9 statements.
# Partially parsed test_constructor_accepts_negative_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_accepts_zero_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_accepts_same_currency_with_non_one_value. Retrieved 2/7 statements.
# Partially parsed test_constructor_accepts_non_currency_types_for_ccy1. Retrieved 3/7 statements.
# Partially parsed test_constructor_accepts_non_currency_types_for_ccy2. Retrieved 3/7 statements.
# Partially parsed test_constructor_accepts_non_date_types_for_date. Retrieved 4/8 statements.
# Partially parsed test_constructor_accepts_non_decimal_types_for_value. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2023-01-01'
    var_3 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2.5



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 14/30 statements.
# Partially parsed test_queries_with_strict_true_raises_error. Retrieved 9/16 statements.
# Partially parsed test_queries_with_strict_false_returns_none. Retrieved 7/14 statements.
# Partially parsed test_queries_empty_iterable. Retrieved 2/5 statements.
# Partially parsed test_queries_calls_query_for_each_input. Retrieved 11/20 statements.
# Partially parsed test_queries_passes_strict_parameter. Retrieved 11/20 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '1.5'
    var_7 = '0.8'
    var_8 = module_0.Currency(var_0)
    var_9 = module_0.Currency(var_2)
    var_10 = 'GBP'
    var_11 = module_0.Currency(var_10)
    var_12 = module_0.Currency(var_0)
    var_13 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Rate not found'
    var_1 = 'USD'
    var_2 = module_0.Currency(var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = True
    var_8 = list(var_1)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = False

def test_case_0():
    var_0 = []
    var_1 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.0'
    var_1 = 'USD'
    var_2 = module_0.Currency(var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 'GBP'
    var_8 = module_0.Currency(var_7)
    var_9 = module_0.Currency(var_1)
    var_10 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.0'
    var_1 = 'USD'
    var_2 = module_0.Currency(var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = True
    var_8 = module_0.Currency(var_1)
    var_9 = module_0.Currency(var_3)
    var_10 = True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_multiple_queries. Retrieved 18/37 statements.
# Partially parsed test_queries_raises_lookup_error_in_strict_mode. Retrieved 18/34 statements.
# Partially parsed test_queries_handles_empty_queries_list. Retrieved 2/6 statements.
# Partially parsed test_queries_returns_none_for_missing_rates_in_non_strict_mode. Retrieved 16/28 statements.
# Partially parsed test_queries_passes_strict_flag_to_query_method. Retrieved 9/20 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.1'
    var_1 = '1.2'
    var_2 = None
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = 2023
    var_8 = 1
    var_9 = 'GBP'
    var_10 = module_0.Currency(var_9)
    var_11 = module_0.Currency(var_3)
    var_12 = 2
    var_13 = 'JPY'
    var_14 = module_0.Currency(var_13)
    var_15 = module_0.Currency(var_5)
    var_16 = 3
    var_17 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = '1.1'
    var_1 = 'Rate not found'
    var_2 = '1.3'
    var_3 = 'USD'
    var_4 = module_0.Currency(var_3)
    var_5 = 'EUR'
    var_6 = module_0.Currency(var_5)
    var_7 = 2023
    var_8 = 1
    var_9 = 'GBP'
    var_10 = module_0.Currency(var_9)
    var_11 = module_0.Currency(var_3)
    var_12 = 2
    var_13 = 'JPY'
    var_14 = module_0.Currency(var_13)
    var_15 = module_0.Currency(var_5)
    var_16 = 3
    var_17 = True

def test_case_0():
    var_0 = []
    var_1 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = None
    var_1 = 'USD'
    var_2 = module_0.Currency(var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3)
    var_5 = 2023
    var_6 = 1
    var_7 = 'GBP'
    var_8 = module_0.Currency(var_7)
    var_9 = module_0.Currency(var_1)
    var_10 = 2
    var_11 = 'JPY'
    var_12 = module_0.Currency(var_11)
    var_13 = module_0.Currency(var_3)
    var_14 = 3
    var_15 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '1.1'
    var_7 = True
    var_8 = True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_query_returns_fxrate_for_valid_currency_pair_and_date. Retrieved 8/16 statements.
# Partially parsed test_query_returns_none_for_nonexistent_pair_when_strict_false. Retrieved 9/16 statements.
# Partially parsed test_query_raises_error_for_nonexistent_pair_when_strict_true. Retrieved 9/17 statements.
# Partially parsed test_query_returns_same_rate_for_inverse_pair. Retrieved 8/17 statements.
# Partially parsed test_query_returns_rate_for_same_currency. Retrieved 6/12 statements.
# Partially parsed test_query_handles_different_currency_types. Retrieved 9/17 statements.
# Partially parsed test_query_with_currency_having_negative_decimals. Retrieved 9/17 statements.
# Partially parsed test_query_with_currency_having_zero_decimals. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2023
    var_4 = 1
    var_5 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'BTC'
    var_4 = 'Bitcoin'
    var_5 = 8
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Crypto'
    var_2 = -1
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = 2023
    var_7 = 1
    var_8 = False

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = 2023
    var_7 = 1
    var_8 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 3/11 statements.
# Partially parsed test_constructor_allows_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_constructor_creates_fxrate_with_same_currency_and_value_one. Retrieved 2/9 statements.
# Partially parsed test_constructor_creates_fxrate_with_positive_decimal_value. Retrieved 3/11 statements.
# Partially parsed test_constructor_creates_fxrate_with_fractional_value. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '0.75'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_constructor_creates_fxrate_with_correct_attributes. Retrieved 3/11 statements.
# Partially parsed test_constructor_allows_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_constructor_allows_unpacking. Retrieved 3/12 statements.
# Partially parsed test_constructor_does_not_validate_input. Retrieved 3/11 statements.
# Partially parsed test_constructor_accepts_same_currency_with_non_one_value. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '5'



