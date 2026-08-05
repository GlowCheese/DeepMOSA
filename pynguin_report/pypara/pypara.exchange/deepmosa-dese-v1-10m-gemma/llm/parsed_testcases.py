####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_fxrate_invert_basic. Retrieved 4/11 statements.
# Partially parsed test_fxrate_invert_identity. Retrieved 2/7 statements.
# Partially parsed test_fxrate_invert_double_inversion. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = '0.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = '150.5'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_invert_returns_swapped_currencies_and_reciprocal_value. Retrieved 4/10 statements.
# Partially parsed test_invert_is_idempotent_double_inversion. Retrieved 3/8 statements.
# Partially parsed test_invert_with_one_remains_same_structure. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = '2'
    var_3 = '0.5'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = '4'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = '1'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 9/22 statements.
# Partially parsed test_queries_with_strict_true. Retrieved 7/19 statements.
# Partially parsed test_queries_returns_none_for_missing_rates. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 2
    var_6 = '0.95'
    var_7 = '1.25'
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = '140.0'
    var_5 = True
    var_6 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XYZ'
    var_2 = 2023
    var_3 = 1
    var_4 = None
    var_5 = [var_4]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 9/22 statements.
# Partially parsed test_queries_with_strict_mode_true. Retrieved 6/18 statements.
# Partially parsed test_queries_returns_none_for_missing_rate. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 'JPY'
    var_6 = '0.92'
    var_7 = '160.50'
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'CAD'
    var_2 = 2023
    var_3 = 5
    var_4 = '1.35'
    var_5 = True

def test_case_0():
    var_0 = 'XYZ'
    var_1 = 'ABC'
    var_2 = 2000
    var_3 = 1
    var_4 = None
    var_5 = [var_4]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_query_returns_rate_when_found. Retrieved 7/17 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 10/23 statements.
# Partially parsed test_queries_handles_none_values. Retrieved 7/15 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 'JPY'
    var_6 = 2
    var_7 = '0.92'
    var_8 = '160.50'
    var_9 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'CAD'
    var_2 = 2023
    var_3 = 1
    var_4 = None
    var_5 = [var_4]
    var_6 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'AUD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.50'
    var_5 = True
    var_6 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_query_returns_expected_fxrate_when_found. Retrieved 8/18 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 8/17 statements.
# Partially parsed test_query_respects_strict_parameter. Retrieved 9/17 statements.


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
    var_8 = True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_query_returns_rate_when_found. Retrieved 8/19 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 8/17 statements.
# Partially parsed test_query_respects_strict_parameter. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = '0.95'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pound'
    var_5 = 2023
    var_6 = 5
    var_7 = 20
    var_8 = True
    var_9 = False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 10/23 statements.
# Failed to parse test_queries_with_strict_mode_raises_error.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 'JPY'
    var_6 = 2
    var_7 = '0.92'
    var_8 = '165.50'
    var_9 = False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_fxrate_constructor_valid_data. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unvalidated_same_currency. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1.5'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_fxrateservice_query_returns_none_when_no_rate_found. Retrieved 8/17 statements.
# Partially parsed test_fxrateservice_query_returns_rate_when_exists. Retrieved 8/19 statements.
# Partially parsed test_fxrateservice_query_respects_strict_parameter. Retrieved 9/17 statements.


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
    var_8 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 5/12 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 5/12 statements.
# Partially parsed test_fxrate_constructor_untyped_assignment. Retrieved 5/8 statements.


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
    var_4 = '1.5'

def test_case_0():
    var_0 = None
    var_1 = 123
    var_2 = 'not-a-date'
    var_3 = 0
    var_4 = [var_3]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/13 statements.


def test_case_0():
    var_0 = '1.25'
    var_1 = 'USD'
    var_2 = 'EUR'

def test_case_0():
    var_0 = '0.8'
    var_1 = 'EUR'
    var_2 = 'GBP'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_FXRate_constructor_valid. Retrieved 3/11 statements.
# Partially parsed test_FXRate_constructor_tuple_access. Retrieved 3/11 statements.
# Partially parsed test_FXRate_invert_logic. Retrieved 4/14 statements.
# Partially parsed test_FXRate_of_valid. Retrieved 3/11 statements.
# Partially parsed test_FXRate_of_invalid_value_zero. Retrieved 3/12 statements.
# Partially parsed test_FXRate_of_invalid_same_currency. Retrieved 3/12 statements.
# Partially parsed test_FXRate_of_valid_identity. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '1.5'
    var_1 = 'EUR'
    var_2 = 'USD'

def test_case_0():
    var_0 = '2.0'
    var_1 = 'EUR'
    var_2 = 'USD'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = '0.5'

def test_case_0():
    var_0 = '1.25'
    var_1 = 'GBP'
    var_2 = 'JPY'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 3/15 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_fxrate_constructor_valid_data. Retrieved 5/14 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 5/14 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '1.25'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'GBP'
    var_1 = module_0.Currency(var_0)
    var_2 = 'JPY'
    var_3 = module_0.Currency(var_2)
    var_4 = '150.0'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 5/14 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 5/12 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 5/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '2.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '1.5'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'GBP'
    var_1 = module_0.Currency(var_0)
    var_2 = 'JPY'
    var_3 = module_0.Currency(var_2)
    var_4 = '150.0'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_fxrate_constructor_assignment. Retrieved 5/14 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 5/12 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 5/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '2.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '2.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '2.0'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_fxrate_service_query_returns_value_when_exists. Retrieved 4/12 statements.
# Partially parsed test_fxrate_service_query_returns_none_when_not_found. Retrieved 3/10 statements.
# Partially parsed test_fxrate_service_query_parameters_mapping. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = True
    var_3 = True

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = False

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_queries_returns_expected_rates. Retrieved 9/25 statements.
# Partially parsed test_queries_with_strict_mode_true. Retrieved 7/19 statements.
# Partially parsed test_queries_returns_none_for_missing_rate. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 2
    var_6 = '0.92'
    var_7 = '1.25'
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = '130.0'
    var_5 = True
    var_6 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XYZ'
    var_2 = 2023
    var_3 = 1
    var_4 = None
    var_5 = [var_4]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/12 statements.


def test_case_0():
    var_0 = '1.5'
    var_1 = 'EUR'
    var_2 = 'USD'

def test_case_0():
    var_0 = '2.0'
    var_1 = 'USD'
    var_2 = 'GBP'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 3/8 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/10 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = '150'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_queries_returns_correct_values. Retrieved 10/23 statements.
# Partially parsed test_queries_with_strict_true. Retrieved 7/19 statements.
# Partially parsed test_queries_returns_none_for_missing_rate. Retrieved 7/15 statements.
# Partially parsed test_queries_handles_empty_iterable. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 'JPY'
    var_6 = 2
    var_7 = '0.92'
    var_8 = '160.50'
    var_9 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'CAD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.35'
    var_5 = True
    var_6 = True

def test_case_0():
    var_0 = 'XYZ'
    var_1 = 'ABC'
    var_2 = 2023
    var_3 = 1
    var_4 = None
    var_5 = [var_4]
    var_6 = False

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_fxrate_service_query_interface. Retrieved 8/28 statements.
# Partially parsed test_fxrate_service_query_returns_none. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = '1.2'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 3/8 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 3/8 statements.
# Partially parsed test_fxrate_of_valid_creation. Retrieved 3/8 statements.
# Partially parsed test_fxrate_of_same_currency_with_one_is_valid. Retrieved 2/7 statements.
# Partially parsed test_fxrate_of_same_currency_not_one_raises_error. Retrieved 2/7 statements.
# Partially parsed test_fxrate_of_zero_value_raises_error. Retrieved 3/9 statements.
# Partially parsed test_fxrate_of_negative_value_raises_error. Retrieved 3/9 statements.
# Partially parsed test_fxrate_of_invalid_ccy1_type_raises_error. Retrieved 3/8 statements.
# Partially parsed test_fxrate_of_invalid_ccy2_type_raises_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '2.5'
    var_1 = 'EUR'
    var_2 = 'USD'

def test_case_0():
    var_0 = '1.5'
    var_1 = 'GBP'
    var_2 = 'JPY'

def test_case_0():
    var_0 = '1.2'
    var_1 = 'EUR'
    var_2 = 'USD'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-1.0'

def test_case_0():
    var_0 = 'NOT_A_CURRENCY'
    var_1 = 'USD'
    var_2 = '1'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'NOT_A_CURRENCY'
    var_2 = '1'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 3/8 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 3/8 statements.
# Partially parsed test_fxrate_of_valid_input. Retrieved 3/8 statements.
# Partially parsed test_fxrate_of_invalid_ccy1_type. Retrieved 3/9 statements.
# Partially parsed test_fxrate_of_invalid_ccy2_type. Retrieved 3/9 statements.
# Partially parsed test_fxrate_of_zero_value. Retrieved 3/10 statements.
# Partially parsed test_fxrate_of_negative_value. Retrieved 3/10 statements.
# Partially parsed test_fxrate_of_same_currency_invalid_value. Retrieved 2/9 statements.
# Partially parsed test_fxrate_of_same_currency_valid_value. Retrieved 2/8 statements.
# Partially parsed test_fxrate_inversion. Retrieved 4/11 statements.


def test_case_0():
    var_0 = '1.5'
    var_1 = 'EUR'
    var_2 = 'USD'

def test_case_0():
    var_0 = '2.0'
    var_1 = 'EUR'
    var_2 = 'GBP'

def test_case_0():
    var_0 = '1.2'
    var_1 = 'EUR'
    var_2 = 'USD'

def test_case_0():
    var_0 = 'NOT_A_CURRENCY'
    var_1 = 'USD'
    var_2 = '1.0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'NOT_A_CURRENCY'
    var_2 = '1.0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-1.0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1.0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = '0.5'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_fxrate_service_query_returns_rate_when_exists. Retrieved 8/18 statements.
# Partially parsed test_fxrate_service_query_returns_none_when_not_found. Retrieved 9/18 statements.
# Partially parsed test_fxrate_service_query_respects_strict_parameter. Retrieved 9/17 statements.


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
    var_3 = 'GBP'
    var_4 = 'British Pound'
    var_5 = 2023
    var_6 = 5
    var_7 = 20
    var_8 = True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_fxrateservice_query_abstract_method_raises_error. Retrieved 7/25 statements.
# Partially parsed test_fxrateservice_query_interface_definition. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = '1.10'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_fxrate_service_query_interface. Retrieved 5/26 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_query_returns_rate_when_found. Retrieved 2/10 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 2/8 statements.
# Partially parsed test_query_respects_strict_parameter. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = True
    var_3 = True
    var_4 = False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 9/22 statements.
# Failed to parse test_queries_with_strict_mode_raises_error.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 2
    var_6 = '0.94'
    var_7 = '1.22'
    var_8 = False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_valid_input. Retrieved 10/23 statements.
# Partially parsed test_queries_returns_none_when_rate_not_found. Retrieved 7/15 statements.
# Partially parsed test_queries_with_strict_mode_true. Retrieved 7/17 statements.
# Partially parsed test_queries_handles_empty_iterable. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 'JPY'
    var_6 = 2
    var_7 = '0.92'
    var_8 = '160.5'
    var_9 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XYZ'
    var_2 = 2023
    var_3 = 1
    var_4 = None
    var_5 = [var_4]
    var_6 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '0.92'
    var_5 = True
    var_6 = True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_queries_returns_correct_values. Retrieved 10/23 statements.
# Partially parsed test_queries_handles_none_values. Retrieved 7/15 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 'JPY'
    var_6 = 2
    var_7 = '0.92'
    var_8 = '160.50'
    var_9 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'CAD'
    var_2 = 2023
    var_3 = 1
    var_4 = None
    var_5 = [var_4]
    var_6 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'AUD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.50'
    var_5 = True
    var_6 = True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_queries_returns_correct_values. Retrieved 8/21 statements.
# Partially parsed test_queries_with_strict_true. Retrieved 7/19 statements.
# Partially parsed test_queries_returns_none_for_missing_rates. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = '0.92'
    var_6 = '1.22'
    var_7 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = '130.0'
    var_5 = True
    var_6 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XYZ'
    var_2 = 2023
    var_3 = 1
    var_4 = None
    var_5 = [var_4]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_fxrate_constructor_assignment. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_fxrate_constructor_valid_data. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.25'

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = '150.0'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 5/14 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 5/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '2.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '1.5'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 3/15 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 3/15 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.5'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_fxrate_constructor_valid. Retrieved 5/12 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 5/12 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '2.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '1.5'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.


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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_indexing. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_untyped_assignment. Retrieved 3/9 statements.


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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_queries_returns_iterable_of_rates. Retrieved 9/22 statements.
# Partially parsed test_queries_with_strict_true. Retrieved 7/19 statements.
# Partially parsed test_queries_returns_none_for_missing_rates. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 2
    var_6 = '0.95'
    var_7 = '1.25'
    var_8 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = '130.0'
    var_5 = True
    var_6 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XYZ'
    var_2 = 2023
    var_3 = 1
    var_4 = None
    var_5 = [var_4]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_fxrateservice_query_interface_signature. Retrieved 8/24 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = '1.23'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 5/13 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 5/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '2.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '1.5'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/12 statements.


def test_case_0():
    var_0 = '1.5'
    var_1 = 'EUR'
    var_2 = 'USD'

def test_case_0():
    var_0 = '2.0'
    var_1 = 'GBP'
    var_2 = 'JPY'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_fxrate_service_query_returns_rate_when_found. Retrieved 8/18 statements.
# Partially parsed test_fxrate_service_query_returns_none_when_not_found. Retrieved 9/18 statements.
# Partially parsed test_fxrate_service_query_respects_strict_parameter. Retrieved 10/20 statements.


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
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = True
    var_8 = True
    var_9 = False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 9/22 statements.
# Failed to parse test_queries_with_strict_mode_raises_error.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 2
    var_6 = '0.92'
    var_7 = '1.25'
    var_8 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_fxrate_constructor_assignment. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 3/11 statements.


def test_case_0():
    var_0 = '1.5'
    var_1 = 'EUR'
    var_2 = 'USD'

def test_case_0():
    var_0 = '2.0'
    var_1 = 'EUR'
    var_2 = 'USD'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 3/8 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 5/8 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 7/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 5/8 statements.
# Partially parsed test_fxrate_constructor_inversion. Retrieved 6/11 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '2.0'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '1.5'
    var_5 = 'GBP'
    var_6 = module_0.Currency(var_5)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'JPY'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '0.007'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = module_0.Currency(var_0)
    var_2 = 'USD'
    var_3 = module_0.Currency(var_2)
    var_4 = '2'
    var_5 = '0.5'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_fxrate_service_query_interface_definition. Retrieved 7/23 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 10/23 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 7/19 statements.
# Partially parsed test_queries_handles_none_values. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 'GBP'
    var_5 = 'JPY'
    var_6 = 2
    var_7 = '0.94'
    var_8 = '160.50'
    var_9 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'CAD'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.35'
    var_5 = True
    var_6 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XYZ'
    var_2 = 2023
    var_3 = 1
    var_4 = None
    var_5 = [var_4]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.


def test_case_0():
    var_0 = '1.5'
    var_1 = 'EUR'
    var_2 = 'USD'

def test_case_0():
    var_0 = '2.0'
    var_1 = 'GBP'
    var_2 = 'JPY'

def test_case_0():
    var_0 = '0.85'
    var_1 = 'USD'
    var_2 = 'CAD'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_query_returns_rate_when_found. Retrieved 7/17 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 8/17 statements.
# Partially parsed test_query_with_strict_parameter. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pound'
    var_5 = 2023
    var_6 = 5
    var_7 = 20
    var_8 = True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_fxrate_constructor_assignment. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_inversion. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = '0.5'



