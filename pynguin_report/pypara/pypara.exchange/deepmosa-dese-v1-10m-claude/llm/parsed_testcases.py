####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_fxrate_invert. Retrieved 4/16 statements.
# Partially parsed test_fxrate_invert_with_decimal_one. Retrieved 2/12 statements.
# Partially parsed test_fxrate_invert_with_small_decimal. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = '0.5'

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = '0.01'
    var_3 = '100'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_queries_with_empty_iterable. Retrieved 1/9 statements.
# Partially parsed test_queries_with_single_query. Retrieved 5/28 statements.
# Partially parsed test_queries_with_multiple_queries. Retrieved 6/34 statements.
# Partially parsed test_queries_with_none_results. Retrieved 4/22 statements.
# Partially parsed test_queries_strict_mode. Retrieved 7/25 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 'GBP'
    var_3 = 2023
    var_4 = 1
    var_5 = '1.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_0]
    var_5 = True
    var_6 = list(var_3)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_queries_with_single_query. Retrieved 5/27 statements.
# Partially parsed test_queries_with_multiple_queries. Retrieved 8/34 statements.
# Partially parsed test_queries_with_missing_rates. Retrieved 6/30 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 6/28 statements.
# Partially parsed test_queries_empty_list. Retrieved 1/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '0.85'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 'GBP'
    var_3 = 2023
    var_4 = 1
    var_5 = '0.85'
    var_6 = '0.73'
    var_7 = '0.86'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 'JPY'
    var_3 = 2023
    var_4 = 1
    var_5 = '0.85'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = True
    var_5 = list(var_1)

def test_case_0():
    var_0 = []



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_queries_returns_iterable_of_rates. Retrieved 11/43 statements.
# Partially parsed test_queries_with_strict_mode_raises_error. Retrieved 8/35 statements.
# Partially parsed test_queries_without_strict_mode_returns_none. Retrieved 7/35 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 'GBP'
    var_5 = module_0.Currency(var_4)
    var_6 = 2023
    var_7 = 1
    var_8 = False
    var_9 = '0.92'
    var_10 = '0.79'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'JPY'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = True
    var_7 = list(var_2)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_queries_with_empty_iterable. Retrieved 1/18 statements.
# Partially parsed test_queries_with_single_query. Retrieved 6/27 statements.
# Partially parsed test_queries_with_multiple_queries. Retrieved 6/29 statements.
# Partially parsed test_queries_with_strict_false. Retrieved 7/28 statements.


def test_case_0():
    var_0 = []

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 7/15 statements.
# Partially parsed test_fxrateservice_query_signature. Retrieved 7/20 statements.


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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_different_values. Retrieved 6/14 statements.


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
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = '150.5'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_different_values. Retrieved 5/13 statements.


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
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = '150.5'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_different_values. Retrieved 6/14 statements.


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
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = '150.75'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_fxrate_constructor_with_valid_arguments. Retrieved 3/11 statements.
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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_different_values. Retrieved 6/14 statements.


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
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = '150.75'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_different_currencies. Retrieved 6/19 statements.
# Partially parsed test_fxrate_constructor_same_values. Retrieved 3/12 statements.


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
    var_2 = '1.5'
    var_3 = 'GBP'
    var_4 = 'JPY'
    var_5 = '100'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.5'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_fxrate_constructor_basic. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_different_currencies. Retrieved 6/14 statements.
# Partially parsed test_fxrate_constructor_small_decimal_value. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_large_decimal_value. Retrieved 3/12 statements.


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
    var_2 = '3.25'

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = '150.75'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '0.0001'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '999999.999999'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_queries_returns_iterable_of_fx_rates. Retrieved 5/32 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 5/31 statements.
# Partially parsed test_queries_returns_none_for_missing_rates. Retrieved 6/27 statements.
# Partially parsed test_queries_accepts_empty_iterable. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = False
    var_5 = None

def test_case_0():
    var_0 = []
    var_1 = False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 8/27 statements.
# Partially parsed test_fxrateservice_queries_is_abstract. Retrieved 7/25 statements.
# Partially parsed test_fxrateservice_tquery_type. Retrieved 7/19 statements.


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

def test_case_0():
    pass

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_queries_with_empty_iterable. Retrieved 1/15 statements.
# Partially parsed test_queries_with_single_query. Retrieved 5/29 statements.
# Partially parsed test_queries_with_multiple_queries. Retrieved 6/32 statements.
# Partially parsed test_queries_with_strict_mode_false. Retrieved 5/25 statements.
# Partially parsed test_queries_returns_iterable. Retrieved 2/15 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 'GBP'
    var_3 = 2023
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = False

def test_case_0():
    var_0 = []
    var_1 = '__iter__'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_different_currencies. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_small_decimal_value. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_large_decimal_value. Retrieved 3/12 statements.


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
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = '150.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '0.85'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'BRL'
    var_2 = '5.25'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_queries_returns_iterable_of_rates. Retrieved 16/30 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 6/22 statements.
# Partially parsed test_queries_with_empty_iterable. Retrieved 1/13 statements.
# Partially parsed test_queries_accepts_iterable_parameter. Retrieved 11/26 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '2023-01-01'
    var_3 = (var_0, var_1, var_2)
    var_4 = 'GBP'
    var_5 = 'JPY'
    var_6 = '2023-01-02'
    var_7 = (var_4, var_5, var_6)
    var_8 = 'CAD'
    var_9 = 'AUD'
    var_10 = '2023-01-03'
    var_11 = (var_8, var_9, var_10)
    var_12 = [var_3, var_7, var_11]
    var_13 = False
    var_14 = '1.5'
    var_15 = '2.0'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '2023-01-01'
    var_3 = (var_0, var_1, var_2)
    var_4 = [var_3]
    var_5 = False

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '2023-01-01'
    var_3 = (var_0, var_1, var_2)
    var_4 = 'GBP'
    var_5 = 'JPY'
    var_6 = '2023-01-02'
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]
    var_9 = '1.1'
    var_10 = '1.2'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_different_currencies. Retrieved 5/13 statements.
# Partially parsed test_fxrate_constructor_with_small_decimal_value. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_large_decimal_value. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = '150.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '0.001'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '9999.9999'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 7/18 statements.
# Failed to parse test_fxrateservice_query_signature.
# Failed to parse test_fxrateservice_query_return_type_annotation.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/12 statements.


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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 8/27 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_queries_returns_iterable_of_rates. Retrieved 5/36 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 6/34 statements.
# Partially parsed test_queries_with_empty_iterable. Retrieved 1/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.25'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'GBP'
    var_2 = 2023
    var_3 = 1
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = []



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_different_currencies. Retrieved 5/13 statements.


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
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = '150.5'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_fxrate_constructor_valid. Retrieved 3/11 statements.
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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_fxrateservice_query_is_abstract.
# Partially parsed test_fxrateservice_query_with_valid_parameters. Retrieved 7/23 statements.
# Partially parsed test_fxrateservice_query_returns_none_when_not_found. Retrieved 9/24 statements.
# Partially parsed test_fxrateservice_query_with_strict_mode. Retrieved 9/27 statements.
# Partially parsed test_fxrateservice_query_with_same_currency. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
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
    var_7 = 6
    var_8 = 15

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'British Pound'
    var_2 = 2
    var_3 = 'CHF'
    var_4 = 'Swiss Franc'
    var_5 = 2023
    var_6 = 12
    var_7 = 31
    var_8 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2023
    var_4 = 3
    var_5 = 20
    var_6 = '1'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 9/28 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = False
    var_8 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_queries_with_single_query. Retrieved 5/30 statements.
# Partially parsed test_queries_with_multiple_queries. Retrieved 7/35 statements.
# Partially parsed test_queries_with_none_results. Retrieved 4/22 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 9/26 statements.
# Partially parsed test_queries_returns_iterable. Retrieved 5/22 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '0.85'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 'GBP'
    var_3 = 2023
    var_4 = 1
    var_5 = '0.85'
    var_6 = '0.73'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XXX'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'INVALID'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = False
    var_5 = [var_0]
    var_6 = True
    var_7 = list(var_3)
    var_8 = True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '__iter__'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 7/22 statements.
# Partially parsed test_fxrateservice_query_with_strict_false. Retrieved 9/24 statements.
# Partially parsed test_fxrateservice_query_accepts_different_currencies. Retrieved 9/26 statements.
# Partially parsed test_fxrateservice_query_with_different_dates. Retrieved 9/30 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pounds'
    var_5 = 2023
    var_6 = 6
    var_7 = 15
    var_8 = False

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = 2023
    var_7 = 3
    var_8 = 20

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euros'
    var_2 = 2
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2023
    var_6 = 1
    var_7 = 12
    var_8 = 31



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_queries_with_empty_iterable. Retrieved 1/15 statements.
# Partially parsed test_queries_with_single_query. Retrieved 5/26 statements.
# Partially parsed test_queries_with_multiple_queries. Retrieved 8/34 statements.
# Partially parsed test_queries_with_none_values. Retrieved 5/28 statements.
# Partially parsed test_queries_strict_mode_false. Retrieved 6/26 statements.
# Partially parsed test_queries_strict_mode_true. Retrieved 5/24 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 'GBP'
    var_3 = 2023
    var_4 = 1
    var_5 = '1.5'
    var_6 = '1.2'
    var_7 = '0.9'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = False
    var_5 = '1.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_queries_with_single_query. Retrieved 5/30 statements.
# Partially parsed test_queries_with_multiple_queries. Retrieved 8/41 statements.
# Partially parsed test_queries_with_missing_rate. Retrieved 4/22 statements.
# Partially parsed test_queries_with_empty_iterable. Retrieved 1/14 statements.
# Partially parsed test_queries_returns_iterable. Retrieved 5/22 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '0.85'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 'USD'
    var_3 = 'EUR'
    var_4 = 'GBP'
    var_5 = '0.85'
    var_6 = '0.73'
    var_7 = '0.86'

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 'USD'
    var_3 = 'XXX'

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 'USD'
    var_3 = 'EUR'
    var_4 = '__iter__'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_queries_with_empty_iterable. Retrieved 1/9 statements.
# Partially parsed test_queries_with_single_query. Retrieved 7/28 statements.
# Partially parsed test_queries_with_multiple_queries. Retrieved 9/31 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 7/24 statements.
# Partially parsed test_queries_returns_iterable. Retrieved 7/22 statements.


def test_case_0():
    var_0 = []

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '1.5'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 'GBP'
    var_5 = module_0.Currency(var_4)
    var_6 = 2023
    var_7 = 1
    var_8 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = True

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = '__iter__'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_different_values. Retrieved 6/14 statements.


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
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 6
    var_4 = 15
    var_5 = '150.75'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_immutability. Retrieved 4/14 statements.


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
    var_2 = '2'
    var_3 = '3'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
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

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_with_different_currencies. Retrieved 5/13 statements.


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
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = '150.5'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 7/18 statements.
# Failed to parse test_fxrateservice_query_signature.
# Failed to parse test_fxrateservice_query_return_type.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_queries_with_empty_iterable. Retrieved 2/25 statements.
# Partially parsed test_queries_with_single_query. Retrieved 7/34 statements.
# Partially parsed test_queries_with_multiple_queries. Retrieved 9/40 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 7/34 statements.
# Failed to parse test_queries_with_none_results.


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

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = False

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = module_0.Currency(var_0)
    var_2 = 'EUR'
    var_3 = module_0.Currency(var_2)
    var_4 = 2023
    var_5 = 1
    var_6 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_different_currencies. Retrieved 6/14 statements.


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
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = '150.5'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 1/6 statements.
# Failed to parse test_fxrateservice_query_signature.
# Failed to parse test_fxrateservice_cannot_instantiate.
# Failed to parse test_fxrateservice_query_requires_implementation.
# Partially parsed test_fxrateservice_query_with_concrete_implementation. Retrieved 8/25 statements.
# Partially parsed test_fxrateservice_query_with_strict_parameter. Retrieved 9/28 statements.


def test_case_0():
    var_0 = '__isabstractmethod__'

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
    var_8 = True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_queries_with_empty_iterable. Retrieved 1/15 statements.
# Partially parsed test_queries_with_single_query. Retrieved 5/26 statements.
# Partially parsed test_queries_with_multiple_queries. Retrieved 7/35 statements.
# Partially parsed test_queries_with_strict_mode_false. Retrieved 5/24 statements.
# Partially parsed test_queries_returns_iterable. Retrieved 5/24 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '1.5'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 'GBP'
    var_3 = 2023
    var_4 = 1
    var_5 = '0.92'
    var_6 = '1.27'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XXX'
    var_2 = 2023
    var_3 = 1
    var_4 = False

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = '__iter__'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.


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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = 'query'
    var_8 = '__isabstractmethod__'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_queries_returns_iterable_of_fx_rates. Retrieved 5/23 statements.
# Partially parsed test_queries_with_strict_mode_true. Retrieved 5/20 statements.
# Partially parsed test_queries_returns_none_for_missing_rates. Retrieved 5/19 statements.
# Partially parsed test_queries_with_empty_queries_list. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = '1.5'
    var_3 = '0.67'
    var_4 = False

def test_case_0():
    var_0 = 2023
    var_1 = 6
    var_2 = 15
    var_3 = '2.3'
    var_4 = True

def test_case_0():
    var_0 = 2023
    var_1 = 12
    var_2 = 31
    var_3 = None
    var_4 = False

def test_case_0():
    var_0 = []
    var_1 = False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_different_currencies. Retrieved 6/14 statements.
# Partially parsed test_fxrate_constructor_small_value. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_large_value. Retrieved 3/12 statements.


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
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = '150.75'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '0.0001'

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '9999.9999'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 3/11 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 3/12 statements.
# Partially parsed test_fxrate_constructor_with_different_values. Retrieved 6/14 statements.


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
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = '150.75'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_fxrateservice_query_is_abstract. Retrieved 7/18 statements.
# Partially parsed test_fxrateservice_query_signature. Retrieved 7/20 statements.
# Partially parsed test_fxrateservice_query_is_abstract_method. Retrieved 1/5 statements.


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

def test_case_0():
    var_0 = '__isabstractmethod__'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_queries_returns_iterable_of_fx_rates. Retrieved 4/31 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 5/27 statements.
# Partially parsed test_queries_with_empty_iterable. Retrieved 1/17 statements.
# Partially parsed test_queries_returns_optional_fx_rates. Retrieved 4/28 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = True

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_fxrate_constructor. Retrieved 3/11 statements.
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



