####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_fxrate_invert_success. Retrieved 4/7 statements.
# Partially parsed test_fxrate_invert_identity. Retrieved 3/6 statements.


import decimal as module_0

def test_case_0():
    var_0 = '2'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = '0.5'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)

import decimal as module_0

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_valid_inputs. Retrieved 17/22 statements.
# Partially parsed test_queries_returns_none_when_rate_not_found. Retrieved 10/15 statements.
# Partially parsed test_queries_with_strict_mode_true. Retrieved 12/17 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = 'GBP'
    var_9 = 2
    var_10 = [var_2, var_3, var_9]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = (var_8, var_0, var_12)
    var_14 = [var_7, var_13]
    var_15 = '0.92'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = '1.25'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_1.Decimal(*var_20, **var_21)
    var_23 = [var_18, var_22]
    var_24 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = None
    var_10 = [var_9]
    var_11 = False

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = '0.92'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.Decimal(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = True
    var_15 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_fx_rate_service_query_returns_rate_when_exists. Retrieved 11/21 statements.
# Partially parsed test_fx_rate_service_query_returns_none_when_not_found. Retrieved 9/17 statements.
# Partially parsed test_fx_rate_service_query_respects_strict_parameter. Retrieved 10/17 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = '0.92'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Decimal(*var_11, **var_12)
    var_14 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = True
    var_11 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_valid_inputs. Retrieved 10/15 statements.
# Partially parsed test_queries_returns_none_when_rate_not_found. Retrieved 9/14 statements.
# Partially parsed test_queries_handles_multiple_inputs_and_strict_mode. Retrieved 16/21 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '0.92'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = (var_0, var_1, var_6)
    var_12 = [var_11]
    var_13 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = None
    var_10 = False

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 'GBP'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = (var_0, var_1, var_7)
    var_9 = (var_0, var_2, var_7)
    var_10 = [var_8, var_9]
    var_11 = '0.92'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)
    var_15 = '0.82'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = [var_14, var_18]
    var_20 = True
    var_21 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 5/13 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 5/13 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_queries_returns_correct_values_from_iterable. Retrieved 9/18 statements.
# Partially parsed test_queries_with_strict_true_raises_error. Retrieved 8/19 statements.
# Partially parsed test_queries_handles_empty_iterable. Retrieved 1/9 statements.


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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '2023-01-01'
    var_3 = (var_0, var_1, var_2)
    var_4 = [var_3]
    var_5 = True
    var_6 = 'Should have raised ValueError'
    var_7 = AssertionError(var_6)

def test_case_0():
    var_0 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_valid_input. Retrieved 10/16 statements.
# Partially parsed test_queries_returns_none_when_rate_not_found. Retrieved 9/15 statements.
# Partially parsed test_queries_handles_multiple_inputs_and_strict_mode. Retrieved 18/24 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '0.92'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = (var_0, var_1, var_6)
    var_12 = [var_11]
    var_13 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'GBP'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = None
    var_10 = False

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 'JPY'
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = (var_0, var_1, var_7)
    var_9 = (var_0, var_2, var_7)
    var_10 = [var_8, var_9]
    var_11 = '0.92'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)
    var_15 = '130.0'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = [var_14, var_18]
    var_20 = True
    var_21 = [var_11]
    var_22 = {}
    var_23 = module_1.Decimal(*var_21, **var_22)
    var_24 = [var_15]
    var_25 = {}
    var_26 = module_1.Decimal(*var_24, **var_25)
    var_27 = True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_fx_rate_service_query_returns_rate_when_exists. Retrieved 9/18 statements.
# Partially parsed test_fx_rate_service_query_returns_none_when_not_found. Retrieved 9/17 statements.
# Partially parsed test_fx_rate_service_query_with_strict_mode_true. Retrieved 10/17 statements.


import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 5
    var_8 = [var_6, var_7, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_fxrate_service_query_returns_rate_when_found. Retrieved 11/19 statements.
# Partially parsed test_fxrate_service_query_returns_none_when_not_found. Retrieved 9/21 statements.
# Partially parsed test_fxrate_service_query_with_strict_mode_true. Retrieved 12/20 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = '0.95'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Decimal(*var_11, **var_12)
    var_14 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = False

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = '0.95'
    var_11 = [var_10]
    var_12 = {}
    var_13 = True
    var_14 = [var_10]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)
    var_17 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_fxrate_constructor_valid_inputs. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_inversion. Retrieved 6/14 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_query_returns_rate_when_found. Retrieved 9/18 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 9/17 statements.
# Partially parsed test_query_with_strict_parameter. Retrieved 10/17 statements.


import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = True
    var_11 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_untyped_assignment. Retrieved 4/9 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_fxrate_constructor_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/11 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.5'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_fxrate_constructor_valid_data. Retrieved 7/13 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 7/13 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '1.1'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '1.1'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/11 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/13 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 4/13 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 4/12 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 9/15 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'GBP'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '0.85'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = [var_2, var_3, var_3]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = [var_7]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_fx_rate_service_query_returns_rate_when_exists. Retrieved 9/18 statements.
# Partially parsed test_fx_rate_service_query_returns_none_when_not_found. Retrieved 8/15 statements.
# Partially parsed test_fx_rate_service_query_with_strict_true. Retrieved 10/17 statements.


import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = True
    var_11 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_valid_inputs. Retrieved 17/22 statements.
# Partially parsed test_queries_returns_none_when_rate_not_found. Retrieved 10/15 statements.
# Partially parsed test_queries_with_strict_mode_enabled. Retrieved 14/19 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = 'GBP'
    var_9 = 2
    var_10 = [var_2, var_3, var_9]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = (var_8, var_0, var_12)
    var_14 = [var_7, var_13]
    var_15 = '0.92'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = '1.22'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_1.Decimal(*var_20, **var_21)
    var_23 = [var_18, var_22]
    var_24 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XYZ'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = None
    var_10 = [var_9]
    var_11 = False

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = '0.92'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.Decimal(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = True
    var_15 = [var_9]
    var_16 = {}
    var_17 = module_1.Decimal(*var_15, **var_16)
    var_18 = [var_17]
    var_19 = True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 7/14 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 7/13 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.25'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '150.0'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'CHF'
    var_1 = 'CAD'
    var_2 = 2023
    var_3 = 5
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_fxrate_constructor_valid_args. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 4/12 statements.


import decimal as module_0

def test_case_0():
    var_0 = '1.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'EUR'
    var_5 = 'USD'

import decimal as module_0

def test_case_0():
    var_0 = '2.0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'GBP'
    var_5 = 'JPY'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 4/12 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 4/11 statements.


import decimal as module_0

def test_case_0():
    var_0 = '1.25'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'EUR'
    var_5 = 'USD'

import decimal as module_0

def test_case_0():
    var_0 = '0.85'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'USD'
    var_5 = 'GBP'

import decimal as module_0

def test_case_0():
    var_0 = '1.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'JPY'
    var_5 = 'USD'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_queries_returns_iterable_of_rates. Retrieved 9/18 statements.
# Partially parsed test_queries_with_strict_mode_raises_error. Retrieved 7/18 statements.
# Partially parsed test_queries_with_mock_values. Retrieved 9/17 statements.


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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '2023-01-01'
    var_3 = (var_0, var_1, var_2)
    var_4 = [var_3]
    var_5 = True
    var_6 = list(var_1)

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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_fxrate_constructor_assignment. Retrieved 4/18 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 5/22 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_fxrateservice_query_signature_and_return_type_logic. Retrieved 10/34 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = '0.85'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Decimal(*var_9, **var_10)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_fxrate_service_query_returns_rate_when_exists. Retrieved 8/17 statements.
# Partially parsed test_fxrate_service_query_returns_none_when_not_found. Retrieved 8/16 statements.
# Partially parsed test_fxrate_service_query_respects_strict_flag. Retrieved 11/20 statements.


import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = True
    var_11 = True
    var_12 = False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_fxrate_service_query_returns_rate_when_found. Retrieved 9/18 statements.
# Partially parsed test_fxrate_service_query_returns_none_when_not_found. Retrieved 9/17 statements.
# Partially parsed test_fxrate_service_query_with_strict_mode_raises_error. Retrieved 11/20 statements.


import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = 'Rate not found'
    var_11 = [var_10]
    var_12 = {}
    var_13 = True
    var_14 = True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_queries_returns_correct_rates_from_mock_implementation. Retrieved 15/25 statements.
# Failed to parse test_queries_with_strict_true_raises_error_in_mock_implementation.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = 'GBP'
    var_9 = 'JPY'
    var_10 = [var_2, var_3, var_3]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = (var_8, var_9, var_12)
    var_14 = [var_7, var_13]
    var_15 = '1.2'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = '1.0'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_1.Decimal(*var_20, **var_21)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_query_returns_rate_when_found. Retrieved 9/18 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 11/19 statements.
# Partially parsed test_query_called_with_strict_true. Retrieved 10/17 statements.


import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = False
    var_12 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pound'
    var_5 = 2023
    var_6 = 5
    var_7 = 20
    var_8 = [var_5, var_6, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_fxrate_service_query_returns_rate_when_exists. Retrieved 8/17 statements.
# Partially parsed test_fxrate_service_query_returns_none_when_not_found. Retrieved 9/17 statements.
# Partially parsed test_fxrate_service_query_respects_strict_parameter. Retrieved 11/20 statements.


import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pound'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = True
    var_11 = True
    var_12 = False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_fxrate_service_query_interface_definition. Retrieved 10/26 statements.
# Partially parsed test_fxrate_service_query_parameters_usage. Retrieved 9/26 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = '1.10'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Decimal(*var_11, **var_12)

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_queries_returns_correct_values. Retrieved 18/23 statements.
# Failed to parse test_queries_with_strict_mode.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = 'GBP'
    var_9 = 'JPY'
    var_10 = 2
    var_11 = [var_2, var_3, var_10]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = (var_8, var_9, var_13)
    var_15 = [var_7, var_14]
    var_16 = '0.92'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.Decimal(*var_17, **var_18)
    var_20 = '160.50'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_1.Decimal(*var_21, **var_22)
    var_24 = [var_19, var_23]
    var_25 = False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_queries_returns_iterable_of_rates. Retrieved 20/25 statements.
# Partially parsed test_queries_with_strict_mode_raises_error. Retrieved 1/3 statements.


import decimal as module_0
import datetime as module_1

def test_case_0():
    var_0 = '1.2'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = '0.8'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = None
    var_9 = [var_3, var_7, var_8]
    var_10 = 'USD'
    var_11 = 'EUR'
    var_12 = 2023
    var_13 = 1
    var_14 = [var_12, var_13, var_13]
    var_15 = {}
    var_16 = module_1.date(*var_14, **var_15)
    var_17 = (var_10, var_11, var_16)
    var_18 = 'GBP'
    var_19 = [var_12, var_13, var_13]
    var_20 = {}
    var_21 = module_1.date(*var_19, **var_20)
    var_22 = (var_18, var_10, var_21)
    var_23 = 'JPY'
    var_24 = [var_12, var_13, var_13]
    var_25 = {}
    var_26 = module_1.date(*var_24, **var_25)
    var_27 = (var_23, var_10, var_26)
    var_28 = [var_17, var_22, var_27]
    var_29 = False

def test_case_0():
    var_0 = 'Rate not found'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_FXRate_constructor_valid_assignment. Retrieved 5/13 statements.
# Partially parsed test_FXRate_constructor_tuple_unpacking. Retrieved 5/14 statements.
# Partially parsed test_FXRate_constructor_indexed_access. Retrieved 5/13 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)

import decimal as module_0

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = '150'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_queries_returns_expected_rates. Retrieved 17/22 statements.
# Partially parsed test_queries_with_strict_mode_raises_error. Retrieved 11/16 statements.
# Partially parsed test_queries_returns_none_for_missing_rate. Retrieved 10/15 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = 'GBP'
    var_9 = 2
    var_10 = [var_2, var_3, var_9]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = (var_8, var_0, var_12)
    var_14 = [var_7, var_13]
    var_15 = '0.92'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = '1.22'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_1.Decimal(*var_20, **var_21)
    var_23 = [var_18, var_22]
    var_24 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = 'Rate not found'
    var_10 = True
    var_11 = list(var_1)
    var_12 = True

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XYZ'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = None
    var_10 = [var_9]
    var_11 = False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/11 statements.


import decimal as module_0

def test_case_0():
    var_0 = '1.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'EUR'
    var_5 = 'USD'

import decimal as module_0

def test_case_0():
    var_0 = '2.0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'EUR'
    var_5 = 'USD'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 4/12 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 7/13 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '150.0'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_fxrate_constructor_valid_data. Retrieved 9/15 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 9/15 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '1.2'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = [var_2, var_3, var_3]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = [var_7]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '1.2'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = [var_2, var_3, var_3]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = [var_7]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_fxrate_constructor_assignment_and_indexing. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 4/12 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 4/15 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 4/17 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/15 statements.
# Partially parsed test_fxrate_constructor_untyped_access. Retrieved 5/15 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_fxrate_constructor_valid_data. Retrieved 5/13 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 5/13 statements.
# Partially parsed test_fxrate_constructor_equality. Retrieved 5/16 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_valid_input. Retrieved 18/23 statements.
# Partially parsed test_queries_returns_none_for_missing_rates. Retrieved 10/15 statements.
# Partially parsed test_queries_raises_error_when_strict_is_true_and_rate_missing. Retrieved 11/16 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = 'GBP'
    var_9 = 'JPY'
    var_10 = 2
    var_11 = [var_2, var_3, var_10]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = (var_8, var_9, var_13)
    var_15 = [var_7, var_14]
    var_16 = '0.92'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.Decimal(*var_17, **var_18)
    var_20 = '160.50'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_1.Decimal(*var_21, **var_22)
    var_24 = [var_19, var_23]
    var_25 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XYZ'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = None
    var_10 = [var_9]
    var_11 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XYZ'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = 'Rate not found'
    var_10 = True
    var_11 = list(var_1)
    var_12 = True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_fxrate_service_query_returns_expected_rate. Retrieved 10/18 statements.
# Partially parsed test_fxrate_service_query_returns_none_when_not_found. Retrieved 8/16 statements.
# Partially parsed test_fxrate_service_query_with_strict_flag. Retrieved 12/20 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = '0.92'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Decimal(*var_11, **var_12)

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2023
    var_6 = 1
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = '0.92'
    var_11 = [var_10]
    var_12 = {}
    var_13 = True
    var_14 = [var_10]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)
    var_17 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_indexing. Retrieved 4/11 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.5'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_fxrate_constructor_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/11 statements.


import decimal as module_0

def test_case_0():
    var_0 = '1.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'EUR'
    var_5 = 'USD'

import decimal as module_0

def test_case_0():
    var_0 = '2.0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'EUR'
    var_5 = 'USD'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_query_interface_definition. Retrieved 7/25 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 5/20 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = '1.23'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_queries_returns_iterable_of_rates_from_concrete_implementation. Retrieved 15/35 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = 'GBP'
    var_9 = 'JPY'
    var_10 = [var_2, var_3, var_3]
    var_11 = {}
    var_12 = module_0.date(*var_10, **var_11)
    var_13 = (var_8, var_9, var_12)
    var_14 = [var_7, var_13]
    var_15 = '1.2'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = None
    var_20 = [var_15]
    var_21 = {}
    var_22 = module_1.Decimal(*var_20, **var_21)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_fxrate_constructor_valid_data. Retrieved 5/13 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 5/13 statements.
# Partially parsed test_fxrate_constructor_identity_property. Retrieved 4/11 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1.0'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)
    var_5 = [var_1]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_FXRate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_FXRate_constructor_tuple_unpacking. Retrieved 7/14 statements.
# Partially parsed test_FXRate_constructor_indexed_access. Retrieved 5/13 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.25'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '150.5'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '0.9'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_valid_input. Retrieved 18/23 statements.
# Partially parsed test_queries_returns_none_when_rate_not_found. Retrieved 10/15 statements.
# Partially parsed test_queries_with_strict_mode_enabled. Retrieved 12/17 statements.


import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = 'GBP'
    var_9 = 'JPY'
    var_10 = 2
    var_11 = [var_2, var_3, var_10]
    var_12 = {}
    var_13 = module_0.date(*var_11, **var_12)
    var_14 = (var_8, var_9, var_13)
    var_15 = [var_7, var_14]
    var_16 = '0.92'
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.Decimal(*var_17, **var_18)
    var_20 = '160.50'
    var_21 = [var_20]
    var_22 = {}
    var_23 = module_1.Decimal(*var_21, **var_22)
    var_24 = [var_19, var_23]
    var_25 = False

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'XYZ'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = None
    var_10 = [var_9]
    var_11 = False

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = '0.92'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.Decimal(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = True
    var_15 = True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_fxrateservice_query_returns_none_when_no_rate_found. Retrieved 9/17 statements.
# Partially parsed test_fxrateservice_query_returns_rate_when_rate_exists. Retrieved 9/18 statements.
# Partially parsed test_fxrateservice_query_with_strict_parameter_true. Retrieved 11/18 statements.
# Partially parsed test_fxrateservice_query_with_strict_parameter_false. Retrieved 11/18 statements.


import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = True
    var_12 = True

import datetime as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 2023
    var_7 = 1
    var_8 = [var_6, var_7, var_7]
    var_9 = {}
    var_10 = module_0.date(*var_8, **var_9)
    var_11 = False
    var_12 = False



