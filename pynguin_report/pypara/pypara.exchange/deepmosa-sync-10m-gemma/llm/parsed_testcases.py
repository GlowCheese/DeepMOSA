####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_fxrate_invert_logic. Retrieved 4/7 statements.
# Partially parsed test_fxrate_invert_double_inversion. Retrieved 3/7 statements.
# Partially parsed test_fxrate_invert_identity_with_one. Retrieved 3/6 statements.


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
    var_0 = '4'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)

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

# Partially parsed test_queries_returns_correct_rates. Retrieved 18/23 statements.
# Failed to parse test_queries_strict_mode_raises_error.


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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_fxrate_invert_logic. Retrieved 6/11 statements.
# Partially parsed test_fxrate_invert_identity. Retrieved 4/10 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '4'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_query_returns_rate_when_found. Retrieved 10/18 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 8/16 statements.
# Partially parsed test_query_with_strict_true. Retrieved 8/15 statements.


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
    var_10 = '0.95'
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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = True
    var_9 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_invert_fxrate_calculation. Retrieved 6/11 statements.
# Partially parsed test_invert_fxrate_identity. Retrieved 3/7 statements.
# Partially parsed test_invert_fxrate_inverse_property. Retrieved 4/11 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '4'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_queries_returns_expected_rates. Retrieved 18/23 statements.
# Partially parsed test_queries_with_strict_true. Retrieved 14/19 statements.
# Partially parsed test_queries_returns_none_for_missing_rates. Retrieved 9/14 statements.


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
import decimal as module_1

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
    var_9 = '0.78'
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_invert_returns_correct_inverted_rate. Retrieved 4/7 statements.
# Partially parsed test_invert_is_symmetric. Retrieved 2/6 statements.
# Partially parsed test_invert_with_one. Retrieved 3/6 statements.


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
    var_0 = '4'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)

import decimal as module_0

def test_case_0():
    var_0 = '1'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 18/23 statements.
# Partially parsed test_queries_with_strict_true. Retrieved 15/20 statements.
# Partially parsed test_queries_returns_none_for_missing_rate. Retrieved 9/14 statements.


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
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'CAD'
    var_2 = 2023
    var_3 = 5
    var_4 = 1
    var_5 = [var_2, var_3, var_4]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = (var_0, var_1, var_7)
    var_9 = [var_8]
    var_10 = '1.35'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Decimal(*var_11, **var_12)
    var_14 = [var_13]
    var_15 = True
    var_16 = [var_10]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = [var_18]
    var_20 = True

import datetime as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = 'ABC'
    var_2 = 2000
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = None
    var_10 = [var_9]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/15 statements.
# Partially parsed test_fxrate_constructor_tuple_indexing. Retrieved 4/13 statements.
# Partially parsed test_fxrate_constructor_unpacks_correctly. Retrieved 4/14 statements.


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

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_query_returns_rate_when_found. Retrieved 9/18 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 9/17 statements.
# Partially parsed test_query_with_strict_mode_true. Retrieved 10/17 statements.


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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_queries_returns_iterable_of_rates. Retrieved 15/20 statements.
# Partially parsed test_queries_with_strict_mode_raises_error. Retrieved 11/16 statements.
# Partially parsed test_queries_handles_none_values. Retrieved 11/16 statements.


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
    var_8 = (var_0, var_1, var_6)
    var_9 = [var_7, var_8]
    var_10 = '0.92'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Decimal(*var_11, **var_12)
    var_14 = '0.93'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_1.Decimal(*var_15, **var_16)
    var_18 = [var_13, var_17]
    var_19 = False

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
    var_11 = True

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
    var_10 = [var_9]
    var_11 = False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_fxrate_service_query_returns_rate_when_exists. Retrieved 8/17 statements.
# Partially parsed test_fxrate_service_query_returns_none_when_not_found. Retrieved 8/16 statements.
# Partially parsed test_fxrate_service_query_respects_strict_parameter. Retrieved 10/17 statements.
# Partially parsed test_fxrate_service_query_raises_error_in_strict_mode. Retrieved 11/20 statements.


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
    var_10 = 'Rate not found'
    var_11 = [var_10]
    var_12 = {}
    var_13 = True
    var_14 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_query_returns_expected_fxrate. Retrieved 4/11 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 3/8 statements.
# Partially parsed test_query_with_strict_param. Retrieved 5/10 statements.


import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = False

import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)

import datetime as module_0

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = True
    var_6 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/11 statements.


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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_fxrate_constructor_valid_values. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_unvalidated_same_currency. Retrieved 4/11 statements.


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

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = '5.0'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)
    var_5 = [var_1]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 2/10 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 2/10 statements.
# Partially parsed test_fxrate_constructor_inversion. Retrieved 4/13 statements.
# Partially parsed test_fxrate_of_factory_valid. Retrieved 2/10 statements.
# Partially parsed test_fxrate_of_factory_invalid_same_currency_not_one. Retrieved 2/10 statements.
# Partially parsed test_fxrate_constructor_properties. Retrieved 2/10 statements.
# Partially parsed test_fxrate_constructor_indexability. Retrieved 2/10 statements.


import decimal as module_0

def test_case_0():
    var_0 = '2.0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)

import decimal as module_0

def test_case_0():
    var_0 = '1.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)

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
    var_0 = '2'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)

import decimal as module_0

def test_case_0():
    var_0 = '1.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)

import decimal as module_0

def test_case_0():
    var_0 = '10'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)

import decimal as module_0

def test_case_0():
    var_0 = '10'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/15 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 4/14 statements.


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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_fxrate_constructor_assignment. Retrieved 4/14 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/13 statements.
# Partially parsed test_fxrate_constructor_unpacking. Retrieved 7/16 statements.


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

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'JPY'
    var_2 = 2020
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '150.0'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_indexing. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_inversion. Retrieved 6/14 statements.
# Partially parsed test_fxrate_of_valid_creation. Retrieved 4/11 statements.
# Partially parsed test_fxrate_of_invalid_value_raises_error. Retrieved 4/12 statements.
# Partially parsed test_fxrate_of_same_currency_invalid_value_raises_error. Retrieved 3/11 statements.
# Partially parsed test_fxrate_of_same_currency_valid_value_passes. Retrieved 4/11 statements.


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

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)

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
    var_2 = '0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1.5'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)
    var_5 = [var_1]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_FXRate_constructor_valid_inputs. Retrieved 4/11 statements.
# Partially parsed test_FXRate_constructor_tuple_access. Retrieved 5/13 statements.
# Partially parsed test_FXRate_constructor_identity_case. Retrieved 4/11 statements.


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
    var_6 = [var_2]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)
    var_5 = [var_1]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_indexing. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_untyped_assignment. Retrieved 8/12 statements.


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

import datetime as module_0
import decimal as module_1
import pypara.exchange as module_2

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = '2'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = [var_0, var_1, var_6, var_10]
    var_12 = {}
    var_13 = module_2.FXRate(*var_11, **var_12)
    var_14 = var_13.ccy1
    assert var_14 == 'EUR'
    var_15 = var_13.ccy2
    assert var_15 == 'USD'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_query_returns_rate_when_exists. Retrieved 12/21 statements.
# Partially parsed test_query_returns_none_when_rate_not_found. Retrieved 10/17 statements.
# Partially parsed test_query_with_strict_mode_logic_placeholder. Retrieved 11/18 statements.


import decimal as module_0
import pypara.exchange as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 1
    var_8 = 'EUR'
    var_9 = 'Euro'
    var_10 = [var_3]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = '1.10'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_0.Decimal(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = {}
    var_19 = module_1.FXRate(*var_17, **var_18)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 1
    var_8 = 'EUR'
    var_9 = 'Euro'
    var_10 = [var_3]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = {}

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 1
    var_8 = 'EUR'
    var_9 = 'Euro'
    var_10 = [var_3]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = {}
    var_14 = True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_queries_returns_expected_rates. Retrieved 18/23 statements.
# Partially parsed test_queries_with_strict_true. Retrieved 13/18 statements.
# Partially parsed test_queries_returns_none_on_missing_rate. Retrieved 10/15 statements.


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
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'CAD'
    var_2 = 2023
    var_3 = 5
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = '1.35'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.Decimal(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = True
    var_15 = [var_9]
    var_16 = {}
    var_17 = module_1.Decimal(*var_15, **var_16)
    var_18 = [var_17]

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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_fxrate_constructor_valid_assignment. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 4/12 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 4/11 statements.


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

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_query_returns_rate_when_exists. Retrieved 12/23 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 8/15 statements.
# Partially parsed test_query_respects_strict_parameter. Retrieved 10/17 statements.


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
    var_10 = 0
    var_11 = FXRateService.__subclasses__()[var_10]
    var_12 = var_11()
    var_13 = False

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



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_fxrate_invert_success. Retrieved 6/11 statements.
# Partially parsed test_fxrate_invert_equality. Retrieved 9/14 statements.
# Partially parsed test_fxrate_invert_identity. Retrieved 3/6 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)
    var_10 = 'rframe_rrate'
    var_11 = locals()
    var_12 = var_10 in var_11

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = '1'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_fxrate_invert_success. Retrieved 4/7 statements.
# Partially parsed test_fxrate_invert_identity. Retrieved 3/6 statements.
# Partially parsed test_fxrate_invert_double_inversion. Retrieved 2/7 statements.


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

import decimal as module_0

def test_case_0():
    var_0 = '3.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_query_returns_rate_when_found. Retrieved 10/19 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 8/16 statements.
# Partially parsed test_query_respects_strict_parameter. Retrieved 11/20 statements.


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
    var_10 = '0.95'
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
    var_12 = False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_queries_returns_correct_rates_for_valid_inputs. Retrieved 18/23 statements.
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
    var_20 = '150.50'
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_invert_fxrate_logic. Retrieved 6/13 statements.
# Partially parsed test_invert_fxrate_identity. Retrieved 4/10 statements.
# Partially parsed test_invert_fxrate_value_calculation. Retrieved 8/13 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'GBP'
    var_2 = '1'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'USD'
    var_2 = '150'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '1'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)
    var_10 = [var_2]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)
    var_13 = var_9 / var_12



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_query_returns_rate_when_found. Retrieved 9/18 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 9/17 statements.
# Partially parsed test_query_respects_strict_parameter. Retrieved 11/20 statements.


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
    var_12 = False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 18/23 statements.
# Partially parsed test_queries_with_strict_true. Retrieved 14/19 statements.
# Partially parsed test_queries_returns_none_for_missing_rate. Retrieved 9/14 statements.


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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 18/23 statements.
# Partially parsed test_queries_with_strict_true. Retrieved 13/18 statements.
# Partially parsed test_queries_returns_none_when_rate_missing. Retrieved 9/14 statements.


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
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'CAD'
    var_2 = 2023
    var_3 = 5
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = '1.35'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.Decimal(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = True
    var_15 = [var_9]
    var_16 = {}
    var_17 = module_1.Decimal(*var_15, **var_16)
    var_18 = [var_17]

import datetime as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = 'ABC'
    var_2 = 2023
    var_3 = 1
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = None
    var_10 = [var_9]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 18/23 statements.
# Partially parsed test_queries_with_strict_true. Retrieved 13/18 statements.
# Partially parsed test_queries_returns_none_for_missing_rates. Retrieved 9/13 statements.


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
import decimal as module_1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'CAD'
    var_2 = 2023
    var_3 = 5
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = '1.35'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.Decimal(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = True
    var_15 = [var_9]
    var_16 = {}
    var_17 = module_1.Decimal(*var_15, **var_16)
    var_18 = [var_17]

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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_inversion. Retrieved 6/14 statements.


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

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_fxrate_constructor_valid_inputs. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_inversion. Retrieved 6/14 statements.


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

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_fxrate_constructor_valid_data. Retrieved 4/14 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 4/14 statements.
# Partially parsed test_fxrate_invert_operation. Retrieved 6/16 statements.
# Partially parsed test_fxrate_of_valid_creation. Retrieved 4/13 statements.
# Partially parsed test_fxrate_of_same_currency_with_one_fails_if_not_one. Retrieved 3/15 statements.
# Partially parsed test_fxrate_of_invalid_value_fails. Retrieved 4/15 statements.


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

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '1.25'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1.5'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '-1.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_query_interface_definition. Retrieved 8/22 statements.
# Partially parsed test_query_return_type_none. Retrieved 8/19 statements.


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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_query_returns_rate_when_found. Retrieved 9/18 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 9/16 statements.
# Partially parsed test_query_with_strict_true. Retrieved 8/14 statements.


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
    var_3 = 2023
    var_4 = 1
    var_5 = [var_3, var_4, var_4]
    var_6 = {}
    var_7 = module_0.date(*var_5, **var_6)
    var_8 = True
    var_9 = True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_identity_check. Retrieved 5/12 statements.


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

import decimal as module_0

def test_case_0():
    var_0 = '1.0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'USD'
    var_5 = '1'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_fxrate_constructor_assignment. Retrieved 4/8 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/8 statements.
# Partially parsed test_fxrate_invert_operation. Retrieved 6/11 statements.


import decimal as module_0

def test_case_0():
    var_0 = '2.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'EUR'
    var_5 = 'USD'

import decimal as module_0

def test_case_0():
    var_0 = '1.2'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'GBP'
    var_5 = 'JPY'

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_unvalidated_input_possible. Retrieved 4/8 statements.


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
    var_0 = '2'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'NOT_A_CURRENCY'
    var_5 = 'ALSO_NOT_A_CURRENCY'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_fxrate_constructor_valid_data. Retrieved 4/15 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/13 statements.


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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_tuple_access. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_with_inverted_logic. Retrieved 6/14 statements.


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
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 18/23 statements.
# Partially parsed test_queries_with_strict_true. Retrieved 14/19 statements.
# Partially parsed test_queries_returns_none_on_missing_rate. Retrieved 9/14 statements.


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
import decimal as module_1

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
    var_9 = '0.78'
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_query_returns_rate_when_exists. Retrieved 9/18 statements.
# Partially parsed test_query_returns_none_when_not_found. Retrieved 10/18 statements.
# Partially parsed test_query_respects_strict_parameter. Retrieved 10/17 statements.


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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_fxrate_constructor_valid_data. Retrieved 4/15 statements.
# Partially parsed test_fxrate_constructor_tuple_unpacking. Retrieved 7/16 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2.0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

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
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_fxrate_constructor_valid_values. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_indexed_access. Retrieved 4/11 statements.
# Partially parsed test_fxrate_constructor_unpacks_correctly. Retrieved 4/12 statements.


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

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'USD'
    var_2 = '2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_fxrateservice_query_interface_signature. Retrieved 11/23 statements.
# Partially parsed test_fxrateservice_query_return_none. Retrieved 8/19 statements.


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
    var_10 = True
    var_11 = '1.0'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_1.Decimal(*var_12, **var_13)

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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_queries_returns_iterable_of_rates. Retrieved 9/18 statements.
# Partially parsed test_queries_with_strict_mode_logic. Retrieved 7/19 statements.
# Partially parsed test_queries_handles_empty_input. Retrieved 1/9 statements.


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
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = []



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_fxrate_constructor_valid. Retrieved 4/8 statements.
# Partially parsed test_fxrate_constructor_inversion. Retrieved 6/11 statements.
# Partially parsed test_fxrate_of_valid. Retrieved 4/10 statements.
# Partially parsed test_fxrate_of_invalid_value_raises_error. Retrieved 4/12 statements.
# Partially parsed test_fxrate_of_same_currency_invalid_value_raises_error. Retrieved 3/10 statements.


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
    var_6 = '0.5'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Decimal(*var_7, **var_8)

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
    var_2 = '0'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = '1.5'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_fxrate_constructor_valid_input. Retrieved 4/11 statements.
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_queries_returns_correct_rates. Retrieved 11/16 statements.
# Partially parsed test_queries_handles_none_rates. Retrieved 10/15 statements.
# Partially parsed test_queries_with_strict_mode. Retrieved 12/17 statements.
# Partially parsed test_queries_multiple_inputs. Retrieved 18/23 statements.


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
    var_7 = '0.91'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Decimal(*var_8, **var_9)
    var_11 = (var_0, var_1, var_6)
    var_12 = [var_11]
    var_13 = [var_10]
    var_14 = False

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
    var_10 = [var_9]
    var_11 = False

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'AUD'
    var_2 = 2023
    var_3 = 5
    var_4 = [var_2, var_3, var_3]
    var_5 = {}
    var_6 = module_0.date(*var_4, **var_5)
    var_7 = (var_0, var_1, var_6)
    var_8 = [var_7]
    var_9 = '0.01'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.Decimal(*var_10, **var_11)
    var_13 = True
    var_14 = [var_9]
    var_15 = {}
    var_16 = module_1.Decimal(*var_14, **var_15)
    var_17 = [var_16]

import datetime as module_0
import decimal as module_1

def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = [var_0, var_1, var_1]
    var_3 = {}
    var_4 = module_0.date(*var_2, **var_3)
    var_5 = 2
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.date(*var_6, **var_7)
    var_9 = 'USD'
    var_10 = 'EUR'
    var_11 = (var_9, var_10, var_4)
    var_12 = 'GBP'
    var_13 = (var_9, var_12, var_8)
    var_14 = [var_11, var_13]
    var_15 = '0.91'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_1.Decimal(*var_16, **var_17)
    var_19 = '0.82'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_1.Decimal(*var_20, **var_21)
    var_23 = [var_18, var_22]
    var_24 = [var_15]
    var_25 = {}
    var_26 = module_1.Decimal(*var_24, **var_25)
    var_27 = [var_19]
    var_28 = {}
    var_29 = module_1.Decimal(*var_27, **var_28)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_fxrate_constructor_assigns_values_correctly. Retrieved 2/4 statements.
# Partially parsed test_fxrate_constructor_allows_indexed_access. Retrieved 2/4 statements.
# Failed to parse test_fxrate_constructor_with_identical_currencies_and_one_value.
# Partially parsed test_fxrate_constructor_with_identical_currencies_and_non_one_value. Retrieved 3/5 statements.


import decimal as module_0

def test_case_0():
    var_0 = '2.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)

import decimal as module_0

def test_case_0():
    var_0 = '1.5'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)

import decimal as module_0

def test_case_0():
    var_0 = '2.0'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_query_returns_expected_rate. Retrieved 10/19 statements.
# Partially parsed test_query_returns_none_when_no_rate_found. Retrieved 9/17 statements.
# Partially parsed test_query_with_strict_parameter. Retrieved 9/16 statements.


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
    var_10 = '0.95'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Decimal(*var_11, **var_12)

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
    var_6 = 5
    var_7 = [var_5, var_6, var_6]
    var_8 = {}
    var_9 = module_0.date(*var_7, **var_8)
    var_10 = True



