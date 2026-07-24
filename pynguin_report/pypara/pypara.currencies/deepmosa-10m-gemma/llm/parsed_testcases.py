####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_currency_constructor_valid_assignment. Retrieved 7/9 statements.
# Partially parsed test_currency_equality_same_hash. Retrieved 3/9 statements.
# Partially parsed test_currency_equality_different_name. Retrieved 4/10 statements.
# Partially parsed test_currency_quantize_usd. Retrieved 12/19 statements.
# Partially parsed test_currency_quantize_jpy. Retrieved 11/15 statements.
# Partially parsed test_currency_quantize_crypto. Retrieved 11/15 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 12345
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = '1.00'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = '1.015'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_0.Decimal(*var_12, **var_13)
    var_15 = '1.02'
    var_16 = [var_11]
    var_17 = {}
    var_18 = module_0.Decimal(*var_16, **var_17)
    var_19 = [var_15]
    var_20 = {}
    var_21 = module_0.Decimal(*var_19, **var_20)

import decimal as module_0

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = '0'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = '1.5'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_0.Decimal(*var_12, **var_13)
    var_15 = '2'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_0.Decimal(*var_16, **var_17)

import decimal as module_0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = '1.000000000000'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = '1.0000000000015'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_0.Decimal(*var_12, **var_13)
    var_15 = '1.000000000002'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_0.Decimal(*var_16, **var_17)



# Parsed testcases at query #2
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #4
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #5
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_currency_constructor_valid_assignment. Retrieved 7/9 statements.
# Partially parsed test_currency_equality_same_hashcache. Retrieved 6/9 statements.
# Partially parsed test_currency_equality_different_hashcache. Retrieved 7/10 statements.
# Partially parsed test_currency_hash_returns_hashcache. Retrieved 6/9 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 12345
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 999

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 111
    var_8 = 222

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 777



# Parsed testcases at query #8
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #9
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_currency_constructor_valid_initialization. Retrieved 6/8 statements.
# Partially parsed test_currency_equality_same_hashcache. Retrieved 7/11 statements.
# Partially parsed test_currency_inequality_different_hashcache. Retrieved 8/12 statements.
# Partially parsed test_currency_hash_returns_hashcache. Retrieved 6/9 statements.


import decimal as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 12345
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 2

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 100
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 100
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = 200

import decimal as module_0

def test_case_0():
    var_0 = 999
    var_1 = 'EUR'
    var_2 = 'Euro'
    var_3 = 2
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_codes_buffer_is_not_empty_after_registration. Retrieved 8/15 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'AED'
    var_2 = 'UAE Dirham'
    var_3 = 2
    var_4 = 'USD'
    var_5 = 'US Dollar'
    var_6 = var_0.codes
    var_7 = len(var_6)
    var_8 = bool(var_7 != 0)
    assert var_8 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_currency_constructor_valid_assignment. Retrieved 7/9 statements.
# Partially parsed test_currency_constructor_equality_and_hash. Retrieved 6/11 statements.
# Partially parsed test_currency_constructor_inequality. Retrieved 11/17 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 12345
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

import decimal as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 999

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 111
    var_8 = 'USD'
    var_9 = 'UX Dollars'
    var_10 = 2
    var_11 = [var_3]
    var_12 = {}
    var_13 = module_0.Decimal(*var_11, **var_12)
    var_14 = 222



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_currency_constructor_valid_assignment. Retrieved 7/9 statements.
# Partially parsed test_currency_equality_same_hashcache. Retrieved 6/9 statements.
# Partially parsed test_currency_inequality_different_hashcache. Retrieved 7/10 statements.
# Partially parsed test_currency_hash_return_value. Retrieved 6/9 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 12345
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 999

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 111
    var_8 = 222

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 888



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #15
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_currency_constructor_valid_parameters. Retrieved 7/9 statements.
# Partially parsed test_currency_equality_same_hashcache. Retrieved 6/9 statements.
# Partially parsed test_currency_inequality_different_hashcache. Retrieved 7/10 statements.
# Partially parsed test_currency_hash_return_value. Retrieved 6/9 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 12345
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 999

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 111
    var_8 = 222

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 777



# Parsed testcases at query #17
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_currency_constructor_valid_initialization. Retrieved 6/8 statements.
# Partially parsed test_currency_equality_same_hashcache. Retrieved 7/11 statements.
# Partially parsed test_currency_equality_different_hashcache. Retrieved 8/12 statements.
# Partially parsed test_currency_hash_returns_hashcache. Retrieved 6/9 statements.
# Partially parsed test_currency_order_comparison. Retrieved 7/11 statements.


import decimal as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 12345
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 2

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 999
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 999
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = 888

import decimal as module_0

def test_case_0():
    var_0 = 12345
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)

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
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)



# Parsed testcases at query #20
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #21
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_currency_constructor_valid_assignment. Retrieved 6/8 statements.
# Partially parsed test_currency_equality_logic. Retrieved 4/8 statements.
# Partially parsed test_currency_hash_consistency. Retrieved 3/8 statements.
# Partially parsed test_currency_of_factory_creates_correct_quantizer_for_positive_decimals. Retrieved 11/15 statements.
# Partially parsed test_currency_of_factory_creates_correct_quantizer_for_zero_decimals. Retrieved 11/15 statements.
# Partially parsed test_currency_of_factory_creates_correct_quantizer_for_negative_decimals. Retrieved 11/15 statements.


import decimal as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 12345
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = '1.00'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = '1.015'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_0.Decimal(*var_12, **var_13)
    var_15 = '1.02'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_0.Decimal(*var_16, **var_17)

import decimal as module_0

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = '0'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = '1.5'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_0.Decimal(*var_12, **var_13)
    var_15 = '2'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_0.Decimal(*var_16, **var_17)

import decimal as module_0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = '1.000000000000'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = '1.0000000000015'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_0.Decimal(*var_12, **var_13)
    var_15 = '1.000000000002'
    var_16 = [var_15]
    var_17 = {}
    var_18 = module_0.Decimal(*var_16, **var_17)



# Parsed testcases at query #23
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #24
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_currency_eq_equality_same_attributes. Retrieved 3/7 statements.
# Partially parsed test_currency_eq_inequality_different_name. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_inequality_different_code. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_inequality_different_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_eq_inequality_different_type. Retrieved 3/7 statements.
# Partially parsed test_currency_eq_with_different_type_object. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #2
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_currency_constructor_valid_initialization. Retrieved 6/8 statements.
# Partially parsed test_currency_equality_same_hashcache. Retrieved 7/11 statements.
# Partially parsed test_currency_inequality_different_hashcache. Retrieved 9/13 statements.
# Partially parsed test_currency_hash_returns_hashcache. Retrieved 6/9 statements.


import decimal as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 12345
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 2

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 100
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 100
    var_8 = 'UX Dollars'
    var_9 = [var_3]
    var_10 = {}
    var_11 = module_0.Decimal(*var_9, **var_10)
    var_12 = 200

import decimal as module_0

def test_case_0():
    var_0 = 999
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)



# Parsed testcases at query #5
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_currency_constructor_success. Retrieved 6/8 statements.
# Partially parsed test_currency_equality_same_attributes. Retrieved 6/10 statements.
# Partially parsed test_currency_inequality_different_hashcache. Retrieved 7/11 statements.
# Partially parsed test_currency_hash. Retrieved 6/9 statements.


import decimal as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 12345
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 2

import decimal as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'USD'
    var_5 = 'US Dollars'
    var_6 = 2
    var_7 = 100

import decimal as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 'USD'
    var_5 = 'US Dollars'
    var_6 = 2
    var_7 = 100
    var_8 = 200

import decimal as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 999
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 2



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_currency_constructor_valid_assignment. Retrieved 6/8 statements.
# Partially parsed test_currency_equality_same_hash. Retrieved 7/11 statements.
# Partially parsed test_currency_inequality_different_hash. Retrieved 8/12 statements.
# Partially parsed test_currency_hash_consistency. Retrieved 6/9 statements.


import decimal as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 12345
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 2

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 100
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 100
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = 200

import decimal as module_0

def test_case_0():
    var_0 = 12345
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)



# Parsed testcases at query #9
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_currency_constructor_valid_initialization. Retrieved 6/8 statements.
# Partially parsed test_currency_equality_with_same_hashcache. Retrieved 8/12 statements.
# Partially parsed test_currency_inequality_with_different_hashcache. Retrieved 8/12 statements.
# Partially parsed test_currency_hash_returns_hashcache. Retrieved 6/9 statements.


import decimal as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 12345
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 2

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 100
    var_8 = 'Different Name'
    var_9 = [var_3]
    var_10 = {}
    var_11 = module_0.Decimal(*var_9, **var_10)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 100
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = 200

import decimal as module_0

def test_case_0():
    var_0 = 999
    var_1 = 'EUR'
    var_2 = 'Euro'
    var_3 = 2
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)



# Parsed testcases at query #12
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_currency_constructor_success. Retrieved 6/8 statements.
# Partially parsed test_currency_equality_same_hashcache. Retrieved 7/11 statements.
# Partially parsed test_currency_inequality_different_hashcache. Retrieved 8/12 statements.
# Partially parsed test_currency_hash_consistency. Retrieved 6/9 statements.


import decimal as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Decimal(*var_1, **var_2)
    var_4 = 12345
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 2

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 123
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 123
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)
    var_11 = 456

import decimal as module_0

def test_case_0():
    var_0 = 12345
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)



# Parsed testcases at query #16
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True
    var_8 = 'USD'
    var_9 = bool('USD' not in var_0)
    assert var_9 is True
    var_10 = 'USD'
    var_11 = var_0.has(var_10)
    assert var_11 is False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_currency_constructor_valid_assignment. Retrieved 7/9 statements.
# Partially parsed test_currency_equality_same_hashcache. Retrieved 3/7 statements.
# Partially parsed test_currency_inequality_different_name. Retrieved 4/8 statements.
# Partially parsed test_currency_hash_consistency. Retrieved 3/9 statements.
# Partially parsed test_currency_of_factory_creation_usd. Retrieved 5/7 statements.
# Partially parsed test_currency_of_factory_creation_jpy. Retrieved 5/7 statements.
# Partially parsed test_currency_of_factory_creation_crypto. Retrieved 3/5 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 12345
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)

import decimal as module_0

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #19
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_currency_constructor_valid_assignment. Retrieved 7/9 statements.
# Partially parsed test_currency_equality_same_hashcache. Retrieved 6/9 statements.
# Partially parsed test_currency_inequality_different_hashcache. Retrieved 7/10 statements.
# Partially parsed test_currency_hash_returns_hashcache. Retrieved 6/9 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 12345
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 999

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 111
    var_8 = 222

import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 888



# Parsed testcases at query #21
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #22
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #23
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #24
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = bool(var_0.all == [])
    assert var_3 is True
    var_4 = var_0.codes
    var_5 = bool(var_0.codes == [])
    assert var_5 is True
    var_6 = var_0.codenames
    var_7 = bool(var_0.codenames == [])
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_currency_constructor_valid_assignment. Retrieved 7/9 statements.
# Partially parsed test_currency_equality_same_hashcache. Retrieved 3/7 statements.
# Partially parsed test_currency_inequality_different_name. Retrieved 4/8 statements.
# Partially parsed test_currency_hash_consistency. Retrieved 3/9 statements.
# Partially parsed test_currency_hash_inconsistency. Retrieved 4/10 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Decimal(*var_4, **var_5)
    var_7 = 12345
    var_8 = [var_3]
    var_9 = {}
    var_10 = module_0.Decimal(*var_8, **var_9)

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'



