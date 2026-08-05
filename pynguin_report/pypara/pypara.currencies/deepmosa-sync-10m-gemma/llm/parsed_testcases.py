####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = bool('USD' not in var_0)
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_currency_constructor_valid. Retrieved 9/14 statements.
# Partially parsed test_currency_equality_same_hash. Retrieved 9/15 statements.
# Partially parsed test_currency_equality_different_hash. Retrieved 9/15 statements.
# Partially parsed test_currency_hash. Retrieved 7/12 statements.


import decimal as module_0

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = '0.01'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Decimal(*var_3, **var_4)
    var_6 = 12345
    var_7 = 'USD'
    var_8 = 'US Dollars'
    var_9 = 2
    var_10 = [var_2]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)

import decimal as module_0

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = 99
    var_9 = 'Other Name'
    var_10 = [var_4]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)

import decimal as module_0

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = 99
    var_9 = [var_4]
    var_10 = {}
    var_11 = module_0.Decimal(*var_9, **var_10)
    var_12 = 100

import decimal as module_0

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Decimal(*var_5, **var_6)
    var_8 = 888



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_codes_buffer_is_not_empty_after_registration. Retrieved 8/18 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = var_1.codes
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 'USD'
    var_9 = bool('USD' in var_1.codes)
    assert var_9 is True
    var_10 = 'EUR'
    var_11 = bool('EUR' in var_1.codes)
    assert var_11 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_currency_constructor_valid_usd. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_jpy. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_crypto. Retrieved 3/5 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash_consistency. Retrieved 3/9 statements.
# Partially parsed test_currency_quantize_usd. Retrieved 11/15 statements.
# Partially parsed test_currency_quantize_jpy. Retrieved 11/15 statements.
# Partially parsed test_currency_quantize_crypto. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

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



# Parsed testcases at query #7
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #8
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = bool('USD' not in var_0)
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_currency_constructor_valid_usd. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_jpy. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_crypto. Retrieved 3/5 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash. Retrieved 3/9 statements.
# Partially parsed test_currency_quantize_usd. Retrieved 11/16 statements.
# Partially parsed test_currency_quantize_jpy. Retrieved 11/16 statements.
# Partially parsed test_currency_quantize_crypto. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_codes_buffer_is_not_empty_after_registration. Retrieved 8/15 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = var_0.codes
    var_7 = len(var_6)
    var_8 = bool(var_7 != 0)
    assert var_8 is True
    var_9 = 'USD'
    var_10 = bool('USD' in var_0.codes)
    assert var_10 is True



# Parsed testcases at query #11
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.all
    var_2 = bool(var_0.all == [])
    assert var_2 is True
    var_3 = var_0.codes
    var_4 = bool(var_0.codes == [])
    assert var_4 is True
    var_5 = var_0.codenames
    var_6 = bool(var_0.codenames == [])
    assert var_6 is True
    var_7 = len(var_0)
    assert var_7 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = bool('USD' not in var_0)
    assert var_2 is True
    var_3 = 'USD'
    var_4 = var_0.has(var_3)
    assert var_4 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_currency_constructor_valid_assignment. Retrieved 7/9 statements.
# Partially parsed test_currency_equality_same_values. Retrieved 3/7 statements.
# Partially parsed test_currency_equality_different_names. Retrieved 4/8 statements.
# Partially parsed test_currency_hash_consistency. Retrieved 3/9 statements.
# Partially parsed test_currency_quantize_usd. Retrieved 11/15 statements.
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



# Parsed testcases at query #13
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_currency_constructor_valid_usd. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_jpy. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_crypto. Retrieved 3/5 statements.
# Partially parsed test_currency_equality_same_attributes. Retrieved 3/9 statements.
# Partially parsed test_currency_inequality_different_name. Retrieved 4/8 statements.
# Partially parsed test_currency_inequality_different_decimals. Retrieved 4/8 statements.
# Partially parsed test_currency_quantize_usd. Retrieved 11/16 statements.
# Partially parsed test_currency_quantize_jpy. Retrieved 11/16 statements.
# Partially parsed test_currency_quantize_crypto. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

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
    var_3 = 3

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



# Parsed testcases at query #15
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #16
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_currency_constructor_valid_data. Retrieved 6/8 statements.
# Partially parsed test_currency_equality_same_hash. Retrieved 7/11 statements.
# Partially parsed test_currency_inequality_different_hash. Retrieved 9/13 statements.
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
    var_7 = 99
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
    var_7 = 99
    var_8 = 'UX Dollars'
    var_9 = [var_3]
    var_10 = {}
    var_11 = module_0.Decimal(*var_9, **var_10)
    var_12 = 88

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



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #19
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



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_currency_constructor_valid_initialization. Retrieved 6/8 statements.
# Partially parsed test_currency_equality_same_hashcache. Retrieved 7/11 statements.
# Partially parsed test_currency_inequality_different_hashcache. Retrieved 9/13 statements.
# Partially parsed test_currency_hash_returns_hashcache. Retrieved 6/9 statements.
# Partially parsed test_currency_order_comparison. Retrieved 9/13 statements.


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
    var_0 = 98765
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
    var_8 = 'GBP'
    var_9 = 'British Pounds'
    var_10 = [var_3]
    var_11 = {}
    var_12 = module_0.Decimal(*var_10, **var_11)



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

# Partially parsed test_currency_constructor_valid_usd. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_jpy. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_crypto. Retrieved 3/5 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash. Retrieved 3/9 statements.
# Partially parsed test_currency_quantize_usd. Retrieved 11/18 statements.
# Partially parsed test_currency_quantize_jpy. Retrieved 11/16 statements.
# Partially parsed test_currency_quantize_crypto. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

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



# Parsed testcases at query #24
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #25
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = bool('USD' not in var_0)
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_currency_constructor_valid_usd. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_jpy. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_crypto. Retrieved 3/5 statements.
# Partially parsed test_currency_equality_identical. Retrieved 3/9 statements.
# Partially parsed test_currency_inequality_different_name. Retrieved 4/10 statements.
# Partially parsed test_currency_quantize_usd. Retrieved 11/15 statements.
# Partially parsed test_currency_quantize_jpy. Retrieved 11/15 statements.
# Partially parsed test_currency_quantize_crypto. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_getitem_success. Retrieved 3/7 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = var_0['USD']

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'NON-EXISTING'
    var_2 = var_0[var_1]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_currency_equality_same_attributes. Retrieved 3/7 statements.
# Partially parsed test_currency_equality_different_name. Retrieved 4/8 statements.
# Partially parsed test_currency_equality_different_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_equality_different_type. Retrieved 5/10 statements.
# Partially parsed test_currency_equality_different_code. Retrieved 5/9 statements.
# Partially parsed test_currency_equality_with_unrelated_type. Retrieved 3/5 statements.


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
    var_3 = 'US'
    var_4 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'USC'
    var_4 = 'US Crypto'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pounds'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #5
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_get_item_success. Retrieved 3/7 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = var_0['USD']
    var_4 = var_0['USD'].code
    assert var_4 == 'USD'

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'NON-EXISTING'
    var_2 = var_0[var_1]



# Parsed testcases at query #7
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'NON-EXISTING'
    var_2 = var_0[var_1]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_currency_eq_isinstance_check. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = None
    var_4 = 123



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_codes_buffer_is_not_empty_after_registration. Retrieved 4/17 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'AED'
    var_2 = 'UAE Dirham'
    var_3 = 2



# Parsed testcases at query #10
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_currency_constructor_success. Retrieved 6/8 statements.
# Partially parsed test_currency_equality. Retrieved 4/8 statements.
# Partially parsed test_currency_hash. Retrieved 3/8 statements.
# Partially parsed test_currency_quantize_usd. Retrieved 11/15 statements.
# Partially parsed test_currency_quantize_jpy. Retrieved 11/15 statements.
# Partially parsed test_currency_quantize_crypto. Retrieved 11/15 statements.


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_currency_constructor_valid_usd. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_jpy. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_crypto. Retrieved 3/5 statements.
# Partially parsed test_currency_equality_same_attributes. Retrieved 3/9 statements.
# Partially parsed test_currency_inequality_different_name. Retrieved 4/8 statements.
# Partially parsed test_currency_inequality_different_decimals. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

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
    var_3 = 3



# Parsed testcases at query #13
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = bool('USD' not in var_0)
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = bool('USD' not in var_0)
    assert var_2 is True



# Parsed testcases at query #15
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__registry
    var_2 = bool(var_0._CurrencyRegistry__registry == {})
    assert var_2 is True
    var_3 = var_0._CurrencyRegistry__currencies
    var_4 = bool(var_0._CurrencyRegistry__currencies == [])
    assert var_4 is True
    var_5 = var_0._CurrencyRegistry__codes
    var_6 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_6 is True
    var_7 = var_0._CurrencyRegistry__codenames
    var_8 = bool(var_0._CurrencyRegistry__codenames == [])
    assert var_8 is True
    var_9 = var_0._CurrencyRegistry__ctx_open
    assert var_9 is False

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_currency_constructor_valid_usd. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_jpy. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_crypto. Retrieved 3/5 statements.
# Partially parsed test_currency_equality_same_values. Retrieved 3/9 statements.
# Partially parsed test_currency_inequality_different_name. Retrieved 4/10 statements.
# Partially parsed test_currency_inequality_different_decimals. Retrieved 4/8 statements.
# Partially parsed test_currency_quantize_rounding_half_to_even. Retrieved 11/16 statements.
# Partially parsed test_currency_quantize_zero_decimals. Retrieved 11/16 statements.
# Partially parsed test_currency_quantize_negative_decimals. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

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
    var_3 = 0

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



# Parsed testcases at query #17
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

# Partially parsed test_currency_constructor_valid_usd. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_jpy. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_crypto. Retrieved 3/5 statements.
# Partially parsed test_currency_equality_same_values. Retrieved 3/9 statements.
# Partially parsed test_currency_inequality_different_names. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'



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

# Partially parsed test_currency_constructor_valid_usd. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_jpy. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_crypto. Retrieved 3/5 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash. Retrieved 3/9 statements.
# Partially parsed test_currency_quantize_usd. Retrieved 11/16 statements.
# Partially parsed test_currency_quantize_jpy. Retrieved 11/16 statements.
# Partially parsed test_currency_quantize_crypto. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

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



# Parsed testcases at query #24
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #25
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #26
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

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_currency_constructor_valid_usd. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_jpy. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_valid_crypto. Retrieved 3/5 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash. Retrieved 3/9 statements.
# Partially parsed test_currency_quantize_usd. Retrieved 11/16 statements.
# Partially parsed test_currency_quantize_jpy. Retrieved 11/16 statements.
# Partially parsed test_currency_quantize_crypto. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_currency_constructor_valid_usd. Retrieved 8/17 statements.
# Partially parsed test_currency_equality_and_hash. Retrieved 11/26 statements.


import decimal as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Decimal(*var_6, **var_7)
    var_9 = [var_5]
    var_10 = {}
    var_11 = module_0.Decimal(*var_9, **var_10)

import decimal as module_0

def test_case_0():
    var_0 = 1
    var_1 = '0.01'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Decimal(*var_2, **var_3)
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 2
    var_8 = 'EUR'
    var_9 = 'Euro'
    var_10 = 2
    var_11 = [var_1]
    var_12 = {}
    var_13 = module_0.Decimal(*var_11, **var_12)
    var_14 = [var_1]
    var_15 = {}
    var_16 = module_0.Decimal(*var_14, **var_15)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_codes_buffer_is_not_empty_after_registration. Retrieved 5/12 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = var_0.codes
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_0.codes[0]
    assert var_5 == 'USD'



# Parsed testcases at query #30
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



