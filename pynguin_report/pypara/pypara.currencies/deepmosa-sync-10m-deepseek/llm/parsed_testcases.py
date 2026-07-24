####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_currency_registry_constructor. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = []
    var_2 = [var_1]
    var_3 = var_0._CurrencyRegistry__registry
    var_4 = var_0._CurrencyRegistry__currencies
    var_5 = bool(var_0._CurrencyRegistry__currencies == [])
    assert var_5 is True
    var_6 = var_0._CurrencyRegistry__codes
    var_7 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_7 is True
    var_8 = var_0._CurrencyRegistry__codenames
    var_9 = bool(var_0._CurrencyRegistry__codenames == [])
    assert var_9 is True
    var_10 = var_0._CurrencyRegistry__ctx_open
    assert var_10 is False



# Parsed testcases at query #3
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0._CurrencyRegistry__codes)
    assert var_1 is True



# Parsed testcases at query #4
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_currency_constructor. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_currency_constructor_with_valid_arguments. Retrieved 4/16 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 4/12 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 4/9 statements.
# Partially parsed test_currency_constructor_with_cached_hash. Retrieved 4/12 statements.
# Partially parsed test_currency_equality_with_same_hash. Retrieved 4/18 statements.
# Partially parsed test_currency_equality_with_different_hash. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = -1

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = 'UX Dollars'
    var_7 = [var_3]
    var_8 = [var_3]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_currency_registry_constructor_private_attributes. Retrieved 2/3 statements.


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
    var_1 = []
    var_2 = [var_1]
    var_3 = var_0._CurrencyRegistry__registry
    var_4 = var_0._CurrencyRegistry__currencies
    var_5 = bool(var_0._CurrencyRegistry__currencies == [])
    assert var_5 is True
    var_6 = var_0._CurrencyRegistry__codes
    var_7 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_7 is True
    var_8 = var_0._CurrencyRegistry__codenames
    var_9 = bool(var_0._CurrencyRegistry__codenames == [])
    assert var_9 is True
    var_10 = var_0._CurrencyRegistry__ctx_open
    assert var_10 is False



# Parsed testcases at query #10
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0.codes)
    assert var_1 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_currency_constructor. Retrieved 21/45 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = '1.015'
    var_8 = [var_7]
    var_9 = '1.02'
    var_10 = [var_9]
    var_11 = 'JPY'
    var_12 = 'Japanese Yen'
    var_13 = 0
    var_14 = '0.5'
    var_15 = [var_14]
    var_16 = '0'
    var_17 = [var_16]
    var_18 = '1.5'
    var_19 = [var_18]
    var_20 = '2'
    var_21 = [var_20]
    var_22 = 'ZZZ'
    var_23 = 'Some weird currency'
    var_24 = -1
    var_25 = '1.0000000000005'
    var_26 = [var_25]
    var_27 = '1.000000000000'
    var_28 = [var_27]
    var_29 = '1.0000000000015'
    var_30 = [var_29]
    var_31 = '1.000000000002'
    var_32 = [var_31]



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_currency_constructor_valid_input. Retrieved 4/16 statements.
# Partially parsed test_currency_constructor_equality_check. Retrieved 4/20 statements.
# Partially parsed test_currency_constructor_inequality_check. Retrieved 5/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = 'UX Dollars'
    var_7 = [var_3]
    var_8 = [var_3]



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #15
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_currency_constructor_with_valid_inputs. Retrieved 4/16 statements.
# Partially parsed test_currency_constructor_with_invalid_code. Retrieved 4/12 statements.
# Partially parsed test_currency_constructor_with_invalid_name. Retrieved 4/12 statements.
# Partially parsed test_currency_constructor_with_invalid_decimals. Retrieved 5/13 statements.
# Partially parsed test_currency_constructor_with_invalid_type. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = -2
    var_6 = [var_3]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'INVALID_TYPE'
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = [var_4]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



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
    var_1 = bool(not var_0._CurrencyRegistry__codes)
    assert var_1 is True



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_currency_equality. Retrieved 3/7 statements.
# Partially parsed test_currency_inequality. Retrieved 6/10 statements.
# Partially parsed test_currency_hash_equality. Retrieved 3/9 statements.
# Partially parsed test_currency_hash_inequality. Retrieved 6/12 statements.
# Partially parsed test_currency_quantize. Retrieved 7/15 statements.
# Partially parsed test_currency_of_invalid_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_invalid_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_invalid_type. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = '1.015'
    var_8 = [var_7]
    var_9 = '1.02'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'INVALID_TYPE'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_registry_initialization. Retrieved 1/2 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = []
    var_2 = var_0._CurrencyRegistry__registry
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_constructor_initializes_registry. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = []
    var_2 = [var_1]
    var_3 = var_0._CurrencyRegistry__registry
    var_4 = var_0._CurrencyRegistry__currencies
    var_5 = bool(var_0._CurrencyRegistry__currencies == [])
    assert var_5 is True
    var_6 = var_0._CurrencyRegistry__codes
    var_7 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_7 is True
    var_8 = var_0._CurrencyRegistry__codenames
    var_9 = bool(var_0._CurrencyRegistry__codenames == [])
    assert var_9 is True
    var_10 = var_0._CurrencyRegistry__ctx_open
    assert var_10 is False

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_currency_constructor_with_valid_inputs. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/8 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 4/9 statements.
# Partially parsed test_currency_constructor_raises_error_for_invalid_code. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_raises_error_for_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_raises_error_for_invalid_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_raises_error_for_invalid_type. Retrieved 4/6 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = -1

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = bool(False)
    assert var_4 is True



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

# Partially parsed test_currency_constructor_with_valid_arguments. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_with_trimmed_name. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_equality. Retrieved 3/9 statements.
# Partially parsed test_currency_constructor_inequality. Retrieved 4/10 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'



# Parsed testcases at query #26
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

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.all
    var_2 = bool(var_0.all == [])
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = bool(var_0.codes == [])
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codenames
    var_2 = bool(var_0.codenames == [])
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #27
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_CurrencyRegistry_constructor. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = []
    var_2 = [var_1]
    var_3 = var_0._CurrencyRegistry__registry
    var_4 = var_0._CurrencyRegistry__currencies
    var_5 = bool(var_0._CurrencyRegistry__currencies == [])
    assert var_5 is True
    var_6 = var_0._CurrencyRegistry__codes
    var_7 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_7 is True
    var_8 = var_0._CurrencyRegistry__codenames
    var_9 = bool(var_0._CurrencyRegistry__codenames == [])
    assert var_9 is True
    var_10 = var_0._CurrencyRegistry__ctx_open
    assert var_10 is False



# Parsed testcases at query #3
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_currency_constructor_with_valid_arguments. Retrieved 4/7 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 4/7 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 4/7 statements.
# Partially parsed test_currency_equality. Retrieved 5/11 statements.
# Partially parsed test_currency_hash. Retrieved 5/15 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'BTC'
    var_1 = 'Bitcoin'
    var_2 = -1
    var_3 = '0.000000000000'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = 'GBP'
    var_4 = 'British Pound'

def test_case_0():
    var_0 = 'CAD'
    var_1 = 'Canadian Dollar'
    var_2 = 2
    var_3 = 'AUD'
    var_4 = 'Australian Dollar'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = '1.015'
    var_8 = [var_7]
    var_9 = '1.02'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'ETH'
    var_1 = 'Ethereum'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = [var_3]
    var_5 = '1.000000000000'
    var_6 = [var_5]
    var_7 = '1.0000000000015'
    var_8 = [var_7]
    var_9 = '1.000000000002'
    var_10 = [var_9]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_currency_constructor_creates_valid_instance. Retrieved 4/8 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/8 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 3/8 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test___getitem___returns_currency_for_valid_code. Retrieved 4/7 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = var_0['USD']

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'NON-EXISTING'
    var_2 = var_0[var_1]
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0._CurrencyRegistry__codes)
    assert var_1 is True



# Parsed testcases at query #8
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_eq_returns_true_when_currencies_are_identical. Retrieved 3/7 statements.
# Partially parsed test_eq_returns_false_when_currencies_have_different_names. Retrieved 4/8 statements.
# Partially parsed test_eq_returns_false_when_currencies_have_different_codes. Retrieved 4/8 statements.
# Partially parsed test_eq_returns_false_when_currencies_have_different_decimals. Retrieved 4/8 statements.
# Partially parsed test_eq_returns_false_when_currencies_have_different_types. Retrieved 3/7 statements.
# Partially parsed test_eq_returns_false_when_comparing_with_non_currency_object. Retrieved 3/5 statements.


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
    var_3 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #11
#--------------------------

# Partially parsed test___eq___with_same_currency_objects. Retrieved 3/7 statements.
# Partially parsed test___eq___with_different_currency_objects. Retrieved 6/10 statements.
# Partially parsed test___eq___with_non_currency_object. Retrieved 3/5 statements.
# Partially parsed test___eq___with_same_code_but_different_name. Retrieved 4/8 statements.
# Partially parsed test___eq___with_same_code_and_name_but_different_decimals. Retrieved 4/8 statements.
# Partially parsed test___eq___with_same_code_name_decimals_but_different_type. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0

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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_currency_equality_predicate_evaluates_to_true. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_currency_eq_same_instance. Retrieved 3/7 statements.
# Partially parsed test_currency_eq_different_instance. Retrieved 5/9 statements.
# Partially parsed test_currency_eq_same_code_different_name. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_same_code_same_name_different_decimals. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_same_code_same_name_same_decimals_different_type. Retrieved 3/7 statements.
# Partially parsed test_currency_eq_with_non_currency_object. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'

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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'USD'



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
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

# Partially parsed test_currency_constructor. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_currency_constructor_with_valid_arguments. Retrieved 4/10 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 4/10 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/11 statements.
# Partially parsed test_currency_constructor_with_different_currency_type. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '0.000000000001'
    var_4 = [var_3]
    var_5 = -1
    var_6 = [var_3]

def test_case_0():
    var_0 = 'BTC'
    var_1 = 'Bitcoin'
    var_2 = 8
    var_3 = '0.00000001'
    var_4 = [var_3]
    var_5 = [var_3]



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0.codes)
    assert var_1 is True



# Parsed testcases at query #19
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0.codes)
    assert var_1 is True



# Parsed testcases at query #20
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_currency_constructor. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]



# Parsed testcases at query #22
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #23
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True
    var_3 = var_0._CurrencyRegistry__registry
    var_4 = bool(var_0._CurrencyRegistry__registry == {})
    assert var_4 is True
    var_5 = var_0._CurrencyRegistry__currencies
    var_6 = bool(var_0._CurrencyRegistry__currencies == [])
    assert var_6 is True
    var_7 = var_0._CurrencyRegistry__codes
    var_8 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_8 is True
    var_9 = var_0._CurrencyRegistry__codenames
    var_10 = bool(var_0._CurrencyRegistry__codenames == [])
    assert var_10 is True
    var_11 = var_0._CurrencyRegistry__ctx_open
    assert var_11 is False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_currency_constructor. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = [var_3]
    var_7 = [var_3]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_constructor_initializes_registry. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = []
    var_2 = [var_1]
    var_3 = var_0._CurrencyRegistry__registry
    var_4 = var_0._CurrencyRegistry__currencies
    var_5 = bool(var_0._CurrencyRegistry__currencies == [])
    assert var_5 is True
    var_6 = var_0._CurrencyRegistry__codes
    var_7 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_7 is True
    var_8 = var_0._CurrencyRegistry__codenames
    var_9 = bool(var_0._CurrencyRegistry__codenames == [])
    assert var_9 is True
    var_10 = var_0._CurrencyRegistry__ctx_open
    assert var_10 is False



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
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



# Parsed testcases at query #2
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_currency_of_creates_valid_instance. Retrieved 5/10 statements.
# Partially parsed test_currency_of_raises_on_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_alphabetic_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_currency_type. Retrieved 4/6 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash_equality. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

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
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = '1.015'
    var_8 = [var_7]
    var_9 = '1.02'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = [var_3]
    var_5 = '1.000000000000'
    var_6 = [var_5]
    var_7 = '1.0000000000015'
    var_8 = [var_7]
    var_9 = '1.000000000002'
    var_10 = [var_9]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_currency_equality_same_instance. Retrieved 3/7 statements.
# Partially parsed test_currency_equality_different_code. Retrieved 5/9 statements.
# Partially parsed test_currency_equality_different_name. Retrieved 4/8 statements.
# Partially parsed test_currency_equality_different_decimals. Retrieved 6/10 statements.
# Partially parsed test_currency_equality_different_type. Retrieved 6/10 statements.
# Partially parsed test_currency_equality_with_non_currency. Retrieved 3/5 statements.
# Partially parsed test_currency_equality_same_hash. Retrieved 3/7 statements.
# Partially parsed test_currency_equality_different_hash. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'BTC'
    var_4 = 'Bitcoin'
    var_5 = 8

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'



# Parsed testcases at query #6
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test___getitem___returns_currency_for_valid_code. Retrieved 6/10 statements.
# Partially parsed test___getitem___accesses_same_instance_as_registry. Retrieved 6/10 statements.
# Partially parsed test___getitem___after_context_exit_still_works. Retrieved 6/10 statements.
# Partially parsed test___getitem___case_sensitive. Retrieved 6/11 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'USD'
    var_5 = var_0[var_4]
    var_6 = var_5.code
    assert var_6 == 'USD'
    var_7 = var_5.name
    assert var_7 == 'US Dollar'
    var_8 = var_5.type

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'NON-EXISTING'
    var_2 = var_0[var_1]
    var_3 = bool(False)
    assert var_3 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'EUR'
    var_2 = 'Euro'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = var_0[var_4]

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'GBP'
    var_2 = 'British Pound'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = var_0[var_4]
    var_6 = var_5.code
    assert var_6 == 'GBP'

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'usd'
    var_5 = var_0[var_4]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_constructor_singleton_persistence. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test___getitem___returns_currency_for_valid_code. Retrieved 4/9 statements.
# Partially parsed test___getitem___raises_currencylookuperror_for_invalid_code. Retrieved 4/10 statements.
# Partially parsed test___getitem___accesses_same_instance_as_get. Retrieved 4/9 statements.
# Partially parsed test___getitem___after_context_exit_still_works. Retrieved 4/9 statements.
# Partially parsed test___getitem___is_case_sensitive. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'USD'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'XXX'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = 'EUR'

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'British Pound'
    var_2 = 2
    var_3 = 'GBP'

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollar Lower'
    var_2 = 2
    var_3 = 'usd'
    var_4 = 'USD'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_currency_of_creates_valid_instance. Retrieved 5/10 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_on_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_alphabetic_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_currency_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = '1.015'
    var_8 = [var_7]
    var_9 = '1.02'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = [var_3]
    var_5 = '1.000000000000'
    var_6 = [var_5]
    var_7 = '1.0000000000015'
    var_8 = [var_7]
    var_9 = '1.000000000002'
    var_10 = [var_9]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_currency_of_creates_valid_instance. Retrieved 5/10 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_on_invalid_code_type. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_alpha_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_invalid_name_type. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_invalid_decimals_type. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_invalid_currency_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = '1.015'
    var_8 = [var_7]
    var_9 = '1.02'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = [var_3]
    var_5 = '1.000000000000'
    var_6 = [var_5]
    var_7 = '1.0000000000015'
    var_8 = [var_7]
    var_9 = '1.000000000002'
    var_10 = [var_9]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_currency_of_creates_valid_currency. Retrieved 5/10 statements.
# Partially parsed test_currency_of_creates_currency_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_creates_currency_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_alphabetic_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_currencytype_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality_based_on_hashcache. Retrieved 4/10 statements.
# Partially parsed test_currency_hash_based_on_hashcache. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = '1.015'
    var_8 = [var_7]
    var_9 = '1.02'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = [var_3]
    var_5 = '1.000000000000'
    var_6 = [var_5]
    var_7 = '1.0000000000015'
    var_8 = [var_7]
    var_9 = '1.000000000002'
    var_10 = [var_9]



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #22
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



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_constructor_singleton_persistence. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #29
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



# Parsed testcases at query #30
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
    var_1 = var_0._CurrencyRegistry__registry
    var_2 = bool(var_0._CurrencyRegistry__registry is not None)
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__currencies
    var_2 = bool(var_0._CurrencyRegistry__currencies == [])
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codenames
    var_2 = bool(var_0._CurrencyRegistry__codenames == [])
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_0._CurrencyRegistry__registry
    var_3 = bool(var_0._CurrencyRegistry__registry is var_1._CurrencyRegistry__registry)
    assert var_3 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_0._CurrencyRegistry__currencies
    var_3 = bool(var_0._CurrencyRegistry__currencies is var_1._CurrencyRegistry__currencies)
    assert var_3 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_0._CurrencyRegistry__codes
    var_3 = bool(var_0._CurrencyRegistry__codes is var_1._CurrencyRegistry__codes)
    assert var_3 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_0._CurrencyRegistry__codenames
    var_3 = bool(var_0._CurrencyRegistry__codenames is var_1._CurrencyRegistry__codenames)
    assert var_3 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_0._CurrencyRegistry__ctx_open
    var_3 = bool(var_0._CurrencyRegistry__ctx_open is var_1._CurrencyRegistry__ctx_open)
    assert var_3 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_currency_of_valid_arguments. Retrieved 5/9 statements.
# Partially parsed test_currency_of_code_not_string. Retrieved 3/6 statements.
# Partially parsed test_currency_of_code_not_alphabetic. Retrieved 3/6 statements.
# Partially parsed test_currency_of_code_not_uppercase. Retrieved 3/6 statements.
# Partially parsed test_currency_of_name_not_string. Retrieved 3/6 statements.
# Partially parsed test_currency_of_name_empty. Retrieved 3/6 statements.
# Partially parsed test_currency_of_name_has_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_name_has_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_decimals_not_integer. Retrieved 3/6 statements.
# Partially parsed test_currency_of_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_type_not_currency_type. Retrieved 4/6 statements.
# Partially parsed test_currency_of_decimals_zero_quantizer_zero. Retrieved 3/5 statements.
# Partially parsed test_currency_of_decimals_negative_quantizer_max_precision. Retrieved 3/5 statements.
# Partially parsed test_currency_of_decimals_positive_quantizer_custom. Retrieved 4/6 statements.
# Partially parsed test_currency_equality_same. Retrieved 3/8 statements.
# Partially parsed test_currency_equality_different_name. Retrieved 4/9 statements.
# Partially parsed test_currency_quantize_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2.5
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Crypto'
    var_2 = -1

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)

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
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = '1.015'
    var_8 = [var_7]
    var_9 = '1.02'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Crypto'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = [var_3]
    var_5 = '1.000000000000'
    var_6 = [var_5]
    var_7 = '1.0000000000015'
    var_8 = [var_7]
    var_9 = '1.000000000002'
    var_10 = [var_9]



# Parsed testcases at query #32
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
    var_1 = var_0._CurrencyRegistry__registry
    var_2 = bool(var_0._CurrencyRegistry__registry is not None)
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__currencies
    var_2 = bool(var_0._CurrencyRegistry__currencies == [])
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codenames
    var_2 = bool(var_0._CurrencyRegistry__codenames == [])
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #34
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
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__registry
    var_2 = bool(var_0._CurrencyRegistry__registry == {})
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__currencies
    var_2 = bool(var_0._CurrencyRegistry__currencies == [])
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codenames
    var_2 = bool(var_0._CurrencyRegistry__codenames == [])
    assert var_2 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_currency_of_creates_valid_instance. Retrieved 5/10 statements.
# Partially parsed test_currency_of_creates_instance_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_creates_instance_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_alphabetic_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_currencytype_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash_equality. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = '1.015'
    var_8 = [var_7]
    var_9 = '1.02'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = [var_3]
    var_5 = '1.000000000000'
    var_6 = [var_5]
    var_7 = '1.0000000000015'
    var_8 = [var_7]
    var_9 = '1.000000000002'
    var_10 = [var_9]



# Parsed testcases at query #36
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_constructor_singleton_preserves_state. Retrieved 2/3 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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
    var_2 = var_1._CurrencyRegistry__ctx_open
    assert var_2 is True



# Parsed testcases at query #38
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True



# Parsed testcases at query #39
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_currency_of_creates_valid_instance. Retrieved 5/10 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_alphabetic_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_error_for_non_currencytype_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash_equality. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = '1.015'
    var_8 = [var_7]
    var_9 = '1.02'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = [var_3]
    var_5 = '1.000000000000'
    var_6 = [var_5]
    var_7 = '1.0000000000015'
    var_8 = [var_7]
    var_9 = '1.000000000002'
    var_10 = [var_9]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_constructor_singleton_persistence. Retrieved 2/3 statements.
# Partially parsed test_constructor_reinitialization_does_not_override. Retrieved 4/7 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'dummy'
    var_3 = module_0.CurrencyRegistry()
    var_4 = var_3._CurrencyRegistry__ctx_open
    assert var_4 is True
    var_5 = var_3._CurrencyRegistry__registry
    var_6 = bool(var_3._CurrencyRegistry__registry == {'TEST': 'dummy'})
    assert var_6 is True



# Parsed testcases at query #42
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_currency_of_creates_valid_currency. Retrieved 5/10 statements.
# Partially parsed test_currency_of_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_of_raises_on_non_string_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_alpha_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_uppercase_code. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_string_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_name_with_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_raises_on_non_currencytype_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality. Retrieved 3/9 statements.
# Partially parsed test_currency_inequality_different_name. Retrieved 4/10 statements.
# Partially parsed test_currency_quantize_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_negative_decimals. Retrieved 7/15 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = '2'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

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
    var_3 = '1.005'
    var_4 = [var_3]
    var_5 = '1.00'
    var_6 = [var_5]
    var_7 = '1.015'
    var_8 = [var_7]
    var_9 = '1.02'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = [var_3]
    var_5 = '0'
    var_6 = [var_5]
    var_7 = '1.5'
    var_8 = [var_7]
    var_9 = '2'
    var_10 = [var_9]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = [var_3]
    var_5 = '1.000000000000'
    var_6 = [var_5]
    var_7 = '1.0000000000015'
    var_8 = [var_7]
    var_9 = '1.000000000002'
    var_10 = [var_9]



