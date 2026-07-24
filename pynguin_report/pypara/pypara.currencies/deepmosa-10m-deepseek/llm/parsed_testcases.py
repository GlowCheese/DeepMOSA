####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_Currency_constructor_with_valid_input. Retrieved 4/16 statements.
# Partially parsed test_Currency_constructor_with_invalid_code. Retrieved 4/12 statements.
# Partially parsed test_Currency_constructor_with_invalid_name. Retrieved 4/12 statements.
# Partially parsed test_Currency_constructor_with_invalid_decimals. Retrieved 5/13 statements.
# Partially parsed test_Currency_constructor_with_invalid_type. Retrieved 5/11 statements.


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
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_currency_registry_constructor. Retrieved 1/2 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True



# Parsed testcases at query #6
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



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
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #9
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #10
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0._CurrencyRegistry__codes)
    assert var_1 is True



# Parsed testcases at query #11
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_currency_constructor. Retrieved 11/25 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = '1'
    var_9 = [var_8]
    var_10 = 'ZZZ'
    var_11 = 'Some weird currency'
    var_12 = -1



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 4/7 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 4/7 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_equality. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize. Retrieved 21/45 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_currency_constructor. Retrieved 8/23 statements.


import pypara.commons.numbers as module_0

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
    var_11 = module_0.make_quantizer(var_2)



# Parsed testcases at query #15
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0.codes)
    assert var_1 is True



# Parsed testcases at query #16
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0._CurrencyRegistry__codes)
    assert var_1 is True



# Parsed testcases at query #17
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_currency_constructor_with_valid_arguments. Retrieved 4/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/7 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 3/7 statements.
# Partially parsed test_currency_equality. Retrieved 4/10 statements.
# Partially parsed test_currency_hash_equality. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_with_positive_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_zero_decimals. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_with_negative_decimals. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]

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



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_currency_constructor_initializes_fields_correctly. Retrieved 4/8 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 3/8 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/8 statements.
# Partially parsed test_currency_constructor_equality. Retrieved 4/11 statements.
# Partially parsed test_currency_constructor_inequality. Retrieved 9/21 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Currency'
    var_2 = -1

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)

import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = module_0.make_quantizer(var_2)
    var_8 = module_0.make_quantizer(var_2)



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
    var_1 = bool(not var_0.codes)
    assert var_1 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_CurrencyRegistry_constructor. Retrieved 2/4 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_currency_constructor_valid_input. Retrieved 4/11 statements.
# Partially parsed test_currency_constructor_zero_decimals. Retrieved 4/11 statements.
# Partially parsed test_currency_constructor_negative_decimals. Retrieved 5/12 statements.


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



# Parsed testcases at query #25
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0._CurrencyRegistry__codes)
    assert var_1 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_currency_constructor_with_valid_arguments. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/8 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 4/9 statements.
# Partially parsed test_currency_equality. Retrieved 4/14 statements.
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
    var_3 = -1

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



# Parsed testcases at query #27
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0._CurrencyRegistry__codes)
    assert var_1 is True



# Parsed testcases at query #28
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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_equality_with_same_currency. Retrieved 3/7 statements.
# Partially parsed test_equality_with_different_currency. Retrieved 6/10 statements.
# Partially parsed test_equality_with_same_code_but_different_name. Retrieved 4/8 statements.
# Partially parsed test_equality_with_same_code_name_but_different_decimals. Retrieved 4/8 statements.
# Partially parsed test_equality_with_same_code_name_decimals_but_different_type. Retrieved 3/7 statements.
# Partially parsed test_equality_with_non_currency_object. Retrieved 3/5 statements.


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
    var_1 = bool(not var_0.codes)
    assert var_1 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_currency_constructor_with_valid_inputs. Retrieved 4/8 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_with_invalid_code. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_invalid_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_invalid_type. Retrieved 4/6 statements.


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
    var_3 = 'InvalidType'
    var_4 = bool(False)
    assert var_4 is True



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



# Parsed testcases at query #8
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



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_currency_constructor. Retrieved 4/8 statements.


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_currency_constructor_success. Retrieved 4/11 statements.
# Partially parsed test_currency_constructor_invalid_code. Retrieved 4/12 statements.
# Partially parsed test_currency_constructor_invalid_name. Retrieved 4/12 statements.
# Partially parsed test_currency_constructor_invalid_decimals. Retrieved 4/12 statements.
# Partially parsed test_currency_constructor_invalid_type. Retrieved 5/11 statements.
# Partially parsed test_currency_constructor_invalid_quantizer. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 'two'
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = [var_4]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'



# Parsed testcases at query #12
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0.codes)
    assert var_1 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_currency_constructor_with_valid_arguments. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 3/8 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 4/9 statements.
# Partially parsed test_currency_constructor_with_invalid_code. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_invalid_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_with_invalid_type. Retrieved 4/6 statements.


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
    var_3 = 'INVALID_TYPE'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #15
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




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = bool(var_0._CurrencyRegistry__codes == [])
    assert var_2 is True



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #19
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_currency_constructor_creates_immutable_instance. Retrieved 4/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 4/7 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 4/7 statements.
# Partially parsed test_currency_constructor_raises_for_invalid_code. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_raises_for_empty_name. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_raises_for_invalid_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_raises_for_non_integer_decimals. Retrieved 3/6 statements.
# Partially parsed test_currency_constructor_raises_for_invalid_currency_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality_based_on_fields. Retrieved 4/10 statements.
# Partially parsed test_currency_hash_consistency. Retrieved 4/14 statements.
# Partially parsed test_currency_quantize_method. Retrieved 14/30 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '0.000000000001'
    var_4 = [var_3]

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
    var_2 = '2'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = bool(False)
    assert var_4 is True

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



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_CurrencyRegistry_constructor_initializes_registry. Retrieved 2/3 statements.


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



# Parsed testcases at query #24
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



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_currency_constructor_with_valid_arguments. Retrieved 4/16 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 4/12 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 4/9 statements.
# Partially parsed test_currency_equality. Retrieved 4/18 statements.
# Partially parsed test_currency_inequality. Retrieved 8/22 statements.
# Partially parsed test_currency_hash. Retrieved 4/20 statements.
# Partially parsed test_currency_hash_inequality. Retrieved 8/24 statements.


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
    var_6 = [var_3]
    var_7 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = '1'
    var_10 = [var_9]
    var_11 = [var_9]

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
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = '1'
    var_10 = [var_9]
    var_11 = [var_9]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 4/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 4/7 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 4/7 statements.
# Partially parsed test_currency_constructor_with_uppercase_code. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_with_trimmed_name. Retrieved 3/5 statements.
# Partially parsed test_currency_constructor_equality. Retrieved 3/9 statements.
# Partially parsed test_currency_constructor_inequality. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '0.000000000001'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'British Pound'
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



# Parsed testcases at query #28
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(not var_0._CurrencyRegistry__codes)
    assert var_1 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_currency_registry_constructor. Retrieved 1/2 statements.


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



