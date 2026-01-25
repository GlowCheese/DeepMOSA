####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_fields. Retrieved 7/14 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 7/13 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = 12345

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'ZZZ'
    var_3 = 'Crypto Currency'
    var_4 = -1
    var_5 = '1'
    var_6 = 54321

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'JPY'
    var_3 = 'Japanese Yen'
    var_4 = 0
    var_5 = '1'
    var_6 = 99999



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_containers. Retrieved 10/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.all
    var_2 = var_0.codes
    var_3 = var_0.codenames
    var_4 = var_0.all
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = var_0.codes
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_0.codenames
    var_9 = len(var_8)
    assert var_9 == 0



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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_containers. Retrieved 10/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.all
    var_2 = var_0.codes
    var_3 = var_0.codenames
    var_4 = var_0.all
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = var_0.codes
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_0.codenames
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_currency_constructor. Retrieved 6/11 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 6/10 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 6/10 statements.
# Partially parsed test_currency_constructor_frozen. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = '0.01'
    var_5 = 12345

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = 'CRYPTO'
    var_4 = '1E+1'
    var_5 = 54321

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = 'MONEY'
    var_4 = '1'
    var_5 = 99999

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'
    var_4 = '0.01'
    var_5 = 12345



# Parsed testcases at query #8
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #9
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_containers. Retrieved 10/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.all
    var_2 = var_0.codes
    var_3 = var_0.codenames
    var_4 = var_0.all
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = var_0.codes
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_0.codenames
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #11
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_fields. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_different_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = 54321

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Currency'
    var_2 = -1
    var_3 = '0.000000000001'
    var_4 = 99999



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_fields. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_is_orderable. Retrieved 8/14 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 111
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 222

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Currency'
    var_2 = -1
    var_3 = '1E-12'
    var_4 = 54321

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = 99999



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_containers. Retrieved 10/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.all
    var_2 = var_0.codes
    var_3 = var_0.codenames
    var_4 = var_0.all
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = var_0.codes
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_0.codenames
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_currency_constructor_creates_frozen_dataclass. Retrieved 7/14 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 7/15 statements.
# Partially parsed test_currency_constructor_with_different_values. Retrieved 7/14 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = 12345

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = 12345

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'JPY'
    var_3 = 'Japanese Yen'
    var_4 = 0
    var_5 = '1'
    var_6 = 54321

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = '0.000000000001'
    var_3 = 'ZZZ'
    var_4 = 'Crypto'
    var_5 = -1
    var_6 = 99999



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_attributes. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_creates_frozen_instance. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_is_orderable. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 54321

def test_case_0():
    var_0 = 'AAA'
    var_1 = 'Currency A'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 100
    var_5 = 'BBB'
    var_6 = 'Currency B'
    var_7 = 200



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



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




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_currency_of_valid_usd. Retrieved 3/5 statements.
# Partially parsed test_currency_of_valid_jpy. Retrieved 3/5 statements.
# Partially parsed test_currency_of_valid_crypto. Retrieved 3/5 statements.
# Partially parsed test_currency_of_invalid_code_not_string. Retrieved 3/6 statements.
# Partially parsed test_currency_of_invalid_code_not_alpha. Retrieved 3/6 statements.
# Partially parsed test_currency_of_invalid_code_not_uppercase. Retrieved 3/6 statements.
# Partially parsed test_currency_of_invalid_name_not_string. Retrieved 3/6 statements.
# Partially parsed test_currency_of_invalid_name_empty. Retrieved 3/6 statements.
# Partially parsed test_currency_of_invalid_name_leading_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_invalid_name_trailing_space. Retrieved 3/6 statements.
# Partially parsed test_currency_of_invalid_decimals_not_int. Retrieved 3/6 statements.
# Partially parsed test_currency_of_invalid_decimals_less_than_minus_one. Retrieved 3/6 statements.
# Partially parsed test_currency_of_invalid_type_not_currency_type. Retrieved 4/6 statements.
# Partially parsed test_currency_equality_same_values. Retrieved 3/7 statements.
# Partially parsed test_currency_equality_different_name. Retrieved 4/8 statements.
# Partially parsed test_currency_hash_same_values. Retrieved 3/9 statements.
# Partially parsed test_currency_hash_different_name. Retrieved 4/10 statements.
# Partially parsed test_currency_quantize_usd. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_jpy. Retrieved 7/15 statements.
# Partially parsed test_currency_quantize_crypto. Retrieved 7/15 statements.


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
    var_0 = 123
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'US1'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'usd'
    var_1 = 'US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 123
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ''
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = ' US Dollars'
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars '
    var_2 = 2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2.5

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = -2

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'MONEY'

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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '0.5'
    var_4 = '0'
    var_5 = '1.5'
    var_6 = '2'

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1.0000000000005'
    var_4 = '1.000000000000'
    var_5 = '1.0000000000015'
    var_6 = '1.000000000002'



# Parsed testcases at query #22
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



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

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_creates_frozen_object. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_different_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_crypto_type. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 54321

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = 99999

def test_case_0():
    var_0 = 'BTC'
    var_1 = 'Bitcoin'
    var_2 = 8
    var_3 = '0.00000001'
    var_4 = 77777

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Currency'
    var_2 = -1
    var_3 = '0.000000000001'
    var_4 = 55555



# Parsed testcases at query #25
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_currency_eq_same_currencies. Retrieved 3/7 statements.
# Partially parsed test_currency_eq_different_names. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_different_decimals. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_different_types. Retrieved 5/9 statements.
# Partially parsed test_currency_eq_with_non_currency_object. Retrieved 3/5 statements.
# Partially parsed test_currency_eq_reflexive. Retrieved 3/5 statements.
# Partially parsed test_currency_eq_symmetric. Retrieved 3/7 statements.
# Partially parsed test_currency_eq_transitive. Retrieved 3/9 statements.


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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'ZZZ'
    var_4 = 'Weird Currency'

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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #2
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0[var_1]

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'NON-EXISTING'
    var_2 = var_0[var_1]

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0[var_1]
    var_3 = 'EUR'
    var_4 = var_0[var_3]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_state. Retrieved 5/8 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = var_0.all
    var_3 = var_0.codes
    var_4 = var_0.codenames



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_currency_equality_with_same_currencies. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #5
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #6
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_attributes. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_maintains_order. Retrieved 8/14 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'AUD'
    var_1 = 'Australian Dollar'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 100
    var_5 = 'USD'
    var_6 = 'US Dollars'
    var_7 = 200

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Currency'
    var_2 = -1
    var_3 = '1E+1'
    var_4 = 54321

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = 99999



# Parsed testcases at query #8
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_is_frozen. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1'
    var_4 = 54321

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = 99999

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_registry. Retrieved 4/7 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.all
    var_2 = var_0.codes
    var_3 = var_0.codenames



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_attributes. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_ordering. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = 54321

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '0.000000000001'
    var_4 = 99999

def test_case_0():
    var_0 = 'AAA'
    var_1 = 'Currency A'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 1000
    var_5 = 'ZZZ'
    var_6 = 'Currency Z'
    var_7 = 2000



# Parsed testcases at query #12
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_currency_constructor_creates_frozen_dataclass. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 5/11 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_all_attributes_assigned. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 54321

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Currency'
    var_2 = -1
    var_3 = '1'
    var_4 = 99999

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = 11111

def test_case_0():
    var_0 = 'GBP'
    var_1 = 'British Pound'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 77777
    var_5 = 'code'
    var_6 = 'name'
    var_7 = 'decimals'
    var_8 = 'type'
    var_9 = 'quantizer'
    var_10 = 'hashcache'



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #15
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



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

# Partially parsed test_currency_constructor_creates_instance_with_all_attributes. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_creates_frozen_instance. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_instances_are_orderable. Retrieved 8/14 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 54321

def test_case_0():
    var_0 = 'AAA'
    var_1 = 'Currency A'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 111
    var_5 = 'BBB'
    var_6 = 'Currency B'
    var_7 = 222

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Currency'
    var_2 = -1
    var_3 = '1'
    var_4 = 99999

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = 77777



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




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_attributes. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_preserves_hashcache. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1E+1'
    var_4 = 54321

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = 99999

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 777



# Parsed testcases at query #21
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #22
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    var_2 = 0
    var_3 = var_1 > var_2



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_attributes. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_different_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = 54321

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Crypto'
    var_2 = -1
    var_3 = '0.000000000001'
    var_4 = 99999



# Parsed testcases at query #24
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_containers. Retrieved 10/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.all
    var_2 = var_0.codes
    var_3 = var_0.codenames
    var_4 = var_0.all
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = var_0.codes
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_0.codenames
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #26
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_fields. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_different_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 12345

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = 67890

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '0.000000000001'
    var_4 = 11111



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_containers. Retrieved 10/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.all
    var_2 = var_0.codes
    var_3 = var_0.codenames
    var_4 = var_0.all
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = var_0.codes
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_0.codenames
    var_9 = len(var_8)
    assert var_9 == 0



