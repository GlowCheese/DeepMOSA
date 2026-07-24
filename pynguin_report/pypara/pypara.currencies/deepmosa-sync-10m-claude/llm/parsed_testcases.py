####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
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

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    var_2 = 0
    var_3 = var_1 == var_2
    var_4 = bool('USD' not in var_0 or var_3)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_currency_constructor_creates_frozen_instance. Retrieved 7/14 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 7/15 statements.
# Partially parsed test_currency_constructor_with_different_decimals. Retrieved 7/14 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = [var_5]
    var_7 = 12345
    var_8 = [var_5]

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = [var_5]
    var_7 = 12345
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'JPY'
    var_3 = 'Japanese Yen'
    var_4 = 0
    var_5 = '1'
    var_6 = [var_5]
    var_7 = 54321
    var_8 = [var_5]

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'ZZZ'
    var_3 = 'Some weird currency'
    var_4 = -1
    var_5 = '0.000000000001'
    var_6 = [var_5]
    var_7 = 99999



# Parsed testcases at query #3
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

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
    var_5 = var_0.get(var_3)
    assert var_5 is None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_currency_constructor_basic. Retrieved 6/15 statements.
# Partially parsed test_currency_constructor_frozen. Retrieved 6/16 statements.
# Partially parsed test_currency_constructor_with_crypto. Retrieved 6/14 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = [var_5]
    var_7 = [var_5]

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'EUR'
    var_3 = 'Euro'
    var_4 = 2
    var_5 = '0.01'
    var_6 = [var_5]
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'BTC'
    var_3 = 'Bitcoin'
    var_4 = -1
    var_5 = '0.000000000001'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'JPY'
    var_3 = 'Japanese Yen'
    var_4 = 0
    var_5 = '1'
    var_6 = [var_5]
    var_7 = [var_5]



# Parsed testcases at query #5
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_fields. Retrieved 6/10 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 6/11 statements.
# Partially parsed test_currency_constructor_fields_are_ordered. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 0
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = 12345

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 0
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = 12345
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = 0
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = 11111
    var_7 = 'USD'
    var_8 = 'US Dollars'
    var_9 = [var_4]
    var_10 = 12345



# Parsed testcases at query #7
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = var_0.codes
    var_4 = bool(var_0.codes == [])
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_currency_constructor_creates_frozen_dataclass. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 5/11 statements.
# Partially parsed test_currency_constructor_with_different_values. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_crypto_type. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = 54321
    var_6 = [var_3]

def test_case_0():
    var_0 = 'BTC'
    var_1 = 'Bitcoin'
    var_2 = -1
    var_3 = '1E+1'
    var_4 = [var_3]
    var_5 = 99999
    var_6 = [var_3]



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
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = var_0.codes
    var_4 = bool(var_0.codes == [])
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

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
    var_5 = var_0.get(var_3)
    assert var_5 is None



# Parsed testcases at query #12
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

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
    var_2 = 0
    var_3 = var_1 > var_2
    var_4 = bool('USD' not in var_0 or var_3)
    assert var_4 is True



# Parsed testcases at query #13
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

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
    var_2 = 0
    var_3 = var_1 == var_2
    var_4 = bool('USD' not in var_0 or var_3)
    assert var_4 is True
    var_5 = 'NONEXISTENT'
    var_6 = var_0.get(var_5)
    assert var_6 is None



# Parsed testcases at query #15
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_different_currency_types. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = [var_3]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = 54321
    var_6 = [var_3]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '0.000000000001'
    var_4 = [var_3]
    var_5 = 99999

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 111
    var_6 = 'BTC'
    var_7 = 'Bitcoin'
    var_8 = 8
    var_9 = '0.00000001'
    var_10 = [var_9]
    var_11 = 222



# Parsed testcases at query #17
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = bool(var_0 is var_1)
    assert var_3 is True
    var_4 = bool(var_1 is var_2)
    assert var_4 is True
    var_5 = bool(var_0 is var_2)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_creates_different_instances. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = [var_3]

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Crypto Currency'
    var_2 = -1
    var_3 = '1'
    var_4 = [var_3]
    var_5 = 54321

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = 99999
    var_6 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 111
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = [var_3]
    var_9 = 222



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_containers. Retrieved 10/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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



# Parsed testcases at query #20
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

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
    var_2 = 0
    var_3 = var_1 > var_2
    var_4 = bool('USD' not in var_0 or var_3)
    assert var_4 is True
    var_5 = 'NONEXISTENT'
    var_6 = var_0.has(var_5)
    assert var_6 is False
    var_7 = var_0.get(var_5)
    assert var_7 is None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_containers. Retrieved 10/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 4/10 statements.
# Partially parsed test_currency_constructor_frozen. Retrieved 4/11 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 4/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 4/10 statements.
# Partially parsed test_currency_constructor_ordering. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Currency'
    var_2 = -1
    var_3 = '0.000000000001'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = 'AAA'
    var_1 = 'First'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 'BBB'
    var_6 = 'Second'
    var_7 = 2
    var_8 = [var_3]



# Parsed testcases at query #23
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = var_0.codes
    var_4 = bool(var_0.codes == [])
    assert var_4 is True



# Parsed testcases at query #24
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_fields. Retrieved 6/10 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 6/11 statements.
# Partially parsed test_currency_constructor_preserves_field_types. Retrieved 6/19 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 6/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 0
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = 12345

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 0
    var_4 = '0.01'
    var_5 = [var_4]
    var_6 = 12345
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = 1
    var_4 = '1'
    var_5 = [var_4]
    var_6 = 54321

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Crypto'
    var_2 = -1
    var_3 = 2
    var_4 = '0.000000000001'
    var_5 = [var_4]
    var_6 = 99999

def test_case_0():
    var_0 = 'XXX'
    var_1 = 'Test'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = 11111
    var_6 = [var_3]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = bool(var_0 is var_1)
    assert var_3 is True
    var_4 = bool(var_1 is var_2)
    assert var_4 is True
    var_5 = bool(var_0 is var_2)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_fields. Retrieved 7/14 statements.
# Partially parsed test_currency_constructor_is_frozen. Retrieved 7/15 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 7/13 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = [var_5]
    var_7 = 12345
    var_8 = [var_5]

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = [var_5]
    var_7 = 12345
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'ZZZ'
    var_3 = 'Weird Currency'
    var_4 = -1
    var_5 = '1'
    var_6 = [var_5]
    var_7 = 54321

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'JPY'
    var_3 = 'Japanese Yen'
    var_4 = 0
    var_5 = '1'
    var_6 = [var_5]
    var_7 = 99999
    var_8 = [var_5]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_currency_eq_same_currencies. Retrieved 3/7 statements.
# Partially parsed test_currency_eq_different_name. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_different_code. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_different_decimals. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_different_type. Retrieved 3/7 statements.
# Partially parsed test_currency_eq_not_currency_instance. Retrieved 3/5 statements.
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
    var_3 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 3

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some Currency'
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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_currency_eq_same_currencies. Retrieved 3/7 statements.
# Partially parsed test_currency_eq_different_names. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_different_codes. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_different_decimals. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_different_type. Retrieved 3/7 statements.
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
    var_3 = 'EUR'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 3

def test_case_0():
    var_0 = 'ABC'
    var_1 = 'Test Currency'
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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #5
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0[var_1]
    var_3 = var_2.code
    assert var_3 == 'USD'

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
    var_1 = 'USD'
    var_2 = var_0[var_1]
    var_3 = 'EUR'
    var_4 = var_0[var_3]
    var_5 = var_2.code
    assert var_5 == 'USD'
    var_6 = var_4.code
    assert var_6 == 'EUR'

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0[var_1]
    var_3 = var_0[var_1]
    var_4 = bool(var_2 is var_3)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

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
    var_2 = 0
    var_3 = var_1 >= var_2
    var_4 = bool('USD' not in var_0 or var_3)
    assert var_4 is True
    var_5 = 'NON_EXISTING'
    var_6 = var_0.get(var_5)
    assert var_6 is None



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Some weird currency'
    var_2 = -1
    var_3 = '1E+12'
    var_4 = [var_3]
    var_5 = 54321
    var_6 = [var_3]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = 99999
    var_6 = [var_3]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_currency_eq_same_currencies. Retrieved 3/7 statements.
# Partially parsed test_currency_eq_different_names. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_different_codes. Retrieved 5/9 statements.
# Partially parsed test_currency_eq_different_decimals. Retrieved 4/8 statements.
# Partially parsed test_currency_eq_different_type. Retrieved 3/7 statements.
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
    var_3 = 'EUR'
    var_4 = 'Euro'

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 3

def test_case_0():
    var_0 = 'ABC'
    var_1 = 'Test Currency'
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

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #9
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_fields. Retrieved 4/9 statements.
# Partially parsed test_currency_constructor_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 4/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 4/10 statements.
# Partially parsed test_currency_constructor_hashcache_is_stored. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Weird Currency'
    var_2 = -1
    var_3 = '1'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = [var_3]

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 999999



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_containers. Retrieved 10/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.codes
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #14
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_currency_registry_constructor. Retrieved 2/3 statements.


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

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = '_CurrencyRegistry__registry'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_CurrencyRegistry__currencies'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = '_CurrencyRegistry__codes'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = '_CurrencyRegistry__codenames'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = '_CurrencyRegistry__ctx_open'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True

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

# Partially parsed test_currency_constructor_creates_frozen_dataclass. Retrieved 7/14 statements.
# Partially parsed test_currency_constructor_frozen_dataclass_prevents_modification. Retrieved 7/15 statements.
# Partially parsed test_currency_constructor_with_different_decimals. Retrieved 7/14 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = [var_5]
    var_7 = 12345
    var_8 = [var_5]

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = [var_5]
    var_7 = 12345
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'JPY'
    var_3 = 'Japanese Yen'
    var_4 = 0
    var_5 = '1'
    var_6 = [var_5]
    var_7 = 54321
    var_8 = [var_5]

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'ZZZ'
    var_3 = 'Crypto'
    var_4 = -1
    var_5 = '0.000000000001'
    var_6 = [var_5]
    var_7 = 99999



# Parsed testcases at query #18
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

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
    var_2 = 0
    var_3 = var_1 > var_2
    var_4 = bool('USD' not in var_0 or var_3)
    assert var_4 is True
    var_5 = 'NONEXISTENT'
    var_6 = var_0.get(var_5)
    assert var_6 is None



# Parsed testcases at query #19
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #20
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
    var_6 = [var_5]
    var_7 = 12345
    var_8 = [var_5]

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = '0.01'
    var_6 = [var_5]
    var_7 = 12345
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = 'JPY'
    var_3 = 'Japanese Yen'
    var_4 = 0
    var_5 = '1'
    var_6 = [var_5]
    var_7 = 54321
    var_8 = [var_5]

def test_case_0():
    var_0 = 'MONEY'
    var_1 = 'CRYPTO'
    var_2 = '0.000000000001'
    var_3 = [var_2]
    var_4 = 'ZZZ'
    var_5 = 'Crypto Currency'
    var_6 = -1
    var_7 = 99999



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_containers. Retrieved 10/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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



# Parsed testcases at query #22
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_currency_constructor_with_valid_parameters. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_frozen. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = [var_3]

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Crypto Currency'
    var_2 = -1
    var_3 = '1'
    var_4 = [var_3]
    var_5 = 54321

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = 99999
    var_6 = [var_3]



# Parsed testcases at query #24
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



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

# Partially parsed test_currency_constructor_creates_instance_with_correct_attributes. Retrieved 5/8 statements.
# Partially parsed test_currency_constructor_creates_frozen_instance. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_is_orderable. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 100
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = [var_3]
    var_9 = 200



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_currency_registry_constructor_initializes_empty_containers. Retrieved 10/13 statements.


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = bool(var_0 is not None)
    assert var_1 is True
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = var_0.all
    var_4 = bool(var_0.all == [])
    assert var_4 is True
    var_5 = var_0.codes
    var_6 = bool(var_0.codes == [])
    assert var_6 is True
    var_7 = var_0.codenames
    var_8 = bool(var_0.codenames == [])
    assert var_8 is True

import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = bool(var_0 is var_1)
    assert var_2 is True

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



# Parsed testcases at query #28
#--------------------------




import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0._CurrencyRegistry__ctx_open
    assert var_1 is False



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_currency_constructor_creates_instance_with_all_fields. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_creates_frozen_instance. Retrieved 5/11 statements.
# Partially parsed test_currency_constructor_with_negative_decimals. Retrieved 5/9 statements.
# Partially parsed test_currency_constructor_with_zero_decimals. Retrieved 5/10 statements.
# Partially parsed test_currency_constructor_preserves_all_parameters. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 12345
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'ZZZ'
    var_1 = 'Crypto'
    var_2 = -1
    var_3 = '1'
    var_4 = [var_3]
    var_5 = 99999

def test_case_0():
    var_0 = 'JPY'
    var_1 = 'Japanese Yen'
    var_2 = 0
    var_3 = '1'
    var_4 = [var_3]
    var_5 = 54321
    var_6 = [var_3]

def test_case_0():
    var_0 = 'EUR'
    var_1 = 'Euro'
    var_2 = 2
    var_3 = '0.01'
    var_4 = [var_3]
    var_5 = 11111



