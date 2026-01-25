####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __enter__ returns the __register method and opens the context.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = callable(var_2)
    var_4 = None
    var_5 = var_1.__exit__(var_4, var_4, var_4)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test Currency.__eq__ method'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'United States Dollars'
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 3



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test the has method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'XXX'
    var_3 = 'EUR'
    var_4 = 'GBP'
    var_5 = 'JPY'
    var_6 = 'INVALID'
    var_7 = 'ZZZ'
    var_8 = ''



# Parsed testcases at query #4
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __contains__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test the __gt__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'GBP'
    var_7 = 'British Pound'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8
    var_14 = 'XAU'
    var_15 = 'Gold'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test the __le__ (less than or equal) ordering method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '\n    Test that Currency is frozen and does not allow attribute deletion.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2



# Parsed testcases at query #8
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __exit__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'TST'
    var_3 = 'Test Currency'
    var_4 = 2
    var_5 = 'ZZA'
    var_6 = 'ZZA Currency'
    var_7 = 'AAA'
    var_8 = 'AAA Currency'
    var_9 = var_1.all
    var_10 = len(var_9)
    assert var_10 == 3

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __exit__ properly sorts currencies by code.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'XYZ'
    var_3 = 'XYZ Currency'
    var_4 = 2
    var_5 = 'ABC'
    var_6 = 'ABC Currency'
    var_7 = 'MNO'
    var_8 = 'MNO Currency'
    var_9 = var_1.codes
    var_10 = sorted(var_9)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __exit__ updates all internal buffers.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'CCC'
    var_3 = 'CCC Currency'
    var_4 = 2
    var_5 = 'AAA'
    var_6 = 'AAA Currency'
    var_7 = 'BBB'
    var_8 = 'BBB Currency'
    var_9 = var_1.all
    var_10 = len(var_9)
    assert var_10 == 3
    var_11 = var_1.codes
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = var_1.codenames
    var_14 = len(var_13)
    assert var_14 == 3



# Parsed testcases at query #9
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyLookupError constructor and behavior.'
    var_1 = 'XYZ'
    var_2 = module_0.CurrencyLookupError(var_1)
    var_3 = str(var_2)
    assert var_3 == "Currency identified by code 'XYZ' does not exist"
    var_4 = 'ABC'
    var_5 = module_0.CurrencyLookupError(var_4)
    var_6 = str(var_5)
    assert var_6 == "Currency identified by code 'ABC' does not exist"
    var_7 = ''
    var_8 = module_0.CurrencyLookupError(var_7)
    var_9 = str(var_8)
    assert var_9 == "Currency identified by code '' does not exist"
    var_10 = 'EUR'
    var_11 = module_0.CurrencyLookupError(var_10)



# Parsed testcases at query #10
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __contains__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #11
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry constructor and singleton behavior.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = len(var_1)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = var_1.has(var_4)
    var_6 = var_1.get(var_4)
    assert var_6 is None
    var_7 = None
    var_8 = var_1.get(var_4, var_7)
    assert var_8 is None
    var_9 = 'USD'
    var_10 = var_1[var_9]
    var_11 = 'USD'
    var_12 = 'US Dollar'
    var_13 = 2
    var_14 = len(var_1)
    assert var_14 == 1
    var_15 = var_1.has(var_12)
    var_16 = var_1.get(var_12)
    var_17 = 'EUR'
    var_18 = 'Euro'
    var_19 = 2
    var_20 = 'USD'
    var_21 = 'US Dollar'
    var_22 = 2



# Parsed testcases at query #12
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __contains__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = None
    var_3 = var_2 in var_1
    assert var_3 is False



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the CurrencyRegistry constructor and singleton behavior.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = len(var_1)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = var_1.has(var_4)
    var_6 = var_1.get(var_4)
    assert var_6 is None
    var_7 = None
    var_8 = var_1.get(var_4, var_7)
    assert var_8 is None
    var_9 = 'USD'
    var_10 = var_1[var_9]
    var_11 = 'USD'
    var_12 = 'US Dollar'
    var_13 = 2
    var_14 = len(var_1)
    assert var_14 == 1
    var_15 = var_1.has(var_12)
    var_16 = var_1.get(var_12)
    var_17 = var_1.all
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'USD'
    var_20 = 'US Dollar'
    var_21 = 2
    var_22 = str(var_19)
    var_23 = 'EUR'
    var_24 = 'Euro'
    var_25 = 2
    var_26 = 'XXX'
    var_27 = 'Unknown'
    var_28 = 2
    var_29 = 'NON_EXISTING'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test CurrencyRegistry.__getitem__ method.'
    var_1 = 'USD'
    var_2 = 'JPY'
    var_3 = 'NON-EXISTING'
    var_4 = 'usd'
    var_5 = 'code'
    var_6 = 'name'
    var_7 = 'decimals'
    var_8 = 'type'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test the get method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'XXX'
    var_3 = 'EUR'
    var_4 = 'NON-EXISTING'
    var_5 = 'INVALID'
    var_6 = None
    var_7 = 'GBP'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test the has method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 'JPY'
    var_4 = 'GBP'
    var_5 = 'XXX'
    var_6 = 'ZZZ'
    var_7 = 'INVALID'
    var_8 = ''
    var_9 = 'usd'
    var_10 = 'Usd'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test the __eq__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'United States Dollars'
    var_5 = 3
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test the get method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'XXX'
    var_3 = 'NON_EXISTENT'
    var_4 = None
    var_5 = 'EUR'
    var_6 = 'JPY'



# Parsed testcases at query #20
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __contains__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #21
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __len__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_1)
    var_3 = var_1.all
    var_4 = len(var_3)
    var_5 = var_1.codes
    var_6 = len(var_5)
    var_7 = var_1.codenames
    var_8 = len(var_7)
    var_9 = len(var_1)
    var_10 = len(var_1)



# Parsed testcases at query #22
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry constructor and singleton behavior.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = len(var_1)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = var_1.has(var_4)
    var_6 = 'EUR'
    var_7 = var_1.has(var_6)
    var_8 = var_1.get(var_4)
    assert var_8 is None
    var_9 = var_1.get(var_6)
    assert var_9 is None
    var_10 = 'USD'
    var_11 = var_1[var_10]
    var_12 = 'USD'
    var_13 = 'US Dollar'
    var_14 = 2
    var_15 = 'EUR'
    var_16 = 'Euro'
    var_17 = len(var_1)
    assert var_17 == 2
    var_18 = var_1.has(var_14)
    var_19 = var_1.has(var_6)
    var_20 = var_1.all
    var_21 = len(var_20)
    assert var_21 == 2
    var_22 = var_1.get(var_14)
    var_23 = var_1[var_6]
    var_24 = 'USD'
    var_25 = 'US Dollar'
    var_26 = 2
    var_27 = 'GBP'
    var_28 = 'British Pound'
    var_29 = 2



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test Currency class creation and validation'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = '1.005'
    var_11 = '1.00'
    var_12 = '1.015'
    var_13 = '1.02'
    var_14 = '0.5'
    var_15 = '0'
    var_16 = '1.5'
    var_17 = '2'
    var_18 = '1.0000000000005'
    var_19 = '1.000000000000'
    var_20 = '1.0000000000015'
    var_21 = '1.000000000002'
    var_22 = 'UX Dollars'
    var_23 = 'US1'
    var_24 = 'US Dollars'
    var_25 = 2
    var_26 = 'Usd'
    var_27 = 'US Dollars'
    var_28 = 2
    var_29 = 123
    var_30 = 'US Dollars'
    var_31 = 2
    var_32 = 'USD'
    var_33 = ''
    var_34 = 2
    var_35 = 'USD'
    var_36 = ' US Dollars'
    var_37 = 2
    var_38 = 'USD'
    var_39 = 'US Dollars '
    var_40 = 2
    var_41 = 'USD'
    var_42 = 123
    var_43 = 2
    var_44 = 'USD'
    var_45 = 'US Dollars'
    var_46 = 2.5
    var_47 = 'USD'
    var_48 = 'US Dollars'
    var_49 = -2
    var_50 = 'USD'
    var_51 = 'US Dollars'
    var_52 = 2
    var_53 = 'MONEY'
    var_54 = 'XAU'
    var_55 = 'Gold'
    var_56 = 'BTC'
    var_57 = 'Bitcoin'
    var_58 = 8
    var_59 = 'ALT'
    var_60 = 'Alternative'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test the __gt__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'GBP'
    var_7 = 'British Pound'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8
    var_14 = 'CHF'
    var_15 = 'Swiss Franc'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test the __hash__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'United States Dollar'
    var_8 = 3
    var_9 = 'Dollar'
    var_10 = 'Yen'



# Parsed testcases at query #26
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry constructor and singleton behavior.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = len(var_1)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = var_1.has(var_4)
    assert var_5 is False
    var_6 = var_1.get(var_4)
    assert var_6 is None
    var_7 = None
    var_8 = var_1.get(var_4, var_7)
    assert var_8 is None
    var_9 = 'TST'
    var_10 = 'Test Currency'
    var_11 = 2
    var_12 = len(var_1)
    assert var_12 == 1
    var_13 = var_1.has(var_9)
    assert var_13 is True
    var_14 = var_1.get(var_9)
    var_15 = 'AAA'
    var_16 = 'First Currency'
    var_17 = 'ZZZ'
    var_18 = 'Last Currency'
    var_19 = var_1.codenames
    var_20 = 'NON_EXISTENT'
    var_21 = var_1[var_20]



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test that Currency instances are immutable (frozen dataclass).'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.001'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test the __hash__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'United States Dollar'
    var_8 = 3
    var_9 = 'First'
    var_10 = 'Second'
    var_11 = 'Third'
    var_12 = 'ZZZ'
    var_13 = 'Some weird currency'
    var_14 = -1



# Parsed testcases at query #29
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyLookupError constructor.'
    var_1 = 'XYZ'
    var_2 = module_0.CurrencyLookupError(var_1)
    var_3 = str(var_2)

def test_case_0():
    var_0 = 'Test CurrencyLookupError with different currency codes.'
    var_1 = 'ABC'
    var_2 = 'USD'
    var_3 = 'EUR'
    var_4 = 'GBP'
    var_5 = [var_1, var_2, var_3, var_4]

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that CurrencyLookupError can be raised and caught.'
    var_1 = 'XYZ'
    var_2 = module_0.CurrencyLookupError(var_1)
    var_3 = str(var_2)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that CurrencyLookupError is a LookupError.'
    var_1 = 'TEST'
    var_2 = module_0.CurrencyLookupError(var_1)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test the __eq__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'United States Dollar'
    var_5 = 3
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'ZZZ'
    var_12 = 'Some weird currency'
    var_13 = -1



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __enter__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = callable(var_2)
    var_4 = '__enter__ should return a callable'
    var_5 = None
    var_6 = var_1.__exit__(var_5, var_5, var_5)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __enter__ method works correctly in a with statement.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'Context manager should return a callable'
    var_3 = 'TST'
    var_4 = 'Test Currency'
    var_5 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __enter__ sets the context open flag correctly.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = None
    var_4 = var_1.__exit__(var_3, var_3, var_3)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test the __eq__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'UX Dollars'
    var_5 = 3
    var_6 = 'EUR'
    var_7 = 'Euros'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test the has method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 'JPY'
    var_4 = 'XXX'
    var_5 = 'NON'
    var_6 = 'INVALID'
    var_7 = ''
    var_8 = 'usd'



# Parsed testcases at query #4
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __contains__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test the __gt__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'United States Dollars'
    var_12 = 'BTC'
    var_13 = 'Bitcoin'
    var_14 = 8
    var_15 = 'ZZZ'
    var_16 = 'Some weird currency'
    var_17 = -1



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test the has method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 'JPY'
    var_4 = 'XXX'
    var_5 = 'ZZZ'
    var_6 = 'INVALID'
    var_7 = ''
    var_8 = 'usd'
    var_9 = 'GBP'
    var_10 = 'CHF'
    var_11 = 'AUD'



# Parsed testcases at query #7
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the CurrencyRegistry constructor and singleton behavior.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = len(var_1)
    assert var_3 == 0
    var_4 = var_1._CurrencyRegistry__registry
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = var_1._CurrencyRegistry__currencies
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = var_1._CurrencyRegistry__codes
    var_9 = len(var_8)
    assert var_9 == 0
    var_10 = var_1._CurrencyRegistry__codenames
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = 'USD'
    var_13 = var_1.has(var_12)
    var_14 = 'EUR'
    var_15 = var_1.has(var_14)
    var_16 = var_1.get(var_12)
    assert var_16 is None
    var_17 = var_1.get(var_14)
    assert var_17 is None
    var_18 = 'USD'
    var_19 = var_1[var_18]



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test Currency class constructor and functionality.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.01'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1
    var_11 = '1.005'
    var_12 = '1.00'
    var_13 = '1.015'
    var_14 = '1.02'
    var_15 = '0.5'
    var_16 = '0'
    var_17 = '1.5'
    var_18 = '2'
    var_19 = 'UX Dollars'
    var_20 = 'US1'
    var_21 = 'US Dollars'
    var_22 = 2
    var_23 = 'usd'
    var_24 = 'US Dollars'
    var_25 = 2
    var_26 = 123
    var_27 = 'US Dollars'
    var_28 = 2
    var_29 = 'USD'
    var_30 = ''
    var_31 = 2
    var_32 = 'USD'
    var_33 = ' US Dollars'
    var_34 = 2
    var_35 = 'USD'
    var_36 = 'US Dollars '
    var_37 = 2
    var_38 = 'USD'
    var_39 = 123
    var_40 = 2
    var_41 = 'USD'
    var_42 = 'US Dollars'
    var_43 = 2.5
    var_44 = 'USD'
    var_45 = 'US Dollars'
    var_46 = -2
    var_47 = 'USD'
    var_48 = 'US Dollars'
    var_49 = 2
    var_50 = 'MONEY'
    var_51 = 'AU'
    var_52 = 'Gold'
    var_53 = 'ALT'
    var_54 = 'Alternative'



# Parsed testcases at query #9
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __enter__ returns the __register method and sets context as open.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = callable(var_2)
    var_4 = 'TST'
    var_5 = 'Test Currency'
    var_6 = 2
    var_7 = None
    var_8 = var_1.__exit__(var_7, var_7, var_7)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test the __le__ (less than or equal to) comparison of Currency objects.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'GBP'
    var_7 = 'British Pound'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test the __lt__ method of Currency class for ordering.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8
    var_14 = 'XAU'
    var_15 = 'Gold'
    var_16 = 4
    var_17 = 'American Dollars'



# Parsed testcases at query #12
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __enter__ returns the __register method and opens the context.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = callable(var_2)
    var_4 = 'TST'
    var_5 = 'Test Currency'
    var_6 = 2
    var_7 = None
    var_8 = var_1.__exit__(var_7, var_7, var_7)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test the get method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'XXX'
    var_3 = 'EUR'
    var_4 = 'GBP'
    var_5 = 'NON_EXISTENT'
    var_6 = 'JPY'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test the quantize method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '1.005'
    var_5 = '1.00'
    var_6 = '1.015'
    var_7 = '1.02'
    var_8 = '1.234'
    var_9 = '1.23'
    var_10 = '1.235'
    var_11 = '1.24'
    var_12 = '0'
    var_13 = '0.00'
    var_14 = 'JPY'
    var_15 = 'Japanese Yen'
    var_16 = 0
    var_17 = '0.5'
    var_18 = '1.5'
    var_19 = '2'
    var_20 = '2.4'
    var_21 = '2.5'
    var_22 = '100'
    var_23 = 'ZZZ'
    var_24 = 'Some weird currency'
    var_25 = -1
    var_26 = '1.0000000000005'
    var_27 = '1.000000000000'
    var_28 = '1.0000000000015'
    var_29 = '1.000000000002'
    var_30 = '1.123456789'
    var_31 = 'BTC'
    var_32 = 'Bitcoin'
    var_33 = 8
    var_34 = '0.123456789'
    var_35 = '0.12345679'
    var_36 = '1'
    var_37 = '1.00000000'
    var_38 = 'GBP'
    var_39 = 'British Pounds'
    var_40 = '10.126'
    var_41 = '10.13'
    var_42 = '10.124'
    var_43 = '10.12'
    var_44 = 'AUD'
    var_45 = 'Australian Dollars'
    var_46 = '0.999'
    var_47 = '0.001'
    var_48 = 'EUR'
    var_49 = 'Euros'
    var_50 = '-1.234'
    var_51 = '-1.23'
    var_52 = '-1.235'
    var_53 = '-1.24'



# Parsed testcases at query #15
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that CurrencyRegistry.__new__ creates and returns a singleton instance.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = module_0.CurrencyRegistry()
    var_4 = id(var_1)
    var_5 = id(var_2)
    var_6 = id(var_3)



# Parsed testcases at query #16
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry.__getitem__ method'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = var_1[var_2]
    var_4 = var_1[var_2]
    var_5 = 'JPY'
    var_6 = var_1[var_5]
    var_7 = 'NON-EXISTING'
    var_8 = var_1[var_7]
    var_9 = 'XYZ'
    var_10 = var_1[var_9]
    var_11 = 'ABC'
    var_12 = var_1[var_11]
    var_13 = ''
    var_14 = var_1[var_13]
    var_15 = 'usd'
    var_16 = var_1[var_15]



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test the __ge__ (greater than or equal) comparison operator for Currency.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'GBP'
    var_7 = 'British Pound'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test the __repr__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'XAU'
    var_5 = 'Gold'
    var_6 = 4
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0
    var_13 = 'ALT'
    var_14 = 'Alternative Currency'
    var_15 = 3



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test Currency class constructor and related functionality.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = '1.005'
    var_11 = '1.00'
    var_12 = '1.015'
    var_13 = '1.02'
    var_14 = '0.5'
    var_15 = '0'
    var_16 = '1.5'
    var_17 = '2'
    var_18 = '1.0000000000005'
    var_19 = '1.000000000000'
    var_20 = '1.0000000000015'
    var_21 = '1.000000000002'
    var_22 = 'UX Dollars'
    var_23 = 'US1'
    var_24 = 'US Dollars'
    var_25 = 2
    var_26 = 'Usd'
    var_27 = 'US Dollars'
    var_28 = 2
    var_29 = 123
    var_30 = 'US Dollars'
    var_31 = 2
    var_32 = 'USD'
    var_33 = ''
    var_34 = 2
    var_35 = 'USD'
    var_36 = ' US Dollars'
    var_37 = 2
    var_38 = 'USD'
    var_39 = 'US Dollars '
    var_40 = 2
    var_41 = 'USD'
    var_42 = 123
    var_43 = 2
    var_44 = 'USD'
    var_45 = 'US Dollars'
    var_46 = 2.5
    var_47 = 'USD'
    var_48 = 'US Dollars'
    var_49 = -2
    var_50 = 'USD'
    var_51 = 'US Dollars'
    var_52 = 2
    var_53 = 'MONEY'
    var_54 = 'GOLD'
    var_55 = 'Gold'
    var_56 = 5
    var_57 = 'BTC'
    var_58 = 'Bitcoin'
    var_59 = -1
    var_60 = 'ALT'
    var_61 = 'Alternative Currency'
    var_62 = 3



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test that Currency instances are immutable (frozen dataclass).'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.001'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test the __eq__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'UX Dollars'
    var_5 = 3
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0



# Parsed testcases at query #22
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __exit__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'TST'
    var_3 = 'Test Currency 1'
    var_4 = 2
    var_5 = 'TSA'
    var_6 = 'Test Currency 2'
    var_7 = 'TSZ'
    var_8 = 'Test Currency 3'
    var_9 = var_1.codes
    var_10 = sorted(var_9)
    var_11 = var_1.all
    var_12 = len(var_11)
    assert var_12 == 3
    var_13 = 'NEW'
    var_14 = 'New Currency'
    var_15 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __exit__ properly closes context even when exception occurs.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'EXC'
    var_3 = 'Exception Currency'
    var_4 = 2
    var_5 = 'Test exception'
    var_6 = ValueError(var_5)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __exit__ properly sorts registry items.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'ZZZ'
    var_3 = 'Last'
    var_4 = 2
    var_5 = 'AAA'
    var_6 = 'First'
    var_7 = 'MMM'
    var_8 = 'Middle'
    var_9 = var_1.codes



# Parsed testcases at query #23
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __enter__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = callable(var_2)
    var_4 = '__enter__ should return a callable'
    var_5 = None
    var_6 = var_1.__exit__(var_5, var_5, var_5)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test CurrencyRegistry.__getitem__ method'
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 'NON-EXISTING'
    var_4 = 'XYZ123'
    var_5 = 'usd'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test the quantize method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '1.005'
    var_5 = '1.00'
    var_6 = '1.015'
    var_7 = '1.02'
    var_8 = '1.234'
    var_9 = '1.23'
    var_10 = '1.235'
    var_11 = '1.24'
    var_12 = '0'
    var_13 = '0.00'
    var_14 = '100.999'
    var_15 = '101.00'
    var_16 = 'JPY'
    var_17 = 'Japanese Yen'
    var_18 = 0
    var_19 = '0.5'
    var_20 = '1.5'
    var_21 = '2'
    var_22 = '1.4'
    var_23 = '1'
    var_24 = '100.6'
    var_25 = '101'
    var_26 = 'ZZZ'
    var_27 = 'Some weird currency'
    var_28 = -1
    var_29 = '1.0000000000005'
    var_30 = '1.000000000000'
    var_31 = '1.0000000000015'
    var_32 = '1.000000000002'
    var_33 = '0.123456789012345'
    var_34 = '0.123456789012'
    var_35 = 'BTC'
    var_36 = 'Bitcoin'
    var_37 = 8
    var_38 = '0.123456789'
    var_39 = '0.12345679'
    var_40 = '0.000000001'
    var_41 = '0.00000000'
    var_42 = '21000000'
    var_43 = '21000000.00000000'
    var_44 = 'GBP'
    var_45 = 'British Pounds'
    var_46 = '50.125'
    var_47 = '50.12'
    var_48 = '50.135'
    var_49 = '50.14'
    var_50 = '0.01'
    var_51 = 'KWD'
    var_52 = 'Kuwaiti Dinar'
    var_53 = 3
    var_54 = '1.2345'
    var_55 = '1.2344'
    var_56 = '0.0005'
    var_57 = '0.000'
    var_58 = '-1.005'
    var_59 = '-1.00'
    var_60 = '-1.015'
    var_61 = '-1.02'
    var_62 = '-1.5'
    var_63 = '-2'
    var_64 = '999999999.999'
    var_65 = '1000000000.00'
    var_66 = '1000000000'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Test the __lt__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'GBP'
    var_7 = 'British Pound'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8
    var_14 = 'XAU'
    var_15 = 'Gold'
    var_16 = 3



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test that Currency instances are immutable (frozen dataclass).'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.001'



# Parsed testcases at query #28
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the CurrencyRegistry constructor and singleton behavior.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = len(var_1)
    assert var_3 == 0
    var_4 = var_1._CurrencyRegistry__registry
    var_5 = var_1._CurrencyRegistry__registry
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = var_1._CurrencyRegistry__currencies
    var_8 = var_1._CurrencyRegistry__currencies
    var_9 = len(var_8)
    assert var_9 == 0
    var_10 = var_1._CurrencyRegistry__codes
    var_11 = var_1._CurrencyRegistry__codes
    var_12 = len(var_11)
    assert var_12 == 0
    var_13 = var_1._CurrencyRegistry__codenames
    var_14 = var_1._CurrencyRegistry__codenames
    var_15 = len(var_14)
    assert var_15 == 0



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test the get method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'XXX'
    var_3 = 'NON-EXISTING'
    var_4 = 'JPY'
    var_5 = 'EUR'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test the __eq__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'United States Dollar'
    var_5 = 3
    var_6 = 'EUR'
    var_7 = 'Euro'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test that Currency instances are immutable and __delattr__ raises an error.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __enter__ returns the __register method and opens the context.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = callable(var_2)
    var_4 = 'TST'
    var_5 = 'Test Currency'
    var_6 = 2
    var_7 = None
    var_8 = var_1.__exit__(var_7, var_7, var_7)



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'Test the __repr__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'XAU'
    var_5 = 'Gold'
    var_6 = 4
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = -1
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'United States Dollar'
    var_4 = 3
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = 'GBP'
    var_11 = 'British Pound'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test the has method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 'JPY'
    var_4 = 'XXX'
    var_5 = 'NON'
    var_6 = 'INVALID'
    var_7 = ''
    var_8 = 'usd'



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test the quantize method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '1.005'
    var_5 = '1.00'
    var_6 = '1.015'
    var_7 = '1.02'
    var_8 = '1.234'
    var_9 = '1.23'
    var_10 = '1.235'
    var_11 = '1.24'
    var_12 = '0'
    var_13 = '0.00'
    var_14 = '100.5'
    var_15 = '100.50'
    var_16 = 'JPY'
    var_17 = 'Japanese Yen'
    var_18 = 0
    var_19 = '0.5'
    var_20 = '1.5'
    var_21 = '2'
    var_22 = '2.4'
    var_23 = '2.5'
    var_24 = '2.6'
    var_25 = '3'
    var_26 = '100'
    var_27 = 'ZZZ'
    var_28 = 'Some weird currency'
    var_29 = -1
    var_30 = '1.0000000000005'
    var_31 = '1.000000000000'
    var_32 = '1.0000000000015'
    var_33 = '1.000000000002'
    var_34 = '0.123456789'
    var_35 = '1'
    var_36 = 'GBP'
    var_37 = 'British Pound'
    var_38 = '10.125'
    var_39 = '10.12'
    var_40 = 'BHD'
    var_41 = 'Bahraini Dinar'
    var_42 = 3
    var_43 = '1.0005'
    var_44 = '1.000'
    var_45 = '1.0015'
    var_46 = '1.002'
    var_47 = '-1.005'
    var_48 = '-1.00'
    var_49 = '-1.5'
    var_50 = '-2'
    var_51 = '0.001'
    var_52 = '0.009'
    var_53 = '0.01'



# Parsed testcases at query #4
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __contains__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #34
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __enter__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_1)
    var_3 = 'TST'
    var_4 = 'Test Currency'
    var_5 = 2



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test the __gt__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test the quantize method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '1.005'
    var_5 = '1.00'
    var_6 = '1.015'
    var_7 = '1.02'
    var_8 = '10.999'
    var_9 = '11.00'
    var_10 = '0.001'
    var_11 = '0.00'
    var_12 = '0.005'
    var_13 = 'JPY'
    var_14 = 'Japanese Yen'
    var_15 = 0
    var_16 = '0.5'
    var_17 = '0'
    var_18 = '1.5'
    var_19 = '2'
    var_20 = '10.4'
    var_21 = '10'
    var_22 = '10.6'
    var_23 = '11'
    var_24 = 'ZZZ'
    var_25 = 'Some weird currency'
    var_26 = -1
    var_27 = '1.0000000000005'
    var_28 = '1.000000000000'
    var_29 = '1.0000000000015'
    var_30 = '1.000000000002'
    var_31 = '1.123456789'
    var_32 = 'XXX'
    var_33 = 'Test Currency'
    var_34 = 3
    var_35 = '1.0005'
    var_36 = '1.000'
    var_37 = '1.0015'
    var_38 = '1.002'
    var_39 = '100.9999'
    var_40 = '101.000'
    var_41 = '-1.234'
    var_42 = '-1.23'
    var_43 = '-1.6'
    var_44 = '-2'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test the __gt__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'GBP'
    var_7 = 'British Pound'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = -1
    var_14 = 'XAU'
    var_15 = 'Gold'



# Parsed testcases at query #7
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyLookupError constructor.'
    var_1 = 'XYZ'
    var_2 = module_0.CurrencyLookupError(var_1)
    var_3 = str(var_2)
    var_4 = 'ABC'
    var_5 = module_0.CurrencyLookupError(var_4)
    var_6 = str(var_5)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '\n    Test the __repr__ method of Currency class.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'XAU'
    var_5 = 'Gold'
    var_6 = 4
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = -1
    var_10 = 'ALT'
    var_11 = 'Alternative Currency'
    var_12 = 0
    var_13 = 'JPY'
    var_14 = 'Japanese Yen'



# Parsed testcases at query #9
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that CurrencyRegistry.__new__ returns a singleton instance.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = module_0.CurrencyRegistry()
    var_4 = id(var_1)
    var_5 = id(var_2)
    var_6 = id(var_3)



# Parsed testcases at query #10
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Test that __enter__ returns the __register method and sets context as open.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = callable(var_2)
    var_4 = None
    var_5 = var_1.__exit__(var_4, var_4, var_4)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test the __le__ (less than or equal) method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = -1
    var_14 = 'American Dollars'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test the has method of CurrencyRegistry class.'
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 'JPY'
    var_4 = 'XXX'
    var_5 = 'NON'
    var_6 = 'INVALID'
    var_7 = ''
    var_8 = 'usd'
    var_9 = 'Usd'



# Parsed testcases at query #13
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"
    var_3 = 'ABC'
    var_4 = module_0.CurrencyLookupError(var_3)
    var_5 = str(var_4)
    assert var_5 == "Currency identified by code 'ABC' does not exist"
    var_6 = 'X'
    var_7 = module_0.CurrencyLookupError(var_6)
    var_8 = str(var_7)
    assert var_8 == "Currency identified by code 'X' does not exist"
    var_9 = 'EUR'
    var_10 = module_0.CurrencyLookupError(var_9)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'Test CurrencyRegistry.get method'
    var_1 = 'USD'
    var_2 = 'XXX'
    var_3 = 'JPY'
    var_4 = 'NON-EXISTING'
    var_5 = None
    var_6 = 'EUR'
    var_7 = 'GBP'
    var_8 = 'INVALID'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test that Currency instances are immutable (frozen dataclass).'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.001'



# Parsed testcases at query #16
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry.__getitem__ method'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = var_1[var_2]
    var_4 = 'JPY'
    var_5 = var_1[var_4]
    var_6 = 'NON-EXISTING'
    var_7 = var_1[var_6]
    var_8 = 'XYZ'
    var_9 = var_1[var_8]
    var_10 = ''
    var_11 = var_1[var_10]
    var_12 = var_1[var_11]
    var_13 = var_1[var_11]
    var_14 = hash(var_12)
    var_15 = hash(var_13)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test Currency.__eq__ method'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'United States Dollars'
    var_5 = 3
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'ZZZ'
    var_12 = 'Some weird currency'
    var_13 = -1



# Parsed testcases at query #18
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __enter__ returns the __register method and opens the context.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = callable(var_2)
    var_4 = 'TST'
    var_5 = 'Test Currency'
    var_6 = 2
    var_7 = None
    var_8 = var_1.__exit__(var_7, var_7, var_7)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'Test the __repr__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = -1
    var_10 = 'XAU'
    var_11 = 'Gold'
    var_12 = 4



# Parsed testcases at query #20
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry.__exit__ method'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'GBP'
    var_8 = 'British Pound'
    var_9 = len(var_1)
    assert var_9 == 3
    var_10 = (var_5, var_6)
    var_11 = (var_7, var_8)
    var_12 = (var_2, var_3)
    var_13 = [var_10, var_11, var_12]
    var_14 = 'JPY'
    var_15 = 'Japanese Yen'
    var_16 = 0

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry.__exit__ handles exceptions properly'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'Test exception'
    var_6 = ValueError(var_5)
    var_7 = len(var_1)
    assert var_7 == 1

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry.__exit__ sorts currencies correctly'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'JPY'
    var_3 = 'Japanese Yen'
    var_4 = 0
    var_5 = 'AED'
    var_6 = 'UAE Dirham'
    var_7 = 2
    var_8 = 'ZAR'
    var_9 = 'South African Rand'



# Parsed testcases at query #36
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __enter__ returns the __register method and sets context open flag.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = callable(var_2)
    var_4 = 'TST'
    var_5 = 'Test Currency'
    var_6 = 2
    var_7 = None
    var_8 = var_1.__exit__(var_7, var_7, var_7)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __enter__ works correctly with context manager protocol.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'CTX'
    var_3 = 'Context Test'
    var_4 = 2
    var_5 = var_1.has(var_2)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __enter__ returns the __register method specifically.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = 'RGM'
    var_4 = 'Register Method Test'
    var_5 = 2
    var_6 = var_1.has(var_3)
    var_7 = None
    var_8 = var_1.__exit__(var_7, var_7, var_7)



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'Test the __lt__ method of Currency class for ordering.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = -1



# Parsed testcases at query #38
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __len__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_1)
    var_3 = len(var_1)
    var_4 = var_1.all
    var_5 = len(var_4)
    var_6 = len(var_1)
    var_7 = var_1.codes
    var_8 = len(var_7)
    var_9 = len(var_1)
    var_10 = var_1.codenames
    var_11 = len(var_10)
    var_12 = len(var_1)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test that Currency instances are immutable (frozen dataclass).'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.001'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = '\n    Test that Currency objects are immutable and cannot have attributes deleted.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test the __repr__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = -1



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test the __gt__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8
    var_14 = False
    var_15 = 'USDA'
    var_16 = 'US Dollars A'
    var_17 = None
    var_18 = 'CHF'
    var_19 = 'Swiss Franc'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test the __ge__ (greater than or equal) comparison for Currency objects.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'GBP'
    var_7 = 'British Pound'
    var_8 = 'United States Dollar'
    var_9 = 'JPY'
    var_10 = 'Japanese Yen'
    var_11 = 0
    var_12 = 'CAD'
    var_13 = 'Canadian Dollar'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Test the __gt__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'GBP'
    var_7 = 'British Pound'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = -1



# Parsed testcases at query #27
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __exit__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = var_1.codes
    var_11 = sorted(var_10)
    var_12 = var_1.all
    var_13 = len(var_12)
    var_14 = 1
    var_15 = var_13 - var_14
    var_16 = range(var_15)
    var_17 = lambda x: x.code
    var_18 = sorted(var_12, key=var_17)
    var_19 = [c.code for c in var_18]
    var_20 = lambda x: x.code
    var_21 = sorted(var_12, key=var_20)
    var_22 = [(c.code, c.name) for c in var_21]
    var_23 = var_1.get(var_5)
    var_24 = len(var_1)
    assert var_24 == 3

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __exit__ closes context even when exception occurs.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'GBP'
    var_3 = 'British Pound'
    var_4 = 2
    var_5 = 'Test exception'
    var_6 = ValueError(var_5)

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __exit__ properly sorts currencies by code.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'ZZZ'
    var_3 = 'Last Currency'
    var_4 = 2
    var_5 = 'AAA'
    var_6 = 'First Currency'
    var_7 = 'MMM'
    var_8 = 'Middle Currency'
    var_9 = var_1.codes
    var_10 = var_1.all

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __exit__ updates all internal buffers correctly.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'BTC'
    var_3 = 'Bitcoin'
    var_4 = 8
    var_5 = 'ETH'
    var_6 = 'Ethereum'
    var_7 = 18
    var_8 = var_1.all
    var_9 = len(var_8)
    var_10 = var_1.codes
    var_11 = len(var_10)
    var_12 = var_1.all
    var_13 = len(var_12)
    var_14 = var_1.codenames
    var_15 = len(var_14)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test the __gt__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'ZZZ'
    var_12 = 'Weird Currency'
    var_13 = -1
    var_14 = 'AAA'
    var_15 = 'Normal Currency'



# Parsed testcases at query #29
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry constructor and singleton behavior.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = len(var_1)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = var_1.has(var_4)
    var_6 = var_1.get(var_4)
    assert var_6 is None
    var_7 = None
    var_8 = var_1.get(var_4, var_7)
    assert var_8 is None
    var_9 = 'USD'
    var_10 = var_1[var_9]
    var_11 = 'USD'
    var_12 = 'US Dollar'
    var_13 = 2
    var_14 = len(var_1)
    assert var_14 == 1
    var_15 = var_1.has(var_12)
    var_16 = var_1.get(var_12)
    var_17 = var_1.all
    var_18 = len(var_17)
    assert var_18 == 1
    var_19 = 'USD'
    var_20 = 'US Dollar'
    var_21 = 2
    var_22 = str(var_19)
    var_23 = 'JPY'
    var_24 = 'Japanese Yen'
    var_25 = 0
    var_26 = 'EUR'
    var_27 = 'Euro'
    var_28 = 2
    var_29 = len(var_1)
    assert var_29 == 3
    var_30 = 'GBP'
    var_31 = 'British Pound'
    var_32 = 2
    var_33 = 'NON-EXISTING'
    var_34 = 'CHF'
    var_35 = 'Swiss Franc'
    var_36 = 2



# Parsed testcases at query #30
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that __enter__ returns the __register method and sets context as open.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = None



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test the __hash__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'UX Dollars'
    var_5 = 3
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'ZZZ'
    var_10 = 'Some weird currency'
    var_11 = -1
    var_12 = -1
    var_13 = 'first'
    var_14 = 'second'



# Parsed testcases at query #32
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry constructor and singleton behavior.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = len(var_1)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = var_1.has(var_4)
    assert var_5 is False
    var_6 = 'EUR'
    var_7 = var_1.has(var_6)
    assert var_7 is False
    var_8 = var_1.get(var_4)
    assert var_8 is None
    var_9 = var_1.get(var_6)
    assert var_9 is None
    var_10 = 'USD'
    var_11 = var_1[var_10]
    var_12 = 'US Dollar'
    var_13 = 2
    var_14 = 'Euro'
    var_15 = len(var_1)
    assert var_15 == 2
    var_16 = var_1.has(var_4)
    assert var_16 is True
    var_17 = var_1.has(var_6)
    assert var_17 is True
    var_18 = var_1.all
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_1.get(var_4)
    var_21 = var_1.get(var_6)
    var_22 = 'GBP'
    var_23 = var_1.get(var_22)
    assert var_23 is None
    var_24 = 'GBP'
    var_25 = 'British Pound'
    var_26 = 2



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Unit tests for Currency class constructor and methods.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = '1.005'
    var_11 = '1.00'
    var_12 = '1.015'
    var_13 = '1.02'
    var_14 = '0.5'
    var_15 = '0'
    var_16 = '1.5'
    var_17 = '2'
    var_18 = '1.0000000000005'
    var_19 = '1.000000000000'
    var_20 = '1.0000000000015'
    var_21 = '1.000000000002'
    var_22 = 'UX Dollars'
    var_23 = 'US1'
    var_24 = 'US Dollars'
    var_25 = 2
    var_26 = 'usd'
    var_27 = 'US Dollars'
    var_28 = 2
    var_29 = 123
    var_30 = 'US Dollars'
    var_31 = 2
    var_32 = 'USD'
    var_33 = ''
    var_34 = 2
    var_35 = 'USD'
    var_36 = ' US Dollars'
    var_37 = 2
    var_38 = 'USD'
    var_39 = 'US Dollars '
    var_40 = 2
    var_41 = 'USD'
    var_42 = 123
    var_43 = 2
    var_44 = 'USD'
    var_45 = 'US Dollars'
    var_46 = 2.5
    var_47 = 'USD'
    var_48 = 'US Dollars'
    var_49 = -2
    var_50 = 'USD'
    var_51 = 'US Dollars'
    var_52 = 2
    var_53 = 'MONEY'
    var_54 = 'GLD'
    var_55 = 'Gold'
    var_56 = 'ALT'
    var_57 = 'Alternative'
    var_58 = 'AAA'
    var_59 = 'Currency A'
    var_60 = 'BBB'
    var_61 = 'Currency B'



# Parsed testcases at query #34
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __enter__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'TST'
    var_3 = 'Test Currency'
    var_4 = 2
    var_5 = len(var_1)
    var_6 = len(var_1)



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'Test the __gt__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test the __getitem__ method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 'NON-EXISTING'
    var_4 = 'XYZ'
    var_5 = 'INVALID'
    var_6 = 'usd'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test the __eq__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'United States Dollar'
    var_5 = 3
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test the __eq__ method of Currency class'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'Different Name'
    var_5 = 3
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'GBP'
    var_12 = 'British Pound'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test the __eq__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'United States Dollar'
    var_5 = 3
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test the __eq__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'UX Dollars'
    var_5 = 3
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'ZZZ'
    var_12 = 'Some weird currency'
    var_13 = -1
    var_14 = -1



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test CurrencyRegistry.__getitem__ method'
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 'NON-EXISTING'
    var_4 = 'XYZ'
    var_5 = 'JPY'
    var_6 = 'GBP'



# Parsed testcases at query #7
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry.__getitem__ method'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = var_1[var_2]
    var_4 = 'JPY'
    var_5 = var_1[var_4]
    var_6 = 'NON-EXISTING'
    var_7 = var_1[var_6]
    var_8 = 'XYZ'
    var_9 = var_1[var_8]
    var_10 = 'usd'
    var_11 = var_1[var_10]
    var_12 = ''
    var_13 = var_1[var_12]
    var_14 = var_1[var_13]
    var_15 = var_1[var_13]



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test CurrencyRegistry.__getitem__ method'
    var_1 = 'USD'
    var_2 = 'JPY'
    var_3 = 'NON-EXISTING'
    var_4 = 'XYZ'
    var_5 = ''
    var_6 = 'INVALID_CODE'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'Test the __eq__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'United States Dollars'
    var_5 = 3
    var_6 = 'EUR'
    var_7 = 'Euros'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test CurrencyRegistry.__getitem__ method.'
    var_1 = 'USD'
    var_2 = 'JPY'
    var_3 = 'NON-EXISTING'
    var_4 = 'XYZ'
    var_5 = 'EUR'
    var_6 = 'GBP'
    var_7 = 'usd'
    var_8 = 'Usd'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'Test the __repr__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'XAU'
    var_5 = 'Gold'
    var_6 = 4
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = -1
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0



# Parsed testcases at query #12
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __len__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_1)
    var_3 = len(var_1)
    var_4 = len(var_1)
    var_5 = var_1.all
    var_6 = len(var_5)
    var_7 = len(var_1)
    var_8 = var_1.codes
    var_9 = len(var_8)
    var_10 = len(var_1)
    var_11 = var_1.codenames
    var_12 = len(var_11)



# Parsed testcases at query #13
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that CurrencyRegistry.__new__ creates and returns a singleton instance.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = module_0.CurrencyRegistry()
    var_4 = id(var_1)
    var_5 = id(var_2)
    var_6 = id(var_2)
    var_7 = id(var_3)



# Parsed testcases at query #14
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyLookupError constructor and properties.'
    var_1 = 'XYZ'
    var_2 = module_0.CurrencyLookupError(var_1)
    var_3 = str(var_2)
    assert var_3 == "Currency identified by code 'XYZ' does not exist"
    var_4 = 'ABC'
    var_5 = module_0.CurrencyLookupError(var_4)
    var_6 = str(var_5)
    assert var_6 == "Currency identified by code 'ABC' does not exist"
    var_7 = 'INVALID'
    var_8 = module_0.CurrencyLookupError(var_7)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test the get method of CurrencyRegistry class.'
    var_1 = 'USD'
    var_2 = 'XXX'
    var_3 = 'EUR'
    var_4 = 'NON-EXISTING'
    var_5 = 'INVALID'
    var_6 = None



# Parsed testcases at query #16
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyLookupError constructor and properties.'
    var_1 = 'XYZ'
    var_2 = module_0.CurrencyLookupError(var_1)
    var_3 = str(var_2)
    assert var_3 == "Currency identified by code 'XYZ' does not exist"
    var_4 = 'ABC'
    var_5 = module_0.CurrencyLookupError(var_4)
    var_6 = str(var_5)
    assert var_6 == "Currency identified by code 'ABC' does not exist"
    var_7 = '123'
    var_8 = module_0.CurrencyLookupError(var_7)
    var_9 = str(var_8)
    assert var_9 == "Currency identified by code '123' does not exist"
    var_10 = 'USD'
    var_11 = module_0.CurrencyLookupError(var_10)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test Currency class constructor and methods'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '1.005'
    var_5 = '1.00'
    var_6 = '1.015'
    var_7 = '1.02'
    var_8 = '99.999'
    var_9 = '100.00'
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0
    var_13 = '0.5'
    var_14 = '0'
    var_15 = '1.5'
    var_16 = '2'
    var_17 = 'ZZZ'
    var_18 = 'Some weird currency'
    var_19 = -1
    var_20 = '1.0000000000005'
    var_21 = '1.000000000000'
    var_22 = '1.0000000000015'
    var_23 = '1.000000000002'
    var_24 = 'UX Dollars'
    var_25 = 'BTC'
    var_26 = 'Bitcoin'
    var_27 = 8
    var_28 = 'XAU'
    var_29 = 'Gold'
    var_30 = 4
    var_31 = 'ALT'
    var_32 = 'Alternative'
    var_33 = 3
    var_34 = 'US1'
    var_35 = 'US Dollars'
    var_36 = 2
    var_37 = 'usd'
    var_38 = 'US Dollars'
    var_39 = 2
    var_40 = 123
    var_41 = 'US Dollars'
    var_42 = 2
    var_43 = 'USD'
    var_44 = ''
    var_45 = 2
    var_46 = 'USD'
    var_47 = ' US Dollars'
    var_48 = 2
    var_49 = 'USD'
    var_50 = 'US Dollars '
    var_51 = 2
    var_52 = 'USD'
    var_53 = 123
    var_54 = 2
    var_55 = 'USD'
    var_56 = 'US Dollars'
    var_57 = -2
    var_58 = 'USD'
    var_59 = 'US Dollars'
    var_60 = 2.5
    var_61 = 'USD'
    var_62 = 'US Dollars'
    var_63 = 2
    var_64 = 'MONEY'
    var_65 = 'EUR'
    var_66 = 'Euro'
    var_67 = 'GBP'
    var_68 = 'British Pound'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test that Currency instances are immutable (frozen dataclass).'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.1'



# Parsed testcases at query #19
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that CurrencyRegistry.__new__ returns a singleton instance.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = module_0.CurrencyRegistry()
    var_4 = id(var_1)
    var_5 = id(var_2)
    var_6 = id(var_3)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'Test the __lt__ method of Currency class for ordering.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'United States Dollar'
    var_12 = 3
    var_13 = 'BTC'
    var_14 = 'Bitcoin'
    var_15 = -1
    var_16 = 'XAU'
    var_17 = 'Gold'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test the __le__ (less than or equal) comparison method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test __getitem__ method of CurrencyRegistry class.'
    var_1 = 'USD'
    var_2 = 'NON-EXISTING'
    var_3 = 'JPY'
    var_4 = 'XYZ'
    var_5 = ''
    var_6 = 'usd'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test the has method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'XXX'
    var_3 = 'EUR'
    var_4 = 'GBP'
    var_5 = 'JPY'
    var_6 = 'ZZZ'
    var_7 = 'ABC'
    var_8 = 'XYZ'
    var_9 = 'usd'
    var_10 = 'eur'
    var_11 = ''
    var_12 = 'U'
    var_13 = 'US1'
    var_14 = '123'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test the quantize method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '1.005'
    var_5 = '1.00'
    var_6 = '1.015'
    var_7 = '1.02'
    var_8 = '1.234'
    var_9 = '1.23'
    var_10 = '1.235'
    var_11 = '1.24'
    var_12 = '0'
    var_13 = '0.00'
    var_14 = '99.999'
    var_15 = '100.00'
    var_16 = 'JPY'
    var_17 = 'Japanese Yen'
    var_18 = 0
    var_19 = '0.5'
    var_20 = '1.5'
    var_21 = '2'
    var_22 = '1.4'
    var_23 = '1'
    var_24 = '2.5'
    var_25 = '100'
    var_26 = 'ZZZ'
    var_27 = 'Some weird currency'
    var_28 = -1
    var_29 = '1.0000000000005'
    var_30 = '1.000000000000'
    var_31 = '1.0000000000015'
    var_32 = '1.000000000002'
    var_33 = '1.123456789'
    var_34 = 'BTC'
    var_35 = 'Bitcoin'
    var_36 = 8
    var_37 = '1.12345679'
    var_38 = '0.00000001'
    var_39 = '0.000000001'
    var_40 = '0.00000000'
    var_41 = 'GBP'
    var_42 = 'British Pound'
    var_43 = '10.125'
    var_44 = '10.12'
    var_45 = '10.135'
    var_46 = '10.14'
    var_47 = 'EUR'
    var_48 = 'Euro'
    var_49 = '0.001'
    var_50 = '0.005'
    var_51 = '0.015'
    var_52 = '0.02'
    var_53 = '-1.005'
    var_54 = '-1.00'
    var_55 = '-1.015'
    var_56 = '-1.02'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test the __le__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'AED'
    var_12 = 'UAE Dirham'
    var_13 = 'BTC'
    var_14 = 'Bitcoin'
    var_15 = 8



# Parsed testcases at query #26
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that CurrencyRegistry.__new__ creates and returns singleton instance.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = module_0.CurrencyRegistry()



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test the __lt__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = -1



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test the has method of CurrencyRegistry.'
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 'JPY'
    var_4 = 'XXX'
    var_5 = 'ZZZ'
    var_6 = 'INVALID'
    var_7 = ''
    var_8 = 'usd'
    var_9 = 'eur'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test the get method of CurrencyRegistry class.'
    var_1 = 'USD'
    var_2 = 'NON_EXISTING'
    var_3 = 'EUR'
    var_4 = 'INVALID_CODE'
    var_5 = None
    var_6 = 'JPY'
    var_7 = 'GBP'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test the __hash__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'Different Name'
    var_8 = 3
    var_9 = 'dollars'
    var_10 = 'yen'
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = -1
    var_14 = -1
    var_15 = 'ALT'
    var_16 = 'Alternative Currency'
    var_17 = 4



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'Test the __ge__ (greater than or equal) comparison method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'Test the __hash__ method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'United States Dollars'
    var_7 = 3
    var_8 = 'dollar'
    var_9 = 'euro'
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0
    var_13 = 'ZZZ'
    var_14 = 'Some weird currency'
    var_15 = -1



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test that Currency instances are immutable (frozen dataclass).'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.001'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test CurrencyRegistry.__getitem__ method'
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 'NON-EXISTING'
    var_4 = 'XYZ'
    var_5 = 'INVALID'
    var_6 = ''
    var_7 = 'usd'



# Parsed testcases at query #35
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry constructor and singleton behavior.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = len(var_1)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = var_1.has(var_4)
    var_6 = var_1.get(var_4)
    assert var_6 is None
    var_7 = None
    var_8 = var_1.get(var_4, var_7)
    assert var_8 is None
    var_9 = 'USD'
    var_10 = var_1[var_9]

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test CurrencyRegistry context manager functionality.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = '__enter__ should return a callable'
    var_3 = 'TST'
    var_4 = 'Test Currency'
    var_5 = 2
    var_6 = 'TST'
    var_7 = var_1.get(var_6)
    var_8 = var_7.code
    assert var_8 == 'TST'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that registering outside context raises error.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'OUT'
    var_3 = 'Outside Currency'
    var_4 = 2

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that registering duplicate currency raises error.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'DUP'
    var_3 = 'Duplicate Currency'
    var_4 = 2
    var_5 = 'Another Duplicate'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'Test the get method of CurrencyRegistry class.'
    var_1 = 'USD'
    var_2 = 'NON_EXISTING'
    var_3 = 'JPY'
    var_4 = 'EUR'
    var_5 = ''
    var_6 = 'INVALID'
    var_7 = None



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'Test that Currency instances are immutable (frozen dataclass).'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '0.001'



# Parsed testcases at query #38
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __len__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_1)
    var_3 = len(var_1)
    var_4 = var_1.all
    var_5 = len(var_4)
    var_6 = len(var_1)
    var_7 = var_1.codes
    var_8 = len(var_7)
    var_9 = len(var_1)
    var_10 = var_1.codenames
    var_11 = len(var_10)



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'Test the __lt__ method of Currency class for ordering.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8
    var_14 = 'XAU'
    var_15 = 'Gold'



# Parsed testcases at query #40
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that CurrencyRegistry.__new__ creates and returns a singleton instance.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = id(var_1)
    var_4 = id(var_2)



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'Test the __ge__ (greater than or equal) comparison method of Currency class.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euros'
    var_6 = 'GBP'
    var_7 = 'British Pounds'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = -1



# Parsed testcases at query #42
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test that CurrencyRegistry.__new__ creates and returns a singleton instance.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = '\n    Test that Currency instances are immutable and prevent attribute deletion.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2



# Parsed testcases at query #44
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'Test the __contains__ method of CurrencyRegistry.'
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = var_2 in var_1
    assert var_3 is True
    var_4 = 'NONEXISTENT'
    var_5 = var_4 in var_1
    assert var_5 is False



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'Test that Currency instances are immutable and cannot have attributes deleted.'
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2



