####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 3
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 3
    var_7 = 'AAA'
    var_8 = 'Currency A'
    var_9 = 'BBB'
    var_10 = 'Currency B'
    var_11 = 'hashcache'
    var_12 = 999



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = 'UX Dollars'
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8
    var_12 = 'ETH'
    var_13 = 'Ethereum'
    var_14 = 'XAU'
    var_15 = 'Gold'
    var_16 = 4
    var_17 = 'XAG'
    var_18 = 'Silver'
    var_19 = 'AAA'
    var_20 = 'Currency A'
    var_21 = 'ZZZ'
    var_22 = 'Currency Z'
    var_23 = 'ABC'
    var_24 = 'Test Currency'
    var_25 = -1
    var_26 = 'First'
    var_27 = 'BBB'
    var_28 = 'Second'
    var_29 = 1
    var_30 = 'CCC'
    var_31 = 'Third'



# Parsed testcases at query #3
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = len(var_0)
    assert var_7 == 2
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8
    var_14 = len(var_0)
    assert var_14 == 4
    var_15 = 'USD'
    var_16 = 'US Dollar'
    var_17 = 2
    var_18 = len(var_0)
    assert var_18 == 4



# Parsed testcases at query #4
#--------------------------


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = -1
    var_10 = 'UX Dollars'
    var_11 = 3
    var_12 = '1.005'
    var_13 = '1.00'
    var_14 = '1.015'
    var_15 = '1.02'
    var_16 = '0.5'
    var_17 = '0'
    var_18 = '1.5'
    var_19 = '2'
    var_20 = '1.0000000000005'
    var_21 = '1.000000000000'
    var_22 = '1.0000000000015'
    var_23 = '1.000000000002'
    var_24 = 'AAA'
    var_25 = 'Currency A'
    var_26 = 'BBB'
    var_27 = 'Currency B'
    var_28 = 'CCC'
    var_29 = 'Currency C'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0.01'), hashcache="
    var_4 = ')'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0'), hashcache="
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = -1
    var_12 = "Currency(code='BTC', name='Bitcoin', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=Decimal('0.000000000000000000000000000000000000000000000000000000000000'), hashcache="
    var_13 = 'XAU'
    var_14 = 'Gold'
    var_15 = 4
    var_16 = "Currency(code='XAU', name='Gold', decimals=4, type=<CurrencyType.METAL: 'Precious Metal'>, quantizer=Decimal('0.0001'), hashcache="
    var_17 = 'ALT'
    var_18 = 'Alternative Currency'
    var_19 = 3
    var_20 = "Currency(code='ALT', name='Alternative Currency', decimals=3, type=<CurrencyType.ALTERNATIVE: 'Alternative'>, quantizer=Decimal('0.001'), hashcache="



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0.01'), hashcache="
    var_4 = ')'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0'), hashcache="
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = -1
    var_12 = "Currency(code='BTC', name='Bitcoin', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=MaxPrecisionQuantizer, hashcache="
    var_13 = 'XAU'
    var_14 = 'Gold'
    var_15 = 4
    var_16 = "Currency(code='XAU', name='Gold', decimals=4, type=<CurrencyType.METAL: 'Precious Metal'>, quantizer=Decimal('0.0001'), hashcache="
    var_17 = 'LTS'
    var_18 = 'Local Trade System'
    var_19 = "Currency(code='LTS', name='Local Trade System', decimals=2, type=<CurrencyType.ALTERNATIVE: 'Alternative'>, quantizer=Decimal('0.01'), hashcache="



# Parsed testcases at query #7
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = 'USD'
    var_4 = 'US Dollar'
    var_5 = 2
    var_6 = len(var_0)
    assert var_6 == 1
    var_7 = 'USD'
    var_8 = var_0.has(var_7)
    assert var_8 is True
    var_9 = var_0.get(var_7)
    var_10 = 'EUR'
    var_11 = 'Euro'
    var_12 = 2
    var_13 = var_0.get(var_10)
    assert var_13 is None
    var_14 = var_0.has(var_10)
    assert var_14 is False
    var_15 = 'EUR'
    var_16 = var_0[var_15]
    var_17 = 'EUR'
    var_18 = 'Euro'
    var_19 = 2
    var_20 = 'JPY'
    var_21 = 'Japanese Yen'
    var_22 = 0



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = 'US Dollars'
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8
    var_12 = 10
    var_13 = 'XAU'
    var_14 = 'Gold'
    var_15 = -1
    var_16 = -1
    var_17 = -1
    var_18 = 'LTS'
    var_19 = 'Local Time'
    var_20 = 'XYZ'
    var_21 = 'Test Currency'
    var_22 = 'ZZZ'
    var_23 = 'Weird Currency'
    var_24 = -1



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = 'EUR'
    var_3 = 'Euro'
    var_4 = 2
    var_5 = None
    var_6 = 'JPY'
    var_7 = 'usd'
    var_8 = 'Usd'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = 'BTC'
    var_9 = 'Bitcoin'
    var_10 = 8
    var_11 = 'XAU'
    var_12 = 'Gold'
    var_13 = -1
    var_14 = 'not a currency'
    var_15 = 'AUD'
    var_16 = 'Australian Dollar'
    var_17 = 'CAD'
    var_18 = 'Canadian Dollar'
    var_19 = 'GBP'
    var_20 = 'British Pound'



# Parsed testcases at query #11
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
    var_6 = 'TEST'
    var_7 = module_0.CurrencyLookupError(var_6)
    var_8 = 'ERR'
    var_9 = module_0.CurrencyLookupError(var_8)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXIST'
    var_2 = 'EUR'
    var_3 = 'Euro'
    var_4 = 2
    var_5 = 'NONEXIST'
    var_6 = ''
    var_7 = None
    var_8 = None



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = 'United States Dollar'
    var_9 = 3
    var_10 = 'not a currency'



# Parsed testcases at query #14
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = '_CurrencyRegistry__registry'
    var_3 = hasattr(var_0, var_2)
    var_4 = var_0._CurrencyRegistry__registry
    var_5 = var_0._CurrencyRegistry__registry
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = '_CurrencyRegistry__currencies'
    var_8 = hasattr(var_0, var_7)
    var_9 = '_CurrencyRegistry__codes'
    var_10 = hasattr(var_0, var_9)
    var_11 = '_CurrencyRegistry__codenames'
    var_12 = hasattr(var_0, var_11)
    var_13 = '_CurrencyRegistry__ctx_open'
    var_14 = hasattr(var_0, var_13)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'USD'
    var_2 = 'XXX'



# Parsed testcases at query #16
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = '_CurrencyRegistry__registry'
    var_3 = hasattr(var_0, var_2)
    var_4 = var_0._CurrencyRegistry__registry
    var_5 = var_0._CurrencyRegistry__registry
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = '_CurrencyRegistry__currencies'
    var_8 = hasattr(var_0, var_7)
    var_9 = '_CurrencyRegistry__codes'
    var_10 = hasattr(var_0, var_9)
    var_11 = '_CurrencyRegistry__codenames'
    var_12 = hasattr(var_0, var_11)
    var_13 = '_CurrencyRegistry__ctx_open'
    var_14 = hasattr(var_0, var_13)
    var_15 = module_0.CurrencyRegistry()



# Parsed testcases at query #17
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #20
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2
    var_4 = 'TEST2'
    var_5 = 'Test Currency 2'
    var_6 = 2
    var_7 = str(var_4)
    var_8 = 'AAA'
    var_9 = 'AAA Currency'
    var_10 = 0
    var_11 = 'ZZZ'
    var_12 = 'ZZZ Currency'
    var_13 = 2



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.001'
    var_4 = '1.005'
    var_5 = '1.00'
    var_6 = '1.015'
    var_7 = '1.02'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'EUR'
    var_8 = 'Euro'
    var_9 = 'AAA'
    var_10 = 'Currency A'
    var_11 = 'BBB'
    var_12 = 'Currency B'
    var_13 = 'CCC'
    var_14 = 'Currency C'
    var_15 = 'BTC'
    var_16 = 'Bitcoin'
    var_17 = 8
    var_18 = 'XAU'
    var_19 = 'Gold'
    var_20 = 4



# Parsed testcases at query #23
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = len(var_0)
    assert var_6 == 2



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'not a currency'
    var_10 = 'BTC'
    var_11 = 'Bitcoin'
    var_12 = 8
    var_13 = 'XAU'
    var_14 = 'Gold'
    var_15 = -1



# Parsed testcases at query #25
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = len(var_0)
    assert var_5 == 1
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = len(var_0)
    assert var_8 == 2
    var_9 = 'JPY'
    var_10 = 'Japanese Yen'
    var_11 = 0
    var_12 = len(var_0)
    assert var_12 == 3
    var_13 = len(var_0)
    assert var_13 == 3
    var_14 = 'USD'
    var_15 = 'US Dollar'
    var_16 = 2
    var_17 = len(var_0)
    assert var_17 == 3
    var_18 = 'BTC'
    var_19 = 'Bitcoin'
    var_20 = 8
    var_21 = len(var_0)
    assert var_21 == 4
    var_22 = 'XAU'
    var_23 = 'Gold'
    var_24 = -1
    var_25 = len(var_0)
    assert var_25 == 5
    var_26 = len(var_0)
    assert var_26 == 5



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = ''
    var_3 = 'usd'
    var_4 = '123'
    var_5 = 'EUR'
    var_6 = 'USD-'
    var_7 = 'USD/EUR'
    var_8 = ' USD '
    var_9 = '\tUSD'
    var_10 = 'TEST'



# Parsed testcases at query #27
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'ZZZ'
    var_2 = 'Test Currency Z'
    var_3 = 2
    var_4 = 'AAA'
    var_5 = 'Test Currency A'
    var_6 = 0
    var_7 = 'MMM'
    var_8 = 'Test Currency M'
    var_9 = -1
    var_10 = module_0.CurrencyRegistry()
    var_11 = len(var_10)
    assert var_11 == 0



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = 'UX Dollars'
    var_11 = '1.005'
    var_12 = '1.00'
    var_13 = '1.015'
    var_14 = '1.02'
    var_15 = '0.5'
    var_16 = '0'
    var_17 = '1.5'
    var_18 = '2'
    var_19 = '1.0000000000005'
    var_20 = '1.000000000000'
    var_21 = '1.0000000000015'
    var_22 = '1.000000000002'
    var_23 = 'XAU'
    var_24 = 'Gold'
    var_25 = 4
    var_26 = 'LOC'
    var_27 = 'Local Currency'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'UX Dollars'
    var_9 = 3
    var_10 = 'BTC'
    var_11 = 'Bitcoin'
    var_12 = 8
    var_13 = 'XAU'
    var_14 = 'Gold'
    var_15 = 4
    var_16 = 'ZZZ'
    var_17 = 'Some weird currency'
    var_18 = -1



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.001'
    var_4 = '1.005'
    var_5 = '1.00'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 'GBP'
    var_3 = 'JPY'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 'XYZ'
    var_6 = 'ABC'
    var_7 = '123'
    var_8 = 'usd'
    var_9 = [var_5, var_6, var_7, var_8]



# Parsed testcases at query #32
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.has(var_1)
    var_3 = 'XXX'
    var_4 = var_0.has(var_3)



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    assert var_1 is False
    var_2 = ''
    var_3 = 'usd'
    var_4 = 'EUR'
    var_5 = 'USD$'
    var_6 = '123'
    var_7 = None



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'ZZZ'
    var_7 = 'Some weird currency'
    var_8 = -1
    var_9 = 'XAU'
    var_10 = 'Gold'
    var_11 = 4
    var_12 = 'ALT'
    var_13 = 'Alternative Currency'
    var_14 = 3



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = '0.000'
    var_8 = '0.00'
    var_9 = '123.456'
    var_10 = '123.46'
    var_11 = '-1.005'
    var_12 = '-1.00'
    var_13 = '-1.015'
    var_14 = '-1.02'
    var_15 = 'JPY'
    var_16 = 'Japanese Yen'
    var_17 = 0
    var_18 = '0.5'
    var_19 = '0'
    var_20 = '1.5'
    var_21 = '2'
    var_22 = '123'
    var_23 = '-0.5'
    var_24 = '-1.5'
    var_25 = '-2'
    var_26 = 'ZZZ'
    var_27 = 'Some weird currency'
    var_28 = -1
    var_29 = '1.0000000000005'
    var_30 = '1.000000000000'
    var_31 = '1.0000000000015'
    var_32 = '1.000000000002'
    var_33 = '0.0000000000005'
    var_34 = '0.000000000000'
    var_35 = '-1.0000000000005'
    var_36 = '-1.000000000000'
    var_37 = 'ABC'
    var_38 = 'One Decimal'
    var_39 = 1
    var_40 = '1.05'
    var_41 = '1.0'
    var_42 = '1.15'
    var_43 = '1.2'
    var_44 = '0.05'
    var_45 = '0.0'
    var_46 = 'XYZ'
    var_47 = 'Three Decimals'
    var_48 = 3
    var_49 = '1.0005'
    var_50 = '1.000'
    var_51 = '1.0015'
    var_52 = '1.002'
    var_53 = '0.0005'
    var_54 = '100.00'
    var_55 = '100'



# Parsed testcases at query #36
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = 'USD'
    var_4 = 'US Dollar'
    var_5 = 2
    var_6 = len(var_0)
    assert var_6 == 1
    var_7 = 'USD'
    var_8 = var_0.has(var_7)
    assert var_8 is True
    var_9 = var_0.get(var_7)
    var_10 = 'EUR'
    var_11 = 'Euro'
    var_12 = 2
    var_13 = 'JPY'
    var_14 = 'Japanese Yen'
    var_15 = 0
    var_16 = len(var_0)
    assert var_16 == 3
    var_17 = 'USD'
    var_18 = 'US Dollar'
    var_19 = 2
    var_20 = 'GBP'
    var_21 = 'British Pound'
    var_22 = 2
    var_23 = 'EUR'
    var_24 = var_0.has(var_23)
    assert var_24 is True
    var_25 = 'XYZ'
    var_26 = var_0.has(var_25)
    assert var_26 is False
    var_27 = var_0.get(var_23)
    var_28 = var_0.get(var_25)
    assert var_28 is None
    var_29 = 'XYZ'
    var_30 = var_0[var_29]
    var_31 = len(var_1)
    assert var_31 == 3



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = ''
    var_3 = 'usd'
    var_4 = '123'
    var_5 = 'USD$'
    var_6 = 'EUR'
    var_7 = 'GBP'
    var_8 = 'JPY'



# Parsed testcases at query #38
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = module_0.CurrencyRegistry()
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 2
    var_8 = 'GBP'
    var_9 = 'British Pound'
    var_10 = 'AUD'
    var_11 = 'Australian Dollar'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'EUR'
    var_8 = 'Euro'
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8
    var_12 = 'ETH'
    var_13 = 'Ethereum'
    var_14 = 18
    var_15 = 'XAU'
    var_16 = 'Gold'
    var_17 = -1



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #41
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = '_CurrencyRegistry__registry'
    var_3 = hasattr(var_0, var_2)
    var_4 = var_0._CurrencyRegistry__registry
    var_5 = var_0._CurrencyRegistry__registry
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = 'has'
    var_8 = hasattr(var_0, var_7)
    var_9 = 'get'
    var_10 = hasattr(var_0, var_9)
    var_11 = '__contains__'
    var_12 = hasattr(var_0, var_11)
    var_13 = '__getitem__'
    var_14 = hasattr(var_0, var_13)
    var_15 = '__len__'
    var_16 = hasattr(var_0, var_15)
    var_17 = var_0.all
    var_18 = var_0.codes
    var_19 = var_0.codenames



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8
    var_10 = 'XAU'
    var_11 = 'Gold'
    var_12 = -1
    var_13 = -1
    var_14 = 'JPY'
    var_15 = 'Japanese Yen'
    var_16 = 'LOC'
    var_17 = 'Local Currency'



# Parsed testcases at query #3
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'NONEXISTENT'
    var_5 = var_0[var_4]
    var_6 = 'usd'
    var_7 = var_0[var_6]
    var_8 = module_0.CurrencyRegistry()
    var_9 = 'EUR'
    var_10 = 'Euro'
    var_11 = 2
    var_12 = 'JPY'
    var_13 = 'Japanese Yen'
    var_14 = 0



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'usd'
    var_1 = 'NON-EXISTING'
    var_2 = 'XYZ'
    var_3 = 'USD'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 'ZZZ'
    var_10 = 'Weird Currency'
    var_11 = -1



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'EUR'
    var_8 = 'Euro'
    var_9 = 'ABC'
    var_10 = 'Test A'
    var_11 = 'DEF'
    var_12 = 'Test B'
    var_13 = 'GHI'
    var_14 = 'Test C'
    var_15 = 'BTC'
    var_16 = 'Bitcoin'
    var_17 = 8
    var_18 = 'XAU'
    var_19 = 'Gold'
    var_20 = 4
    var_21 = 'USD'
    var_22 = 'ZZZ'
    var_23 = 'Weird Currency'
    var_24 = -1



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = ''
    var_3 = 'usd'
    var_4 = 'USD$'
    var_5 = 'EUR'
    var_6 = '123'
    var_7 = 'A'
    var_8 = 100
    var_9 = var_7 * var_8



# Parsed testcases at query #9
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = module_0.CurrencyRegistry()
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 2
    var_8 = 'GBP'
    var_9 = 'British Pound'
    var_10 = 'AUD'
    var_11 = 'Australian Dollar'
    var_12 = module_0.CurrencyRegistry()
    var_13 = 'TEST'
    var_14 = 'Test Currency'
    var_15 = 2
    var_16 = 'Test Currency Duplicate'
    var_17 = module_0.CurrencyRegistry()
    var_18 = 'XYZ'
    var_19 = 'Test'
    var_20 = 2



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 'usd'
    var_3 = 'NONEXISTENT'
    var_4 = 'XYZ'
    var_5 = 'JPY'
    var_6 = 'BTC'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'AA Dollars'
    var_9 = 'ZZ Dollars'
    var_10 = 1
    var_11 = 3
    var_12 = 'AAA'
    var_13 = 'Currency A'
    var_14 = 'BBB'
    var_15 = 'Currency B'
    var_16 = 'CCC'
    var_17 = 'Currency C'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = '1.025'
    var_8 = '1.035'
    var_9 = '1.04'
    var_10 = '1.01'
    var_11 = '1.99'
    var_12 = 'JPY'
    var_13 = 'Japanese Yen'
    var_14 = 0
    var_15 = '0.5'
    var_16 = '0'
    var_17 = '1.5'
    var_18 = '2'
    var_19 = '2.5'
    var_20 = '3.5'
    var_21 = '4'
    var_22 = '100'
    var_23 = 'ZZZ'
    var_24 = 'Some weird currency'
    var_25 = -1
    var_26 = '1.0000000000005'
    var_27 = '1.000000000000'
    var_28 = '1.0000000000015'
    var_29 = '1.000000000002'
    var_30 = '1.0000000000025'
    var_31 = '1.0000000000035'
    var_32 = '1.000000000004'
    var_33 = '0.0000000000005'
    var_34 = '0.000000000000'
    var_35 = '0.0000000000015'
    var_36 = '0.000000000002'
    var_37 = 'TST'
    var_38 = 'Test Currency'
    var_39 = 1
    var_40 = '1.05'
    var_41 = '1.0'
    var_42 = '1.15'
    var_43 = '1.2'
    var_44 = '1.25'
    var_45 = '1.35'
    var_46 = '1.4'
    var_47 = 'THR'
    var_48 = 'Three Decimal'
    var_49 = 3
    var_50 = '1.0005'
    var_51 = '1.000'
    var_52 = '1.0015'
    var_53 = '1.002'
    var_54 = '1.0025'
    var_55 = '1.0035'
    var_56 = '1.004'
    var_57 = '-1.005'
    var_58 = '-1.00'
    var_59 = '-1.015'
    var_60 = '-1.02'
    var_61 = '-0.5'
    var_62 = '-1.5'
    var_63 = '-2'
    var_64 = '0.00'



# Parsed testcases at query #14
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = 'USD'
    var_4 = 'US Dollar'
    var_5 = 2
    var_6 = len(var_0)
    assert var_6 == 1
    var_7 = 'USD'
    var_8 = var_0.has(var_7)
    assert var_8 is True
    var_9 = var_0.get(var_7)
    var_10 = 'USD'
    var_11 = 'US Dollar'
    var_12 = 2
    var_13 = 'EUR'
    var_14 = 'Euro'
    var_15 = 2
    var_16 = 'JPY'
    var_17 = 'Japanese Yen'
    var_18 = 0
    var_19 = 'XYZ'
    var_20 = var_0[var_19]
    var_21 = 'XYZ'
    var_22 = var_0.get(var_21)
    assert var_22 is None
    var_23 = 'EUR'
    var_24 = var_0.has(var_23)
    assert var_24 is True
    var_25 = var_0.has(var_21)
    assert var_25 is False



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'UX Dollars'
    var_9 = 3
    var_10 = 'not a currency'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'United States Dollars'
    var_6 = 0
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = -1
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 'GBP'
    var_13 = 'British Pound'



# Parsed testcases at query #17
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = module_0.CurrencyRegistry()
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 2
    var_8 = 'GBP'
    var_9 = 'British Pound'
    var_10 = 'AUD'
    var_11 = 'Australian Dollar'
    var_12 = module_0.CurrencyRegistry()
    var_13 = 'TEST'
    var_14 = 'Test'
    var_15 = 2
    var_16 = module_0.CurrencyRegistry()
    var_17 = 'TEST'
    var_18 = 'Test'
    var_19 = 2



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'not a currency'



# Parsed testcases at query #19
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = 'TEST1'
    var_3 = 'Test Currency 1'
    var_4 = 2
    var_5 = len(var_0)
    assert var_5 == 1
    var_6 = 'TEST2'
    var_7 = 'Test Currency 2'
    var_8 = 0
    var_9 = len(var_0)
    assert var_9 == 2
    var_10 = 'TEST3'
    var_11 = 'Test Currency 3'
    var_12 = -1
    var_13 = len(var_0)
    assert var_13 == 3
    var_14 = len(var_0)
    assert var_14 == 3
    var_15 = len(var_0)
    var_16 = var_0.all
    var_17 = len(var_16)
    var_18 = len(var_0)
    var_19 = var_0.codes
    var_20 = len(var_19)
    var_21 = len(var_0)
    var_22 = var_0.codenames
    var_23 = len(var_22)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = 'UX Dollars'
    var_11 = '1.005'
    var_12 = '1.00'
    var_13 = '1.015'
    var_14 = '1.02'
    var_15 = '0.5'
    var_16 = '0'
    var_17 = '1.5'
    var_18 = '2'
    var_19 = '1.0000000000005'
    var_20 = '1.000000000000'
    var_21 = '1.0000000000015'
    var_22 = '1.000000000002'
    var_23 = 'XAU'
    var_24 = 'Gold'
    var_25 = 4
    var_26 = 'LOC'
    var_27 = 'Local Currency'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = 'UX Dollars'
    var_11 = '1.005'
    var_12 = '1.00'
    var_13 = '1.015'
    var_14 = '1.02'
    var_15 = '0.5'
    var_16 = '0'
    var_17 = '1.5'
    var_18 = '2'
    var_19 = 'XAU'
    var_20 = 'Gold'
    var_21 = 4
    var_22 = 'ALT'
    var_23 = 'Alternative'



# Parsed testcases at query #22
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = len(var_0)
    assert var_5 == 1
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = len(var_0)
    assert var_8 == 2
    var_9 = 'JPY'
    var_10 = 'Japanese Yen'
    var_11 = 0
    var_12 = len(var_0)
    assert var_12 == 3
    var_13 = len(var_0)
    assert var_13 == 3
    var_14 = 'GBP'
    var_15 = 'British Pound'
    var_16 = 2
    var_17 = len(var_0)
    assert var_17 == 3
    var_18 = 'USD'
    var_19 = 'US Dollar'
    var_20 = 2
    var_21 = len(var_0)
    assert var_21 == 3



# Parsed testcases at query #23
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'EUR'
    var_2 = 'Euro'
    var_3 = 2
    var_4 = 'USD'
    var_5 = 'US Dollar'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'AED'
    var_10 = 'UAE Dirham'
    var_11 = module_0.CurrencyRegistry()



# Parsed testcases at query #24
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'ZZZ'
    var_2 = 'Z Currency'
    var_3 = 2
    var_4 = 'AAA'
    var_5 = 'A Currency'
    var_6 = 'MMM'
    var_7 = 'M Currency'
    var_8 = list(var_1)
    var_9 = len(var_0)
    assert var_9 == 3
    var_10 = 'AAA'
    var_11 = var_0.has(var_10)
    assert var_11 is True
    var_12 = 'BBB'
    var_13 = var_0.has(var_12)
    assert var_13 is False
    var_14 = var_0.get(var_10)
    var_15 = var_14.code
    assert var_15 == 'AAA'
    var_16 = var_0.get(var_12)
    assert var_16 is None
    var_17 = var_0[var_10]
    var_18 = var_0.get(var_12, var_17)
    var_19 = var_18.code
    assert var_19 == 'AAA'



# Parsed testcases at query #25
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 2



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'A Dollars'
    var_9 = 'B Dollars'
    var_10 = 3
    var_11 = 'BTC'
    var_12 = 'Bitcoin'
    var_13 = 8
    var_14 = 'ETH'
    var_15 = 'Ethereum'
    var_16 = 'XRP'
    var_17 = 'Ripple'
    var_18 = 6
    var_19 = 'not a currency'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollar'
    var_6 = 0
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = -1
    var_12 = 'ETH'
    var_13 = 'Ethereum'
    var_14 = -1
    var_15 = 'CAD'
    var_16 = 'Canadian Dollar'
    var_17 = 'GBP'
    var_18 = 'British Pound'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'EUR'
    var_7 = 'UX Dollars'
    var_8 = 3
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = -1



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'EUR'
    var_8 = 'Euro'
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8
    var_12 = 'XAU'
    var_13 = 'Gold'
    var_14 = -1



# Parsed testcases at query #32
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = module_0.CurrencyRegistry()
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 2
    var_8 = 'GBP'
    var_9 = 'British Pound'
    var_10 = module_0.CurrencyRegistry()
    var_11 = 'TEST'
    var_12 = 'Test Currency'
    var_13 = 2
    var_14 = 'Another Test'
    var_15 = module_0.CurrencyRegistry()



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0.01'), hashcache="
    var_4 = ')'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0'), hashcache="
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = -1
    var_12 = "Currency(code='BTC', name='Bitcoin', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=MaxPrecisionQuantizer, hashcache="
    var_13 = 'XAU'
    var_14 = 'Gold'
    var_15 = 4
    var_16 = "Currency(code='XAU', name='Gold', decimals=4, type=<CurrencyType.METAL: 'Precious Metal'>, quantizer=Decimal('0.0001'), hashcache="
    var_17 = 'LVC'
    var_18 = 'Local Currency'
    var_19 = "Currency(code='LVC', name='Local Currency', decimals=2, type=<CurrencyType.ALTERNATIVE: 'Alternative'>, quantizer=Decimal('0.01'), hashcache="



