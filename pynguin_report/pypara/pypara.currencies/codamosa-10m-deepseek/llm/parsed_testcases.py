####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2
    var_4 = 'FAIL'
    var_5 = 'Fail Currency'
    var_6 = 2



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'



# Parsed testcases at query #3
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.has(var_1)
    assert var_2 is True
    var_3 = 'XXX'
    var_4 = var_0.has(var_3)
    assert var_4 is False
    var_5 = 'usd'
    var_6 = var_0.has(var_5)
    assert var_6 is False
    var_7 = ''
    var_8 = var_0.has(var_7)
    assert var_8 is False
    var_9 = None
    var_10 = var_0.has(var_9)
    var_11 = '123'
    var_12 = var_0.has(var_11)
    assert var_12 is False
    var_13 = '@#!'
    var_14 = var_0.has(var_13)
    assert var_14 is False
    var_15 = 'TEST'
    var_16 = 'Test Currency'
    var_17 = 2
    var_18 = 'TEST'
    var_19 = var_0.has(var_18)
    assert var_19 is True



# Parsed testcases at query #4
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'ABC'
    var_2 = 'Test Currency'
    var_3 = 2



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'USDX'
    var_4 = 'UX Dollars'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'XXX'
    var_2 = ''
    var_3 = 'usd'
    var_4 = '123'
    var_5 = None



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
    var_9 = var_0.get(var_7)
    var_10 = 'NON_EXISTING'
    var_11 = var_0[var_10]
    var_12 = 'NON_EXISTING'
    var_13 = var_0.get(var_12)
    assert var_13 is None
    var_14 = 'USD'
    var_15 = 'US Dollar'
    var_16 = 2



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = '0.5'
    var_11 = '0'
    var_12 = '1.5'
    var_13 = '2'
    var_14 = 'ZZZ'
    var_15 = 'Some weird currency'
    var_16 = -1
    var_17 = '1.0000000000005'
    var_18 = '1.000000000000'
    var_19 = '1.0000000000015'
    var_20 = '1.000000000002'
    var_21 = 'UX Dollars'
    var_22 = 'usd'
    var_23 = 'US Dollars'
    var_24 = 2
    var_25 = 'US1'
    var_26 = 'US Dollars'
    var_27 = 2
    var_28 = ''
    var_29 = 'US Dollars'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = ' US Dollars'
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars '
    var_39 = 2
    var_40 = 'USD'
    var_41 = 'US Dollars'
    var_42 = -2
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = '2'
    var_46 = 'USD'
    var_47 = 'US Dollars'
    var_48 = 2
    var_49 = 'MONEY'



# Parsed testcases at query #9
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 3



# Parsed testcases at query #12
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.has(var_1)
    assert var_2 is True
    var_3 = 'XXX'
    var_4 = var_0.has(var_3)
    assert var_4 is False
    var_5 = ''
    var_6 = var_0.has(var_5)
    assert var_6 is False
    var_7 = None
    var_8 = var_0.has(var_7)
    var_9 = 'usd'
    var_10 = var_0.has(var_9)
    assert var_10 is False



# Parsed testcases at query #13
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
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = len(var_0)
    assert var_10 == 3
    var_11 = 'GBP'
    var_12 = 'British Pound'
    var_13 = 2
    var_14 = len(var_0)
    assert var_14 == 4



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 2



# Parsed testcases at query #15
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
    var_9 = 'XYZ'
    var_10 = var_0.has(var_9)
    assert var_10 is False
    var_11 = var_0.get(var_7)
    var_12 = var_0.get(var_9)
    assert var_12 is None
    var_13 = 'XYZ'
    var_14 = var_0[var_13]



# Parsed testcases at query #16
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.001'



# Parsed testcases at query #18
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"
    var_3 = 'USD'
    var_4 = module_0.CurrencyLookupError(var_3)
    var_5 = str(var_4)
    assert var_5 == "Currency identified by code 'USD' does not exist"



# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'



# Parsed testcases at query #21
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"



# Parsed testcases at query #22
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'GBP'
    var_10 = 'British Pound'
    var_11 = 2



# Parsed testcases at query #23
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'EUR'
    var_8 = 'Euro'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



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
    var_8 = 2
    var_9 = len(var_0)
    assert var_9 == 2



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = '0.5'
    var_11 = '0'
    var_12 = '1.5'
    var_13 = '2'
    var_14 = 'ZZZ'
    var_15 = 'Some weird currency'
    var_16 = -1
    var_17 = '1.0000000000005'
    var_18 = '1.000000000000'
    var_19 = '1.0000000000015'
    var_20 = '1.000000000002'



# Parsed testcases at query #27
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
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = len(var_0)
    assert var_10 == 3
    var_11 = 'GBP'
    var_12 = 'British Pound'
    var_13 = 2
    var_14 = len(var_0)
    assert var_14 == 4



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    pass



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.__enter__()
    var_2 = None
    var_3 = var_0.__exit__(var_2, var_2, var_2)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 0



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'XXX'
    var_2 = ''
    var_3 = None
    var_4 = 123



# Parsed testcases at query #4
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #7
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.001'



# Parsed testcases at query #11
#--------------------------


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1
    var_11 = -1
    var_12 = 'UX Dollars'
    var_13 = '123'
    var_14 = 'Invalid Code'
    var_15 = 2
    var_16 = 'usd'
    var_17 = 'Invalid Code'
    var_18 = 2
    var_19 = 'USD'
    var_20 = ''
    var_21 = 2
    var_22 = 'USD'
    var_23 = ' US Dollars '
    var_24 = 2
    var_25 = 'USD'
    var_26 = 'US Dollars'
    var_27 = -2
    var_28 = 'USD'
    var_29 = 'US Dollars'
    var_30 = 'two'
    var_31 = 'USD'
    var_32 = 'US Dollars'
    var_33 = 2
    var_34 = 'Invalid Type'



# Parsed testcases at query #12
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'XXX'
    var_2 = ''
    var_3 = 'usd'
    var_4 = '123'
    var_5 = 'U$D'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 123



# Parsed testcases at query #16
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
    var_11 = 'not_a_currency'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('1E-2'), hashcache={})"
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('1E-0'), hashcache={})"
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1
    var_11 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=Decimal('1E-12'), hashcache={})"



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = '0.5'
    var_11 = '0'
    var_12 = '1.5'
    var_13 = '2'
    var_14 = 'ZZZ'
    var_15 = 'Some weird currency'
    var_16 = -1
    var_17 = '1.0000000000005'
    var_18 = '1.000000000000'
    var_19 = '1.0000000000015'
    var_20 = '1.000000000002'
    var_21 = 'UX Dollars'
    var_22 = 'usd'
    var_23 = 'US Dollars'
    var_24 = 2
    var_25 = 'USD1'
    var_26 = 'US Dollars'
    var_27 = 2
    var_28 = 'USD'
    var_29 = ''
    var_30 = 2
    var_31 = 'USD'
    var_32 = ' US Dollars '
    var_33 = 2
    var_34 = 'USD'
    var_35 = 'US Dollars'
    var_36 = -2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = 2
    var_40 = 'InvalidType'



# Parsed testcases at query #20
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
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0
    var_13 = '0.5'
    var_14 = '0'
    var_15 = '1.5'
    var_16 = '2'
    var_17 = '2.5'
    var_18 = '3.5'
    var_19 = '4'
    var_20 = 'ZZZ'
    var_21 = 'Some weird currency'
    var_22 = -1
    var_23 = '1.0000000000005'
    var_24 = '1.000000000000'
    var_25 = '1.0000000000015'
    var_26 = '1.000000000002'
    var_27 = '1.0000000000025'
    var_28 = '1.0000000000035'
    var_29 = '1.000000000004'
    var_30 = '1'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 3
    var_5 = '0.01'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'XXX'
    var_2 = ''
    var_3 = None
    var_4 = 123



# Parsed testcases at query #23
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
    var_9 = 'UX Dollars'



# Parsed testcases at query #24
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #25
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'XYZ'
    var_5 = var_0[var_4]



# Parsed testcases at query #26
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = var_0.get(var_1)
    var_5 = 'XXX'
    var_6 = var_0.get(var_5)
    assert var_6 is None
    var_7 = 'EUR'
    var_8 = 'Euro'



# Parsed testcases at query #27
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = len(var_0)
    assert var_5 == 1
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 2
    var_9 = len(var_0)
    assert var_9 == 2
    var_10 = module_0.CurrencyRegistry()
    var_11 = len(var_10)
    assert var_11 == 0



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = '0.5'
    var_11 = '0'
    var_12 = '1.5'
    var_13 = '2'
    var_14 = 'ZZZ'
    var_15 = 'Some weird currency'
    var_16 = -1
    var_17 = '1.0000000000005'
    var_18 = '1.000000000000'
    var_19 = '1.0000000000015'
    var_20 = '1.000000000002'



# Parsed testcases at query #29
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #30
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



# Parsed testcases at query #31
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



# Parsed testcases at query #32
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = '0.5'
    var_11 = '0'
    var_12 = '1.5'
    var_13 = '2'
    var_14 = 'ZZZ'
    var_15 = 'Some weird currency'
    var_16 = -1
    var_17 = '1.0000000000005'
    var_18 = '1.000000000000'
    var_19 = '1.0000000000015'
    var_20 = '1.000000000002'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'BTC'
    var_5 = 'Bitcoin'
    var_6 = 8
    var_7 = 0
    var_8 = 'American Dollars'



# Parsed testcases at query #35
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



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NON-EXISTING'
    var_2 = 'usd'
    var_3 = ''



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



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



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'UX Dollars'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 0



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'



# Parsed testcases at query #11
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"



# Parsed testcases at query #12
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
    var_9 = 'EUR'
    var_10 = var_0.has(var_9)
    var_11 = var_0.get(var_7)
    var_12 = var_0.get(var_9)
    assert var_12 is None
    var_13 = 'USD'
    var_14 = 'US Dollar'
    var_15 = 2
    var_16 = 'EUR'
    var_17 = 'Euro'
    var_18 = 2
    var_19 = 'XYZ'
    var_20 = var_0[var_19]



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'BTC'
    var_7 = 'Bitcoin'
    var_8 = -1



# Parsed testcases at query #15
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0



# Parsed testcases at query #16
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



# Parsed testcases at query #17
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = '_CurrencyRegistry__registry'
    var_3 = hasattr(var_0, var_2)
    var_4 = '_CurrencyRegistry__currencies'
    var_5 = hasattr(var_0, var_4)
    var_6 = '_CurrencyRegistry__codes'
    var_7 = hasattr(var_0, var_6)
    var_8 = '_CurrencyRegistry__codenames'
    var_9 = hasattr(var_0, var_8)
    var_10 = '_CurrencyRegistry__ctx_open'
    var_11 = hasattr(var_0, var_10)



# Parsed testcases at query #18
#--------------------------


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = module_0.make_quantizer(var_2)
    var_4 = module_0.make_quantizer(var_2)
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1
    var_11 = 'EUR'
    var_12 = 'Euro'
    var_13 = 1
    var_14 = '1.005'
    var_15 = '1.00'
    var_16 = '1.015'
    var_17 = '1.02'
    var_18 = 'usd'
    var_19 = 'US Dollars'
    var_20 = 2
    var_21 = 'US1'
    var_22 = 'US Dollars'
    var_23 = 2
    var_24 = 'USD'
    var_25 = ''
    var_26 = 2
    var_27 = 'USD'
    var_28 = ' US Dollars '
    var_29 = 2
    var_30 = 'USD'
    var_31 = 'US Dollars'
    var_32 = -2
    var_33 = 'USD'
    var_34 = 'US Dollars'
    var_35 = 2
    var_36 = 'InvalidType'



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



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'USX'
    var_7 = 'UX Dollars'
    var_8 = 3



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



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = None
    var_1 = 123



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #26
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'USD'



# Parsed testcases at query #28
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



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'XXX'
    var_2 = ''
    var_3 = 'usd'
    var_4 = '123'
    var_5 = 'US$'



# Parsed testcases at query #30
#--------------------------




