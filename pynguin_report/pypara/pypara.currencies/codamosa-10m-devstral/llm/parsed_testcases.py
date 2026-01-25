####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = ''
    var_3 = None
    var_4 = 'usd'



# Parsed testcases at query #4
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 3
    var_7 = -1



# Parsed testcases at query #7
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'USD'
    var_7 = var_0[var_6]
    var_8 = 'NON-EXISTING'
    var_9 = var_0[var_8]
    var_10 = str(var_9)
    assert var_10 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 3
    var_4 = 0
    var_5 = 'EUR'
    var_6 = 'Euro'



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'TEST'
    var_1 = 'Test Currency'
    var_2 = 2
    var_3 = 'NONEXIST'
    var_4 = 'DEFAULT'
    var_5 = 'Default Currency'
    var_6 = 0



# Parsed testcases at query #11
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
    var_22 = '123'
    var_23 = 'Invalid Code'
    var_24 = 2
    var_25 = 'abc'
    var_26 = 'Invalid Code'
    var_27 = 2
    var_28 = 'Abc'
    var_29 = 'Invalid Code'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = '  '
    var_36 = 2
    var_37 = 'USD'
    var_38 = ' US Dollars'
    var_39 = 2
    var_40 = 'USD'
    var_41 = 'US Dollars '
    var_42 = 2
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = -2
    var_46 = 'USD'
    var_47 = 'US Dollars'
    var_48 = '2'
    var_49 = 'USD'
    var_50 = 'US Dollars'
    var_51 = 2
    var_52 = 'MONEY'



# Parsed testcases at query #12
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
    var_9 = len(var_0)
    assert var_9 == 3



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8



# Parsed testcases at query #15
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #16
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #17
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #19
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'NON-EXISTING'
    var_5 = var_0[var_4]
    var_6 = str(var_4)
    assert var_6 == "Currency identified by code 'NON-EXISTING' does not exist"



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
    var_28 = 123
    var_29 = 'US Dollars'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = '  US Dollars'
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars  '
    var_39 = 2
    var_40 = 'USD'
    var_41 = 123
    var_42 = 2
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = -2
    var_46 = 'USD'
    var_47 = 'US Dollars'
    var_48 = '2'
    var_49 = 'USD'
    var_50 = 'US Dollars'
    var_51 = 2
    var_52 = 'MONEY'



# Parsed testcases at query #21
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache="
    var_4 = ')'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache="
    var_9 = 'ZZZ'
    var_10 = 'Some weird currency'
    var_11 = -1
    var_12 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=Decimal('1E-28'), hashcache="



# Parsed testcases at query #23
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
    var_11 = 'UX Dollars'



# Parsed testcases at query #24
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #28
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



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'code'
    var_4 = 'name'
    var_5 = 'decimals'
    var_6 = 'type'
    var_7 = 'quantizer'
    var_8 = 'hashcache'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'Different Name'
    var_7 = 'ZZZ'
    var_8 = 'Weird Currency'
    var_9 = -1
    var_10 = 'XAU'
    var_11 = 'Gold'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 0
    var_2 = CurrencyRegistry()[var_1][var_0]
    var_3 = 'NON-EXISTING'
    var_4 = 0
    var_5 = CurrencyRegistry()[var_4][var_3]
    var_6 = str(var_5)
    assert var_6 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 0
    var_6 = 'BTC'
    var_7 = 'Bitcoin'
    var_8 = -1



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'NON-EXISTING'
    var_7 = var_0[var_6]



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



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 3



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
    var_7 = 'US Dollars Different'
    var_8 = 3



# Parsed testcases at query #5
#--------------------------


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
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'NON-EXISTING'
    var_5 = var_0[var_4]
    var_6 = str(var_4)
    assert var_6 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #7
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'NON-EXISTING'
    var_7 = var_0[var_6]
    var_8 = str(var_6)
    assert var_8 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #8
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'NON-EXISTING'
    var_7 = var_0[var_6]
    var_8 = str(var_6)
    assert var_8 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'NON-EXISTING'
    var_1 = str(var_0)
    assert var_1 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 0
    var_2 = CurrencyRegistry()[var_1][var_0]
    var_3 = 'NON-EXISTING'
    var_4 = 0
    var_5 = CurrencyRegistry()[var_4][var_3]
    var_6 = str(var_5)
    assert var_6 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #11
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = 'TEST'
    var_4 = 'Test Currency'
    var_5 = 2
    var_6 = len(var_0)
    assert var_6 == 1



# Parsed testcases at query #12
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'USD'
    var_7 = var_0.get(var_6)
    var_8 = 'EUR'
    var_9 = var_0.get(var_8)
    var_10 = 'XYZ'
    var_11 = var_0.get(var_10)
    assert var_11 is None
    var_12 = 'ABC'



# Parsed testcases at query #13
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
    var_8 = 8
    var_9 = 3



# Parsed testcases at query #14
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'NON-EXISTING'
    var_1 = str(var_0)
    assert var_1 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #16
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache="
    var_4 = ')'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache="
    var_9 = 'ZZZ'
    var_10 = 'Some weird currency'
    var_11 = -1
    var_12 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=Decimal('1E-28'), hashcache="



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache="
    var_4 = '0.01'
    var_5 = ')'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 3



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = '1.000'
    var_8 = '1.999'
    var_9 = '2.00'
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0
    var_13 = '0.5'
    var_14 = '0'
    var_15 = '1.5'
    var_16 = '2'
    var_17 = '1.0'
    var_18 = '1'
    var_19 = '2.9'
    var_20 = '3'
    var_21 = 'ZZZ'
    var_22 = 'Some weird currency'
    var_23 = -1
    var_24 = '1.0000000000005'
    var_25 = '1.000000000000'
    var_26 = '1.0000000000015'
    var_27 = '1.000000000002'
    var_28 = '1.0000000000000'
    var_29 = '1.0000000000009'
    var_30 = '1.000000000001'
    var_31 = 'EUR'
    var_32 = 'Euro'
    var_33 = '0.001'
    var_34 = '0.00'
    var_35 = '0.009'
    var_36 = '0.01'
    var_37 = '-1.005'
    var_38 = '-1.00'
    var_39 = '-1.015'
    var_40 = '-1.02'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 3



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
    var_6 = 'USD'
    var_7 = var_0.get(var_6)
    var_8 = 'EUR'
    var_9 = 'XYZ'
    var_10 = var_0.get(var_9)
    assert var_10 is None



# Parsed testcases at query #24
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.get(var_1)
    var_3 = module_0.CurrencyRegistry()
    var_4 = 'NONEXISTENT'
    var_5 = var_3.get(var_4)
    assert var_5 is None
    var_6 = 'XYZ'
    var_7 = 'Test Currency'
    var_8 = 2
    var_9 = module_0.CurrencyRegistry()
    var_10 = module_0.CurrencyRegistry()
    var_11 = module_0.CurrencyRegistry()
    var_12 = len(var_11)
    var_13 = module_0.CurrencyRegistry()
    var_14 = var_13.all
    var_15 = len(var_14)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'Different Name'
    var_7 = 3



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #27
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
    var_11 = 'UX Dollars'
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
    var_24 = 123
    var_25 = 'US Dollars'
    var_26 = 2
    var_27 = 'usd'
    var_28 = 'US Dollars'
    var_29 = 2
    var_30 = 'USD'
    var_31 = ''
    var_32 = 2
    var_33 = 'USD'
    var_34 = ' US Dollars'
    var_35 = 2
    var_36 = 'USD'
    var_37 = 'US Dollars '
    var_38 = 2
    var_39 = 'USD'
    var_40 = 'US Dollars'
    var_41 = -2
    var_42 = 'USD'
    var_43 = 'US Dollars'
    var_44 = 2
    var_45 = 'MONEY'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = '1.000'
    var_8 = '1.001'
    var_9 = '1.009'
    var_10 = '1.01'
    var_11 = 'JPY'
    var_12 = 'Japanese Yen'
    var_13 = 0
    var_14 = '0.5'
    var_15 = '0'
    var_16 = '1.5'
    var_17 = '2'
    var_18 = '1.0'
    var_19 = '1'
    var_20 = '1.9'
    var_21 = 'ZZZ'
    var_22 = 'Some weird currency'
    var_23 = -1
    var_24 = '1.0000000000005'
    var_25 = '1.000000000000'
    var_26 = '1.0000000000015'
    var_27 = '1.000000000002'
    var_28 = '1.00000000000000000000000000001'
    var_29 = '-1.005'
    var_30 = '-1.00'
    var_31 = '-1.015'
    var_32 = '-1.02'
    var_33 = '-1.5'
    var_34 = '-2'
    var_35 = '-1.0000000000005'
    var_36 = '-1.000000000000'
    var_37 = '0.00'



# Parsed testcases at query #30
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #31
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



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'
    var_7 = 3



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1



