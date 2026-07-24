####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'US Dollars 2'
    var_6 = 3



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



# Parsed testcases at query #4
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD$'
    var_29 = 'Special Char Code'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = '  Trim Me  '
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = -2
    var_40 = 'USD'
    var_41 = 'US Dollars'
    var_42 = '2'
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = 2
    var_46 = 'MONEY'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = '0.01'



# Parsed testcases at query #6
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



# Parsed testcases at query #7
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'U$D'
    var_29 = 'Non-Alpha Code'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = '  Trimmed Name  '
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = -2
    var_40 = 'USD'
    var_41 = 'US Dollars'
    var_42 = 2
    var_43 = 'Invalid Type'



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
    var_6 = 'XAU'
    var_7 = 'Gold'
    var_8 = 'BTC'
    var_9 = 'Bitcoin'
    var_10 = 8
    var_11 = 'ZZZ'
    var_12 = 'Some weird currency'
    var_13 = -1
    var_14 = 'YYY'
    var_15 = 'Another weird currency'



# Parsed testcases at query #10
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #12
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
    var_35 = ' US Dollars'
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars '
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



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #14
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = 'TEST'
    var_4 = 'Test Currency'
    var_5 = 2



# Parsed testcases at query #15
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.get(var_1)
    var_3 = CurrencyRegistry()[var_1]
    var_4 = module_0.CurrencyRegistry()
    var_5 = 'NON-EXISTING'
    var_6 = var_4.get(var_5)
    assert var_6 is None
    var_7 = 'US Dollar'
    var_8 = 2
    var_9 = module_0.CurrencyRegistry()
    var_10 = module_0.CurrencyRegistry()
    var_11 = CurrencyRegistry()[var_1]



# Parsed testcases at query #16
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8
    var_10 = 'ZZZ'
    var_11 = 'Some weird currency'
    var_12 = -1



# Parsed testcases at query #19
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
    var_7 = 'TEST'
    var_8 = var_0.has(var_7)
    var_9 = var_0.get(var_7)
    var_10 = 'NONEXISTENT'
    var_11 = var_0.get(var_10)
    assert var_11 is None
    var_12 = 'NONEXISTENT'
    var_13 = var_0[var_12]



# Parsed testcases at query #20
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #21
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
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = len(var_0)
    assert var_8 == 2
    var_9 = 'USD'
    var_10 = var_0.has(var_9)
    var_11 = 'EUR'
    var_12 = var_0.has(var_11)
    var_13 = 'JPY'
    var_14 = var_0.has(var_13)
    var_15 = var_0.get(var_9)
    var_16 = var_0.get(var_13)
    assert var_16 is None
    var_17 = 'JPY'
    var_18 = var_0[var_17]



# Parsed testcases at query #22
#--------------------------


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 0
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = module_0.make_quantizer(var_2)



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 3
    var_7 = '0.01'
    var_8 = 'not a currency'



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



# Parsed testcases at query #28
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST1'
    var_2 = 'Test Currency 1'
    var_3 = 2
    var_4 = 'TEST2'
    var_5 = 'Test Currency 2'
    var_6 = 0



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'code'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = 'usd'



# Parsed testcases at query #33
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



# Parsed testcases at query #34
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
    var_28 = 'USD'
    var_29 = ''
    var_30 = 2
    var_31 = 'USD'
    var_32 = '  Trimmed  '
    var_33 = 2
    var_34 = 'USD'
    var_35 = 'US Dollars'
    var_36 = -2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = 2
    var_40 = 'Invalid Type'



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #36
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



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #38
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
    var_9 = None
    var_10 = var_0.__exit__(var_9, var_9, var_9)



# Parsed testcases at query #39
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
    var_8 = var_7.code
    assert var_8 == 'USD'
    var_9 = 'EUR'
    var_10 = var_0.get(var_9)
    var_11 = var_10.code
    assert var_11 == 'EUR'
    var_12 = 'XYZ'
    var_13 = var_0.get(var_12)
    assert var_13 is None
    var_14 = 'GBP'
    var_15 = 'British Pound'
    var_16 = 2



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = ''
    var_3 = 'usd'



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.001'



# Parsed testcases at query #43
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
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = 'EUR'
    var_11 = 'Euro'
    var_12 = 2
    var_13 = 'USD'
    var_14 = var_0.get(var_13)
    var_15 = 'NONEXISTENT'
    var_16 = var_0.get(var_15)
    assert var_16 is None
    var_17 = var_0.has(var_13)
    var_18 = var_0.has(var_15)
    var_19 = 'NONEXISTENT'
    var_20 = var_0[var_19]



# Parsed testcases at query #44
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



# Parsed testcases at query #45
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"



# Parsed testcases at query #46
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



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'



# Parsed testcases at query #48
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #49
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD$'
    var_29 = 'Special Char Code'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = '  Trimmed  '
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = -2
    var_40 = 'USD'
    var_41 = 'US Dollars'
    var_42 = '2'
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = 2
    var_46 = 'MONEY'



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 3
    var_4 = 0
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'



# Parsed testcases at query #51
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
    var_7 = 'TEST'
    var_8 = var_0.has(var_7)
    var_9 = 'NONEXISTENT'
    var_10 = var_0.has(var_9)
    var_11 = var_0.get(var_7)
    var_12 = var_0.get(var_9)
    assert var_12 is None
    var_13 = 'NONEXISTENT'
    var_14 = var_0[var_13]



# Parsed testcases at query #52
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #54
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



# Parsed testcases at query #55
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
    var_10 = 'XAU'
    var_11 = 'Gold'



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = '0.01'



# Parsed testcases at query #57
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2



# Parsed testcases at query #58
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #59
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



# Parsed testcases at query #60
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
    var_8 = 'XYZ'
    var_9 = var_0[var_8]
    var_10 = str(var_2)
    assert var_10 == "Currency identified by code 'XYZ' does not exist"



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #62
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 3
    var_6 = 'US Dollar'



# Parsed testcases at query #64
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



# Parsed testcases at query #65
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'



# Parsed testcases at query #66
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
    var_31 = '-1.005'
    var_32 = '-1.00'
    var_33 = '-1.015'
    var_34 = '-1.02'
    var_35 = '-0.5'
    var_36 = '-1.5'
    var_37 = '-2'
    var_38 = '-1.0000000000005'
    var_39 = '-1.000000000000'
    var_40 = '-1.0000000000015'
    var_41 = '-1.000000000002'



# Parsed testcases at query #67
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



# Parsed testcases at query #68
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #70
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
    var_35 = ' '
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
    var_48 = 2
    var_49 = 'Invalid Type'



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #72
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #73
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
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1



# Parsed testcases at query #74
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #75
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
    var_10 = 'XYZ'
    var_11 = var_0.get(var_10)
    assert var_11 is None



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'
    var_7 = 'XAU'
    var_8 = 'Gold'



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8
    var_12 = 'ZZZ'
    var_13 = 'Some weird currency'
    var_14 = -1



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = 'UX Dollars'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = '0.5'
    var_12 = '0'
    var_13 = '1.5'
    var_14 = '2'
    var_15 = 'ZZZ'
    var_16 = 'Some weird currency'
    var_17 = -1
    var_18 = '1.0000000000005'
    var_19 = '1.000000000000'
    var_20 = '1.0000000000015'
    var_21 = '1.000000000002'
    var_22 = '123'
    var_23 = 'Invalid Code'
    var_24 = 2
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD '
    var_29 = 'Code with Space'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = ' '
    var_36 = 2
    var_37 = 'USD'
    var_38 = ' Leading Space'
    var_39 = 2
    var_40 = 'USD'
    var_41 = 'Trailing Space '
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



# Parsed testcases at query #80
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #81
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



# Parsed testcases at query #82
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.__enter__()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = None
    var_11 = var_0.__exit__(var_10, var_10, var_10)
    var_12 = 'GBP'
    var_13 = 'British Pound'
    var_14 = 2



# Parsed testcases at query #83
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = CurrencyRegistry()[var_0]
    var_2 = 'NON-EXISTING'
    var_3 = CurrencyRegistry()[var_2]
    var_4 = str(var_3)
    assert var_4 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #84
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



# Parsed testcases at query #85
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})"
    var_8 = 'ALT'
    var_9 = 'Alternative Currency'
    var_10 = 3
    var_11 = "Currency(code='ALT', name='Alternative Currency', decimals=3, type=CurrencyType.ALTERNATIVE, quantizer=Decimal('0.001'), hashcache={})"
    var_12 = 'BTC'
    var_13 = 'Bitcoin'
    var_14 = 8
    var_15 = "Currency(code='BTC', name='Bitcoin', decimals=8, type=CurrencyType.CRYPTO, quantizer=Decimal('0.00000001'), hashcache={})"



# Parsed testcases at query #86
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0
    var_7 = -1



# Parsed testcases at query #87
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
    var_10 = str(var_2)
    assert var_10 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #88
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'AED'
    var_2 = 'UAE Dirham'
    var_3 = 2
    var_4 = 'USD'
    var_5 = 'US Dollar'
    var_6 = len(var_0)
    assert var_6 == 2



# Parsed testcases at query #89
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 3
    var_7 = 'UX Dollars'



# Parsed testcases at query #90
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #91
#--------------------------


def test_case_0():
    var_0 = 'NON-EXISTING'
    var_1 = str(var_0)
    assert var_1 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #92
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD '
    var_29 = 'Code with Space'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = ' '
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



# Parsed testcases at query #93
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #94
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
    var_17 = '0.0'
    var_18 = '999.9'
    var_19 = '1000'
    var_20 = 'ZZZ'
    var_21 = 'Some weird currency'
    var_22 = -1
    var_23 = '1.0000000000005'
    var_24 = '1.000000000000'
    var_25 = '1.0000000000015'
    var_26 = '1.000000000002'
    var_27 = '1.0000000000000'
    var_28 = '1.0000000000001'
    var_29 = 'EUR'
    var_30 = 'Euro'
    var_31 = '0.001'
    var_32 = '0.00'
    var_33 = '0.009'
    var_34 = '0.01'
    var_35 = '-1.005'
    var_36 = '-1.00'
    var_37 = '-1.015'
    var_38 = '-1.02'



# Parsed testcases at query #95
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = 'usd'



# Parsed testcases at query #96
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache="
    var_4 = '0.01'
    var_5 = ')'



# Parsed testcases at query #97
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_0)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"



# Parsed testcases at query #98
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #99
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #100
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



# Parsed testcases at query #101
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTING'
    var_2 = ''
    var_3 = None



# Parsed testcases at query #102
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
    var_9 = 'ZZZ'
    var_10 = 'Some weird currency'
    var_11 = -1
    var_12 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=Decimal('1E-28'), hashcache="



# Parsed testcases at query #103
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #104
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #105
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #106
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #107
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #108
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
    var_7 = len(var_0)
    assert var_7 == 1
    var_8 = 'USD'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = var_0.get(var_8)
    var_13 = var_0.get(var_10)
    assert var_13 is None
    var_14 = 'XYZ'
    var_15 = var_0[var_14]



# Parsed testcases at query #109
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #110
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #111
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



# Parsed testcases at query #112
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #113
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'XAU'
    var_6 = 'Gold'
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0
    var_13 = 'US Dollars (Alternative)'



# Parsed testcases at query #114
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
    var_10 = str(var_2)
    assert var_10 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #115
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



# Parsed testcases at query #116
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2



# Parsed testcases at query #117
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #118
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #119
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



# Parsed testcases at query #120
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
    var_7 = len(var_0)
    assert var_7 == 1
    var_8 = 'USD'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = var_0.get(var_8)
    var_13 = var_0.get(var_10)
    assert var_13 is None
    var_14 = 'XYZ'
    var_15 = var_0[var_14]



# Parsed testcases at query #121
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #122
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #123
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #124
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars Different'
    var_6 = 3



# Parsed testcases at query #125
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 3
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8



# Parsed testcases at query #126
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #127
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



# Parsed testcases at query #128
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
    var_10 = str(var_2)
    assert var_10 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #129
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2



# Parsed testcases at query #130
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
    var_11 = 'ZZZ'
    var_12 = 'Some weird currency'
    var_13 = -1



# Parsed testcases at query #131
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



# Parsed testcases at query #132
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
    var_19 = '9.9'
    var_20 = '10'
    var_21 = 'ZZZ'
    var_22 = 'Some weird currency'
    var_23 = -1
    var_24 = '1.0000000000005'
    var_25 = '1.000000000000'
    var_26 = '1.0000000000015'
    var_27 = '1.000000000002'
    var_28 = '1.0000000000000'
    var_29 = '1.0000000000001'
    var_30 = '-1.005'
    var_31 = '-1.00'
    var_32 = '-1.015'
    var_33 = '-1.02'
    var_34 = '-0.5'
    var_35 = '-1.5'
    var_36 = '-2'
    var_37 = '-1.0000000000005'
    var_38 = '-1.000000000000'
    var_39 = '-1.0000000000015'
    var_40 = '-1.000000000002'



# Parsed testcases at query #133
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
    var_11 = 'ABC'



# Parsed testcases at query #134
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



# Parsed testcases at query #135
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



# Parsed testcases at query #136
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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #2
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



# Parsed testcases at query #4
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD '
    var_29 = 'Code with Space'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = ' '
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



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=<CurrencyType.MONEY: 'Money'>, quantizer=Decimal('0'), hashcache={})"
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1
    var_11 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=<CurrencyType.CRYPTO: 'Crypto Currency'>, quantizer=Decimal('0.000000000000000000000000000001'), hashcache={})"



# Parsed testcases at query #6
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



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTING'
    var_2 = ''
    var_3 = None



# Parsed testcases at query #9
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'AED'
    var_2 = 'UAE Dirham'
    var_3 = 2
    var_4 = 'BHD'
    var_5 = 'Bahraini Dinar'
    var_6 = 3
    var_7 = 'CUC'
    var_8 = 'Cuban Convertible Peso'
    var_9 = len(var_0)
    assert var_9 == 3



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars Different'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'ZZZ'
    var_10 = 'Some weird currency'
    var_11 = -1



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache="
    var_4 = '0.01'
    var_5 = ')'



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
    var_19 = '9.9'
    var_20 = '10'
    var_21 = 'ZZZ'
    var_22 = 'Some weird currency'
    var_23 = -1
    var_24 = '1.0000000000005'
    var_25 = '1.000000000000'
    var_26 = '1.0000000000015'
    var_27 = '1.000000000002'
    var_28 = '1.0000000000000'
    var_29 = '1.0000000000001'
    var_30 = '-1.005'
    var_31 = '-1.00'
    var_32 = '-1.5'
    var_33 = '-2'
    var_34 = '-1.0000000000005'
    var_35 = '-1.000000000000'
    var_36 = '999999999999.995'
    var_37 = '999999999999.99'
    var_38 = '999999999999.5'
    var_39 = '999999999999'



# Parsed testcases at query #14
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
    var_1 = 'TST'
    var_2 = 'Test Currency 1'
    var_3 = 2
    var_4 = 'TST2'
    var_5 = 'Test Currency 2'
    var_6 = 0
    var_7 = var_0.has(var_1)
    var_8 = var_0.has(var_4)
    var_9 = len(var_0)
    assert var_9 == 2
    var_10 = 'AAA'
    var_11 = 'Test Currency 3'
    var_12 = 1



# Parsed testcases at query #17
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
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = len(var_0)
    assert var_11 == 3
    var_12 = len(var_0)
    assert var_12 == 3
    var_13 = 'USD'
    var_14 = var_0.has(var_13)
    var_15 = 'XYZ'
    var_16 = var_0.has(var_15)
    var_17 = var_0.get(var_13)
    var_18 = var_0.get(var_15)
    assert var_18 is None
    var_19 = 'XYZ'
    var_20 = var_0[var_19]



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'NON-EXISTING'



# Parsed testcases at query #19
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
    var_23 = '123'
    var_24 = 'Invalid Code'
    var_25 = 2
    var_26 = 'usd'
    var_27 = 'Lowercase Code'
    var_28 = 2
    var_29 = 'USD '
    var_30 = 'Code with space'
    var_31 = 2
    var_32 = 'USD'
    var_33 = ''
    var_34 = 2
    var_35 = 'USD'
    var_36 = ' '
    var_37 = 2
    var_38 = 'USD'
    var_39 = ' US Dollars'
    var_40 = 2
    var_41 = 'USD'
    var_42 = 'US Dollars '
    var_43 = 2
    var_44 = 'USD'
    var_45 = 'US Dollars'
    var_46 = -2
    var_47 = 'USD'
    var_48 = 'US Dollars'
    var_49 = '2'
    var_50 = 'USD'
    var_51 = 'US Dollars'
    var_52 = 2
    var_53 = 'MONEY'



# Parsed testcases at query #20
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
    var_7 = len(var_0)
    assert var_7 == 1
    var_8 = 'USD'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = var_0.get(var_8)
    var_13 = var_0.get(var_10)
    assert var_13 is None
    var_14 = 'XYZ'
    var_15 = var_0[var_14]



# Parsed testcases at query #21
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
    var_7 = len(var_0)
    assert var_7 == 1
    var_8 = 'USD'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = var_0.get(var_8)
    var_13 = var_0.get(var_10)
    assert var_13 is None
    var_14 = 'XYZ'
    var_15 = var_0[var_14]



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 3
    var_4 = 'US Dollars 2'
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = 'ZZZ'
    var_11 = 'Some weird currency'
    var_12 = -1



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
    var_6 = 'USD'
    var_7 = var_0.get(var_6)
    var_8 = var_7.code
    assert var_8 == 'USD'
    var_9 = 'EUR'
    var_10 = var_0.get(var_9)
    var_11 = var_10.name
    assert var_11 == 'Euro'
    var_12 = 'XYZ'
    var_13 = var_0.get(var_12)
    assert var_13 is None
    var_14 = var_0[var_6]
    var_15 = var_0.get(var_12, var_14)
    var_16 = 'NON-EXISTING'
    var_17 = var_0.get(var_16)
    assert var_17 is None



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
    var_6 = 'USD'
    var_7 = var_0.get(var_6)
    var_8 = var_7.code
    assert var_8 == 'USD'
    var_9 = 'EUR'
    var_10 = var_0.get(var_9)
    var_11 = var_10.name
    assert var_11 == 'Euro'
    var_12 = 'XYZ'
    var_13 = var_0.get(var_12)
    assert var_13 is None
    var_14 = var_0[var_6]
    var_15 = var_0.get(var_12, var_14)
    var_16 = 'NON-EXISTING'
    var_17 = var_0.get(var_16)
    assert var_17 is None



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #28
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #29
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #30
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = 'TEST'
    var_4 = 'Test Currency'
    var_5 = 2



# Parsed testcases at query #31
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
    var_7 = len(var_0)
    assert var_7 == 1
    var_8 = 'USD'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = var_0.get(var_8)
    var_13 = var_0.get(var_10)
    assert var_13 is None
    var_14 = 'XYZ'
    var_15 = var_0[var_14]



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
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = 'XAU'
    var_11 = 'Gold'



# Parsed testcases at query #34
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
    var_10 = 'XAU'
    var_11 = 'Gold'



# Parsed testcases at query #35
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
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #37
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #39
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2



# Parsed testcases at query #40
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
    var_7 = 'EUR'
    var_8 = 'Euro'
    var_9 = len(var_0)
    assert var_9 == 2
    var_10 = 'USD'
    var_11 = var_0.has(var_10)
    var_12 = 'EUR'
    var_13 = var_0.has(var_12)
    var_14 = 'XYZ'
    var_15 = var_0.has(var_14)
    var_16 = var_0.get(var_10)
    var_17 = var_0.get(var_12)
    var_18 = var_0.get(var_14)
    assert var_18 is None
    var_19 = 'XYZ'
    var_20 = var_0[var_19]



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'



# Parsed testcases at query #42
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #43
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



# Parsed testcases at query #44
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
    var_31 = '-1.005'
    var_32 = '-1.00'
    var_33 = '-1.015'
    var_34 = '-1.02'
    var_35 = '-0.5'
    var_36 = '-1.5'
    var_37 = '-2'
    var_38 = '-1.0000000000005'
    var_39 = '-1.000000000000'
    var_40 = '-1.0000000000015'
    var_41 = '-1.000000000002'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 3
    var_7 = 'US Dollars '



# Parsed testcases at query #46
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
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = len(var_0)
    assert var_8 == 2
    var_9 = 'USD'
    var_10 = var_0.has(var_9)
    var_11 = 'EUR'
    var_12 = var_0.get(var_11)
    var_13 = 'NONEXISTENT'
    var_14 = var_0.get(var_13)
    assert var_14 is None
    var_15 = 'NONEXISTENT'
    var_16 = var_0[var_15]



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'not a currency'



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars Different'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #51
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
    var_12 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache="



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 3
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #55
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



# Parsed testcases at query #56
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



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = ''
    var_3 = 123



# Parsed testcases at query #58
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2
    var_4 = module_0.CurrencyRegistry()
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #59
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #60
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
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = len(var_0)
    assert var_8 == 2
    var_9 = 'USD'
    var_10 = var_0.has(var_9)
    var_11 = 'XYZ'
    var_12 = var_0.has(var_11)
    var_13 = var_0.get(var_9)
    var_14 = var_0.get(var_11)
    assert var_14 is None
    var_15 = 'XYZ'
    var_16 = var_0[var_15]



# Parsed testcases at query #61
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



# Parsed testcases at query #62
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
    var_7 = var_0.has(var_6)
    assert var_7 is True
    var_8 = 'EUR'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = 'NONEXISTENT'
    var_13 = var_0.has(var_12)
    assert var_13 is False
    var_14 = 'usd'
    var_15 = var_0.has(var_14)
    assert var_15 is False
    var_16 = 'eur'
    var_17 = var_0.has(var_16)
    assert var_17 is False



# Parsed testcases at query #63
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'NON_EXISTENT'
    var_2 = var_0.has(var_1)
    var_3 = 'USD'
    var_4 = 'US Dollar'
    var_5 = 2
    var_6 = 'USD'
    var_7 = var_0.has(var_6)
    var_8 = 'EUR'
    var_9 = var_0.has(var_8)



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 0



# Parsed testcases at query #65
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
    var_12 = 'DEF'
    var_13 = 'Default'
    var_14 = 2



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})"
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1
    var_11 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})"



# Parsed testcases at query #67
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



# Parsed testcases at query #68
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
    var_19 = '1.9'
    var_20 = 'ZZZ'
    var_21 = 'Some weird currency'
    var_22 = -1
    var_23 = '1.0000000000005'
    var_24 = '1.000000000000'
    var_25 = '1.0000000000015'
    var_26 = '1.000000000002'
    var_27 = '1.0000000000000'
    var_28 = '1.0000000000009'
    var_29 = '1.000000000001'
    var_30 = '-1.005'
    var_31 = '-1.00'
    var_32 = '-1.5'
    var_33 = '-2'
    var_34 = '-1.0000000000005'
    var_35 = '-1.000000000000'
    var_36 = '999999999.995'
    var_37 = '999999999.99'
    var_38 = '999999999.5'
    var_39 = '999999999'
    var_40 = '999999999.0000000000005'
    var_41 = '999999999.000000000000'



# Parsed testcases at query #69
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
    var_8 = 'US Dollars Different'
    var_9 = 3



# Parsed testcases at query #70
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



# Parsed testcases at query #71
#--------------------------


def test_case_0():
    var_0 = 'NON-EXISTING'



# Parsed testcases at query #72
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = module_0.CurrencyRegistry()



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #74
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = 'UX Dollars'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = '0.5'
    var_12 = '0'
    var_13 = '1.5'
    var_14 = '2'
    var_15 = 'ZZZ'
    var_16 = 'Some weird currency'
    var_17 = -1
    var_18 = '1.0000000000005'
    var_19 = '1.000000000000'
    var_20 = '1.0000000000015'
    var_21 = '1.000000000002'
    var_22 = '123'
    var_23 = 'Invalid Code'
    var_24 = 2
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD '
    var_29 = 'Code with Space'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = ' '
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



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'
    var_7 = 'XAU'
    var_8 = 'Gold'



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = '0.01'



# Parsed testcases at query #77
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 3
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'US Dollars Different'
    var_7 = 0



# Parsed testcases at query #78
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



# Parsed testcases at query #79
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"



# Parsed testcases at query #80
#--------------------------


def test_case_0():
    var_0 = 'NON-EXISTING'
    var_1 = str(var_0)
    assert var_1 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #81
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'AED'
    var_2 = 'UAE Dirham'
    var_3 = 2
    var_4 = 'USD'
    var_5 = 'US Dollar'
    var_6 = 'EUR'
    var_7 = 'Euro'



# Parsed testcases at query #82
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



# Parsed testcases at query #83
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = CurrencyRegistry()[var_0]
    var_2 = 'NON-EXISTING'
    var_3 = CurrencyRegistry()[var_2]
    var_4 = str(var_3)
    assert var_4 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #84
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #85
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #86
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #87
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})"
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1
    var_11 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})"



# Parsed testcases at query #88
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
    var_8 = 'XYZ'
    var_9 = var_0.get(var_8)
    assert var_9 is None
    var_10 = 'GBP'
    var_11 = 'British Pound'
    var_12 = 2
    var_13 = 'EUR'



# Parsed testcases at query #89
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



# Parsed testcases at query #90
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #91
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = CurrencyRegistry()[var_0]
    var_2 = 'NON-EXISTING'
    var_3 = CurrencyRegistry()[var_2]
    var_4 = str(var_3)
    assert var_4 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #92
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = 'EUR'
    var_3 = 'ANOTHERNONEXISTENT'



# Parsed testcases at query #93
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



# Parsed testcases at query #94
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #95
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #96
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #97
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
    var_21 = '-1.005'
    var_22 = '-1.00'
    var_23 = '-1.5'
    var_24 = '-2'
    var_25 = '0.00'



# Parsed testcases at query #98
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
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = len(var_0)
    assert var_8 == 1
    var_9 = len(var_0)
    assert var_9 == 2
    var_10 = len(var_0)
    assert var_10 == 2
    var_11 = 'USD'
    var_12 = var_0.has(var_11)
    var_13 = var_0.get(var_11)
    var_14 = 'XXX'
    var_15 = var_0.get(var_14)
    assert var_15 is None
    var_16 = 'XXX'
    var_17 = var_0[var_16]



# Parsed testcases at query #99
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
    var_19 = '9.9'
    var_20 = '10'
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
    var_31 = '-1.005'
    var_32 = '-1.00'
    var_33 = '-1.015'
    var_34 = '-1.02'
    var_35 = '-0.5'
    var_36 = '-1.5'
    var_37 = '-2'
    var_38 = '-1.0000000000005'
    var_39 = '-1.000000000000'
    var_40 = '-1.0000000000015'
    var_41 = '-1.000000000002'



# Parsed testcases at query #100
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"



# Parsed testcases at query #101
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'NONEXISTENT'
    var_7 = var_0[var_6]



# Parsed testcases at query #102
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = var_0.has(var_1)
    assert var_2 is True
    var_3 = module_0.CurrencyRegistry()
    var_4 = 'NONEXISTENT'
    var_5 = var_3.has(var_4)
    assert var_5 is False
    var_6 = module_0.CurrencyRegistry()
    var_7 = ''
    var_8 = var_6.has(var_7)
    assert var_8 is False
    var_9 = module_0.CurrencyRegistry()
    var_10 = None
    var_11 = var_9.has(var_10)
    assert var_11 is False



# Parsed testcases at query #103
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



# Parsed testcases at query #104
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'



# Parsed testcases at query #105
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #106
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #107
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



# Parsed testcases at query #108
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache="
    var_4 = '0.01'
    var_5 = ')'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache="
    var_10 = '0'
    var_11 = 'ZZZ'
    var_12 = 'Some weird currency'
    var_13 = -1
    var_14 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache="
    var_15 = -1



# Parsed testcases at query #109
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



# Parsed testcases at query #110
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 3
    var_7 = '0.01'



# Parsed testcases at query #111
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



# Parsed testcases at query #112
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



# Parsed testcases at query #113
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #114
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



# Parsed testcases at query #115
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'



# Parsed testcases at query #116
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
    var_10 = 'NONEXISTENT'
    var_11 = var_0.get(var_10)
    assert var_11 is None
    var_12 = 'NONEXISTENT'
    var_13 = var_0[var_12]



# Parsed testcases at query #117
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.001'



# Parsed testcases at query #118
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 3
    var_7 = -1



# Parsed testcases at query #119
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #120
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = 'XAU'
    var_11 = 'Gold'



# Parsed testcases at query #121
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #122
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'ABC'
    var_2 = 'Test Currency 1'
    var_3 = 2
    var_4 = 'DEF'
    var_5 = 'Test Currency 2'
    var_6 = 0
    var_7 = 'GHI'
    var_8 = 'Test Currency 3'
    var_9 = 4
    var_10 = 'JKL'
    var_11 = 'Test Currency 4'
    var_12 = 2



# Parsed testcases at query #123
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 3



# Parsed testcases at query #124
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #125
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



# Parsed testcases at query #126
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



# Parsed testcases at query #127
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD!'
    var_29 = 'Special Char Code'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = '  Trimmed Name  '
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = -2
    var_40 = 'USD'
    var_41 = 'US Dollars'
    var_42 = '2'
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = 2
    var_46 = 'MONEY'



# Parsed testcases at query #128
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
    var_19 = '999.9'
    var_20 = '1000'
    var_21 = 'ZZZ'
    var_22 = 'Some weird currency'
    var_23 = -1
    var_24 = '1.0000000000005'
    var_25 = '1.000000000000'
    var_26 = '1.0000000000015'
    var_27 = '1.000000000002'
    var_28 = '1.0000000000000'
    var_29 = '1.0000000000001'
    var_30 = 'EUR'
    var_31 = 'Euro'
    var_32 = '0.00'
    var_33 = '0.001'
    var_34 = '0.009'
    var_35 = '0.01'
    var_36 = '999999999999.999'
    var_37 = '1000000000000.00'



# Parsed testcases at query #129
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD$'
    var_29 = 'Special Char Code'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = '  Trimmed  '
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = -2
    var_40 = 'USD'
    var_41 = 'US Dollars'
    var_42 = '2'
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = 2
    var_46 = 'MONEY'



# Parsed testcases at query #130
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



# Parsed testcases at query #131
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #132
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
    var_9 = 'EUR'
    var_10 = 'Euro'
    var_11 = 2
    var_12 = len(var_0)
    assert var_12 == 2
    var_13 = 'EUR'
    var_14 = var_0.has(var_13)
    assert var_14 is True
    var_15 = var_0.get(var_11)
    var_16 = 'XYZ'
    var_17 = var_0.get(var_16)
    assert var_17 is None
    var_18 = 'XYZ'
    var_19 = var_0[var_18]



# Parsed testcases at query #133
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
    var_7 = var_0.has(var_6)
    assert var_7 is True
    var_8 = 'EUR'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = ''
    var_13 = var_0.has(var_12)
    assert var_13 is False



# Parsed testcases at query #134
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'



# Parsed testcases at query #135
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
    var_23 = '123'
    var_24 = 'Invalid Code'
    var_25 = 2
    var_26 = 'usd'
    var_27 = 'Lowercase Code'
    var_28 = 2
    var_29 = 'USD 1'
    var_30 = 'Code with space'
    var_31 = 2
    var_32 = 'USD'
    var_33 = ''
    var_34 = 2
    var_35 = 'USD'
    var_36 = '  Trimmed Name  '
    var_37 = 2
    var_38 = 'USD'
    var_39 = 'US Dollars'
    var_40 = -2
    var_41 = 'USD'
    var_42 = 'US Dollars'
    var_43 = '2'
    var_44 = 'USD'
    var_45 = 'US Dollars'
    var_46 = 2
    var_47 = 'MONEY'



# Parsed testcases at query #136
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD '
    var_29 = 'Code with space'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = ' '
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



# Parsed testcases at query #137
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD$'
    var_29 = 'Special Char Code'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = '  Trim Me  '
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = -2
    var_40 = 'USD'
    var_41 = 'US Dollars'
    var_42 = '2'
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = 2
    var_46 = 'MONEY'



# Parsed testcases at query #138
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #139
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



# Parsed testcases at query #140
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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'UX Dollars'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8
    var_12 = 'not a currency'



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
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = len(var_0)
    assert var_10 == 3



# Parsed testcases at query #4
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
    var_13 = '1.005'
    var_14 = '1.00'
    var_15 = '1.015'
    var_16 = '1.02'
    var_17 = 'usd'
    var_18 = 'US Dollars'
    var_19 = 2
    var_20 = '123'
    var_21 = 'US Dollars'
    var_22 = 2
    var_23 = 'USD '
    var_24 = 'US Dollars'
    var_25 = 2
    var_26 = 'USD'
    var_27 = ''
    var_28 = 2
    var_29 = 'USD'
    var_30 = ' US Dollars'
    var_31 = 2
    var_32 = 'USD'
    var_33 = 'US Dollars '
    var_34 = 2
    var_35 = 'USD'
    var_36 = 'US Dollars'
    var_37 = -2
    var_38 = 'USD'
    var_39 = 'US Dollars'
    var_40 = '2'
    var_41 = 'USD'
    var_42 = 'US Dollars'
    var_43 = 2
    var_44 = 'MONEY'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache="
    var_4 = '0.01'
    var_5 = ')'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache="
    var_10 = '0'
    var_11 = 'ZZZ'
    var_12 = 'Some weird currency'
    var_13 = -1
    var_14 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer="
    var_15 = ', hashcache='
    var_16 = -1



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'US Dollars 2'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8
    var_12 = 'ZZZ'
    var_13 = 'Weird'
    var_14 = -1



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #9
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = 'TEST'
    var_4 = 'Test Currency'
    var_5 = 2



# Parsed testcases at query #10
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2
    var_4 = var_0.has(var_1)
    var_5 = var_0.codes
    var_6 = sorted(var_5)
    var_7 = var_0.codenames
    var_8 = 0
    var_9 = lambda x: x[var_8]
    var_10 = sorted(var_7, key=var_9)
    var_11 = var_0.all
    var_12 = lambda x: x.code
    var_13 = sorted(var_11, key=var_12)



# Parsed testcases at query #11
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #12
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'AED'
    var_2 = 'UAE Dirham'
    var_3 = 2
    var_4 = 'BHD'
    var_5 = 'Bahraini Dinar'
    var_6 = 3
    var_7 = 'CUC'
    var_8 = 'Cuban Convertible Peso'
    var_9 = len(var_0)
    assert var_9 == 3
    var_10 = 'AED'
    var_11 = 'UAE Dirham'
    var_12 = 2
    var_13 = 'BHD'
    var_14 = 'Bahraini Dinar'
    var_15 = 3
    var_16 = 'CUC'
    var_17 = 'Cuban Convertible Peso'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #14
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST1'
    var_2 = 'Test Currency 1'
    var_3 = 2
    var_4 = 'TEST2'
    var_5 = 'Test Currency 2'
    var_6 = 0
    var_7 = 'TEST3'
    var_8 = 'Test Currency 3'
    var_9 = -1
    var_10 = len(var_0)
    assert var_10 == 3
    var_11 = 'TEST4'
    var_12 = 'Test Currency 4'
    var_13 = 2



# Parsed testcases at query #15
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
    var_11 = 'US Dollars (Different)'



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
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 3



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #19
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TST'
    var_2 = 'Test Currency 1'
    var_3 = 2
    var_4 = 'TST2'
    var_5 = 'Test Currency 2'
    var_6 = 0
    var_7 = len(var_0)
    assert var_7 == 2
    var_8 = 'AAA'
    var_9 = 'Test Currency 3'
    var_10 = 1



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8



# Parsed testcases at query #21
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #22
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
    var_17 = '0.0'
    var_18 = '999.9'
    var_19 = '1000'
    var_20 = 'ZZZ'
    var_21 = 'Some weird currency'
    var_22 = -1
    var_23 = '1.0000000000005'
    var_24 = '1.000000000000'
    var_25 = '1.0000000000015'
    var_26 = '1.000000000002'
    var_27 = '1.0000000000000'
    var_28 = '1.0000000000001'
    var_29 = '-1.005'
    var_30 = '-1.00'
    var_31 = '-1.5'
    var_32 = '-2'
    var_33 = '-1.0000000000005'
    var_34 = '-1.000000000000'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 3



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})"
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1
    var_11 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})"



# Parsed testcases at query #25
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TST'
    var_2 = 'Test Currency 1'
    var_3 = 2
    var_4 = 'TST2'
    var_5 = 'Test Currency 2'
    var_6 = 0
    var_7 = len(var_0)
    assert var_7 == 2
    var_8 = 'AAA'
    var_9 = 'Test Currency 3'



# Parsed testcases at query #26
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
    var_23 = 123
    var_24 = 'US Dollars'
    var_25 = 2
    var_26 = 'usd'
    var_27 = 'US Dollars'
    var_28 = 2
    var_29 = 'USD1'
    var_30 = 'US Dollars'
    var_31 = 2
    var_32 = 'USD'
    var_33 = ''
    var_34 = 2
    var_35 = 'USD'
    var_36 = '  US Dollars'
    var_37 = 2
    var_38 = 'USD'
    var_39 = 'US Dollars  '
    var_40 = 2
    var_41 = 'USD'
    var_42 = 'US Dollars'
    var_43 = '2'
    var_44 = 'USD'
    var_45 = 'US Dollars'
    var_46 = -2
    var_47 = 'USD'
    var_48 = 'US Dollars'
    var_49 = 2
    var_50 = 'MONEY'



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
    var_7 = len(var_0)
    assert var_7 == 2



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #29
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



# Parsed testcases at query #30
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
    var_7 = len(var_0)
    assert var_7 == 1
    var_8 = 'USD'
    var_9 = var_0.has(var_8)
    var_10 = 'EUR'
    var_11 = var_0.has(var_10)
    var_12 = var_0.get(var_8)
    var_13 = var_0.get(var_10)
    assert var_13 is None
    var_14 = 'EUR'
    var_15 = var_0[var_14]



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'



# Parsed testcases at query #32
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



# Parsed testcases at query #33
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"
    var_3 = ''
    var_4 = module_0.CurrencyLookupError(var_3)
    var_5 = str(var_4)
    assert var_5 == "Currency identified by code '' does not exist"
    var_6 = 'A$C'
    var_7 = module_0.CurrencyLookupError(var_6)
    var_8 = str(var_7)
    assert var_8 == "Currency identified by code 'A$C' does not exist"



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 3



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 'BTC'
    var_7 = 'Bitcoin'
    var_8 = 8



# Parsed testcases at query #36
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
    var_7 = var_0.has(var_6)
    assert var_7 is True
    var_8 = 'EUR'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = ''
    var_13 = var_0.has(var_12)
    assert var_13 is False
    var_14 = 'usd'
    var_15 = var_0.has(var_14)
    assert var_15 is False
    var_16 = 'eur'
    var_17 = var_0.has(var_16)
    assert var_17 is False



# Parsed testcases at query #37
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'NON-EXISTING'
    var_5 = var_0[var_4]



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'ZZZ'
    var_10 = 'Some weird currency'
    var_11 = -1



# Parsed testcases at query #39
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2



# Parsed testcases at query #40
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
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = len(var_0)
    assert var_8 == 0
    var_9 = len(var_0)
    assert var_9 == 2
    var_10 = 'USD'
    var_11 = var_0.get(var_10)
    var_12 = 'NONEXISTENT'
    var_13 = var_0.get(var_12)
    assert var_13 is None
    var_14 = 'NONEXISTENT'
    var_15 = var_0[var_14]



# Parsed testcases at query #41
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
    var_28 = 'U$D'
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
    var_41 = 'US Dollars'
    var_42 = -2
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = '2'
    var_46 = 'USD'
    var_47 = 'US Dollars'
    var_48 = 2
    var_49 = 'Money'



# Parsed testcases at query #42
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
    var_7 = len(var_0)
    assert var_7 == 1
    var_8 = 'USD'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = var_0.get(var_8)
    var_13 = var_0.get(var_10)
    assert var_13 is None
    var_14 = 'XYZ'
    var_15 = var_0[var_14]



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #45
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #46
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
    var_7 = 'TEST'
    var_8 = var_0.has(var_7)
    var_9 = 'NONEXISTENT'
    var_10 = var_0.has(var_9)
    var_11 = var_0.get(var_7)
    var_12 = var_0.get(var_9)
    assert var_12 is None
    var_13 = 'NONEXISTENT'
    var_14 = var_0[var_13]



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.001'



# Parsed testcases at query #48
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
    var_23 = 'Invalid'
    var_24 = 2
    var_25 = 'abc'
    var_26 = 'Invalid'
    var_27 = 2
    var_28 = 'Abc'
    var_29 = 'Invalid'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = ' '
    var_36 = 2
    var_37 = 'USD'
    var_38 = ' Invalid'
    var_39 = 2
    var_40 = 'USD'
    var_41 = 'Invalid '
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



# Parsed testcases at query #49
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'



# Parsed testcases at query #50
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



# Parsed testcases at query #51
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
    var_7 = 'TEST'
    var_8 = var_0.has(var_7)
    assert var_8 is True
    var_9 = 'NONEXISTENT'
    var_10 = var_0.has(var_9)
    assert var_10 is False
    var_11 = var_0.get(var_7)
    var_12 = var_0.get(var_9)
    assert var_12 is None
    var_13 = 'NONEXISTENT'
    var_14 = var_0[var_13]



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'Different Name'
    var_4 = 3
    var_5 = 'EUR'
    var_6 = 'Euro'



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = '0.01'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})"
    var_9 = '0'
    var_10 = 'ZZZ'
    var_11 = 'Some weird currency'
    var_12 = -1
    var_13 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})"
    var_14 = -1



# Parsed testcases at query #54
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



# Parsed testcases at query #55
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



# Parsed testcases at query #56
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



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #58
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = CurrencyRegistry()[var_0]
    var_2 = 'NON-EXISTING'
    var_3 = CurrencyRegistry()[var_2]
    var_4 = str(var_3)
    assert var_4 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #59
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #60
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
    var_28 = 'U$D'
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
    var_41 = 'US Dollars'
    var_42 = -2
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = '2'
    var_46 = 'USD'
    var_47 = 'US Dollars'
    var_48 = 2
    var_49 = 'MONEY'



# Parsed testcases at query #61
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



# Parsed testcases at query #62
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
    var_11 = 'ZZZ'
    var_12 = 'Some weird currency'
    var_13 = -1



# Parsed testcases at query #63
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #65
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #66
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



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = '1.001'
    var_8 = '1.009'
    var_9 = '1.01'
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0
    var_13 = '0.5'
    var_14 = '0'
    var_15 = '1.5'
    var_16 = '2'
    var_17 = '1.0'
    var_18 = '1'
    var_19 = '1.4'
    var_20 = '1.6'
    var_21 = 'ZZZ'
    var_22 = 'Some weird currency'
    var_23 = -1
    var_24 = '1.0000000000005'
    var_25 = '1.000000000000'
    var_26 = '1.0000000000015'
    var_27 = '1.000000000002'
    var_28 = '1.0000000000001'
    var_29 = '1.0000000000009'
    var_30 = '1.000000000001'
    var_31 = 'TND'
    var_32 = 'Tunisian Dinar'
    var_33 = 3
    var_34 = '1.0005'
    var_35 = '1.000'
    var_36 = '1.0015'
    var_37 = '1.002'
    var_38 = '1.0001'
    var_39 = '1.0009'



# Parsed testcases at query #68
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



# Parsed testcases at query #69
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



# Parsed testcases at query #70
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #71
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST1'
    var_2 = 'Test Currency 1'
    var_3 = 2
    var_4 = 'TEST2'
    var_5 = 'Test Currency 2'
    var_6 = 0
    var_7 = len(var_0)
    assert var_7 == 2
    var_8 = 'AAA'
    var_9 = 'Test Currency AAA'
    var_10 = 2
    var_11 = 'ZZZ'
    var_12 = 'Test Currency ZZZ'
    var_13 = 0
    var_14 = 'FAIL'
    var_15 = 'Should Fail'
    var_16 = 2
    var_17 = 'EXC'
    var_18 = 'Exception Test'
    var_19 = 2
    var_20 = 'Test exception'
    var_21 = ValueError(var_20)



# Parsed testcases at query #72
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
    var_8 = 'XYZ'
    var_9 = var_0[var_8]



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8
    var_12 = 'ZZZ'
    var_13 = 'Some weird currency'
    var_14 = -1



# Parsed testcases at query #74
#--------------------------




# Parsed testcases at query #75
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



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #77
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #80
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
    var_7 = len(var_0)
    assert var_7 == 1
    var_8 = 'USD'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = var_0.get(var_8)
    var_13 = var_0.get(var_10)
    assert var_13 is None
    var_14 = 'XYZ'
    var_15 = var_0[var_14]



# Parsed testcases at query #81
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
    var_22 = 123
    var_23 = 'US Dollars'
    var_24 = 2
    var_25 = 'usd'
    var_26 = 'US Dollars'
    var_27 = 2
    var_28 = 'USD1'
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



# Parsed testcases at query #82
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #83
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST1'
    var_2 = 'Test Currency 1'
    var_3 = 2
    var_4 = 'TEST2'
    var_5 = 'Test Currency 2'
    var_6 = 0
    var_7 = 'TEST3'
    var_8 = 'Test Currency 3'
    var_9 = -1
    var_10 = len(var_0)
    assert var_10 == 3
    var_11 = 'AAA'
    var_12 = 'Unsorted Currency'



# Parsed testcases at query #84
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
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1



# Parsed testcases at query #85
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



# Parsed testcases at query #86
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #87
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2



# Parsed testcases at query #88
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
    var_23 = '123'
    var_24 = 'Invalid Code'
    var_25 = 2
    var_26 = 'abc'
    var_27 = 'Invalid Code'
    var_28 = 2
    var_29 = 'Abc'
    var_30 = 'Invalid Code'
    var_31 = 2
    var_32 = 'USD'
    var_33 = ''
    var_34 = 2
    var_35 = 'USD'
    var_36 = '  Invalid Name  '
    var_37 = 2
    var_38 = 'USD'
    var_39 = 'US Dollars'
    var_40 = -2
    var_41 = 'USD'
    var_42 = 'US Dollars'
    var_43 = '2'
    var_44 = 'USD'
    var_45 = 'US Dollars'
    var_46 = 2
    var_47 = 'MONEY'



# Parsed testcases at query #89
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'XAU'
    var_6 = 'Gold'
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8
    var_10 = 0
    var_11 = 'JPY'
    var_12 = 'Japanese Yen'



# Parsed testcases at query #90
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8
    var_10 = 'ZZZ'
    var_11 = 'Some weird currency'
    var_12 = -1



# Parsed testcases at query #91
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #92
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #93
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = '1.234567'
    var_8 = '1.23'
    var_9 = 'JPY'
    var_10 = 'Japanese Yen'
    var_11 = 0
    var_12 = '0.5'
    var_13 = '0'
    var_14 = '1.5'
    var_15 = '2'
    var_16 = '123.456'
    var_17 = '123'
    var_18 = 'ZZZ'
    var_19 = 'Some weird currency'
    var_20 = -1
    var_21 = '1.0000000000005'
    var_22 = '1.000000000000'
    var_23 = '1.0000000000015'
    var_24 = '1.000000000002'
    var_25 = '1.12345678901234567890123456789'
    var_26 = '-1.005'
    var_27 = '-1.00'
    var_28 = '-1.5'
    var_29 = '-2'
    var_30 = '-1.0000000000005'
    var_31 = '-1.000000000000'
    var_32 = '0.00'



# Parsed testcases at query #94
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
    var_8 = var_7.code
    assert var_8 == 'USD'
    var_9 = var_0.get(var_6)
    var_10 = var_9.name
    assert var_10 == 'US Dollar'
    var_11 = 'XYZ'
    var_12 = var_0.get(var_11)
    assert var_12 is None
    var_13 = 'GBP'
    var_14 = 'British Pound'
    var_15 = 2
    var_16 = 'EUR'



# Parsed testcases at query #95
#--------------------------


def test_case_0():
    var_0 = 'TEST'
    var_1 = 'Test Currency'
    var_2 = 2
    var_3 = 'NONEXISTENT'
    var_4 = 'DEFAULT'
    var_5 = 'Default Currency'
    var_6 = 0



# Parsed testcases at query #96
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
    var_10 = 'XXX'
    var_11 = var_0.get(var_10)
    assert var_11 is None
    var_12 = 'XXX'
    var_13 = var_0[var_12]



# Parsed testcases at query #97
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = var_0.__enter__()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = None
    var_8 = var_0.__exit__(var_7, var_7, var_7)



# Parsed testcases at query #98
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TST'
    var_2 = 'Test Currency 1'
    var_3 = 2
    var_4 = 'TST2'
    var_5 = 'Test Currency 2'
    var_6 = 0
    var_7 = len(var_0)
    assert var_7 == 2
    var_8 = 'TST3'
    var_9 = 'Test Currency 3'
    var_10 = 3



# Parsed testcases at query #99
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = 'TEST1'
    var_3 = 'Test Currency 1'
    var_4 = 2
    var_5 = 'TEST2'
    var_6 = 'Test Currency 2'
    var_7 = 0
    var_8 = len(var_0)
    assert var_8 == 2



# Parsed testcases at query #100
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #101
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



# Parsed testcases at query #102
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #103
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #104
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



# Parsed testcases at query #105
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"



# Parsed testcases at query #106
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
    var_19 = '9.9'
    var_20 = '10'
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
    var_31 = '-1.005'
    var_32 = '-1.00'
    var_33 = '-1.015'
    var_34 = '-1.02'
    var_35 = '-0.5'
    var_36 = '-1.5'
    var_37 = '-2'
    var_38 = '-1.0000000000005'
    var_39 = '-1.000000000000'
    var_40 = '-1.0000000000015'
    var_41 = '-1.000000000002'



# Parsed testcases at query #107
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #108
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #109
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"



# Parsed testcases at query #110
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = ''
    var_3 = 123



# Parsed testcases at query #111
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2
    var_4 = var_0.codes
    var_5 = sorted(var_4)
    var_6 = var_0.all
    var_7 = lambda x: x.code
    var_8 = sorted(var_6, key=var_7)
    var_9 = var_0.codenames
    var_10 = 0
    var_11 = lambda x: x[var_10]
    var_12 = sorted(var_9, key=var_11)



# Parsed testcases at query #112
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'AAA'
    var_6 = 'BBB'
    var_7 = 1
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0



# Parsed testcases at query #113
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = '0.01'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})"
    var_9 = '0'
    var_10 = 'ZZZ'
    var_11 = 'Some weird currency'
    var_12 = -1
    var_13 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})"
    var_14 = -1



# Parsed testcases at query #114
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"
    var_3 = 'ABC'
    var_4 = module_0.CurrencyLookupError(var_3)



# Parsed testcases at query #115
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
    var_10 = 'XXX'
    var_11 = var_0.get(var_10)
    assert var_11 is None
    var_12 = 'YYY'



# Parsed testcases at query #116
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



# Parsed testcases at query #117
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
    var_10 = 'XXX'
    var_11 = var_0.get(var_10)
    assert var_11 is None
    var_12 = 'NON-EXISTING'
    var_13 = var_0.get(var_12)
    assert var_13 is None



# Parsed testcases at query #118
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #119
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)
    assert var_2 == "Currency identified by code 'XYZ' does not exist"



# Parsed testcases at query #120
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #121
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



# Parsed testcases at query #122
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2
    var_4 = 'AAA'
    var_5 = 'First Currency'
    var_6 = 2
    var_7 = 'ZZZ'
    var_8 = 'Last Currency'



# Parsed testcases at query #123
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #124
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = ''
    var_3 = 'usd'



# Parsed testcases at query #125
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
    var_7 = var_0.has(var_6)
    assert var_7 is True
    var_8 = 'EUR'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = 'NONEXISTENT'
    var_13 = var_0.has(var_12)
    assert var_13 is False
    var_14 = 'usd'
    var_15 = var_0.has(var_14)
    assert var_15 is False
    var_16 = 'eur'
    var_17 = var_0.has(var_16)
    assert var_17 is False



# Parsed testcases at query #126
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = module_0.CurrencyRegistry()



# Parsed testcases at query #127
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #128
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



# Parsed testcases at query #129
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
    var_10 = 'XXX'
    var_11 = var_0.get(var_10)
    assert var_11 is None
    var_12 = 'XXX'
    var_13 = var_0[var_12]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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
    var_1 = 'US Dollar'
    var_2 = 2
    var_3 = 'NON-EXISTING'



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
    var_6 = 'NON-EXISTING'
    var_7 = var_0[var_6]
    var_8 = str(var_6)
    assert var_8 == "Currency identified by code 'NON-EXISTING' does not exist"



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
    var_0 = 'TEST'
    var_1 = 'Test Currency'
    var_2 = 2
    var_3 = CurrencyRegistry()[var_0]
    var_4 = 'NONEXISTENT'
    var_5 = CurrencyRegistry()[var_4]



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
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = '0.01'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})"
    var_9 = '0'
    var_10 = 'ZZZ'
    var_11 = 'Some weird currency'
    var_12 = -1
    var_13 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})"
    var_14 = -1



# Parsed testcases at query #12
#--------------------------


import pypara.commons.numbers as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = module_0.make_quantizer(var_2)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = 'XAU'
    var_11 = 'Gold'



# Parsed testcases at query #14
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
    var_35 = ' '
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



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXIST'
    var_2 = ''
    var_3 = 'USD!'
    var_4 = 'USD '



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
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 0



# Parsed testcases at query #19
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
    var_7 = 'UX Dollars'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = '0.5'
    var_12 = '0'
    var_13 = '1.5'
    var_14 = '2'
    var_15 = 'ZZZ'
    var_16 = 'Some weird currency'
    var_17 = -1
    var_18 = '1.0000000000005'
    var_19 = '1.000000000000'
    var_20 = '1.0000000000015'
    var_21 = '1.000000000002'
    var_22 = '123'
    var_23 = 'Invalid Code'
    var_24 = 2
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD '
    var_29 = 'Code with Space'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = '  Trim Me  '
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = -2
    var_40 = 'USD'
    var_41 = 'US Dollars'
    var_42 = '2'
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = 2
    var_46 = 'MONEY'



# Parsed testcases at query #21
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
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #23
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
    var_23 = '123'
    var_24 = 'Invalid Code'
    var_25 = 2
    var_26 = 'usd'
    var_27 = 'Lowercase Code'
    var_28 = 2
    var_29 = 'USD '
    var_30 = 'Code with Space'
    var_31 = 2
    var_32 = 'USD'
    var_33 = ''
    var_34 = 2
    var_35 = 'USD'
    var_36 = ' '
    var_37 = 2
    var_38 = 'USD'
    var_39 = ' Leading Space'
    var_40 = 2
    var_41 = 'USD'
    var_42 = 'Trailing Space '
    var_43 = 2
    var_44 = 'USD'
    var_45 = 'US Dollars'
    var_46 = -2
    var_47 = 'USD'
    var_48 = 'US Dollars'
    var_49 = '2'
    var_50 = 'USD'
    var_51 = 'US Dollars'
    var_52 = 2
    var_53 = 'MONEY'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.0001'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0



# Parsed testcases at query #26
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 3



# Parsed testcases at query #28
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
    var_10 = str(var_2)
    assert var_10 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #29
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
    var_7 = var_0.has(var_6)
    assert var_7 is True
    var_8 = 'EUR'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = ''
    var_13 = var_0.has(var_12)
    assert var_13 is False



# Parsed testcases at query #30
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #31
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


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



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
    var_52 = 'Money'



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.001'



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'AAA'
    var_6 = 'BBB'
    var_7 = 1
    var_8 = 3
    var_9 = 'ZZZ'
    var_10 = 'Weird'
    var_11 = -1
    var_12 = 0



# Parsed testcases at query #38
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
    var_19 = '9.9'
    var_20 = '10'
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
    var_31 = '-1.005'
    var_32 = '-1.00'
    var_33 = '-1.015'
    var_34 = '-1.02'
    var_35 = '-0.5'
    var_36 = '-1.5'
    var_37 = '-2'
    var_38 = '-1.0000000000005'
    var_39 = '-1.000000000000'
    var_40 = '-1.0000000000015'
    var_41 = '-1.000000000002'



# Parsed testcases at query #39
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



# Parsed testcases at query #40
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



# Parsed testcases at query #41
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #44
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST'
    var_2 = 'Test Currency'
    var_3 = 2
    var_4 = module_0.CurrencyRegistry()
    var_5 = 'ZAR'
    var_6 = 'South African Rand'
    var_7 = 2
    var_8 = 'AED'
    var_9 = 'UAE Dirham'
    var_10 = 'MAD'
    var_11 = 'Moroccan Dirham'



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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
    var_12 = 'NON-EXISTING'
    var_13 = var_0.get(var_12)
    assert var_13 is None



# Parsed testcases at query #48
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #49
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
    var_7 = len(var_0)
    assert var_7 == 1
    var_8 = 'USD'
    var_9 = var_0.has(var_8)
    var_10 = 'EUR'
    var_11 = var_0.has(var_10)
    var_12 = var_0.get(var_8)
    var_13 = var_0.get(var_10)
    assert var_13 is None
    var_14 = 'EUR'
    var_15 = var_0[var_14]



# Parsed testcases at query #50
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



# Parsed testcases at query #51
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
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = len(var_0)
    assert var_8 == 2
    var_9 = 'USD'
    var_10 = var_0.has(var_9)
    var_11 = 'XYZ'
    var_12 = var_0.has(var_11)
    var_13 = var_0.get(var_9)
    var_14 = var_0.get(var_11)
    assert var_14 is None
    var_15 = 'XYZ'
    var_16 = var_0[var_15]



# Parsed testcases at query #52
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
    var_28 = '1.0000000000000'
    var_29 = '1.0000000000001'
    var_30 = '-1.005'
    var_31 = '-1.00'
    var_32 = '-1.015'
    var_33 = '-1.02'
    var_34 = '-0.5'
    var_35 = '-1.5'
    var_36 = '-2'
    var_37 = '-1.0000000000005'
    var_38 = '-1.000000000000'



# Parsed testcases at query #53
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
    var_7 = var_0.has(var_6)
    assert var_7 is True
    var_8 = 'EUR'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = ''
    var_13 = var_0.has(var_12)
    assert var_13 is False



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars Different'
    var_6 = 3



# Parsed testcases at query #56
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



# Parsed testcases at query #57
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



# Parsed testcases at query #58
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'TEST1'
    var_2 = 'Test Currency 1'
    var_3 = 2
    var_4 = 'TEST2'
    var_5 = 'Test Currency 2'
    var_6 = 0
    var_7 = var_0.has(var_1)
    var_8 = var_0.has(var_4)
    var_9 = len(var_0)
    assert var_9 == 2
    var_10 = 'AAA'
    var_11 = 'First Currency'
    var_12 = 'ZZZ'
    var_13 = 'Last Currency'
    var_14 = 3



# Parsed testcases at query #59
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



# Parsed testcases at query #60
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
    var_9 = 'YYY'
    var_10 = 'Another weird currency'
    var_11 = 'XAU'
    var_12 = 'Gold'
    var_13 = 'ABC'
    var_14 = 'Alternative Currency'



# Parsed testcases at query #61
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #62
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
    var_12 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache="



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.001'



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #65
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



# Parsed testcases at query #66
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 3
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8



# Parsed testcases at query #67
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
    var_31 = '-1.005'
    var_32 = '-1.00'
    var_33 = '-1.5'
    var_34 = '-2'
    var_35 = '-1.0000000000005'
    var_36 = '-1.000000000000'
    var_37 = '0.00'



# Parsed testcases at query #68
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
    var_8 = var_7.code
    assert var_8 == 'USD'
    var_9 = var_0.get(var_6)
    var_10 = var_9.name
    assert var_10 == 'US Dollar'
    var_11 = 'XYZ'
    var_12 = var_0.get(var_11)
    assert var_12 is None
    var_13 = 'GBP'
    var_14 = 'British Pound'
    var_15 = 2
    var_16 = 'EUR'



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 3



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #71
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
    var_10 = str(var_2)
    assert var_10 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #72
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTING'
    var_2 = ''
    var_3 = 123



# Parsed testcases at query #73
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



# Parsed testcases at query #74
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
    var_7 = var_0.has(var_6)
    assert var_7 is True
    var_8 = 'EUR'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = 'NONEXISTENT'
    var_13 = var_0.has(var_12)
    assert var_13 is False
    var_14 = 'usd'
    var_15 = var_0.has(var_14)
    assert var_15 is False
    var_16 = 'eur'
    var_17 = var_0.has(var_16)
    assert var_17 is False



# Parsed testcases at query #75
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
    var_11 = module_0.CurrencyRegistry()
    var_12 = len(var_11)
    assert var_12 == 3



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'USDA'
    var_4 = 'US Dollars A'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1



# Parsed testcases at query #77
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



# Parsed testcases at query #78
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = 'TEST'
    var_3 = 'Test Currency'
    var_4 = 2
    var_5 = len(var_0)
    assert var_5 == 1
    var_6 = len(var_0)
    assert var_6 == 1
    var_7 = 'AED'
    var_8 = 'UAE Dirham'
    var_9 = 2
    var_10 = 'BHD'
    var_11 = 'Bahraini Dinar'
    var_12 = 3
    var_13 = 'AED'
    var_14 = 'Duplicate'
    var_15 = 2



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = 'BTC'
    var_9 = 'Bitcoin'
    var_10 = 8



# Parsed testcases at query #80
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'



# Parsed testcases at query #81
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = CurrencyRegistry()[var_0]
    var_2 = 'NON-EXISTING'
    var_3 = CurrencyRegistry()[var_2]
    var_4 = str(var_3)
    assert var_4 == "Currency identified by code 'NON-EXISTING' does not exist"



# Parsed testcases at query #82
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8



# Parsed testcases at query #83
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'UX Dollars'
    var_6 = 0



# Parsed testcases at query #84
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD$'
    var_29 = 'Special Char Code'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = '  Trimmed  '
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = -2
    var_40 = 'USD'
    var_41 = 'US Dollars'
    var_42 = '2'
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = 2
    var_46 = 'MONEY'



# Parsed testcases at query #85
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'UX Dollars'



# Parsed testcases at query #86
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = 'AED'
    var_3 = 'UAE Dirham'
    var_4 = 2
    var_5 = 'USD'
    var_6 = 'US Dollar'
    var_7 = len(var_0)
    assert var_7 == 2



# Parsed testcases at query #87
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = '0.01'



# Parsed testcases at query #88
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
    var_7 = len(var_0)
    assert var_7 == 1
    var_8 = 'USD'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = var_0.get(var_8)
    var_13 = var_0.get(var_10)
    assert var_13 is None
    var_14 = 'XYZ'
    var_15 = var_0[var_14]



# Parsed testcases at query #89
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



# Parsed testcases at query #90
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'US Dollars High'
    var_7 = 3
    var_8 = 'US Dollars Low'
    var_9 = 1



# Parsed testcases at query #91
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2



# Parsed testcases at query #92
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'NONEXISTENT'
    var_2 = var_0.has(var_1)
    var_3 = 'USD'
    var_4 = 'US Dollar'
    var_5 = 2
    var_6 = 'USD'
    var_7 = var_0.has(var_6)
    var_8 = 'EUR'
    var_9 = var_0.has(var_8)



# Parsed testcases at query #93
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



# Parsed testcases at query #94
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 3



# Parsed testcases at query #95
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollar'
    var_6 = 3



# Parsed testcases at query #96
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



# Parsed testcases at query #97
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 3
    var_7 = 'not a currency'



# Parsed testcases at query #98
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #99
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
    var_21 = '0.0'
    var_22 = 'ZZZ'
    var_23 = 'Some weird currency'
    var_24 = -1
    var_25 = '1.0000000000005'
    var_26 = '1.000000000000'
    var_27 = '1.0000000000015'
    var_28 = '1.000000000002'
    var_29 = '1.0000000000000'
    var_30 = '1.0000000000001'
    var_31 = '1.0000000000009'
    var_32 = '1.000000000001'
    var_33 = 'EUR'
    var_34 = 'Euro'
    var_35 = '0.001'
    var_36 = '0.00'
    var_37 = '0.009'
    var_38 = '0.01'
    var_39 = '999.999'
    var_40 = '1000.00'
    var_41 = '-1.001'
    var_42 = '-1.00'
    var_43 = '-1.009'
    var_44 = '-1.01'



# Parsed testcases at query #100
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
    var_19 = '9.9'
    var_20 = '10'
    var_21 = 'ZZZ'
    var_22 = 'Some weird currency'
    var_23 = -1
    var_24 = '1.0000000000005'
    var_25 = '1.000000000000'
    var_26 = '1.0000000000015'
    var_27 = '1.000000000002'
    var_28 = '1.0000000000000'
    var_29 = '1.0000000000001'
    var_30 = '-1.005'
    var_31 = '-1.00'
    var_32 = '-1.015'
    var_33 = '-1.02'
    var_34 = '-0.5'
    var_35 = '-1.5'
    var_36 = '-2'
    var_37 = '-1.0000000000005'
    var_38 = '-1.000000000000'
    var_39 = '-1.0000000000015'
    var_40 = '-1.000000000002'
    var_41 = '0.00'



# Parsed testcases at query #101
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
    var_7 = var_0.has(var_6)
    assert var_7 is True
    var_8 = 'EUR'
    var_9 = var_0.has(var_8)
    assert var_9 is True
    var_10 = 'XYZ'
    var_11 = var_0.has(var_10)
    assert var_11 is False
    var_12 = ''
    var_13 = var_0.has(var_12)
    assert var_13 is False



# Parsed testcases at query #102
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()



# Parsed testcases at query #103
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD!'
    var_29 = 'Special Char Code'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = '  Trimmed  '
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = -2
    var_40 = 'USD'
    var_41 = 'US Dollars'
    var_42 = '2'
    var_43 = 'USD'
    var_44 = 'US Dollars'
    var_45 = 2
    var_46 = 'MONEY'



# Parsed testcases at query #104
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



# Parsed testcases at query #105
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'XAU'
    var_7 = 'Gold'
    var_8 = 'BTC'
    var_9 = 'Bitcoin'
    var_10 = 8
    var_11 = 'ZZZ'
    var_12 = 'Weird Currency 1'
    var_13 = -1
    var_14 = 'YYY'
    var_15 = 'Weird Currency 2'
    var_16 = 5



# Parsed testcases at query #106
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



# Parsed testcases at query #107
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
    var_9 = 'YYY'
    var_10 = 'Some other weird currency'
    var_11 = -1
    var_12 = 'XAU'
    var_13 = 'Gold'
    var_14 = 'ALT'
    var_15 = 'Alternative Currency'



# Parsed testcases at query #108
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



# Parsed testcases at query #109
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 3



# Parsed testcases at query #110
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = ''
    var_3 = 'USD!'
    var_4 = 'USD '



# Parsed testcases at query #111
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'US Dollars 2'
    var_6 = 3
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0



# Parsed testcases at query #112
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



# Parsed testcases at query #113
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = 'XAU'
    var_11 = 'Gold'



# Parsed testcases at query #114
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



# Parsed testcases at query #115
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #116
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euros'
    var_5 = 'US Dollars 2'
    var_6 = 3
    var_7 = -1



# Parsed testcases at query #117
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #118
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
    var_8 = 'XYZ'
    var_9 = var_0.get(var_8)
    assert var_9 is None
    var_10 = var_0[var_6]
    var_11 = var_0.get(var_8, var_10)



# Parsed testcases at query #119
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
    var_6 = var_0.has(var_3)
    var_7 = 'USD'
    var_8 = var_0.has(var_7)
    var_9 = len(var_0)
    assert var_9 == 1
    var_10 = var_0.get(var_7)
    var_11 = 'XYZ'
    var_12 = var_0.get(var_11)
    assert var_12 is None
    var_13 = 'XYZ'
    var_14 = var_0[var_13]



# Parsed testcases at query #120
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
    var_25 = 'usd'
    var_26 = 'Lowercase Code'
    var_27 = 2
    var_28 = 'USD '
    var_29 = 'Code with space'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = ' '
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



# Parsed testcases at query #121
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = '0.01'



# Parsed testcases at query #122
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #123
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
    var_8 = 'ZZZ'
    var_9 = 'Crypto1'
    var_10 = -1
    var_11 = 'Crypto2'
    var_12 = 'XAU'
    var_13 = 'Gold'
    var_14 = 'Gold Alt'
    var_15 = 'GBP'
    var_16 = 'British Pound'
    var_17 = 'CHF'
    var_18 = 'Swiss Franc'



# Parsed testcases at query #124
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = ''
    var_3 = None
    var_4 = 123



# Parsed testcases at query #125
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NON-EXISTING'



# Parsed testcases at query #126
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
    var_17 = '0.0'
    var_18 = '999.9'
    var_19 = '1000'
    var_20 = 'ZZZ'
    var_21 = 'Some weird currency'
    var_22 = -1
    var_23 = '1.0000000000005'
    var_24 = '1.000000000000'
    var_25 = '1.0000000000015'
    var_26 = '1.000000000002'
    var_27 = '1.0000000000000'
    var_28 = '1.0000000000001'
    var_29 = 'EUR'
    var_30 = 'Euro'
    var_31 = '0.001'
    var_32 = '0.00'
    var_33 = '0.009'
    var_34 = '0.01'
    var_35 = '-1.005'
    var_36 = '-1.00'
    var_37 = '-1.015'
    var_38 = '-1.02'



# Parsed testcases at query #127
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
    var_23 = 123
    var_24 = 'US Dollars'
    var_25 = 2
    var_26 = 'usd'
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
    var_39 = 'US Dollars'
    var_40 = '2'
    var_41 = 'USD'
    var_42 = 'US Dollars'
    var_43 = -2
    var_44 = 'USD'
    var_45 = 'US Dollars'
    var_46 = 2
    var_47 = 'MONEY'



# Parsed testcases at query #128
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



# Parsed testcases at query #129
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



# Parsed testcases at query #130
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'EUR'
    var_4 = 'Euro'
    var_5 = 'GBP'
    var_6 = 'British Pound'
    var_7 = 'US Dollars Alternative'
    var_8 = 3



# Parsed testcases at query #131
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #132
#--------------------------


def test_case_0():
    var_0 = 0
    var_1 = CurrencyRegistry()[var_0]
    var_2 = 'USD'
    var_3 = CurrencyRegistry()[var_2]
    var_4 = 'NON-EXISTING'
    var_5 = CurrencyRegistry()[var_4]



# Parsed testcases at query #133
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
    var_19 = '9.9'
    var_20 = '10'
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
    var_31 = '-1.005'
    var_32 = '-1.00'
    var_33 = '-1.5'
    var_34 = '-2'
    var_35 = '-1.0000000000005'
    var_36 = '-1.000000000000'



# Parsed testcases at query #134
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
    var_19 = '9.9'
    var_20 = '10'
    var_21 = 'ZZZ'
    var_22 = 'Some weird currency'
    var_23 = -1
    var_24 = '1.0000000000005'
    var_25 = '1.000000000000'
    var_26 = '1.0000000000015'
    var_27 = '1.000000000002'
    var_28 = '1.0000000000000'
    var_29 = '1.0000000000001'
    var_30 = '-1.005'
    var_31 = '-1.00'
    var_32 = '-1.5'
    var_33 = '-2'
    var_34 = '-1.0000000000005'
    var_35 = '-1.000000000000'
    var_36 = '0.00'



# Parsed testcases at query #135
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



# Parsed testcases at query #136
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = '1.23456'
    var_8 = '1.23'
    var_9 = '1.23556'
    var_10 = '1.24'
    var_11 = 'JPY'
    var_12 = 'Japanese Yen'
    var_13 = 0
    var_14 = '0.5'
    var_15 = '0'
    var_16 = '1.5'
    var_17 = '2'
    var_18 = '123.456'
    var_19 = '123'
    var_20 = 'ZZZ'
    var_21 = 'Some weird currency'
    var_22 = -1
    var_23 = '1.0000000000005'
    var_24 = '1.000000000000'
    var_25 = '1.0000000000015'
    var_26 = '1.000000000002'
    var_27 = '1.123456789012345678901234567890'
    var_28 = '-1.005'
    var_29 = '-1.00'
    var_30 = '-1.015'
    var_31 = '-1.02'
    var_32 = '-0.5'
    var_33 = '-1.5'
    var_34 = '-2'



# Parsed testcases at query #137
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.001'



# Parsed testcases at query #138
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



# Parsed testcases at query #139
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'
    var_7 = 4
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1



# Parsed testcases at query #140
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



# Parsed testcases at query #141
#--------------------------


def test_case_0():
    var_0 = 'NON-EXISTING'



# Parsed testcases at query #142
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



# Parsed testcases at query #143
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.001'



# Parsed testcases at query #144
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #145
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'NONEXISTENT'
    var_2 = ''
    var_3 = 'USD!'
    var_4 = 'usd'



# Parsed testcases at query #146
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'
    var_7 = 'ZZZ'
    var_8 = 'Some weird currency'
    var_9 = -1
    var_10 = 'XAU'
    var_11 = 'Gold'



# Parsed testcases at query #147
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = "Currency(code='USD', name='US Dollars', decimals=2, type=CurrencyType.MONEY, quantizer=Decimal('0.01'), hashcache={})"
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = "Currency(code='JPY', name='Japanese Yen', decimals=0, type=CurrencyType.MONEY, quantizer=Decimal('0'), hashcache={})"
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1
    var_11 = "Currency(code='ZZZ', name='Some weird currency', decimals=-1, type=CurrencyType.CRYPTO, quantizer=MaxPrecisionQuantizer, hashcache={})"
    var_12 = 'XAU'
    var_13 = 'Gold'
    var_14 = 4
    var_15 = "Currency(code='XAU', name='Gold', decimals=4, type=CurrencyType.METAL, quantizer=Decimal('0.0001'), hashcache={})"
    var_16 = 'ALT'
    var_17 = 'Alternative Currency'
    var_18 = 3
    var_19 = "Currency(code='ALT', name='Alternative Currency', decimals=3, type=CurrencyType.ALTERNATIVE, quantizer=Decimal('0.001'), hashcache={})"



# Parsed testcases at query #148
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()



# Parsed testcases at query #149
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
    var_10 = 'XXX'
    var_11 = var_0.get(var_10)
    assert var_11 is None
    var_12 = 'XXX'
    var_13 = var_0[var_12]



# Parsed testcases at query #150
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = 'UX Dollars'
    var_7 = 'XAU'
    var_8 = 'Gold'
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8



