####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the equality logic of the Currency class, ensuring that two currencies \n    are considered equal only if they share the same hash (derived from their attributes)\n    and that different currencies or non-currency types return False.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'UX Dollars'
    var_5 = 'GBP'
    var_6 = 'British Pounds'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __gt__ (greater than) implementation for the Currency class.\n    Since the class is decorated with @dataclass(order=True), \n    it uses the order of fields defined in the dataclass: \n    code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pounds'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'United States Dollars'



# Parsed testcases at query #3
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __len__ method of CurrencyRegistry.\n    Ensures it correctly reports the number of registered currencies \n    within a registry context.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = len(var_1)
    assert var_11 == 1
    var_12 = len(var_1)
    assert var_12 == 2
    var_13 = len(var_1)
    assert var_13 == 3
    var_14 = len(var_1)
    assert var_14 == 3



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'JPY'
    var_4 = 'Japanese Yen'
    var_5 = 0
    var_6 = '1.5'
    var_7 = '2'
    var_8 = '1.4'
    var_9 = '1'
    var_10 = 'ZZZ'
    var_11 = 'Crypto'
    var_12 = -1
    var_13 = '1.0000000000005'
    var_14 = '1.000000000000'
    var_15 = 'Different Name'
    var_16 = 'usd'
    var_17 = 'Lower Case'
    var_18 = 2
    var_19 = 'USD'
    var_20 = ' Leading Space'
    var_21 = 2
    var_22 = 'USD'
    var_23 = 'Trailing Space '
    var_24 = 2
    var_25 = 'USD1'
    var_26 = 'Has Numbers'
    var_27 = 2
    var_28 = 'USD'
    var_29 = ''
    var_30 = 2
    var_31 = 'USD'
    var_32 = 'US Dollars'
    var_33 = -2
    var_34 = 'USD'
    var_35 = 'US Dollars'
    var_36 = 2
    var_37 = 'NotAType'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)

def test_case_0():
    pass



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __lt__ (less than) implementation of the Currency class.\n    Since the class is decorated with @dataclass(order=True), \n    the comparison follows the order of fields defined in the dataclass:\n    code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'ABC'
    var_5 = 'Alpha Currency'
    var_6 = 'US X Dollars'
    var_7 = 0
    var_8 = 'A'
    var_9 = 'Name'
    var_10 = 'B'
    var_11 = 'Z Name'
    var_12 = 1



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '\n    Tests that attempting to delete an attribute from a frozen Currency instance\n    raises a FrozenInstanceError (via AttributeError).\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __lt__ (less than) implementation of the Currency class.\n    Since Currency is a frozen dataclass with order=True, it implements \n    comparison based on the order of fields defined in the dataclass.\n    The fields are: code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pounds'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8
    var_12 = 'AAA'
    var_13 = 'Alpha'
    var_14 = '0.01'
    var_15 = 1
    var_16 = 'Beta'
    var_17 = '1'
    var_18 = 3
    var_19 = 4
    var_20 = 5
    var_21 = 6



# Parsed testcases at query #9
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __getitem__ method of CurrencyRegistry.\n    Verifies that it returns the correct Currency object for existing codes\n    and raises CurrencyLookupError for non-existent codes.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'XYZ'
    var_8 = var_1[var_7]



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __gt__ (greater than) implementation for the Currency class.\n    Note: Since the class is decorated with @dataclass(order=True), \n    the comparison order is determined by the order of fields in the dataclass definition:\n    code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'ZAR'
    var_5 = 'South African Rand'
    var_6 = 'US Dollars Premium'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0



# Parsed testcases at query #11
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests that the __new__ method implements a singleton pattern, \n    ensuring that multiple instantiations return the exact same instance.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()

import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests that the singleton instance persists across different variable assignments.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = id(var_1)
    var_4 = id(var_2)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the greater-than-or-equal operator (__ge__) for the Currency class.\n    Since Currency is a dataclass with order=True, __ge__ uses the \n    comparison of fields in order: code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'ABC'
    var_5 = 'Alpha Currency'
    var_6 = 'XYZ'
    var_7 = 'X Currency'
    var_8 = 'US'
    var_9 = 1
    var_10 = 'Not a currency'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the less-than-or-equal comparison (__le__) for the Currency class.\n    Since the Currency class is decorated with @dataclass(order=True), \n    it implements __le__ based on the order of fields in the dataclass:\n    code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'AUD'
    var_5 = 'Australian Dollars'
    var_6 = 'US Dollar'



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
    var_14 = 'BTC'
    var_15 = 'Bitcoin'
    var_16 = -1
    var_17 = '1.0000000000005'
    var_18 = '1.000000000000'
    var_19 = 'UX Dollars'
    var_20 = 'usd'
    var_21 = 'US Dollars'
    var_22 = 2
    var_23 = 'US1'
    var_24 = 'US Dollars'
    var_25 = 2
    var_26 = 123
    var_27 = 'US Dollars'
    var_28 = 2
    var_29 = 'USD'
    var_30 = ''
    var_31 = 2
    var_32 = 'USD'
    var_33 = ' US Dollars '
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



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '\n    Tests that attempting to delete an attribute from a frozen dataclass \n    raises a FrozenInstanceError (or AttributeError).\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2



# Parsed testcases at query #16
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests that CurrencyRegistry implements the Singleton pattern via __new__.\n    Ensures that multiple instantiations return the exact same object instance.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #17
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #18
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'XYZ'
    var_5 = var_0[var_1]
    var_6 = var_0[var_4]
    var_7 = module_0.CurrencyRegistry()
    var_8 = 'USD'
    var_9 = var_7[var_8]



# Parsed testcases at query #19
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'XYZ'
    var_5 = var_0[var_4]
    var_6 = 'NON-EXISTENT'
    var_7 = var_0[var_6]



# Parsed testcases at query #20
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #21
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pound'
    var_6 = 'XYZ'
    var_7 = var_0[var_6]
    var_8 = 'NON-EXISTENT'
    var_9 = var_0[var_8]



# Parsed testcases at query #22
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
    var_7 = 'XYZ'
    var_8 = var_0[var_7]
    var_9 = 'NON_EXISTENT'
    var_10 = var_0[var_9]



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '1.005'
    var_4 = '1.00'
    var_5 = '1.015'
    var_6 = '1.02'
    var_7 = '1.006'
    var_8 = '1.01'
    var_9 = '1.004'
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0
    var_13 = '0.5'
    var_14 = '0'
    var_15 = '1.5'
    var_16 = '2'
    var_17 = '0.4'
    var_18 = '1.4'
    var_19 = '1'
    var_20 = 'ZZZ'
    var_21 = 'Some weird currency'
    var_22 = -1
    var_23 = '1.0000000000005'
    var_24 = '1.000000000000'
    var_25 = '1.0000000000015'
    var_26 = '1.000000000002'
    var_27 = '0.000'
    var_28 = '0.00'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = '\n    Tests that the __hash__ method of the Currency class returns the pre-computed \n    hashcache and ensures consistency for identical currency objects.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'USD'
    var_5 = 'US Dollars Modified'
    var_6 = 'hashcache'



# Parsed testcases at query #25
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'NON_EXISTENT'
    var_5 = var_0[var_4]
    var_6 = ''
    var_7 = var_0[var_6]
    var_8 = '123'
    var_9 = var_0[var_8]



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = '\n    Tests that the Currency class, being a frozen dataclass, \n    raises FrozenInstanceError (or TypeError) when attempting to modify attributes.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2



# Parsed testcases at query #27
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests that the __enter__ method of CurrencyRegistry returns the \n    internal __register method and sets the context flag to True.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = None
    var_4 = var_1.__exit__(var_3, var_3, var_3)



# Parsed testcases at query #28
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



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __lt__ (less than) implementation of the Currency class.\n    Since Currency is a dataclass with order=True, it uses the \n    order of fields defined in the class for comparison.\n    The field order is: code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'US Dollars Alt'
    var_7 = 0
    var_8 = 1



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __gt__ (greater than) magic method of the Currency class.\n    Since the class is decorated with @dataclass(order=True), \n    the comparison follows the order of fields defined in the dataclass:\n    code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pounds'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'United States Dollars'
    var_10 = 4
    var_11 = 'AAA'
    var_12 = 'Alpha'
    var_13 = 'MONARY'
    var_14 = 'BBB'
    var_15 = 'Beta'
    var_16 = 'USD'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'US Dollars Updated'
    var_4 = 0
    var_5 = 'EUR'
    var_6 = 'Euro'



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the quantize method of the Currency class with various scenarios including\n    standard money, zero-decimal currencies (JPY), and negative-decimal (crypto) currencies.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '1.005'
    var_5 = '1.00'
    var_6 = '1.015'
    var_7 = '1.02'
    var_8 = '1.000'
    var_9 = 'JPY'
    var_10 = 'Japanese Yen'
    var_11 = 0
    var_12 = '0.5'
    var_13 = '0'
    var_14 = '1.5'
    var_15 = '2'
    var_16 = '1.9'
    var_17 = 'ZZZ'
    var_18 = 'Some weird currency'
    var_19 = -1
    var_20 = '1.0000000000005'
    var_21 = '1.000000000000'
    var_22 = '1.0000000000015'
    var_23 = '1.000000000002'
    var_24 = 'EUR'
    var_25 = 'Euro'
    var_26 = '10.555'
    var_27 = '10.56'
    var_28 = '10.554'
    var_29 = '10.55'
    var_30 = 'GBP'
    var_31 = 'British Pound'
    var_32 = '100.123456'
    var_33 = '100.12'



# Parsed testcases at query #33
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pound'
    var_6 = 'XYZ'
    var_7 = var_0[var_6]
    var_8 = 'NON-EXISTING'
    var_9 = var_0[var_8]



# Parsed testcases at query #34
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = len(var_0)
    assert var_1 == 0
    var_2 = module_0.CurrencyRegistry()
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
    var_12 = var_0.has(var_6)
    assert var_12 is True
    var_13 = 'GBP'
    var_14 = var_0.has(var_13)
    assert var_14 is False
    var_15 = var_0.get(var_8)
    var_16 = 'NON_EXISTENT'



# Parsed testcases at query #35
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __getitem__ method of CurrencyRegistry.\n    Verifies that it returns the correct currency when the code exists\n    and raises CurrencyLookupError when the code does not exist.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = 'XYZ'
    var_6 = var_1[var_5]
    var_7 = var_1[var_2]
    var_8 = var_1[var_5]



# Parsed testcases at query #36
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __len__ method of the CurrencyRegistry class.\n    Ensures that it correctly returns the count of registered currencies.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = 'USD'
    var_4 = 'US Dollar'
    var_5 = 2
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = len(var_1)
    assert var_8 == 2
    var_9 = 'GBP'
    var_10 = 'British Pound'
    var_11 = 2
    var_12 = len(var_1)
    assert var_12 == 3



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'US Dollars Updated'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 'XYZ'
    var_5 = 'Different Name'
    var_6 = 5
    var_7 = '0.00001'
    var_8 = '1.0'
    var_9 = 'USD'
    var_10 = 'US Dollars'
    var_11 = 2



# Parsed testcases at query #3
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __getitem__ method of CurrencyRegistry, verifying that it returns \n    the correct Currency object for a valid code and raises CurrencyLookupError \n    for an invalid code.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'NON_EXISTENT'
    var_8 = var_1[var_7]



# Parsed testcases at query #4
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pound'
    var_6 = 'XYZ'
    var_7 = var_0[var_6]
    var_8 = 123
    var_9 = var_0[var_8]



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 'GBP'
    var_5 = 'British Pounds'
    var_6 = 0



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __le__ (less than or equal) method of the Currency class.\n    Since the class is decorated with @dataclass(order=True), \n    the order is determined by the field order: code, name, decimals, type...\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pounds'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'United States Dollars'



# Parsed testcases at query #7
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __contains__ method of CurrencyRegistry.\n    Verifies that it correctly identifies registered and unregistered currency codes.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0



# Parsed testcases at query #8
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __contains__ method of CurrencyRegistry.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = 'BTC'
    var_9 = 'Bitcoin'
    var_10 = 8



# Parsed testcases at query #9
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = "\n    Tests that the __exit__ method correctly re-sorts and synchronizes \n    the registry's internal buffers (registry, currencies, codes, codenames) \n    after a population context has closed.\n    "
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'ZZZ'
    var_3 = 'Z Currency'
    var_4 = 2
    var_5 = 'AAA'
    var_6 = 'A Currency'
    var_7 = 0
    var_8 = 'BBB'
    var_9 = 'B Currency'
    var_10 = var_1.all
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = (var_5, var_6)
    var_13 = (var_8, var_9)
    var_14 = (var_2, var_3)
    var_15 = [var_12, var_13, var_14]
    var_16 = 'CCC'
    var_17 = 'C Currency'
    var_18 = 2



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '\n    Tests that attempting to delete an attribute from a frozen Currency instance \n    raises a FrozenInstanceError (or AttributeError, which is how dataclass \n    frozen=True manifests).\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the quantize method of the Currency class with various scenarios \n    including standard money, zero-decimal currency (JPY), and crypto/high precision.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = '1.005'
    var_5 = '1.00'
    var_6 = '1.015'
    var_7 = '1.02'
    var_8 = '1.555'
    var_9 = '1.56'
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0
    var_13 = '0.5'
    var_14 = '0'
    var_15 = '1.5'
    var_16 = '2'
    var_17 = '1.4'
    var_18 = '1'
    var_19 = 'ZZZ'
    var_20 = 'Some weird currency'
    var_21 = -1
    var_22 = '1.0000000000005'
    var_23 = '1.000000000000'
    var_24 = '1.0000000000015'
    var_25 = '1.000000000002'
    var_26 = 'GBP'
    var_27 = 'British Pounds'
    var_28 = '2.5'
    var_29 = '3.5'
    var_30 = '4'
    var_31 = 'EUR'
    var_32 = 'Euro'
    var_33 = '100.123456789'
    var_34 = '100.12'



# Parsed testcases at query #12
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests that the __new__ method implements a singleton pattern,\n    ensuring that multiple calls to CurrencyRegistry() return the same instance.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = module_0.CurrencyRegistry()



# Parsed testcases at query #13
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __contains__ method of CurrencyRegistry.\n    Verifies that it correctly identifies existing and non-existing currency codes.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0



# Parsed testcases at query #14
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = len(var_0)
    assert var_11 == 3
    var_12 = var_0.get(var_6)
    var_13 = var_0.has(var_8)
    assert var_13 is True
    var_14 = 'GBP'
    var_15 = var_0.has(var_14)
    assert var_15 is False
    var_16 = 'NON_EXISTENT'
    var_17 = var_0[var_16]
    var_18 = str(var_3)



# Parsed testcases at query #15
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = "\n    Tests the 'get' method of the CurrencyRegistry class.\n    Verifies retrieving an existing currency, returning None for non-existent keys,\n    and returning a default value when provided.\n    "
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = var_1.get(var_2)
    var_11 = var_1.get(var_5)
    var_12 = var_11.name
    assert var_12 == 'Euro'
    var_13 = var_1.get(var_7)
    var_14 = var_13.decimals
    assert var_14 == 0
    var_15 = 'GBP'
    var_16 = var_1.get(var_15)
    assert var_16 is None
    var_17 = 'XYZ'
    var_18 = var_1.get(var_17)
    assert var_18 is None
    var_19 = 'ABC'
    var_20 = 'Fallback'
    var_21 = 'usd'
    var_22 = var_1.get(var_21)
    assert var_22 is None



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = '\n    Tests that attempting to delete an attribute from a frozen dataclass \n    raises AttributeError.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = "can't set attribute"
    var_5 = 'immutable'



# Parsed testcases at query #17
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __contains__ implementation of CurrencyRegistry.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = 'BTC'
    var_9 = 'Bitcoin'
    var_10 = 8



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __ge__ (greater than or equal) operator for the Currency class.\n    Since Currency is decorated with @dataclass(order=True), \n    it uses the fields in order: code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pounds'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'US Dollars Alt'
    var_10 = 'USD Different Name'
    var_11 = '0.01'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '\n    Tests that the __hash__ method returns the pre-computed hashcache \n    and maintains consistency for identical currency definitions.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __gt__ (greater than) magic method of the Currency class.\n    Since the class is decorated with @dataclass(order=True), \n    it implements comparison based on the order of fields in the dataclass definition.\n    The field order is: code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pounds'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'Alternative US Dollars'
    var_10 = 4



# Parsed testcases at query #21
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __len__ method of the CurrencyRegistry class.\n    Ensures that len() correctly returns the number of registered currencies\n    within a registry context.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8
    var_12 = len(var_1)
    assert var_12 == 1
    var_13 = len(var_1)
    assert var_13 == 2
    var_14 = len(var_1)
    assert var_14 == 3
    var_15 = len(var_1)
    assert var_15 == 3



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __lt__ (less than) magic method of the Currency class.\n    Since the class is decorated with @dataclass(order=True), \n    it uses the order of its fields for comparison.\n    The field order is: code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pounds'
    var_6 = 'Alternative USD'
    var_7 = 4
    var_8 = 'ZZZ'
    var_9 = 'Some weird currency'
    var_10 = -1
    var_11 = -1



# Parsed testcases at query #23
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'BTC'
    var_8 = 'Bitcoin'
    var_9 = 8
    var_10 = var_0.get(var_1)
    var_11 = var_0.get(var_4)
    var_12 = var_0.get(var_7)
    var_13 = 'XYZ'
    var_14 = var_0.get(var_13)
    assert var_14 is None
    var_15 = 'EUR'
    var_16 = 'Euro'
    var_17 = 'NON_EXISTENT'
    var_18 = var_0.get(var_1)



# Parsed testcases at query #24
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __len__ method of CurrencyRegistry to ensure it correctly \n    returns the number of registered currencies.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = 'USD'
    var_4 = 'US Dollar'
    var_5 = 2
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'JPY'
    var_9 = 'Japanese Yen'
    var_10 = 0
    var_11 = len(var_1)
    assert var_11 == 1
    var_12 = len(var_1)
    assert var_12 == 2
    var_13 = len(var_1)
    assert var_13 == 3
    var_14 = 'GBP'
    var_15 = 'British Pound'
    var_16 = 2
    var_17 = len(var_1)
    assert var_17 == 4



# Parsed testcases at query #25
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __exit__ method of CurrencyRegistry to ensure it correctly \n    finalizes, sorts, and populates the registry buffers after a context manager block.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = 'AED'
    var_6 = 'UAE Dirham'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = [var_5, var_7, var_2]
    var_11 = (var_5, var_6)
    var_12 = (var_7, var_8)
    var_13 = (var_2, var_3)
    var_14 = [var_11, var_12, var_13]



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = '\n    Tests that the __hash__ method of Currency returns the pre-computed hashcache\n    and that identical currencies produce the same hash.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'



# Parsed testcases at query #27
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = "\n    Tests the 'has' method of the CurrencyRegistry class.\n    Verifies that it correctly identifies registered and unregistered currency codes.\n    "
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = var_1.has(var_2)
    assert var_10 is False
    var_11 = var_1.has(var_2)
    assert var_11 is True
    var_12 = var_1.has(var_5)
    assert var_12 is True
    var_13 = var_1.has(var_7)
    assert var_13 is False
    var_14 = 'GBP'
    var_15 = var_1.has(var_14)
    assert var_15 is False
    var_16 = 'NONEXISTENT'
    var_17 = var_1.has(var_16)
    assert var_17 is False



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __gt__ (greater than) method of the Currency class.\n    Since the class is decorated with @dataclass(order=True), \n    comparison operators are generated based on the order of fields in the class definition.\n    The field order is: code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'ZWD'
    var_5 = 'Zimbabwean Dollars'
    var_6 = 'US Alt Dollars'
    var_7 = 4
    var_8 = 'BTC'
    var_9 = 'Bitcoin'
    var_10 = 8



# Parsed testcases at query #29
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
    assert var_11 == 2
    var_12 = var_0.has(var_3)
    assert var_12 is True
    var_13 = 'GBP'
    var_14 = var_0.has(var_13)
    assert var_14 is False
    var_15 = var_0.get(var_6)
    var_16 = 'XYZ'
    var_17 = var_0[var_16]



# Parsed testcases at query #30
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __len__ method of CurrencyRegistry.\n    Ensures that len() correctly returns the count of registered currencies\n    after they are added via the registry context.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = len(var_1)
    assert var_2 == 0
    var_3 = 'USD'
    var_4 = 'US Dollar'
    var_5 = 2
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'BTC'
    var_10 = 'Bitcoin'
    var_11 = 8
    var_12 = len(var_1)
    assert var_12 == 1
    var_13 = len(var_1)
    assert var_13 == 2
    var_14 = len(var_1)
    assert var_14 == 3
    var_15 = len(var_1)
    assert var_15 == 3
    var_16 = len(var_1)
    assert var_16 == 3



# Parsed testcases at query #31
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests that the __enter__ method of CurrencyRegistry returns the \n    internal register method and sets the context open flag.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 2
    var_8 = 'ABC'
    var_9 = 'Alpha'
    var_10 = 0



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __gt__ (greater than) magic method of the Currency class.\n    Note: The provided implementation uses @dataclass(order=True), \n    which implements comparison methods based on field order.\n    The fields in Currency are: code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = 'GBP'
    var_8 = 'British Pound'
    var_9 = 'US Dollars Plus'
    var_10 = 4
    var_11 = 'Z'
    var_12 = 'A'
    var_13 = 'Y'



# Parsed testcases at query #33
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests that the __new__ method implements a singleton pattern, \n    ensuring that multiple calls return the exact same instance.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #34
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests that the __exit__ method correctly finalizes the registry by:\n    1. Re-sorting the registry by currency code.\n    2. Updating internal buffers (currencies, codes, codenames).\n    3. Closing the population context flag.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'BHD'
    var_3 = 'Bahraini Dinar'
    var_4 = 3
    var_5 = 'AUD'
    var_6 = 'Australian Dollar'
    var_7 = 2
    var_8 = 'CAD'
    var_9 = 'Canadian Dollar'
    var_10 = [var_5, var_2, var_8]
    var_11 = (var_5, var_6)
    var_12 = (var_2, var_3)
    var_13 = (var_8, var_9)
    var_14 = [var_11, var_12, var_13]



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
    var_7 = '1.555'
    var_8 = '1.56'
    var_9 = 'JPY'
    var_10 = 'Japanese Yen'
    var_11 = 0
    var_12 = '0.5'
    var_13 = '0'
    var_14 = '1.5'
    var_15 = '2'
    var_16 = '100.9'
    var_17 = '101'
    var_18 = 'ZZZ'
    var_19 = 'Some weird currency'
    var_20 = -1
    var_21 = '1.0000000000005'
    var_22 = '1.000000000000'
    var_23 = '1.0000000000015'
    var_24 = '1.000000000002'
    var_25 = '0.000'
    var_26 = '0.00'



# Parsed testcases at query #36
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __getitem__ method of CurrencyRegistry.\n    Verifies that it returns the correct currency when the code exists,\n    and raises CurrencyLookupError when the code does not exist.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = 'XYZ'
    var_6 = var_1[var_5]



