####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __enter__ method of CurrencyRegistry.\n    The method should set the internal context flag to True and return the register method.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = var_1.__enter__()
    var_3 = 'USD'
    var_4 = 'US Dollars'
    var_5 = 2
    var_6 = var_1.has(var_3)
    var_7 = None
    var_8 = var_1.__exit__(var_7, var_7, var_7)



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = "\n    Tests the 'has' method of the CurrencyRegistry class.\n    "
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = 'BTC'
    var_11 = 'Bitcoin'
    var_12 = 8
    var_13 = var_1.has(var_2)
    assert var_13 is False
    var_14 = var_1.has(var_2)
    assert var_14 is True
    var_15 = var_1.has(var_5)
    assert var_15 is True
    var_16 = var_1.has(var_7)
    assert var_16 is True
    var_17 = var_1.has(var_10)
    assert var_17 is True
    var_18 = 'GBP'
    var_19 = var_1.has(var_18)
    assert var_19 is False
    var_20 = 'XYZ'
    var_21 = var_1.has(var_20)
    assert var_21 is False
    var_22 = ''
    var_23 = var_1.has(var_22)
    assert var_23 is False
    var_24 = 'usd'
    var_25 = var_1.has(var_24)
    assert var_25 is False



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __gt__ (greater than) magic method of the Currency class.\n    Note: Since the class is decorated with @dataclass(order=True), \n    __gt__ is automatically implemented based on the order of fields \n    defined in the class (code, name, decimals, type, quantizer, hashcache).\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'ABC'
    var_5 = 'Alpha Currency'
    var_6 = 0



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    Tests that the Currency class, being a frozen dataclass, \n    raises FrozenInstanceError (or AttributeError) when attempting \n    to set an attribute after instantiation.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2



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
    var_21 = 'USD'
    var_22 = 'US Dollars'
    var_23 = 2
    var_24 = 'UX Dollars'
    var_25 = 'usd'
    var_26 = 'US Dollars'
    var_27 = 2
    var_28 = 'US1'
    var_29 = 'US Dollars'
    var_30 = 2
    var_31 = 'USD'
    var_32 = ''
    var_33 = 2
    var_34 = 'USD'
    var_35 = ' US Dollars'
    var_36 = 2
    var_37 = 'USD'
    var_38 = 'US Dollars'
    var_39 = -2
    var_40 = 'USD'
    var_41 = 'US Dollars'
    var_42 = 2
    var_43 = 'NotAType'

import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #8
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __contains__ method of CurrencyRegistry.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'



# Parsed testcases at query #9
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
    var_11 = var_2 in var_1



# Parsed testcases at query #10
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __enter__ method of CurrencyRegistry.\n    It should set the internal context flag to True and return the __register method.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'USD'
    var_6 = 'US Dollar'
    var_7 = 2
    var_8 = 'EUR'
    var_9 = 'Euro'
    var_10 = 2



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __le__ (less than or equal to) magic method of the Currency class.\n    Since the class is decorated with @dataclass(order=True), the comparison\n    is based on the order of fields defined in the class:\n    code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'USD Alternative'
    var_5 = 'AUD'
    var_6 = 'Australian Dollars'
    var_7 = 0



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __lt__ method of the Currency class.\n    Since Currency is a dataclass with order=True, it implements __lt__ \n    based on the order of fields defined in the class: \n    (code, name, decimals, type, quantizer, hashcache).\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'ABC'
    var_5 = 'Alpha Currency'
    var_6 = 'United States Dollars'
    var_7 = 0



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __hash__ method of the Currency class to ensure it returns\n    the pre-computed hashcache and maintains consistency for identical objects.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 2



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __ge__ (greater than or equal) magic method of the Currency class.\n    Since the class is decorated with @dataclass(order=True), __ge__ is \n    automatically implemented based on the order of fields in the class definition.\n    The order of fields is: code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pounds'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'United States Dollars'
    var_10 = 'US Dollars Z'
    var_11 = '0.01'
    var_12 = 'USD_Z'
    var_13 = hash(var_12)
    var_14 = '1'
    var_15 = 'USD_0'
    var_16 = hash(var_15)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __ge__ (greater than or equal) magic method of the Currency class.\n    Since Currency is a dataclass with order=True, it implements __ge__ \n    based on the order of fields in the class definition.\n    The field order is: code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'USDT'
    var_5 = 'Tether'
    var_6 = 'US Dollars Plus'
    var_7 = 3
    var_8 = 'XAU'
    var_9 = 'Gold'



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
    var_0 = '\n    Tests the constructor and singleton behavior of CurrencyRegistry.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



# Parsed testcases at query #18
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = "\n    Tests the 'get' method of CurrencyRegistry for existing,\n    non-existing, and default value scenarios.\n    "
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'MONEX'
    var_3 = 'USD'
    var_4 = 'US Dollar'
    var_5 = 2
    var_6 = 'EUR'
    var_7 = 'Euro'
    var_8 = 'GBP'
    var_9 = 'British Pound'
    var_10 = var_1.get(var_3)
    var_11 = 'JPY'
    var_12 = var_1.get(var_11)
    assert var_12 is None



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __le__ (less than or equal) method of the Currency class.\n    Since the class is decorated with @dataclass(order=True), \n    the comparison follows the order of fields defined in the class:\n    code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'US Dollars (Alt)'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = 'ABC'
    var_11 = 'Alpha'
    var_12 = 1



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __ge__ (greater than or equal) magic method of the Currency class.\n    Since the class is decorated with @dataclass(order=True), the order is \n    determined by the order of fields in the class definition.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'US Dollars Premium'
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = 0
    var_8 = 'Not a currency'



# Parsed testcases at query #21
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __enter__ method of CurrencyRegistry.\n    It should set the context as open and return the __register method.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = var_1.__enter__()
    var_6 = callable(var_5)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __getitem__ method of CurrencyRegistry for both\n    successful retrieval and raising CurrencyLookupError.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollars'
    var_4 = 2
    var_5 = 'NON_EXISTENT'
    var_6 = var_1[var_5]



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = "\n    Tests the __eq__ implementation of the Currency class.\n    Ensures that equality is based on the hashcache (which represents the object's identity \n    in this implementation) and handles comparisons with non-Currency objects.\n    "
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'US Dollars Modified'
    var_5 = 3
    var_6 = '0.01'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'US Dollars Alt'
    var_4 = 0
    var_5 = 'EUR'
    var_6 = 'Euro'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = 'UX Dollars'
    var_4 = 0
    var_5 = 'EUR'
    var_6 = 'Euro'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __eq__ method of the Currency class to ensure it correctly\n    compares currency objects based on their hashcache (representing identity).\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'USD'
    var_5 = 'UX Dollars'
    var_6 = 2
    var_7 = 'US Dollars'



# Parsed testcases at query #6
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = "\n    Tests the 'has' method of the CurrencyRegistry class.\n    "
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'JPY'
    var_6 = 'Japanese Yen'
    var_7 = 0
    var_8 = var_1.has(var_2)
    assert var_8 is False
    var_9 = var_1.has(var_2)
    assert var_9 is True
    var_10 = var_1.has(var_5)
    assert var_10 is True
    var_11 = 'EUR'
    var_12 = var_1.has(var_11)
    assert var_12 is False
    var_13 = ''
    var_14 = var_1.has(var_13)
    assert var_14 is False
    var_15 = None
    var_16 = var_1.has(var_15)
    assert var_16 is False
    var_17 = 123
    var_18 = var_1.has(var_17)
    assert var_18 is False



# Parsed testcases at query #7
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the __exit__ method of CurrencyRegistry to ensure it correctly \n    finalizes the registry by sorting and updating internal buffers \n    after a context manager block completes.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'AED'
    var_6 = 'UAE Dirham'
    var_7 = 'JPY'
    var_8 = 'Japanese Yen'
    var_9 = 0
    var_10 = (var_5, var_6)
    var_11 = (var_7, var_8)
    var_12 = (var_2, var_3)
    var_13 = [var_10, var_11, var_12]
    var_14 = 'GBP'
    var_15 = 'British Pound'
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
    var_7 = '1.1234'
    var_8 = '1.12'
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
    var_24 = 'GBP'
    var_25 = 'British Pound'
    var_26 = 4
    var_27 = '1.123456'
    var_28 = '1.1235'
    var_29 = '1.123444'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __le__ (less than or equal) magic method of the Currency class.\n    Since Currency is a dataclass with order=True, it implements comparison \n    operators based on the order of fields in the class definition.\n    The field order is: code, name, decimals, type, quantizer, hashcache.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'ABC'
    var_5 = 'Alpha Currency'
    var_6 = 'AAA Currency'
    var_7 = 0
    var_8 = 'Not a Currency object'



# Parsed testcases at query #10
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'XYZ'
    var_1 = module_0.CurrencyLookupError(var_0)
    var_2 = str(var_1)



# Parsed testcases at query #11
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests that the __new__ method correctly implements the Singleton pattern\n    for the CurrencyRegistry class.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = module_0.CurrencyRegistry()



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'US Dollars'
    var_2 = 2
    var_3 = '0.01'
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0
    var_7 = '1'
    var_8 = 'BTC'
    var_9 = 'Bitcoin'
    var_10 = -1
    var_11 = 'US Dollar'
    var_12 = '1.005'
    var_13 = '1.00'
    var_14 = '1.015'
    var_15 = '1.02'
    var_16 = '1.5'
    var_17 = '2'
    var_18 = '0.5'
    var_19 = '0'
    var_20 = 'usd'
    var_21 = 'US Dollars'
    var_22 = 2
    var_23 = 'U1D'
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
    var_39 = 'US Dollars'
    var_40 = -2
    var_41 = 'USD'
    var_42 = 'US Dollars'
    var_43 = '2'
    var_44 = 'USD'
    var_45 = 'US Dollars'
    var_46 = 2
    var_47 = 'MONEY'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '\n    Tests that the __hash__ method of the Currency class returns the pre-computed hashcache\n    and that identical currency definitions result in the same hash.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pounds'
    var_6 = 2



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __repr__ method of the Currency class.\n    Note: Since the provided code does not explicitly implement __repr__, \n    this test verifies the default behavior of a dataclass __repr__.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'Currency('
    var_5 = ')'



# Parsed testcases at query #15
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = "\n    Tests the 'get' method of CurrencyRegistry for:\n    1. Retrieving an existing currency.\n    2. Returning the default value when a currency is not found.\n    3. Returning None when a currency is not found and no default is provided.\n    "
    var_1 = module_0.CurrencyRegistry()
    var_2 = 'USD'
    var_3 = 'US Dollar'
    var_4 = 2
    var_5 = 'EUR'
    var_6 = 'Euro'
    var_7 = var_1.get(var_2)
    var_8 = var_1.get(var_5)
    var_9 = 'GBP'
    var_10 = 'British Pound'
    var_11 = 'JPY'
    var_12 = var_1.get(var_11)
    assert var_12 is None
    var_13 = 'NON_EXISTENT'
    var_14 = var_1.get(var_13)
    assert var_14 is None



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __hash__ method of the Currency class to ensure it returns\n    the pre-computed hashcache and maintains consistency.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'JPY'
    var_5 = 'Japanese Yen'
    var_6 = 0



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __repr__ method of the Currency class.\n    Note: Since the provided code does not explicitly implement __repr__, \n    it relies on the default dataclass implementation.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2



# Parsed testcases at query #18
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = module_0.CurrencyRegistry()
    var_1 = 'USD'
    var_2 = 'US Dollar'
    var_3 = 2
    var_4 = 'GBP'
    var_5 = 'British Pound'



# Parsed testcases at query #19
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the singleton behavior of CurrencyRegistry.__new__.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()



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
    var_7 = '1.5'
    var_8 = '1.50'
    var_9 = '1'
    var_10 = 'JPY'
    var_11 = 'Japanese Yen'
    var_12 = 0
    var_13 = '0.5'
    var_14 = '0'
    var_15 = '2'
    var_16 = '1.1'
    var_17 = 'ZZZ'
    var_18 = 'Some weird currency'
    var_19 = -1
    var_20 = '1.0000000000005'
    var_21 = '1.000000000000'
    var_22 = '1.0000000000015'
    var_23 = '1.000000000002'
    var_24 = 'GBP'
    var_25 = 'British Pound'
    var_26 = '1.23456'
    var_27 = '1.23'
    var_28 = '1.23556'
    var_29 = '1.24'
    var_30 = 'EUR'
    var_31 = 'Euro'
    var_32 = '0.00'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __gt__ (greater than) implementation for the Currency class.\n    Note: Since the class uses @dataclass(order=True), the comparison \n    is based on the order of fields defined in the class.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'EUR'
    var_5 = 'Euro'
    var_6 = 'JPY'
    var_7 = 'Japanese Yen'
    var_8 = 0
    var_9 = 'Alpha'
    var_10 = 'Zeta'



# Parsed testcases at query #22
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the constructor and singleton behavior of CurrencyRegistry.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = len(var_1)
    assert var_3 == 0



# Parsed testcases at query #23
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = '\n    Tests the constructor and initialization logic of the CurrencyRegistry singleton.\n    '
    var_1 = module_0.CurrencyRegistry()
    var_2 = module_0.CurrencyRegistry()
    var_3 = len(var_1)
    assert var_3 == 0
    var_4 = 'USD'
    var_5 = var_1.has(var_4)
    assert var_5 is False
    var_6 = var_1._CurrencyRegistry__registry
    var_7 = 'collections'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __hash__ method of the Currency class to ensure it returns \n    the pre-computed hashcache and maintains consistency for identical objects.\n    '
    var_1 = 'USD'
    var_2 = 'US Dollars'
    var_3 = 2
    var_4 = 'USD'
    var_5 = 'UX Dollars'



