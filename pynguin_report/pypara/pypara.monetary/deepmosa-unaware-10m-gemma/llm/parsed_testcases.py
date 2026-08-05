####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '10.5'
    var_4 = '0'
    var_5 = '99'
    var_6 = '5'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '\n    Tests that NoneMoney.dov_or returns the default date provided,\n    since NoneMoney does not have a defined DOV.\n    '
    var_1 = 2001
    var_2 = 1



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '100'
    var_4 = 2000



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = '10.5'
    var_3 = 2023
    var_4 = 1
    var_5 = '11'
    var_6 = 2



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the round method of the Money class for both defined (SomeMoney) \n    and undefined (NoMoney/na) instances.\n    '
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 2023
    var_4 = 1
    var_5 = '1.255'
    var_6 = '1.245'
    var_7 = 0
    var_8 = '1'
    var_9 = 2
    var_10 = '1.26'

def test_case_0():
    var_0 = '\n    Tests edge cases like negative values and zero for the round method.\n    '
    var_1 = 'USD'
    var_2 = 2023
    var_3 = 1
    var_4 = '0.00'
    var_5 = 2
    var_6 = '-1.555'
    var_7 = '-1.56'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 2
    var_5 = '10'
    var_6 = '5'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the boolean evaluation of Price objects.\n    Since the provided code defines __bool__, we test both defined \n    and undefined (na) instances.\n    '
    var_1 = 'USD'
    var_2 = 2019
    var_3 = 1
    var_4 = '10.5'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __truediv__ method of the Money class.\n    Since the provided code is an abstract base class, we use a Mock \n    to simulate the behavior described in the docstrings.\n    '
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'
    var_5 = '5.00'
    var_6 = '0'
    var_7 = 'not a number'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __lt__ (less than) implementation of the Money class.\n    Testing various scenarios including defined/undefined objects and \n    incompatible currencies.\n    '
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 2023
    var_4 = 1
    var_5 = '10.00'
    var_6 = '20.00'
    var_7 = None
    var_8 = '5.00'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '10'
    var_1 = 10
    var_2 = '10.5'
    var_3 = '0'
    var_4 = 0
    var_5 = '-5.9'
    var_6 = -5
    var_7 = 2023
    var_8 = 1



# Parsed testcases at query #11
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3, var_1)
    var_5 = 2023
    var_6 = 1
    var_7 = '10.00'
    var_8 = '4.00'
    var_9 = '5.00'
    var_10 = '6.00'
    var_11 = '0.00'
    var_12 = '-5.00'
    var_13 = '9.00'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '10.00'
    var_1 = '20.00'
    var_2 = True
    var_3 = None
    var_4 = '0'
    var_5 = False
    var_6 = '5.00'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 2019
    var_1 = 1
    var_2 = 2
    var_3 = '10'
    var_4 = '5'
    var_5 = None
    var_6 = True
    var_7 = '15'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dov_or method of the Money class.\n    Ensures it returns the value date (dov) if defined, \n    and the provided default if undefined.\n    '
    var_1 = '100.00'
    var_2 = 2000
    var_3 = 1
    var_4 = 2025
    var_5 = 12
    var_6 = 31



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 2
    var_5 = '10'
    var_6 = '20'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = '0.01'
    var_2 = 0
    var_3 = '1'
    var_4 = 2023
    var_5 = 1
    var_6 = '1.2345'
    var_7 = '1.23'
    var_8 = 5
    var_9 = '150.75'
    var_10 = '150'
    var_11 = -1
    var_12 = '0.00'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 2
    var_5 = '10.00'
    var_6 = '5.00'
    var_7 = '15.00'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the qty_map method of the Price class.\n    \n    Scenario 1: Defined price - applies function to quantity.\n    Scenario 2: Undefined price - returns value from the error/fallback handler.\n    '
    var_1 = '1'
    var_2 = '42'
    var_3 = False
    var_4 = lambda : var_3



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the gte (greater than or equal to) method of the Money class.\n    \n    Scenarios covered:\n    1. Defined money >= same defined money (Equality)\n    2. Defined money > same defined money (Greater than)\n    3. Defined money < same defined (Less than - should be False)\n    4. Undefined money >= defined money (Should be False per docstring)\n    5. Undefined money >= undefined money (Should be True per docstring)\n    6. Incompatible currencies (Should raise IncompatibleCurrencyError)\n    '



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the lte (less than or equal to) method of the Money class.\n    Covers:\n    1. Undefined money is always <= other defined money.\n    2. Defined money with same currency and smaller quantity.\n    3. Defined money with same currency and equal quantity.\n    4. IncompatibleCurrencyError when comparing different currencies.\n    '
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 2023
    var_4 = 1
    var_5 = None
    var_6 = '1.0'
    var_7 = '10.0'
    var_8 = '20.0'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __float__ method of the Price class.\n    Since Price is an abstract base class, we test the behavior \n    on a concrete implementation (SomePrice).\n    '
    var_1 = 10.5
    var_2 = str(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '123.456789'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = "\n    Tests the __bool__ method (truthiness) of the Money class.\n    Based on the abstract definition, a Money object's truthiness \n    is implementation-dependent, but typically relates to whether it is defined.\n    "
    var_1 = True
    var_2 = False
    var_3 = 'Defined money should evaluate to True in boolean context'
    var_4 = 'Undefined money should evaluate to False in boolean context'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = '5.00'
    var_1 = 10
    var_2 = '0'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dov_or method of the Money class.\n    The method should return the value date (dov) if defined, \n    otherwise return the provided default date.\n    '
    var_1 = '10.0'
    var_2 = 2000
    var_3 = 1
    var_4 = 2025
    var_5 = 12
    var_6 = 31



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 2019
    var_2 = 1
    var_3 = '100.00'
    var_4 = '0.00'
    var_5 = '-50.00'



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = "\n    Tests the 'convert' method of the Money class.\n    Since the provided code is an abstract base class (ABC), \n    we test against a Mock implementation that follows the documented behavior.\n    "
    var_1 = 2023
    var_2 = 1
    var_3 = 2
    var_4 = 'asof'
    var_5 = None
    var_6 = True



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 2023
    var_2 = 1
    var_3 = '100.00'
    var_4 = True
    var_5 = None
    var_6 = False
    var_7 = '-50.00'
    var_8 = True



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.00'
    var_2 = 2023
    var_3 = 1
    var_4 = '2'
    var_5 = '0'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the with_dov method of the Money class.\n    The test covers:\n    1. Returning a new object with the updated date when money is defined.\n    2. Returning the same object (itself) when money is undefined.\n    '
    var_1 = 2023
    var_2 = 1
    var_3 = 2024
    var_4 = '100.00'
    var_5 = True
    var_6 = None
    var_7 = False



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the __abs__ method of the Money class.\n    Since the class is abstract, we mock the behavior based on the \n    provided docstrings and signature requirements.\n    '



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Undefined'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2019
    var_3 = 1
    var_4 = '1'
    var_5 = '2'
    var_6 = '5'
    var_7 = '10'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = 'EUR'
    var_2 = 2023
    var_3 = 1
    var_4 = 2
    var_5 = '10.00'
    var_6 = '5.00'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dimap method of the Money class.\n    dimap should apply f to the value if defined, and e if undefined.\n    '
    var_1 = lambda x: x.ccy.code
    var_2 = 'EUR'
    var_3 = lambda : var_2
    var_4 = True
    var_5 = 'GBP'
    var_6 = False
    var_7 = None
    var_8 = lambda x: x.ccy_code
    var_9 = 'DEFAULT'
    var_10 = lambda : var_9
    var_11 = lambda x: x.ccy_code
    var_12 = lambda : var_9



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the fmap method of the Money class.\n    The test covers:\n    1. Applying a function to a defined Money object (SomeMoney).\n    2. Handling an undefined Money object (NoMoney/na) using fmap.\n    '
    var_1 = 'USD'
    var_2 = 2019
    var_3 = 1
    var_4 = '1.00'
    var_5 = True
    var_6 = '2.00'
    var_7 = 11
    var_8 = False



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the ccy_or_none method of the Money class.\n    The method should return the currency if defined, and None if undefined.\n    '
    var_1 = None



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '10.5'
    var_1 = False
    var_2 = '-5.9'
    var_3 = '0.00'
    var_4 = None
    var_5 = True



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the negative() method and __neg__ operator of the Money class.\n    Since the class is abstract, we test against a mock implementation \n    representing both defined (SomeMoney) and undefined (NoMoney) states.\n    '
    var_1 = '5.00'
    var_2 = '-5.00'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = "\n    Tests the __int__ method of the Money class.\n    Since the class is abstract, we mock the behavior or use a concrete implementation \n    if available. This test assumes 'SomeMoney' and 'NoMoney' are the concrete subclasses.\n    "
    var_1 = 'Undefined money'
    var_2 = '15.99'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = '10.5'
    var_1 = 2023
    var_2 = 1
    var_3 = True
    var_4 = '-5.25'
    var_5 = True
    var_6 = '5.25'
    var_7 = None
    var_8 = False



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the qty_or_else method of the Money class.\n    The method should return the quantity if defined, \n    otherwise return the result of the provided callable.\n    '
    var_1 = '1.00'
    var_2 = '42'
    var_3 = True
    var_4 = lambda : var_3
    var_5 = False
    var_6 = lambda : var_5



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the with_dov method of the Money class.\n    Ensures that:\n    1. It returns a new money object with the updated date if defined.\n    2. It returns itself (the same instance) if the money object is undefined.\n    3. The original object remains unchanged (immutability).\n    '
    var_1 = 2023
    var_2 = 1
    var_3 = 2024
    var_4 = '100.00'



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the dov_or method of the Money class.\n    Checks that it returns the value date if defined, and the default \n    date if the money object is undefined.\n    '
    var_1 = 2023
    var_2 = 5
    var_3 = 20
    var_4 = '100.00'
    var_5 = 1999
    var_6 = 1



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the lt (less than) method of the Money class.\n    Covers:\n    1. Undefined money < defined money is True.\n    2. Defined money < undefined money is False.\n    3. Comparison of two defined money objects with same currency.\n    4. Comparison of two defined money objects with different currencies (raises IncompatibleCurrencyError).\n    '
    var_1 = 'USD'
    var_2 = 'EUR'
    var_3 = 2023
    var_4 = 1
    var_5 = '10.00'
    var_6 = '20.00'
    var_7 = '5.00'
    var_8 = False
    var_9 = True
    var_10 = True



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the floor_divide method of the Price class.\n    Since Price is an abstract base class, we test against a mock \n    implementation that follows the documented behavior:\n    1. Performs floor division if defined.\n    2. Returns an undefined price object if division by zero occurs.\n    '
    var_1 = '10'
    var_2 = 3
    var_3 = '3'
    var_4 = 0
    var_5 = 5
    var_6 = '2'
    var_7 = '5'



# Parsed testcases at query #22
#--------------------------


import pypara.currencies as module_0

def test_case_0():
    var_0 = 'USD'
    var_1 = 2
    var_2 = module_0.Currency(var_0, var_1)
    var_3 = 'EUR'
    var_4 = module_0.Currency(var_3, var_1)
    var_5 = 2023
    var_6 = 1
    var_7 = '10.00'
    var_8 = '20.00'
    var_9 = '5.00'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'USD'
    var_1 = '10.5'
    var_2 = 2023
    var_3 = 1
    var_4 = True
    var_5 = '1'
    var_6 = '42'
    var_7 = '11.5'
    var_8 = None
    var_9 = False
    var_10 = '2'
    var_11 = '21.0'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the round functionality of the Money class for both \n    defined (SomeMoney) and undefined (NoMoney/NA) instances,\n    ensuring it handles ndigits correctly using HALF_EVEN logic.\n    '
    var_1 = '1.5'
    var_2 = 0
    var_3 = '2'
    var_4 = '2.5'
    var_5 = '1.2345'
    var_6 = 2
    var_7 = '1.23'
    var_8 = '1.2355'
    var_9 = '1.24'
    var_10 = '0'
    var_11 = False
    var_12 = '1.678'
    var_13 = '1.68'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the abs() method for the Money class.\n    The logic should return the absolute money if defined, and itself otherwise.\n    '



