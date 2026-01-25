####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 'invalid'
    var_6 = 'All tests passed for BaseProvider.validate_enum()'
    var_7 = print(var_6)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    assert var_2 == 'en'
    var_3 = var_1.get_current_locale()
    assert var_3 == 'ru'
    var_4 = var_1.get_current_locale()
    assert var_4 == 'en'
    var_5 = module_0.BaseDataProvider(var_3)
    var_6 = '_dataset'
    var_7 = delattr(var_5, var_6)
    var_8 = module_0.BaseDataProvider(var_3)
    var_9 = var_8.get_current_locale()
    assert var_9 == 'en'
    var_10 = var_8.get_current_locale()
    assert var_10 == 'en'
    var_11 = var_8.get_current_locale()
    assert var_11 == 'en'



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 'test2'
    var_2 = module_0.BaseProvider()
    var_3 = None
    var_4 = 'invalid'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test method override_locale of class BaseDataProvider.'



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = 'test_datafile.json'
    var_2 = None
    assert var_2 == 'fr'
    var_3 = 'en'
    var_4 = module_0.BaseProvider()



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Unit test for method override_locale of class BaseDataProvider.'
    var_1 = 'Test exception'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'mock'
    var_1 = 'mock.json'
    var_2 = 'non_locale'
    var_3 = None



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 4



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'mock'
    var_1 = 'mock.json'
    var_2 = 'non_locale'
    var_3 = None



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    assert var_2 == 'en'
    var_3 = var_1.get_current_locale()
    assert var_3 == 'ru'
    var_4 = var_1.get_current_locale()
    assert var_4 == 'en'



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 4



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.base as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'Test the BaseProvider class.'
    var_1 = module_0.BaseProvider()
    var_2 = module_0.BaseProvider()
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = module_0.BaseProvider(seed=var_3)
    var_6 = module_0.BaseProvider(seed=var_3)
    var_7 = 43
    var_8 = var_6.reseed(var_7)
    var_9 = module_1.Random()
    var_10 = module_0.BaseProvider(random=var_9)
    var_11 = 1
    var_12 = 2
    var_13 = None
    var_14 = 3
    var_15 = 'address.json'
    var_16 = module_0.BaseProvider(seed=var_13)
    var_17 = str(var_1)
    assert var_17 == 'BaseProvider'



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 4



# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'test.json'



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'test_data.json'
    var_2 = 'Test exception'
    var_3 = module_0.BaseProvider()



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Unit test for method override_locale of class BaseDataProvider.'
    var_1 = module_0.BaseProvider()



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Test reseed method of BaseProvider.'
    var_1 = module_0.BaseProvider()
    var_2 = var_1.reseed()
    var_3 = 42
    var_4 = var_1.reseed(var_3)
    var_5 = None
    var_6 = var_1.reseed(var_5)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'test.json'



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = 'C'
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 'D'



# Parsed testcases at query #23
#--------------------------




# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test method override_locale().'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = '\n    Test that the override_locale method correctly changes the locale temporarily.\n    '
    var_1 = 'Test error'



# Parsed testcases at query #26
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = 'C'
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 'D'
    var_7 = 'D'



# Parsed testcases at query #27
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random.seed
    var_2 = var_0.reseed()
    var_3 = 42
    var_4 = var_0.reseed(var_3)
    var_5 = None
    var_6 = var_0.reseed(var_5)
    var_7 = module_0.BaseProvider()
    var_8 = var_7.reseed()
    var_9 = 456
    var_10 = module_0.BaseProvider(seed=var_9)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    var_1 = 'test_provider'
    var_2 = 'test_data.json'



# Parsed testcases at query #29
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'ru'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    assert var_2 == 'en'



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test override_locale method of BaseDataProvider.'



# Parsed testcases at query #31
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    var_1 = module_0.BaseProvider()
    var_2 = 'All tests passed.'
    var_3 = print(var_2)



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'test_data.json'



# Parsed testcases at query #33
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Test reseed method of BaseProvider.'
    var_1 = module_0.BaseProvider()
    var_2 = 42
    var_3 = var_1.reseed(var_2)
    var_4 = None
    var_5 = var_1.reseed(var_4)



# Parsed testcases at query #34
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.get_current_locale()
    var_2 = var_0.get_current_locale()



# Parsed testcases at query #35
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = var_1.reseed()



# Parsed testcases at query #36
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Test method override_locale of class BaseDataProvider.'
    var_1 = module_0.BaseDataProvider(var_0)



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'test_data.json'
    var_2 = 'en'
    assert var_2 == 'ru'
    var_3 = 'Test exception'
    assert var_3 == 'ru_RU'
    var_4 = 'en_US'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test the `override_locale` method of `BaseDataProvider`.\n\n    This test ensures that the `override_locale` method temporarily changes the locale\n    of a `BaseDataProvider` instance and restores the original locale after exiting the context.\n    '
    var_1 = 'test_provider'
    var_2 = 'test_data.json'



# Parsed testcases at query #39
#--------------------------


def test_case_0():
    var_0 = 'mock'
    var_1 = 'mock.json'



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    var_1 = 'mock'
    var_2 = 'mock.json'



# Parsed testcases at query #41
#--------------------------


def test_case_0():
    var_0 = 'mock'
    var_1 = 'mock.json'



# Parsed testcases at query #42
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Test method validate_enum of class BaseProvider.'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = module_0.BaseProvider()
    var_5 = None
    var_6 = 'invalid'
    var_7 = 'Expected NonEnumerableError'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #43
#--------------------------




# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'



# Parsed testcases at query #45
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    assert var_0 == 'en'
    assert var_0 == 'ru'
    var_1 = module_0.BaseDataProvider()
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    var_4 = 'Expected ValueError for invalid locale'
    var_5 = AssertionError(var_4)
    var_6 = var_1.get_current_locale()



# Parsed testcases at query #46
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Test reseed method of BaseProvider.'
    var_1 = module_0.BaseProvider()
    var_2 = var_1.reseed()
    var_3 = None
    var_4 = var_1.reseed(var_3)
    var_5 = 42
    var_6 = var_1.reseed(var_5)
    var_7 = 'test_seed'
    var_8 = var_1.reseed(var_7)
    var_9 = 123
    var_10 = var_1.reseed(var_9)
    var_11 = 456
    var_12 = var_1.reseed(var_11)



# Parsed testcases at query #47
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = None
    var_5 = var_1.reseed(var_4)



# Parsed testcases at query #48
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #49
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = var_1.reseed(var_0)
    var_3 = None
    var_4 = var_1.reseed(var_3)



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 'test.json'
    var_1 = 'test_provider'



# Parsed testcases at query #51
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)
    var_4 = var_3.reseed(var_2)
    var_5 = None
    var_6 = module_0.BaseProvider(seed=var_5)
    var_7 = var_6.reseed(var_5)
    var_8 = 123
    var_9 = module_0.BaseProvider()
    var_10 = var_9.reseed()
    var_11 = 'test_BaseProvider_reseed passed successfully'
    var_12 = print(var_11)



# Parsed testcases at query #52
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Unit test for method reseed of class BaseProvider.'
    var_1 = 12345
    var_2 = module_0.BaseProvider(seed=var_1)
    var_3 = var_2.reseed(var_1)
    var_4 = None
    var_5 = module_0.BaseProvider(seed=var_4)
    var_6 = var_5.reseed(var_4)
    var_7 = var_5.reseed(var_1)
    var_8 = module_0.BaseProvider(seed=var_1)
    var_9 = module_0.BaseProvider(seed=var_1)
    var_10 = var_9.reseed(var_4)
    var_11 = module_0.BaseProvider(seed=var_4)
    var_12 = var_11.reseed(var_1)
    var_13 = module_0.BaseProvider(seed=var_4)
    var_14 = var_13.reseed(var_4)
    var_15 = 67890
    var_16 = module_0.BaseProvider(seed=var_1)
    var_17 = var_16.reseed(var_15)



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'test_data.json'
    var_2 = 'Test exception'



# Parsed testcases at query #54
#--------------------------


def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    var_1 = 'test_provider'
    var_2 = 'test.json'



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    assert var_0 == 'ru'
    var_1 = 'mock'
    var_2 = 'mock.json'
    var_3 = 'en'
    var_4 = 'ru'
    var_5 = 'key'
    var_6 = 'value_en'
    var_7 = {var_5: var_6}
    var_8 = 'value_ru'
    var_9 = {var_5: var_8}
    var_10 = {var_3: var_7, var_4: var_9}
    var_11 = 'mock.json'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'invalid_locale'



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.base as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = module_1.Random()
    var_3 = module_0.BaseProvider(random=var_2)
    var_4 = module_0.BaseProvider(seed=var_0, random=var_2)
    var_5 = 'invalid'
    var_6 = module_0.BaseProvider(random=var_5)



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'locale_dependent'
    var_1 = 'datafile.json'
    var_2 = 'non_locale_dependent'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    var_1 = 'mock'
    var_2 = 'mock.json'



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    assert var_2 == 'ru'
    var_3 = var_1.get_current_locale()
    assert var_3 == 'en'
    var_4 = 'Test exception'
    var_5 = var_1.get_current_locale()
    assert var_5 == 'en'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    var_1 = 'mock'
    var_2 = 'mock.json'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Unit test for method override_locale of class BaseDataProvider'
    var_1 = 'test_provider'
    var_2 = 'test.json'
    var_3 = 'test_provider_no_locale'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'test_data.json'



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Unit test for method validate_enum of class BaseProvider.'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = module_0.BaseProvider()
    var_5 = None
    var_6 = 4



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    assert var_2 == 'ru'
    var_3 = var_1.get_current_locale()
    assert var_3 == 'en'



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Test reseed method of BaseProvider.'
    var_1 = module_0.BaseProvider()
    var_2 = var_1.reseed()
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = var_4.reseed(var_3)
    var_6 = None
    var_7 = module_0.BaseProvider(seed=var_6)
    var_8 = var_7.reseed(var_6)
    var_9 = module_0.BaseProvider()
    var_10 = var_9.reseed()



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.BaseProvider()
    var_3 = None
    var_4 = 3



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    assert var_2 == 'ru'
    var_3 = var_1.get_current_locale()
    assert var_3 == 'en'



# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------




# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'mock'
    var_1 = 'mock_data.json'
    var_2 = 'en'
    assert var_2 == 'en_us_value'
    var_3 = 'en-US'
    var_4 = 'key'
    var_5 = 'en_value'
    var_6 = {var_4: var_5}
    var_7 = 'en_us_value'
    var_8 = {var_4: var_7}



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Test reseed method of BaseProvider.'
    var_1 = module_0.BaseProvider()
    var_2 = var_1.random
    var_3 = var_1.reseed()
    var_4 = 42
    var_5 = module_0.BaseProvider(seed=var_4)
    var_6 = var_5.random
    var_7 = 123
    var_8 = var_5.reseed(var_7)
    var_9 = None
    var_10 = module_0.BaseProvider(seed=var_9)
    var_11 = var_10.random
    var_12 = var_10.reseed(var_9)



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = None
    var_5 = var_1.reseed(var_4)



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 4



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 'D'



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 4
    var_7 = 'A'



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 4



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'



# Parsed testcases at query #26
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.BaseProvider()
    var_2 = None
    var_3 = 'invalid'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = 'Test exception'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    var_1 = 'test_provider'
    var_2 = 'test_data.json'
    var_3 = 'non_locale_provider'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'test_provider'
    var_1 = 'test.json'



# Parsed testcases at query #30
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = module_0.BaseProvider()
    var_5 = None
    var_6 = var_4.reseed(var_5)
    var_7 = module_0.BaseProvider()
    var_8 = module_0.BaseProvider()



# Parsed testcases at query #31
#--------------------------




# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    var_1 = 'mock'
    var_2 = 'mock.json'
    var_3 = 'non_locale'
    var_4 = None



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    var_1 = 'mock'
    var_2 = 'mock.json'
    var_3 = 'non_locale'
    var_4 = None



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'Test override_locale method of BaseDataProvider.'
    assert var_0 == 'ru'
    var_1 = 'test_provider'
    var_2 = 'test_data.json'
    var_3 = 'Test exception'



# Parsed testcases at query #35
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 42
    var_2 = var_0.reseed(var_1)
    var_3 = None
    var_4 = var_0.reseed(var_3)



# Parsed testcases at query #36
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 4
    var_6 = 'A'



# Parsed testcases at query #37
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'Test validate_enum method of BaseProvider.'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = 'C'
    var_4 = module_0.BaseProvider()
    var_5 = None
    var_6 = 'D'



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'Test the override_locale method of BaseDataProvider.'
    var_1 = 'mock'
    var_2 = 'mock.json'



# Parsed testcases at query #39
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 'invalid'



# Parsed testcases at query #40
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 4



