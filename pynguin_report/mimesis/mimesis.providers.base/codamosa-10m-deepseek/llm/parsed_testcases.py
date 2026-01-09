####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 'invalid'
    var_7 = 'test_BaseProvider_validate_enum passed'
    var_8 = print(var_7)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'Enum'
    var_2 = ()
    var_3 = 'value'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = None
    var_7 = 'invalid'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'mock'
    var_1 = 'test.json'
    var_2 = 'Test exception'
    var_3 = 'All tests passed!'
    var_4 = print(var_3)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'non_locale'
    var_1 = 'nonexistent.json'



# Parsed testcases at query #7
#--------------------------


import mimesis.random as module_1


def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)
    var_4 = module_1.Random()
    var_5 = module_0.BaseProvider(random=var_4)
    var_6 = module_0.BaseProvider(seed=var_2)
    var_7 = 100
    var_8 = var_6.reseed(var_7)
    var_9 = module_0.BaseProvider(seed=var_2)
    var_10 = module_0.BaseProvider(seed=var_2)
    var_11 = None
    var_12 = module_0.BaseProvider(seed=var_11)
    var_13 = module_0.BaseProvider()
    var_14 = str(var_13)
    assert var_14 == 'BaseProvider'



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = module_0.BaseProvider()
    var_3 = 42
    var_4 = var_2.reseed(var_3)
    var_5 = var_2.reseed(var_3)
    var_6 = module_0.BaseProvider()
    var_7 = None
    var_8 = var_6.reseed(var_7)
    var_9 = module_0.BaseProvider()
    var_10 = 12345
    var_11 = var_9.reseed(var_10)
    var_12 = 1
    var_13 = 100
    var_14 = var_9.reseed(var_10)
    var_15 = module_0.BaseProvider()
    var_16 = 'test_seed'
    var_17 = var_15.reseed(var_16)
    var_18 = var_15.reseed(var_16)
    var_19 = module_0.BaseProvider()
    var_20 = module_0.BaseProvider()



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)
    var_4 = module_1.Random()
    var_5 = module_0.BaseProvider(random=var_4)
    var_6 = 'not_a_random_instance'
    var_7 = module_0.BaseProvider(random=var_6)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)
    var_4 = var_3.random
    var_5 = module_1.Random()
    var_6 = module_0.BaseProvider(random=var_5)
    var_7 = module_0.BaseProvider(seed=var_2)
    var_8 = 100
    var_9 = var_7.reseed(var_8)
    var_10 = 'a'
    var_11 = 'b'
    var_12 = None
    var_13 = 'invalid'
    var_14 = module_0.BaseProvider()
    var_15 = module_0.BaseProvider(seed=var_2)
    var_16 = str(var_15)
    assert var_16 == 'BaseProvider'



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = module_0.BaseProvider()
    var_3 = 42
    var_4 = var_2.reseed(var_3)
    var_5 = var_2.reseed(var_3)
    var_6 = module_0.BaseProvider()
    var_7 = None
    var_8 = var_6.reseed(var_7)
    var_9 = module_0.BaseProvider()
    var_10 = 12345
    var_11 = var_9.reseed(var_10)
    var_12 = module_0.BaseProvider()
    var_13 = 'test_seed'
    var_14 = var_12.reseed(var_13)
    var_15 = module_0.BaseProvider()
    var_16 = var_15.reseed()
    var_17 = 'All tests passed for BaseProvider.reseed()'
    var_18 = print(var_17)



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = var_0.reseed()
    var_3 = module_0.BaseProvider()
    var_4 = 42
    var_5 = var_3.reseed(var_4)
    var_6 = module_0.BaseProvider()
    var_7 = None
    var_8 = var_6.reseed(var_7)
    var_9 = module_0.BaseProvider()
    var_10 = var_9.reseed()



# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'value_a'
    var_2 = 'value_b'
    var_3 = 'value_c'
    var_4 = None
    var_5 = 'invalid'
    var_6 = 'All tests passed for validate_enum method.'
    var_7 = print(var_6)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = 123
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 456
    var_3 = var_1.reseed(var_2)
    var_4 = module_0.BaseProvider(seed=var_0)
    var_5 = None
    var_6 = var_4.reseed(var_5)
    var_7 = 100
    var_8 = module_0.BaseProvider(seed=var_7)
    var_9 = module_0.BaseProvider(seed=var_7)
    var_10 = 5
    var_11 = range(var_10)
    var_12 = [provider1.random.random() for _ in var_11]
    var_13 = range(var_10)
    var_14 = [provider2.random.random() for _ in var_13]
    var_15 = 200
    var_16 = var_8.reseed(var_15)
    var_17 = var_9.reseed(var_15)
    var_18 = range(var_10)
    var_19 = [provider1.random.random() for _ in var_18]
    var_20 = range(var_10)
    var_21 = [provider2.random.random() for _ in var_20]



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = 'value_a'
    var_1 = 'value_b'
    var_2 = 'value_c'
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 'invalid'
    var_7 = 'value_d'



# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 'invalid'
    var_6 = 'd'



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'Test exception'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'Test exception'



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 'invalid'
    var_7 = 'd'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'All tests passed!'
    var_1 = print(var_0)



# Parsed testcases at query #26
#--------------------------




# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test exception'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Test exception'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Test exception'
    assert var_0 == 'fr'



# Parsed testcases at query #30
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 'invalid'



# Parsed testcases at query #31
#--------------------------




# Parsed testcases at query #32
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'Test exception'



# Parsed testcases at query #34
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = None
    var_2 = 'invalid'



# Parsed testcases at query #35
#--------------------------




####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'non_locale_dependent'
    var_1 = ''
    var_2 = 'Test exception'
    var_3 = RuntimeError(var_2)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)
    var_4 = module_1.Random()
    var_5 = module_0.BaseProvider(random=var_4)
    var_6 = 'not a random'
    var_7 = module_0.BaseProvider(random=var_6)



# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test exception'
    assert var_0 == 'de'
    assert var_0 == 'ru'
    assert var_0 == 'en_US'
    var_1 = 'non_locale_dependent'
    var_2 = ''
    var_3 = 'locale_dependent'
    var_4 = 'test.json'
    var_5 = None
    var_6 = 'All tests passed!'
    var_7 = print(var_6)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'non_locale'
    var_1 = None



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Test exception'
    var_1 = 'Test exception'
    var_2 = 'Test exception'
    var_3 = 'Test exception'
    var_4 = 'Test exception'



# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = var_0.reseed()
    var_3 = module_0.BaseProvider()
    var_4 = 42
    var_5 = var_3.reseed(var_4)
    var_6 = module_0.BaseProvider()
    var_7 = None
    var_8 = var_6.reseed(var_7)
    var_9 = 123
    var_10 = module_0.BaseProvider(seed=var_9)
    var_11 = module_0.BaseProvider(seed=var_9)
    var_12 = 1
    var_13 = 100
    var_14 = 456
    var_15 = var_10.reseed(var_14)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = module_0.BaseProvider(seed=var_0)
    var_5 = None
    var_6 = var_4.reseed(var_5)
    var_7 = module_0.BaseProvider(seed=var_0)
    var_8 = module_0.BaseProvider(seed=var_0)
    var_9 = var_8.random
    var_10 = var_8.reseed(var_2)
    var_11 = module_0.BaseProvider()
    var_12 = module_0.BaseProvider()
    var_13 = 456
    var_14 = var_12.reseed(var_13)
    var_15 = module_1.Random()
    var_16 = module_0.BaseProvider(random=var_15)
    var_17 = 789
    var_18 = var_16.reseed(var_17)
    var_19 = module_0.BaseProvider()
    var_20 = 'invalid_seed'
    var_21 = var_19.reseed(var_20)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'random_value'
    var_2 = None
    var_3 = 'enum'
    var_4 = var_0.validate_enum(var_2, var_3)
    assert var_4 == 'random_value'
    assert var_4 == 'enum_value'
    assert var_4 == 'different_value'
    assert var_4 == 'subclass_value'
    assert var_4 == 123
    assert var_4 is None
    assert var_4 == ''
    assert var_4 is True
    assert var_4 == b'hello'
    var_5 = 'enum_value'
    var_6 = 'invalid_item'
    var_7 = 'different_value'
    var_8 = 'subclass_value'
    var_9 = 123
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = 'key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = ''
    var_18 = True
    var_19 = 3.14
    var_20 = (var_18, var_11, var_12)
    var_21 = {var_18, var_11, var_12}
    var_22 = [var_18, var_11, var_12]
    var_23 = frozenset(var_22)
    var_24 = [var_18, var_11, var_12]
    var_25 = frozenset(var_24)
    var_26 = b'hello'
    var_27 = bytearray(var_26)
    var_28 = bytearray(var_26)
    var_29 = memoryview(var_26)
    var_30 = memoryview(var_26)
    var_31 = 5
    var_32 = range(var_31)
    var_33 = range(var_31)
    var_34 = 10
    var_35 = slice(var_18, var_34, var_11)
    var_36 = slice(var_18, var_34, var_11)



# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = module_0.BaseProvider(seed=var_0)
    var_5 = None
    var_6 = var_4.reseed(var_5)
    var_7 = module_0.BaseProvider()
    var_8 = var_7.reseed()
    var_9 = 'test_seed'
    var_10 = module_0.BaseProvider(seed=var_9)
    var_11 = 'new_seed'
    var_12 = var_10.reseed(var_11)
    var_13 = module_0.BaseProvider(seed=var_0)
    var_14 = module_0.BaseProvider(seed=var_0)
    var_15 = var_13.reseed(var_0)
    var_16 = var_14.reseed(var_0)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'All tests passed!'
    var_1 = print(var_0)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test exception'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'Test exception'
    var_1 = 'non_locale_dependent'
    var_2 = ''



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = module_0.BaseProvider(seed=var_0)
    var_5 = None
    var_6 = var_4.reseed(var_5)
    var_7 = module_0.BaseProvider(seed=var_0)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test override_locale method of BaseDataProvider.'
    var_1 = 'mock'
    var_2 = 'test.json'
    var_3 = 'All tests passed!'
    var_4 = print(var_3)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = module_0.BaseProvider(seed=var_3)
    var_5 = None
    var_6 = 'invalid'



# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'value_a'
    var_2 = 'value_b'
    var_3 = 'value_c'
    var_4 = None
    var_5 = 'invalid'
    var_6 = 'value_d'



# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'non_locale'
    var_1 = 'nonexistent.json'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'non_locale'
    var_1 = ''



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'non_locale'



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'locale_independent'
    var_1 = 'test.json'



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'non_locale'
    var_1 = None



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'non_locale'
    var_1 = None



# Parsed testcases at query #31
#--------------------------




# Parsed testcases at query #32
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = None
    var_5 = var_1.reseed(var_4)



# Parsed testcases at query #33
#--------------------------



def test_case_0():
    var_0 = 12345
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 54321
    var_3 = var_1.reseed(var_2)
    var_4 = 'Test case 1 passed: Reseed with specific seed'
    var_5 = print(var_4)
    var_6 = module_0.BaseProvider(seed=var_0)
    var_7 = None
    var_8 = var_6.reseed(var_7)
    var_9 = 'Test case 2 passed: Reseed with None'
    var_10 = print(var_9)
    var_11 = 'Test case 3 passed: Reseed with MissingSeed and global seed set'
    var_12 = print(var_11)
    var_13 = 'Test case 4 passed: Reseed with MissingSeed and no global seed'
    var_14 = print(var_13)
    var_15 = 42
    var_16 = module_0.BaseProvider(seed=var_15)
    var_17 = 5
    var_18 = range(var_17)
    var_19 = 1
    var_20 = 100
    var_21 = [provider1.random.randint(var_19, var_20) for _ in var_18]
    var_22 = module_0.BaseProvider(seed=var_15)
    var_23 = range(var_17)
    var_24 = [provider2.random.randint(var_19, var_20) for _ in var_23]
    var_25 = 'Test case 5 passed: Same seed produces same random sequence'
    var_26 = print(var_25)
    var_27 = module_0.BaseProvider(seed=var_15)
    var_28 = range(var_17)
    var_29 = [provider.random.randint(var_19, var_20) for _ in var_28]
    var_30 = var_27.reseed(var_20)
    var_31 = range(var_17)
    var_32 = [provider.random.randint(var_19, var_20) for _ in var_31]
    var_33 = 'Test case 6 passed: Reseed changes random sequence'
    var_34 = print(var_33)
    var_35 = 'All test cases passed!'
    var_36 = print(var_35)



# Parsed testcases at query #34
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'value_a'
    var_2 = 'value_b'
    var_3 = 'value_c'
    var_4 = None
    var_5 = 'invalid'
    var_6 = 'value_d'
    var_7 = 'All tests passed for validate_enum'
    var_8 = print(var_7)



# Parsed testcases at query #35
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = module_0.BaseProvider(seed=var_0)
    var_5 = None
    var_6 = var_4.reseed(var_5)
    var_7 = module_0.BaseProvider(seed=var_0)



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'Test exception'
    var_1 = 'non_locale_dependent'
    var_2 = ''



# Parsed testcases at query #37
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 123
    var_3 = var_1.reseed(var_2)
    var_4 = module_0.BaseProvider(seed=var_0)
    var_5 = None
    var_6 = var_4.reseed(var_5)
    var_7 = module_0.BaseProvider(seed=var_0)



# Parsed testcases at query #38
#--------------------------



def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = module_0.BaseProvider()
    var_3 = 42
    var_4 = var_2.reseed(var_3)
    var_5 = var_2.reseed(var_3)
    var_6 = module_0.BaseProvider()
    var_7 = None
    var_8 = var_6.reseed(var_7)
    var_9 = module_0.BaseProvider()
    var_10 = 12345
    var_11 = var_9.reseed(var_10)
    var_12 = 'All tests passed for BaseProvider.reseed()'
    var_13 = print(var_12)



# Parsed testcases at query #39
#--------------------------



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




