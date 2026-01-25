####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_6 = 1



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = None
    var_5 = 'invalid'
    var_6 = 'invalid'



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = None
    var_4 = 'invalid'



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    assert var_2 == 'en'



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = None
    var_3 = 'key'
    var_4 = 'value_en'



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.base as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)
    var_4 = var_3.random
    var_5 = module_1.Random()
    var_6 = module_0.BaseProvider(random=var_5)
    var_7 = 'not_a_random_object'
    var_8 = module_0.BaseProvider(random=var_7)
    var_9 = 123
    var_10 = var_0.reseed(var_9)
    var_11 = var_0.reseed()
    var_12 = 42
    var_13 = module_0.BaseProvider(seed=var_12)
    var_14 = str(var_0)
    assert var_14 == 'BaseProvider'



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    var_4 = 'non_locale'
    var_5 = False



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = None
    var_5 = 'd'



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    assert var_2 == 'en'
    var_3 = var_1.get_current_locale()
    assert var_3 == 'en'
    var_4 = 'locale_independent'
    var_5 = False



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = 100
    var_5 = module_0.BaseProvider(seed=var_4)
    var_6 = module_0.BaseProvider(seed=var_4)
    var_7 = 32
    var_8 = 200
    var_9 = var_5.reseed(var_8)
    var_10 = var_6.reseed(var_8)



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.get_current_locale()
    var_2 = var_0.get_current_locale()



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #26
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #27
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #28
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.reseed()
    var_3 = 42
    var_4 = var_0.reseed(var_3)
    var_5 = 100
    var_6 = module_0.BaseProvider(seed=var_5)
    var_7 = module_0.BaseProvider(seed=var_5)
    var_8 = 32
    var_9 = 200
    var_10 = var_6.reseed(var_9)



# Parsed testcases at query #29
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.BaseProvider()
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #30
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseDataProvider()
    var_1 = var_0.get_current_locale()
    var_2 = var_0.get_current_locale()



# Parsed testcases at query #31
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()
    var_5 = 'non_locale'
    var_6 = False



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = None
    var_4 = 'invalid'



# Parsed testcases at query #33
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = None
    var_4 = 'invalid'



# Parsed testcases at query #34
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()
    var_5 = var_1.get_current_locale()



# Parsed testcases at query #35
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = 100
    var_5 = module_0.BaseProvider(seed=var_4)
    var_6 = module_0.BaseProvider(seed=var_4)
    var_7 = 32
    var_8 = module_0.BaseProvider(seed=var_4)
    var_9 = 200
    var_10 = var_8.reseed(var_9)
    var_11 = module_0.BaseProvider()
    var_12 = None
    var_13 = var_11.reseed(var_12)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()
    var_5 = var_1.get_current_locale()



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.base as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.random
    var_2 = 42
    var_3 = module_0.BaseProvider(seed=var_2)
    var_4 = var_3.random
    var_5 = module_1.Random()
    var_6 = module_0.BaseProvider(random=var_5)
    var_7 = module_0.BaseProvider(seed=var_2, random=var_5)
    var_8 = 'not_a_random_object'
    var_9 = module_0.BaseProvider(random=var_8)



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 100
    var_3 = var_1.reseed(var_2)
    var_4 = None
    var_5 = var_1.reseed(var_4)



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.reseed()



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = 100
    var_5 = module_0.BaseProvider(seed=var_4)
    var_6 = module_0.BaseProvider(seed=var_4)
    var_7 = 32
    var_8 = 200
    var_9 = module_0.BaseProvider(seed=var_8)
    var_10 = module_0.BaseProvider()



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = 100
    var_5 = module_0.BaseProvider(seed=var_4)
    var_6 = module_0.BaseProvider(seed=var_4)
    var_7 = 0
    var_8 = 200
    var_9 = module_0.BaseProvider(seed=var_8)



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()



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


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.BaseProvider(seed=var_0)
    var_2 = 100
    var_3 = var_1.reseed(var_2)
    var_4 = None
    var_5 = var_1.reseed(var_4)



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.reseed()
    var_3 = 42
    var_4 = var_0.reseed(var_3)
    var_5 = 100
    var_6 = module_0.BaseProvider(seed=var_5)
    var_7 = module_0.BaseProvider(seed=var_5)
    var_8 = 0
    var_9 = module_0.BaseProvider()
    var_10 = var_9.reseed()



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = None
    var_5 = 'd'



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = None
    var_5 = 'invalid'



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'test.json'
    var_2 = 'en'
    assert var_2 == 'de'



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = None
    var_4 = 'invalid'



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.reseed()
    var_3 = 42
    var_4 = var_0.reseed(var_3)
    var_5 = 10
    var_6 = module_0.BaseProvider(seed=var_5)
    var_7 = module_0.BaseProvider(seed=var_5)
    var_8 = 0
    var_9 = 100
    var_10 = module_0.BaseProvider()
    var_11 = var_10.reseed()



# Parsed testcases at query #26
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = None
    var_4 = 'invalid_item'



# Parsed testcases at query #27
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()



# Parsed testcases at query #28
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = None
    var_5 = var_0.reseed(var_4)



# Parsed testcases at query #29
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()



# Parsed testcases at query #30
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = None
    var_5 = var_0.reseed(var_4)



# Parsed testcases at query #31
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = None
    var_5 = 'd'



# Parsed testcases at query #32
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = var_0.reseed()
    var_3 = 42
    var_4 = var_0.reseed(var_3)
    var_5 = 100
    var_6 = module_0.BaseProvider(seed=var_5)
    var_7 = module_0.BaseProvider(seed=var_5)
    var_8 = 32
    var_9 = 200
    var_10 = var_6.reseed(var_9)



# Parsed testcases at query #33
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #34
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = 'en'
    assert var_0 == 'de'
    var_1 = module_0.BaseDataProvider(var_0)
    var_2 = var_1.get_current_locale()
    var_3 = var_1.get_current_locale()
    assert var_3 == 'de'
    var_4 = var_1.get_current_locale()



# Parsed testcases at query #35
#--------------------------


import mimesis.providers.base as module_0

def test_case_0():
    var_0 = module_0.BaseProvider()
    var_1 = var_0.reseed()
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = 100
    var_5 = module_0.BaseProvider(seed=var_4)
    var_6 = module_0.BaseProvider(seed=var_4)
    var_7 = 0
    var_8 = 200
    var_9 = module_0.BaseProvider(seed=var_8)



