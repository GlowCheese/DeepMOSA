####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.reseed()
    var_5 = 'custom'
    var_6 = 100
    var_7 = var_0.reseed(var_6)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = var_0.personal
    var_3 = var_0.reseed(var_1)
    var_4 = var_0.personal



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'personal'



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'custom'
    var_2 = 'custom'
    var_3 = hasattr(var_0, var_2)
    var_4 = 'custom_kwargs'
    var_5 = 'test_kwargs'
    var_6 = 'custom_kwargs'
    var_7 = hasattr(var_0, var_6)
    var_8 = 'not_a_class'
    var_9 = var_0.add_provider(var_8)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 42
    var_1 = 100



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'custom'
    var_2 = 'custom'
    var_3 = hasattr(var_0, var_2)
    var_4 = var_0.custom
    var_5 = 'not_a_class'
    var_6 = var_0.add_provider(var_5)
    var_7 = 'kwargs'
    var_8 = 'test_value'
    var_9 = 'seedtest'
    var_10 = 999



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = module_0.Generic()



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.personal
    var_2 = var_0.personal



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = var_0.__getattr__(var_1)
    var_3 = hasattr(var_0, var_1)
    var_4 = 'non_existent_provider'
    var_5 = var_0.__getattr__(var_4)
    assert var_5 is None
    var_6 = module_0.Generic()
    var_7 = 'food'
    var_8 = var_6.__getattr__(var_7)
    var_9 = hasattr(var_6, var_7)



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.reseed()



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    assert var_2 is False
    var_3 = var_0.person
    var_4 = hasattr(var_0, var_1)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'custom'
    var_2 = 'custom'
    var_3 = hasattr(var_0, var_2)
    var_4 = var_0.custom
    var_5 = 'another'
    var_6 = 'test'
    var_7 = 'another'
    var_8 = hasattr(var_0, var_7)
    var_9 = var_0.another
    var_10 = 'not a class'
    var_11 = var_0.add_provider(var_10)
    var_12 = 'seedtest'
    var_13 = 42



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.person
    var_2 = var_0._person
    var_3 = callable(var_2)



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'seed'
    var_2 = hasattr(var_0, var_1)
    var_3 = 42
    var_4 = module_0.Generic(seed=var_3)
    var_5 = 'Generic'
    var_6 = hasattr(var_0, var_5)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test __getattr__ method of Generic class.'



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    assert var_2 is False
    var_3 = hasattr(var_0, var_1)
    assert var_3 is True
    var_4 = var_0.person
    var_5 = var_0.locale
    var_6 = var_0.seed



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.reseed()



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    assert var_2 is False
    var_3 = var_0.person
    var_4 = hasattr(var_0, var_1)
    assert var_4 is True
    var_5 = var_0.person
    var_6 = var_0._food
    var_7 = var_0.food



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = var_0.__getattr__(var_1)
    var_3 = var_0.__dir__()
    var_4 = 'non_existent_provider'
    var_5 = var_0.__getattr__(var_4)
    assert var_5 is None
    var_6 = var_0.__getattr__(var_1)
    var_7 = var_0.__getattr__(var_1)



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.personal
    var_2 = var_0.personal
    var_3 = var_0.personal
    var_4 = 42
    var_5 = module_0.Generic(seed=var_4)



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    assert var_2 is False
    var_3 = var_0.person
    var_4 = hasattr(var_0, var_1)
    assert var_4 is True
    var_5 = var_0._person



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = module_0.Generic()
    var_2 = 'custom'
    var_3 = hasattr(var_1, var_2)
    var_4 = var_1.custom
    var_5 = 'custom_kwargs'
    var_6 = 'test'
    var_7 = 'custom_kwargs'
    var_8 = hasattr(var_1, var_7)
    var_9 = var_1.custom_kwargs
    var_10 = 'customproviderno_meta'
    var_11 = hasattr(var_1, var_10)
    var_12 = var_1.customproviderno_meta
    var_13 = 'not_a_class'
    var_14 = var_1.add_provider(var_13)
    var_15 = 'custom_seed'
    var_16 = 42



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.person
    var_4 = 'female'
    var_5 = 'custom'
    var_6 = 'custom'
    var_7 = hasattr(var_0, var_6)
    var_8 = var_0.custom
    var_9 = 'nometa'
    var_10 = hasattr(var_0, var_9)
    var_11 = var_0.nometa



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.reseed()



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test that reseed method correctly resets the seed for all providers.'
    var_1 = 42
    var_2 = 123



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = 'Test adding a custom provider to Generic.'
    var_1 = 'custom'
    var_2 = module_0.Generic()
    var_3 = 'custom'
    var_4 = hasattr(var_2, var_3)
    var_5 = var_2.custom
    var_6 = 'not_a_class'
    var_7 = var_2.add_provider(var_6)
    var_8 = 42
    var_9 = module_0.Generic(seed=var_8)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 42
    var_1 = 100



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = var_0.__getattr__(var_1)
    var_3 = var_0.__getattr__(var_1)
    var_4 = 'non_existent_provider'
    var_5 = var_0.__getattr__(var_4)
    assert var_5 is None



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = hasattr(var_0, var_1)
    assert var_2 is False
    var_3 = var_0.person
    var_4 = hasattr(var_0, var_1)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'seed'
    var_2 = hasattr(var_0, var_1)
    var_3 = 42
    var_4 = module_0.Generic(seed=var_3)
    var_5 = 'Generic'
    var_6 = hasattr(var_0, var_5)



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 'person'
    var_2 = var_0.__getattr__(var_1)
    var_3 = var_0.__getattr__(var_1)
    var_4 = 'non_existent_provider'
    var_5 = var_0.__getattr__(var_4)
    assert var_5 is None
    var_6 = 'fr'
    var_7 = module_0.Generic(var_6)
    var_8 = var_7.__getattr__(var_1)



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.reseed()
    var_5 = 'de'
    var_6 = module_0.Generic(var_5)
    var_7 = 100
    var_8 = var_6.reseed(var_7)
    var_9 = 'custom'
    var_10 = 200
    var_11 = var_0.reseed(var_10)



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 42
    var_1 = 100



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)
    var_4 = var_0.reseed()



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'person'
    var_1 = 'address'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 42
    var_1 = 100



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.personal
    var_2 = var_0.personal
    var_3 = var_0.personal
    var_4 = 42
    var_5 = module_0.Generic(seed=var_4)
    var_6 = var_5.personal



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.generic as module_0

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = var_0.seed
    var_2 = 42
    var_3 = var_0.reseed(var_2)



# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 42



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.generic as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.Generic()
    var_1 = 42
    var_2 = module_0.Generic(seed=var_1)
    var_3 = var_0.reseed(var_1)
    var_4 = var_2.reseed(var_1)
    var_5 = 123
    var_6 = var_0.reseed(var_5)
    var_7 = 100
    var_8 = module_0.Generic(seed=var_7)
    var_9 = 'person'
    var_10 = 'address'
    var_11 = 'datetime'
    var_12 = module_1.date()
    var_13 = 200
    var_14 = var_8.reseed(var_13)
    var_15 = module_1.date()



