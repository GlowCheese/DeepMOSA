####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'



# Parsed testcases at query #2
#--------------------------


import factory.builder as module_0
import mimesis.plugins.factory as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Resolver()
    var_2 = 0
    var_3 = None
    var_4 = module_0.BuildStep(var_3)
    var_5 = 'MockBuilder'
    var_6 = ()
    var_7 = 'factory_meta'
    var_8 = 'MockMeta'
    var_9 = ()
    var_10 = 'declarations'
    var_11 = {}
    var_12 = {var_10: var_11}
    var_13 = type(var_8, var_9, var_12)
    var_14 = 'gender'
    var_15 = 'female'
    var_16 = {var_14: var_15}
    var_17 = 'custom_field'
    var_18 = 'custom_value'
    var_19 = lambda : var_18
    var_20 = {var_17: var_19}
    var_21 = module_1.FactoryField(var_17)
    var_22 = var_21.evaluate(var_1, var_4)
    assert var_22 == 'custom_value'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'gender'
    var_2 = 'female'
    var_3 = {var_1: var_2}
    var_4 = 'another_param'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 'field_handlers'
    var_8 = []



# Parsed testcases at query #4
#--------------------------


import factory.builder as module_0
import mimesis.plugins.factory as module_1

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.Resolver()
    var_2 = None
    var_3 = module_0.BuildStep(var_2, var_2, var_2)
    var_4 = 'gender'
    var_5 = 'female'
    var_6 = {var_4: var_5}
    var_7 = 'custom_field'
    var_8 = 'custom_value'
    var_9 = lambda : var_8
    var_10 = {var_7: var_9}
    var_11 = 'field_handlers'
    var_12 = module_1.FactoryField(var_7)
    var_13 = var_12.evaluate(var_1, var_3)
    assert var_13 == 'custom_value'



# Parsed testcases at query #5
#--------------------------


import factory.builder as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Resolver()
    var_2 = None
    var_3 = module_0.BuildStep(var_2)
    var_4 = 'MockBuilder'
    var_5 = ()
    var_6 = 'factory_meta'
    var_7 = 'MockMeta'
    var_8 = ()
    var_9 = 'declarations'
    var_10 = {}
    var_11 = {var_9: var_10}
    var_12 = type(var_7, var_8, var_11)
    var_13 = 'gender'
    var_14 = 'female'
    var_15 = {var_13: var_14}
    var_16 = 'custom_field'
    var_17 = 'custom_value'
    var_18 = lambda : var_17
    var_19 = {var_16: var_18}



# Parsed testcases at query #6
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'address.street_name'
    var_2 = 'value'
    var_3 = 'datetime.date'
    var_4 = module_0.FactoryField(var_3)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'gender'
    var_2 = 'female'
    var_3 = {var_1: var_2}
    var_4 = 'age'
    var_5 = 30
    var_6 = {var_4: var_5}



# Parsed testcases at query #9
#--------------------------


import factory.builder as module_0
import mimesis.plugins.factory as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Resolver()
    var_2 = None
    var_3 = 'test_step'
    var_4 = module_0.BuildStep(var_2)
    var_5 = 'gender'
    var_6 = 'female'
    var_7 = {var_5: var_6}
    var_8 = 'custom_field'
    var_9 = 'custom_value'
    var_10 = lambda : var_9
    var_11 = {var_8: var_10}
    var_12 = 'field_handlers'
    var_13 = module_1.FactoryField(var_8)
    var_14 = var_13.evaluate(var_1, var_4)
    assert var_14 == 'custom_value'



# Parsed testcases at query #10
#--------------------------


import mimesis.plugins.factory as module_0
import factory.builder as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = module_1.Resolver()
    var_3 = None
    var_4 = module_1.BuildStep(var_3)
    var_5 = 'gender'
    var_6 = 'female'
    var_7 = {var_5: var_6}
    var_8 = var_1.evaluate(var_2, var_4, var_7)
    var_9 = 'male'
    var_10 = module_0.FactoryField(var_0)
    var_11 = var_10.evaluate(var_2, var_4)
    var_12 = module_0.FactoryField(var_0)
    var_13 = var_12.evaluate(var_2, var_4)
    var_14 = 'custom_field'
    var_15 = 'custom_value'
    var_16 = lambda : var_15
    var_17 = {var_14: var_16}
    var_18 = module_1.BuildStep(var_3)
    var_19 = 'field_handlers'
    var_20 = module_0.FactoryField(var_14)
    var_21 = var_20.evaluate(var_2, var_18)
    assert var_21 == 'custom_value'



# Parsed testcases at query #11
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 'person'
    var_4 = 30
    var_5 = 'female'
    var_6 = module_0.FactoryField(var_3)
    var_7 = 'datetime'
    var_8 = 1990
    var_9 = 2000
    var_10 = '%Y-%m-%d'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'gender'
    var_2 = 'female'
    var_3 = {var_1: var_2}
    var_4 = 'another_param'
    var_5 = 'value'
    var_6 = {var_4: var_5}



# Parsed testcases at query #13
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'



# Parsed testcases at query #14
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 'Berlin'



# Parsed testcases at query #15
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'text.word'
    var_1 = 'length'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = 10
    var_5 = {var_1: var_4}



# Parsed testcases at query #17
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'



# Parsed testcases at query #18
#--------------------------


import factory.builder as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Resolver()
    var_2 = None
    var_3 = module_0.BuildStep(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}



# Parsed testcases at query #19
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'gender'
    var_2 = 'female'
    var_3 = {var_1: var_2}
    var_4 = 'person.email'
    var_5 = 'custom_field'
    var_6 = 'custom_value'
    var_7 = lambda : var_6
    var_8 = {var_5: var_7}
    var_9 = module_0.FactoryField(var_5)



# Parsed testcases at query #20
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'
    var_5 = 30



# Parsed testcases at query #21
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 'Berlin'



# Parsed testcases at query #22
#--------------------------


import mimesis.plugins.factory as module_0
import factory.builder as module_1

def test_case_0():
    var_0 = 'person.name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = module_1.Resolver()
    var_3 = None
    var_4 = module_1.BuildStep(var_3, var_3, var_3)
    var_5 = var_1.evaluate(var_2, var_4)
    var_6 = len(var_5)
    var_7 = 'gender'
    var_8 = 'female'
    var_9 = {var_7: var_8}
    var_10 = var_1.evaluate(var_2, var_4, var_9)
    var_11 = len(var_10)
    var_12 = 'field_handlers'
    var_13 = 'custom_field'
    var_14 = module_0.FactoryField(var_13)
    var_15 = var_14.evaluate(var_2, var_4)
    assert var_15 == 'custom_value'



# Parsed testcases at query #23
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'



# Parsed testcases at query #24
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'person.age'
    var_3 = 18
    var_4 = 99
    var_5 = module_0.FactoryField(var_2)
    var_6 = 'address.city'
    var_7 = 'FR'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'text.word'
    var_1 = 'length'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = 'uppercase'
    var_5 = True
    var_6 = {var_4: var_5}
    var_7 = 'custom'
    var_8 = 'custom_value'
    var_9 = lambda : var_8



# Parsed testcases at query #26
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'gender'
    var_2 = 'female'
    var_3 = {var_1: var_2}
    var_4 = 'person.email'
    var_5 = 'custom_field'
    var_6 = 'custom_value'
    var_7 = lambda : var_6
    var_8 = {var_5: var_7}
    var_9 = 'field_handlers'
    var_10 = module_0.FactoryField(var_5)



# Parsed testcases at query #27
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'



# Parsed testcases at query #28
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 10



# Parsed testcases at query #29
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 'person'
    var_4 = 'female'
    var_5 = module_0.FactoryField(var_3)
    var_6 = 'datetime'
    var_7 = '%Y-%m-%d'



# Parsed testcases at query #30
#--------------------------


import factory.builder as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Resolver()
    var_2 = None
    var_3 = module_0.BuildStep(var_2)
    var_4 = 'gender'
    var_5 = 'female'
    var_6 = {var_4: var_5}



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 'person'
    var_4 = 30
    var_5 = 'male'
    var_6 = module_0.FactoryField(var_3)
    var_7 = 'datetime'
    var_8 = 2020
    var_9 = 2023



# Parsed testcases at query #2
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #3
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'person.age'
    var_3 = 18
    var_4 = 65
    var_5 = module_0.FactoryField(var_2)
    var_6 = 'address.city'
    var_7 = 'ES'



# Parsed testcases at query #4
#--------------------------


import factory.builder as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Resolver()
    var_2 = None
    var_3 = module_0.BuildStep(var_2)
    var_4 = 'gender'
    var_5 = 'female'
    var_6 = {var_4: var_5}



# Parsed testcases at query #5
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'
    var_5 = 30



# Parsed testcases at query #6
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'



# Parsed testcases at query #7
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'
    var_5 = 30



# Parsed testcases at query #8
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'female'
    var_2 = 'age'
    var_3 = 30
    var_4 = {var_2: var_3}



# Parsed testcases at query #10
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address.city'
    var_3 = 'datetime.date'
    var_4 = 2000
    var_5 = 2020
    var_6 = module_0.FactoryField(var_3)
    var_7 = 'person.email'
    var_8 = True
    var_9 = 10



# Parsed testcases at query #11
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'female'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'male'



# Parsed testcases at query #12
#--------------------------


import mimesis.plugins.factory as module_0
import factory.builder as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = module_1.Resolver()
    var_3 = None
    var_4 = module_1.BuildStep(var_3, var_3, var_3)
    var_5 = var_1.evaluate(var_2, var_4)
    var_6 = 'gender'
    var_7 = 'female'
    var_8 = {var_6: var_7}
    var_9 = var_1.evaluate(var_2, var_4, var_8)
    var_10 = 'custom_field'
    var_11 = 'custom_value'
    var_12 = lambda : var_11
    var_13 = {var_10: var_12}
    var_14 = 'field_handlers'
    var_15 = module_0.FactoryField(var_10)
    var_16 = var_15.evaluate(var_2, var_4)
    assert var_16 == 'custom_value'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'gender'
    var_2 = 'female'
    var_3 = {var_1: var_2}
    var_4 = 'length'
    var_5 = 10
    var_6 = {var_4: var_5}



# Parsed testcases at query #14
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = None
    var_3 = var_1.evaluate(var_2, var_2)
    var_4 = len(var_3)
    var_5 = 'person.age'
    var_6 = 18
    var_7 = 99
    var_8 = module_0.FactoryField(var_5)
    var_9 = var_8.evaluate(var_2, var_2)
    var_10 = 'de'
    var_11 = module_0.FactoryField(var_0, var_10)
    var_12 = var_11.evaluate(var_2, var_2)
    var_13 = len(var_12)
    var_14 = module_0.FactoryField(var_5)
    var_15 = 'minimum'
    var_16 = 'maximum'
    var_17 = 25
    var_18 = 65
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = var_14.evaluate(var_2, var_2, var_19)
    assert var_20 == 'test'
    var_21 = 'custom'
    var_22 = 'test'
    var_23 = lambda : var_22
    var_24 = module_0.FactoryField(var_21)



# Parsed testcases at query #15
#--------------------------


import factory.builder as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Resolver()
    var_2 = None
    var_3 = module_0.BuildStep(var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}



# Parsed testcases at query #16
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 'person'
    var_4 = 'female'
    var_5 = module_0.FactoryField(var_3)
    var_6 = 'datetime'
    var_7 = '%Y-%m-%d'



# Parsed testcases at query #17
#--------------------------


import factory.builder as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Resolver()
    var_2 = None
    var_3 = module_0.BuildStep(var_2, var_2, var_2)
    var_4 = 'gender'
    var_5 = 'female'
    var_6 = {var_4: var_5}
    var_7 = 'field_handlers'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'value'



# Parsed testcases at query #19
#--------------------------


import factory.builder as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.Resolver()
    var_2 = 0
    var_3 = module_0.BuildStep(var_1)
    var_4 = 'gender'
    var_5 = 'female'
    var_6 = {var_4: var_5}



# Parsed testcases at query #20
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'person.age'
    var_3 = 18
    var_4 = 65
    var_5 = module_0.FactoryField(var_2)
    var_6 = 'address.city'
    var_7 = 'Catalonia'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'gender'
    var_2 = 'female'
    var_3 = {var_1: var_2}



# Parsed testcases at query #22
#--------------------------


import factory.builder as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Resolver()
    var_2 = None
    var_3 = module_0.BuildStep(var_2)
    var_4 = 'gender'
    var_5 = 'female'
    var_6 = {var_4: var_5}
    var_7 = 'address'
    var_8 = 'city'
    var_9 = 'London'
    var_10 = {var_8: var_9}
    var_11 = 'custom_field'
    var_12 = 'custom_value'
    var_13 = lambda : var_12
    var_14 = {var_11: var_13}
    var_15 = 'MockMeta'
    var_16 = ()
    var_17 = 'declarations'
    var_18 = 'field_handlers'
    var_19 = {var_18: var_14}
    var_20 = {var_17: var_19}
    var_21 = type(var_15, var_16, var_20)
    var_22 = module_0.BuildStep(var_2)



# Parsed testcases at query #23
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address.city'
    var_3 = 'datetime.date'
    var_4 = 2000
    var_5 = 2020
    var_6 = module_0.FactoryField(var_3)
    var_7 = 'text.word'
    var_8 = 5
    var_9 = 10



# Parsed testcases at query #24
#--------------------------


import factory.builder as module_0
import mimesis.plugins.factory as module_1

def test_case_0():
    var_0 = 'person.name'
    var_1 = module_0.Resolver()
    var_2 = None
    var_3 = module_0.BuildStep(var_2, var_2, var_2)
    var_4 = 'field_handlers'
    var_5 = []
    var_6 = 'gender'
    var_7 = 'female'
    var_8 = {var_6: var_7}
    var_9 = 'custom_field'
    var_10 = 'custom_value'
    var_11 = lambda : var_10
    var_12 = {var_9: var_11}
    var_13 = module_1.FactoryField(var_9)
    var_14 = var_13.evaluate(var_1, var_3)
    assert var_14 == 'custom_value'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'gender'
    var_2 = 'female'
    var_3 = {var_1: var_2}
    var_4 = 'another_param'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = []



# Parsed testcases at query #26
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'gender'
    var_2 = 'female'
    var_3 = {var_1: var_2}
    var_4 = 'age'
    var_5 = 30
    var_6 = {var_4: var_5}
    var_7 = module_0.FactoryField(var_0, **var_3)
    var_8 = 'custom_field'
    var_9 = 'custom_value'
    var_10 = lambda : var_9
    var_11 = {var_8: var_10}
    var_12 = module_0.FactoryField(var_8)



# Parsed testcases at query #27
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 'person'
    var_4 = 30
    var_5 = 'female'
    var_6 = module_0.FactoryField(var_3)
    var_7 = 'datetime'
    var_8 = 2000
    var_9 = 2020



# Parsed testcases at query #28
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 10
    var_4 = 'email'
    var_5 = 'example.com'
    var_6 = True
    var_7 = module_0.FactoryField(var_4)



# Parsed testcases at query #29
#--------------------------


import factory.builder as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Resolver()
    var_2 = 0
    var_3 = None
    var_4 = module_0.BuildStep(var_2, var_3, var_3)
    var_5 = 'gender'
    var_6 = 'female'
    var_7 = {var_5: var_6}
    var_8 = 'custom_field'
    var_9 = 'custom_value'
    var_10 = lambda : var_9
    var_11 = {var_8: var_10}
    var_12 = 'field_handlers'



# Parsed testcases at query #30
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 'value'
    var_4 = 'email'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = module_0.FactoryField(var_4)



# Parsed testcases at query #31
#--------------------------


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'address'
    var_3 = 'person'
    var_4 = 30
    var_5 = 'male'
    var_6 = module_0.FactoryField(var_3)
    var_7 = 'datetime'
    var_8 = 2000
    var_9 = 2020



# Parsed testcases at query #32
#--------------------------


import factory.builder as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'gender'
    var_2 = 'female'
    var_3 = {var_1: var_2}
    var_4 = 'age'
    var_5 = 30
    var_6 = {var_4: var_5}
    var_7 = module_0.Resolver()
    var_8 = None
    var_9 = 'test_step'
    var_10 = False
    var_11 = module_0.BuildStep(var_8)
    var_12 = 'MockBuilder'
    var_13 = ()
    var_14 = 'factory_meta'
    var_15 = 'MockMeta'
    var_16 = ()
    var_17 = 'declarations'
    var_18 = 'field_handlers'
    var_19 = []
    var_20 = {var_18: var_19}
    var_21 = {var_17: var_20}
    var_22 = type(var_15, var_16, var_21)



