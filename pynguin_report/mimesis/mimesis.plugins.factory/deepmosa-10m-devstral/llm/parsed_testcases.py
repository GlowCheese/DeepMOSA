####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'
    var_2 = 'custom_param'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'test_field'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'custom_param': 'value'})
    assert var_8 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_init_sets_locale_and_kwargs_and_field. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_init_sets_locale. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_field'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_init_assigns_locale_field_and_kwargs. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_factory_field_constructor. Retrieved 3/7 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True
    var_7 = 'value'



# Parsed testcases at query #8
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 30
    var_2 = 'male'
    var_3 = 'age'
    var_4 = 'gender'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'age': 30, 'gender': 'male'})
    assert var_10 is True

def test_case_0():
    var_0 = 'datetime'
    var_1 = '%Y-%m-%d'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_factory_field_constructor_with_custom_locale. Retrieved 1/3 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'female'
    var_2 = 30
    var_3 = 'gender'
    var_4 = 'age'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'gender': 'female', 'age': 30})
    assert var_10 is True



# Parsed testcases at query #11
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_factory_field_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 30
    var_2 = 'male'
    var_3 = 'age'
    var_4 = 'gender'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'age': 30, 'gender': 'male'})
    assert var_10 is True

def test_case_0():
    var_0 = 'datetime'
    var_1 = '%Y-%m-%d'



# Parsed testcases at query #13
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_init_assigns_locale_kwargs_and_field. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_all_params. Retrieved 3/5 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 42})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 42})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_init_assigns_locale_kwargs_and_field. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_factory_field_constructor. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'



# Parsed testcases at query #20
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_factory_field_initialization. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'
    var_2 = 'extra_param'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'test_field'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'extra_param': 'value'})
    assert var_8 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_factory_field_constructor_with_custom_locale. Retrieved 1/3 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'female'
    var_2 = 'gender'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'person'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'gender': 'female'})
    assert var_8 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_factory_field_constructor. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'value'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_init_assigns_field_parameter. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'
    var_2 = 'some_param'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'test_field'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'some_param': 'value'})
    assert var_8 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_factory_field_initialization. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #28
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_locale_parameter_is_not_none. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_field'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 42})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42



# Parsed testcases at query #31
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_init_assigns_locale. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_field'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 30
    var_2 = 'male'
    var_3 = 'age'
    var_4 = 'gender'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'age': 30, 'gender': 'male'})
    assert var_10 is True

def test_case_0():
    var_0 = 'datetime'
    var_1 = '%Y-%m-%d'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_init_assigns_locale_kwargs_and_field. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_init_with_locale. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_field'



# Parsed testcases at query #36
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_factory_field_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'female'
    var_2 = 'gender'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'person'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'gender': 'female'})
    assert var_8 is True

def test_case_0():
    var_0 = 'datetime'
    var_1 = '%Y-%m-%d'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_factory_field_constructor. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 123
    var_2 = 'value'
    var_3 = 'some_param'
    var_4 = 'another_param'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'some_param': 123, 'another_param': 'value'})
    assert var_10 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_init_with_locale_and_kwargs. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_factory_field_constructor_with_custom_locale. Retrieved 1/3 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'female'
    var_2 = 30
    var_3 = 'gender'
    var_4 = 'age'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'gender': 'female', 'age': 30})
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'person.name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'address.city'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'address.city'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'datetime.date'
    var_1 = 'female'
    var_2 = 18
    var_3 = 'gender'
    var_4 = 'minimum_age'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'datetime.date'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'gender': 'female', 'minimum_age': 18})
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_factory_field_constructor_with_custom_locale. Retrieved 1/3 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'person.full_name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address.city'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'datetime.date'
    var_1 = 1990
    var_2 = 2000
    var_3 = 'start'
    var_4 = 'end'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'datetime.date'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'start': 1990, 'end': 2000})
    assert var_10 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'
    var_2 = 'custom_param'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'test_field'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'custom_param': 'value'})
    assert var_8 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_factory_field_constructor. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'value'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'address'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'address'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_factory_field_constructor. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'value'



# Parsed testcases at query #7
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field is not None)
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_init_assigns_locale_field_and_kwargs. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_init_with_locale_sets_locale. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_field'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_init_assigns_locale. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_init_assigns_locale. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_field'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 42})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42



# Parsed testcases at query #13
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = None
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'key'
    var_6 = {var_5: var_3}
    var_7 = module_0.FactoryField(var_0, var_1, **var_6)
    var_8 = var_7.field
    var_9 = bool(var_7.field == var_0)
    assert var_9 is True
    var_10 = var_7.locale
    var_11 = bool(var_7.locale == var_1)
    assert var_11 is True
    var_12 = var_7.kwargs
    var_13 = bool(var_7.kwargs == var_4)
    assert var_13 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_init_assigns_field_correctly. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 30
    var_2 = 'male'
    var_3 = 'age'
    var_4 = 'gender'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'age': 30, 'gender': 'male'})
    assert var_10 is True

def test_case_0():
    var_0 = 'datetime'
    var_1 = '%Y-%m-%d'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_factory_field_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'female'
    var_2 = 30
    var_3 = 'gender'
    var_4 = 'age'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'gender': 'female', 'age': 30})
    assert var_10 is True

def test_case_0():
    var_0 = 'food'
    var_1 = 'dessert'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 42})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42



# Parsed testcases at query #18
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_init_with_locale_and_kwargs. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_locale_not_none. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_field'



# Parsed testcases at query #21
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_factory_field_constructor. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_factory_field_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'person.name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'person.name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.name'
    var_1 = 'female'
    var_2 = 'gender'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'person.name'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'gender': 'female'})
    assert var_8 is True

def test_case_0():
    var_0 = 'person.name'
    var_1 = 'male'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'female'
    var_2 = 30
    var_3 = 'gender'
    var_4 = 'age'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'gender': 'female', 'age': 30})
    assert var_10 is True

def test_case_0():
    var_0 = 'datetime'
    var_1 = '%Y-%m-%d'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 42})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'



# Parsed testcases at query #26
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 42})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'



# Parsed testcases at query #28
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_factory_field_constructor. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'value'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'
    var_2 = 'custom_param'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'test_field'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'custom_param': 'value'})
    assert var_8 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val1'
    var_2 = 'val2'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'person.full_name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address.city'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'datetime.date'
    var_1 = 2000
    var_2 = 2020
    var_3 = 'start'
    var_4 = 'end'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'datetime.date'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'start': 2000, 'end': 2020})
    assert var_10 is True

def test_case_0():
    var_0 = 'person.age'
    var_1 = 18
    var_2 = 99



# Parsed testcases at query #32
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field is not None)
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 42
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 42})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'



# Parsed testcases at query #34
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_init_sets_locale. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_field'



