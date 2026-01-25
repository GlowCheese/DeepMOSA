####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 4/6 statements.
# Partially parsed test_factory_field_constructor_with_multiple_kwargs. Retrieved 8/10 statements.
# Partially parsed test_factory_field_constructor_empty_kwargs. Retrieved 3/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'first_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'max_bytes'
    var_2 = 'key'
    var_3 = 100
    var_4 = 'value'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'max_bytes'
    var_7 = 'key'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'phone_number'
    var_1 = 'mask'
    var_2 = '###-###-####'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'address'
    var_1 = 'param1'
    var_2 = 'param2'
    var_3 = 'param3'
    var_4 = 'value1'
    var_5 = 42
    var_6 = True
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'uuid'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.kwargs
    var_4 = bool(var_2.kwargs == {})
    assert var_4 is True
    var_5 = var_2.kwargs



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_multiple_kwargs. Retrieved 4/8 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'first_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'first_name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'first_name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'first_name'
    var_1 = 'female'
    var_2 = 25
    var_3 = 'gender'
    var_4 = 'age'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'first_name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'gender': 'female', 'age': 25})
    assert var_10 is True

def test_case_0():
    var_0 = 'person'
    var_1 = 'male'
    var_2 = 30

def test_case_0():
    var_0 = 'email'

def test_case_0():
    var_0 = 'address'
    var_1 = 'Spain'
    var_2 = 'Madrid'
    var_3 = '28001'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'phone'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'first_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'first_name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'password'
    var_1 = 10
    var_2 = True

def test_case_0():
    var_0 = 'username'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'
    var_3 = 'val3'
    var_4 = 'param1'
    var_5 = 'param2'
    var_6 = 'param3'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.FactoryField(var_0, **var_7)
    var_9 = var_8.field
    assert var_9 == 'address'
    var_10 = var_8.kwargs
    var_11 = bool(var_8.kwargs == {'param1': 'val1', 'param2': 'val2', 'param3': 'val3'})
    assert var_11 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 2/6 statements.
# Partially parsed test_factory_field_constructor_with_multiple_kwargs. Retrieved 3/7 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'first_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'first_name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'first_name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 100
    var_2 = 10
    var_3 = 'max_bytes'
    var_4 = 'min_bytes'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'max_bytes': 100, 'min_bytes': 10})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'Germany'

def test_case_0():
    var_0 = 'password'
    var_1 = 20
    var_2 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person.full_name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'd'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'test_field'
    var_10 = 'a'
    var_11 = 'b'
    var_12 = 'c'
    var_13 = 'd'
    var_14 = {var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_15 = module_0.FactoryField(var_9, **var_14)
    var_16 = var_15.field
    assert var_16 == 'test_field'
    var_17 = var_15.kwargs
    var_18 = bool(var_15.kwargs == var_8)
    assert var_18 is True



# Parsed testcases at query #7
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/9 statements.


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
    var_0 = 'name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'email'
    var_1 = 'test'
    var_2 = 42

def test_case_0():
    var_0 = 'address'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 6/10 statements.
# Partially parsed test_factory_field_constructor_with_multiple_kwargs. Retrieved 4/8 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'address'
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = module_0.FactoryField(var_5, **var_8)
    var_10 = var_9.field
    assert var_10 == 'address'
    var_11 = var_9.locale
    assert var_11 is None
    var_12 = var_9.kwargs
    var_13 = bool(var_9.kwargs == var_4)
    assert var_13 is True

def test_case_0():
    var_0 = 'param1'
    var_1 = 'param2'
    var_2 = 100
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'phone'

def test_case_0():
    var_0 = 'text'
    var_1 = 'val1'
    var_2 = 'val2'
    var_3 = 42

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == ''
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_factory_field_init_stores_field_parameter. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #11
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #12
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 5/6 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'full_name'
    var_1 = 'female'
    var_2 = True

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
    var_6 = var_2.kwargs
    var_7 = len(var_6)
    assert var_7 == 0

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = None
    var_2 = 'val1'
    var_3 = 42
    var_4 = True
    var_5 = 'param1'
    var_6 = 'param2'
    var_7 = 'param3'
    var_8 = {var_5: var_2, var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, var_1, **var_8)
    var_10 = var_9.field
    assert var_10 == 'person'
    var_11 = var_9.kwargs
    var_12 = bool(var_9.kwargs == {'param1': 'val1', 'param2': 42, 'param3': True})
    assert var_12 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.


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
    var_0 = 'name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 42
    var_2 = 'value'
    var_3 = 'seed'
    var_4 = 'key'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'seed': 42, 'key': 'value'})
    assert var_10 is True

def test_case_0():
    var_0 = 'email'
    var_1 = 123
    var_2 = 'test'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'address'
    var_1 = 'value1'
    var_2 = 42
    var_3 = True
    var_4 = 'param1'
    var_5 = 'param2'
    var_6 = 'param3'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.FactoryField(var_0, **var_7)
    var_9 = var_8.field
    assert var_9 == 'address'
    var_10 = var_8.locale
    assert var_10 is None
    var_11 = var_8.kwargs
    var_12 = bool(var_8.kwargs == {'param1': 'value1', 'param2': 42, 'param3': True})
    assert var_12 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factoryfield_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factoryfield_constructor_multiple_instances_independent. Retrieved 4/10 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person.full_name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address.city'
    var_1 = 'test'
    var_2 = 123

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == ''
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'name'
    var_1 = 'value1'
    var_2 = 'email'
    var_3 = 'value2'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_factory_field_init_with_none_locale. Retrieved 3/13 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'test_field'
    var_2 = None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/5 statements.


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
    var_0 = 'name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'email'
    var_1 = 'test'
    var_2 = 42

def test_case_0():
    var_0 = 'username'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'text'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'address'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_factory_field_locale_none_predicate. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_factory_field_locale_is_none. Retrieved 3/13 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'test_field'
    var_2 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_factory_field_locale_assignment. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'name'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'test1'
    var_2 = 'test2'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 100
    var_2 = 10
    var_3 = 'en'
    var_4 = 'max_bytes'
    var_5 = 'min_bytes'
    var_6 = 'lang'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.FactoryField(var_0, **var_7)
    var_9 = var_8.field
    assert var_9 == 'text'
    var_10 = var_8.kwargs
    var_11 = bool(var_8.kwargs == {'max_bytes': 100, 'min_bytes': 10, 'lang': 'en'})
    assert var_11 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_factory_field_init_with_locale_none. Retrieved 5/14 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'first_name'
    var_2 = None
    var_3 = 'value'
    var_4 = 'some_param'
    var_5 = {var_4: var_3}
    var_6 = module_0.FactoryField(var_1, var_2, **var_5)
    var_7 = var_6.locale
    assert var_7 is None
    var_8 = var_6.field
    assert var_8 == 'first_name'
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'some_param': 'value'})
    assert var_10 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factoryfield_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factoryfield_constructor_with_empty_kwargs. Retrieved 1/5 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'email'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'France'
    var_2 = 'Paris'

def test_case_0():
    var_0 = 'username'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_factory_field_locale_assignment. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'name'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 3/6 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'first_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'first_name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'full_name'
    var_1 = 'test'
    var_2 = 123

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'
    var_3 = 'val3'
    var_4 = 'kwarg1'
    var_5 = 'kwarg2'
    var_6 = 'kwarg3'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.FactoryField(var_0, **var_7)
    var_9 = var_8.field
    assert var_9 == 'address'
    var_10 = var_8.kwargs
    var_11 = bool(var_8.kwargs == {'kwarg1': 'val1', 'kwarg2': 'val2', 'kwarg3': 'val3'})
    assert var_11 is True



# Parsed testcases at query #27
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/5 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person.full_name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address.city'
    var_1 = 'test'
    var_2 = 42

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.field
    assert var_4 == 'name'
    var_5 = var_3.locale
    assert var_5 is None
    var_6 = var_3.kwargs
    var_7 = bool(var_3.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'phone'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'phone'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_factory_field_init_with_locale_and_kwargs. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #31
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None
    var_5 = var_3.field
    assert var_5 == 'name'
    var_6 = var_3.kwargs
    var_7 = bool(var_3.kwargs == {})
    assert var_7 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_factory_field_init_locale_assignment. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'email'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'address'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'address'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'person'
    var_1 = 'test1'
    var_2 = 'test2'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'phone_number'
    var_1 = '###-####'
    var_2 = 'US'
    var_3 = None
    var_4 = 'mask'
    var_5 = 'country'
    var_6 = {var_4: var_1, var_5: var_2}
    var_7 = module_0.FactoryField(var_0, var_3, **var_6)
    var_8 = var_7.field
    assert var_8 == 'phone_number'
    var_9 = var_7.locale
    assert var_9 is None
    var_10 = var_7.kwargs
    var_11 = bool(var_7.kwargs == {'mask': '###-####', 'country': 'US'})
    assert var_11 is True



# Parsed testcases at query #34
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None
    var_5 = var_3.field
    assert var_5 == 'name'
    var_6 = var_3.kwargs
    var_7 = bool(var_3.kwargs == {})
    assert var_7 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 6/10 statements.
# Partially parsed test_factory_field_constructor_preserves_kwargs_order. Retrieved 9/13 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'text'
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = module_0.FactoryField(var_5, **var_8)
    var_10 = var_9.field
    assert var_10 == 'text'
    var_11 = var_9.locale
    assert var_11 is None
    var_12 = var_9.kwargs
    var_13 = bool(var_9.kwargs == var_4)
    assert var_13 is True

def test_case_0():
    var_0 = 'param1'
    var_1 = 'param2'
    var_2 = 'test'
    var_3 = 100
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'person'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'z'
    var_1 = 'a'
    var_2 = 'm'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'data'
    var_8 = 'z'
    var_9 = 'a'
    var_10 = 'm'
    var_11 = {var_8: var_3, var_9: var_4, var_10: var_5}
    var_12 = module_0.FactoryField(var_7, **var_11)
    var_13 = var_12.kwargs
    var_14 = bool(var_12.kwargs == var_6)
    assert var_14 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'phone_number'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = 'd'
    var_9 = {var_5: var_1, var_6: var_2, var_7: var_3, var_8: var_4}
    var_10 = module_0.FactoryField(var_0, **var_9)
    var_11 = var_10.field
    assert var_11 == 'text'
    var_12 = var_10.kwargs
    var_13 = bool(var_10.kwargs == {'a': 1, 'b': 2, 'c': 3, 'd': 4})
    assert var_13 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'text'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factoryfield_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factoryfield_constructor_with_empty_kwargs. Retrieved 1/5 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'email'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'first_name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'test'
    var_2 = 42

def test_case_0():
    var_0 = 'phone_number'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_factory_field_init_with_locale_and_kwargs. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'email'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'person'



# Parsed testcases at query #41
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/5 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person.full_name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address.full_address'
    var_1 = 42
    var_2 = 'female'

def test_case_0():
    var_0 = 'text.title'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'text'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_factory_field_locale_is_none. Retrieved 3/13 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'test_field'
    var_2 = None



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value'
    var_2 = 'param'
    var_3 = 'custom_param'
    var_4 = 'another'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'custom_param': 'value', 'another': 'param'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'France'
    var_2 = 'capital'

def test_case_0():
    var_0 = 'username'



# Parsed testcases at query #46
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 6/10 statements.
# Partially parsed test_factory_field_constructor_with_field_and_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_empty_kwargs. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'email'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'phone_number'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'max_bytes'
    var_2 = 'key'
    var_3 = 100
    var_4 = 'value'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'max_bytes'
    var_7 = 'key'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'address'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.field
    var_5 = bool(var_3.field == var_0)
    assert var_5 is True
    var_6 = var_3.locale
    assert var_6 is None
    var_7 = var_3.kwargs
    var_8 = bool(var_3.kwargs == {})
    assert var_8 is True

def test_case_0():
    var_0 = 'uuid'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_factory_field_constructor. Retrieved 4/6 statements.
# Partially parsed test_factory_field_constructor_with_multiple_kwargs. Retrieved 8/10 statements.


def test_case_0():
    var_0 = 'first_name'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.field
    var_5 = bool(var_3.field == var_0)
    assert var_5 is True
    var_6 = var_3.locale
    assert var_6 is None
    var_7 = var_3.kwargs
    var_8 = bool(var_3.kwargs == {})
    assert var_8 is True

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'phone_number'
    var_1 = 'min_digits'
    var_2 = 'max_digits'
    var_3 = 10
    var_4 = 15
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'min_digits'
    var_7 = 'max_digits'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'text'
    var_1 = 'param1'
    var_2 = 'param2'
    var_3 = 'param3'
    var_4 = 'value1'
    var_5 = 42
    var_6 = True
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_factory_field_init_locale_assignment. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'name'



# Parsed testcases at query #50
#--------------------------






# Parsed testcases at query #51
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factory_field_constructor_with_multiple_kwargs. Retrieved 10/14 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'value'
    var_2 = 123
    var_3 = 'some_param'
    var_4 = 'another'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person.full_name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'some_param': 'value', 'another': 123})
    assert var_10 is True

def test_case_0():
    var_0 = 'address.city'
    var_1 = 'test'
    var_2 = 42

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'key4'
    var_4 = 'value1'
    var_5 = 2
    var_6 = True
    var_7 = None
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'test_field'



# Parsed testcases at query #52
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 6/8 statements.
# Partially parsed test_factory_field_constructor_with_field_and_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_multiple_kwargs. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 'email'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'phone_number'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'max_bytes'
    var_2 = 'custom_param'
    var_3 = 100
    var_4 = True
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'max_bytes'
    var_7 = 'custom_param'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'username'
    var_1 = 'param1'
    var_2 = 'param2'
    var_3 = 'param3'
    var_4 = 'value1'
    var_5 = 42
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_1: var_4, var_2: var_5, var_3: var_9}



# Parsed testcases at query #54
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #55
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factory_field_constructor_with_multiple_kwargs. Retrieved 8/12 statements.


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
    var_0 = 'name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 42
    var_2 = 'value'
    var_3 = 'seed'
    var_4 = 'key'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'seed': 42, 'key': 'value'})
    assert var_10 is True

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 123
    var_2 = 'test'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.field
    assert var_4 == 'email'
    var_5 = var_3.locale
    assert var_5 is None
    var_6 = var_3.kwargs
    var_7 = bool(var_3.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'param1'
    var_1 = 'param2'
    var_2 = 'param3'
    var_3 = 'value1'
    var_4 = 42
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'text'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'phone_number'
    var_1 = '+33 # ## ## ## ##'
    var_2 = 'custom_value'

def test_case_0():
    var_0 = 'username'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_factory_field_constructor_with_field_and_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'email'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'France'
    var_2 = 'Paris'

def test_case_0():
    var_0 = 'phone_number'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/5 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'email'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'person'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'address'



# Parsed testcases at query #5
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'first_name'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'phone'



# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------

# Partially parsed test_factory_field_init_with_locale_and_kwargs. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'email'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'full_name'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'username'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 6/10 statements.
# Partially parsed test_factory_field_constructor_with_field_and_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'full_name'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'max_length'
    var_2 = 'min_length'
    var_3 = 100
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'max_length'
    var_7 = 'min_length'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'phone_number'



# Parsed testcases at query #12
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = None
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'key'
    var_6 = {var_5: var_3}
    var_7 = module_0.FactoryField(var_0, var_1, **var_6)
    var_8 = var_7.locale
    assert var_8 is None
    var_9 = var_7.field
    var_10 = bool(var_7.field == var_0)
    assert var_10 is True
    var_11 = var_7.kwargs
    var_12 = bool(var_7.kwargs == var_4)
    assert var_12 is True



# Parsed testcases at query #13
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'first_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'text'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'France'
    var_2 = 'Île-de-France'

def test_case_0():
    var_0 = 'text'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'full_name'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'phone'



# Parsed testcases at query #17
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_factory_field_init_with_none_locale. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'France'
    var_2 = 'city'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'val1'
    var_2 = 'val2'
    var_3 = 'val3'
    var_4 = 'key1'
    var_5 = 'key2'
    var_6 = 'key3'
    var_7 = {var_4: var_1, var_5: var_2, var_6: var_3}
    var_8 = module_0.FactoryField(var_0, **var_7)
    var_9 = var_8.field
    assert var_9 == 'person'
    var_10 = var_8.kwargs
    var_11 = bool(var_8.kwargs == {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'})
    assert var_11 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_factory_field_init_with_explicit_locale. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'name'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factoryfield_constructor_with_kwargs. Retrieved 3/7 statements.
# Partially parsed test_factoryfield_constructor_with_empty_kwargs. Retrieved 1/5 statements.


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
    var_0 = 'person.full_name'

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'female'
    var_2 = True

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'address.postal_code'
    var_1 = 99999
    var_2 = 10000
    var_3 = 'max_value'
    var_4 = 'min_value'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'address.postal_code'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'max_value': 99999, 'min_value': 10000})
    assert var_10 is True

def test_case_0():
    var_0 = 'text.sentence'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'name'
    var_1 = 'value1'
    var_2 = 'value2'

def test_case_0():
    var_0 = 'address'



# Parsed testcases at query #23
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'first_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_factory_field_init_with_none_locale. Retrieved 4/13 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test_field'
    var_2 = None
    var_3 = {}
    var_4 = module_0.FactoryField(var_1, var_2, **var_3)
    var_5 = var_4.locale
    assert var_5 is None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/5 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'phone_number'



# Parsed testcases at query #26
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None
    var_5 = var_3.field
    var_6 = bool(var_3.field == var_0)
    assert var_6 is True
    var_7 = var_3.kwargs
    var_8 = bool(var_3.kwargs == {})
    assert var_8 is True



# Parsed testcases at query #27
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None
    var_5 = var_3.field
    assert var_5 == 'name'
    var_6 = var_3.kwargs
    var_7 = bool(var_3.kwargs == {})
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'text'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_factory_field_init_stores_field_parameter. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 6/8 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'max_bytes'
    var_2 = 'seed'
    var_3 = 100
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'max_bytes'
    var_7 = 'seed'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'full_name'
    var_1 = 'gender'
    var_2 = 'seed'
    var_3 = 'female'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'address'
    var_1 = 'country'
    var_2 = 'postal_code'
    var_3 = 'seed'
    var_4 = 'USA'
    var_5 = True
    var_6 = 999
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'country'
    var_9 = 'postal_code'
    var_10 = 'seed'
    var_11 = {var_8: var_4, var_9: var_5, var_10: var_6}
    var_12 = module_0.FactoryField(var_0, **var_11)
    var_13 = var_12.field
    var_14 = bool(var_12.field == var_0)
    assert var_14 is True
    var_15 = var_12.kwargs
    var_16 = bool(var_12.kwargs == var_7)
    assert var_16 is True
    var_17 = var_12.kwargs
    var_18 = len(var_17)
    assert var_18 == 3



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_factory_field_init_with_none_locale. Retrieved 4/13 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'name'
    var_2 = None
    var_3 = {}
    var_4 = module_0.FactoryField(var_1, var_2, **var_3)
    var_5 = var_4.locale
    assert var_5 is None
    var_6 = var_4.locale is not None
    assert var_6 is False



# Parsed testcases at query #33
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = 'value1'
    var_3 = 'key1'
    var_4 = {var_3: var_2}
    var_5 = module_0.FactoryField(var_0, var_1, **var_4)
    var_6 = var_5.locale
    assert var_6 is None
    var_7 = var_5.kwargs
    var_8 = bool(var_5.kwargs == {'key1': 'value1'})
    assert var_8 is True
    var_9 = var_5.field
    assert var_9 == 'name'



# Parsed testcases at query #34
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None
    var_5 = var_3.field
    assert var_5 == 'name'
    var_6 = var_3.kwargs
    var_7 = bool(var_3.kwargs == {})
    assert var_7 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_factory_field_init_locale_assignment. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'name'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factoryfield_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factoryfield_constructor_with_empty_kwargs. Retrieved 1/5 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'email'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'first_name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'password'
    var_1 = 12
    var_2 = True

def test_case_0():
    var_0 = 'username'



# Parsed testcases at query #37
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'Test that FactoryField can be initialized with locale=None.'
    var_1 = 'name'
    var_2 = None
    var_3 = {}
    var_4 = module_0.FactoryField(var_1, var_2, **var_3)
    var_5 = var_4.locale
    assert var_5 is None
    var_6 = var_4.field
    assert var_6 == 'name'
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {})
    assert var_8 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 6/8 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/3 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'first_name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'text'
    var_1 = 'length'
    var_2 = 'seed'
    var_3 = 100
    var_4 = 12345
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 'address'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = 'param1'
    var_2 = 'param2'
    var_3 = 'param3'
    var_4 = 'val1'
    var_5 = 'val2'
    var_6 = 123
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'param1'
    var_9 = 'param2'
    var_10 = 'param3'
    var_11 = {var_8: var_4, var_9: var_5, var_10: var_6}
    var_12 = module_0.FactoryField(var_0, **var_11)
    var_13 = var_12.field
    var_14 = bool(var_12.field == var_0)
    assert var_14 is True
    var_15 = var_12.kwargs
    var_16 = bool(var_12.kwargs == var_7)
    assert var_16 is True



# Parsed testcases at query #39
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/6 statements.
# Partially parsed test_factory_field_constructor_with_empty_kwargs. Retrieved 1/4 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'test1'
    var_2 = 'test2'

def test_case_0():
    var_0 = 'date'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_factory_field_init_locale_assignment. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'email'



# Parsed testcases at query #42
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = 'value'
    var_3 = 'extra_param'
    var_4 = {var_3: var_2}
    var_5 = module_0.FactoryField(var_0, var_1, **var_4)
    var_6 = var_5.locale
    assert var_6 is None
    var_7 = var_5.field
    assert var_7 == 'name'
    var_8 = var_5.kwargs
    var_9 = bool(var_5.kwargs == {'extra_param': 'value'})
    assert var_9 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_locale_and_kwargs. Retrieved 2/5 statements.
# Partially parsed test_factory_field_constructor_with_multiple_kwargs. Retrieved 4/7 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'first_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'first_name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 100
    var_2 = 'length'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'text'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'length': 100})
    assert var_8 is True

def test_case_0():
    var_0 = 'password'
    var_1 = 50

def test_case_0():
    var_0 = 'address'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 123

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = 'value'
    var_3 = 'key'
    var_4 = {var_3: var_2}
    var_5 = module_0.FactoryField(var_0, var_1, **var_4)
    var_6 = var_5.field
    assert var_6 == 'name'
    var_7 = var_5.locale
    assert var_7 is None
    var_8 = var_5.kwargs
    var_9 = bool(var_5.kwargs == {'key': 'value'})
    assert var_9 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_factory_field_init_with_locale_and_kwargs. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'email'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 2/6 statements.
# Partially parsed test_factory_field_constructor_with_field_and_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_multiple_kwargs. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 'email'
    var_1 = 'custom_value'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'phone_number'

def test_case_0():
    var_0 = 'text'
    var_1 = 'param1'
    var_2 = 'param2'
    var_3 = 'param3'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = 42
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}



# Parsed testcases at query #46
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #47
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factoryfield_constructor_with_all_parameters. Retrieved 3/7 statements.
# Partially parsed test_factoryfield_constructor_with_empty_kwargs. Retrieved 1/5 statements.


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
    var_0 = 'name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'name'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'email'
    var_1 = 'val1'
    var_2 = 'val2'

def test_case_0():
    var_0 = 'address'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_factory_field_init_stores_field_parameter. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_factory_field_init_locale_assignment. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'name'



# Parsed testcases at query #51
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None
    var_5 = var_3.field
    assert var_5 == 'name'
    var_6 = var_3.kwargs
    var_7 = bool(var_3.kwargs == {})
    assert var_7 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_factory_field_init_with_locale_and_kwargs. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'param1'
    var_1 = 'param2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'email'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/5 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 3/7 statements.


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
    var_0 = 'email'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'person'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'address'
    var_1 = 'val1'
    var_2 = 'val2'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'test_field'
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = {var_8: var_3, var_9: var_4, var_10: var_5}
    var_12 = module_0.FactoryField(var_7, **var_11)
    var_13 = var_12.field
    assert var_13 == 'test_field'
    var_14 = var_12.kwargs
    var_15 = bool(var_12.kwargs == var_6)
    assert var_15 is True

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'phone'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.field
    assert var_4 == 'phone'
    var_5 = var_3.locale
    assert var_5 is None



# Parsed testcases at query #54
#--------------------------




import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_factory_field_init_locale_is_none. Retrieved 3/12 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test_field'
    var_2 = {}
    var_3 = module_0.FactoryField(var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_factory_field_constructor_with_locale. Retrieved 1/4 statements.
# Partially parsed test_factory_field_constructor_with_all_parameters. Retrieved 6/9 statements.


import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'name'

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = 'max_bytes'
    var_2 = 'key'
    var_3 = 100
    var_4 = 'value'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'max_bytes'
    var_7 = 'key'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = 'param1'
    var_2 = 'param2'
    var_3 = 'val1'
    var_4 = 42
    var_5 = {var_1: var_3, var_2: var_4}

import mimesis.plugins.factory as module_0

def test_case_0():
    var_0 = 'address'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'key3'
    var_4 = 'key4'
    var_5 = 'value1'
    var_6 = 123
    var_7 = True
    var_8 = None
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'key1'
    var_11 = 'key2'
    var_12 = 'key3'
    var_13 = 'key4'
    var_14 = {var_10: var_5, var_11: var_6, var_12: var_7, var_13: var_8}
    var_15 = module_0.FactoryField(var_0, **var_14)
    var_16 = var_15.field
    var_17 = bool(var_15.field == var_0)
    assert var_17 is True
    var_18 = var_15.kwargs
    var_19 = bool(var_15.kwargs == var_9)
    assert var_19 is True
    var_20 = var_15.kwargs
    var_21 = len(var_20)
    assert var_21 == 4



