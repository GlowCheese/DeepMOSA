####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = var_2.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'default_value'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'a': 'default_value'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'valid_value'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'a': 'valid_value'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'a'
    var_7 = 'not_an_integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'test_key'
    var_7 = 'other_key'
    var_8 = 'valid_value'
    var_9 = 123
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = bool(var_11 == {'test_key': 'valid_value'})
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'test_key'
    var_7 = 'not_an_integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, additional_properties=var_0, **var_5)
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'valid'
    var_10 = 'invalid'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 'a'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {var_2: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, additional_properties=var_1, **var_6)
    var_8 = 'b'
    var_9 = 'valid'
    var_10 = 123
    var_11 = {var_2: var_9, var_8: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(var_12 == {'a': 'valid', 'b': 123})
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 'a'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {var_2: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, additional_properties=var_1, **var_6)
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'valid'
    var_11 = 'not_an_integer'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = var_7.validate(var_12)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = '^test_'
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {var_4: var_6}
    var_8 = {}
    var_9 = module_0.Boolean(**var_8)
    var_10 = [var_0]
    var_11 = 1
    var_12 = 5
    var_13 = {}
    var_14 = module_0.String(min_length=var_11, **var_13)
    var_15 = {}
    var_16 = module_0.Object(properties=var_3, pattern_properties=var_7, additional_properties=var_9, property_names=var_14, min_properties=var_11, max_properties=var_12, required=var_10, **var_15)
    var_17 = 'test_key'
    var_18 = 'other'
    var_19 = 'valid'
    var_20 = 123
    var_21 = True
    var_22 = {var_0: var_19, var_17: var_20, var_18: var_21}
    var_23 = var_16.validate(var_22)
    var_24 = bool(var_23 == {'a': 'valid', 'test_key': 123, 'other': True})
    assert var_24 is True



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    assert var_3 is True
    var_4 = False
    var_5 = var_1.validate(var_4)
    assert var_5 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'true'
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'false'
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'on'
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'off'
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = '1'
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = '0'
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 1
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 0
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = 'null'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = 'none'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'invalid'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'not a field'
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_float_as_integer. Retrieved 1/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.14159
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 3.14)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 3
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.2
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    assert var_3 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 3.14
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 3.14)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '42'
    var_3 = var_1.validate(var_2)
    assert var_3 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = -10
    var_3 = var_1.validate(var_2)
    assert var_3 == -10

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 0
    var_3 = var_1.validate(var_2)
    assert var_3 == 0



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, coerce_types=var_3, **var_5)
    var_7 = ''
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_3}
    var_7 = module_0.Choice(choices=var_2, coerce_types=var_4, **var_6)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = ''
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)
    var_7 = var_6.validate(var_0)
    assert var_7 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)
    var_7 = 'c'
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'true'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'false'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'list'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = [var_0, var_1]
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == ['a', 'b'])
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {var_0: var_1}
    var_3 = 'dict'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = {var_0: var_1}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'a': 'b'})
    assert var_10 is True



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = {}
    var_3 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'blank'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'a\x00b'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'ab'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  hello  '
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  hello  '
    var_4 = var_2.validate(var_3)
    assert var_4 == '  hello  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hi'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_length'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'max_length'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'Hello123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'pattern'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'invalid-email'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'format'



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.serialize(var_7)
    var_9 = bool(var_8 == ['a', 'b', 'c'])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 'a'
    var_8 = 1
    var_9 = [var_7, var_8]
    var_10 = var_6.serialize(var_9)
    var_11 = bool(var_10 == ['a', 1])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.serialize(var_5)
    var_7 = bool(var_6 == ['a', 'b', 'c'])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = {}
    var_6 = module_0.Array(var_2, var_4, **var_5)
    var_7 = 'a'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = var_6.serialize(var_11)
    var_13 = bool(var_12 == ['a', 1, 2, 3])
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'fields'
    var_8 = {var_7: var_6}
    var_9 = module_0.Object(**var_8)
    var_10 = {}
    var_11 = module_0.Array(var_9, **var_10)
    var_12 = 'Alice'
    var_13 = 30
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = 'Bob'
    var_16 = 25
    var_17 = {var_0: var_15, var_1: var_16}
    var_18 = [var_14, var_17]
    var_19 = var_11.serialize(var_18)
    var_20 = bool(var_19 == [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}])
    assert var_20 is True



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_0]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = 3
    var_5 = [var_3, var_0, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_0]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(var_8 == [1, 2, 3])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 'two'
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 1
    var_8 = 'two'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == [1, 'two'])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Array(var_2, var_4, **var_5)
    var_7 = 1
    var_8 = 'two'
    var_9 = 'three'
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(var_11 == [1, 'two', 'three'])
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Array(var_2, var_4, **var_5)
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = 1
    var_7 = [var_6]
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == [1])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = '1'
    var_4 = [var_0, var_3, var_0]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [1, '1', True])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = True
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = 'c'
    var_10 = var_8.validate(var_9)



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Union(var_2, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = 0
    var_8 = excinfo.value.messages()[var_7]
    var_9 = var_8.code
    assert var_9 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'hello'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 3.14
    var_8 = var_6.validate(var_7)
    var_9 = 0
    var_10 = excinfo.value.messages()[var_9]
    var_11 = var_10.code
    assert var_11 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = 0
    var_12 = excinfo.value.messages()[var_11]
    var_13 = var_12.code
    assert var_13 == 'union'



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_raises_type_error_for_invalid_value. Retrieved 5/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Boolean(coerce_types=var_0, **var_3)
    var_5 = 'invalid'
    var_6 = var_4.validate(var_5)
    var_7 = 'type'



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = '123'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = var_2.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'default'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'a': 'default'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'a'
    var_7 = 123
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^a.*$'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'abc'
    var_7 = 'def'
    var_8 = 'value'
    var_9 = 123
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = bool(var_11 == {'abc': 'value'})
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^a.*$'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'abc'
    var_7 = 123
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'value'
    var_10 = 'extra'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = 'b'
    var_9 = 'value'
    var_10 = 123
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(var_12 == {'a': 'value', 'b': 123})
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'value'
    var_11 = 'not an integer'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = var_7.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = 'value'
    var_10 = 123
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = bool(var_12 == {'a': 'value', 'b': 123})
    assert var_13 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_string_constructor_with_defaults. Retrieved 1/2 statements.
# Partially parsed test_string_constructor_with_custom_values. Retrieved 10/12 statements.
# Partially parsed test_string_constructor_with_allow_blank_and_no_default. Retrieved 2/4 statements.
# Partially parsed test_string_constructor_with_allow_null_and_no_default. Retrieved 2/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = bool(not var_1.allow_null)
    assert var_4 is True
    var_5 = bool(not var_1.read_only)
    assert var_5 is True
    var_6 = bool(not var_1.allow_blank)
    assert var_6 is True
    var_7 = bool(var_1.trim_whitespace)
    assert var_7 is True
    var_8 = var_1.max_length
    assert var_8 is None
    var_9 = var_1.min_length
    assert var_9 is None
    var_10 = var_1.pattern
    assert var_10 is None
    var_11 = var_1.pattern_regex
    assert var_11 is None
    var_12 = var_1.format
    assert var_12 is None
    var_13 = bool(var_1.coerce_types)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = 'default_value'
    var_3 = True
    var_4 = False
    var_5 = 100
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = 'title'
    var_10 = 'description'
    var_11 = 'default'
    var_12 = 'allow_null'
    var_13 = 'read_only'
    var_14 = {var_9: var_0, var_10: var_1, var_11: var_2, var_12: var_3, var_13: var_3}
    var_15 = module_0.String(allow_blank=var_3, trim_whitespace=var_4, max_length=var_5, min_length=var_6, pattern=var_7, format=var_8, coerce_types=var_4, **var_14)
    var_16 = var_15.title
    assert var_16 == 'Test Title'
    var_17 = var_15.description
    assert var_17 == 'Test Description'
    var_18 = bool(var_15.allow_null)
    assert var_18 is True
    var_19 = bool(var_15.read_only)
    assert var_19 is True
    var_20 = bool(var_15.allow_blank)
    assert var_20 is True
    var_21 = bool(not var_15.trim_whitespace)
    assert var_21 is True
    var_22 = var_15.max_length
    assert var_22 == 100
    var_23 = var_15.min_length
    assert var_23 == 10
    var_24 = var_15.pattern
    assert var_24 == '^[a-z]+$'
    var_25 = var_15.pattern_regex
    var_26 = bool(var_15.pattern_regex is not None)
    assert var_26 is True
    var_27 = var_15.format
    assert var_27 == 'email'
    var_28 = bool(not var_15.coerce_types)
    assert var_28 is True

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = var_3.pattern
    assert var_4 == '^[a-z]+$'
    var_5 = var_3.pattern_regex
    var_6 = bool(var_3.pattern_regex == var_1)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = {}
    var_3 = module_0.Array(var_0, var_1, **var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(var_8 == [1, 2, 3])
    assert var_9 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_choice_constructor_with_default_value. Retrieved 2/4 statements.
# Partially parsed test_choice_constructor_with_callable_default. Retrieved 3/5 statements.
# Partially parsed test_choice_constructor_with_allow_null_and_no_default. Retrieved 2/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.choices
    var_10 = bool(var_8.choices == [('a', 'A'), ('b', 'B')])
    assert var_10 is True
    var_11 = var_8.coerce_types
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Choice(choices=var_0, **var_1)
    var_3 = var_2.choices
    var_4 = bool(var_2.choices == [])
    assert var_4 is True
    var_5 = var_2.coerce_types
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Choice(**var_0)
    var_2 = var_1.choices
    var_3 = bool(var_1.choices == [])
    assert var_3 is True
    var_4 = var_1.coerce_types
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.choices
    var_6 = bool(var_4.choices == [('a', 'a'), ('b', 'b')])
    assert var_6 is True
    var_7 = var_4.coerce_types
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = {}
    var_9 = module_0.Choice(choices=var_6, coerce_types=var_7, **var_8)
    var_10 = var_9.choices
    var_11 = bool(var_9.choices == [('a', 'A'), ('b', 'B')])
    assert var_11 is True
    var_12 = var_9.coerce_types
    assert var_12 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Description'
    var_2 = True
    var_3 = 'title'
    var_4 = 'description'
    var_5 = 'allow_null'
    var_6 = 'read_only'
    var_7 = {var_3: var_0, var_4: var_1, var_5: var_2, var_6: var_2}
    var_8 = module_0.Choice(**var_7)
    var_9 = var_8.title
    assert var_9 == 'Test'
    var_10 = var_8.description
    assert var_10 == 'Description'
    var_11 = var_8.allow_null
    assert var_11 is True
    var_12 = var_8.read_only
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Choice(**var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = lambda : var_0
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.Choice(**var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Choice(**var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = 'extra'
    var_3 = (var_0, var_1, var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 10.0
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 9.9
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_with_null_value_and_allow_null_false. Retrieved 6/8 statements.
# Partially parsed test_validate_with_invalid_value_and_single_candidate_error. Retrieved 7/9 statements.
# Partially parsed test_validate_with_invalid_value_and_multiple_candidate_errors. Retrieved 8/10 statements.
# Partially parsed test_validate_with_invalid_value_and_no_candidate_errors. Retrieved 6/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Union(var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Union(var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'test'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 42
    var_8 = var_6.validate(var_7)
    assert var_8 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Union(var_5, **var_6)
    var_8 = 'abc'
    var_9 = var_7.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 10
    var_4 = 'min_value'
    var_5 = {var_4: var_3}
    var_6 = module_0.Integer(**var_5)
    var_7 = [var_2, var_6]
    var_8 = {}
    var_9 = module_0.Union(var_7, **var_8)
    var_10 = 'abc'
    var_11 = var_9.validate(var_10)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = []
    var_8 = var_6.validate(var_7)



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == {})
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = '123'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = var_2.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'a'
    var_7 = 'not an integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'test_a'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'test_a': 'value'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'test_a'
    var_7 = 'not an integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 1
    var_10 = 2
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = 'b'
    var_9 = 1
    var_10 = 'value'
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(var_12 == {'a': 1, 'b': 'value'})
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 1
    var_11 = 'not an integer'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = var_7.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.Integer(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'a': 0})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = 1
    var_10 = 'value'
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = bool(var_12 == {'a': 1, 'b': 'value'})
    assert var_13 is True



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = lambda : var_0
    var_2 = module_0.Field(default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    assert var_1 is None



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 6
    var_4 = var_2.validate(var_3)
    assert var_4 == 6



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'



# Parsed testcases at query #24
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Choice(choices=var_3, **var_6)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Choice(choices=var_3, **var_6)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Choice(choices=var_3, coerce_types=var_4, **var_6)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = True
    var_5 = False
    var_6 = 'allow_null'
    var_7 = {var_6: var_4}
    var_8 = module_0.Choice(choices=var_3, coerce_types=var_5, **var_7)
    var_9 = ''
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Choice(choices=var_3, **var_6)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    assert var_9 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = 'c'
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_1)
    assert var_9 == 'A'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = 'C'
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'True'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'False'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    assert var_9 is True
    var_10 = var_8.validate(var_3)
    assert var_10 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = 0
    var_4 = 'Zero'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    assert var_9 == 1
    var_10 = var_8.validate(var_3)
    assert var_10 == 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'True'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'False'
    var_5 = (var_3, var_4)
    var_6 = 'One'
    var_7 = (var_0, var_6)
    var_8 = 'Zero'
    var_9 = (var_3, var_8)
    var_10 = [var_2, var_5, var_7, var_9]
    var_11 = {}
    var_12 = module_0.Choice(choices=var_10, **var_11)
    var_13 = var_12.validate(var_0)
    assert var_13 is True
    var_14 = var_12.validate(var_3)
    assert var_14 is False
    var_15 = var_12.validate(var_0)
    assert var_15 == 1
    var_16 = var_12.validate(var_3)
    assert var_16 == 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'List'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = [var_0, var_1]
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == ['a', 'b'])
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {var_0: var_1}
    var_3 = 'Dict'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = {var_0: var_1}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'a': 'b'})
    assert var_10 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_unique_items_error_when_duplicate_found. Retrieved 7/9 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_3]
    var_6 = var_2.validate(var_5)
    var_7 = len(var_4)
    assert var_7 == 1



# Parsed testcases at query #26
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(var_8 == [1, 2, 3])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = '1'
    var_8 = 'hello'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == [1, 'hello'])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Array(var_2, var_4, **var_5)
    var_7 = '1'
    var_8 = 'hello'
    var_9 = 'world'
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(var_11 == [1, 'hello', 'world'])
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = '1'
    var_7 = 'hello'
    var_8 = [var_6, var_7]
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = '1'
    var_5 = 'invalid'
    var_6 = '3'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = '1'
    var_5 = 'invalid1'
    var_6 = 'invalid2'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = var_1.allow_null
    assert var_4 is False
    var_5 = var_1.read_only
    assert var_5 is False
    var_6 = var_1.items
    assert var_6 is None
    var_7 = var_1.additional_items
    assert var_7 is False
    var_8 = var_1.min_items
    assert var_8 is None
    var_9 = var_1.max_items
    assert var_9 is None
    var_10 = var_1.unique_items
    assert var_10 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 5
    var_3 = 'Custom Title'
    var_4 = 'Custom Description'
    var_5 = 'title'
    var_6 = 'description'
    var_7 = 'allow_null'
    var_8 = 'read_only'
    var_9 = {var_5: var_3, var_6: var_4, var_7: var_1, var_8: var_1}
    var_10 = module_0.Array(var_0, var_1, var_1, var_2, unique_items=var_1, **var_9)
    var_11 = var_10.title
    assert var_11 == 'Custom Title'
    var_12 = var_10.description
    assert var_12 == 'Custom Description'
    var_13 = var_10.allow_null
    assert var_13 is True
    var_14 = var_10.read_only
    assert var_14 is True
    var_15 = module_0.Field()
    var_16 = var_10.items
    var_17 = bool(var_10.items == var_15)
    assert var_17 is True
    var_18 = var_10.additional_items
    assert var_18 is True
    var_19 = var_10.min_items
    assert var_19 == 1
    var_20 = var_10.max_items
    assert var_20 == 5
    var_21 = var_10.unique_items
    assert var_21 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Array(var_2, **var_3)
    var_5 = var_4.items
    var_6 = bool(var_4.items == var_2)
    assert var_6 is True
    var_7 = var_4.min_items
    assert var_7 == 2
    var_8 = var_4.max_items
    assert var_8 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 == 3
    var_4 = var_2.max_items
    assert var_4 == 3

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Array(var_2, var_0, **var_3)
    var_5 = var_4.additional_items
    var_6 = bool(var_4.additional_items == var_0)
    assert var_6 is True
    var_7 = var_4.min_items
    assert var_7 == 1
    var_8 = var_4.max_items
    assert var_8 is None



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, var_2, **var_3)
    var_5 = var_4.max_items
    assert var_5 is None



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_0]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = 3
    var_5 = [var_3, var_0, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_0]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(var_8 == ['a', 'b', 'c'])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Boolean(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = 'a'
    var_10 = 1
    var_11 = True
    var_12 = False
    var_13 = [var_9, var_10, var_11, var_12]
    var_14 = var_8.validate(var_13)
    var_15 = bool(var_14 == ['a', 1, True, False])
    assert var_15 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Boolean(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = 'a'
    var_10 = 1
    var_11 = 'invalid'
    var_12 = [var_9, var_10, var_11]
    var_13 = var_8.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = False
    var_6 = {}
    var_7 = module_0.Array(var_4, var_5, **var_6)
    var_8 = 'a'
    var_9 = 1
    var_10 = [var_8, var_9]
    var_11 = var_7.validate(var_10)
    var_12 = bool(var_11 == ['a', 1])
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = False
    var_6 = {}
    var_7 = module_0.Array(var_4, var_5, **var_6)
    var_8 = 'a'
    var_9 = 1
    var_10 = 'extra'
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #30
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #31
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'not a field'
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #32
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, var_2, **var_3)
    var_5 = var_4.max_items
    assert var_5 is None



# Parsed testcases at query #33
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_unique_items_validation. Retrieved 8/10 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = len(var_4)
    assert var_8 == 1



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_validate_raises_type_error_for_invalid_value. Retrieved 5/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Boolean(coerce_types=var_0, **var_3)
    var_5 = 'invalid'
    var_6 = var_4.validate(var_5)
    var_7 = 'type'



# Parsed testcases at query #36
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_0]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = 3
    var_5 = [var_3, var_0, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_0]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_3]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = False
    var_4 = [var_0, var_3, var_0, var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [True, False, 1, 0])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = '1'
    var_5 = '2'
    var_6 = '3'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(var_8 == [1, 2, 3])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = '1'
    var_5 = 'two'
    var_6 = '3'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Boolean(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = 1
    var_10 = 'two'
    var_11 = True
    var_12 = False
    var_13 = [var_9, var_10, var_11, var_12]
    var_14 = var_8.validate(var_13)
    var_15 = bool(var_14 == [1, 'two', True, False])
    assert var_15 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = False
    var_6 = {}
    var_7 = module_0.Array(var_4, var_5, **var_6)
    var_8 = 1
    var_9 = 'two'
    var_10 = True
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = lambda : var_0
    var_2 = module_0.Field(default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    assert var_1 is None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_array_constructor_with_default_parameters. Retrieved 1/2 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Field()
    var_4 = 1
    var_5 = 5
    var_6 = 3
    var_7 = True
    var_8 = 'Test Array'
    var_9 = 'A test array field'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = True
    var_15 = True
    var_16 = 'title'
    var_17 = 'description'
    var_18 = 'default'
    var_19 = 'allow_null'
    var_20 = 'read_only'
    var_21 = {var_16: var_8, var_17: var_9, var_18: var_13, var_19: var_14, var_20: var_15}
    var_22 = module_0.Array(var_2, var_3, var_4, var_5, var_6, var_7, **var_21)
    var_23 = var_22.items
    var_24 = bool(var_22.items == var_2)
    assert var_24 is True
    var_25 = var_22.additional_items
    var_26 = bool(var_22.additional_items == var_3)
    assert var_26 is True
    var_27 = var_22.min_items
    var_28 = bool(var_22.min_items == var_6)
    assert var_28 is True
    var_29 = var_22.max_items
    var_30 = bool(var_22.max_items == var_6)
    assert var_30 is True
    var_31 = var_22.unique_items
    var_32 = bool(var_22.unique_items == var_7)
    assert var_32 is True
    var_33 = var_22.title
    var_34 = bool(var_22.title == var_8)
    assert var_34 is True
    var_35 = var_22.description
    var_36 = bool(var_22.description == var_9)
    assert var_36 is True
    var_37 = var_22.default
    var_38 = bool(var_22.default == var_13)
    assert var_38 is True
    var_39 = var_22.allow_null
    var_40 = bool(var_22.allow_null == var_14)
    assert var_40 is True
    var_41 = var_22.read_only
    var_42 = bool(var_22.read_only == var_15)
    assert var_42 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = var_2.items
    var_4 = bool(var_2.items == var_0)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = var_2.items
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = var_5.additional_items
    assert var_6 is False
    var_7 = var_5.max_items
    assert var_7 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = {}
    var_5 = module_0.Array(var_2, exact_items=var_3, **var_4)
    var_6 = var_5.min_items
    assert var_6 == 3
    var_7 = var_5.max_items
    assert var_7 == 3

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = {}
    var_5 = module_0.Array(var_2, unique_items=var_3, **var_4)
    var_6 = var_5.unique_items
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = var_1.items
    assert var_2 is None
    var_3 = var_1.additional_items
    assert var_3 is False
    var_4 = var_1.min_items
    assert var_4 is None
    var_5 = var_1.max_items
    assert var_5 is None
    var_6 = var_1.unique_items
    assert var_6 is False
    var_7 = var_1.title
    assert var_7 == ''
    var_8 = var_1.description
    assert var_8 == ''
    var_9 = var_1.allow_null
    assert var_9 is False
    var_10 = var_1.read_only
    assert var_10 is False



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = var_2.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = 'email'
    var_5 = 'test@example.com'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = '123'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'username'
    var_8 = 'extra'
    var_9 = 'test'
    var_10 = 'value'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = 'extra'
    var_9 = 'test'
    var_10 = 'value'
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(var_12 == {'username': 'test', 'extra': 'value'})
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^user_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'user_name'
    var_7 = 'other'
    var_8 = 'test'
    var_9 = 123
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = bool(var_11 == {'user_name': 'test'})
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = 'test'
    var_10 = 25
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = bool(var_12 == {'username': 'test', 'age': 25})
    assert var_13 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_choice_constructor_with_default. Retrieved 8/9 statements.
# Partially parsed test_choice_constructor_with_callable_default. Retrieved 9/10 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.choices
    var_10 = bool(var_8.choices == [('a', 'A'), ('b', 'B')])
    assert var_10 is True
    var_11 = var_8.coerce_types
    assert var_11 is True
    var_12 = var_8.title
    assert var_12 == ''
    var_13 = var_8.description
    assert var_13 == ''
    var_14 = var_8.allow_null
    assert var_14 is False
    var_15 = var_8.read_only
    assert var_15 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Choice(choices=var_1, **var_2)
    var_4 = var_3.choices
    var_5 = bool(var_3.choices == [('a', 'a')])
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = {}
    var_9 = module_0.Choice(choices=var_6, coerce_types=var_7, **var_8)
    var_10 = var_9.coerce_types
    assert var_10 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.allow_null
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'default'
    var_8 = {var_7: var_0}
    var_9 = module_0.Choice(choices=var_6, **var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda : var_0
    var_8 = 'default'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'Test Title'
    var_8 = 'Test Description'
    var_9 = 'title'
    var_10 = 'description'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_0.Choice(choices=var_6, **var_11)
    var_13 = var_12.title
    assert var_13 == 'Test Title'
    var_14 = var_12.description
    assert var_14 == 'Test Description'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = 'read_only'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.read_only
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Choice(choices=var_0, **var_1)
    var_3 = var_2.choices
    var_4 = bool(var_2.choices == [])
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Choice(choices=var_0, **var_1)
    var_3 = var_2.choices
    var_4 = bool(var_2.choices == [])
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = 'B'
    var_3 = (var_0, var_1, var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_array_constructor_with_defaults. Retrieved 1/2 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Boolean(**var_2)
    var_4 = 1
    var_5 = 5
    var_6 = None
    var_7 = True
    var_8 = 'Test Array'
    var_9 = 'A test array field'
    var_10 = []
    var_11 = True
    var_12 = False
    var_13 = 'title'
    var_14 = 'description'
    var_15 = 'default'
    var_16 = 'allow_null'
    var_17 = 'read_only'
    var_18 = {var_13: var_8, var_14: var_9, var_15: var_10, var_16: var_11, var_17: var_12}
    var_19 = module_0.Array(var_1, var_3, var_4, var_5, var_6, var_7, **var_18)
    var_20 = var_19.items
    var_21 = bool(var_19.items == var_1)
    assert var_21 is True
    var_22 = var_19.additional_items
    var_23 = bool(var_19.additional_items == var_3)
    assert var_23 is True
    var_24 = var_19.min_items
    assert var_24 == 1
    var_25 = var_19.max_items
    assert var_25 == 5
    var_26 = var_19.exact_items
    assert var_26 is None
    var_27 = var_19.unique_items
    assert var_27 is True
    var_28 = var_19.title
    assert var_28 == 'Test Array'
    var_29 = var_19.description
    assert var_29 == 'A test array field'
    var_30 = var_19.default
    var_31 = bool(var_19.default == [])
    assert var_31 is True
    var_32 = var_19.allow_null
    assert var_32 is True
    var_33 = var_19.read_only
    assert var_33 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = var_6.items
    var_8 = bool(var_6.items == var_4)
    assert var_8 is True
    var_9 = var_6.min_items
    assert var_9 == 2
    var_10 = var_6.max_items
    assert var_10 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 == 3
    var_4 = var_2.max_items
    assert var_4 == 3

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = var_1.items
    assert var_2 is None
    var_3 = var_1.additional_items
    assert var_3 is False
    var_4 = var_1.min_items
    assert var_4 is None
    var_5 = var_1.max_items
    assert var_5 is None
    var_6 = var_1.exact_items
    assert var_6 is None
    var_7 = var_1.unique_items
    assert var_7 is False
    var_8 = var_1.title
    assert var_8 == ''
    var_9 = var_1.description
    assert var_9 == ''
    var_10 = var_1.allow_null
    assert var_10 is False
    var_11 = var_1.read_only
    assert var_11 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = var_3.default
    assert var_4 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_non_integer_float_with_int_type. Retrieved 1/4 statements.
# Partially parsed test_validate_valid_string_with_int_type. Retrieved 1/3 statements.
# Partially parsed test_validate_valid_string_with_float_type. Retrieved 1/3 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.00'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.14159
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 3.14)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.2
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    assert var_3 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 3.14
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 3.14)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '42'
    var_3 = var_1.validate(var_2)
    assert var_3 == 42

def test_case_0():
    var_0 = '42'

def test_case_0():
    var_0 = '3.14'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 11
    var_4 = var_2.validate(var_3)
    assert var_4 == 11

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    assert var_4 == 9

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 6
    var_4 = var_2.validate(var_3)
    assert var_4 == 6

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.0
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.0)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Array(var_2, **var_3)
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = var_4.serialize(var_7)
    var_9 = bool(var_8 == [1, 2])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.serialize(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.serialize(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Field()
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = var_5.serialize(var_9)
    var_11 = bool(var_10 == [1, 2, 3])
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_string_constructor_with_defaults. Retrieved 1/2 statements.
# Partially parsed test_string_constructor_with_custom_values. Retrieved 10/12 statements.
# Partially parsed test_string_constructor_with_allow_blank_and_no_default. Retrieved 2/4 statements.
# Partially parsed test_string_constructor_with_allow_null_and_no_default. Retrieved 2/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = bool(not var_1.allow_null)
    assert var_4 is True
    var_5 = bool(not var_1.read_only)
    assert var_5 is True
    var_6 = bool(not var_1.allow_blank)
    assert var_6 is True
    var_7 = bool(var_1.trim_whitespace)
    assert var_7 is True
    var_8 = var_1.max_length
    assert var_8 is None
    var_9 = var_1.min_length
    assert var_9 is None
    var_10 = var_1.pattern
    assert var_10 is None
    var_11 = var_1.pattern_regex
    assert var_11 is None
    var_12 = var_1.format
    assert var_12 is None
    var_13 = bool(var_1.coerce_types)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = 'default_value'
    var_3 = True
    var_4 = False
    var_5 = 100
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = 'title'
    var_10 = 'description'
    var_11 = 'default'
    var_12 = 'allow_null'
    var_13 = 'read_only'
    var_14 = {var_9: var_0, var_10: var_1, var_11: var_2, var_12: var_3, var_13: var_3}
    var_15 = module_0.String(allow_blank=var_3, trim_whitespace=var_4, max_length=var_5, min_length=var_6, pattern=var_7, format=var_8, coerce_types=var_4, **var_14)
    var_16 = var_15.title
    assert var_16 == 'Test Title'
    var_17 = var_15.description
    assert var_17 == 'Test Description'
    var_18 = bool(var_15.allow_null)
    assert var_18 is True
    var_19 = bool(var_15.read_only)
    assert var_19 is True
    var_20 = bool(var_15.allow_blank)
    assert var_20 is True
    var_21 = bool(not var_15.trim_whitespace)
    assert var_21 is True
    var_22 = var_15.max_length
    assert var_22 == 100
    var_23 = var_15.min_length
    assert var_23 == 10
    var_24 = var_15.pattern
    assert var_24 == '^[a-z]+$'
    var_25 = var_15.pattern_regex.pattern
    assert var_25 == '^[a-z]+$'
    var_26 = var_15.format
    assert var_26 == 'email'
    var_27 = bool(not var_15.coerce_types)
    assert var_27 is True

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = var_3.pattern
    assert var_4 == '^[0-9]+$'
    var_5 = var_3.pattern_regex.pattern
    assert var_5 == '^[0-9]+$'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_field_or_creates_union_with_two_fields. Retrieved 5/6 statements.
# Partially parsed test_field_or_with_existing_union. Retrieved 8/9 statements.
# Partially parsed test_field_or_with_two_unions. Retrieved 11/12 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
    var_3 = module_0.Field(title=var_2)
    var_4 = var_1 | var_3
    var_5 = var_4.any_of
    var_6 = bool(var_4.any_of == [var_1, var_3])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
    var_3 = module_0.Field(title=var_2)
    var_4 = 'Field 3'
    var_5 = module_0.Field(title=var_4)
    var_6 = var_1 | var_3
    var_7 = var_6 | var_5
    var_8 = var_7.any_of
    var_9 = bool(var_7.any_of == [var_1, var_3, var_5])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
    var_3 = module_0.Field(title=var_2)
    var_4 = 'Field 3'
    var_5 = module_0.Field(title=var_4)
    var_6 = 'Field 4'
    var_7 = module_0.Field(title=var_6)
    var_8 = var_1 | var_3
    var_9 = var_5 | var_7
    var_10 = var_8 | var_9
    var_11 = var_10.any_of
    var_12 = bool(var_10.any_of == [var_1, var_3, var_5, var_7])
    assert var_12 is True



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = 'invalid@key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'key1'
    var_4 = 'value1'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'key3'
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = 'value3'
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = var_2.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'required_key'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = 'other_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 5
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, **var_2)
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = 'key'
    var_8 = 'too long value'
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^pattern_.*'
    var_1 = 5
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, **var_2)
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.Object(pattern_properties=var_4, **var_5)
    var_7 = 'pattern_key'
    var_8 = 'too long value'
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'allowed_key'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'allowed_key'
    var_8 = 'extra_key'
    var_9 = 'value'
    var_10 = {var_7: var_9, var_8: var_9}
    var_11 = var_6.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'allowed_key'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = 5
    var_5 = {}
    var_6 = module_0.String(max_length=var_4, **var_5)
    var_7 = {}
    var_8 = module_0.Object(properties=var_3, additional_properties=var_6, **var_7)
    var_9 = 'allowed_key'
    var_10 = 'extra_key'
    var_11 = 'value'
    var_12 = 'too long value'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = var_8.validate(var_13)
    var_15 = bool(False)
    assert var_15 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = 'value'
    var_10 = 123
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = bool(var_12 == {'key1': 'value', 'key2': 123})
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'key': 'default_value'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'extra_key'
    var_8 = 'value'
    var_9 = 'extra_value'
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = var_6.validate(var_10)
    var_12 = bool(var_11 == {'key': 'value', 'extra_key': 'extra_value'})
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^pattern_.*'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'pattern_key1'
    var_7 = 'pattern_key2'
    var_8 = 'value1'
    var_9 = 'value2'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = bool(var_11 == {'pattern_key1': 'value1', 'pattern_key2': 'value2'})
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'a\x00b'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'ab'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = '  hello  '
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  hello  '
    var_4 = var_2.validate(var_3)
    assert var_4 == '  hello  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hi'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'invalid-email'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)
    var_7 = var_6.validate(var_0)
    assert var_7 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)
    var_7 = 'c'
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, coerce_types=var_3, **var_5)
    var_7 = ''
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_3}
    var_7 = module_0.Choice(choices=var_2, coerce_types=var_4, **var_6)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_3}
    var_7 = module_0.Choice(choices=var_2, coerce_types=var_4, **var_6)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'true'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'false'
    var_5 = (var_3, var_4)
    var_6 = 'one'
    var_7 = (var_0, var_6)
    var_8 = 'zero'
    var_9 = (var_3, var_8)
    var_10 = [var_2, var_5, var_7, var_9]
    var_11 = {}
    var_12 = module_0.Choice(choices=var_10, **var_11)
    var_13 = var_12.validate(var_0)
    assert var_13 is True
    var_14 = var_12.validate(var_3)
    assert var_14 is False
    var_15 = var_12.validate(var_0)
    assert var_15 == 1
    var_16 = var_12.validate(var_3)
    assert var_16 == 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'list'
    var_4 = (var_2, var_3)
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'dict'
    var_8 = (var_6, var_7)
    var_9 = [var_4, var_8]
    var_10 = {}
    var_11 = module_0.Choice(choices=var_9, **var_10)
    var_12 = [var_0, var_1]
    var_13 = var_11.validate(var_12)
    var_14 = bool(var_13 == [1, 2])
    assert var_14 is True
    var_15 = {var_5: var_0}
    var_16 = var_11.validate(var_15)
    var_17 = bool(var_16 == {'a': 1})
    assert var_17 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pattern_regex_search_returns_false. Retrieved 5/7 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    var_6 = 'Must match the pattern /^[a-z]+$/.'
    var_7 = bool('Must match the pattern /^[a-z]+$/.' in var_5)
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Union(var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Union(var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'test'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 3.14
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Union(var_5, **var_6)
    var_8 = 'abc'
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'code'
    var_1 = 'index'
    var_2 = 'type'
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 1



# Parsed testcases at query #16
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    assert var_3 is True
    var_4 = False
    var_5 = var_1.validate(var_4)
    assert var_5 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'true'
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'false'
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'on'
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'off'
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = '1'
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = '0'
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 1
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 0
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = 'null'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = 'none'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'invalid'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'invalid'
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_0]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = 3
    var_5 = [var_3, var_0, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_0]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(var_8 == [1, 2, 3])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 'not an integer'
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 1
    var_8 = 'two'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == [1, 'two'])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 1
    var_8 = 'two'
    var_9 = 'extra'
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Boolean(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = 1
    var_10 = 'two'
    var_11 = True
    var_12 = False
    var_13 = [var_9, var_10, var_11, var_12]
    var_14 = var_8.validate(var_13)
    var_15 = bool(var_14 == [1, 'two', True, False])
    assert var_15 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Boolean(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = 1
    var_10 = 'two'
    var_11 = 'not a boolean'
    var_12 = [var_9, var_10, var_11]
    var_13 = var_8.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 1
    var_3 = 'two'
    var_4 = True
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)
    var_7 = bool(var_6 == [1, 'two', True])
    assert var_7 is True



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, coerce_types=var_3, **var_5)
    var_7 = ''
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_3}
    var_7 = module_0.Choice(choices=var_2, coerce_types=var_4, **var_6)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = ''
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)
    var_7 = var_6.validate(var_0)
    assert var_7 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)
    var_7 = 'c'
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'true'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'false'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    assert var_9 is True
    var_10 = var_8.validate(var_3)
    assert var_10 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'true'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'false'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = 1
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'list'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = [var_0, var_1]
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == ['a', 'b'])
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {var_0: var_1}
    var_3 = 'dict'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = {var_0: var_1}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'a': 'b'})
    assert var_10 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_with_float_and_numeric_type_int. Retrieved 1/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '42'
    var_4 = var_2.validate(var_3)
    assert var_4 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '42'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.14159
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 3.14)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    assert var_3 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 3.14
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 3.14)
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict or mapping'
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == var_5)
    assert var_7 is True



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'not a field'
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #24
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = '[a-z]+'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #26
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    assert var_9 == 'a'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_raises_type_error_for_invalid_value. Retrieved 5/7 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    var_6 = 'type'
    var_7 = bool('type' in var_5)
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.Number(coerce_types=var_1, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.validate(var_0)
    assert var_11 == 'a'
    var_12 = var_10.validate(var_3)
    assert var_12 == 'b'
    var_13 = 'c'
    var_14 = var_10.validate(var_13)
    assert var_14 is None



