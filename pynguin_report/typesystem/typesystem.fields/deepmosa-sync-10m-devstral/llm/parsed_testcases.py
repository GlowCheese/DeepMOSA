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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = False
    var_3 = var_1.validate(var_2)
    assert var_3 is False

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
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = 'invalid'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)

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



# Parsed testcases at query #2
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
    var_5 = 1
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
    var_1 = 'b'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_0]
    var_8 = 1
    var_9 = 3
    var_10 = True
    var_11 = {}
    var_12 = module_0.Object(properties=var_6, additional_properties=var_10, min_properties=var_8, max_properties=var_9, required=var_7, **var_11)
    var_13 = 'c'
    var_14 = 'test'
    var_15 = 'extra'
    var_16 = {var_0: var_10, var_1: var_14, var_13: var_15}
    var_17 = var_12.validate(var_16)
    var_18 = bool(var_17 == {'a': 1, 'b': 'test', 'c': 'extra'})
    assert var_18 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 10
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.Integer(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'a': 10})
    assert var_10 is True



# Parsed testcases at query #3
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
    var_0 = 'Custom Title'
    var_1 = 'Custom Description'
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
    assert var_16 == 'Custom Title'
    var_17 = var_15.description
    assert var_17 == 'Custom Description'
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



# Parsed testcases at query #4
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)

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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)

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



# Parsed testcases at query #5
#--------------------------




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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_object_constructor_with_valid_properties. Retrieved 21/23 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '^S_'
    var_6 = '^I_'
    var_7 = module_0.Field()
    var_8 = module_0.Field()
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = [var_0]
    var_11 = True
    var_12 = module_0.Field()
    var_13 = 10
    var_14 = 'Test Object'
    var_15 = 'A test object'
    var_16 = 'default'
    var_17 = {var_0: var_16}
    var_18 = False
    var_19 = 'title'
    var_20 = 'description'
    var_21 = 'default'
    var_22 = 'allow_null'
    var_23 = 'read_only'
    var_24 = {var_19: var_14, var_20: var_15, var_21: var_17, var_22: var_11, var_23: var_18}
    var_25 = module_0.Object(properties=var_4, pattern_properties=var_9, additional_properties=var_11, property_names=var_12, min_properties=var_11, max_properties=var_13, required=var_10, **var_24)
    var_26 = var_25.properties
    var_27 = bool(var_25.properties == var_4)
    assert var_27 is True
    var_28 = var_25.pattern_properties
    var_29 = bool(var_25.pattern_properties == var_9)
    assert var_29 is True
    var_30 = var_25.additional_properties
    assert var_30 is True
    var_31 = var_25.property_names
    var_32 = var_25.min_properties
    assert var_32 == 1
    var_33 = var_25.max_properties
    assert var_33 == 10
    var_34 = var_25.required
    var_35 = bool(var_25.required == var_10)
    assert var_35 is True
    var_36 = var_25.title
    assert var_36 == 'Test Object'
    var_37 = var_25.description
    assert var_37 == 'A test object'
    var_38 = var_25.allow_null
    assert var_38 is True
    var_39 = var_25.read_only
    assert var_39 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = var_1.properties
    var_3 = bool(var_1.properties == {})
    assert var_3 is True
    var_4 = var_1.pattern_properties
    var_5 = bool(var_1.pattern_properties == {})
    assert var_5 is True
    var_6 = var_1.additional_properties
    assert var_6 is True
    var_7 = var_1.property_names
    assert var_7 is None
    var_8 = var_1.min_properties
    assert var_8 is None
    var_9 = var_1.max_properties
    assert var_9 is None
    var_10 = var_1.required
    var_11 = bool(var_1.required == [])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = var_2.additional_properties
    var_4 = bool(var_2.additional_properties is var_0)
    assert var_4 is True
    var_5 = var_2.properties
    var_6 = bool(var_2.properties == var_0)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'not a Field'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'not a Field'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict or mapping'
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #9
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
    var_8 = 'a'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == [1, 'a'])
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
    var_8 = 'a'
    var_9 = 'b'
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(var_11 == [1, 'a', 'b'])
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
    var_7 = 'a'
    var_8 = [var_6, var_7]
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

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
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = [var_3, var_4]
    var_7 = [var_5, var_6]
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

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



# Parsed testcases at query #10
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
    var_10 = var_8.validate(var_3)
    assert var_10 == 'b'

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
    var_0 = True
    var_1 = 'True'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'False'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 'Zero'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 == 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 == 1

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'List'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = [var_0, var_1]
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == [1, 2])
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'Dict'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = {var_0: var_1}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'a': 1})
    assert var_10 is True



# Parsed testcases at query #11
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



# Parsed testcases at query #12
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
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
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 'b'
    var_7 = 2
    var_8 = {var_6: var_7}
    var_9 = [var_5, var_8]
    var_10 = var_2.serialize(var_9)
    var_11 = bool(var_10 == [{'a': 1}, {'b': 2}])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Field()
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = 1
    var_7 = 'a'
    var_8 = 3.14
    var_9 = [var_6, var_7, var_8]
    var_10 = var_5.serialize(var_9)
    var_11 = bool(var_10 == [1, 'a', 3.14])
    assert var_11 is True



# Parsed testcases at query #13
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
    var_0 = 1.5
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
    var_2 = '123'
    var_3 = var_1.validate(var_2)
    assert var_3 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    assert var_4 == 10



# Parsed testcases at query #14
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hi'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello world'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'Hello123'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'invalid-email'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pattern_properties_validation_error. Retrieved 8/11 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'test_key'
    var_7 = 123
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)



# Parsed testcases at query #16
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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'not_a_number'
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
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = True
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_0.Union(var_4, **var_7)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = False
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_0.Union(var_4, **var_7)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = e.messages()[0].code
    assert var_11 == 'null'

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
    var_9 = 123
    var_10 = var_6.validate(var_9)
    assert var_10 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 123.45
    var_8 = var_6.validate(var_7)
    var_9 = e.messages()[0].code
    assert var_9 == 'type'

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
    var_11 = e.messages()[0].code
    assert var_11 == 'union'



# Parsed testcases at query #18
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
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 105
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 100
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 7
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #19
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



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'invalid'
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_array_constructor_with_defaults. Retrieved 3/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 5
    var_3 = 'Test'
    var_4 = 'Test Description'
    var_5 = []
    var_6 = 'title'
    var_7 = 'description'
    var_8 = 'default'
    var_9 = 'allow_null'
    var_10 = 'read_only'
    var_11 = {var_6: var_3, var_7: var_4, var_8: var_5, var_9: var_1, var_10: var_1}
    var_12 = module_0.Array(var_0, var_1, var_1, var_2, unique_items=var_1, **var_11)
    var_13 = module_0.Field()
    var_14 = var_12.items
    var_15 = bool(var_12.items == var_13)
    assert var_15 is True
    var_16 = var_12.additional_items
    assert var_16 is True
    var_17 = var_12.min_items
    assert var_17 == 1
    var_18 = var_12.max_items
    assert var_18 == 5
    var_19 = var_12.unique_items
    assert var_19 is True
    var_20 = var_12.title
    assert var_20 == 'Test'
    var_21 = var_12.description
    assert var_21 == 'Test Description'
    var_22 = var_12.default
    var_23 = bool(var_12.default == [])
    assert var_23 is True
    var_24 = var_12.allow_null
    assert var_24 is True
    var_25 = var_12.read_only
    assert var_25 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = var_5.items
    var_7 = bool(var_5.items == [var_0, var_1])
    assert var_7 is True
    var_8 = var_5.additional_items
    assert var_8 is False
    var_9 = var_5.min_items
    assert var_9 == 2
    var_10 = var_5.max_items
    assert var_10 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 3
    var_2 = {}
    var_3 = module_0.Array(var_0, exact_items=var_1, **var_2)
    var_4 = var_3.min_items
    assert var_4 == 3
    var_5 = var_3.max_items
    assert var_5 == 3

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = module_0.Field()
    var_4 = var_2.items
    var_5 = bool(var_2.items == var_3)
    assert var_5 is True
    var_6 = var_2.additional_items
    assert var_6 is False
    var_7 = var_2.min_items
    assert var_7 is None
    var_8 = var_2.max_items
    assert var_8 is None
    var_9 = var_2.unique_items
    assert var_9 is False
    var_10 = var_2.title
    assert var_10 == ''
    var_11 = var_2.description
    assert var_11 == ''
    var_12 = var_2.allow_null
    assert var_12 is False
    var_13 = var_2.read_only
    assert var_13 is False



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



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {}
    var_4 = module_0.Array(var_2, unique_items=var_0, **var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == ['a', 'b', 'c'])
    assert var_10 is True



# Parsed testcases at query #26
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = 43
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = 42
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 34/38 statements.


import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'MockError'
    var_1 = ()
    var_2 = 'messages'
    var_3 = 'MockMessage'
    var_4 = ()
    var_5 = 'code'
    var_6 = 'index'
    var_7 = 'type'
    var_8 = None
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = [var_3, var_4, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = var_12()
    var_14 = [var_13]
    var_15 = lambda self: var_14
    var_16 = {var_2: var_15}
    var_17 = [var_0, var_1, var_16]
    var_18 = {}
    var_19 = module_0.type(*var_17, **var_18)
    var_20 = var_19()
    var_21 = 'MockChild'
    var_22 = ()
    var_23 = 'validate_or_error'
    var_24 = (var_8, var_20)
    var_25 = lambda self, value: var_24
    var_26 = {var_23: var_25}
    var_27 = [var_21, var_22, var_26]
    var_28 = {}
    var_29 = module_0.type(*var_27, **var_28)
    var_30 = var_29()
    var_31 = [var_30]
    var_32 = {}
    var_33 = module_1.Union(var_31, **var_32)
    var_34 = 1
    var_35 = 0
    var_36 = var_20.messages()[var_35]
    var_37 = var_36.code
    var_38 = var_37 != var_7
    var_39 = var_20.messages()[var_35]
    var_40 = var_39.index



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 is None



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_const_constructor_with_default. Retrieved 3/5 statements.
# Partially parsed test_const_constructor_with_callable_default. Retrieved 4/6 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'Test'
    var_2 = 'A test field'
    var_3 = 'title'
    var_4 = 'description'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Const(var_0, **var_5)
    var_7 = var_6.const
    assert var_7 == 42
    var_8 = var_6.title
    assert var_8 == 'Test'
    var_9 = var_6.description
    assert var_9 == 'A test field'
    var_10 = var_6.allow_null
    assert var_10 is False
    var_11 = var_6.read_only
    assert var_11 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = 'Null Test'
    var_2 = 'title'
    var_3 = {var_2: var_1}
    var_4 = module_0.Const(var_0, **var_3)
    var_5 = var_4.const
    assert var_5 is None
    var_6 = var_4.title
    assert var_6 == 'Null Test'
    var_7 = var_4.allow_null
    assert var_7 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = 50
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.Const(var_0, **var_3)
    var_5 = var_4.const
    assert var_5 == 100

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 'default_value'
    var_2 = lambda : var_1
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.Const(var_0, **var_4)
    var_6 = var_5.const
    assert var_6 == 'value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.Const(var_0, **var_2)
    var_4 = var_3.const
    assert var_4 is True
    var_5 = var_3.read_only
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Const(var_0, **var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 12/16 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'type'
    var_8 = 'code'
    var_9 = 'index'
    var_10 = None
    var_11 = {var_8: var_7, var_9: var_10}
    var_12 = [var_11]
    var_13 = {var_7: var_12}
    var_14 = 1



# Parsed testcases at query #31
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(allow_blank=var_1, trim_whitespace=var_0, coerce_types=var_0, **var_3)
    var_5 = '   '
    var_6 = var_4.validate(var_5)
    assert var_6 is None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_raises_type_error_for_invalid_value. Retrieved 5/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Boolean(coerce_types=var_0, **var_3)
    var_5 = 'invalid_value'
    var_6 = var_4.validate(var_5)



# Parsed testcases at query #33
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'not a Field instance'
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #34
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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_union_predicate_evaluates_to_true. Retrieved 8/18 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'type'
    var_3 = [var_0]
    var_4 = {}
    var_5 = module_0.Union(var_3, **var_4)
    var_6 = 'not_type'
    var_7 = 'test_value'
    var_8 = var_5.validate(var_7)
    assert var_8 is None



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_validate_non_integer_float_with_int_type. Retrieved 1/4 statements.
# Partially parsed test_validate_with_int_type. Retrieved 1/3 statements.
# Partially parsed test_validate_with_float_type. Retrieved 1/3 statements.


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
    var_2 = '3.14'
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 3.14)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.14159
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 3.14)
    assert var_5 is True

def test_case_0():
    var_0 = 3.0

def test_case_0():
    var_0 = 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 2.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 2.5)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    assert var_4 == 9



# Parsed testcases at query #37
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



# Parsed testcases at query #38
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = [var_0, var_3, var_0]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #39
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
    var_10 = var_8.validate(var_3)
    assert var_10 == 'b'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = [var_0, var_1]
    var_3 = 'b'
    var_4 = 'B'
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    assert var_9 == 'a'
    var_10 = var_8.validate(var_3)
    assert var_10 == 'b'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    assert var_9 == 'a'
    var_10 = var_8.validate(var_3)
    assert var_10 == 'b'

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
    var_0 = 1.5
    var_1 = 'One Point Five'
    var_2 = (var_0, var_1)
    var_3 = 0.0
    var_4 = 'Zero'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    var_10 = bool(var_9 == 1.5)
    assert var_10 is True
    var_11 = var_8.validate(var_3)
    var_12 = bool(var_11 == 0.0)
    assert var_12 is True

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
    var_10 = var_8.validate(var_3)
    assert var_10 == 'b'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'List 1,2'
    var_4 = (var_2, var_3)
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = 'List 3,4'
    var_9 = (var_7, var_8)
    var_10 = [var_4, var_9]
    var_11 = {}
    var_12 = module_0.Choice(choices=var_10, **var_11)
    var_13 = [var_0, var_1]
    var_14 = var_12.validate(var_13)
    var_15 = bool(var_14 == [1, 2])
    assert var_15 is True
    var_16 = [var_5, var_6]
    var_17 = var_12.validate(var_16)
    var_18 = bool(var_17 == [3, 4])
    assert var_18 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'Dict a:1'
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = 'Dict b:2'
    var_9 = (var_7, var_8)
    var_10 = [var_4, var_9]
    var_11 = {}
    var_12 = module_0.Choice(choices=var_10, **var_11)
    var_13 = {var_0: var_1}
    var_14 = var_12.validate(var_13)
    var_15 = bool(var_14 == {'a': 1})
    assert var_15 is True
    var_16 = {var_5: var_6}
    var_17 = var_12.validate(var_16)
    var_18 = bool(var_17 == {'b': 2})
    assert var_18 is True



# Parsed testcases at query #40
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
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'age'
    var_7 = 'not an integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^num_'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'num_1'
    var_7 = 'other'
    var_8 = '123'
    var_9 = 'value'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = bool(var_11 == {'num_1': 123, 'other': 'value'})
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'name'
    var_8 = 'age'
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = 'age'
    var_9 = 'John'
    var_10 = '30'
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(var_12 == {'name': 'John', 'age': 30})
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = 18
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.Integer(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = 'name'
    var_9 = 'John'
    var_10 = {var_8: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = bool(var_11 == {'name': 'John', 'age': 18})
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = bool(var_12 == {'name': 'John', 'age': 30})
    assert var_13 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'abc'
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
    var_3 = '3.14159'
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



# Parsed testcases at query #2
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
    var_10 = bool(e.messages() == [{'code': 'null', 'message': 'May not be null.'}])
    assert var_10 is True

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
    var_10 = bool(e.messages() == [{'code': 'union', 'message': 'Did not match any valid type.'}])
    assert var_10 is True

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
    var_11 = bool(e.messages() == [{'code': 'min_length', 'message': 'Shorter than minimum length 5.'}])
    assert var_11 is True

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
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(e.messages() == [{'code': 'union', 'message': 'Did not match any valid type.'}])
    assert var_13 is True



# Parsed testcases at query #3
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
    var_2 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)

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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hi'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    assert var_4 == '123'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)

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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hel\x00lo'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None



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
    var_0 = 'username'
    var_1 = 'anonymous'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'username': 'anonymous'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'john'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'username': 'john'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'username'
    var_7 = 123
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^user_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'user_name'
    var_7 = 'john'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'user_name': 'john'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^user_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'name'
    var_7 = 'john'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'username'
    var_4 = 'extra'
    var_5 = 'john'
    var_6 = 'value'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'username'
    var_5 = 'extra'
    var_6 = 'john'
    var_7 = 'value'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = var_3.validate(var_8)
    var_10 = bool(var_9 == {'username': 'john', 'extra': 'value'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'username'
    var_5 = 'extra'
    var_6 = 'john'
    var_7 = 123
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = var_3.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serialize_with_valid_decimal. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_zero. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_negative_decimal. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_large_decimal. Retrieved 2/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = '10.5'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = '0'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = '-5.25'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = '999999999.999999'



# Parsed testcases at query #3
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



# Parsed testcases at query #4
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
    var_8 = var_6.validate(var_2)
    assert var_8 == 'b'

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
    var_0 = 1
    var_1 = 'one'
    var_2 = (var_0, var_1)
    var_3 = 0
    var_4 = 'zero'
    var_5 = (var_3, var_4)
    var_6 = True
    var_7 = 'true'
    var_8 = (var_6, var_7)
    var_9 = False
    var_10 = 'false'
    var_11 = (var_9, var_10)
    var_12 = [var_2, var_5, var_8, var_11]
    var_13 = {}
    var_14 = module_0.Choice(choices=var_12, **var_13)
    var_15 = var_14.validate(var_6)
    assert var_15 == 1
    var_16 = var_14.validate(var_9)
    assert var_16 == 0
    var_17 = True
    var_18 = var_14.validate(var_17)
    assert var_18 is True
    var_19 = False
    var_20 = var_14.validate(var_19)
    assert var_20 is False

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



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = var_1.properties
    var_3 = bool(var_1.properties == {})
    assert var_3 is True
    var_4 = var_1.pattern_properties
    var_5 = bool(var_1.pattern_properties == {})
    assert var_5 is True
    var_6 = var_1.additional_properties
    assert var_6 is True
    var_7 = var_1.property_names
    assert var_7 is None
    var_8 = var_1.min_properties
    assert var_8 is None
    var_9 = var_1.max_properties
    assert var_9 is None
    var_10 = var_1.required
    var_11 = bool(var_1.required == [])
    assert var_11 is True
    var_12 = var_1.title
    assert var_12 == ''
    var_13 = var_1.description
    assert var_13 == ''
    var_14 = var_1.allow_null
    assert var_14 is False
    var_15 = var_1.read_only
    assert var_15 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = var_4.properties
    var_6 = bool(var_4.properties == {'key': var_0})
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'pattern'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = var_4.pattern_properties
    var_6 = bool(var_4.pattern_properties == {'pattern': var_0})
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = var_2.additional_properties
    var_4 = bool(var_2.additional_properties == var_0)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Object(property_names=var_0, **var_1)
    var_3 = var_2.property_names
    var_4 = bool(var_2.property_names == var_0)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = var_2.min_properties
    assert var_3 == 1

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = var_2.max_properties
    assert var_3 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Object(required=var_2, **var_3)
    var_5 = var_4.required
    var_6 = bool(var_4.required == ['key1', 'key2'])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Object(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Test Title'
    var_7 = var_5.description
    assert var_7 == 'Test Description'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Object(properties=var_0, **var_1)
    var_3 = var_2.properties
    var_4 = bool(var_2.properties == {})
    assert var_4 is True
    var_5 = var_2.additional_properties
    var_6 = bool(var_2.additional_properties == var_0)
    assert var_6 is True



# Parsed testcases at query #6
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



# Parsed testcases at query #7
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)

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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'
    var_5 = 'hi'
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'
    var_5 = 'hello world'
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'
    var_5 = 'Hello123'
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'
    var_5 = 'invalid-email'
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'helloworld'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_float_value_and_numeric_type_int. Retrieved 1/4 statements.


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
    var_3 = '3.14159'
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
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    assert var_3 == 42
    var_4 = '42'
    var_5 = var_1.validate(var_4)
    assert var_5 == 42
    var_6 = 3.14
    var_7 = var_1.validate(var_6)
    var_8 = bool(var_7 == 3.14)
    assert var_8 is True



# Parsed testcases at query #9
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
    var_9 = 'c'
    var_10 = var_8.validate(var_9)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_11. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1.5



# Parsed testcases at query #11
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
    var_3 = False
    var_4 = 'True'
    var_5 = 'False'
    var_6 = [var_0, var_3, var_0, var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(var_7 == [True, False, 1, 0, 'True', 'False'])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = True
    var_4 = [var_3, var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #12
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
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True



# Parsed testcases at query #13
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
    var_3 = {}
    var_4 = module_0.Union(var_2, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)

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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_float_value_and_numeric_type_int. Retrieved 1/4 statements.


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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)

def test_case_0():
    var_0 = 3.14

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)

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

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)

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
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 6
    var_4 = var_2.validate(var_3)
    assert var_4 == 6



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_field_or_creates_union_with_two_fields. Retrieved 3/4 statements.
# Partially parsed test_field_or_with_existing_union. Retrieved 5/6 statements.
# Partially parsed test_field_or_with_two_unions. Retrieved 7/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = var_0 | var_1
    var_3 = var_2.any_of
    var_4 = bool(var_2.any_of == [var_0, var_1])
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = var_0 | var_1
    var_4 = var_3 | var_2
    var_5 = var_4.any_of
    var_6 = bool(var_4.any_of == [var_0, var_1, var_2])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = var_0 | var_1
    var_5 = var_2 | var_3
    var_6 = var_4 | var_5
    var_7 = var_6.any_of
    var_8 = bool(var_6.any_of == [var_0, var_1, var_2, var_3])
    assert var_8 is True



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = False
    var_2 = 1
    var_3 = 5
    var_4 = None
    var_5 = True
    var_6 = 'Test Array'
    var_7 = 'A test array field'
    var_8 = []
    var_9 = True
    var_10 = 'title'
    var_11 = 'description'
    var_12 = 'default'
    var_13 = 'allow_null'
    var_14 = 'read_only'
    var_15 = {var_10: var_6, var_11: var_7, var_12: var_8, var_13: var_9, var_14: var_1}
    var_16 = module_0.Array(var_0, var_1, var_2, var_3, var_4, var_5, **var_15)
    var_17 = module_0.Field()
    var_18 = var_16.items
    var_19 = bool(var_16.items == var_17)
    assert var_19 is True
    var_20 = var_16.additional_items
    assert var_20 is False
    var_21 = var_16.min_items
    assert var_21 == 1
    var_22 = var_16.max_items
    assert var_22 == 5
    var_23 = var_16.exact_items
    assert var_23 is None
    var_24 = var_16.unique_items
    assert var_24 is True
    var_25 = var_16.title
    assert var_25 == 'Test Array'
    var_26 = var_16.description
    assert var_26 == 'A test array field'
    var_27 = var_16.default
    var_28 = bool(var_16.default == [])
    assert var_28 is True
    var_29 = var_16.allow_null
    assert var_29 is True
    var_30 = var_16.read_only
    assert var_30 is False

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
    var_2 = {}
    var_3 = module_0.Array(var_1, var_0, **var_2)
    var_4 = var_3.additional_items
    var_5 = bool(var_3.additional_items == var_0)
    assert var_5 is True

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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_with_invalid_choice. Retrieved 11/14 statements.


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
    var_11 = 0



# Parsed testcases at query #19
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
    var_9 = 123
    var_10 = var_6.validate(var_9)
    assert var_10 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 12.34
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
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_string_constructor_defaults. Retrieved 1/2 statements.
# Partially parsed test_string_constructor_with_all_params. Retrieved 10/11 statements.
# Partially parsed test_string_constructor_allow_blank_sets_default. Retrieved 2/3 statements.
# Partially parsed test_string_constructor_allow_null_without_default. Retrieved 2/3 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = var_1.allow_null
    assert var_4 is False
    var_5 = var_1.read_only
    assert var_5 is False
    var_6 = var_1.allow_blank
    assert var_6 is False
    var_7 = var_1.trim_whitespace
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
    var_13 = var_1.coerce_types
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
    var_18 = var_15.default
    assert var_18 == 'default_value'
    var_19 = var_15.allow_null
    assert var_19 is True
    var_20 = var_15.read_only
    assert var_20 is True
    var_21 = var_15.allow_blank
    assert var_21 is True
    var_22 = var_15.trim_whitespace
    assert var_22 is False
    var_23 = var_15.max_length
    assert var_23 == 100
    var_24 = var_15.min_length
    assert var_24 == 10
    var_25 = var_15.pattern
    assert var_25 == '^[a-z]+$'
    var_26 = var_15.pattern_regex.pattern
    assert var_26 == '^[a-z]+$'
    var_27 = var_15.format
    assert var_27 == 'email'
    var_28 = var_15.coerce_types
    assert var_28 is False

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = var_3.pattern
    assert var_4 == '^[0-9]+$'
    var_5 = var_3.pattern_regex
    var_6 = bool(var_3.pattern_regex == var_1)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = var_2.default
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = var_3.default
    assert var_4 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)



# Parsed testcases at query #22
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
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
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
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 'a'
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == ['a', 2])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 1
    var_8 = 'a'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

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
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(var_11 == ['a', 2, 3])
    assert var_12 is True

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
    var_8 = 'b'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

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
    var_9 = 2
    var_10 = 'b'
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

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
    var_3 = False
    var_4 = [var_0, var_0, var_3, var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [True, 1, False, 0])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = [var_0, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = var_2.validate(var_8)
    var_10 = bool(var_9 == [[1, 2], [3, 4]])
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = [var_3, var_4]
    var_7 = [var_5, var_6]
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = [var_4, var_7]
    var_9 = var_2.validate(var_8)
    var_10 = bool(var_9 == [{'a': 1}, {'b': 2}])
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}
    var_7 = [var_5, var_6]
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = var_4.min_items
    assert var_5 == 1



# Parsed testcases at query #24
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = var_5.max_items
    assert var_6 is None



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = {}
    var_3 = module_0.Array(var_0, var_1, **var_2)
    var_4 = []
    var_5 = var_3.validate(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True



# Parsed testcases at query #26
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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'abc'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

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
    var_0 = 'age'
    var_1 = 0
    var_2 = 'min_value'
    var_3 = {var_2: var_1}
    var_4 = module_0.Integer(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = 'age'
    var_9 = -1
    var_10 = {var_8: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^num_'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'num_age'
    var_7 = 'not an integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'name'
    var_8 = 'age'
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = 'name'
    var_9 = 'age'
    var_10 = 'John'
    var_11 = 'not an integer'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = var_7.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_0]
    var_8 = 1
    var_9 = 3
    var_10 = True
    var_11 = {}
    var_12 = module_0.Object(properties=var_6, additional_properties=var_10, min_properties=var_8, max_properties=var_9, required=var_7, **var_11)
    var_13 = 'city'
    var_14 = 'John'
    var_15 = 30
    var_16 = 'NYC'
    var_17 = {var_0: var_14, var_1: var_15, var_13: var_16}
    var_18 = var_12.validate(var_17)
    var_19 = bool(var_18 == {'name': 'John', 'age': 30, 'city': 'NYC'})
    assert var_19 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = 18
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.Integer(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'age': 18})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = '^num_'
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {var_4: var_6}
    var_8 = {}
    var_9 = module_0.Object(properties=var_3, pattern_properties=var_7, **var_8)
    var_10 = 'num_age'
    var_11 = 'John'
    var_12 = 30
    var_13 = {var_0: var_11, var_10: var_12}
    var_14 = var_9.validate(var_13)
    var_15 = bool(var_14 == {'name': 'John', 'num_age': 30})
    assert var_15 is True



# Parsed testcases at query #27
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
    var_2 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)

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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'
    var_5 = 'hi'
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'
    var_5 = 'hello world'
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    assert var_4 == '123'
    var_5 = 'abc'
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'
    var_5 = 'invalid-email'
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'helloworld'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_with_null_value_and_allow_null. Retrieved 2/6 statements.
# Partially parsed test_validate_with_null_value_and_no_allow_null. Retrieved 2/7 statements.
# Partially parsed test_validate_with_matching_child. Retrieved 1/6 statements.
# Partially parsed test_validate_with_non_matching_children. Retrieved 1/7 statements.
# Partially parsed test_validate_with_single_candidate_error. Retrieved 2/8 statements.
# Partially parsed test_validate_with_multiple_candidate_errors. Retrieved 3/9 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 5
    var_1 = 'abc'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 5
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #29
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
    var_5 = True
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
    var_6 = True
    var_7 = 'True'
    var_8 = (var_6, var_7)
    var_9 = False
    var_10 = 'False'
    var_11 = (var_9, var_10)
    var_12 = [var_2, var_5, var_8, var_11]
    var_13 = {}
    var_14 = module_0.Choice(choices=var_12, **var_13)
    var_15 = var_14.validate(var_6)
    assert var_15 == 1
    var_16 = var_14.validate(var_9)
    assert var_16 == 0
    var_17 = True
    var_18 = var_14.validate(var_17)
    assert var_18 is True
    var_19 = False
    var_20 = var_14.validate(var_19)
    assert var_20 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'List'
    var_4 = (var_2, var_3)
    var_5 = 'a'
    var_6 = {var_5: var_0}
    var_7 = 'Dict'
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



# Parsed testcases at query #30
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    assert var_4 == '123'

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



# Parsed testcases at query #31
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
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'ab'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcd'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '123'
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = '  hello  '
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'he\x00llo'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'



# Parsed testcases at query #32
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 is None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_validate_raises_type_error_for_invalid_string. Retrieved 5/7 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    assert var_5 == 'Must be a number.'



# Parsed testcases at query #34
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = var_5.max_items
    assert var_6 is None



