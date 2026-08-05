####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_object_constructor_type_conversions. Retrieved 11/14 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Prop 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Prop 2'
    var_3 = 'default'
    var_4 = module_0.Field(title=var_2, default=var_3)
    var_5 = 'p1'
    var_6 = 'p2'
    var_7 = {var_5: var_1, var_6: var_4}
    var_8 = '^\\d+$'
    var_9 = {var_8: var_1}
    var_10 = False
    var_11 = 'Name'
    var_12 = module_0.Field(title=var_11)
    var_13 = 1
    var_14 = 5
    var_15 = [var_5]
    var_16 = module_0.Object(properties=var_7, pattern_properties=var_9, additional_properties=var_10, property_names=var_12, min_properties=var_13, max_properties=var_14, required=var_15)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Extra'
    var_1 = module_0.Field(title=var_0)
    var_2 = module_0.Object(additional_properties=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Single'
    var_1 = module_0.Field(title=var_0)
    var_2 = module_0.Object(properties=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'P1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'p1'
    var_3 = {var_2: var_1}
    var_4 = '^a$'
    var_5 = {var_4: var_1}
    var_6 = (var_2,)
    var_7 = module_0.Object(properties=var_3, pattern_properties=var_5, required=var_6)
    var_8 = var_7.properties
    var_9 = var_7.pattern_properties
    var_10 = var_7.required



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 2
    var_2 = module_0.String(max_length=var_0, min_length=var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  spaced  '
    var_3 = var_1.validate(var_2)
    assert var_3 == 'spaced'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  spaced  '
    var_3 = var_1.validate(var_2)
    assert var_3 == '  spaced  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = '   '
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = '   '
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0, coerce_types=var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 123
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'abcd'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)
    assert var_3 == '123'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'a\x00b'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'ab'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.String(allow_blank=var_1, coerce_types=var_0)
    var_3 = '  '
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_valid_int. Retrieved 1/3 statements.
# Partially parsed test_validate_valid_float. Retrieved 1/3 statements.
# Partially parsed test_validate_valid_string_coercion. Retrieved 2/4 statements.
# Partially parsed test_validate_precision. Retrieved 2/4 statements.
# Partially parsed test_validate_integer_type_check_fails_on_float_fraction. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10.5

def test_case_0():
    var_0 = True
    var_1 = '10.5'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = True
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(minimum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 5
    var_3 = 4
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(exclusive_minimum=var_0)
    var_2 = 5.1
    var_3 = var_1.validate(var_2)
    var_4 = 5
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(maximum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 10
    var_3 = 11
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 9.9
    var_3 = var_1.validate(var_2)
    var_4 = 10
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 4
    var_3 = var_1.validate(var_2)
    assert var_3 == 4
    var_4 = 3
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 1.5
    var_3 = var_1.validate(var_2)
    var_4 = 1.2
    var_5 = var_1.validate(var_4)

def test_case_0():
    var_0 = '0.01'
    var_1 = 1.23456

def test_case_0():
    var_0 = 10.5

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'not-a-number'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_choice_validate_invalid_choice. Retrieved 6/10 statements.
# Partially parsed test_choice_validate_null_not_allowed. Retrieved 6/10 statements.
# Partially parsed test_choice_validate_empty_string_error_not_allowed_null. Retrieved 7/11 statements.
# Partially parsed test_choice_validate_empty_string_error_no_coercion. Retrieved 7/11 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = module_0.Choice(choices=var_2)
    var_4 = var_3.validate(var_0)
    assert var_4 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Alpha'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Beta'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = module_0.Choice(choices=var_2)
    var_4 = 'c'
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Choice(choices=var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Choice(choices=var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Choice(choices=var_1, coerce_types=var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = False
    var_3 = True
    var_4 = module_0.Choice(choices=var_1, coerce_types=var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = module_0.Choice(choices=var_1, coerce_types=var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = [var_0, var_1]
    var_3 = module_0.Choice(choices=var_2)
    var_4 = var_3.validate(var_0)
    assert var_4 is True
    var_5 = var_3.validate(var_1)
    assert var_5 is False
    var_6 = 1
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = [var_0, var_1]
    var_3 = module_0.Choice(choices=var_2)
    var_4 = var_3.validate(var_0)
    assert var_4 == 1
    var_5 = var_3.validate(var_1)
    assert var_5 == 0
    var_6 = True
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = [var_0, var_1]
    var_9 = var_7.validate(var_8)
    var_10 = {var_3: var_4}
    var_11 = var_7.validate(var_10)
    var_12 = 1
    var_13 = 3
    var_14 = [var_12, var_13]
    var_15 = var_7.validate(var_14)



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 2
    var_2 = module_0.String(max_length=var_0, min_length=var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  spaced  '
    var_3 = var_1.validate(var_2)
    assert var_3 == 'spaced'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  spaced  '
    var_3 = var_1.validate(var_2)
    assert var_3 == '  spaced  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'abc\x00def'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'abcdef'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 123
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = '   '
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = '   '
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'abcd'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = '12345'
    var_3 = var_1.validate(var_2)
    assert var_3 == '12345'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0, coerce_types=var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.String(allow_blank=var_1, coerce_types=var_0)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = False
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'true'
    var_3 = var_1.validate(var_2)
    assert var_3 is True
    var_4 = 'TRUE'
    var_5 = var_1.validate(var_4)
    assert var_5 is True
    var_6 = 'on'
    var_7 = var_1.validate(var_6)
    assert var_7 is True
    var_8 = '1'
    var_9 = var_1.validate(var_8)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'false'
    var_3 = var_1.validate(var_2)
    assert var_3 is False
    var_4 = 'off'
    var_5 = var_1.validate(var_4)
    assert var_5 is False
    var_6 = '0'
    var_7 = var_1.validate(var_6)
    assert var_7 is False
    var_8 = ''
    var_9 = var_1.validate(var_8)
    assert var_9 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 is True
    var_3 = 0
    var_4 = var_1.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = 'null'
    var_5 = var_1.validate(var_4)
    assert var_5 is None
    var_6 = 'none'
    var_7 = var_1.validate(var_6)
    assert var_7 is None
    var_8 = ''
    var_9 = var_1.validate(var_8)
    assert var_9 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'true'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Boolean(coerce_types=var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'not_a_boolean'
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 1
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_0.Object(properties=var_2, required=var_3)
    var_5 = 'age'
    var_6 = 30
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_1.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'val'
    var_8 = 'extra'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_4.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Integer()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'b'
    var_6 = 'val'
    var_7 = 123
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = var_4.validate(var_8)
    var_10 = 'a'
    var_11 = 'key'
    var_12 = 'val'
    var_13 = 'not_int'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = var_4.validate(var_14)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = module_0.Object(properties=var_4, required=var_5)
    var_7 = 'extra'
    var_8 = 'John'
    var_9 = 30
    var_10 = 'allowed'
    var_11 = {var_0: var_8, var_1: var_9, var_7: var_10}
    var_12 = var_6.validate(var_11)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Array'
    var_1 = 'A test array'
    var_2 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Item'
    var_1 = module_0.Field(title=var_0, description=var_0)
    var_2 = [var_1]
    var_3 = 'List Array'
    var_4 = module_0.Array(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Item'
    var_1 = module_0.Field(title=var_0, description=var_0)
    var_2 = module_0.Array(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Array(exact_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = 5
    var_2 = module_0.Array(min_items=var_0, max_items=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Extra'
    var_1 = module_0.Field(title=var_0, description=var_0)
    var_2 = module_0.Array(additional_items=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(additional_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'test_key'
    var_5 = 123
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = 'ValidationError was not raised for invalid pattern property value'
    var_9 = AssertionError(var_8)



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = 2
    var_4 = module_0.Array(var_2, exact_items=var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = var_4.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = 2
    var_4 = module_0.Array(var_2, exact_items=var_3)
    var_5 = 'a'
    var_6 = [var_5]
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 2
    var_2 = module_0.Array(var_0, min_items=var_1)
    var_3 = 'a'
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = module_0.Array(var_0, max_items=var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = module_0.Array(var_0, min_items=var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = True
    var_2 = module_0.Array(var_0, unique_items=var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = True
    var_2 = module_0.Array(var_0, unique_items=var_1)
    var_3 = 'a'
    var_4 = [var_3, var_3]
    var_5 = var_2.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = module_0.Integer()
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 'a'
    var_5 = 1
    var_6 = 2
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = module_0.Integer()
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 'a'
    var_5 = 'not_an_int'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = module_0.Array(var_1)
    var_3 = 'a'
    var_4 = 123
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_numeric_type_conversion_success. Retrieved 2/5 statements.
# Partially parsed test_validate_numeric_type_conversion_float_success. Retrieved 2/5 statements.
# Partially parsed test_validate_no_numeric_type_success. Retrieved 5/6 statements.


def test_case_0():
    var_0 = True
    var_1 = '123'

def test_case_0():
    var_0 = True
    var_1 = '123.45'

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.Number(coerce_types=var_1)
    var_3 = '123.45'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_string_constructor_pattern_as_string. Retrieved 5/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Name'
    var_1 = 'The user name'
    var_2 = module_0.String()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Code'
    var_1 = 'A code'
    var_2 = 10
    var_3 = 5
    var_4 = module_0.String(max_length=var_2, min_length=var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Empty'
    var_1 = 'Allowed to be empty'
    var_2 = True
    var_3 = module_0.String(allow_blank=var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Pattern'
    var_1 = 'Regex string'
    var_2 = '^\\d+$'
    var_3 = module_0.String(pattern=var_2)
    var_4 = var_3.pattern_regex

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '[a-z]+'
    var_1 = module_0.compile(var_0)
    var_2 = 'Regex'
    var_3 = 'Compiled regex'
    var_4 = module_1.String(pattern=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Trim'
    var_1 = 'No trim'
    var_2 = False
    var_3 = module_0.String(trim_whitespace=var_2, coerce_types=var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Format'
    var_1 = 'With format'
    var_2 = 'email'
    var_3 = module_0.String(format=var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Fail'
    var_1 = 'not_an_int'
    var_2 = module_0.String(max_length=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Fail'
    var_1 = 'not_an_int'
    var_2 = module_0.String(min_length=var_1)



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_.*'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'other_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_array_validate_unique_items_not_duplicate. Retrieved 5/16 statements.


def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_array_init_items_not_list. Retrieved 5/6 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Single Field'
    var_1 = module_0.Field(title=var_0)
    var_2 = 5
    var_3 = module_0.Array(var_1, min_items=var_2)
    var_4 = var_3.items



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_numeric_type_conversion_success. Retrieved 2/5 statements.
# Partially parsed test_validate_numeric_type_conversion_float_success. Retrieved 2/5 statements.
# Partially parsed test_validate_string_to_decimal_success. Retrieved 4/5 statements.


def test_case_0():
    var_0 = True
    var_1 = '123'

def test_case_0():
    var_0 = True
    var_1 = '123.45'

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.Number(coerce_types=var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    assert var_4 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '123.456'
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 2
    var_2 = module_0.String(max_length=var_0, min_length=var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  space  '
    var_3 = var_1.validate(var_2)
    assert var_3 == 'space'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  space  '
    var_3 = var_1.validate(var_2)
    assert var_3 == '  space  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = '   '
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = '   '
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0, coerce_types=var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 123
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'abcd'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)
    assert var_3 == '123'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello\x00world'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'helloworld'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.String(allow_blank=var_1, coerce_types=var_0)
    var_3 = '  '
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_numeric_type_int_float_not_integer. Retrieved 4/9 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = 1.5
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_array_validate_null_error. Retrieved 2/8 statements.
# Partially parsed test_array_validate_type_error. Retrieved 1/7 statements.
# Partially parsed test_array_validate_exact_items_success. Retrieved 4/10 statements.
# Partially parsed test_array_validate_exact_items_error. Retrieved 3/11 statements.
# Partially parsed test_array_validate_min_items_error. Retrieved 3/9 statements.


def test_case_0():
    var_0 = False
    var_1 = None

def test_case_0():
    var_0 = 'not a list'

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = [var_1]

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = [var_1]

def test_case_0():
    pass



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_numeric_type_int_float_not_integer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 1.5



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_valid_string_conversion. Retrieved 4/5 statements.
# Partially parsed test_validate_error_integer_type_constraint. Retrieved 1/4 statements.
# Partially parsed test_validate_precision_success. Retrieved 2/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = 10
    var_3 = var_1.validate(var_2)
    assert var_3 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = 10.5
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '10.5'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = True
    var_2 = var_0.validate(var_1)

def test_case_0():
    var_0 = 10.5

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(minimum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 5
    var_3 = 6
    var_4 = var_1.validate(var_3)
    assert var_4 == 6

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(minimum=var_0)
    var_2 = 4
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(exclusive_minimum=var_0)
    var_2 = 5.1
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(exclusive_minimum=var_0)
    var_2 = 5
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(maximum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 10
    var_3 = 9
    var_4 = var_1.validate(var_3)
    assert var_4 == 9

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(maximum=var_0)
    var_2 = 11
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 9.9
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 10
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 4
    var_3 = var_1.validate(var_2)
    assert var_3 == 4
    var_4 = 5
    var_5 = var_1.validate(var_4)
    assert var_5 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 5
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 1.0
    var_3 = var_1.validate(var_2)
    var_4 = 1.5
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 0.7
    var_3 = var_1.validate(var_2)

def test_case_0():
    var_0 = '0.01'
    var_1 = 1.2345

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '10'
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_const_constructor_valid_params. Retrieved 5/6 statements.
# Partially parsed test_const_constructor_type_checks. Retrieved 3/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 'Test Field'
    var_2 = 'A test description'
    var_3 = 5
    var_4 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = 'Null Field'
    var_2 = True
    var_3 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = True
    var_2 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'string_value'
    var_1 = module_0.Const(var_0)
    var_2 = var_1.const



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_raises_null_error_when_value_is_none_and_allow_null_is_false. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = None



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_string_constructor_with_default_value. Retrieved 4/6 statements.
# Partially parsed test_string_constructor_allow_blank_sets_empty_string_default. Retrieved 4/5 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Field'
    var_1 = 'Test Description'
    var_2 = module_0.String()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Custom'
    var_1 = 'Desc'
    var_2 = True
    var_3 = False
    var_4 = 10
    var_5 = 2
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(allow_blank=var_2, trim_whitespace=var_3, max_length=var_4, min_length=var_5, pattern=var_6, format=var_7, coerce_types=var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'T'
    var_1 = 'D'
    var_2 = 'hello'
    var_3 = module_0.String()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'T'
    var_1 = 'D'
    var_2 = True
    var_3 = module_0.String(allow_blank=var_2)

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'T'
    var_1 = 'D'
    var_2 = '^\\d+$'
    var_3 = module_0.compile(var_2)
    var_4 = module_1.String(pattern=var_3)
    var_5 = module_0.compile(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = 'D'
    var_2 = module_0.String()
    var_3 = 'T'
    var_4 = 'D'
    var_5 = 'long'
    var_6 = module_0.String(max_length=var_5)



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 2
    var_2 = module_0.String(max_length=var_0, min_length=var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  space  '
    var_3 = var_1.validate(var_2)
    assert var_3 == 'space'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  space  '
    var_3 = var_1.validate(var_2)
    assert var_3 == '  space  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = '   '
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = '   '
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0, coerce_types=var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 123
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'abcd'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)
    assert var_3 == '123'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'a\x00b'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'ab'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.String(allow_blank=var_1, coerce_types=var_0)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'extra'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = var_1.validate()

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'extra'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_0.Object(properties=var_2, required=var_3)
    var_5 = 'John'
    var_6 = {var_0: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_2: var_0, var_3: var_4}
    var_6 = {var_2: var_0}
    var_7 = var_1.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'extra'
    var_6 = 'John'
    var_7 = 'not allowed'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = {var_0: var_6}
    var_10 = var_4.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(property_names=var_0)
    var_2 = 'valid_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^pre_.*'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'pre_test'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = module_0.Object(properties=var_4, required=var_5)
    var_7 = 'John'
    var_8 = 30
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = var_6.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Array'
    var_1 = 'A test array'
    var_2 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Item 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Item 2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 0
    var_6 = module_0.Array(var_4, min_items=var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Array(exact_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Extra'
    var_1 = module_0.Field(title=var_0)
    var_2 = 1
    var_3 = module_0.Array(additional_items=var_1, min_items=var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Item 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = module_0.Array(var_2, var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = 5
    var_2 = module_0.Array(min_items=var_0, max_items=var_1)



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = 'not a list'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 'one'
    var_5 = [var_4]
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(min_items=var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(max_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = True
    var_2 = module_0.Array(var_0, unique_items=var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4, var_3]
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 'hello'
    var_5 = 42
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = module_0.Integer()
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 'first'
    var_5 = 100
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = module_0.Array(var_1)
    var_3 = 123
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = module_0.String(allow_blank=var_0, coerce_types=var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_object_validate_not_none. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_valid_int. Retrieved 1/4 statements.
# Partially parsed test_validate_valid_float. Retrieved 1/4 statements.
# Partially parsed test_validate_string_coercion. Retrieved 1/3 statements.
# Partially parsed test_validate_integer_type_constraint. Retrieved 1/4 statements.
# Partially parsed test_validate_precision. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10.5

def test_case_0():
    var_0 = '12.34'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = True
    var_2 = var_0.validate(var_1)

def test_case_0():
    var_0 = 10.5

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(minimum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 5
    var_3 = 4
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(exclusive_minimum=var_0)
    var_2 = 5.1
    var_3 = var_1.validate(var_2)
    var_4 = 5
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(maximum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 10
    var_3 = 11
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 9.9
    var_3 = var_1.validate(var_2)
    var_4 = 10
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 4
    var_3 = var_1.validate(var_2)
    assert var_3 == 4
    var_4 = 5
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 1.5
    var_3 = var_1.validate(var_2)
    var_4 = 1.2
    var_5 = var_1.validate(var_4)

def test_case_0():
    var_0 = '0.01'
    var_1 = 1.23456

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'not-a-number'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 'first'
    var_5 = 'second'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_object_property_missing_and_has_default. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'default_val'
    var_1 = 'test_key'
    var_2 = {}



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_object_validate_null_error. Retrieved 8/10 statements.
# Partially parsed test_object_validate_type_error. Retrieved 11/13 statements.
# Partially parsed test_object_validate_invalid_key_type. Retrieved 4/11 statements.
# Partially parsed test_object_validate_max_properties. Retrieved 13/15 statements.
# Partially parsed test_object_validate_min_properties_empty. Retrieved 8/10 statements.
# Partially parsed test_object_validate_required_property. Retrieved 9/16 statements.
# Partially parsed test_object_validate_success_simple. Retrieved 6/11 statements.
# Partially parsed test_object_validate_additional_properties_false. Retrieved 10/17 statements.
# Partially parsed test_object_validate_additional_properties_field. Retrieved 13/21 statements.
# Partially parsed test_object_validate_property_names_validation. Retrieved 9/15 statements.


import typesystem.fields as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = 'null'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = None
    var_7 = var_1.validate(var_6)

import typesystem.fields as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = 'type'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = 'not'
    var_7 = 'a'
    var_8 = 'dict'
    var_9 = [var_6, var_7, var_8]
    var_10 = var_1.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}

import typesystem.fields as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = 'max'
    var_3 = 'max_properties'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 1
    var_10 = 2
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_1.validate(var_11)

import typesystem.fields as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = 'empty'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = {}
    var_7 = var_1.validate(var_6)

import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = [var_1]
    var_3 = 'req'
    var_4 = 'required'
    var_5 = module_0.Message(text=var_3, code=var_4, key=var_1)
    var_6 = [var_5]
    var_7 = module_0.ValidationError(messages=var_6)
    var_8 = {}

def test_case_0():
    var_0 = 'valid_val'
    var_1 = None
    var_2 = 'name'
    var_3 = 'extra'
    var_4 = 'allowed'
    var_5 = {var_2: var_0, var_3: var_4}

def test_case_0():
    var_0 = 'val'
    var_1 = None
    var_2 = 'name'
    var_3 = False
    var_4 = 'invalid'
    var_5 = 'name'
    var_6 = 'extra'
    var_7 = 'val'
    var_8 = 'not_allowed'
    var_9 = {var_5: var_7, var_6: var_8}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'val'
    var_1 = None
    var_2 = 'err'
    var_3 = 'extra'
    var_4 = module_0.Message(text=var_2, code=var_2, key=var_3)
    var_5 = [var_4]
    var_6 = module_0.ValidationError(messages=var_5)
    var_7 = 'name'
    var_8 = 'name'
    var_9 = 'extra'
    var_10 = 'val'
    var_11 = 'bad'
    var_12 = {var_8: var_10, var_9: var_11}

import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'bad'
    var_2 = module_0.Message(text=var_1, code=var_1)
    var_3 = [var_2]
    var_4 = module_0.ValidationError(messages=var_3)
    var_5 = 'invalid property name'
    var_6 = 'invalid_key_name'
    var_7 = 1
    var_8 = {var_6: var_7}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_object_validate_invalid_key_type. Retrieved 8/11 statements.
# Partially parsed test_object_validate_required_property. Retrieved 4/10 statements.
# Partially parsed test_object_validate_successful_properties. Retrieved 5/10 statements.
# Partially parsed test_object_validate_additional_properties_schema. Retrieved 5/9 statements.
# Partially parsed test_object_validate_property_names_constraint. Retrieved 8/15 statements.
# Partially parsed test_object_validate_pattern_properties. Retrieved 6/12 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = 'not'
    var_3 = 'a'
    var_4 = 'dict'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)
    var_5 = 'err'
    var_6 = 'invalid_key'
    var_7 = module_1.Message(text=var_5, code=var_6, key=var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_1.validate(var_6)

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = [var_1]
    var_3 = {}

def test_case_0():
    var_0 = 'valid_val'
    var_1 = None
    var_2 = 'name'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'extra'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

def test_case_0():
    var_0 = 'validated_extra'
    var_1 = None
    var_2 = 'extra'
    var_3 = 1
    var_4 = {var_2: var_3}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'invalid_property'
    var_2 = 'bad'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_4 = None
    var_5 = 'bad'
    var_6 = 1
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 'val'
    var_1 = None
    var_2 = '^pre_'
    var_3 = 'pre_test'
    var_4 = 123
    var_5 = {var_3: var_4}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = 123
    var_3 = [var_2]
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Array(var_0)
    var_2 = []
    var_3 = module_0.Array(var_2)
    var_4 = []
    var_5 = module_0.Array(var_4)
    var_6 = module_0.String()
    var_7 = [var_6]
    var_8 = module_0.Array(var_7)
    var_9 = 123
    var_10 = [var_9]
    var_11 = var_8.validate(var_10)



# Parsed testcases at query #16
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 'Int Field'
    var_2 = 'A field with int const'
    var_3 = module_0.Const(var_0)
    var_4 = 'hello'
    var_5 = 'world'
    var_6 = module_0.Const(var_4)
    var_7 = None
    var_8 = True
    var_9 = module_0.Const(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = True
    var_2 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 123
    var_2 = module_0.Const(var_0)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_choice_constructor_with_default_value. Retrieved 4/5 statements.


import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'Test'
    var_2 = 'Desc'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = module_1.Choice(choices=var_5)

import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'val1'
    var_2 = 'Label 1'
    var_3 = (var_1, var_2)
    var_4 = 'val2'
    var_5 = 'Label 2'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_1.Choice(choices=var_7)

import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = None
    var_2 = module_1.Choice(choices=var_1)

import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = False
    var_4 = module_1.Choice(choices=var_2, coerce_types=var_3)

import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_1.Choice(choices=var_2)

import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = True
    var_4 = module_1.Choice(choices=var_2)



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Array'
    var_1 = 'A test array'
    var_2 = 2
    var_3 = module_0.Array(min_items=var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Item'
    var_1 = module_0.Field(title=var_0, description=var_0)
    var_2 = [var_1]
    var_3 = 1
    var_4 = module_0.Array(var_2, min_items=var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Array(exact_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Item'
    var_1 = module_0.Field(title=var_0, description=var_0)
    var_2 = [var_1, var_1]
    var_3 = False
    var_4 = module_0.Array(var_2, var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Item'
    var_1 = module_0.Field(title=var_0, description=var_0)
    var_2 = module_0.Array(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'not a field or list of fields'
    var_1 = module_0.Array(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'not an int'
    var_1 = module_0.Array(min_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.Array(additional_items=var_0)



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = 'not a list'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = 2
    var_3 = module_0.Array(var_1, exact_items=var_2)
    var_4 = 'one'
    var_5 = [var_4]
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 2
    var_2 = module_0.Array(var_0, min_items=var_1)
    var_3 = 'one'
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = module_0.Array(var_0, max_items=var_1)
    var_3 = 'one'
    var_4 = 'two'
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = module_0.Array(var_0, min_items=var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = True
    var_2 = module_0.Array(var_0, unique_items=var_1)
    var_3 = 'a'
    var_4 = [var_3, var_3]
    var_5 = var_2.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 'first'
    var_5 = 'second'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = module_0.String()
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 'one'
    var_5 = 'two'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = module_0.Array(var_1)
    var_3 = 123
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Array(var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = 'not a list'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 'a'
    var_3 = [var_2]
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Array(min_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(max_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 'a'
    var_3 = [var_2, var_2]
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 'hello'
    var_5 = 'world'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.String()
    var_2 = module_0.Array(var_0, var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Desc'
    var_2 = module_0.String()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Desc'
    var_2 = True
    var_3 = False
    var_4 = 10
    var_5 = 2
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = module_0.String(allow_blank=var_2, trim_whitespace=var_3, max_length=var_4, min_length=var_5, pattern=var_6, format=var_7, coerce_types=var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Desc'
    var_2 = True
    var_3 = module_0.String(allow_blank=var_2)

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.compile(var_0)
    var_2 = 'Test'
    var_3 = 'Desc'
    var_4 = module_1.String(pattern=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Desc'
    var_2 = 'not_an_int'
    var_3 = module_0.String(max_length=var_2)
    var_4 = 'Test'
    var_5 = 'Desc'
    var_6 = 'not_an_int'
    var_7 = module_0.String(min_length=var_6)



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = module_0.Choice(choices=var_2)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'a'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'b'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Alpha'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Beta'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = module_0.Choice(choices=var_6)
    var_9 = var_8.validate(var_0)
    assert var_9 == 'a'
    var_10 = var_8.validate(var_3)
    assert var_10 == 'b'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = module_0.Choice(choices=var_2)
    var_5 = 'c'
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.Choice(choices=var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = module_0.Choice(choices=var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.Choice(choices=var_2, coerce_types=var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = True
    var_5 = module_0.Choice(choices=var_2, coerce_types=var_4)
    var_6 = ''
    var_7 = var_5.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = module_0.Choice(choices=var_2, coerce_types=var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = [var_0, var_1, var_0, var_1]
    var_3 = module_0.Choice(choices=var_2)
    var_4 = var_3.validate(var_0)
    assert var_4 is True
    var_5 = var_3.validate(var_1)
    assert var_5 is False
    var_6 = var_3.validate(var_0)
    assert var_6 == 1
    var_7 = var_3.validate(var_1)
    assert var_7 == 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = module_0.Choice(choices=var_6)
    var_9 = [var_0, var_1]
    var_10 = var_8.validate(var_9)
    var_11 = {var_3: var_4}
    var_12 = var_8.validate(var_11)



