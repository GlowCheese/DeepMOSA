####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'not a dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(property_names=var_0)
    var_2 = 'invalid_key!'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = var_1.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'required_key'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'test_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allowed_key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_0.Object(properties=var_3, additional_properties=var_0)
    var_5 = 'allowed_key'
    var_6 = 'invalid_key'
    var_7 = 'value'
    var_8 = {var_5: var_7, var_6: var_7}
    var_9 = var_4.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'allowed_key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_0.Object(properties=var_3, additional_properties=var_0)
    var_5 = 'additional_key'
    var_6 = 'value'
    var_7 = {var_1: var_6, var_5: var_6}
    var_8 = var_4.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = 'value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = None
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 123
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'a\x00b'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'ab'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  hello  '
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.String(min_length=var_0)
    var_2 = 'ab'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'abcd'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = 'invalid-email'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'valid string'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'valid string'



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_0.Choice(choices=var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = module_0.Choice(choices=var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_0.Choice(choices=var_2, coerce_types=var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = False
    var_5 = module_0.Choice(choices=var_2, coerce_types=var_4)
    var_6 = ''
    var_7 = var_5.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = module_0.Choice(choices=var_2)
    var_5 = ''
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.Choice(choices=var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.Choice(choices=var_4)
    var_6 = 'c'
    var_7 = var_5.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = True
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.Choice(choices=var_4, coerce_types=var_2)
    var_6 = var_5.validate(var_2)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = False
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = True
    var_6 = module_0.Choice(choices=var_4, coerce_types=var_5)
    var_7 = var_6.validate(var_2)
    assert var_7 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 1
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = True
    var_6 = module_0.Choice(choices=var_4, coerce_types=var_5)
    var_7 = var_6.validate(var_5)
    assert var_7 == 1

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 0
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = True
    var_6 = module_0.Choice(choices=var_4, coerce_types=var_5)
    var_7 = var_6.validate(var_2)
    assert var_7 == 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_2, var_3]
    var_5 = [var_2, var_3]
    var_6 = (var_4, var_5)
    var_7 = [var_1, var_6]
    var_8 = module_0.Choice(choices=var_7)
    var_9 = [var_2, var_3]
    var_10 = var_8.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = 'c'
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = (var_4, var_5)
    var_7 = [var_1, var_6]
    var_8 = module_0.Choice(choices=var_7)
    var_9 = {var_2: var_3}
    var_10 = var_8.validate(var_9)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Boolean()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = True
    var_2 = var_0.validate(var_1)
    assert var_2 is True
    var_3 = False
    var_4 = var_0.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'true'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'true'
    var_2 = var_0.validate(var_1)
    assert var_2 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'false'
    var_2 = var_0.validate(var_1)
    assert var_2 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'on'
    var_2 = var_0.validate(var_1)
    assert var_2 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'off'
    var_2 = var_0.validate(var_1)
    assert var_2 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = '1'
    var_2 = var_0.validate(var_1)
    assert var_2 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = '0'
    var_2 = var_0.validate(var_1)
    assert var_2 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = ''
    var_2 = var_0.validate(var_1)
    assert var_2 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 1
    var_2 = var_0.validate(var_1)
    assert var_2 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 0
    var_2 = var_0.validate(var_1)
    assert var_2 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = 'null'
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = 'none'
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'invalid'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.validate(var_4)



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_array_constructor_with_callable_default.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Field()
    var_4 = 1
    var_5 = 10
    var_6 = 5
    var_7 = True
    var_8 = 'Test Array'
    var_9 = 'A test array field'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = True
    var_15 = True
    var_16 = module_0.Array(var_2, var_3, var_4, var_5, var_6, var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Array(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = len(var_2)
    var_5 = len(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = module_0.Array(var_2, var_3)
    var_5 = len(var_2)
    var_6 = len(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Array(exact_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Array()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_none_and_allow_null. Retrieved 2/6 statements.
# Partially parsed test_validate_with_none_and_not_allow_null. Retrieved 2/7 statements.
# Partially parsed test_validate_with_matching_child. Retrieved 1/6 statements.
# Partially parsed test_validate_with_non_matching_child. Retrieved 2/8 statements.
# Partially parsed test_validate_with_candidate_error. Retrieved 2/8 statements.
# Partially parsed test_validate_with_multiple_candidate_errors. Retrieved 3/9 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = None

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]

def test_case_0():
    var_0 = 5
    var_1 = 'test'

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 'test'



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(properties=var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'pattern'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(pattern_properties=var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Object(additional_properties=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Object(property_names=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Object(max_properties=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Description'
    var_2 = True
    var_3 = module_0.Object()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Object(properties=var_0)



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.Array(var_2, var_3)



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

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
    var_0 = 3
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3, var_0]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(min_items=var_0)
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = var_1.validate(var_4)

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
    var_1 = module_0.Array(min_items=var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(max_items=var_0)
    var_2 = 1
    var_3 = [var_2, var_0]
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(max_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 'not an integer'
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 1
    var_5 = 'two'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = module_0.String()
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 1
    var_5 = 'two'
    var_6 = 'three'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 1
    var_5 = 'two'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.validate(var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_float_and_numeric_type_int. Retrieved 1/4 statements.


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
    var_0 = 3.14

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '42'
    var_3 = var_1.validate(var_2)
    assert var_3 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = module_0.Number(precision=var_0)
    var_2 = 3.14159
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(minimum=var_0)
    var_2 = 5
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_minimum=var_0)
    var_2 = 10
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.Number(maximum=var_0)
    var_2 = 105
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 100
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 7
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_string_constructor_with_defaults. Retrieved 1/2 statements.
# Partially parsed test_string_constructor_with_custom_values. Retrieved 10/11 statements.
# Partially parsed test_string_constructor_with_allow_blank_and_no_default. Retrieved 2/3 statements.
# Partially parsed test_string_constructor_with_allow_null_and_no_default. Retrieved 2/3 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()

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
    var_9 = module_0.String(allow_blank=var_3, trim_whitespace=var_4, max_length=var_5, min_length=var_6, pattern=var_7, format=var_8, coerce_types=var_4)

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.String(pattern=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = 'null'
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'not a dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(property_names=var_0)
    var_2 = 'invalid@key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = 'key3'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 'value3'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = var_1.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'required_key'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 5
    var_2 = module_0.String(max_length=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = 'key'
    var_6 = 'too long value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = 5
    var_2 = module_0.String(max_length=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(pattern_properties=var_3)
    var_5 = 'test_key'
    var_6 = 'too long value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'allowed_key'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'allowed_key'
    var_6 = 'extra_key'
    var_7 = 'value'
    var_8 = 'extra_value'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_4.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'allowed_key'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = 5
    var_4 = module_0.String(max_length=var_3)
    var_5 = module_0.Object(properties=var_2, additional_properties=var_4)
    var_6 = 'allowed_key'
    var_7 = 'extra_key'
    var_8 = 'value'
    var_9 = 'too long value'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = False
    var_7 = module_0.Object(properties=var_4, additional_properties=var_6, required=var_5)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = var_7.validate(var_10)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'optional'
    var_1 = 'default_value'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)



# Parsed testcases at query #15
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
    var_7 = False
    var_8 = True
    var_9 = module_0.Choice(choices=var_6, coerce_types=var_8)
    var_10 = 'invalid'
    var_11 = var_9.validate(var_10)



# Parsed testcases at query #16
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

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
    var_0 = 3
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3, var_0]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(min_items=var_0)
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = var_1.validate(var_4)

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
    var_1 = module_0.Array(min_items=var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Array(max_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3, var_0]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(max_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 'not an integer'
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 1
    var_5 = 'two'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = module_0.String()
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 1
    var_5 = 'two'
    var_6 = 'three'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 1
    var_5 = 'two'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3, var_3]
    var_5 = var_1.validate(var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_union_raises_single_candidate_error. Retrieved 11/20 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error1'
    var_3 = 'type'
    var_4 = module_0.Field()
    var_5 = 'error2'
    var_6 = 'other'
    var_7 = [var_0, var_4]
    var_8 = module_0.Union(var_7)
    var_9 = 'invalid'
    var_10 = var_8.validate(var_9)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_with_float_and_numeric_type_int. Retrieved 1/4 statements.


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
    var_0 = 3.14

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'abc'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = module_0.Number(precision=var_0)
    var_2 = '3.14159'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(minimum=var_0)
    var_2 = 3
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
    var_2 = 15
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 10
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 5
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 1.2
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42
    var_3 = '42'
    var_4 = var_0.validate(var_3)
    assert var_4 == 42
    var_5 = 3.14
    var_6 = var_0.validate(var_5)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 4/24 statements.


def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 0
    var_3 = var_7.validate_or_error(var_0)[var_1]



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'not a dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(property_names=var_0)
    var_2 = '123'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

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
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = '123'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'a'
    var_5 = 'abc'
    var_6 = '123'
    var_7 = '456'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = var_3.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 123
    var_8 = 456
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_4.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'b'
    var_6 = 123
    var_7 = 456
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = var_4.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 100
    var_2 = module_0.Integer()
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_union_predicate_evaluates_to_true. Retrieved 12/16 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_0.Union(var_1)
    var_3 = 'messages'
    var_4 = 'code'
    var_5 = 'index'
    var_6 = 'not_type'
    var_7 = None
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = [var_8]
    var_10 = {var_3: var_9}
    var_11 = 1



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = module_0.Array(var_2, min_items=var_3)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_raises_error_for_invalid_value. Retrieved 5/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Boolean(coerce_types=var_0)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #24
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = []
    var_2 = var_0.serialize(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.serialize(var_5)
    var_7 = var_0.serialize(var_2)
    var_8 = var_0.serialize(var_3)
    var_9 = var_0.serialize(var_4)
    var_10 = [var_7, var_8, var_9]

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = var_3.serialize(var_6)
    var_8 = var_0.serialize(var_4)
    var_9 = var_1.serialize(var_5)
    var_10 = [var_8, var_9]

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0]
    var_3 = module_0.Array(var_2, var_1)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.serialize(var_7)
    var_9 = var_0.serialize(var_4)
    var_10 = var_1.serialize(var_5)
    var_11 = var_1.serialize(var_6)
    var_12 = [var_9, var_10, var_11]

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.serialize(var_5)



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'not a dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(property_names=var_0)
    var_2 = '123'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

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
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = var_1.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = module_0.Object(required=var_2)
    var_4 = 'a'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.Integer()
    var_3 = module_0.String()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 1
    var_7 = 'hello'
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = 'a'
    var_5 = 'not an integer'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^a.*$'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'a'
    var_5 = 'ab'
    var_6 = 1
    var_7 = 2
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = var_3.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^a.*$'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'a'
    var_5 = 'not an integer'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 1
    var_8 = 2
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_4.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'b'
    var_6 = 1
    var_7 = 'hello'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = var_4.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 1
    var_8 = 123
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_4.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Integer()
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = [var_0, var_3]
    var_5 = False
    var_6 = module_0.Object(properties=var_2, additional_properties=var_5, required=var_4)
    var_7 = 'a'
    var_8 = 'c'
    var_9 = 'not an integer'
    var_10 = 1
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_numeric_type_is_int_and_value_is_non_integer_float. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1.5



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = '^a'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'b'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Union(var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = module_0.Union(var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 'test'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 3.14
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = 'abc'
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = 10
    var_3 = module_0.Integer()
    var_4 = [var_1, var_3]
    var_5 = module_0.Union(var_4)
    var_6 = 'abc'
    var_7 = var_5.validate(var_6)



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = 'null'
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_with_invalid_string_raises_type_error. Retrieved 2/7 statements.


def test_case_0():
    var_0 = True
    var_1 = 'invalid'



# Parsed testcases at query #31
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #32
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #33
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #34
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)



# Parsed testcases at query #35
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #36
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'not a dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = var_1.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = 'email'
    var_4 = 'test@example.com'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_0.Object(property_names=var_1)
    var_3 = 'abc'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'username'
    var_6 = 'extra'
    var_7 = 'test'
    var_8 = 'value'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_4.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'extra'
    var_6 = 'test'
    var_7 = 'value'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = var_4.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^user_'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'user_name'
    var_5 = 'other'
    var_6 = 'test'
    var_7 = 123
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = var_3.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'test'
    var_7 = 25
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = 'age'
    var_2 = 'guest'
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = {var_0: var_3, var_1: var_4}
    var_6 = module_0.Object(properties=var_5)
    var_7 = 25
    var_8 = {var_1: var_7}
    var_9 = var_6.validate(var_8)



# Parsed testcases at query #37
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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_float_non_integer_with_int_type. Retrieved 1/4 statements.


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
    var_0 = 3.14

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'abc'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.00'
    var_1 = module_0.Number(precision=var_0)
    var_2 = 3.14159
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(minimum=var_0)
    var_2 = 3
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
    var_2 = 15
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 10
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 5
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 1.2
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 3.14
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = '123'
    var_2 = var_0.validate(var_1)
    assert var_2 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 10
    var_3 = var_1.validate(var_2)
    assert var_3 == 10



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = None
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 123
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'a\x00b'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'ab'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  hello  '
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.String(min_length=var_0)
    var_2 = 'ab'
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
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = 'invalid-email'
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_string_constructor_defaults. Retrieved 1/2 statements.
# Partially parsed test_string_constructor_with_all_params. Retrieved 10/12 statements.
# Partially parsed test_string_constructor_with_allow_blank_no_default. Retrieved 2/4 statements.
# Partially parsed test_string_constructor_with_allow_null_no_default. Retrieved 2/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = 'default_value'
    var_3 = True
    var_4 = False
    var_5 = 10
    var_6 = 2
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_0.String(allow_blank=var_3, trim_whitespace=var_4, max_length=var_5, min_length=var_6, pattern=var_7, format=var_8, coerce_types=var_4)

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.String(pattern=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.String(max_length=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.String(min_length=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.String(pattern=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.String(format=var_0)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'not a dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_0.Object(property_names=var_1)
    var_3 = 'abc'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

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
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = var_1.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'username'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = 'email'
    var_4 = 'test@example.com'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = 18
    var_2 = module_0.Integer()
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = 'age'
    var_6 = 17
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'test_key'
    var_5 = 'other'
    var_6 = 'value'
    var_7 = 123
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = var_3.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'test_key'
    var_5 = 'not an integer'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'allowed'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'allowed'
    var_6 = 'not_allowed'
    var_7 = 'value'
    var_8 = {var_5: var_7, var_6: var_7}
    var_9 = var_4.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'allowed'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Integer()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'extra'
    var_6 = 'value'
    var_7 = 123
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = var_4.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'allowed'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Integer()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'allowed'
    var_6 = 'extra'
    var_7 = 'value'
    var_8 = 'not an integer'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_4.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'optional'
    var_1 = 'default_value'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

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
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'extra'
    var_6 = 'John'
    var_7 = 'value'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = var_4.validate(var_8)



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_0.Choice(choices=var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = module_0.Choice(choices=var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = module_0.Choice(choices=var_2, coerce_types=var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = True
    var_4 = False
    var_5 = module_0.Choice(choices=var_2, coerce_types=var_4)
    var_6 = ''
    var_7 = var_5.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = module_0.Choice(choices=var_2)
    var_5 = ''
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.Choice(choices=var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.Choice(choices=var_4)
    var_6 = 'c'
    var_7 = var_5.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'True'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'False'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 is True
    var_9 = var_7.validate(var_3)
    assert var_9 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = '1'
    var_2 = (var_0, var_1)
    var_3 = 0
    var_4 = '0'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 1
    var_9 = var_7.validate(var_3)
    assert var_9 == 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'list'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = module_0.Choice(choices=var_5)
    var_7 = [var_0, var_1]
    var_8 = var_6.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {var_0: var_1}
    var_3 = 'dict'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = module_0.Choice(choices=var_5)
    var_7 = {var_0: var_1}
    var_8 = var_6.validate(var_7)



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Union(var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Union(var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 'test'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 3.14
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = 'abc'
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = 10
    var_3 = module_0.Integer()
    var_4 = [var_1, var_3]
    var_5 = module_0.Union(var_4)
    var_6 = 'abc'
    var_7 = var_5.validate(var_6)



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Union(var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Boolean()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = True
    var_2 = var_0.validate(var_1)
    assert var_2 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = False
    var_2 = var_0.validate(var_1)
    assert var_2 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'true'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'true'
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'false'
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'on'
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'off'
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = '1'
    var_3 = var_1.validate(var_2)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = '0'
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is False

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
    var_2 = 0
    var_3 = var_1.validate(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'null'
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'none'
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'invalid'
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = []
    var_2 = var_0.serialize(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.serialize(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.serialize(var_5)
    var_7 = var_0.serialize(var_2)
    var_8 = var_0.serialize(var_3)
    var_9 = var_0.serialize(var_4)
    var_10 = [var_7, var_8, var_9]

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = var_3.serialize(var_6)
    var_8 = var_0.serialize(var_4)
    var_9 = var_1.serialize(var_5)
    var_10 = [var_8, var_9]

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.serialize(var_7)
    var_9 = var_0.serialize(var_4)
    var_10 = var_1.serialize(var_5)
    var_11 = [var_9, var_10, var_6]

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1]
    var_4 = module_0.Array(var_3, var_2)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.serialize(var_8)
    var_10 = var_0.serialize(var_5)
    var_11 = var_1.serialize(var_6)
    var_12 = var_2.serialize(var_7)
    var_13 = [var_10, var_11, var_12]



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

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
    var_0 = 3
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3, var_0]
    var_5 = var_1.validate(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = var_1.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(min_items=var_0)
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = var_1.validate(var_4)
    var_6 = 1
    var_7 = [var_6]
    var_8 = var_1.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(max_items=var_0)
    var_2 = 1
    var_3 = [var_2, var_0]
    var_4 = var_1.validate(var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = var_1.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)
    var_7 = 1
    var_8 = 'not an integer'
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = var_1.validate(var_10)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 1
    var_5 = 'two'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)
    var_8 = 1
    var_9 = 2
    var_10 = [var_8, var_9]
    var_11 = var_3.validate(var_10)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = module_0.String()
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 1
    var_5 = 'two'
    var_6 = 'three'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = 1
    var_10 = 2
    var_11 = [var_9, var_10]
    var_12 = var_3.validate(var_11)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 1
    var_5 = [var_4]
    var_6 = var_3.validate(var_5)
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = var_3.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = var_1.validate(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7, var_6]
    var_9 = var_1.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = var_1.validate(var_7)
    var_9 = 1
    var_10 = 2
    var_11 = [var_9, var_10]
    var_12 = [var_9, var_10]
    var_13 = [var_11, var_12]
    var_14 = var_1.validate(var_13)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_0.Array(var_1)
    var_3 = None
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.serialize(var_5)
    var_7 = module_0.Integer()
    var_8 = module_0.String()
    var_9 = [var_7, var_8]
    var_10 = module_0.Array(var_9)
    var_11 = 'two'
    var_12 = [var_2, var_11]
    var_13 = var_10.serialize(var_12)
    var_14 = module_0.Integer()
    var_15 = module_0.String()
    var_16 = module_0.Array(var_14, var_15)
    var_17 = 'three'
    var_18 = [var_2, var_11, var_17]
    var_19 = var_16.serialize(var_18)



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Array(var_0)
    var_2 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = module_0.Field()
    var_5 = module_0.Field()
    var_6 = [var_4, var_5]

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Array(var_0, var_1)
    var_3 = module_0.Field()
    var_4 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 5
    var_2 = module_0.Array(var_0, min_items=var_1)
    var_3 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 10
    var_2 = module_0.Array(var_0, max_items=var_1)
    var_3 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 3
    var_2 = module_0.Array(var_0, exact_items=var_1)
    var_3 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_0.Array(var_0, unique_items=var_1)
    var_3 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'Test'
    var_2 = 'Description'
    var_3 = []
    var_4 = True
    var_5 = module_0.Array(var_0)
    var_6 = module_0.Field()

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Array(var_0)



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #16
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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = 'c'
    var_9 = var_7.validate(var_8)



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Boolean()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = True
    var_2 = var_0.validate(var_1)
    assert var_2 is True
    var_3 = False
    var_4 = var_0.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'true'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'true'
    var_2 = var_0.validate(var_1)
    assert var_2 is True
    var_3 = 'false'
    var_4 = var_0.validate(var_3)
    assert var_4 is False
    var_5 = 'on'
    var_6 = var_0.validate(var_5)
    assert var_6 is True
    var_7 = 'off'
    var_8 = var_0.validate(var_7)
    assert var_8 is False
    var_9 = '1'
    var_10 = var_0.validate(var_9)
    assert var_10 is True
    var_11 = '0'
    var_12 = var_0.validate(var_11)
    assert var_12 is False
    var_13 = ''
    var_14 = var_0.validate(var_13)
    assert var_14 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 1
    var_2 = var_0.validate(var_1)
    assert var_2 is True
    var_3 = 0
    var_4 = var_0.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = 'null'
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = 'none'
    var_5 = var_1.validate(var_4)
    assert var_5 is None
    var_6 = ''
    var_7 = var_1.validate(var_6)
    assert var_7 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'invalid'
    var_2 = var_0.validate(var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_with_none_and_allow_null. Retrieved 2/6 statements.
# Partially parsed test_validate_with_none_and_not_allow_null. Retrieved 2/7 statements.
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

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 3.14

def test_case_0():
    var_0 = 5
    var_1 = 'abc'

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 'abc'



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'Custom Title'
    var_2 = 'Custom Description'
    var_3 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = True
    var_2 = module_0.Const(var_0)



# Parsed testcases at query #20
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 is None



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.Union(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 7/11 statements.


import typesystem.fields as module_0
import locale as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_0.Union(var_1)
    var_3 = module_1.Error()
    var_4 = 'type'
    var_5 = None
    var_6 = 1



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #24
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Boolean(coerce_types=var_0)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #25
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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = 'c'
    var_9 = var_7.validate(var_8)
    var_10 = str(var_9)



# Parsed testcases at query #26
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = None
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 123
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = ''
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'a\x00b'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'ab'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = '  hello  '
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  hello  '
    var_3 = var_1.validate(var_2)
    assert var_3 == '  hello  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.String(min_length=var_0)
    var_2 = 'ab'
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
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = 'invalid-email'
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'not a dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(property_names=var_0)
    var_2 = 'invalid@key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = 'key3'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 'value3'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = var_1.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'required_key'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 5
    var_2 = module_0.String(max_length=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = 'key'
    var_6 = 'too long value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'test_key'
    var_5 = 'other_key'
    var_6 = 'value'
    var_7 = 123
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = var_3.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'allowed'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'allowed'
    var_6 = 'not_allowed'
    var_7 = 'value'
    var_8 = {var_5: var_7, var_6: var_7}
    var_9 = var_4.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'allowed'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Integer()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'additional'
    var_6 = 'value'
    var_7 = 123
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = var_4.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'allowed'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'additional'
    var_6 = 'value'
    var_7 = 123
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = var_4.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = 'value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

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
    var_0 = 3
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3, var_0]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(min_items=var_0)
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = var_1.validate(var_4)

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
    var_1 = module_0.Array(min_items=var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Array(max_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3, var_0]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(max_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 'not an integer'
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 1
    var_5 = 'two'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = module_0.String()
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 1
    var_5 = 'two'
    var_6 = 'three'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_0, var_2]
    var_8 = [var_3, var_6, var_7]
    var_9 = var_1.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_2, var_3]
    var_9 = [var_4, var_7, var_8]
    var_10 = var_1.validate(var_9)



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #30
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3, var_2]
    var_5 = var_1.validate(var_4)



# Parsed testcases at query #31
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #32
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'not a dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(property_names=var_0)
    var_2 = '123'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

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
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = var_1.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = 'a'
    var_5 = 'not an integer'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'abc'
    var_5 = 'not an integer'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 1
    var_8 = 2
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_4.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.Integer()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 1
    var_8 = 'not an integer'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_4.validate(var_9)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = module_0.Object(properties=var_2, additional_properties=var_3)
    var_5 = 'b'
    var_6 = 1
    var_7 = 'valid'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = var_4.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 0
    var_2 = module_0.Integer()
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = 'b'
    var_6 = 'valid'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)



# Parsed testcases at query #33
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 5
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_union_predicate_evaluates_to_true. Retrieved 11/17 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'code'
    var_3 = 'index'
    var_4 = 'type'
    var_5 = {var_2: var_4, var_3: var_1}
    var_6 = [var_0]
    var_7 = module_0.Union(var_6)
    var_8 = {var_2: var_4, var_3: var_1}
    var_9 = 'test_value'
    var_10 = var_7.validate(var_9)
    assert var_10 is None



# Parsed testcases at query #35
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_validate_non_integer_float_with_int_type_raises_error. Retrieved 1/4 statements.


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
    var_0 = 3.14

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = '-inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'nan'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

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
    var_2 = 5
    var_3 = var_1.validate(var_2)

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
    var_2 = 10
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 5
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 3.14
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = '42'
    var_2 = var_0.validate(var_1)
    assert var_2 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = '3.14'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = module_0.Number(precision=var_0)
    var_2 = 3.14159
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(minimum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(exclusive_minimum=var_0)
    var_2 = 6
    var_3 = var_1.validate(var_2)
    assert var_3 == 6

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(maximum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 9
    var_3 = var_1.validate(var_2)
    assert var_3 == 9

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 6
    var_3 = var_1.validate(var_2)
    assert var_3 == 6



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_validate_float_with_integer_type. Retrieved 1/4 statements.
# Partially parsed test_validate_valid_string_with_integer_type. Retrieved 2/4 statements.
# Partially parsed test_validate_valid_string_with_float_type. Retrieved 2/4 statements.


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
    var_0 = 3.14

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = module_0.Number(precision=var_0)
    var_2 = 3.14159
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(minimum=var_0)
    var_2 = 3
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
    var_2 = 15
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 10
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 5
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 1.2
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 3.14
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '42'
    var_3 = var_1.validate(var_2)
    assert var_3 == 42

def test_case_0():
    var_0 = True
    var_1 = '42'

def test_case_0():
    var_0 = True
    var_1 = '3.14'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 6
    var_3 = var_1.validate(var_2)
    assert var_3 == 6

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 1.5
    var_3 = var_1.validate(var_2)



