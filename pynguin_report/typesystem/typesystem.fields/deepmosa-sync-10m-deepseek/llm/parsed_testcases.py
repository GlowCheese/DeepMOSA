####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'helloworld'

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
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

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
    var_3 = module_0.String(trim_whitespace=var_0, coerce_types=var_0, **var_2)
    var_4 = '   '
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'abc'

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcde'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'abcde'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcdef'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

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
    var_3 = 'hello123'
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
    var_6 = 'Must be a valid email.'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'



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
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = len(e.messages())
    assert var_4 == 1
    var_5 = e.messages()[0].code
    assert var_5 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'invalid_key'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = ''
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'invalid_property'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'empty'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'min_properties'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'max_properties'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'required_key'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'required'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'key': 'default_value'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'key'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^a.*'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = 'abc'
    var_6 = 'def'
    var_7 = 'value'
    var_8 = 'ignored'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = var_4.validate(var_9)
    var_11 = bool(var_10 == {'abc': 'value'})
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'invalid_property'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = 'required_key'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, required=var_5, **var_6)
    var_8 = 'key'
    var_9 = None
    var_10 = {var_8: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = len(e.messages())
    assert var_12 == 2
    var_13 = {msg.code for msg in e.messages()}
    var_14 = bool(var_13 == {'null', 'required'})
    assert var_14 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_float_for_integer_field_raises_error. Retrieved 1/4 statements.
# Partially parsed test_validate_valid_integer_field. Retrieved 1/3 statements.
# Partially parsed test_validate_valid_float_field. Retrieved 1/3 statements.


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
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    assert var_4 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 123.5
    var_1 = bool(False)
    assert var_1 is True

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
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 12
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
    var_0 = '0.00'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 1.234
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.23)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    assert var_4 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = 20
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)
    assert var_4 == 15

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    assert var_4 == 9

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 42.5



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_child_type_error_and_other_errors. Retrieved 1/15 statements.
# Partially parsed test_validate_with_multiple_candidate_errors. Retrieved 1/24 statements.
# Partially parsed test_validate_with_single_candidate_error. Retrieved 1/22 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = e.messages()[0].code
    assert var_10 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 123
    var_8 = var_6.validate(var_7)
    assert var_8 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'hello'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 3.14
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = e.messages()[0].code
    assert var_10 == 'union'

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True
    var_2 = e.messages()[0].code
    assert var_2 == 'custom'

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True
    var_2 = e.messages()[0].code
    assert var_2 == 'union'

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True
    var_2 = e.messages()[0].code
    assert var_2 == 'custom1'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.String(**var_6)
    var_8 = [var_3, var_7]
    var_9 = {}
    var_10 = module_0.Union(var_8, **var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None



# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Partially parsed test_array_constructor_with_default_value. Retrieved 5/6 statements.


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
    var_7 = var_1.allow_null
    assert var_7 is False
    var_8 = var_1.read_only
    assert var_8 is False
    var_9 = var_1.title
    assert var_9 == ''
    var_10 = var_1.description
    assert var_10 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = var_2.items
    var_4 = bool(var_2.items == var_0)
    assert var_4 is True
    var_5 = var_2.additional_items
    assert var_5 is False
    var_6 = var_2.min_items
    assert var_6 is None
    var_7 = var_2.max_items
    assert var_7 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Array(var_2, **var_3)
    var_5 = var_4.items
    var_6 = bool(var_4.items == [var_0, var_1])
    assert var_6 is True
    var_7 = var_4.additional_items
    assert var_7 is False
    var_8 = var_4.min_items
    assert var_8 == 2
    var_9 = var_4.max_items
    assert var_9 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(additional_items=var_0, **var_1)
    var_3 = var_2.additional_items
    var_4 = bool(var_2.additional_items == var_0)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Array(additional_items=var_0, **var_1)
    var_3 = var_2.additional_items
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 == 5
    var_4 = var_2.max_items
    assert var_4 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 is None
    var_4 = var_2.max_items
    assert var_4 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = 7
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 == 7
    var_4 = var_2.max_items
    assert var_4 == 7

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = var_2.unique_items
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Array(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Test Title'
    var_7 = var_5.description
    assert var_7 == 'Test Description'

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
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1]
    var_4 = {}
    var_5 = module_0.Array(var_3, var_2, **var_4)
    var_6 = var_5.items
    var_7 = bool(var_5.items == [var_0, var_1])
    assert var_7 is True
    var_8 = var_5.additional_items
    var_9 = bool(var_5.additional_items == var_2)
    assert var_9 is True
    var_10 = var_5.min_items
    assert var_10 == 2
    var_11 = var_5.max_items
    assert var_11 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = {}
    var_5 = module_0.Array(var_2, min_items=var_3, **var_4)
    var_6 = var_5.items
    var_7 = bool(var_5.items == [var_0, var_1])
    assert var_7 is True
    var_8 = var_5.min_items
    assert var_8 == 5
    var_9 = var_5.max_items
    assert var_9 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 10
    var_4 = {}
    var_5 = module_0.Array(var_2, max_items=var_3, **var_4)
    var_6 = var_5.items
    var_7 = bool(var_5.items == [var_0, var_1])
    assert var_7 is True
    var_8 = var_5.min_items
    assert var_8 == 2
    var_9 = var_5.max_items
    assert var_9 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 7
    var_4 = {}
    var_5 = module_0.Array(var_2, exact_items=var_3, **var_4)
    var_6 = var_5.items
    var_7 = bool(var_5.items == [var_0, var_1])
    assert var_7 is True
    var_8 = var_5.min_items
    assert var_8 == 7
    var_9 = var_5.max_items
    assert var_9 == 7

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'default'
    var_5 = {var_4: var_3}
    var_6 = module_0.Array(**var_5)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_true_for_multiple_messages. Retrieved 5/25 statements.
# Partially parsed test_predicate_at_line_17_evaluates_true_for_single_message_with_non_type_code. Retrieved 5/25 statements.
# Partially parsed test_predicate_at_line_17_evaluates_true_for_single_message_with_type_code_and_index. Retrieved 5/25 statements.
# Partially parsed test_predicate_at_line_17_evaluates_false_for_single_message_with_type_code_and_no_index. Retrieved 5/25 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = 1
    var_3 = 0
    var_4 = 'type'

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = 1
    var_3 = 0
    var_4 = 'type'

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = 1
    var_3 = 0
    var_4 = 'type'

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = 1
    var_3 = 0
    var_4 = 'type'



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 6
    var_4 = var_2.validate(var_3)
    assert var_4 == 6



# Parsed testcases at query #9
#--------------------------




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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

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
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

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
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_or_with_two_simple_fields. Retrieved 5/6 statements.
# Partially parsed test_or_with_field_and_union. Retrieved 7/8 statements.
# Partially parsed test_or_with_union_and_field. Retrieved 7/8 statements.
# Partially parsed test_or_with_two_unions. Retrieved 9/10 statements.
# Partially parsed test_or_chaining. Retrieved 7/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = var_0 | var_1
    var_3 = var_2.any_of
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_2.any_of[0]
    var_6 = bool(var_2.any_of[0] is var_0)
    assert var_6 is True
    var_7 = var_2.any_of[1]
    var_8 = bool(var_2.any_of[1] is var_1)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = var_0 | var_1
    var_4 = var_3 | var_2
    var_5 = var_4.any_of
    var_6 = len(var_5)
    assert var_6 == 3
    var_7 = var_4.any_of[0]
    var_8 = bool(var_4.any_of[0] is var_0)
    assert var_8 is True
    var_9 = var_4.any_of[1]
    var_10 = bool(var_4.any_of[1] is var_1)
    assert var_10 is True
    var_11 = var_4.any_of[2]
    var_12 = bool(var_4.any_of[2] is var_2)
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = var_0 | var_1
    var_4 = var_2 | var_3
    var_5 = var_4.any_of
    var_6 = len(var_5)
    assert var_6 == 3
    var_7 = var_4.any_of[0]
    var_8 = bool(var_4.any_of[0] is var_2)
    assert var_8 is True
    var_9 = var_4.any_of[1]
    var_10 = bool(var_4.any_of[1] is var_0)
    assert var_10 is True
    var_11 = var_4.any_of[2]
    var_12 = bool(var_4.any_of[2] is var_1)
    assert var_12 is True

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
    var_8 = len(var_7)
    assert var_8 == 4
    var_9 = var_6.any_of[0]
    var_10 = bool(var_6.any_of[0] is var_0)
    assert var_10 is True
    var_11 = var_6.any_of[1]
    var_12 = bool(var_6.any_of[1] is var_1)
    assert var_12 is True
    var_13 = var_6.any_of[2]
    var_14 = bool(var_6.any_of[2] is var_2)
    assert var_14 is True
    var_15 = var_6.any_of[3]
    var_16 = bool(var_6.any_of[3] is var_3)
    assert var_16 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = var_0 | var_1
    var_4 = var_3 | var_2
    var_5 = var_4.any_of
    var_6 = len(var_5)
    assert var_6 == 3
    var_7 = var_4.any_of[0]
    var_8 = bool(var_4.any_of[0] is var_0)
    assert var_8 is True
    var_9 = var_4.any_of[1]
    var_10 = bool(var_4.any_of[1] is var_1)
    assert var_10 is True
    var_11 = var_4.any_of[2]
    var_12 = bool(var_4.any_of[2] is var_2)
    assert var_12 is True



# Parsed testcases at query #11
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
    var_7 = True
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None

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
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

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
    var_11 = 'c'
    var_12 = var_10.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

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
    var_10 = module_0.Choice(choices=var_6, coerce_types=var_7, **var_9)
    var_11 = ''
    var_12 = var_10.validate(var_11)
    assert var_12 is None

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
    var_8 = True
    var_9 = 'allow_null'
    var_10 = {var_9: var_7}
    var_11 = module_0.Choice(choices=var_6, coerce_types=var_8, **var_10)
    var_12 = ''
    var_13 = var_11.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True

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
    var_8 = False
    var_9 = 'allow_null'
    var_10 = {var_9: var_7}
    var_11 = module_0.Choice(choices=var_6, coerce_types=var_8, **var_10)
    var_12 = ''
    var_13 = var_11.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'Display 1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'Display 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.validate(var_0)
    assert var_11 == 'key1'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = var_6.validate(var_0)
    assert var_7 == 'x'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'Yes'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'No'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'allow_null'
    var_8 = {var_7: var_3}
    var_9 = module_0.Choice(choices=var_6, **var_8)
    var_10 = var_9.validate(var_0)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'Yes'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'No'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'allow_null'
    var_8 = {var_7: var_3}
    var_9 = module_0.Choice(choices=var_6, **var_8)
    var_10 = var_9.validate(var_3)
    assert var_10 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'Two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.validate(var_0)
    assert var_11 == 1

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1.5
    var_1 = 'One and half'
    var_2 = (var_0, var_1)
    var_3 = 2.5
    var_4 = 'Two and half'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.validate(var_0)
    var_12 = bool(var_11 == 1.5)
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'List 1'
    var_4 = (var_2, var_3)
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = 'List 2'
    var_9 = (var_7, var_8)
    var_10 = [var_4, var_9]
    var_11 = False
    var_12 = 'allow_null'
    var_13 = {var_12: var_11}
    var_14 = module_0.Choice(choices=var_10, **var_13)
    var_15 = [var_0, var_1]
    var_16 = var_14.validate(var_15)
    var_17 = bool(var_16 == [1, 2])
    assert var_17 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'Dict A'
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = 'Dict B'
    var_9 = (var_7, var_8)
    var_10 = [var_4, var_9]
    var_11 = False
    var_12 = 'allow_null'
    var_13 = {var_12: var_11}
    var_14 = module_0.Choice(choices=var_10, **var_13)
    var_15 = {var_0: var_1}
    var_16 = var_14.validate(var_15)
    var_17 = bool(var_16 == {'a': 1})
    assert var_17 is True



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Choice(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = var_1.allow_null
    assert var_4 is False
    var_5 = var_1.read_only
    assert var_5 is False
    var_6 = var_1.choices
    var_7 = bool(var_1.choices == [])
    assert var_7 is True
    var_8 = var_1.coerce_types
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = True
    var_3 = 'a'
    var_4 = 'A'
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = 'B'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = False
    var_11 = 'title'
    var_12 = 'description'
    var_13 = 'allow_null'
    var_14 = 'read_only'
    var_15 = {var_11: var_0, var_12: var_1, var_13: var_2, var_14: var_2}
    var_16 = module_0.Choice(choices=var_9, coerce_types=var_10, **var_15)
    var_17 = var_16.title
    assert var_17 == 'Test Title'
    var_18 = var_16.description
    assert var_18 == 'Test Description'
    var_19 = var_16.allow_null
    assert var_19 is True
    var_20 = var_16.read_only
    assert var_20 is True
    var_21 = var_16.choices
    var_22 = bool(var_16.choices == [('a', 'A'), ('b', 'B')])
    assert var_22 is True
    var_23 = var_16.coerce_types
    assert var_23 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.choices
    var_6 = bool(var_4.choices == [('option1', 'option1'), ('option2', 'option2')])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.choices
    var_10 = bool(var_8.choices == [('key1', 'value1'), ('key2', 'value2')])
    assert var_10 is True

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Choice(coerce_types=var_0, **var_1)
    var_3 = var_2.coerce_types
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Choice(coerce_types=var_0, **var_1)
    var_3 = var_2.coerce_types
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Choice(**var_0)
    var_2 = 'default'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Choice(**var_2)
    var_4 = var_3.default
    assert var_4 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Choice(**var_2)
    var_4 = var_3.default
    assert var_4 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'explicit_default'
    var_2 = 'allow_null'
    var_3 = 'default'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Choice(**var_4)
    var_6 = var_5.default
    assert var_6 == 'explicit_default'



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(var_5 == {'key': 'value'})
    assert var_6 is True

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
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'

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
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'invalid_key'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'longkey'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'invalid_property'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'

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
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_properties'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'max_properties'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'required'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 123
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'key': 123})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 456
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = 'key'
    var_5 = {var_4: var_3}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'key': 456})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'not an integer'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = '^a'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'abc'
    var_7 = 'def'
    var_8 = 123
    var_9 = 456
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = bool(var_11 == {'abc': 123})
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'invalid_property'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'key'
    var_5 = 123
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'key': 123})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'key'
    var_5 = 'not an integer'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Object(required=var_2, **var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 2
    var_9 = {msg.code for msg in e.messages()}
    var_10 = bool(var_9 == {'required'})
    assert var_10 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_serialize_with_single_item_field. Retrieved 4/9 statements.
# Partially parsed test_serialize_with_list_of_item_fields. Retrieved 3/11 statements.
# Partially parsed test_serialize_with_list_of_item_fields_and_longer_obj. Retrieved 4/12 statements.
# Partially parsed test_serialize_with_list_of_item_fields_and_shorter_obj. Retrieved 2/10 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.serialize(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.serialize(var_4)
    assert var_5 is None

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

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = [var_1, var_0, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_with_single_item_validator. Retrieved 7/8 statements.
# Partially parsed test_validate_with_list_item_validators. Retrieved 8/10 statements.
# Partially parsed test_validate_with_list_item_validators_and_additional_items_false. Retrieved 12/15 statements.
# Partially parsed test_validate_with_list_item_validators_and_additional_items_field. Retrieved 11/14 statements.
# Partially parsed test_validate_with_item_validation_error. Retrieved 12/16 statements.
# Partially parsed test_validate_with_list_item_validators_and_validation_errors. Retrieved 15/20 statements.
# Partially parsed test_validate_with_additional_items_field_validation_error. Retrieved 15/20 statements.


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
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_items'

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
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'max_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'exact_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3, var_0]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [1, 2])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 2
    var_2 = {}
    var_3 = module_0.Array(var_0, **var_2)
    var_4 = 1
    var_5 = 3
    var_6 = [var_4, var_1, var_5]
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == [2, 4, 6])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 1
    var_2 = module_0.Field()
    var_3 = 2
    var_4 = [var_0, var_2]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = [var_1, var_3]
    var_8 = var_6.validate(var_7)
    var_9 = bool(var_8 == [2, 4])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 1
    var_2 = module_0.Field()
    var_3 = 2
    var_4 = [var_0, var_2]
    var_5 = False
    var_6 = {}
    var_7 = module_0.Array(var_4, var_5, **var_6)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 1
    var_15 = e.messages()[0].code
    assert var_15 == 'additional_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 1
    var_2 = module_0.Field()
    var_3 = 2
    var_4 = module_0.Field()
    var_5 = [var_0, var_2]
    var_6 = {}
    var_7 = module_0.Array(var_5, var_4, **var_6)
    var_8 = 3
    var_9 = 4
    var_10 = [var_1, var_3, var_8, var_9]
    var_11 = var_7.validate(var_10)
    var_12 = bool(var_11 == [2, 4, 2, 3])
    assert var_12 is True

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
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'unique_items'

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
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'Invalid'
    var_3 = 'invalid'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = {}
    var_8 = module_0.Array(var_0, **var_7)
    var_9 = 1
    var_10 = 2
    var_11 = [var_9, var_10]
    var_12 = var_8.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 2
    var_15 = e.messages()[0].code
    assert var_15 == 'invalid'
    var_16 = e.messages()[0].index
    var_17 = bool(e.messages()[0].index == [0])
    assert var_17 is True
    var_18 = e.messages()[1].code
    assert var_18 == 'invalid'
    var_19 = e.messages()[1].index
    var_20 = bool(e.messages()[1].index == [1])
    assert var_20 is True

import typesystem.fields as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'Error1'
    var_3 = 'error1'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = module_0.Field()
    var_8 = 2
    var_9 = [var_0, var_7]
    var_10 = {}
    var_11 = module_0.Array(var_9, **var_10)
    var_12 = 1
    var_13 = 2
    var_14 = [var_12, var_13]
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True
    var_17 = len(e.messages())
    assert var_17 == 1
    var_18 = e.messages()[0].code
    assert var_18 == 'error1'
    var_19 = e.messages()[0].index
    var_20 = bool(e.messages()[0].index == [0])
    assert var_20 is True

import typesystem.fields as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 1
    var_2 = module_0.Field()
    var_3 = ()
    var_4 = 'Additional error'
    var_5 = 'additional_error'
    var_6 = module_1.Message(text=var_4, code=var_5)
    var_7 = [var_6]
    var_8 = module_1.ValidationError(messages=var_7)
    var_9 = [var_0]
    var_10 = {}
    var_11 = module_0.Array(var_9, var_2, **var_10)
    var_12 = 1
    var_13 = 2
    var_14 = [var_12, var_13]
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True
    var_17 = len(e.messages())
    assert var_17 == 1
    var_18 = e.messages()[0].code
    assert var_18 == 'additional_error'
    var_19 = e.messages()[0].index
    var_20 = bool(e.messages()[0].index == [1])
    assert var_20 is True

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = 2
    var_6 = [var_0, var_5]
    var_7 = {var_3: var_0}
    var_8 = [var_0, var_5]
    var_9 = [var_4, var_6, var_7, var_8]
    var_10 = var_2.validate(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = var_10[0]
    var_13 = bool(var_10[0] == {'a': 1})
    assert var_13 is True
    var_14 = var_10[1]
    var_15 = bool(var_10[1] == [1, 2])
    assert var_15 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_assert_all_choices_have_length_two. Retrieved 12/14 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.choices
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = var_8.choices
    var_12 = 2



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------




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
    var_8 = var_0.serialize(var_3)
    var_9 = var_0.serialize(var_4)
    var_10 = var_0.serialize(var_5)
    var_11 = [var_8, var_9, var_10]
    var_12 = bool(var_7 == var_11)
    assert var_12 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_native_type_with_format. Retrieved 3/6 statements.


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
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_3 = '  test  '
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  test  '
    var_4 = var_2.validate(var_3)
    assert var_4 == '  test  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

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
    var_3 = module_0.String(trim_whitespace=var_0, coerce_types=var_0, **var_2)
    var_4 = '   '
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'abc'

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcde'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'abcde'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcdef'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'abc'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc123'
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
    var_6 = 'Must be a valid email.'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = (var_0, var_1)
    var_3 = 'c'
    var_4 = 'd'
    var_5 = (var_3, var_4)
    var_6 = 'e'
    var_7 = 'f'
    var_8 = (var_6, var_7)
    var_9 = [var_2, var_5, var_8]
    var_10 = {}
    var_11 = module_0.Choice(choices=var_9, **var_10)
    var_12 = var_11.choices
    var_13 = len(var_12)
    assert var_13 == 3



# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------




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
    var_14 = 'default'
    var_15 = hasattr(var_1, var_14)
    var_16 = bool(not var_15)
    assert var_16 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = var_2.allow_blank
    assert var_3 is True
    var_4 = var_2.default
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Name'
    var_1 = 'Enter your name'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.String(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Name'
    var_7 = var_5.description
    assert var_7 == 'Enter your name'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = var_2.max_length
    assert var_3 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = var_2.min_length
    assert var_3 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = var_2.pattern
    assert var_3 == '^[a-z]+$'
    var_4 = var_2.pattern_regex
    var_5 = bool(var_2.pattern_regex is not None)
    assert var_5 is True

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
    var_6 = bool(var_3.pattern_regex is var_1)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = var_2.format
    assert var_3 == 'email'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(coerce_types=var_0, **var_1)
    var_3 = var_2.coerce_types
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, **var_3)
    var_5 = var_4.allow_blank
    assert var_5 is True
    var_6 = var_4.default
    assert var_6 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'allow_null'
    var_3 = 'default'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.String(**var_4)
    var_6 = var_5.allow_null
    assert var_6 is True
    var_7 = var_5.default
    assert var_7 is None

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[A-Z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = 'Title'
    var_3 = 'Description'
    var_4 = 'DEFAULT'
    var_5 = True
    var_6 = False
    var_7 = 100
    var_8 = 'uppercase'
    var_9 = 'title'
    var_10 = 'description'
    var_11 = 'default'
    var_12 = 'allow_null'
    var_13 = 'read_only'
    var_14 = {var_9: var_2, var_10: var_3, var_11: var_4, var_12: var_5, var_13: var_5}
    var_15 = module_1.String(allow_blank=var_5, trim_whitespace=var_6, max_length=var_7, min_length=var_5, pattern=var_1, format=var_8, coerce_types=var_6, **var_14)
    var_16 = var_15.title
    assert var_16 == 'Title'
    var_17 = var_15.description
    assert var_17 == 'Description'
    var_18 = var_15.default
    assert var_18 == 'DEFAULT'
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
    assert var_24 == 1
    var_25 = var_15.pattern
    assert var_25 == '^[A-Z]+$'
    var_26 = var_15.pattern_regex
    var_27 = bool(var_15.pattern_regex is var_1)
    assert var_27 is True
    var_28 = var_15.format
    assert var_28 == 'uppercase'
    var_29 = var_15.coerce_types
    assert var_29 is False



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------

# Partially parsed test_union_validate_candidate_errors_condition_true. Retrieved 8/19 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'not_an_integer'
    var_8 = 1
    var_9 = 0
    var_10 = 'type'



# Parsed testcases at query #26
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = False
    var_3 = var_1.validate(var_2)
    assert var_3 is False

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
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

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
    var_2 = []
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is False



# Parsed testcases at query #28
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
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'invalid_key'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = ''
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'invalid_property'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'

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
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_properties'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'max_properties'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'required'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'key': 'default_value'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'key'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0].code
    assert var_12 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^a.*'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = 'abc'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'abc': 'value'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^a.*'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = 'xyz'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'xyz': 'value'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'invalid_property'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = [var_0]
    var_2 = False
    var_3 = {}
    var_4 = module_0.Object(additional_properties=var_2, required=var_1, **var_3)
    var_5 = 'extra'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 2
    var_11 = {msg.code for msg in e.messages()}
    var_12 = bool(var_11 == {'required', 'invalid_property'})
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = False
    var_4 = {}
    var_5 = module_0.Object(properties=var_2, additional_properties=var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_1: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'key': 'value'})
    assert var_9 is True



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True



# Parsed testcases at query #30
#--------------------------






# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_integer_type_with_float_non_integer. Retrieved 1/4 statements.


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
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
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
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '123.45'
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 123.45)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 123.45
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
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
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
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10

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
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 11
    var_4 = var_2.validate(var_3)
    assert var_4 == 11

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
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10

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
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    assert var_4 == 9

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 12
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)
    assert var_4 == 15

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
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.5)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 1.234
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.23)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 1.235
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.24)
    assert var_5 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_array_constructor_with_default_value. Retrieved 5/7 statements.
# Failed to parse test_array_constructor_with_callable_default.


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
    var_7 = var_1.allow_null
    assert var_7 is False
    var_8 = var_1.read_only
    assert var_8 is False
    var_9 = var_1.title
    assert var_9 == ''
    var_10 = var_1.description
    assert var_10 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = var_2.items
    var_4 = bool(var_2.items == var_0)
    assert var_4 is True
    var_5 = var_2.additional_items
    assert var_5 is False
    var_6 = var_2.min_items
    assert var_6 is None
    var_7 = var_2.max_items
    assert var_7 is None
    var_8 = var_2.unique_items
    assert var_8 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Array(var_2, **var_3)
    var_5 = var_4.items
    var_6 = bool(var_4.items == [var_0, var_1])
    assert var_6 is True
    var_7 = var_4.additional_items
    assert var_7 is False
    var_8 = var_4.min_items
    assert var_8 == 2
    var_9 = var_4.max_items
    assert var_9 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = {}
    var_3 = module_0.Array(var_0, var_1, **var_2)
    var_4 = var_3.items
    var_5 = bool(var_3.items == var_0)
    assert var_5 is True
    var_6 = var_3.additional_items
    var_7 = bool(var_3.additional_items == var_1)
    assert var_7 is True
    var_8 = var_3.min_items
    assert var_8 is None
    var_9 = var_3.max_items
    assert var_9 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = False
    var_2 = {}
    var_3 = module_0.Array(var_0, var_1, **var_2)
    var_4 = var_3.items
    var_5 = bool(var_3.items == var_0)
    assert var_5 is True
    var_6 = var_3.additional_items
    assert var_6 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 == 3
    var_4 = var_2.max_items
    assert var_4 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 is None
    var_4 = var_2.max_items
    assert var_4 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = 4
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 == 4
    var_4 = var_2.max_items
    assert var_4 == 4

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = var_2.unique_items
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Array'
    var_1 = 'An array for testing'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Array(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Test Array'
    var_7 = var_5.description
    assert var_7 == 'An array for testing'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = var_5.min_items
    assert var_6 == 2
    var_7 = var_5.max_items
    assert var_7 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1]
    var_4 = {}
    var_5 = module_0.Array(var_3, var_2, **var_4)
    var_6 = var_5.min_items
    assert var_6 == 2
    var_7 = var_5.max_items
    assert var_7 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 7
    var_1 = 1
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(min_items=var_1, max_items=var_2, exact_items=var_0, **var_3)
    var_5 = var_4.min_items
    assert var_5 == 7
    var_6 = var_4.max_items
    assert var_6 == 7

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'default'
    var_5 = {var_4: var_3}
    var_6 = module_0.Array(**var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_returns_validated_value_when_one_child_validates. Retrieved 6/8 statements.
# Partially parsed test_validate_raises_child_error_when_one_child_has_non_type_error. Retrieved 7/13 statements.
# Partially parsed test_validate_raises_union_error_when_no_child_validates_and_multiple_candidate_errors. Retrieved 9/20 statements.
# Partially parsed test_validate_raises_child_error_when_only_one_child_has_non_type_error. Retrieved 9/20 statements.
# Partially parsed test_validate_raises_union_error_when_all_children_have_type_errors_without_index. Retrieved 8/19 statements.
# Partially parsed test_validate_raises_child_error_when_child_type_error_has_index. Retrieved 9/15 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
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
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = [var_1]
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_0.Union(var_2, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = e.messages()[0].code
    assert var_9 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = [var_0]
    var_3 = {}
    var_4 = module_0.Union(var_2, **var_3)
    var_5 = 'test'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'custom'
    var_2 = None
    var_3 = [var_0]
    var_4 = {}
    var_5 = module_0.Union(var_3, **var_4)
    var_6 = 'test'
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'custom1'
    var_2 = None
    var_3 = module_0.Field()
    var_4 = 'custom2'
    var_5 = [var_0, var_3]
    var_6 = {}
    var_7 = module_0.Union(var_5, **var_6)
    var_8 = 'test'
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = e.messages()[0].code
    assert var_11 == 'union'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'type'
    var_2 = None
    var_3 = module_0.Field()
    var_4 = 'custom'
    var_5 = [var_0, var_3]
    var_6 = {}
    var_7 = module_0.Union(var_5, **var_6)
    var_8 = 'test'
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'type'
    var_2 = None
    var_3 = module_0.Field()
    var_4 = [var_0, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'test'
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = e.messages()[0].code
    assert var_10 == 'union'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'type'
    var_2 = 0
    var_3 = [var_2]
    var_4 = None
    var_5 = [var_0]
    var_6 = {}
    var_7 = module_0.Union(var_5, **var_6)
    var_8 = 'test'
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = False
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = var_6.allow_null
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_default'
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 'test_default'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'callable_result'
    var_1 = lambda : var_0
    var_2 = module_0.Field(default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 'callable_result'

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serialize_with_single_item_field. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_list_of_item_fields. Retrieved 5/12 statements.
# Partially parsed test_serialize_with_list_of_item_fields_and_shorter_obj. Retrieved 4/11 statements.
# Partially parsed test_serialize_with_list_of_item_fields_and_longer_obj. Retrieved 6/13 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.serialize(var_4)
    assert var_5 is None

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
    var_1 = module_0.Array(**var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.serialize(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = 3
    var_3 = [var_1, var_0, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5
    var_3 = 10
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5
    var_3 = 10
    var_4 = 15
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #5
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
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].text
    assert var_6 == 'May not be null.'
    var_7 = e.messages()[0].code
    assert var_7 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].text
    assert var_6 == 'Must be an array.'
    var_7 = e.messages()[0].code
    assert var_7 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].text
    assert var_7 == 'Must not be empty.'
    var_8 = e.messages()[0].code
    assert var_8 == 'empty'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].text
    assert var_9 == 'Must have at least 3 items.'
    var_10 = e.messages()[0].code
    assert var_10 == 'min_items'

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
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].text
    assert var_10 == 'Must have no more than 2 items.'
    var_11 = e.messages()[0].code
    assert var_11 == 'max_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].text
    assert var_8 == 'Must have 2 items.'
    var_9 = e.messages()[0].code
    assert var_9 == 'exact_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].text
    assert var_10 == 'Must have 2 items.'
    var_11 = e.messages()[0].code
    assert var_11 == 'exact_items'

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
    var_5 = 'invalid'
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].text
    assert var_11 == 'Must be a valid integer.'
    var_12 = e.messages()[0].index
    var_13 = bool(e.messages()[0].index == [1])
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
    var_8 = 'hello'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == [1, 'hello'])
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
    var_7 = 'invalid'
    var_8 = 123
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0].text
    assert var_13 == 'Must be a valid integer.'
    var_14 = e.messages()[0].index
    var_15 = bool(e.messages()[0].index == [0])
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
    var_9 = 'hello'
    var_10 = 'extra'
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 1
    var_15 = e.messages()[0].text
    assert var_15 == 'May not contain additional items.'
    var_16 = e.messages()[0].code
    assert var_16 == 'additional_items'

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
    var_10 = 'hello'
    var_11 = True
    var_12 = False
    var_13 = [var_9, var_10, var_11, var_12]
    var_14 = var_8.validate(var_13)
    var_15 = bool(var_14 == [1, 'hello', True, False])
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
    var_10 = 'hello'
    var_11 = 'not_bool'
    var_12 = [var_9, var_10, var_11]
    var_13 = var_8.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True
    var_15 = len(e.messages())
    assert var_15 == 1
    var_16 = e.messages()[0].text
    assert var_16 == 'Must be a valid boolean.'
    var_17 = e.messages()[0].index
    var_18 = bool(e.messages()[0].index == [2])
    assert var_18 is True

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
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].text
    assert var_9 == 'Items must be unique.'
    var_10 = e.messages()[0].code
    assert var_10 == 'unique_items'
    var_11 = e.messages()[0].key
    assert var_11 == 2

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = [var_0, var_3]
    var_5 = [var_0, var_3]
    var_6 = [var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(var_7 == [[1, 2], [1, 2]])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = {var_3: var_0}
    var_6 = [var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(var_7 == [{'a': 1}, {'a': 1}])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 2
    var_3 = True
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, unique_items=var_3, **var_4)
    var_6 = 1
    var_7 = 'invalid'
    var_8 = [var_6, var_7, var_6]
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = len(e.messages())
    assert var_11 == 2
    var_12 = {msg.code for msg in e.messages()}
    var_13 = bool(var_12 == {'invalid', 'unique_items'})
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 1
    var_3 = 'hello'
    var_4 = True
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)
    var_7 = bool(var_6 == [1, 'hello', True])
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
    var_5 = 'invalid'
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].index
    var_12 = bool(e.messages()[0].index == [1])
    assert var_12 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_condition_true_when_multiple_messages. Retrieved 1/12 statements.
# Partially parsed test_condition_true_when_single_message_not_type. Retrieved 1/12 statements.
# Partially parsed test_condition_true_when_single_type_message_with_index. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serialize_with_decimal. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_integer_decimal. Retrieved 2/4 statements.


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
    var_2 = '7'



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_property_names_invalid. Retrieved 12/16 statements.
# Partially parsed test_validate_properties_valid. Retrieved 7/8 statements.
# Partially parsed test_validate_properties_invalid. Retrieved 15/17 statements.
# Partially parsed test_validate_pattern_properties_valid. Retrieved 8/9 statements.
# Partially parsed test_validate_pattern_properties_invalid. Retrieved 15/17 statements.
# Partially parsed test_validate_additional_properties_field_valid. Retrieved 6/7 statements.
# Partially parsed test_validate_additional_properties_field_invalid. Retrieved 13/15 statements.


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
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'invalid_key'

import typesystem.fields as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'Invalid'
    var_3 = 'custom'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = {}
    var_8 = module_0.Object(property_names=var_0, **var_7)
    var_9 = 'invalid_key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 1
    var_15 = e.messages()[0].code
    assert var_15 == 'invalid_property'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'

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
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_properties'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'max_properties'

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
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'required'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'key': 'default_value'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'Error'
    var_3 = 'custom'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = (var_1, var_6)
    var_8 = 'key'
    var_9 = {var_8: var_0}
    var_10 = {}
    var_11 = module_0.Object(properties=var_9, **var_10)
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True
    var_17 = len(e.messages())
    assert var_17 == 1
    var_18 = e.messages()[0].code
    assert var_18 == 'custom'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^a.*'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = 'abc'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'abc': 'value'})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'Error'
    var_3 = 'custom'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = (var_1, var_6)
    var_8 = '^a.*'
    var_9 = {var_8: var_0}
    var_10 = {}
    var_11 = module_0.Object(pattern_properties=var_9, **var_10)
    var_12 = 'abc'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True
    var_17 = len(e.messages())
    assert var_17 == 1
    var_18 = e.messages()[0].code
    assert var_18 == 'custom'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'invalid_property'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'Error'
    var_3 = 'custom'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = (var_1, var_6)
    var_8 = {}
    var_9 = module_0.Object(additional_properties=var_0, **var_8)
    var_10 = 'extra'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True
    var_15 = len(e.messages())
    assert var_15 == 1
    var_16 = e.messages()[0].code
    assert var_16 == 'custom'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = [var_0]
    var_2 = False
    var_3 = {}
    var_4 = module_0.Object(additional_properties=var_2, required=var_1, **var_3)
    var_5 = 'extra'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 2
    var_11 = {msg.code for msg in e.messages()}
    var_12 = bool(var_11 == {'required', 'invalid_property'})
    assert var_12 is True



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Array(var_1, var_0, **var_2)
    var_4 = var_3.max_items
    assert var_4 is None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_float_for_int_numeric_type. Retrieved 1/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

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
    var_4 = ''
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)
    assert var_3 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '123.45'
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 123.45)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 123.5
    var_1 = bool(False)
    assert var_1 is True

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
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
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
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10

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
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 11
    var_4 = var_2.validate(var_3)
    assert var_4 == 11

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
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10

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
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    assert var_4 == 9

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 12
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)
    assert var_4 == 15

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
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.5)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 1.234
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.23)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #12
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
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'helloworld'

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
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

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
    var_3 = module_0.String(trim_whitespace=var_0, coerce_types=var_0, **var_2)
    var_4 = '  '
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
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
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hey'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hey'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello world'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

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
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'hello123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

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
    var_6 = 'email'



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Field()
    var_2 = {}
    var_3 = module_0.Array(var_0, var_1, **var_2)
    var_4 = var_3.max_items
    assert var_4 is None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_union_validation_with_one_candidate_error. Retrieved 6/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'not_an_integer'
    var_8 = var_6.validate(var_7)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_with_numeric_type_int_and_float_integer. Retrieved 1/3 statements.
# Partially parsed test_validate_with_numeric_type_int_and_string_integer. Retrieved 1/3 statements.
# Partially parsed test_validate_with_numeric_type_float_and_integer. Retrieved 1/3 statements.
# Partially parsed test_validate_with_numeric_type_int_and_integer. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 5.0

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 5
    var_3 = var_1.validate(var_2)
    assert var_3 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 5.5
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 5.5)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '5.5'
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 5.5)
    assert var_4 is True

def test_case_0():
    var_0 = 5



# Parsed testcases at query #16
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 == ''



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_serialize_with_items_as_list. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_true_for_multiple_messages. Retrieved 4/22 statements.
# Partially parsed test_predicate_at_line_17_evaluates_true_for_non_type_code. Retrieved 4/22 statements.
# Partially parsed test_predicate_at_line_17_evaluates_true_for_index_present. Retrieved 4/22 statements.
# Partially parsed test_predicate_at_line_17_evaluates_false_for_single_type_no_index. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 0
    var_3 = 'type'

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 0
    var_3 = 'type'

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 0
    var_3 = 'type'

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 0
    var_3 = 'type'



# Parsed testcases at query #19
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
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'

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
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'exact_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_items'

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
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'max_items'

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
    var_5 = 'invalid'
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].index
    var_12 = bool(e.messages()[0].index == [1])
    assert var_12 is True

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
    var_8 = 'hello'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == [1, 'hello'])
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
    var_7 = 'invalid'
    var_8 = 123
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0].index
    var_14 = bool(e.messages()[0].index == [0])
    assert var_14 is True

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
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0].code
    assert var_12 == 'additional_items'

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
    var_8 = 'extra'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == [1, 'extra'])
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
    var_8 = 123
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(e.messages())
    assert var_12 == 1
    var_13 = e.messages()[0].index
    var_14 = bool(e.messages()[0].index == [1])
    assert var_14 is True

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
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'unique_items'

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = [var_0, var_3]
    var_5 = [var_0, var_3]
    var_6 = [var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(var_7 == [[1, 2], [1, 2]])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = {var_3: var_0}
    var_6 = [var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(var_7 == [{'a': 1}, {'a': 1}])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = True
    var_6 = {}
    var_7 = module_0.Array(var_4, unique_items=var_5, **var_6)
    var_8 = 'invalid'
    var_9 = 123
    var_10 = 'duplicate'
    var_11 = [var_8, var_9, var_10, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 3
    var_15 = {msg.code for msg in e.messages()}
    var_16 = bool(var_15 == {'type', 'type', 'unique_items'})
    assert var_16 is True



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 2.0
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 2.0)
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_serialize_when_items_is_list_and_obj_is_shorter. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = [var_0]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_string_constructor_allow_blank_sets_default. Retrieved 2/4 statements.
# Partially parsed test_string_constructor_allow_blank_with_explicit_default. Retrieved 3/4 statements.
# Partially parsed test_string_constructor_allow_null_sets_default. Retrieved 2/4 statements.
# Partially parsed test_string_constructor_allow_null_with_explicit_default. Retrieved 3/4 statements.


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
    var_0 = 'Name'
    var_1 = 'Full name'
    var_2 = True
    var_3 = False
    var_4 = 10
    var_5 = '^[a-z]+$'
    var_6 = 'email'
    var_7 = 'title'
    var_8 = 'description'
    var_9 = 'allow_null'
    var_10 = 'read_only'
    var_11 = {var_7: var_0, var_8: var_1, var_9: var_2, var_10: var_2}
    var_12 = module_0.String(allow_blank=var_2, trim_whitespace=var_3, max_length=var_4, min_length=var_2, pattern=var_5, format=var_6, coerce_types=var_3, **var_11)
    var_13 = var_12.title
    assert var_13 == 'Name'
    var_14 = var_12.description
    assert var_14 == 'Full name'
    var_15 = bool(var_12.allow_null)
    assert var_15 is True
    var_16 = bool(var_12.read_only)
    assert var_16 is True
    var_17 = bool(var_12.allow_blank)
    assert var_17 is True
    var_18 = bool(not var_12.trim_whitespace)
    assert var_18 is True
    var_19 = var_12.max_length
    assert var_19 == 10
    var_20 = var_12.min_length
    assert var_20 == 1
    var_21 = var_12.pattern
    assert var_21 == '^[a-z]+$'
    var_22 = var_12.pattern_regex
    var_23 = bool(var_12.pattern_regex is not None)
    assert var_23 is True
    var_24 = var_12.format
    assert var_24 == 'email'
    var_25 = bool(not var_12.coerce_types)
    assert var_25 is True

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
    var_6 = bool(var_3.pattern_regex is var_1)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'custom'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, **var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'not null'
    var_2 = 'allow_null'
    var_3 = 'default'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.String(**var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '10'
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '1'
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



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_array_constructor_with_default_value. Retrieved 5/7 statements.
# Partially parsed test_array_constructor_with_callable_default. Retrieved 5/7 statements.
# Partially parsed test_array_constructor_with_allow_null_and_default. Retrieved 4/6 statements.


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
    var_7 = var_1.allow_null
    assert var_7 is False
    var_8 = var_1.read_only
    assert var_8 is False
    var_9 = var_1.title
    assert var_9 == ''
    var_10 = var_1.description
    assert var_10 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = var_2.items
    var_4 = bool(var_2.items == var_0)
    assert var_4 is True
    var_5 = var_2.additional_items
    assert var_5 is False
    var_6 = var_2.min_items
    assert var_6 is None
    var_7 = var_2.max_items
    assert var_7 is None

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
    var_7 = var_4.additional_items
    assert var_7 is False
    var_8 = var_4.min_items
    assert var_8 == 2
    var_9 = var_4.max_items
    assert var_9 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = {}
    var_3 = module_0.Array(var_0, var_1, **var_2)
    var_4 = var_3.items
    var_5 = bool(var_3.items == var_0)
    assert var_5 is True
    var_6 = var_3.additional_items
    var_7 = bool(var_3.additional_items == var_1)
    assert var_7 is True
    var_8 = var_3.min_items
    assert var_8 is None
    var_9 = var_3.max_items
    assert var_9 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = False
    var_2 = {}
    var_3 = module_0.Array(var_0, var_1, **var_2)
    var_4 = var_3.items
    var_5 = bool(var_3.items == var_0)
    assert var_5 is True
    var_6 = var_3.additional_items
    assert var_6 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 == 3
    var_4 = var_2.max_items
    assert var_4 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 is None
    var_4 = var_2.max_items
    assert var_4 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = 4
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 == 4
    var_4 = var_2.max_items
    assert var_4 == 4

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = var_2.unique_items
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Array(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Test Title'
    var_7 = var_5.description
    assert var_7 == 'Test Description'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'default'
    var_5 = {var_4: var_3}
    var_6 = module_0.Array(**var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = lambda : var_2
    var_4 = 'default'
    var_5 = {var_4: var_3}
    var_6 = module_0.Array(**var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'allow_null'
    var_4 = 'default'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = module_0.Array(**var_5)
    var_7 = var_6.allow_null
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Array(var_3, **var_4)
    var_6 = var_5.min_items
    assert var_6 == 3
    var_7 = var_5.max_items
    assert var_7 == 3

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = var_5.min_items
    assert var_6 == 2
    var_7 = var_5.max_items
    assert var_7 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Field()
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = var_5.min_items
    assert var_6 == 2
    var_7 = var_5.max_items
    assert var_7 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(min_items=var_1, max_items=var_2, exact_items=var_0, **var_3)
    var_5 = var_4.min_items
    assert var_5 == 5
    var_6 = var_4.max_items
    assert var_6 == 5



# Parsed testcases at query #25
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = False
    var_3 = var_1.validate(var_2)
    assert var_3 is False

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
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

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
    var_2 = []
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_choice_constructor_inherits_field_default. Retrieved 2/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Choice(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = var_1.allow_null
    assert var_4 is False
    var_5 = var_1.read_only
    assert var_5 is False
    var_6 = var_1.choices
    var_7 = bool(var_1.choices == [])
    assert var_7 is True
    var_8 = var_1.coerce_types
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = True
    var_3 = 'a'
    var_4 = 'A'
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = 'B'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = False
    var_11 = 'title'
    var_12 = 'description'
    var_13 = 'allow_null'
    var_14 = 'read_only'
    var_15 = {var_11: var_0, var_12: var_1, var_13: var_2, var_14: var_2}
    var_16 = module_0.Choice(choices=var_9, coerce_types=var_10, **var_15)
    var_17 = var_16.title
    assert var_17 == 'Test Title'
    var_18 = var_16.description
    assert var_18 == 'Test Description'
    var_19 = var_16.allow_null
    assert var_19 is True
    var_20 = var_16.read_only
    assert var_20 is True
    var_21 = var_16.choices
    var_22 = bool(var_16.choices == [('a', 'A'), ('b', 'B')])
    assert var_22 is True
    var_23 = var_16.coerce_types
    assert var_23 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.choices
    var_6 = bool(var_4.choices == [('option1', 'option1'), ('option2', 'option2')])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.choices
    var_10 = bool(var_8.choices == [('key1', 'value1'), ('key2', 'value2')])
    assert var_10 is True

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
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Choice(**var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Choice(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True
    var_5 = 'default'
    var_6 = hasattr(var_3, var_5)
    var_7 = bool(not var_6)
    assert var_7 is True



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------






# Parsed testcases at query #29
#--------------------------






# Parsed testcases at query #30
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
    var_0 = 'key1'
    var_1 = 'Display 1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'Display 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_3)
    assert var_9 == 'key2'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.validate(var_1)
    assert var_5 == 'y'



# Parsed testcases at query #31
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Choice(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = var_1.allow_null
    assert var_4 is False
    var_5 = var_1.read_only
    assert var_5 is False
    var_6 = var_1.choices
    var_7 = bool(var_1.choices == [])
    assert var_7 is True
    var_8 = var_1.coerce_types
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = True
    var_3 = 'a'
    var_4 = 'A'
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = 'B'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = False
    var_11 = 'title'
    var_12 = 'description'
    var_13 = 'allow_null'
    var_14 = 'read_only'
    var_15 = {var_11: var_0, var_12: var_1, var_13: var_2, var_14: var_2}
    var_16 = module_0.Choice(choices=var_9, coerce_types=var_10, **var_15)
    var_17 = var_16.title
    assert var_17 == 'Test Title'
    var_18 = var_16.description
    assert var_18 == 'Test Description'
    var_19 = var_16.allow_null
    assert var_19 is True
    var_20 = var_16.read_only
    assert var_20 is True
    var_21 = var_16.choices
    var_22 = bool(var_16.choices == [('a', 'A'), ('b', 'B')])
    assert var_22 is True
    var_23 = var_16.coerce_types
    assert var_23 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.choices
    var_6 = bool(var_4.choices == [('option1', 'option1'), ('option2', 'option2')])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.choices
    var_10 = bool(var_8.choices == [('key1', 'value1'), ('key2', 'value2')])
    assert var_10 is True

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
    var_0 = {}
    var_1 = module_0.Choice(**var_0)
    var_2 = 'default'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Choice(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True
    var_5 = var_3.default
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Choice(**var_2)
    var_4 = var_3.default
    assert var_4 == 'default_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Choice(**var_0)
    var_2 = var_1.coerce_types
    assert var_2 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Choice(coerce_types=var_0, **var_1)
    var_3 = var_2.coerce_types
    assert var_3 is False



# Parsed testcases at query #32
#--------------------------






# Parsed testcases at query #33
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Field()
    var_2 = {}
    var_3 = module_0.Array(var_0, var_1, **var_2)
    var_4 = var_3.max_items
    assert var_4 is None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_validate_combined_errors. Retrieved 12/13 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(var_5 == {'key': 'value'})
    assert var_6 is True

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
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    assert var_5 == 1
    var_6 = e.messages()[0].code
    assert var_6 == 'type'

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
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'invalid_key'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'longkey'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'invalid_property'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'empty'

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
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'min_properties'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'max_properties'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'required'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'key'
    var_5 = {var_4: var_3}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'key': 'default_value'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'key': 'value'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'not an integer'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = '^a.*'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'abc'
    var_7 = 123
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'abc': 123})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = '^a.*'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'bcd'
    var_7 = 123
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'bcd': 123})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'extra': 'value'})
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'invalid_property'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 123
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'extra': 123})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 'not an integer'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'req'
    var_1 = [var_0]
    var_2 = 'prop'
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = {var_2: var_4}
    var_6 = False
    var_7 = {}
    var_8 = module_0.Object(properties=var_5, additional_properties=var_6, required=var_1, **var_7)
    var_9 = 'extra'
    var_10 = 'not int'
    var_11 = 'value'
    var_12 = {var_2: var_10, var_9: var_11}
    var_13 = var_8.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'required'
    var_16 = 'type'
    var_17 = 'invalid_property'



# Parsed testcases at query #35
#--------------------------






