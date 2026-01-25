####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_0 = False
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)

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
    var_0 = False
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)

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
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  hello  '
    var_3 = var_1.validate(var_2)
    assert var_3 == '  hello  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = 'hi'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello world'
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
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = 'test@example.com'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'test@example.com'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'world'
    var_1 = lambda : var_0
    var_2 = module_0.Field(default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 'world'

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



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = [var_1]
    var_3 = module_0.Union(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_0.Union(var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_0.Union(var_1)
    var_3 = 'test'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_0.Union(var_1)
    var_3 = 123
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 'test'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 123
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_0.Union(var_1)
    var_3 = 123
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #4
#--------------------------




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
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)
    assert var_3 == 123

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
    var_0 = 5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 7
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 0.7
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = module_0.Number(precision=var_0)
    var_2 = 1.234
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_array_constructor_with_items. Retrieved 3/4 statements.
# Partially parsed test_array_constructor_with_list_of_items. Retrieved 8/11 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Array(var_0)
    var_2 = var_1.items

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = var_3.items
    var_5 = var_3.items
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_3.items

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(additional_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = module_0.Array(min_items=var_0, max_items=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Array(exact_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = module_0.Array()



# Parsed testcases at query #6
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
    var_0 = module_0.Array()
    var_1 = []
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)

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
    var_0 = 2
    var_1 = module_0.Array(max_items=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 1
    var_3 = [var_2, var_2]
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'integer'
    var_1 = module_0.Field()
    var_2 = module_0.Array(var_1)
    var_3 = 'not an integer'
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'integer'
    var_1 = module_0.Field()
    var_2 = [var_1]
    var_3 = 'string'
    var_4 = module_0.Field()
    var_5 = module_0.Array(var_2, var_4)
    var_6 = 1
    var_7 = 'not a string'
    var_8 = [var_6, var_7]
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'integer'
    var_1 = module_0.Field()
    var_2 = [var_1]
    var_3 = False
    var_4 = module_0.Array(var_2, var_3)
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = var_4.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'integer'
    var_1 = module_0.Field()
    var_2 = [var_1]
    var_3 = True
    var_4 = module_0.Array(var_2, unique_items=var_3)
    var_5 = 1
    var_6 = [var_5, var_5]
    var_7 = var_4.validate(var_6)



# Parsed testcases at query #7
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
    var_0 = module_0.Boolean()
    var_1 = 'invalid'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 123
    var_2 = var_0.validate(var_1)

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
    var_0 = False
    var_1 = module_0.Boolean()
    var_2 = 'null'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Boolean()
    var_2 = 'none'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Boolean()
    var_2 = ''
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Boolean(coerce_types=var_0)
    var_2 = 'true'
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #8
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
    var_1 = 'not_a_dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 1
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

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
    var_0 = 'key'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = 'value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '.*'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pattern_properties_validation_success. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'a+'
    var_1 = 'a'
    var_2 = 'valid'
    var_3 = {var_1: var_2}



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'invalid_value'



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = 3
    var_2 = module_0.Array(var_0, exact_items=var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #16
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = True
    var_3 = False
    var_4 = 10
    var_5 = 5
    var_6 = 'test'
    var_7 = 'email'
    var_8 = module_0.String(allow_blank=var_2, trim_whitespace=var_3, max_length=var_4, min_length=var_5, pattern=var_6, format=var_7, coerce_types=var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)

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



# Parsed testcases at query #17
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
    var_0 = False
    var_1 = module_0.String()
    var_2 = None
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
    var_0 = module_0.String()
    var_1 = 123
    var_2 = var_0.validate(var_1)

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
    var_0 = True
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  test  '
    var_3 = var_1.validate(var_2)
    assert var_3 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  test  '
    var_3 = var_1.validate(var_2)
    assert var_3 == '  test  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'te\x00st'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = 'test'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'test'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = 'test123'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = 'test'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = 'invalid-email'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = 'test@example.com'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'test@example.com'



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 5
    var_3 = True
    var_4 = module_0.Array(var_1, var_3, var_2)



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = None
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'not a dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 1
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = 'key1'
    var_3 = 'value1'
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
    var_0 = 'key1'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'key1'
    var_3 = 'not an integer'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(trim_whitespace=var_0, coerce_types=var_0)
    var_2 = '   '
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.String(trim_whitespace=var_1, coerce_types=var_0)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_raises_integer_error_for_float_with_non_integer_value_when_numeric_type_is_int. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 3.14



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_integer_type_with_float. Retrieved 1/4 statements.


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
    var_0 = 1.5

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)
    assert var_3 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(minimum=var_0)
    var_2 = 9
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_minimum=var_0)
    var_2 = 10
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
    var_0 = 2
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 3
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 1.2
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = module_0.Number(precision=var_0)
    var_2 = 1.234
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 'Example'
    var_2 = 'Test Description'
    var_3 = True
    var_4 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = module_0.Const(var_0)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'invalid_value'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_pattern_properties_with_error. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'pattern'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_allow_blank_sets_default_to_empty_string_when_no_default. Retrieved 2/3 statements.
# Partially parsed test_allow_blank_does_not_override_existing_default. Retrieved 3/4 statements.
# Partially parsed test_allow_blank_false_does_not_set_default. Retrieved 4/5 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'existing'
    var_2 = module_0.String(allow_blank=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(allow_blank=var_0)
    var_2 = 'default'
    var_3 = hasattr(var_1, var_2)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_validate_integer. Retrieved 1/3 statements.
# Partially parsed test_validate_float_not_integer. Retrieved 1/4 statements.


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
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number()
    var_2 = ''
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = True
    var_2 = var_0.validate(var_1)

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 42.5

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = '42'
    var_2 = var_0.validate(var_1)
    assert var_2 == 42

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
    var_2 = 42.123
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(minimum=var_0)
    var_2 = 15
    var_3 = var_1.validate(var_2)
    assert var_3 == 15

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
    var_2 = 15
    var_3 = var_1.validate(var_2)
    assert var_3 == 15

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_minimum=var_0)
    var_2 = 10
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.Number(maximum=var_0)
    var_2 = 15
    var_3 = var_1.validate(var_2)
    assert var_3 == 15

import typesystem.fields as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.Number(maximum=var_0)
    var_2 = 25
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 15
    var_3 = var_1.validate(var_2)
    assert var_3 == 15

import typesystem.fields as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 20
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 15
    var_3 = var_1.validate(var_2)
    assert var_3 == 15

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 16
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #30
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = [var_0, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = 'valid'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'valid'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = 'invalid'
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = module_0.Field()
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = 'invalid'
    var_6 = var_4.validate(var_5)



# Parsed testcases at query #31
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
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)

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
    var_3 = [var_2]
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
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Array(var_1, var_2)
    var_4 = '2'
    var_5 = 3
    var_6 = [var_2, var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = module_0.String()
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 1
    var_5 = '2'
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = module_0.Array(var_1)
    var_3 = 1
    var_4 = '2'
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = True
    var_3 = module_0.Array(min_items=var_0, max_items=var_1, unique_items=var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = [var_4, var_5, var_6, var_7, var_4]
    var_9 = var_3.validate(var_8)



# Parsed testcases at query #32
#--------------------------




import typesystem.fields as module_0
import typesystem.unique as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = 2
    var_3 = [var_0, var_2, var_2]
    var_4 = [var_0, var_2]
    var_5 = module_1.Uniqueness(var_4)
    var_6 = var_1.validate(var_3)
    assert var_6 is None



# Parsed testcases at query #33
#--------------------------






# Parsed testcases at query #34
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = '^a.*'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'abc'
    var_5 = 'invalid'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #35
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Field()
    var_2 = module_0.Array(var_0, var_1)
    var_3 = 'max_items'
    var_4 = hasattr(var_2, var_3)



# Parsed testcases at query #36
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #37
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_validate_child_schema_with_error. Retrieved 7/11 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 5
    var_2 = module_0.String(max_length=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = 'too_long_string'
    var_6 = {var_0: var_5}



# Parsed testcases at query #39
#--------------------------






# Parsed testcases at query #40
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 123
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #41
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_allow_blank_without_default_sets_default_to_empty_string. Retrieved 2/3 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(properties=var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^test_'
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
    var_1 = 10
    var_2 = module_0.Object(min_properties=var_0, max_properties=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = [var_0, var_1]
    var_3 = module_0.Object(required=var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Test description'
    var_2 = True
    var_3 = module_0.Object()



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
    var_0 = False
    var_1 = module_0.String()
    var_2 = None
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
    var_0 = module_0.String()
    var_1 = 123
    var_2 = var_0.validate(var_1)

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
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'abcdef'
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
    var_2 = 'not-an-email'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  abc  '
    var_3 = var_1.validate(var_2)
    assert var_3 == 'abc'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.String(trim_whitespace=var_0)
    var_2 = '  abc  '
    var_3 = var_1.validate(var_2)
    assert var_3 == '  abc  '



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'not a field'
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = None
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = 'not an array'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(min_items=var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(max_items=var_0)
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(exact_items=var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = [var_0, var_0]
    var_3 = var_1.validate(var_2)

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
    var_3 = 'invalid'
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 2
    var_5 = 3
    var_6 = [var_2, var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = module_0.String()
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 1
    var_5 = 'valid'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = [var_0]
    var_2 = module_0.String()
    var_3 = module_0.Array(var_1, var_2)
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_integer_type_constraint. Retrieved 2/6 statements.


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
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)
    assert var_3 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '123'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 123
    var_2 = var_0.validate(var_1)
    assert var_2 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 123.45
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(minimum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 10
    var_3 = 9
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_minimum=var_0)
    var_2 = 11
    var_3 = var_1.validate(var_2)
    assert var_3 == 11
    var_4 = 10
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.Number(maximum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 20
    var_3 = 21
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 20
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 19
    var_3 = var_1.validate(var_2)
    assert var_3 == 19
    var_4 = 20
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 15
    var_3 = var_1.validate(var_2)
    assert var_3 == 15
    var_4 = 16
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 1.5
    var_3 = var_1.validate(var_2)
    var_4 = 1.6
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = module_0.Number(precision=var_0)
    var_2 = 123.45
    var_3 = var_1.validate(var_2)
    var_4 = 123.456
    var_5 = var_1.validate(var_4)

def test_case_0():
    var_0 = 123
    var_1 = 123.45



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_properties_keys_are_not_all_strings. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'key2'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_additional_properties_is_not_field_instance. Retrieved 3/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = var_1.additional_properties



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'default'
    var_2 = hasattr(var_0, var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Title'
    var_1 = 'Description'
    var_2 = 'default'
    var_3 = True
    var_4 = False
    var_5 = 100
    var_6 = 10
    var_7 = 'pattern'
    var_8 = 'format'
    var_9 = module_0.String(allow_blank=var_3, trim_whitespace=var_4, max_length=var_5, min_length=var_6, pattern=var_7, format=var_8, coerce_types=var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String(allow_blank=var_0)

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.String(pattern=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'invalid'
    var_3 = module_0.String(min_length=var_2)
    var_4 = 123
    var_5 = module_0.String(pattern=var_4)
    var_6 = 123
    var_7 = module_0.String(format=var_6)



# Parsed testcases at query #10
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
    var_8 = 'default'
    var_9 = hasattr(var_7, var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = 'Test Description'
    var_2 = 'a'
    var_3 = True
    var_4 = 'A'
    var_5 = (var_2, var_4)
    var_6 = 'b'
    var_7 = 'B'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = False
    var_11 = module_0.Choice(choices=var_9, coerce_types=var_10)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = module_0.Choice(choices=var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Choice(choices=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.Choice(choices=var_0)



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #12
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
    var_2 = ''
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = 'null'
    var_5 = var_1.validate(var_4)
    assert var_5 is None
    var_6 = 'none'
    var_7 = var_1.validate(var_6)
    assert var_7 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'invalid'
    var_2 = var_0.validate(var_1)



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
    var_0 = module_0.Array()

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

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Array(additional_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = module_0.Array(min_items=var_0, max_items=var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Array(exact_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Test description'
    var_2 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = {}
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #16
#--------------------------




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
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'not a dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 1
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

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
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^test_'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'test_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'extra_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'extra_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'extra_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^valid_'
    var_1 = module_0.Field()
    var_2 = module_0.Object(property_names=var_1)
    var_3 = 'invalid_name'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = True
    var_5 = module_0.Choice(choices=var_3)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = False
    var_5 = module_0.Choice(choices=var_3)
    var_6 = None
    var_7 = var_5.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = 'b'
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = False
    var_5 = True
    var_6 = module_0.Choice(choices=var_3, coerce_types=var_5)
    var_7 = ''
    var_8 = var_6.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = True
    var_5 = module_0.Choice(choices=var_3, coerce_types=var_4)
    var_6 = ''
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = (var_0, var_1)
    var_3 = 'AB'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = module_0.Choice(choices=var_5)
    var_7 = (var_0, var_1)
    var_8 = var_6.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'AB'
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
    var_3 = 'AB'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = module_0.Choice(choices=var_5)
    var_7 = {var_0: var_1}
    var_8 = var_6.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 1
    var_4 = 'One'
    var_5 = (var_3, var_4)
    var_6 = True
    var_7 = 'True'
    var_8 = (var_6, var_7)
    var_9 = [var_2, var_5, var_8]
    var_10 = module_0.Choice(choices=var_9)
    var_11 = var_10.validate(var_0)
    assert var_11 == 'a'
    var_12 = var_10.validate(var_6)
    assert var_12 == 1
    var_13 = True
    var_14 = var_10.validate(var_13)
    assert var_14 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_invalid_integer. Retrieved 1/4 statements.


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
    var_0 = 1.23

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = '123'
    var_2 = var_0.validate(var_1)
    assert var_2 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(minimum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 10
    var_3 = 9
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Number(exclusive_minimum=var_0)
    var_2 = 11
    var_3 = var_1.validate(var_2)
    assert var_3 == 11
    var_4 = 10
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.Number(maximum=var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 100
    var_3 = 101
    var_4 = var_1.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.Number(exclusive_maximum=var_0)
    var_2 = 99
    var_3 = var_1.validate(var_2)
    assert var_3 == 99
    var_4 = 100
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 10
    var_3 = var_1.validate(var_2)
    assert var_3 == 10
    var_4 = 12
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 1.0
    var_3 = var_1.validate(var_2)
    var_4 = 1.2
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.00'
    var_1 = module_0.Number(precision=var_0)
    var_2 = 1.234
    var_3 = var_1.validate(var_2)
    var_4 = 1.235
    var_5 = var_1.validate(var_4)



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()

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

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(additional_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Array(additional_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Array(min_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Array(max_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = module_0.Array(exact_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Title'
    var_1 = module_0.Array()

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Description'
    var_1 = module_0.Array()



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'Answer'
    var_2 = 'The answer to everything'
    var_3 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.Const(var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Const(var_0)



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = [var_1]
    var_3 = module_0.Union(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = [var_1]
    var_3 = module_0.Union(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_0.Union(var_1)
    var_3 = 'valid'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'valid'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = module_0.Union(var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 'valid'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 'invalid'
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 'invalid'
    var_5 = var_3.validate(var_4)



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = 5
    var_3 = module_0.Array(var_1, min_items=var_2)



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Boolean(coerce_types=var_0)
    var_3 = 'invalid_value'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #24
#--------------------------




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
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'not a dict'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 1
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0.validate(var_3)

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
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^test_'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = 'test_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'req_key'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Object(additional_properties=var_2, required=var_1)
    var_4 = 'invalid_key'
    var_5 = 456
    var_6 = 123
    var_7 = 'value'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = var_3.validate(var_8)
    var_10 = {msg.code for msg in e.messages()}



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_numeric_type_int_with_float_non_integer. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 3.14



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_validate_float_with_non_integer_value_when_numeric_type_is_int.




# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_returns_validated_value_when_one_child_validates_successfully. Retrieved 1/8 statements.
# Partially parsed test_validate_raises_child_error_when_one_child_has_non_type_error. Retrieved 1/9 statements.
# Partially parsed test_validate_raises_union_error_when_no_child_validates_successfully. Retrieved 1/9 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = module_0.Union(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Union(var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = lambda : var_0
    var_2 = module_0.Field(default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 100

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    assert var_1 is None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_integer_required. Retrieved 1/4 statements.


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
    var_1 = 'not a number'
    var_2 = var_0.validate(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = True
    var_2 = var_0.validate(var_1)

def test_case_0():
    var_0 = 3.14

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 'inf'
    var_2 = float(var_1)
    var_3 = var_0.validate(var_2)

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
    var_0 = 5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 7
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Number(multiple_of=var_0)
    var_2 = 0.7
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number(coerce_types=var_0)
    var_2 = '42'
    var_3 = var_1.validate(var_2)
    assert var_3 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = module_0.Number(precision=var_0)
    var_2 = 3.14159
    var_3 = var_1.validate(var_2)



# Parsed testcases at query #31
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = var_1.validate(var_3)



# Parsed testcases at query #32
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_validate_float_with_non_integer_value_when_numeric_type_is_int. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 3.14



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_validate_does_not_raise_type_error_for_valid_numeric_type_conversion. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = '42'



# Parsed testcases at query #35
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)



# Parsed testcases at query #36
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
    var_8 = module_0.Choice(choices=var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None

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
    var_8 = module_0.Choice(choices=var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)

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
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'

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
    var_8 = module_0.Choice(choices=var_6, coerce_types=var_7)
    var_9 = ''
    var_10 = var_8.validate(var_9)
    assert var_10 is None

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
    var_8 = module_0.Choice(choices=var_6)
    var_9 = ''
    var_10 = var_8.validate(var_9)



# Parsed testcases at query #37
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



