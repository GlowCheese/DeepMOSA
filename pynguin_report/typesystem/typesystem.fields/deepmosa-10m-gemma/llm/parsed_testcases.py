####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = False
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True
    var_5 = 'TRUE'
    var_6 = var_2.validate(var_5)
    assert var_6 is True
    var_7 = 'on'
    var_8 = var_2.validate(var_7)
    assert var_8 is True
    var_9 = '1'
    var_10 = var_2.validate(var_9)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'false'
    var_4 = var_2.validate(var_3)
    assert var_4 is False
    var_5 = 'off'
    var_6 = var_2.validate(var_5)
    assert var_6 is False
    var_7 = '0'
    var_8 = var_2.validate(var_7)
    assert var_8 is False
    var_9 = ''
    var_10 = var_2.validate(var_9)
    assert var_10 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is True
    var_4 = 0
    var_5 = var_2.validate(var_4)
    assert var_5 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = 'null'
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = 'none'
    var_9 = var_3.validate(var_8)
    assert var_9 is None
    var_10 = ''
    var_11 = var_3.validate(var_10)
    assert var_11 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Boolean(coerce_types=var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    var_5 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'not_a_boolean'
    var_4 = var_2.validate(var_3)
    var_5 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = True
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = 'type'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 2/11 statements.
# Partially parsed test_validate_raises_null_error_when_value_is_none_and_allow_null_is_false. Retrieved 9/34 statements.
# Partially parsed test_validate_returns_first_valid_child_value. Retrieved 3/15 statements.
# Partially parsed test_validate_raises_union_error_when_no_children_match. Retrieved 9/18 statements.
# Partially parsed test_validate_raises_specific_child_error_when_exactly_one_non_type_error_exists. Retrieved 3/15 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = 'Err'
    var_2 = ()
    var_3 = 'messages'
    var_4 = 'M'
    var_5 = ()
    var_6 = 'code'
    var_7 = 'index'
    var_8 = None
    var_9 = e.messages()[0].code
    assert var_9 == 'null'

def test_case_0():
    var_0 = None
    var_1 = 'success'
    var_2 = 'some_input'

def test_case_0():
    var_0 = 'Err'
    var_1 = ()
    var_2 = 'messages'
    var_3 = 'M'
    var_4 = ()
    var_5 = 'code'
    var_6 = 'index'
    var_7 = False
    var_8 = 'input'
    var_9 = e.messages()[0].code
    assert var_9 == 'union'

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'input'
    var_3 = e.messages()[0].code
    assert var_3 == 'other'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 2/11 statements.
# Partially parsed test_validate_raises_null_error_when_value_is_none_and_allow_null_is_false. Retrieved 2/22 statements.
# Partially parsed test_validate_returns_validated_value_on_first_success. Retrieved 1/21 statements.
# Partially parsed test_validate_raises_candidate_error_when_single_non_type_error_exists. Retrieved 1/26 statements.
# Partially parsed test_validate_raises_union_error_when_all_fields_fail_with_type_errors. Retrieved 1/26 statements.
# Partially parsed test_validate_raises_union_error_when_all_fields_fail_with_multiple_messages. Retrieved 1/26 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = e.messages()[0].code
    assert var_2 == 'null'

def test_case_0():
    var_0 = 'success'

def test_case_0():
    var_0 = 123
    var_1 = e.messages()[0].code
    assert var_1 == 'custom'

def test_case_0():
    var_0 = 123
    var_1 = e.messages()[0].code
    assert var_1 == 'union'

def test_case_0():
    var_0 = 123
    var_1 = e.messages()[0].code
    assert var_1 == 'union'



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 3
    var_2 = {}
    var_3 = module_0.String(max_length=var_0, min_length=var_1, **var_2)
    var_4 = 'hello'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  spaced  '
    var_4 = var_2.validate(var_3)
    assert var_4 == 'spaced'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  spaced  '
    var_4 = var_2.validate(var_3)
    assert var_4 == '  spaced  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = 'null'

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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = '   '
    var_4 = var_2.validate(var_3)
    var_5 = 'blank'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = '   '
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = 'min_length'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcdef'
    var_4 = var_2.validate(var_3)
    var_5 = 'max_length'

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
    var_5 = 'pattern'

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
    var_2 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 == ''



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_object_validate_additional_properties_field. Retrieved 12/17 statements.
# Partially parsed test_object_validate_properties_success. Retrieved 3/7 statements.
# Partially parsed test_object_validate_properties_with_default. Retrieved 3/7 statements.
# Partially parsed test_object_validate_pattern_properties. Retrieved 6/10 statements.


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
    var_2 = 'not'
    var_3 = 'a'
    var_4 = 'dict'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = 'age'
    var_5 = 30
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

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
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'extra'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)

import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'err'
    var_2 = 'child_err'
    var_3 = 'extra'
    var_4 = [var_3]
    var_5 = module_0.Message(text=var_1, code=var_2, index=var_4)
    var_6 = [var_5]
    var_7 = module_0.ValidationError(messages=var_6)
    var_8 = (var_0, var_7)
    var_9 = 'extra'
    var_10 = 123
    var_11 = {var_9: var_10}

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'default_val'
    var_1 = 'name'
    var_2 = {}

def test_case_0():
    var_0 = '^user_'
    var_1 = 'user_1'
    var_2 = 'other'
    var_3 = 'active'
    var_4 = 'ignore'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_array_constructor_with_allow_null_and_default. Retrieved 5/6 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Item'
    var_1 = 'Description'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = 'Array'
    var_4 = 'Array Description'
    var_5 = 2
    var_6 = 'title'
    var_7 = 'description'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.Array(var_2, min_items=var_5, **var_8)
    var_10 = var_9.items
    var_11 = bool(var_9.items == var_2)
    assert var_11 is True
    var_12 = var_9.min_items
    assert var_12 == 2
    var_13 = var_9.max_items
    assert var_13 is None
    var_14 = var_9.additional_items
    assert var_14 is False
    var_15 = var_9.unique_items
    assert var_15 is False
    var_16 = var_9.title
    assert var_16 == 'Array'
    var_17 = var_9.description
    assert var_17 == 'Array Description'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'I1'
    var_1 = 'D1'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = 'I2'
    var_4 = 'D2'
    var_5 = module_0.Field(title=var_3, description=var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = {}
    var_9 = module_0.Array(var_6, var_7, **var_8)
    var_10 = var_9.items
    var_11 = bool(var_9.items == var_6)
    assert var_11 is True
    var_12 = var_9.min_items
    assert var_12 == 2
    var_13 = var_9.max_items
    assert var_13 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'I1'
    var_1 = 'D1'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = 5
    var_4 = {}
    var_5 = module_0.Array(var_2, exact_items=var_3, **var_4)
    var_6 = var_5.min_items
    assert var_6 == 5
    var_7 = var_5.max_items
    assert var_7 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'I1'
    var_1 = 'D1'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = 'Add'
    var_4 = module_0.Field(title=var_3, description=var_3)
    var_5 = {}
    var_6 = module_0.Array(var_2, var_4, **var_5)
    var_7 = var_6.additional_items
    var_8 = bool(var_6.additional_items == var_4)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = {}
    var_3 = module_0.Array(var_0, unique_items=var_1, **var_2)
    var_4 = var_3.unique_items
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_0, min_items=var_1, max_items=var_2, **var_3)
    var_5 = var_4.min_items
    assert var_5 == 1
    var_6 = var_4.max_items
    assert var_6 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'NullAllowed'
    var_1 = 'Desc'
    var_2 = True
    var_3 = None
    var_4 = 'title'
    var_5 = 'description'
    var_6 = 'allow_null'
    var_7 = 'default'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Array(**var_8)
    var_10 = var_9.allow_null
    assert var_10 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_object_validate_required_error. Retrieved 4/11 statements.
# Partially parsed test_object_validate_property_success. Retrieved 5/12 statements.
# Partially parsed test_object_validate_additional_properties_field. Retrieved 4/9 statements.
# Partially parsed test_object_validate_property_names_error. Retrieved 11/18 statements.


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
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = 'not'
    var_5 = 'a'
    var_6 = 'dict'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)

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
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = [var_1]
    var_3 = {}

def test_case_0():
    var_0 = 'valid_value'
    var_1 = None
    var_2 = 'name'
    var_3 = 'test'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

def test_case_0():
    var_0 = 123
    var_1 = None
    var_2 = 'extra'
    var_3 = {var_2: var_0}

import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = 'err'
    var_2 = 'invalid_property'
    var_3 = 'bad_key'
    var_4 = [var_3]
    var_5 = module_0.Message(text=var_1, code=var_2, index=var_4)
    var_6 = [var_5]
    var_7 = lambda add_prefix: var_6
    var_8 = 'bad_key'
    var_9 = 1
    var_10 = {var_8: var_9}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_valid_string_coercion. Retrieved 4/5 statements.
# Partially parsed test_validate_numeric_type_int_constraint. Retrieved 2/6 statements.
# Partially parsed test_validate_precision_rounding. Retrieved 5/6 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    assert var_4 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 10.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 10.5)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '10.5'
    var_4 = var_2.validate(var_3)

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
    var_6 = 'null'

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
    var_4 = 'type'

def test_case_0():
    var_0 = 10.0
    var_1 = 10.5
    var_2 = 'integer'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 5
    var_4 = 4
    var_5 = var_2.validate(var_4)
    var_6 = 'minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 5.1
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 5.1)
    assert var_5 is True
    var_6 = 5
    var_7 = var_2.validate(var_6)
    var_8 = 'exclusive_minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10
    var_4 = 11
    var_5 = var_2.validate(var_4)
    var_6 = 'maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 9.9
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 9.9)
    assert var_5 is True
    var_6 = 10
    var_7 = var_2.validate(var_6)
    var_8 = 'exclusive_maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 4
    var_4 = var_2.validate(var_3)
    assert var_4 == 4
    var_5 = 3
    var_6 = var_2.validate(var_5)
    var_7 = 'multiple_of'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.5)
    assert var_5 is True
    var_6 = 1.2
    var_7 = var_2.validate(var_6)
    var_8 = 'multiple_of'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 1.234
    var_4 = var_2.validate(var_3)
    var_5 = '1.23'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)
    var_4 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '10'
    var_4 = var_2.validate(var_3)
    var_5 = 'type'



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = False
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 'not a list'
    var_5 = var_3.validate(var_4)
    var_6 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, exact_items=var_2, **var_3)
    var_5 = 'one'
    var_6 = [var_5]
    var_7 = var_4.validate(var_6)
    var_8 = 'exact_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, exact_items=var_2, **var_3)
    var_5 = 'one'
    var_6 = 'two'
    var_7 = [var_5, var_6]
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == ['one', 'two'])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 3
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = 'one'
    var_6 = 'two'
    var_7 = [var_5, var_6]
    var_8 = var_4.validate(var_7)
    var_9 = 'min_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = []
    var_6 = var_4.validate(var_5)
    var_7 = 'empty'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = 'one'
    var_6 = 'two'
    var_7 = [var_5, var_6]
    var_8 = var_4.validate(var_7)
    var_9 = 'max_items'

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
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(var_11 == ['a', 1, 2])
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



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = 'null'



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'test'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'test'})
    assert var_9 is True



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Username'
    var_1 = "The user's unique name"
    var_2 = True
    var_3 = False
    var_4 = 10
    var_5 = 3
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = 'title'
    var_9 = 'description'
    var_10 = {var_8: var_0, var_9: var_1}
    var_11 = module_0.String(allow_blank=var_2, trim_whitespace=var_3, max_length=var_4, min_length=var_5, pattern=var_6, format=var_7, coerce_types=var_3, **var_10)
    var_12 = var_11.title
    assert var_12 == 'Username'
    var_13 = var_11.description
    assert var_13 == "The user's unique name"
    var_14 = var_11.allow_blank
    assert var_14 is True
    var_15 = var_11.trim_whitespace
    assert var_15 is False
    var_16 = var_11.max_length
    assert var_16 == 10
    var_17 = var_11.min_length
    assert var_17 == 3
    var_18 = var_11.pattern
    assert var_18 == '^[a-z]+$'
    var_19 = var_11.format
    assert var_19 == 'email'
    var_20 = var_11.coerce_types
    assert var_20 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = var_1.title
    assert var_2 == ''
    var_3 = var_1.description
    assert var_3 == ''
    var_4 = var_1.allow_blank
    assert var_4 is False
    var_5 = var_1.trim_whitespace
    assert var_5 is True
    var_6 = var_1.max_length
    assert var_6 is None
    var_7 = var_1.min_length
    assert var_7 is None
    var_8 = var_1.pattern
    assert var_8 is None
    var_9 = var_1.format
    assert var_9 is None
    var_10 = var_1.coerce_types
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = var_2.default
    assert var_3 == ''

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = var_3.pattern
    assert var_4 == '\\d+'
    var_5 = var_3.pattern_regex
    var_6 = bool(var_3.pattern_regex == var_1)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'Should have failed due to invalid max_length type'
    var_4 = AssertionError(var_3)
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = {}
    var_9 = module_0.String(min_length=var_7, **var_8)
    var_10 = 'Should have failed due to invalid min_length type'
    var_11 = AssertionError(var_10)
    var_12 = 123
    var_13 = {}
    var_14 = module_0.String(pattern=var_12, **var_13)
    var_15 = 'Should have failed due to invalid pattern type'
    var_16 = AssertionError(var_15)
    var_17 = 'not_a_string'
    var_18 = [var_17]
    var_19 = {}
    var_20 = module_0.String(format=var_18, **var_19)
    var_21 = 'Should have failed due to invalid format type'
    var_22 = AssertionError(var_21)



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'property_name_field'
    var_3 = {var_2: var_1}
    var_4 = module_0.Object(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {}
    var_8 = module_0.Object(property_names=var_6, **var_7)
    var_9 = 'valid_key'
    var_10 = 'some_value'
    var_11 = {var_9: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = bool(var_12 == {'valid_key': 'some_value'})
    assert var_13 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_object_validate_property_names_validation. Retrieved 5/16 statements.


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
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = 'age'
    var_8 = 30
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)

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
    var_10 = 30
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(var_12 == {'name': 'John', 'age': 30})
    assert var_13 is True
    var_14 = 'name'
    var_15 = 'age'
    var_16 = 'John'
    var_17 = 'not_an_int'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = var_7.validate(var_18)

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

def test_case_0():
    var_0 = 'good'
    var_1 = 'bad'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_array_constructor_allow_null_default_handling. Retrieved 3/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Array'
    var_1 = 'A test array'
    var_2 = 2
    var_3 = 'title'
    var_4 = 'description'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.Array(min_items=var_2, **var_5)
    var_7 = var_6.title
    assert var_7 == 'Test Array'
    var_8 = var_6.description
    assert var_8 == 'A test array'
    var_9 = var_6.min_items
    assert var_9 == 2
    var_10 = var_6.max_items
    assert var_10 is None
    var_11 = var_6.items
    assert var_11 is None
    var_12 = var_6.additional_items
    assert var_12 is False
    var_13 = var_6.unique_items
    assert var_13 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Item'
    var_1 = 'Item field'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = [var_2]
    var_4 = 5
    var_5 = {}
    var_6 = module_0.Array(var_3, max_items=var_4, **var_5)
    var_7 = var_6.items
    var_8 = bool(var_6.items == [var_2])
    assert var_8 is True
    var_9 = var_6.min_items
    assert var_9 == 1
    var_10 = var_6.max_items
    assert var_10 == 5

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
    var_0 = 'Extra'
    var_1 = 'Extra field'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = None
    var_4 = {}
    var_5 = module_0.Array(var_3, var_2, **var_4)
    var_6 = var_5.additional_items
    var_7 = bool(var_5.additional_items == var_2)
    assert var_7 is True
    var_8 = var_5.min_items
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = var_2.unique_items
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Item'
    var_1 = 'Item field'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = [var_2, var_2]
    var_4 = False
    var_5 = {}
    var_6 = module_0.Array(var_3, var_4, **var_5)
    var_7 = var_6.min_items
    assert var_7 == 2
    var_8 = var_6.max_items
    assert var_8 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'allow_null'
    var_3 = 'default'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Array(**var_4)
    var_6 = var_5.allow_null
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 2/11 statements.
# Partially parsed test_validate_raises_null_error_when_value_is_none_and_allow_null_is_false. Retrieved 4/28 statements.
# Partially parsed test_validate_returns_value_when_first_child_matches. Retrieved 1/9 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = 'Did not raise null error'
    var_3 = AssertionError(var_2)

def test_case_0():
    var_0 = 10

def test_case_0():
    pass



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_union_raises_candidate_error_when_exactly_one_error_is_not_a_type_error. Retrieved 2/30 statements.


def test_case_0():
    var_0 = None
    var_1 = 'some_value'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_array_init_items_not_list. Retrieved 5/6 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Single Field'
    var_1 = module_0.Field(title=var_0)
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = var_4.items
    var_6 = bool(var_4.items is not None)
    assert var_6 is True
    var_7 = var_4.items
    var_8 = var_4.min_items
    assert var_8 == 5



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_valid_int. Retrieved 1/3 statements.
# Partially parsed test_validate_valid_float. Retrieved 1/3 statements.
# Partially parsed test_validate_string_coercion. Retrieved 1/3 statements.
# Partially parsed test_validate_precision_success. Retrieved 2/4 statements.
# Partially parsed test_validate_int_type_float_not_integer_failure. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10.5

def test_case_0():
    var_0 = '10.5'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 4
    var_4 = var_2.validate(var_3)
    var_5 = 'minimum'

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
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 11
    var_4 = var_2.validate(var_3)
    var_5 = 'maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 5.1
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 5.1)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = 'exclusive_minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 9.9
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 9.9)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = 'exclusive_maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 4
    var_4 = var_2.validate(var_3)
    assert var_4 == 4

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 3
    var_4 = var_2.validate(var_3)
    var_5 = 'multiple_of'

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
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 0.7
    var_4 = var_2.validate(var_3)
    var_5 = 'multiple_of'

def test_case_0():
    var_0 = '0.01'
    var_1 = 10.555

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = 'null'

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

def test_case_0():
    var_0 = 10.5
    var_1 = 'integer'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'abc'
    var_3 = var_1.validate(var_2)
    var_4 = 'type'



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 2
    var_2 = {}
    var_3 = module_0.String(max_length=var_0, min_length=var_1, **var_2)
    var_4 = 'hello'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  trimmed  '
    var_4 = var_2.validate(var_3)
    assert var_4 == 'trimmed'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  not_trimmed  '
    var_4 = var_2.validate(var_3)
    assert var_4 == '  not_trimmed  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = 'null'

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
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = '   '
    var_4 = var_2.validate(var_3)
    var_5 = 'blank'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = '   '
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = 'min_length'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcdef'
    var_4 = var_2.validate(var_3)
    var_5 = 'max_length'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '12345'
    var_4 = var_2.validate(var_3)
    assert var_4 == '12345'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc12'
    var_4 = var_2.validate(var_3)
    var_5 = 'pattern'

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
    var_2 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(allow_blank=var_1, coerce_types=var_0, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 is None



