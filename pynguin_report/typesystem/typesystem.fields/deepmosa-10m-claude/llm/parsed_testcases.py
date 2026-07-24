####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_object_validate_with_min_properties_one. Retrieved 4/7 statements.
# Partially parsed test_object_validate_with_mapping_type. Retrieved 5/9 statements.


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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(var_5 == {'key': 'value'})
    assert var_6 is True

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
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Unknown'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'Unknown'})
    assert var_10 is True

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

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.Object(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'extra'
    var_8 = {}
    var_9 = 'value'
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = var_6.validate(var_10)
    var_12 = bool(var_11 == {'name': {}, 'extra': 'value'})
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.Object(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'name'
    var_8 = 'extra'
    var_9 = {}
    var_10 = 'value'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    var_15 = bool(len(e.messages()) > 0)
    assert var_15 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.Object(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = 'extra'
    var_9 = {}
    var_10 = 'value'
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(var_12 == {'name': {}, 'extra': 'value'})
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'S_test'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'S_test': 'value'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'key': 'value'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'toolongkey'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_0.Object(properties=var_7, **var_8)
    var_10 = 'John'
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = var_9.validate(var_12)
    var_14 = bool(var_13 == {'user': {'name': 'John'}})
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_0.Object(properties=var_7, **var_8)
    var_10 = 'user'
    var_11 = 'age'
    var_12 = 'not_a_number'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = var_9.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True
    var_17 = len(e.messages())
    var_18 = bool(len(e.messages()) > 0)
    assert var_18 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'S_test'
    var_7 = 'not_a_number'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = len(e.messages())
    var_12 = bool(len(e.messages()) > 0)
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_0, var_1]
    var_8 = {}
    var_9 = module_0.Object(properties=var_6, required=var_7, **var_8)
    var_10 = 'name'
    var_11 = 'John'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True
    var_15 = len(e.messages())
    var_16 = bool(len(e.messages()) > 0)
    assert var_16 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_0, var_1]
    var_8 = {}
    var_9 = module_0.Object(properties=var_6, required=var_7, **var_8)
    var_10 = 'John'
    var_11 = 'john@example.com'
    var_12 = {var_0: var_10, var_1: var_11}
    var_13 = var_9.validate(var_12)
    var_14 = bool(var_13 == {'name': 'John', 'email': 'john@example.com'})
    assert var_14 is True



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
    var_8 = bool('null' in str(e).lower())
    assert var_8 is True

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
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'false'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'on'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'off'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '1'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '0'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is False

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
    var_3 = 0
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'TRUE'
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
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'null'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'none'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_object_constructor_allow_null_sets_default_none. Retrieved 2/3 statements.


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
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Field 2'
    var_3 = module_0.Field(title=var_2)
    var_4 = 'key1'
    var_5 = 'key2'
    var_6 = {var_4: var_1, var_5: var_3}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = var_8.properties
    var_10 = bool(var_8.properties == var_6)
    assert var_10 is True
    var_11 = var_8.pattern_properties
    var_12 = bool(var_8.pattern_properties == {})
    assert var_12 is True
    var_13 = var_8.additional_properties
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Pattern Field'
    var_1 = module_0.Field(title=var_0)
    var_2 = '^test_.*'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = var_5.pattern_properties
    var_7 = bool(var_5.pattern_properties == var_3)
    assert var_7 is True
    var_8 = var_5.properties
    var_9 = bool(var_5.properties == {})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Object(required=var_2, **var_3)
    var_5 = var_4.required
    var_6 = bool(var_4.required == var_2)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = module_0.Object(required=var_2, **var_3)
    var_5 = var_4.required
    var_6 = bool(var_4.required == ['key1', 'key2'])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {}
    var_3 = module_0.Object(min_properties=var_0, max_properties=var_1, **var_2)
    var_4 = var_3.min_properties
    assert var_4 == 1
    var_5 = var_3.max_properties
    assert var_5 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = var_2.additional_properties
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Additional'
    var_1 = module_0.Field(title=var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = var_3.additional_properties
    var_5 = bool(var_3.additional_properties is var_1)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Property Names'
    var_1 = module_0.Field(title=var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = var_3.property_names
    var_5 = bool(var_3.property_names is var_1)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Object'
    var_1 = 'A test object'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Object(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Test Object'
    var_7 = var_5.description
    assert var_7 == 'A test object'

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
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.Object(**var_4)
    var_6 = var_5.default
    var_7 = bool(var_5.default == var_2)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Field 1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Pattern Field'
    var_3 = module_0.Field(title=var_2)
    var_4 = 'Additional Field'
    var_5 = module_0.Field(title=var_4)
    var_6 = 'Property Names'
    var_7 = module_0.Field(title=var_6)
    var_8 = 'Complete Object'
    var_9 = 'A complete test object'
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = False
    var_14 = True
    var_15 = 'prop1'
    var_16 = {var_15: var_1}
    var_17 = '^pat_.*'
    var_18 = {var_17: var_3}
    var_19 = 20
    var_20 = [var_15]
    var_21 = 'title'
    var_22 = 'description'
    var_23 = 'default'
    var_24 = 'allow_null'
    var_25 = 'read_only'
    var_26 = {var_21: var_8, var_22: var_9, var_23: var_12, var_24: var_13, var_25: var_14}
    var_27 = module_0.Object(properties=var_16, pattern_properties=var_18, additional_properties=var_5, property_names=var_7, min_properties=var_14, max_properties=var_19, required=var_20, **var_26)
    var_28 = var_27.title
    assert var_28 == 'Complete Object'
    var_29 = var_27.description
    assert var_29 == 'A complete test object'
    var_30 = var_27.default
    var_31 = bool(var_27.default == {'key': 'value'})
    assert var_31 is True
    var_32 = var_27.allow_null
    assert var_32 is False
    var_33 = var_27.read_only
    assert var_33 is True
    var_34 = var_27.properties
    var_35 = bool(var_27.properties == {'prop1': var_1})
    assert var_35 is True
    var_36 = var_27.pattern_properties
    var_37 = bool(var_27.pattern_properties == {'^pat_.*': var_3})
    assert var_37 is True
    var_38 = var_27.additional_properties
    var_39 = bool(var_27.additional_properties is var_5)
    assert var_39 is True
    var_40 = var_27.property_names
    var_41 = bool(var_27.property_names is var_7)
    assert var_41 is True
    var_42 = var_27.min_properties
    assert var_42 == 1
    var_43 = var_27.max_properties
    assert var_43 == 20
    var_44 = var_27.required
    var_45 = bool(var_27.required == ['prop1'])
    assert var_45 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = module_0.Field(title=var_0)
    var_2 = {}
    var_3 = module_0.Object(properties=var_1, **var_2)
    var_4 = var_3.properties
    var_5 = bool(var_3.properties == {})
    assert var_5 is True
    var_6 = var_3.additional_properties
    var_7 = bool(var_3.additional_properties is var_1)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = var_3.default
    assert var_4 is None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_union_validate_returns_first_valid_match. Retrieved 6/8 statements.


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
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'null'
    var_13 = bool('null' in str(e).lower())
    assert var_13 is True

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
    var_7 = 42
    var_8 = var_6.validate(var_7)
    assert var_8 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'union'
    var_14 = bool('union' in str(e).lower())
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = var_6.allow_null
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Union(var_5, **var_6)
    var_8 = 'this is too long'
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool('max_length' in str(e).lower() or 'length' in str(e).lower())
    assert var_11 is True

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_float_with_int_numeric_type_non_integer. Retrieved 1/4 statements.
# Partially parsed test_validate_float_with_int_numeric_type_integer. Retrieved 1/3 statements.
# Partially parsed test_validate_numeric_type_conversion. Retrieved 1/4 statements.
# Partially parsed test_validate_decimal_input. Retrieved 2/5 statements.


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
    var_7 = 'null'
    var_8 = bool('null' in str(e).lower())
    assert var_8 is True

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
    var_5 = 'type'
    var_6 = bool('type' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'integer'
    var_3 = bool('integer' in str(e).lower())
    assert var_3 is True

def test_case_0():
    var_0 = 3.0

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

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
    var_2 = '123.45'
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 123.45)
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
    var_6 = 'finite'
    var_7 = bool('finite' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'
    var_7 = bool('finite' in str(e).lower())
    assert var_7 is True

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
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'minimum'
    var_7 = bool('minimum' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 10.1
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 10.1)
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
    var_6 = 'exclusive_minimum'
    var_7 = bool('exclusive_minimum' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 100

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 101
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'maximum'
    var_7 = bool('maximum' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 99.9
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 99.9)
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
    var_6 = 'exclusive_maximum'
    var_7 = bool('exclusive_maximum' in str(e).lower())
    assert var_7 is True

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 17
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'
    var_7 = bool('multiple_of' in str(e).lower())
    assert var_7 is True

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
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 2.3
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'
    var_7 = bool('multiple_of' in str(e).lower())
    assert var_7 is True

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
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'not_a_number'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'
    var_6 = bool('type' in str(e).lower())
    assert var_6 is True

def test_case_0():
    var_0 = 3.0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 50
    var_6 = var_4.validate(var_5)
    assert var_6 == 50

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '123.45'



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Choice(choices=var_3, **var_6)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Choice(choices=var_3, **var_6)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'null'
    var_12 = bool('null' in str(e).lower())
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 == 'red'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'r'
    var_1 = 'Red'
    var_2 = (var_0, var_1)
    var_3 = 'g'
    var_4 = 'Green'
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = 'Blue'
    var_8 = (var_6, var_7)
    var_9 = [var_2, var_5, var_8]
    var_10 = {}
    var_11 = module_0.Choice(choices=var_9, **var_10)
    var_12 = var_11.validate(var_0)
    assert var_12 == 'r'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = 'yellow'
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'choice'
    var_10 = bool('choice' in str(e).lower())
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Choice(choices=var_3, coerce_types=var_4, **var_6)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = False
    var_6 = 'allow_null'
    var_7 = {var_6: var_4}
    var_8 = module_0.Choice(choices=var_3, coerce_types=var_5, **var_7)
    var_9 = ''
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'required'
    var_13 = bool('required' in str(e).lower())
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Choice(choices=var_3, **var_6)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'required'
    var_12 = bool('required' in str(e).lower())
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.validate(var_1)
    assert var_5 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = True
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'choice'
    var_10 = bool('choice' in str(e).lower())
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = False
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'choice'
    var_10 = bool('choice' in str(e).lower())
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.validate(var_1)
    assert var_6 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = [var_0, var_1]
    var_10 = var_8.validate(var_9)
    var_11 = bool(var_10 == [1, 2])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = {var_0: var_1}
    var_10 = var_8.validate(var_9)
    var_11 = bool(var_10 == {'a': 1})
    assert var_11 is True



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
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
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
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
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = 'd'
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
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
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
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
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
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
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    assert var_9 == 'key1'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'B Label'
    var_3 = (var_1, var_2)
    var_4 = 'c'
    var_5 = [var_0, var_3, var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = var_7.validate(var_1)
    assert var_8 == 'b'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.validate(var_1)
    assert var_6 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Choice(choices=var_0, **var_1)
    var_3 = 'a'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #8
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
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'null'
    var_6 = bool('null' in str(e).lower())
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'
    var_6 = bool('type' in str(e).lower())
    assert var_6 is True

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
    var_7 = bool('blank' in str(e).lower())
    assert var_7 is True

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
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_length'
    var_7 = bool('min_length' in str(e).lower())
    assert var_7 is True

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
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'max_length'
    var_7 = bool('max_length' in str(e).lower())
    assert var_7 is True

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
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'pattern'
    var_7 = bool('pattern' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hel\x00lo'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'
    var_4 = '\x00'
    var_5 = bool('\x00' not in var_3)
    assert var_5 is True

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
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_union_validate_predicate_line_18_true. Retrieved 1/19 statements.


def test_case_0():
    var_0 = 'test_value'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_float_when_integer_required. Retrieved 1/4 statements.
# Partially parsed test_validate_numeric_type_coercion. Retrieved 1/4 statements.
# Partially parsed test_validate_decimal_input. Retrieved 2/6 statements.


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
    var_7 = 'null'

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
    var_5 = 'type'

def test_case_0():
    var_0 = 3.5
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'integer'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    assert var_4 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10
    var_4 = 9
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 11
    var_4 = var_2.validate(var_3)
    assert var_4 == 11
    var_5 = 10
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'exclusive_minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 100
    var_4 = 101
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 99
    var_4 = var_2.validate(var_3)
    assert var_4 == 99
    var_5 = 100
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'exclusive_maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 15
    var_4 = var_2.validate(var_3)
    assert var_4 == 15
    var_5 = 17
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'multiple_of'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 2.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 2.5)
    assert var_5 is True
    var_6 = 2.3
    var_7 = var_2.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'multiple_of'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.145
    var_4 = var_2.validate(var_3)
    var_5 = 3.14
    var_6 = var_4 - var_5
    var_7 = abs(var_6)
    var_8 = 0.001
    var_9 = var_7 < var_8
    var_10 = bool(var_4 == 3.14 or var_9)
    assert var_10 is True

def test_case_0():
    var_0 = '42'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '123.45'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'not_a_number'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'



# Parsed testcases at query #11
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
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'allow_null'
    var_9 = {var_8: var_6}
    var_10 = module_0.Number(coerce_types=var_5, **var_9)
    var_11 = ''
    var_12 = var_10.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'allow_null'
    var_15 = {var_14: var_12}
    var_16 = module_0.Number(coerce_types=var_12, **var_15)
    var_17 = var_16.validate(var_11)
    assert var_17 == 0



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/9 statements.
# Partially parsed test_serialize_with_list_of_custom_serializers. Retrieved 3/14 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None

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
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.serialize(var_7)
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
    var_7 = 42
    var_8 = 'hello'
    var_9 = [var_7, var_8]
    var_10 = var_6.serialize(var_9)
    var_11 = bool(var_10 == [42, 'hello'])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = [var_1, var_3, var_5]
    var_7 = {}
    var_8 = module_0.Array(var_6, **var_7)
    var_9 = 1
    var_10 = 'test'
    var_11 = 2
    var_12 = [var_9, var_10, var_11]
    var_13 = var_8.serialize(var_12)
    var_14 = bool(var_13 == [1, 'test', 2])
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = []
    var_4 = var_2.serialize(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'Hello'
    var_1 = 'WORLD'
    var_2 = [var_0, var_1]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_object_validate_with_required_field_missing. Retrieved 7/10 statements.
# Partially parsed test_object_validate_with_additional_properties_false. Retrieved 11/14 statements.
# Partially parsed test_object_validate_with_property_names_invalid. Retrieved 7/10 statements.
# Partially parsed test_object_validate_with_invalid_nested_property. Retrieved 14/17 statements.


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
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

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
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_0, var_4: var_5}
    var_7 = var_2.validate(var_6)
    var_8 = bool(var_7 == {'a': 1, 'b': 2})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.Object(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'a'
    var_8 = 'b'
    var_9 = {}
    var_10 = 2
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 1
    var_7 = 2
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = var_3.validate(var_8)
    var_10 = bool(var_9 == {'a': 1, 'b': 2})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'S_name'
    var_7 = 'other'
    var_8 = 'John'
    var_9 = 'value'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = 'S_name'
    var_13 = bool('S_name' in var_11)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'ab'
    var_6 = 'abc'
    var_7 = 1
    var_8 = {var_5: var_7, var_6: var_0}
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == {'ab': 1, 'abc': 2})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'ab'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'inner_name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'outer'
    var_7 = {var_6: var_5}
    var_8 = {}
    var_9 = module_0.Object(properties=var_7, **var_8)
    var_10 = 'value'
    var_11 = {var_0: var_10}
    var_12 = {var_6: var_11}
    var_13 = var_9.validate(var_12)
    var_14 = bool(var_13 == {'outer': {'inner_name': 'value'}})
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'default_name'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'default_name'})
    assert var_10 is True

import typesystem.fields as module_0
import collections as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_1.UserDict(var_4, **var_5)
    var_7 = var_1.validate(var_6)
    var_8 = bool(var_7 == {'a': 1})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'z'
    var_1 = 'a'
    var_2 = 'm'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = {var_0: var_4, var_1: var_6, var_2: var_8}
    var_10 = {}
    var_11 = module_0.Object(properties=var_9, **var_10)
    var_12 = '1'
    var_13 = '2'
    var_14 = '3'
    var_15 = {var_0: var_12, var_1: var_13, var_2: var_14}
    var_16 = var_11.validate(var_15)
    var_17 = bool('z' in var_16 and 'a' in var_16 and ('m' in var_16))
    assert var_17 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 5
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, **var_2)
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = 'inner'
    var_8 = {var_7: var_6}
    var_9 = {}
    var_10 = module_0.Object(properties=var_8, **var_9)
    var_11 = 'inner'
    var_12 = 'name'
    var_13 = 'toolongname'
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = var_10.validate(var_15)
    var_17 = bool(False)
    assert var_17 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_object_validate_with_mapping_type. Retrieved 8/12 statements.


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
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

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
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'extra': 'value'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'S_name'
    var_7 = 'John'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'S_name': 'John'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'name'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'name': 'value'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'name'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Default'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'Default'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_0.Object(properties=var_7, **var_8)
    var_10 = {var_0: var_9}
    var_11 = {}
    var_12 = module_0.Object(properties=var_10, **var_11)
    var_13 = 'John'
    var_14 = 30
    var_15 = {var_1: var_13, var_2: var_14}
    var_16 = {var_0: var_15}
    var_17 = var_12.validate(var_16)
    var_18 = bool(var_17 == {'user': {'name': 'John', 'age': 30}})
    assert var_18 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_0.Object(properties=var_7, **var_8)
    var_10 = 'user'
    var_11 = 'age'
    var_12 = 'not an int'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = var_9.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'a'
    var_3 = 1
    var_4 = (var_2, var_3)
    var_5 = 'b'
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = [var_8]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_float_not_integer_with_int_numeric_type. Retrieved 1/4 statements.
# Partially parsed test_validate_precision_with_numeric_type. Retrieved 2/4 statements.
# Partially parsed test_validate_with_numeric_type_int. Retrieved 1/3 statements.
# Partially parsed test_validate_with_numeric_type_float. Retrieved 1/3 statements.
# Partially parsed test_validate_decimal_value. Retrieved 3/9 statements.


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
    var_2 = 'invalid'
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
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '-inf'
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
    var_3 = 9
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
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 101
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 100

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
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 99
    var_4 = var_2.validate(var_3)
    assert var_4 == 99

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 17
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.1
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.5)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.1
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.55
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.146
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 3.15)
    assert var_5 is True

def test_case_0():
    var_0 = '0.01'
    var_1 = 3.146

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
    var_3 = '3.14'
    var_4 = var_2.validate(var_3)
    var_5 = 3.14
    var_6 = var_4 - var_5
    var_7 = abs(var_6)
    var_8 = bool(var_7 < 0.0001)
    assert var_8 is True

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
    var_4 = var_3 - var_2
    var_5 = abs(var_4)
    var_6 = bool(var_5 < 0.0001)
    assert var_6 is True

def test_case_0():
    var_0 = 42.0

def test_case_0():
    var_0 = 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '3.14'
    var_3 = 3.14

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 50
    var_6 = var_4.validate(var_5)
    assert var_6 == 50

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Number(exclusive_minimum=var_0, exclusive_maximum=var_1, **var_2)
    var_4 = 50
    var_5 = var_3.validate(var_4)
    assert var_5 == 50

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Number(exclusive_minimum=var_0, exclusive_maximum=var_1, **var_2)
    var_4 = 0
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Number(exclusive_minimum=var_0, exclusive_maximum=var_1, **var_2)
    var_4 = 100
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_array_constructor_with_additional_items_field. Retrieved 5/7 statements.
# Partially parsed test_array_constructor_with_default. Retrieved 3/6 statements.
# Partially parsed test_array_constructor_with_default_callable. Retrieved 1/7 statements.


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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = var_3.items
    var_5 = bool(var_3.items is var_1)
    assert var_5 is True
    var_6 = var_3.additional_items
    assert var_6 is False
    var_7 = var_3.min_items
    assert var_7 is None
    var_8 = var_3.max_items
    assert var_8 is None

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
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_6.min_items
    assert var_9 == 2
    var_10 = var_6.max_items
    assert var_10 == 2
    var_11 = var_6.additional_items
    assert var_11 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = (var_1, var_3)
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = var_6.items
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = var_6.min_items
    assert var_9 == 2
    var_10 = var_6.max_items
    assert var_10 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = {}
    var_6 = module_0.Array(var_2, var_4, **var_5)
    var_7 = var_6.items
    var_8 = var_6.additional_items
    var_9 = var_6.max_items
    assert var_9 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, max_items=var_3, **var_4)
    var_6 = var_5.min_items
    assert var_6 == 1
    var_7 = var_5.max_items
    assert var_7 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Array(var_1, exact_items=var_2, **var_3)
    var_5 = var_4.min_items
    assert var_5 == 5
    var_6 = var_4.max_items
    assert var_6 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = var_4.unique_items
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'Test Array'
    var_3 = 'A test array field'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Array(var_1, **var_6)
    var_8 = var_7.title
    assert var_8 == 'Test Array'
    var_9 = var_7.description
    assert var_9 == 'A test array field'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = var_5.allow_null
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)
    var_6 = var_5.read_only
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = []
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(var_1, **var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = [var_1, var_3, var_5]
    var_7 = False
    var_8 = {}
    var_9 = module_0.Array(var_6, var_7, **var_8)
    var_10 = var_9.max_items
    assert var_10 == 3

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = var_8.max_items
    assert var_9 is None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_line_28_predicate_evaluates_to_false. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = None



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
    var_1 = module_0.Array(**var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

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
    var_5 = 'not an int'
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
    var_5 = False
    var_6 = {}
    var_7 = module_0.Array(var_4, var_5, **var_6)
    var_8 = 1
    var_9 = 'hello'
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 1

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = 1
    var_10 = 'hello'
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = var_8.validate(var_12)
    var_14 = bool(var_13 == [1, 'hello', 3])
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6, var_5]
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'unique_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 2
    var_6 = 3
    var_7 = [var_2, var_5, var_6]
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == [1, 2, 3])
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = False
    var_6 = [var_2, var_5]
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == [True, False])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = True
    var_5 = {}
    var_6 = module_0.Array(var_3, unique_items=var_4, **var_5)
    var_7 = 2
    var_8 = [var_4, var_7]
    var_9 = 3
    var_10 = 4
    var_11 = [var_9, var_10]
    var_12 = [var_8, var_11]
    var_13 = var_6.validate(var_12)
    var_14 = bool(var_13 == [[1, 2], [3, 4]])
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = True
    var_5 = {}
    var_6 = module_0.Array(var_3, unique_items=var_4, **var_5)
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = [var_7, var_8]
    var_11 = [var_9, var_10]
    var_12 = var_6.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 1
    var_15 = e.messages()[0].code
    assert var_15 == 'unique_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = True
    var_7 = {}
    var_8 = module_0.Array(var_5, unique_items=var_6, **var_7)
    var_9 = {var_0: var_6}
    var_10 = 2
    var_11 = {var_0: var_10}
    var_12 = [var_9, var_11]
    var_13 = var_8.validate(var_12)
    var_14 = bool(var_13 == [{'key': 1}, {'key': 2}])
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = 1
    var_4 = 'hello'
    var_5 = [var_3, var_4, var_0]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 'hello', None])
    assert var_7 is True

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
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 'not int'
    var_8 = 123
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(e.messages())
    var_13 = bool(len(e.messages()) >= 1)
    assert var_13 is True

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



# Parsed testcases at query #19
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
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'blank'
    var_9 = 'allow_null'
    var_10 = {var_9: var_6}
    var_11 = module_0.String(allow_blank=var_5, coerce_types=var_5, **var_10)
    var_12 = ''
    var_13 = var_11.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True
    var_15 = 'blank'
    var_16 = 'allow_null'
    var_17 = {var_16: var_12}
    var_18 = module_0.String(allow_blank=var_12, coerce_types=var_12, **var_17)
    var_19 = ''
    var_20 = var_18.validate(var_19)
    var_21 = bool(False)
    assert var_21 is True
    var_22 = 'blank'



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 'test_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = 'different_value'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'const'
    var_7 = bool('const' in str(e).lower())
    assert var_7 is True

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
    var_3 = 'some_value'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'only_null'
    var_7 = bool('only_null' in str(e).lower())
    assert var_7 is True

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
    var_6 = 'const'
    var_7 = bool('const' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Const(var_3, **var_4)
    var_6 = [var_0, var_1, var_2]
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.Const(var_2, **var_3)
    var_5 = {var_0: var_1}
    var_6 = var_4.validate(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_string_constructor_with_default. Retrieved 2/3 statements.
# Partially parsed test_string_constructor_with_allow_blank_no_default. Retrieved 2/3 statements.


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
    var_0 = 'Name'
    var_1 = 'User name'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.String(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Name'
    var_7 = var_5.description
    assert var_7 == 'User name'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True
    var_5 = var_3.default
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = var_3.default
    assert var_4 == 'hello'

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
    var_0 = True
    var_1 = 'custom'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, **var_3)
    var_5 = var_4.allow_blank
    assert var_5 is True
    var_6 = var_4.default
    assert var_6 == 'custom'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = var_2.max_length
    assert var_3 == 100

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = var_2.min_length
    assert var_3 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = var_2.pattern
    assert var_3 == '^\\d+$'
    var_4 = var_2.pattern_regex
    var_5 = bool(var_2.pattern_regex is not None)
    assert var_5 is True

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^\\w+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = var_3.pattern
    assert var_4 == '^\\w+$'
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
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = var_2.trim_whitespace
    assert var_3 is False

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
    var_1 = 'read_only'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = var_3.read_only
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Email'
    var_1 = 'User email address'
    var_2 = 'test@example.com'
    var_3 = False
    var_4 = True
    var_5 = 255
    var_6 = 5
    var_7 = '^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$'
    var_8 = 'email'
    var_9 = 'title'
    var_10 = 'description'
    var_11 = 'default'
    var_12 = 'allow_null'
    var_13 = 'read_only'
    var_14 = {var_9: var_0, var_10: var_1, var_11: var_2, var_12: var_3, var_13: var_3}
    var_15 = module_0.String(allow_blank=var_3, trim_whitespace=var_4, max_length=var_5, min_length=var_6, pattern=var_7, format=var_8, coerce_types=var_4, **var_14)
    var_16 = var_15.title
    assert var_16 == 'Email'
    var_17 = var_15.description
    assert var_17 == 'User email address'
    var_18 = var_15.default
    assert var_18 == 'test@example.com'
    var_19 = var_15.allow_null
    assert var_19 is False
    var_20 = var_15.read_only
    assert var_20 is False
    var_21 = var_15.allow_blank
    assert var_21 is False
    var_22 = var_15.trim_whitespace
    assert var_22 is True
    var_23 = var_15.max_length
    assert var_23 == 255
    var_24 = var_15.min_length
    assert var_24 == 5
    var_25 = var_15.pattern
    assert var_25 == '^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$'
    var_26 = var_15.format
    assert var_26 == 'email'
    var_27 = var_15.coerce_types
    assert var_27 is True



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'hello'
    var_6 = 'world'
    var_7 = [var_5, var_6, var_5]
    var_8 = var_4.validate(var_7)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_numeric_type_int_with_non_integer_float_raises_error. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_union_validate_predicate_line_18_true. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'type'
    var_1 = True
    var_2 = 'test_value'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_object_validate_with_mapping_type. Retrieved 5/9 statements.


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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(var_5 == {'key': 'value'})
    assert var_6 is True

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

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = 'John'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)
    var_7 = bool(var_6 == {'name': 'John'})
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
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == {'a': 1})
    assert var_6 is True

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

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = {var_3: var_5, var_4: var_0}
    var_7 = var_2.validate(var_6)
    var_8 = bool(var_7 == {'a': 1, 'b': 2})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

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
    var_0 = 'name'
    var_1 = 'Default'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'Default'})
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'extra': 'value'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'S_name'
    var_7 = 'John'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'S_name': 'John'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = {}
    var_6 = module_0.Object(pattern_properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'other'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == {'other': 'value'})
    assert var_11 is True

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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'extra'
    var_8 = 'John'
    var_9 = 'value'
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = var_6.validate(var_10)
    var_12 = bool(var_11 == {'name': 'John', 'extra': 'value'})
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'key': 'value'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'toolongkey'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #28
#--------------------------






# Parsed testcases at query #29
#--------------------------

# Partially parsed test_union_validate_returns_first_valid_match. Retrieved 6/8 statements.


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
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = [var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
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
    var_7 = 42
    var_8 = var_6.validate(var_7)
    assert var_8 == 42

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
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Union(var_5, **var_6)
    var_8 = 'toolong'
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
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
    var_7 = '123'
    var_8 = var_6.validate(var_7)
    assert var_8 == '123'



# Parsed testcases at query #30
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0, var_0]
    var_2 = 5
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_1, var_3, max_items=var_2, **var_4)
    var_6 = var_5.max_items
    assert var_6 == 5
    var_7 = module_0.Field()
    var_8 = module_0.Field()
    var_9 = [var_7, var_7]
    var_10 = None
    var_11 = {}
    var_12 = module_0.Array(var_9, var_8, max_items=var_10, **var_11)
    var_13 = var_12.max_items
    assert var_13 is None
    var_14 = module_0.Field()
    var_15 = module_0.Field()
    var_16 = [var_14, var_14]
    var_17 = 10
    var_18 = {}
    var_19 = module_0.Array(var_16, var_15, max_items=var_17, **var_18)
    var_20 = var_19.max_items
    assert var_20 == 10



# Parsed testcases at query #31
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test that the except block at line 25 evaluates to False (no exception is raised)'
    var_1 = {}
    var_2 = module_0.Number(**var_1)
    var_3 = 42
    var_4 = var_2.validate(var_3)
    assert var_4 == 42



# Parsed testcases at query #32
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    assert var_3 == 42



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_array_init_min_items_not_none. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 5



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_array_validate_nested_validation_error. Retrieved 7/8 statements.
# Partially parsed test_array_validate_multiple_validation_errors. Retrieved 7/8 statements.


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
    var_5 = False
    var_6 = {}
    var_7 = module_0.Array(var_4, var_5, **var_6)
    var_8 = 1
    var_9 = 'hello'
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 1

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = 1
    var_10 = 'hello'
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = var_8.validate(var_12)
    var_14 = bool(var_13 == [1, 'hello', 3])
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6, var_5]
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'unique_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 2
    var_6 = 3
    var_7 = [var_2, var_5, var_6]
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == [1, 2, 3])
    assert var_9 is True

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
    var_5 = [var_0, var_3]
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = [var_4, var_5, var_8]
    var_10 = var_2.validate(var_9)
    var_11 = len(var_10)
    assert var_11 == 3

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
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 'not_int'
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
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 'not_int'
    var_6 = 'also_not_int'
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #35
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Boolean(coerce_types=var_0, **var_3)
    var_5 = 'true'
    var_6 = var_4.validate(var_5)
    assert var_6 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_get_default_value_with_complex_callable. Retrieved 1/5 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 42
    var_2 = module_0.Field(title=var_0, default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'dynamic_value'
    var_2 = lambda : var_1
    var_3 = module_0.Field(title=var_0, default=var_2)
    var_4 = var_3.get_default_value()
    assert var_4 == 'dynamic_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = module_0.Field(title=var_0, default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = module_0.Field(title=var_0, allow_null=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field(title=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'string_value'
    var_2 = module_0.Field(title=var_0, default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 'string_value'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.Field(title=var_0, default=var_4)
    var_6 = var_5.get_default_value()
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Field(title=var_0, default=var_3)
    var_5 = var_4.get_default_value()
    var_6 = bool(var_5 == {'key': 'value'})
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = module_0.Field(title=var_0, default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = module_0.Field(title=var_0, default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 is False

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #37
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = 'hello'
    var_7 = 'world'
    var_8 = [var_6, var_7]
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_object_validate_with_mapping_type. Retrieved 5/9 statements.


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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(var_5 == {'key': 'value'})
    assert var_6 is True

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
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

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
    var_9 = bool(len(e.messages()) > 0)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'extra': 'value'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'S_name'
    var_7 = 'John'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'S_name': 'John'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = 'name'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'name': 'value'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'default_name'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'default_name'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'inner_key'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'outer_key'
    var_7 = {var_6: var_5}
    var_8 = {}
    var_9 = module_0.Object(properties=var_7, **var_8)
    var_10 = 'value'
    var_11 = {var_0: var_10}
    var_12 = {var_6: var_11}
    var_13 = var_9.validate(var_12)
    var_14 = bool(var_13 == {'outer_key': {'inner_key': 'value'}})
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'num'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'outer'
    var_7 = {var_6: var_5}
    var_8 = {}
    var_9 = module_0.Object(properties=var_7, **var_8)
    var_10 = 'outer'
    var_11 = 'num'
    var_12 = 'not_a_number'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = var_9.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_0, var_1]
    var_8 = {}
    var_9 = module_0.Object(properties=var_6, required=var_7, **var_8)
    var_10 = 'name'
    var_11 = 'John'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True
    var_15 = len(e.messages())
    var_16 = bool(len(e.messages()) > 0)
    assert var_16 is True

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
    var_0 = '^S_'
    var_1 = '^I_'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(pattern_properties=var_6, **var_7)
    var_9 = 'S_name'
    var_10 = 'I_id'
    var_11 = 'John'
    var_12 = '123'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = var_8.validate(var_13)
    var_15 = bool(var_14 == {'S_name': 'John', 'I_id': '123'})
    assert var_15 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = 'extra'
    var_9 = 'John'
    var_10 = 'value'
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(var_12 == {'name': 'John', 'extra': 'value'})
    assert var_13 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_union_validate_returns_first_matching_type. Retrieved 6/8 statements.


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
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'null'
    var_13 = bool('null' in str(e).lower())
    assert var_13 is True

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
    var_7 = 42
    var_8 = var_6.validate(var_7)
    assert var_8 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'union'
    var_14 = bool('union' in str(e).lower())
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = [var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
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
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = '123'
    var_8 = var_6.validate(var_7)
    assert var_8 == '123'



# Parsed testcases at query #40
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = 'option3'
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = 'invalid_option'
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'Not a valid choice'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_numeric_type_is_int_with_non_integer_float. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #42
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #43
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #44
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Boolean(coerce_types=var_0, **var_3)
    var_5 = 'true'
    var_6 = var_4.validate(var_5)
    assert var_6 is True
    var_7 = 'false'
    var_8 = var_4.validate(var_7)
    assert var_8 is False
    var_9 = var_4.validate(var_0)
    assert var_9 is True
    var_10 = var_4.validate(var_1)
    assert var_10 is False
    var_11 = 'on'
    var_12 = var_4.validate(var_11)
    assert var_12 is True
    var_13 = 'off'
    var_14 = var_4.validate(var_13)
    assert var_14 is False



# Parsed testcases at query #45
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'Option 1'
    var_2 = (var_0, var_1)
    var_3 = 'option2'
    var_4 = 'Option 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = 'invalid_value'
    var_12 = var_10.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #46
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
    var_7 = len(e.messages())
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    var_6 = bool(len(e.messages()) > 0)
    assert var_6 is True

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
    var_9 = bool(len(e.messages()) > 0)
    assert var_9 is True

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
    var_7 = bool(len(e.messages()) > 0)
    assert var_7 is True

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
    var_9 = bool(len(e.messages()) > 0)
    assert var_9 is True

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
    var_10 = bool(len(e.messages()) > 0)
    assert var_10 is True

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
    var_9 = bool(len(e.messages()) > 0)
    assert var_9 is True

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
    var_3 = False
    var_4 = [var_0, var_0, var_3, var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [True, 1, False, 0])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {}
    var_4 = module_0.Array(var_0, var_2, **var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == [1, 2, 3])
    assert var_10 is True

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
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]
    var_9 = var_1.validate(var_8)
    var_10 = bool(var_9 == [[1, 2], [3, 4]])
    assert var_10 is True

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
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

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
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

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
    var_0 = 2
    var_1 = 4
    var_2 = {}
    var_3 = module_0.Array(min_items=var_0, max_items=var_1, **var_2)
    var_4 = 1
    var_5 = 3
    var_6 = [var_4, var_0, var_5]
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_line_26_predicate_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 5



# Parsed testcases at query #48
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
    var_7 = len(e.messages())
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    var_6 = bool(len(e.messages()) > 0)
    assert var_6 is True

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
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

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
    var_7 = len(e.messages())
    var_8 = bool(len(e.messages()) > 0)
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
    var_6 = len(e.messages())
    var_7 = bool(len(e.messages()) > 0)
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
    var_9 = len(e.messages())
    var_10 = bool(len(e.messages()) > 0)
    assert var_10 is True

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    var_9 = bool(len(e.messages()) > 0)
    assert var_9 is True

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
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Array(var_2, var_4, **var_5)
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
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == [1, 2])
    assert var_11 is True

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
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

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
    var_3 = True
    var_4 = [var_3, var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = 1
    var_4 = 'string'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = [var_3, var_4, var_7]
    var_9 = var_2.validate(var_8)
    var_10 = bool(var_9 == [1, 'string', {'key': 'value'}])
    assert var_10 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_array_init_min_items_not_none. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 5



# Parsed testcases at query #50
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = 'Must be a boolean.'
    var_3 = 'May not be null.'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'true'
    var_6 = 'false'
    var_7 = 'on'
    var_8 = 'off'
    var_9 = '1'
    var_10 = '0'
    var_11 = ''
    var_12 = 1
    var_13 = 0
    var_14 = True
    var_15 = False
    var_16 = True
    var_17 = False
    var_18 = True
    var_19 = False
    var_20 = False
    var_21 = True
    var_22 = False
    var_23 = {var_5: var_14, var_6: var_15, var_7: var_16, var_8: var_17, var_9: var_18, var_10: var_19, var_11: var_20, var_12: var_21, var_13: var_22}
    var_24 = ''
    var_25 = 'null'
    var_26 = 'none'
    var_27 = {var_24, var_25, var_26}
    var_28 = True
    var_29 = False
    var_30 = 'allow_null'
    var_31 = {var_30: var_29}
    var_32 = module_0.Boolean(coerce_types=var_28, **var_31)
    var_33 = 'true'
    var_34 = var_32.validate(var_33)
    assert var_34 is True
    var_35 = var_32.validate(var_28)
    assert var_35 is True
    var_36 = 'false'
    var_37 = var_32.validate(var_36)
    assert var_37 is False



# Parsed testcases at query #51
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = '   '
    var_6 = var_4.validate(var_5)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_union_validate_predicate_line_18_true. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'validate_or_error'
    var_1 = 'allow_null'
    var_2 = [var_0, var_1]
    var_3 = None
    var_4 = 'test_value'
    var_5 = [var_4, var_1]
    var_6 = 'test_value'



# Parsed testcases at query #53
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = '  '
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #54
#--------------------------




import typesystem.fields as module_0
import typesystem.unique as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.Uniqueness()
    var_6 = 'test'
    var_7 = var_5.add(var_6)
    var_8 = bool(var_6 in var_5)
    assert var_8 is True
    var_9 = 'hello'
    var_10 = 'world'
    var_11 = [var_9, var_10, var_9]
    var_12 = var_4.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'unique'
    var_15 = bool('unique' in str(e).lower())
    assert var_15 is True
    var_16 = 'hello'
    var_17 = 'world'
    var_18 = 'goodbye'
    var_19 = [var_16, var_17, var_18]
    var_20 = var_4.validate(var_19)
    var_21 = bool(var_20 == ['hello', 'world', 'goodbye'])
    assert var_21 is True
    var_22 = None
    var_23 = {}
    var_24 = module_0.Array(var_22, unique_items=var_10, **var_23)
    var_25 = False
    var_26 = [var_10, var_10, var_25, var_25]
    var_27 = var_24.validate(var_26)
    var_28 = bool(var_27 == [True, 1, False, 0])
    assert var_28 is True
    var_29 = True
    var_30 = [var_29, var_29]
    var_31 = var_24.validate(var_30)
    var_32 = bool(False)
    assert var_32 is True
    var_33 = 'unique'
    var_34 = bool('unique' in str(e).lower())
    assert var_34 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_line_11_predicate_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #56
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
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'null'
    var_6 = bool('null' in str(e).lower())
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'
    var_6 = bool('type' in str(e).lower())
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'helloworld'

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
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'blank'
    var_7 = bool('blank' in str(e).lower())
    assert var_7 is True

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
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_length'
    var_7 = bool('min_length' in str(e).lower())
    assert var_7 is True

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
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'max_length'
    var_7 = bool('max_length' in str(e).lower())
    assert var_7 is True

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
    var_6 = 'pattern'
    var_7 = bool('pattern' in str(e).lower())
    assert var_7 is True

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = '12345'
    var_5 = var_3.validate(var_4)
    assert var_5 == '12345'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = '^[a-z]+$'
    var_3 = False
    var_4 = True
    var_5 = {}
    var_6 = module_0.String(allow_blank=var_3, trim_whitespace=var_4, max_length=var_1, min_length=var_0, pattern=var_2, **var_5)
    var_7 = '  hello  '
    var_8 = var_6.validate(var_7)
    assert var_8 == 'hello'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_array_init_min_items_not_none. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 5



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_validate_first_matching_type_wins. Retrieved 6/8 statements.
# Partially parsed test_validate_integer_before_string. Retrieved 6/8 statements.


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
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = bool('null' in str(e).lower() or 'may not be null' in str(e).lower())
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = [var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
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
    var_7 = 42
    var_8 = var_6.validate(var_7)
    assert var_8 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool('union' in str(e).lower() or 'did not match' in str(e).lower())
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Union(var_5, **var_6)
    var_8 = 'this is too long'
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool('length' in str(e).lower() or 'max_length' in str(e).lower())
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = '123'
    var_8 = var_6.validate(var_7)
    assert var_8 == '123'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 99
    var_8 = var_6.validate(var_7)
    assert var_8 == 99



# Parsed testcases at query #59
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test that the except clause at line 20 is NOT executed (predicate evaluates to False)'
    var_1 = 'type'
    var_2 = 'null'
    var_3 = 'Must be a boolean.'
    var_4 = 'May not be null.'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'true'
    var_7 = 'false'
    var_8 = 'on'
    var_9 = 'off'
    var_10 = '1'
    var_11 = '0'
    var_12 = ''
    var_13 = 1
    var_14 = 0
    var_15 = True
    var_16 = False
    var_17 = True
    var_18 = False
    var_19 = True
    var_20 = False
    var_21 = False
    var_22 = True
    var_23 = False
    var_24 = {var_6: var_15, var_7: var_16, var_8: var_17, var_9: var_18, var_10: var_19, var_11: var_20, var_12: var_21, var_13: var_22, var_14: var_23}
    var_25 = ''
    var_26 = 'null'
    var_27 = 'none'
    var_28 = {var_25, var_26, var_27}
    var_29 = True
    var_30 = False
    var_31 = 'allow_null'
    var_32 = {var_31: var_30}
    var_33 = module_0.Boolean(coerce_types=var_29, **var_32)
    var_34 = 'true'
    var_35 = var_33.validate(var_34)
    assert var_35 is True
    var_36 = 'false'
    var_37 = var_33.validate(var_36)
    assert var_37 is False
    var_38 = var_33.validate(var_29)
    assert var_38 is True
    var_39 = var_33.validate(var_30)
    assert var_39 is False



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_union_validate_predicate_line_17_true. Retrieved 6/48 statements.


def test_case_0():
    var_0 = 'type'
    var_1 = 'test_value'
    var_2 = len(e.messages())
    assert var_2 == 1
    var_3 = e.messages()[0].code
    assert var_3 == 'union'
    var_4 = 'custom_error'
    var_5 = 'test_value'
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'union'
    var_8 = 0
    var_9 = 'test_value'
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'union'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_union_validate_predicate_line_17_true. Retrieved 1/20 statements.
# Partially parsed test_union_validate_predicate_line_17_true_with_non_type_code. Retrieved 1/20 statements.
# Partially parsed test_union_validate_predicate_line_17_true_with_index. Retrieved 1/20 statements.


def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 'test_value'

def test_case_0():
    var_0 = 'test_value'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_numeric_type_is_int_with_non_integer_float. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #63
#--------------------------




import typesystem.fields as module_0
import typesystem.unique as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.Uniqueness()
    var_6 = 'duplicate'
    var_7 = var_5.add(var_6)
    var_8 = bool(var_6 in var_5)
    assert var_8 is True
    var_9 = module_1.Uniqueness()
    var_10 = var_9.add(var_2)
    var_11 = True
    var_12 = bool(True in var_9)
    assert var_12 is True
    var_13 = 1
    var_14 = bool(1 not in var_9)
    assert var_14 is True
    var_15 = module_1.Uniqueness()
    var_16 = False
    var_17 = var_15.add(var_16)
    var_18 = False
    var_19 = bool(False in var_15)
    assert var_19 is True
    var_20 = 0
    var_21 = bool(0 not in var_15)
    assert var_21 is True
    var_22 = module_1.Uniqueness()
    var_23 = 2
    var_24 = 3
    var_25 = [var_2, var_23, var_24]
    var_26 = var_22.add(var_25)
    var_27 = bool(var_25 in var_22)
    assert var_27 is True
    var_28 = module_1.Uniqueness()
    var_29 = 'key'
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = var_28.add(var_31)
    var_33 = bool(var_31 in var_28)
    assert var_33 is True



# Parsed testcases at query #64
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'apple'
    var_6 = 'banana'
    var_7 = 'cherry'
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == ['apple', 'banana', 'cherry'])
    assert var_10 is True



# Parsed testcases at query #65
#--------------------------




import typesystem.fields as module_0
import typesystem.unique as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.Uniqueness()
    var_6 = 'test'
    var_7 = var_5.add(var_6)
    var_8 = var_6 in var_5
    assert var_8 is True



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_validate_float_when_integer_required. Retrieved 1/4 statements.
# Partially parsed test_validate_valid_float_when_integer_required. Retrieved 1/3 statements.
# Partially parsed test_validate_integer_coerced_to_numeric_type. Retrieved 1/4 statements.
# Partially parsed test_validate_valid_integer. Retrieved 1/3 statements.


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
    var_7 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'

def test_case_0():
    var_0 = 3.5
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'integer'

def test_case_0():
    var_0 = 3.0

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'

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
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '-inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

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
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 6
    var_4 = var_2.validate(var_3)
    assert var_4 == 6

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exclusive_minimum'

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
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'maximum'

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
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exclusive_maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.456
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 3.46)
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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 13
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'

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
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 2.3
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'

def test_case_0():
    var_0 = 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'not_a_number'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123.45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 123.45)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 50
    var_6 = var_4.validate(var_5)
    assert var_6 == 50

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 55
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'multiple_of'

def test_case_0():
    var_0 = 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 3.14
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == 3.14)
    assert var_4 is True



# Parsed testcases at query #67
#--------------------------




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
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 'not a list'
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    assert var_7 == 1
    var_8 = e.messages()[0].code
    assert var_8 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, exact_items=var_2, **var_3)
    var_5 = 'a'
    var_6 = [var_5]
    var_7 = var_4.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'exact_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = 'a'
    var_6 = [var_5]
    var_7 = var_4.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0].code
    assert var_10 == 'min_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = []
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0].code
    assert var_9 == 'empty'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, max_items=var_2, **var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = len(e.messages())
    assert var_11 == 1
    var_12 = e.messages()[0].code
    assert var_12 == 'max_items'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 'hello'
    var_5 = 'world'
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == ['hello', 'world'])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 'hello'
    var_8 = 42
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == ['hello', 42])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = 'hello'
    var_10 = 42
    var_11 = 'extra'
    var_12 = [var_9, var_10, var_11]
    var_13 = var_8.validate(var_12)
    var_14 = bool(var_13 == ['hello', 42, 'extra'])
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
    var_8 = 'hello'
    var_9 = 42
    var_10 = 'extra'
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    assert var_14 == 1

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == ['a', 'b', 'c'])
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6, var_5]
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'unique_items'

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
    var_4 = 1
    var_5 = 'not an int'
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = []
    var_5 = var_3.validate(var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = 1
    var_4 = 'string'
    var_5 = True
    var_6 = [var_3, var_4, var_5, var_0]
    var_7 = var_2.validate(var_6)
    var_8 = bool(var_7 == [1, 'string', True, None])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 2
    var_3 = {}
    var_4 = module_0.Array(var_1, exact_items=var_2, **var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == ['a', 'b'])
    assert var_9 is True



# Parsed testcases at query #68
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'apple'
    var_6 = 'banana'
    var_7 = 'cherry'
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == ['apple', 'banana', 'cherry'])
    assert var_10 is True



# Parsed testcases at query #69
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
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

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
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2])
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
    var_6 = module_0.Integer(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = 1
    var_10 = 'hello'
    var_11 = 42
    var_12 = [var_9, var_10, var_11]
    var_13 = var_8.validate(var_12)
    var_14 = bool(var_13 == [1, 'hello', 42])
    assert var_14 is True

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
    var_10 = 42
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

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
    var_3 = False
    var_4 = [var_0, var_0, var_3, var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [True, 1, False, 0])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 1
    var_3 = 3
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, max_items=var_3, **var_4)
    var_6 = 1
    var_7 = 'not_int'
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = var_5.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = 1
    var_6 = 'invalid'
    var_7 = [var_5, var_6]
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = 1
    var_4 = 'string'
    var_5 = 3.14
    var_6 = [var_3, var_4, var_0, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = bool(var_7 == [1, 'string', None, 3.14])
    assert var_8 is True



# Parsed testcases at query #70
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #71
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = 'Must be a boolean.'
    var_3 = 'May not be null.'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'true'
    var_6 = 'false'
    var_7 = 'on'
    var_8 = 'off'
    var_9 = '1'
    var_10 = '0'
    var_11 = ''
    var_12 = 1
    var_13 = 0
    var_14 = True
    var_15 = False
    var_16 = True
    var_17 = False
    var_18 = True
    var_19 = False
    var_20 = False
    var_21 = True
    var_22 = False
    var_23 = {var_5: var_14, var_6: var_15, var_7: var_16, var_8: var_17, var_9: var_18, var_10: var_19, var_11: var_20, var_12: var_21, var_13: var_22}
    var_24 = ''
    var_25 = 'null'
    var_26 = 'none'
    var_27 = {var_24, var_25, var_26}
    var_28 = True
    var_29 = False
    var_30 = 'allow_null'
    var_31 = {var_30: var_29}
    var_32 = module_0.Boolean(coerce_types=var_28, **var_31)
    var_33 = 'true'
    var_34 = var_32.validate(var_33)
    assert var_34 is True
    var_35 = var_32.validate(var_28)
    assert var_35 is True
    var_36 = 'false'
    var_37 = var_32.validate(var_36)
    assert var_37 is False



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_validate_returns_value_when_format_is_native_type. Retrieved 4/5 statements.


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
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'null'
    var_6 = bool('null' in str(e).lower())
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'uuid'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = '550e8400-e29b-41d4-a716-446655440000'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'
    var_6 = bool('type' in str(e).lower())
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    var_4 = '\x00'
    var_5 = bool('\x00' not in var_3)
    assert var_5 is True
    var_6 = 'helloworld'
    var_7 = bool('helloworld' in var_3)
    assert var_7 is True

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
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = '   '
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'blank'
    var_7 = bool('blank' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(coerce_types=var_0, **var_2)
    var_4 = '   '
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_length'
    var_7 = bool('min_length' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcdefgh'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'max_length'
    var_7 = bool('max_length' in str(e).lower())
    assert var_7 is True

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
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'pattern'
    var_7 = bool('pattern' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '12345'
    var_4 = var_2.validate(var_3)
    assert var_4 == '12345'

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
    var_2 = 'hello world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello world'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_union_validate_returns_first_matching_type. Retrieved 6/8 statements.


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
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = [var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
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
    var_7 = 42
    var_8 = var_6.validate(var_7)
    assert var_8 == 42

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
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'union'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Union(var_5, **var_6)
    var_8 = 'this is a very long string'
    var_9 = var_7.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

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



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_array_init_min_items_not_none. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 5



# Parsed testcases at query #75
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = '\n    Test that the predicate at line 33 evaluates to False.\n    Line 33: elif isinstance(self.additional_items, Field):\n    \n    This evaluates to False when additional_items is False (a boolean).\n    This occurs when self.items is a list, pos >= len(self.items),\n    and self.additional_items is False (not a Field instance).\n    '
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = [var_2]
    var_4 = False
    var_5 = {}
    var_6 = module_0.Array(var_3, var_4, **var_5)
    var_7 = 'hello'
    var_8 = 'world'
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)



# Parsed testcases at query #76
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'apple'
    var_6 = 'banana'
    var_7 = 'cherry'
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == ['apple', 'banana', 'cherry'])
    assert var_10 is True



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_numeric_type_int_with_non_integer_float. Retrieved 3/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 3.14
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #78
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra_key'
    var_5 = 'extra_value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'extra_key': 'extra_value'})
    assert var_8 is True



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_no_exception_in_try_block. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 42



# Parsed testcases at query #80
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3, var_3]
    var_5 = var_2.validate(var_4)
    var_6 = len(e.messages())
    var_7 = bool(len(e.messages()) > 0)
    assert var_7 is True



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_union_predicate_line_17_evaluates_true. Retrieved 6/48 statements.


def test_case_0():
    var_0 = 'type'
    var_1 = 'test_value'
    var_2 = len(e.messages())
    assert var_2 == 1
    var_3 = e.messages()[0].code
    assert var_3 == 'union'
    var_4 = 'custom_error'
    var_5 = 'test_value'
    var_6 = len(e.messages())
    assert var_6 == 1
    var_7 = e.messages()[0].code
    assert var_7 == 'custom_error'
    var_8 = 0
    var_9 = 'test_value'
    var_10 = len(e.messages())
    assert var_10 == 1
    var_11 = e.messages()[0].code
    assert var_11 == 'type'
    var_12 = e.messages()[0].index
    assert var_12 == 0



# Parsed testcases at query #82
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'Option 1'
    var_2 = (var_0, var_1)
    var_3 = 'option2'
    var_4 = 'Option 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.validate(var_0)
    assert var_11 == 'option1'



# Parsed testcases at query #83
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = var_5.validate(var_9)
    var_11 = bool(var_10 == [1, 2, 3])
    assert var_11 is True



# Parsed testcases at query #84
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'Option 1'
    var_2 = (var_0, var_1)
    var_3 = 'option2'
    var_4 = 'Option 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.validate(var_0)
    assert var_11 == 'option1'



# Parsed testcases at query #85
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #86
#--------------------------




import typesystem.fields as module_0
import typesystem.unique as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.Uniqueness()
    var_6 = 'duplicate'
    var_7 = var_5.add(var_6)
    var_8 = var_6 in var_5
    assert var_8 is True



# Parsed testcases at query #87
#--------------------------






# Parsed testcases at query #88
#--------------------------

# Partially parsed test_numeric_type_int_with_non_integer_float. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_array_init_min_items_not_none. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 5



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_union_validate_predicate_at_line_17_evaluates_to_true. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'type'
    var_1 = None
    var_2 = 'test_value'



# Parsed testcases at query #91
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Boolean(coerce_types=var_0, **var_3)
    var_5 = 'true'
    var_6 = var_4.validate(var_5)
    assert var_6 is True



# Parsed testcases at query #92
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
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'null'
    var_6 = bool('null' in str(e).lower())
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'
    var_6 = bool('type' in str(e).lower())
    assert var_6 is True

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
    var_6 = 'blank'
    var_7 = bool('blank' in str(e).lower())
    assert var_7 is True

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
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_length'
    var_7 = bool('min_length' in str(e).lower())
    assert var_7 is True

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
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'max_length'
    var_7 = bool('max_length' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '12345'
    var_4 = var_2.validate(var_3)
    assert var_4 == '12345'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'pattern'
    var_7 = bool('pattern' in str(e).lower())
    assert var_7 is True

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = 'hello'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello'

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = 'Hello123'
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'pattern'
    var_8 = bool('pattern' in str(e).lower())
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 5
    var_2 = {}
    var_3 = module_0.String(trim_whitespace=var_0, min_length=var_1, **var_2)
    var_4 = '  hello world  '
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello world'
    var_6 = len(var_5)
    var_7 = bool(var_6 >= 5)
    assert var_7 is True



# Parsed testcases at query #93
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'hello'
    var_6 = 'world'
    var_7 = [var_5, var_6]
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == ['hello', 'world'])
    assert var_9 is True



# Parsed testcases at query #94
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    assert var_3 == 42



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_union_validate_predicate_line_18_evaluates_true. Retrieved 2/14 statements.


def test_case_0():
    var_0 = None
    var_1 = 'test_value'



####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
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
    var_5 = 'invalid'
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
    var_7 = 'hello'
    var_8 = 1
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
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

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
    var_3 = False
    var_4 = [var_0, var_0, var_3, var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [True, 1, False, 0])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 'not_int'
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = 4
    var_2 = {}
    var_3 = module_0.Array(min_items=var_0, max_items=var_1, **var_2)
    var_4 = 1
    var_5 = 3
    var_6 = [var_4, var_0, var_5]
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == [1, 2, 3])
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3, var_0]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [1, 2])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3, var_0]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [1, 2])
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/9 statements.
# Partially parsed test_serialize_with_list_custom_serializers. Retrieved 3/14 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None

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
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.serialize(var_7)
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
    var_7 = 42
    var_8 = 'hello'
    var_9 = [var_7, var_8]
    var_10 = var_6.serialize(var_9)
    var_11 = bool(var_10 == [42, 'hello'])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = [var_1, var_3, var_5]
    var_7 = {}
    var_8 = module_0.Array(var_6, **var_7)
    var_9 = 42
    var_10 = 'hello'
    var_11 = 99
    var_12 = [var_9, var_10, var_11]
    var_13 = var_8.serialize(var_12)
    var_14 = bool(var_13 == [42, 'hello', 99])
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = []
    var_4 = var_2.serialize(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'Hello'
    var_1 = 'WORLD'
    var_2 = [var_0, var_1]

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = {}
    var_5 = module_0.Array(var_3, **var_4)
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = 3
    var_10 = 4
    var_11 = [var_9, var_10]
    var_12 = [var_8, var_11]
    var_13 = var_5.serialize(var_12)
    var_14 = bool(var_13 == [[1, 2], [3, 4]])
    assert var_14 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_float_when_integer_required. Retrieved 1/4 statements.
# Partially parsed test_validate_string_to_decimal. Retrieved 2/6 statements.
# Partially parsed test_validate_numeric_type_conversion. Retrieved 1/4 statements.
# Partially parsed test_validate_decimal_numeric_type. Retrieved 2/8 statements.


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
    var_7 = 'null'

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
    var_5 = 'type'

def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'integer'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    assert var_4 == 123

def test_case_0():
    var_0 = True
    var_1 = '123.45'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'not_a_number'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '-inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

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
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'minimum'

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
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exclusive_minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 100

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 101
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 99
    var_4 = var_2.validate(var_3)
    assert var_4 == 99

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 100
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exclusive_maximum'

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 16
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'

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
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 2.3
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'

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
    var_0 = '0.1'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.25
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 3.2 or var_4 == 3.3)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 50
    var_6 = var_4.validate(var_5)
    assert var_6 == 50

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 55
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'multiple_of'

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 42.5
    var_1 = '42.5'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_exception_handler_not_triggered. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 42



# Parsed testcases at query #5
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
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hel\x00lo'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

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
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

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
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hello'
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
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello'
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
    var_3 = 'Hello123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = '12345'
    var_5 = var_3.validate(var_4)
    assert var_5 == '12345'



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = 'option3'
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.choices
    var_7 = bool(var_5.choices == [('option1', 'option1'), ('option2', 'option2'), ('option3', 'option3')])
    assert var_7 is True
    var_8 = var_5.coerce_types
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'display1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'display2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.choices
    var_10 = bool(var_8.choices == [('key1', 'display1'), ('key2', 'display2')])
    assert var_10 is True
    var_11 = var_8.coerce_types
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'key2'
    var_2 = 'display2'
    var_3 = (var_1, var_2)
    var_4 = 'option3'
    var_5 = [var_0, var_3, var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = var_7.choices
    var_9 = bool(var_7.choices == [('option1', 'option1'), ('key2', 'display2'), ('option3', 'option3')])
    assert var_9 is True

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
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Choice(choices=var_2, coerce_types=var_3, **var_4)
    var_6 = var_5.coerce_types
    assert var_6 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = [var_0]
    var_2 = 'My Choice'
    var_3 = 'Choose one'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.Choice(choices=var_1, **var_6)
    var_8 = var_7.title
    assert var_8 == 'My Choice'
    var_9 = var_7.description
    assert var_9 == 'Choose one'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = 'default'
    var_4 = {var_3: var_0}
    var_5 = module_0.Choice(choices=var_2, **var_4)
    var_6 = var_5.default
    assert var_6 == 'option1'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Choice(choices=var_1, **var_4)
    var_6 = var_5.allow_null
    assert var_6 is True
    var_7 = var_5.default
    assert var_7 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_0.Choice(choices=var_1, **var_4)
    var_6 = var_5.read_only
    assert var_6 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_object_validate_with_mapping_type. Retrieved 5/9 statements.


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
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = 'John'
    var_10 = '30'
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = bool(var_12 == {'name': 'John', 'age': '30'})
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

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
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
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
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'extra'
    var_8 = 'John'
    var_9 = 'field'
    var_10 = {var_0: var_8, var_7: var_9}
    var_11 = var_6.validate(var_10)
    var_12 = bool(var_11 == {'name': 'John', 'extra': 'field'})
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
    var_8 = 'extra'
    var_9 = 'John'
    var_10 = 'field'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = len(e.messages())
    var_15 = bool(len(e.messages()) > 0)
    assert var_15 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = 'extra'
    var_9 = 'John'
    var_10 = 'field'
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(var_12 == {'name': 'John', 'extra': 'field'})
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Unknown'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'Unknown'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'S_key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'S_key': 'value'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = 'key'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'age'
    var_7 = 'not_an_integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = len(e.messages())
    var_12 = bool(len(e.messages()) > 0)
    assert var_12 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_object_validate_with_non_string_key. Retrieved 5/8 statements.
# Partially parsed test_object_validate_with_required_field_missing. Retrieved 7/11 statements.
# Partially parsed test_object_validate_with_min_properties. Retrieved 6/9 statements.
# Partially parsed test_object_validate_with_max_properties. Retrieved 8/11 statements.
# Partially parsed test_object_validate_with_invalid_property_value. Retrieved 8/12 statements.
# Partially parsed test_object_validate_with_additional_properties_false. Retrieved 11/15 statements.
# Partially parsed test_object_validate_with_property_names. Retrieved 7/11 statements.
# Partially parsed test_object_validate_with_min_properties_one. Retrieved 4/7 statements.
# Partially parsed test_object_validate_with_mapping_type. Retrieved 5/9 statements.


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
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = len(e.messages())
    var_6 = bool(len(e.messages()) > 0)
    assert var_6 is True

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
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = 'John'
    var_8 = {var_0: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = bool(var_9 == {'name': 'John'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(var_8 == {'key1': 'value1', 'key2': 'value2'})
    assert var_9 is True

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
    var_8 = 'extra'
    var_9 = 'John'
    var_10 = 'field'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'key1'
    var_5 = 'key2'
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = var_3.validate(var_8)
    var_10 = bool(var_9 == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Unknown'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'Unknown'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'S_name'
    var_7 = 'other'
    var_8 = 'John'
    var_9 = 'value'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = 'S_name'
    var_13 = bool('S_name' in var_11)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'verylongkey'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'May not be null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'option1'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = 'option3'
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Not a valid choice'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, coerce_types=var_3, **var_5)
    var_7 = ''
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'option2'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, coerce_types=var_3, **var_5)
    var_7 = ''
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'This field is required'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'display1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'display2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    assert var_9 == 'key1'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'key2'
    var_2 = 'display2'
    var_3 = (var_1, var_2)
    var_4 = [var_0, var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)
    var_7 = var_6.validate(var_0)
    assert var_7 == 'option1'
    var_8 = var_6.validate(var_1)
    assert var_8 == 'key2'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 == 1

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 is True
    var_6 = var_4.validate(var_1)
    assert var_6 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_object_constructor_allow_null_sets_default_none. Retrieved 2/3 statements.


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
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = var_4.properties
    var_6 = bool(var_4.properties == var_2)
    assert var_6 is True
    var_7 = var_4.pattern_properties
    var_8 = bool(var_4.pattern_properties == {})
    assert var_8 is True
    var_9 = var_4.additional_properties
    assert var_9 is True
    var_10 = var_4.required
    var_11 = bool(var_4.required == [])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^S_'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = var_4.pattern_properties
    var_6 = bool(var_4.pattern_properties == var_2)
    assert var_6 is True
    var_7 = var_4.properties
    var_8 = bool(var_4.properties == {})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Object(required=var_2, **var_3)
    var_5 = var_4.required
    var_6 = bool(var_4.required == ['name', 'email'])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = module_0.Object(required=var_2, **var_3)
    var_5 = var_4.required
    var_6 = bool(var_4.required == ['name', 'email'])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {}
    var_3 = module_0.Object(min_properties=var_0, max_properties=var_1, **var_2)
    var_4 = var_3.min_properties
    assert var_4 == 1
    var_5 = var_3.max_properties
    assert var_5 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = var_2.additional_properties
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = var_2.additional_properties
    var_4 = bool(var_2.additional_properties is var_0)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Object(property_names=var_0, **var_1)
    var_3 = var_2.property_names
    var_4 = bool(var_2.property_names is var_0)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 'A user object'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Object(**var_4)
    var_6 = var_5.title
    assert var_6 == 'User'
    var_7 = var_5.description
    assert var_7 == 'A user object'

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
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.Object(**var_4)
    var_6 = var_5.default
    var_7 = bool(var_5.default == var_2)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = var_3.default
    assert var_4 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = var_4.properties
    var_6 = bool(var_4.properties == var_2)
    assert var_6 is True
    var_7 = var_4.properties
    var_8 = bool(var_4.properties is not var_2)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = '^S_'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Object(pattern_properties=var_2, **var_3)
    var_5 = var_4.pattern_properties
    var_6 = bool(var_4.pattern_properties == var_2)
    assert var_6 is True
    var_7 = var_4.pattern_properties
    var_8 = bool(var_4.pattern_properties is not var_2)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Object(required=var_2, **var_3)
    var_5 = var_4.required
    var_6 = bool(var_4.required == ['name', 'email'])
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = '^S_'
    var_4 = {var_3: var_0}
    var_5 = [var_1]
    var_6 = 'User'
    var_7 = 'A user object'
    var_8 = 'John'
    var_9 = {var_1: var_8}
    var_10 = False
    var_11 = 1
    var_12 = 10
    var_13 = 'title'
    var_14 = 'description'
    var_15 = 'default'
    var_16 = 'allow_null'
    var_17 = 'read_only'
    var_18 = {var_13: var_6, var_14: var_7, var_15: var_9, var_16: var_10, var_17: var_10}
    var_19 = module_0.Object(properties=var_2, pattern_properties=var_4, additional_properties=var_10, property_names=var_0, min_properties=var_11, max_properties=var_12, required=var_5, **var_18)
    var_20 = var_19.title
    assert var_20 == 'User'
    var_21 = var_19.description
    assert var_21 == 'A user object'
    var_22 = var_19.default
    var_23 = bool(var_19.default == {'name': 'John'})
    assert var_23 is True
    var_24 = var_19.allow_null
    assert var_24 is False
    var_25 = var_19.read_only
    assert var_25 is False
    var_26 = var_19.properties
    var_27 = bool(var_19.properties == var_2)
    assert var_27 is True
    var_28 = var_19.pattern_properties
    var_29 = bool(var_19.pattern_properties == var_4)
    assert var_29 is True
    var_30 = var_19.additional_properties
    assert var_30 is False
    var_31 = var_19.property_names
    var_32 = bool(var_19.property_names is var_0)
    assert var_32 is True
    var_33 = var_19.min_properties
    assert var_33 == 1
    var_34 = var_19.max_properties
    assert var_34 == 10
    var_35 = var_19.required
    var_36 = bool(var_19.required == var_5)
    assert var_36 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_format_is_native_type_returns_value. Retrieved 4/18 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_format'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 12345
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == var_3)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_returns_first_matching_type. Retrieved 6/8 statements.


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
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'null'

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
    var_7 = 42
    var_8 = var_6.validate(var_7)
    assert var_8 == 42

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
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'union'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = [var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
    var_9 = var_8.allow_null
    assert var_9 is True
    var_10 = None
    var_11 = var_8.validate(var_10)
    assert var_11 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 123
    var_8 = var_6.validate(var_7)
    assert var_8 == 123

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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_float_when_int_required. Retrieved 1/4 statements.
# Partially parsed test_validate_float_when_int_required_but_is_integer. Retrieved 1/3 statements.
# Partially parsed test_validate_string_to_float. Retrieved 3/7 statements.
# Partially parsed test_validate_decimal_input. Retrieved 2/6 statements.


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
    var_7 = 'null'

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
    var_5 = 'type'

def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'integer'

def test_case_0():
    var_0 = 3.0

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    assert var_4 == 123

def test_case_0():
    var_0 = True
    var_1 = '3.14'
    var_2 = 3.14

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '-inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

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
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'minimum'

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
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exclusive_minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 100

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 101
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 99
    var_4 = var_2.validate(var_3)
    assert var_4 == 99

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 100
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exclusive_maximum'

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 13
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'

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
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 2.3
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.14159
    var_4 = var_2.validate(var_3)
    var_5 = 3.14
    var_6 = var_4 - var_5
    var_7 = abs(var_6)
    var_8 = bool(var_7 < 0.001)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'not_a_number'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '123.45'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 100
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 50
    var_6 = var_4.validate(var_5)
    assert var_6 == 50

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 100
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 13
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'multiple_of'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_get_default_value_with_callable_default. Retrieved 1/5 statements.
# Partially parsed test_get_default_value_with_callable_returning_none. Retrieved 1/5 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 42
    var_2 = module_0.Field(title=var_0, default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 42

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field(title=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = module_0.Field(title=var_0, allow_null=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'default_string'
    var_2 = module_0.Field(title=var_0, default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 'default_string'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'test'
    var_5 = module_0.Field(title=var_4, default=var_3)
    var_6 = var_5.get_default_value()
    var_7 = bool(var_6 == var_3)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'test'
    var_4 = module_0.Field(title=var_3, default=var_2)
    var_5 = var_4.get_default_value()
    var_6 = bool(var_5 == var_2)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = module_0.Field(title=var_0, default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 == 0

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = module_0.Field(title=var_0, default=var_1)
    var_3 = var_2.get_default_value()
    assert var_3 is False



# Parsed testcases at query #15
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
    var_8 = bool('null' in str(e).lower())
    assert var_8 is True

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
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'false'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'on'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'off'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '1'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '0'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is False

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
    var_3 = 0
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'TRUE'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'FaLsE'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

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
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'null'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'none'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------




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

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = var_2.items
    var_4 = bool(var_2.items is var_0)
    assert var_4 is True
    var_5 = var_2.additional_items
    assert var_5 is False

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
    var_7 = var_4.min_items
    assert var_7 == 2
    var_8 = var_4.max_items
    assert var_8 == 2

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
    var_9 = bool(var_5.additional_items is var_2)
    assert var_9 is True
    var_10 = var_5.min_items
    assert var_10 == 2
    var_11 = var_5.max_items
    assert var_11 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = module_0.Array(var_2, **var_3)
    var_5 = var_4.items
    var_6 = bool(var_4.items == [var_0, var_1])
    assert var_6 is True
    var_7 = var_4.min_items
    assert var_7 == 2
    var_8 = var_4.max_items
    assert var_8 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 5
    var_2 = {}
    var_3 = module_0.Array(var_0, min_items=var_1, **var_2)
    var_4 = var_3.min_items
    assert var_4 == 5
    var_5 = var_3.max_items
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 10
    var_2 = {}
    var_3 = module_0.Array(var_0, max_items=var_1, **var_2)
    var_4 = var_3.max_items
    assert var_4 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 7
    var_2 = {}
    var_3 = module_0.Array(var_0, exact_items=var_1, **var_2)
    var_4 = var_3.min_items
    assert var_4 == 7
    var_5 = var_3.max_items
    assert var_5 == 7

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = {}
    var_3 = module_0.Array(var_0, unique_items=var_1, **var_2)
    var_4 = var_3.unique_items
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'Test Array'
    var_2 = 'A test array'
    var_3 = 'title'
    var_4 = 'description'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Array(var_0, **var_5)
    var_7 = var_6.title
    assert var_7 == 'Test Array'
    var_8 = var_6.description
    assert var_8 == 'A test array'

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Array(var_0, **var_3)
    var_5 = var_4.allow_null
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = True
    var_2 = 'read_only'
    var_3 = {var_2: var_1}
    var_4 = module_0.Array(var_0, **var_3)
    var_5 = var_4.read_only
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'default'
    var_6 = {var_5: var_4}
    var_7 = module_0.Array(var_0, **var_6)
    var_8 = var_7.default
    var_9 = bool(var_7.default == var_4)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = {}
    var_3 = module_0.Array(var_0, var_1, **var_2)
    var_4 = var_3.additional_items
    var_5 = bool(var_3.additional_items is var_1)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 2
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Array(var_0, min_items=var_1, max_items=var_2, **var_3)
    var_5 = var_4.min_items
    assert var_5 == 2
    var_6 = var_4.max_items
    assert var_6 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = {}
    var_5 = module_0.Array(var_2, min_items=var_3, **var_4)
    var_6 = var_5.min_items
    assert var_6 == 5
    var_7 = var_5.max_items
    assert var_7 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = 10
    var_4 = {}
    var_5 = module_0.Array(var_2, max_items=var_3, **var_4)
    var_6 = var_5.min_items
    assert var_6 == 2
    var_7 = var_5.max_items
    assert var_7 == 10



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'Option 1'
    var_2 = (var_0, var_1)
    var_3 = 'option2'
    var_4 = 'Option 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = 'invalid_value'
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_get_default_value_with_callable_default.




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
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3, var_0]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [1, 2])
    assert var_6 is True

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
    var_1 = 3
    var_2 = {}
    var_3 = module_0.Array(min_items=var_0, max_items=var_1, **var_2)
    var_4 = 2
    var_5 = [var_0, var_4]
    var_6 = var_3.validate(var_5)
    var_7 = bool(var_6 == [1, 2])
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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = 1
    var_10 = 'hello'
    var_11 = 42
    var_12 = [var_9, var_10, var_11]
    var_13 = var_8.validate(var_12)
    var_14 = bool(var_13 == [1, 'hello', 42])
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
    var_3 = True
    var_4 = [var_3, var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 'not_int'
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_raises_error_when_value_is_non_integer_float_and_numeric_type_is_int. Retrieved 1/4 statements.
# Partially parsed test_validate_converts_string_to_number. Retrieved 1/3 statements.
# Partially parsed test_validate_with_precision. Retrieved 2/4 statements.
# Partially parsed test_validate_returns_valid_integer. Retrieved 1/3 statements.
# Partially parsed test_validate_returns_valid_float. Retrieved 1/3 statements.
# Partially parsed test_validate_with_decimal_type. Retrieved 1/3 statements.
# Partially parsed test_validate_coerces_float_string_to_int. Retrieved 2/4 statements.


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
    var_7 = bool(True)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = '123'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'not_a_number'
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
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
    var_6 = bool(True)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

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
    var_6 = bool(True)
    assert var_6 is True

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
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 150
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 100

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 100
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 99
    var_4 = var_2.validate(var_3)
    assert var_4 == 99

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 7
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    assert var_4 == 10

def test_case_0():
    var_0 = '0.01'
    var_1 = 3.146

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 3.14

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = True
    var_1 = '42'



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'Option 1'
    var_2 = (var_0, var_1)
    var_3 = 'option2'
    var_4 = 'Option 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.validate(var_0)
    assert var_9 == 'option1'



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    assert var_3 == 42



# Parsed testcases at query #23
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
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'allow_null'
    var_9 = {var_8: var_6}
    var_10 = module_0.String(allow_blank=var_5, coerce_types=var_5, **var_9)
    var_11 = ''
    var_12 = var_10.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'allow_null'
    var_15 = {var_14: var_11}
    var_16 = module_0.String(allow_blank=var_11, coerce_types=var_11, **var_15)
    var_17 = ''
    var_18 = var_16.validate(var_17)
    var_19 = bool(False)
    assert var_19 is True



# Parsed testcases at query #24
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
    var_8 = bool('null' in str(e).lower())
    assert var_8 is True

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
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'false'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'on'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'off'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '1'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '0'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is False

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
    var_3 = 0
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'TRUE'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'FaLsE'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

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
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = True
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'null'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'none'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 1.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'apple'
    var_6 = 'banana'
    var_7 = 'cherry'
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == ['apple', 'banana', 'cherry'])
    assert var_10 is True



# Parsed testcases at query #26
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Choice(choices=var_0, **var_1)
    var_3 = var_2.choices
    var_4 = bool(var_2.choices == [])
    assert var_4 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_string_constructor_allow_blank_sets_default. Retrieved 2/3 statements.
# Partially parsed test_string_constructor_with_allow_null. Retrieved 2/3 statements.
# Partially parsed test_string_constructor_inherits_from_field. Retrieved 1/2 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = var_1.allow_blank
    assert var_2 is False
    var_3 = var_1.trim_whitespace
    assert var_3 is True
    var_4 = var_1.max_length
    assert var_4 is None
    var_5 = var_1.min_length
    assert var_5 is None
    var_6 = var_1.pattern
    assert var_6 is None
    var_7 = var_1.pattern_regex
    assert var_7 is None
    var_8 = var_1.format
    assert var_8 is None
    var_9 = var_1.coerce_types
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 100
    var_3 = 5
    var_4 = 'email'
    var_5 = {}
    var_6 = module_0.String(allow_blank=var_0, trim_whitespace=var_1, max_length=var_2, min_length=var_3, format=var_4, coerce_types=var_1, **var_5)
    var_7 = var_6.allow_blank
    assert var_7 is True
    var_8 = var_6.trim_whitespace
    assert var_8 is False
    var_9 = var_6.max_length
    assert var_9 == 100
    var_10 = var_6.min_length
    assert var_10 == 5
    var_11 = var_6.format
    assert var_11 == 'email'
    var_12 = var_6.coerce_types
    assert var_12 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = var_2.pattern
    assert var_3 == '^\\d+$'
    var_4 = var_2.pattern_regex
    var_5 = bool(var_2.pattern_regex is not None)
    assert var_5 is True

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^\\w+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = var_3.pattern
    assert var_4 == '^\\w+$'
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
    var_1 = 'test'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, **var_3)
    var_5 = var_4.default
    assert var_5 == 'test'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Username'
    var_1 = "User's login name"
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.String(**var_4)
    var_6 = var_5.title
    assert var_6 == 'Username'
    var_7 = var_5.description
    assert var_7 == "User's login name"

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True
    var_5 = var_3.default
    assert var_5 is None

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
    var_0 = {}
    var_1 = module_0.String(**var_0)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Email'
    var_1 = 'User email address'
    var_2 = 'test@example.com'
    var_3 = False
    var_4 = True
    var_5 = 255
    var_6 = 5
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
    assert var_16 == 'Email'
    var_17 = var_15.description
    assert var_17 == 'User email address'
    var_18 = var_15.default
    assert var_18 == 'test@example.com'
    var_19 = var_15.allow_null
    assert var_19 is False
    var_20 = var_15.read_only
    assert var_20 is False
    var_21 = var_15.allow_blank
    assert var_21 is False
    var_22 = var_15.trim_whitespace
    assert var_22 is True
    var_23 = var_15.max_length
    assert var_23 == 255
    var_24 = var_15.min_length
    assert var_24 == 5
    var_25 = var_15.format
    assert var_25 == 'email'
    var_26 = var_15.coerce_types
    assert var_26 is True



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = (var_0, var_1, var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 (self.allow_null and self.coerce_types) evaluates to False'
    var_1 = False
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_1}
    var_5 = module_0.String(allow_blank=var_1, coerce_types=var_2, **var_4)
    var_6 = '   '
    var_7 = var_5.validate(var_6)
    var_8 = var_5.allow_null and var_5.coerce_types
    assert var_8 is False



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_string_validate_format_is_native_type. Retrieved 5/19 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'custom'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test_string'
    var_4 = {}
    var_5 = module_0.String(format=var_0, **var_4)
    var_6 = var_5.validate(var_3)
    var_7 = bool(var_6 == var_3)
    assert var_7 is True



# Parsed testcases at query #31
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
    var_7 = 'May not be null'

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
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'false'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'on'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'off'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '1'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '0'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is False

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
    var_3 = 0
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'TRUE'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'null'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'none'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Must be a boolean'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Must be a boolean'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 1
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Must be a boolean'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Must be a boolean'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'FaLsE'
    var_4 = var_2.validate(var_3)
    assert var_4 is False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_union_validate_predicate_line_18_true. Retrieved 7/47 statements.


def test_case_0():
    var_0 = 'not_type'
    var_1 = None
    var_2 = 'test_value'
    var_3 = 'type'
    var_4 = 0
    var_5 = 'test_value'
    var_6 = 'test_value'



# Parsed testcases at query #33
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)



# Parsed testcases at query #34
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_get_default_value_with_callable_default.




# Parsed testcases at query #36
#--------------------------

# Partially parsed test_union_predicate_line_17_evaluates_to_true. Retrieved 2/17 statements.
# Partially parsed test_union_predicate_line_17_true_with_different_code. Retrieved 2/18 statements.
# Partially parsed test_union_predicate_line_17_true_with_index. Retrieved 2/18 statements.


def test_case_0():
    var_0 = None
    var_1 = 'test_value'

def test_case_0():
    var_0 = None
    var_1 = 'test_value'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = None
    var_1 = 'test_value'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #37
#--------------------------

# Failed to parse test_get_default_value_with_callable_default.




# Parsed testcases at query #38
#--------------------------

# Partially parsed test_union_validate_predicate_line_18_true. Retrieved 8/50 statements.


def test_case_0():
    var_0 = 'custom_error'
    var_1 = None
    var_2 = 'test_value'
    var_3 = 'type'
    var_4 = 0
    var_5 = 'test_value'
    var_6 = 'custom'
    var_7 = 'test_value'
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #39
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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'

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
    var_3 = '   '
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'blank'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(coerce_types=var_0, **var_2)
    var_4 = '   '
    var_5 = var_3.validate(var_4)
    assert var_5 is None

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
    var_0 = 5
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
    var_3 = 'hello world'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'max_length'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

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
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'pattern'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '12345'
    var_4 = var_2.validate(var_3)
    assert var_4 == '12345'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello world'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = False
    var_3 = {}
    var_4 = module_0.String(allow_blank=var_2, max_length=var_1, min_length=var_0, **var_3)
    var_5 = 'hello'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'hello'



# Parsed testcases at query #40
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = '   '
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'blank'
    var_9 = bool('blank' in str(e).lower())
    assert var_9 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_union_predicate_line_17_true. Retrieved 5/21 statements.


def test_case_0():
    var_0 = 'not_type'
    var_1 = None
    var_2 = 'another'
    var_3 = 'valid_value'
    var_4 = 'test_value'



# Parsed testcases at query #42
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_validate_float_when_integer_required. Retrieved 1/4 statements.
# Partially parsed test_validate_valid_integer. Retrieved 1/3 statements.
# Partially parsed test_validate_string_to_integer_with_coerce. Retrieved 2/4 statements.
# Partially parsed test_validate_float_to_float. Retrieved 1/3 statements.
# Partially parsed test_validate_decimal_type. Retrieved 1/5 statements.


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
    var_7 = 'null'

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
    var_5 = 'type'

def test_case_0():
    var_0 = 3.5
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'integer'

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = True
    var_1 = '42'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'minimum'

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
    var_6 = 'exclusive_minimum'

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
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 150
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 100

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 100
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exclusive_maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 99
    var_4 = var_2.validate(var_3)
    assert var_4 == 99

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 13
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'

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
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 2.3
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'

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
    var_0 = 3.14

def test_case_0():
    var_0 = '3.14'

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
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '42'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'



# Parsed testcases at query #44
#--------------------------




import typesystem.fields as module_0
import typesystem.unique as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.Uniqueness()
    var_6 = 'duplicate'
    var_7 = var_5.add(var_6)
    var_8 = var_6 in var_5
    assert var_8 is True



# Parsed testcases at query #45
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'option1'
    var_1 = 'Option 1'
    var_2 = (var_0, var_1)
    var_3 = 'option2'
    var_4 = 'Option 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = 'invalid_value'
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #46
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True



# Parsed testcases at query #47
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
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(allow_blank=var_0, **var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'null'
    var_8 = bool('null' in str(e).lower())
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'
    var_6 = bool('type' in str(e).lower())
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'helloworld'

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
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'blank'
    var_7 = bool('blank' in str(e).lower())
    assert var_7 is True

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
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(allow_blank=var_1, coerce_types=var_0, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hi'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'min_length'
    var_7 = bool('min_length' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello world'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'max_length'
    var_7 = bool('max_length' in str(e).lower())
    assert var_7 is True

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
    var_0 = '^[0-9]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'pattern'
    var_7 = bool('pattern' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[0-9]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '12345'
    var_4 = var_2.validate(var_3)
    assert var_4 == '12345'

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = 'abc'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'abc'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_validate_float_with_int_numeric_type_not_integer. Retrieved 1/4 statements.
# Partially parsed test_validate_float_with_int_numeric_type_is_integer. Retrieved 1/3 statements.
# Partially parsed test_validate_numeric_type_conversion. Retrieved 1/4 statements.
# Partially parsed test_validate_decimal_input. Retrieved 2/5 statements.


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
    var_7 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'

def test_case_0():
    var_0 = 3.14
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'integer'

def test_case_0():
    var_0 = 3.0

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    assert var_4 == 123

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'not_a_number'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'nan'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'finite'

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
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'minimum'

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
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exclusive_minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 100

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 101
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 99
    var_4 = var_2.validate(var_3)
    assert var_4 == 99

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 100
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'exclusive_maximum'

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 16
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'

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
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 2.3
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'multiple_of'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 3.146
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 3.15)
    assert var_5 is True

def test_case_0():
    var_0 = 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 10
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 50
    var_6 = var_4.validate(var_5)
    assert var_6 == 50

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '123.45'



# Parsed testcases at query #49
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'apple'
    var_6 = 'banana'
    var_7 = 'cherry'
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == ['apple', 'banana', 'cherry'])
    assert var_10 is True



# Parsed testcases at query #50
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = '  '
    var_6 = var_4.validate(var_5)



# Parsed testcases at query #51
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
    var_8 = bool('null' in str(e).lower())
    assert var_8 is True

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
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'false'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'on'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'off'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '1'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '0'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is False

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
    var_3 = 0
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'TRUE'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'FaLsE'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

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
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 1.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True



# Parsed testcases at query #52
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test that line 6 predicate evaluates to True when value is not in choices.'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = False
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_0.Choice(choices=var_4, **var_7)
    var_9 = 'd'
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_line_11_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 3.5
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_union_validate_predicate_line_18_true. Retrieved 2/21 statements.


def test_case_0():
    var_0 = None
    var_1 = 'test_value'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_numeric_type_int_with_non_integer_float. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 3.5
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_object_validate_mapping_type. Retrieved 5/9 statements.


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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(var_5 == {'key': 'value'})
    assert var_6 is True

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
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = 'John'
    var_8 = {var_0: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = bool(var_9 == {'name': 'John'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
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
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

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
    var_0 = {}
    var_1 = False
    var_2 = {}
    var_3 = module_0.Object(properties=var_0, additional_properties=var_1, **var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    var_10 = bool(len(e.messages()) > 0)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(additional_properties=var_2, **var_3)
    var_5 = 'key'
    var_6 = 'val'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'key': 'val'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(additional_properties=var_2, **var_3)
    var_5 = 'key'
    var_6 = 'toolong'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'S_name'
    var_7 = 'John'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'S_name': 'John'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = 2
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, **var_2)
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.Object(pattern_properties=var_4, **var_5)
    var_7 = 'S_name'
    var_8 = 'toolong'
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = len(e.messages())
    var_13 = bool(len(e.messages()) > 0)
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'name'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'name': 'value'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'Name'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'John'})
    assert var_10 is True

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
    var_10 = {}
    var_11 = module_0.Object(properties=var_6, min_properties=var_8, max_properties=var_9, required=var_7, **var_10)
    var_12 = 'John'
    var_13 = 30
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(var_15 == {'name': 'John', 'age': 30})
    assert var_16 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_union_predicate_line_17_true. Retrieved 6/25 statements.


def test_case_0():
    var_0 = 'validate_or_error'
    var_1 = 'allow_null'
    var_2 = [var_0, var_1]
    var_3 = [var_0, var_1]
    var_4 = None
    var_5 = 'test_value'



# Parsed testcases at query #58
#--------------------------






# Parsed testcases at query #59
#--------------------------

# Partially parsed test_validate_float_when_numeric_type_is_int. Retrieved 1/4 statements.
# Partially parsed test_validate_integer_when_numeric_type_is_int. Retrieved 1/3 statements.
# Partially parsed test_validate_decimal_input. Retrieved 2/6 statements.
# Partially parsed test_validate_numeric_type_conversion. Retrieved 1/4 statements.


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
    var_0 = 3.5
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 3.0

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
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123.45'
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 123.45)
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
    var_2 = '-inf'
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
    var_3 = 15
    var_4 = var_2.validate(var_3)
    assert var_4 == 15

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
    var_3 = 15
    var_4 = var_2.validate(var_3)
    assert var_4 == 15

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
    var_3 = 50
    var_4 = var_2.validate(var_3)
    assert var_4 == 50

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 150
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 100

import typesystem.fields as module_0

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 50
    var_4 = var_2.validate(var_3)
    assert var_4 == 50

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
    var_3 = 15
    var_4 = var_2.validate(var_3)
    assert var_4 == 15

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 13
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

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
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 2.3
    var_4 = var_2.validate(var_3)
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
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = '123.45'

def test_case_0():
    var_0 = '123'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 100
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 50
    var_6 = var_4.validate(var_5)
    assert var_6 == 50

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 100
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 5
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 100
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 150
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 100
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Number(minimum=var_0, maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = 53
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_numeric_type_int_with_non_integer_float_raises_integer_error. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 3.5
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_object_validate_with_mapping_type. Retrieved 5/8 statements.


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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(var_5 == {'key': 'value'})
    assert var_6 is True

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

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
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
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(var_8 == {'key1': 'value1', 'key2': 'value2'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'key1'
    var_4 = 'value1'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'key1'
    var_5 = 'value1'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'key1': 'value1'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'abc'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'abc': 'value'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'John'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = ''
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_0.Object(properties=var_7, **var_8)
    var_10 = 'John'
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = var_9.validate(var_12)
    var_14 = bool(var_13 == {'user': {'name': 'John'}})
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_0.Object(properties=var_7, **var_8)
    var_10 = 'user'
    var_11 = 'age'
    var_12 = 'not an integer'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = var_9.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True

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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = 'key'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_object_validate_with_required_field_missing. Retrieved 7/9 statements.
# Partially parsed test_object_validate_with_additional_properties_false. Retrieved 6/7 statements.
# Partially parsed test_object_validate_with_property_names. Retrieved 7/9 statements.
# Partially parsed test_object_validate_with_mapping_type. Retrieved 5/8 statements.


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
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(var_5 == {'key': 'value'})
    assert var_6 is True

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
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = 'John'
    var_8 = {var_0: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = bool(var_9 == {'name': 'John'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Unknown'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'Unknown'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = e.messages()[0].code
    assert var_6 == 'empty'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = e.messages()[0].code
    assert var_8 == 'min_properties'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = e.messages()[0].code
    assert var_10 == 'max_properties'

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

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'extra': 'value'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = '^I_'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(pattern_properties=var_6, **var_7)
    var_9 = 'S_name'
    var_10 = 'I_age'
    var_11 = 'John'
    var_12 = 30
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = var_8.validate(var_13)
    var_15 = bool(var_14 == {'S_name': 'John', 'I_age': 30})
    assert var_15 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'Invalid'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'user'
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = {}
    var_9 = module_0.Object(properties=var_7, **var_8)
    var_10 = 'John'
    var_11 = {var_1: var_10}
    var_12 = {var_0: var_11}
    var_13 = var_9.validate(var_12)
    var_14 = bool(var_13 == {'user': {'name': 'John'}})
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
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = 'name'
    var_10 = 'age'
    var_11 = 'John'
    var_12 = 'not an integer'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = var_8.validate(var_13)
    var_15 = bool(False)
    assert var_15 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 5
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, **var_2)
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = 'name'
    var_8 = 'TooLongName'
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^num_'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'num_value'
    var_7 = 'not an integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

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
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = '^age'
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {var_4: var_6}
    var_8 = {}
    var_9 = module_0.Object(properties=var_3, pattern_properties=var_7, **var_8)
    var_10 = 'age_years'
    var_11 = 'John'
    var_12 = 30
    var_13 = {var_0: var_11, var_10: var_12}
    var_14 = var_9.validate(var_13)
    var_15 = bool(var_14 == {'name': 'John', 'age_years': 30})
    assert var_15 is True



# Parsed testcases at query #63
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #64
#--------------------------




import typesystem.fields as module_0
import typesystem.unique as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.Uniqueness()
    var_6 = 'duplicate'
    var_7 = var_5.add(var_6)
    var_8 = var_6 in var_5
    assert var_8 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_union_validate_predicate_line_18_true. Retrieved 9/42 statements.


def test_case_0():
    var_0 = 'type'
    var_1 = None
    var_2 = 'test_value'
    var_3 = 'custom'
    var_4 = []
    var_5 = 'test_value'
    var_6 = len(var_4)
    var_7 = 0
    var_8 = var_6 > var_7
    var_9 = bool(var_8 or True)
    assert var_9 is True



# Parsed testcases at query #66
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'hello'
    var_6 = 'world'
    var_7 = [var_5, var_6]
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == ['hello', 'world'])
    assert var_9 is True



# Parsed testcases at query #67
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'apple'
    var_6 = 'banana'
    var_7 = 'cherry'
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == ['apple', 'banana', 'cherry'])
    assert var_10 is True



# Parsed testcases at query #68
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Integer(**var_4)
    var_6 = [var_1, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
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
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'null'
    var_11 = bool('null' in str(e).lower())
    assert var_11 is True

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
    var_7 = 42
    var_8 = var_6.validate(var_7)
    assert var_8 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'union'
    var_14 = bool('union' in str(e).lower())
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = [var_1, var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
    var_9 = 'test'
    var_10 = var_8.validate(var_9)
    assert var_10 == 'test'
    var_11 = 123
    var_12 = var_8.validate(var_11)
    assert var_12 == 123
    var_13 = True
    var_14 = var_8.validate(var_13)
    assert var_14 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Integer(**var_6)
    var_8 = [var_3, var_7]
    var_9 = {}
    var_10 = module_0.Union(var_8, **var_9)
    var_11 = var_10.allow_null
    assert var_11 is True
    var_12 = None
    var_13 = var_10.validate(var_12)
    assert var_13 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'allow_null'
    var_5 = {var_4: var_0}
    var_6 = module_0.Integer(**var_5)
    var_7 = [var_3, var_6]
    var_8 = {}
    var_9 = module_0.Union(var_7, **var_8)
    var_10 = var_9.allow_null
    assert var_10 is False
    var_11 = None
    var_12 = var_9.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'null'
    var_15 = bool('null' in str(e).lower())
    assert var_15 is True



# Parsed testcases at query #69
#--------------------------




import typesystem.fields as module_0
import typesystem.unique as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.Uniqueness()
    var_6 = 'duplicate'
    var_7 = var_5.add(var_6)
    var_8 = var_6 in var_5
    assert var_8 is True



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_object_validate_with_mapping_type. Retrieved 5/9 statements.


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
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = len(e.messages())
    var_8 = bool(len(e.messages()) > 0)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(properties=var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'John'})
    assert var_10 is True

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
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
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
    var_9 = bool(len(e.messages()) > 0)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'extra': 'value'})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'extra'
    var_5 = 'not_an_int'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    var_10 = bool(len(e.messages()) > 0)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'S_name'
    var_7 = 'John'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {'S_name': 'John'})
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^S_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = {}
    var_6 = module_0.Object(pattern_properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'other'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == {'other': 'value'})
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'abc'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'abc': 'value'})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = 'abcde'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'address'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = 'city'
    var_8 = {}
    var_9 = module_0.String(**var_8)
    var_10 = {var_7: var_9}
    var_11 = {}
    var_12 = module_0.Object(properties=var_10, **var_11)
    var_13 = {var_0: var_4, var_1: var_6, var_2: var_12}
    var_14 = [var_0]
    var_15 = {}
    var_16 = module_0.Object(properties=var_13, required=var_14, **var_15)
    var_17 = 'John'
    var_18 = 30
    var_19 = 'NYC'
    var_20 = {var_7: var_19}
    var_21 = {var_0: var_17, var_1: var_18, var_2: var_20}
    var_22 = var_16.validate(var_21)
    var_23 = var_22['name']
    assert var_23 == 'John'
    var_24 = var_22['age']
    assert var_24 == 30
    var_25 = var_22['address']['city']
    assert var_25 == 'NYC'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_5]



# Parsed testcases at query #71
#--------------------------

# Failed to parse test_get_default_value_with_callable_default.




# Parsed testcases at query #72
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 6 evaluates to True when value is not in choices.'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_0.Choice(choices=var_3, **var_4)
    var_6 = 'invalid_option'
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #73
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = 'Must be a boolean.'
    var_3 = 'May not be null.'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'true'
    var_6 = 'false'
    var_7 = 'on'
    var_8 = 'off'
    var_9 = '1'
    var_10 = '0'
    var_11 = ''
    var_12 = 1
    var_13 = 0
    var_14 = True
    var_15 = False
    var_16 = True
    var_17 = False
    var_18 = True
    var_19 = False
    var_20 = False
    var_21 = True
    var_22 = False
    var_23 = {var_5: var_14, var_6: var_15, var_7: var_16, var_8: var_17, var_9: var_18, var_10: var_19, var_11: var_20, var_12: var_21, var_13: var_22}
    var_24 = ''
    var_25 = 'null'
    var_26 = 'none'
    var_27 = {var_24, var_25, var_26}
    var_28 = False
    var_29 = {}
    var_30 = module_0.Boolean(coerce_types=var_28, **var_29)
    var_31 = 'invalid_value'
    var_32 = var_30.validate(var_31)
    var_33 = bool(False)
    assert var_33 is True



# Parsed testcases at query #74
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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'helloworld'

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
    var_0 = 5
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
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'hello'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'hello world'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'max_length'

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
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'pattern'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '12345'
    var_4 = var_2.validate(var_3)
    assert var_4 == '12345'

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = 'abc'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'abc'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, min_length=var_0, **var_2)
    var_4 = 'hello'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = '^[a-z]+$'
    var_3 = {}
    var_4 = module_0.String(max_length=var_1, min_length=var_0, pattern=var_2, **var_3)
    var_5 = 'hello'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'hello'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_format_in_formats_and_is_native_type. Retrieved 8/30 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = 'FORMATS'
    var_2 = {}
    var_3 = 'custom_format'
    var_4 = {}
    var_5 = module_0.String(format=var_3, **var_4)
    var_6 = 'test_value'
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True
    var_9 = 'FORMATS'



# Parsed testcases at query #76
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #77
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 42
    var_3 = var_1.validate(var_2)
    assert var_3 == 42



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_line_11_predicate_evaluates_to_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 3.5
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #79
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'apple'
    var_6 = 'banana'
    var_7 = 'cherry'
    var_8 = [var_5, var_6, var_7]
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == ['apple', 'banana', 'cherry'])
    assert var_10 is True



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_union_validate_predicate_line_17_true. Retrieved 3/19 statements.


def test_case_0():
    var_0 = None
    var_1 = 'test_value'
    var_2 = 'test_value'



# Parsed testcases at query #81
#--------------------------




import typesystem.fields as module_0
import typesystem.unique as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.Uniqueness()
    var_6 = 'duplicate'
    var_7 = var_5.add(var_6)
    var_8 = var_6 in var_5
    assert var_8 is True



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_union_validate_predicate_line_18_true. Retrieved 5/24 statements.


def test_case_0():
    var_0 = 'validate_or_error'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = None
    var_4 = 'test_value'



# Parsed testcases at query #83
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
    var_8 = bool('null' in str(e).lower())
    assert var_8 is True

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
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'false'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'on'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'off'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '1'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = '0'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is False

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
    var_3 = 0
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'TRUE'
    var_4 = var_2.validate(var_3)
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'FALSE'
    var_4 = var_2.validate(var_3)
    assert var_4 is False

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
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'invalid'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'type'
    var_7 = bool('type' in str(e).lower())
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'null'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'none'
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #84
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
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3, var_0]
    var_5 = var_2.validate(var_4)
    var_6 = bool(var_5 == [1, 2])
    assert var_6 is True

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
    var_1 = 3
    var_2 = {}
    var_3 = module_0.Array(min_items=var_0, max_items=var_1, **var_2)
    var_4 = 2
    var_5 = [var_0, var_4]
    var_6 = var_3.validate(var_5)
    var_7 = bool(var_6 == [1, 2])
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
    var_3 = True
    var_4 = [var_3, var_3]
    var_5 = var_2.validate(var_4)
    var_6 = bool(False)
    assert var_6 is True

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
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_3: var_4}
    var_7 = [var_5, var_6]
    var_8 = var_2.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 1
    var_5 = 'not_int'
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(e.messages())
    var_11 = bool(len(e.messages()) > 0)
    assert var_11 is True



